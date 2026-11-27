#include "video_core/renderer_metal/mtl_scheduler.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <vector>

using namespace Metal;
using namespace std::chrono_literals;

namespace {

auto Deadline() { return std::chrono::steady_clock::now() + 10s; }

template <typename Exception, typename T> bool Throws(T &&callback) {
  try {
    callback();
    return false;
  } catch (const Exception &) {
    return true;
  }
}

void Copy(EncodingContext &ctx, id<MTL4ComputeCommandEncoder> encoder,
          id<MTLBuffer> source, id<MTLBuffer> output) {
  const std::array uses{
      ResourceUse{source, BufferAccess::Read, MTLStageBlit, {0, 64}},
      ResourceUse{output, BufferAccess::Write, MTLStageBlit, {0, 64}},
  };
  ctx.Synchronize(encoder, uses);
  [encoder copyFromBuffer:source
             sourceOffset:0
                 toBuffer:output
        destinationOffset:0
                     size:64];
}

void CheckScheduling(id<MTLDevice> native) {
  DeviceContext device(native);
  Scheduler scheduler(device, 1, 2);
  u32 dispatched = 0;
  std::vector<u32> order;
  scheduler.RegisterOnDispatch([&] { ++dispatched; });
  scheduler.RegisterOnSubmit([&] {
    scheduler.Record([&](EncodingContext &) { order.push_back(300); });
  });
  scheduler.DispatchWork();
  assert(dispatched == 0);
  for (u32 index = 0; index < 300; ++index) {
    scheduler.Record([value = std::make_unique<u32>(index),
                      &order](EncodingContext &) { order.push_back(*value); });
  }
  scheduler.WaitWorker();
  assert(dispatched == 3 && order.size() == 300);
  for (u32 index = 0; index < order.size(); ++index) {
    assert(order[index] == index);
  }

  scheduler.MarkStateNonDirty(StateFlags::Pipeline |
                              StateFlags::ArgumentTables);
  scheduler.MakeDirty(StateFlags::ArgumentTables);
  assert(!scheduler.IsStateDirty(StateFlags::Pipeline));
  assert(scheduler.IsStateDirty(StateFlags::ArgumentTables));
  const auto first = scheduler.CurrentTick();
  assert(scheduler.Wait(first + 1, Deadline()) ==
         CompletionStatus::UnknownTicket);
  assert(Throws<std::invalid_argument>(
      [&] { static_cast<void>(scheduler.Flush({.wait_for_tick = first})); }));
  assert(scheduler.Wait(first, Deadline()) == CompletionStatus::Complete);
  assert(scheduler.IsStateDirty(StateFlags::Pipeline));
  assert(scheduler.IsFree(first) && order.back() == 300 && dispatched == 4);
  assert(scheduler.Shutdown(Deadline()));
}

void CheckVisibility(id<MTLDevice> native, std::size_t command_capacity) {
  DeviceContext device(native);
  Scheduler scheduler(device, command_capacity, 2);
  MemoryAllocator memory(native, MemoryBudget(16 * 1024 * 1024));
  auto source = memory.AllocateBuffer(64, MemoryPool::Transient);
  auto intermediate = memory.AllocateBuffer(64, MemoryPool::Transient);
  auto result = memory.AllocateBuffer(64, MemoryPool::Readback);
  auto destination = memory.AllocateBuffer(64, MemoryPool::Readback);
  auto write = source.AcquireGPUAccess({0, 64}, BufferAccess::ReadWrite);
  auto intermediate_access =
      intermediate.AcquireGPUAccess({0, 64}, BufferAccess::ReadWrite);
  auto result_write = result.AcquireGPUAccess({0, 64}, BufferAccess::Write);

  scheduler.SetQueue(QueueType::Transfer);
  scheduler.Record(
      [write = std::move(write), middle = std::move(intermediate_access),
       output = std::move(result_write)](EncodingContext &ctx) mutable {
        assert(ctx.GetQueue() == QueueType::Transfer && ctx.CurrentTick() == 1);
        auto encoder = [ctx.GetCommandBuffer() computeCommandEncoder];
        const ResourceUse use{
            write.GetBuffer(), BufferAccess::Write, MTLStageBlit, {0, 64}};
        ctx.Synchronize(encoder, {&use, 1});
        [encoder fillBuffer:write.GetBuffer()
                      range:NSMakeRange(0, 64)
                      value:0xA9];
        Copy(ctx, encoder, write.GetBuffer(), middle.GetBuffer());
        ctx.EndPass(encoder);
        encoder = [ctx.GetCommandBuffer() computeCommandEncoder];
        Copy(ctx, encoder, middle.GetBuffer(), output.GetBuffer());
        ctx.EndPass(encoder);
        ctx.Retain(std::move(write));
        ctx.Retain(std::move(middle));
        ctx.Retain(std::move(output));
        ctx.Checkpoint("producer copy");
      });
  assert(scheduler.Flush() == 1);
  scheduler.SetQueue(QueueType::Graphics);
  auto read = result.AcquireGPUAccess({0, 64}, BufferAccess::Read);
  auto final_write = destination.AcquireGPUAccess({0, 64}, BufferAccess::Write);
  scheduler.Record([read = std::move(read), write = std::move(final_write)](
                       EncodingContext &ctx) mutable {
    assert(ctx.GetQueue() == QueueType::Graphics && ctx.CurrentTick() == 2);
    auto encoder = [ctx.GetCommandBuffer() computeCommandEncoder];
    Copy(ctx, encoder, read.GetBuffer(), write.GetBuffer());
    ctx.EndPass(encoder);
    ctx.Retain(std::move(read));
    ctx.Retain(std::move(write));
    ctx.Checkpoint("consumer copy");
  });

  assert(Throws<std::logic_error>(
      [&] { static_cast<void>(destination.MapRead({0, 64})); }));
  assert(scheduler.Finish({}, Deadline()) == CompletionStatus::Complete);
  const auto bytes = destination.MapRead({0, 64});
  assert(std::ranges::all_of(
      bytes.Bytes(), [](std::byte byte) { return byte == std::byte{0xA9}; }));
  const auto report = scheduler.GetLastCompletion();
  assert(report && report->tick == 2 && report->queue == QueueType::Graphics &&
         !report->error);
  assert(report->checkpoint && report->checkpoint->label == "consumer copy");
  assert(report->checkpoint->gpu_timestamp &&
         *report->checkpoint->gpu_timestamp != 0);
  assert(scheduler.Shutdown(Deadline()));
}

void CheckFailures(id<MTLDevice> native) {
  DeviceContext device(native);
  {
    Scheduler scheduler(device, 1, 2);
    scheduler.Record([](EncodingContext &) {
      throw std::runtime_error("encoding fixture");
    });
    const auto tick = scheduler.Flush();
    assert(scheduler.Wait(tick, Deadline()) == CompletionStatus::Faulted);
    const auto report = scheduler.GetFault();
    assert(report && report->tick == tick &&
           report->stage == SubmissionStage::Encoding && report->error);
    assert([report->error.localizedDescription
        isEqualToString:@"encoding fixture"]);
    assert(scheduler.Shutdown(Deadline()) && scheduler.IsFree(tick));
  }
  {
    Scheduler scheduler(device);
    scheduler.Record([](EncodingContext &) {});
    scheduler.WaitWorker();
    assert(device.AdvanceGeneration());
    const auto tick = scheduler.Flush();
    assert(scheduler.Wait(tick, Deadline()) == CompletionStatus::Faulted);
    const auto report = scheduler.GetFault();
    assert(report && report->stage == SubmissionStage::DeviceGeneration);
    assert(scheduler.Shutdown(Deadline()));
  }
  {
    Scheduler scheduler(device, 1, 2);
    auto owner = std::make_shared<int>(4);
    auto weak = std::weak_ptr(owner);
    scheduler.Record([owner = std::move(owner)](EncodingContext &ctx) {
      ctx.Retain(owner);
    });
    const auto tick = scheduler.Flush();
    assert(scheduler.Shutdown(Deadline()));
    assert(scheduler.IsFree(tick) && weak.expired());
    assert(Throws<std::logic_error>(
        [&] { scheduler.Record([](EncodingContext &) {}); }));
  }
}

} // Anonymous namespace

int main() {
  @autoreleasepool {
    id<MTLDevice> native = MTLCreateSystemDefaultDevice();
    if (!native || ![native supportsFamily:MTLGPUFamilyMetal4]) {
      std::puts("PENDING scheduler FIFO, allocator exhaustion, native "
                "feedback, stages, shared "
                "visibility, generation rejection, checkpoints, and shutdown "
                "on Metal 4 GPU");
      return 2;
    }
    CheckScheduling(native);
    CheckVisibility(native, 1);
    CheckVisibility(native, 2);
    CheckFailures(native);
    std::puts("PASS scheduler FIFO, hazards, capacity-one retirement, "
              "visibility, checkpoints, "
              "failure and shutdown");
  }
}
