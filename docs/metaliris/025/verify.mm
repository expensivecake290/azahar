#include "video_core/renderer_metal/mtl_scheduler.h"
#include "video_core/renderer_metal/mtl_stream_buffer.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <optional>
#include <stdexcept>
#include <vector>

using namespace Metal;
using namespace std::chrono_literals;

namespace {

template <typename Exception, typename T> bool Throws(T &&callback) {
  try {
    callback();
    return false;
  } catch (const Exception &) {
    return true;
  }
}

void CheckIntervals() {
  StreamRing ring(96);
  const auto a = ring.Reserve(24, 0, 0);
  assert(a.placement.range.offset == 0 && !a.placement.wrapped);
  assert(Throws<std::logic_error>(
      [&] { static_cast<void>(ring.Reserve(5, 0, 0)); }));
  assert(Throws<std::out_of_range>(
      [&] { static_cast<void>(ring.Commit(a, 25, 5)); }));
  auto first = ring.Commit(a, 12, 5);
  assert(first.range.length == 12);
  ring.NoteUse(first, 7);
  assert(ring.UseTick(first) == 7);
  assert(Throws<std::logic_error>([&] { ring.NoteUse(first, 6); }));

  const auto b = ring.Reserve(16, 48, 0);
  assert(b.placement.range.offset == 48 && ring.UseTick(first) == 7);
  auto pin = std::make_shared<int>(4);
  auto second = ring.Commit(b, 16, 8, pin);
  auto zero = ring.Reserve(0, 0, 0);
  assert(ring.Commit(zero, 0, 0).generation == 0);
  const auto c = ring.Reserve(32, 0, 0);
  assert(c.placement.range.offset == 64 && !c.placement.wrapped);
  auto third = ring.Commit(c, 32, 9);
  assert(ring.RequiredTick(ring.Find(12, 0)) == 7);
  assert(Throws<std::logic_error>(
      [&] { static_cast<void>(ring.Reserve(12, 0, 6)); }));
  assert(Throws<std::overflow_error>(
      [&] { static_cast<void>(ring.Reserve(50, 0, 9)); }));
  const auto d = ring.Reserve(12, 0, 7);
  assert(d.placement.wrapped && d.placement.range.offset == 0);
  assert(Throws<std::logic_error>(
      [&] { static_cast<void>(ring.UseTick(first)); }));
  assert(ring.UseTick(third) == 9);
  ring.Cancel(d);
  assert(Throws<std::logic_error>(
      [&] { static_cast<void>(ring.Commit(d, 12, 10)); }));
  pin.reset();
  assert(ring.RequiredTick(ring.Find(64, 0)) == 8);
  const auto e = ring.Reserve(64, 0, 8);
  assert(e.placement.wrapped && e.placement.range.offset == 0);
  assert(Throws<std::logic_error>(
      [&] { static_cast<void>(ring.UseTick(second)); }));
  assert(ring.UseTick(third) == 9);
  static_cast<void>(ring.Commit(e, 64, 10));
  assert(ring.RequiredTick(ring.Find(32, 0)) == 9);
  assert(
      Throws<std::out_of_range>([&] { static_cast<void>(ring.Find(97, 0)); }));
}

void CheckWatchGrowth() {
  constexpr u32 capacity = 20000;
  StreamRing ring(capacity);
  std::vector<StreamRing::Slice> slices;
  for (u64 tick = 1; tick <= capacity; ++tick) {
    auto reserve = ring.Reserve(1, 1, 0);
    assert(reserve.placement.range.offset == tick - 1);
    slices.push_back(ring.Commit(reserve, 1, tick));
  }
  const auto replacement = ring.Reserve(capacity / 2, 1, capacity / 2);
  assert(replacement.placement.wrapped);
  assert(Throws<std::logic_error>(
      [&] { static_cast<void>(ring.UseTick(slices.front())); }));
  assert(Throws<std::logic_error>(
      [&] { static_cast<void>(ring.UseTick(slices[capacity / 2 - 1])); }));
  assert(ring.UseTick(slices.back()) == capacity);
  auto latest = ring.Commit(replacement, capacity / 2, capacity + 1);
  assert(ring.UseTick(latest) == capacity + 1);
  assert(ring.RequiredTick(ring.Find(capacity / 2, 0)) == capacity);
}

void CheckNative(id<MTLDevice> native) {
  const auto deadline = [] { return std::chrono::steady_clock::now() + 10s; };
  DeviceContext device(native);
  Scheduler scheduler(device, 2);
  MemoryBudget budget(1024 * 1024);
  MemoryAllocator memory(native, budget);
  StreamBuffer upload(memory, scheduler, 128, BufferType::Upload);
  StreamBuffer download(memory, scheduler, 64, BufferType::Download);
  assert(upload.Size() == 128 && download.Size() == 64);
  assert(
      Throws<std::logic_error>([&] { static_cast<void>(download.Map(64)); }));
  assert(Throws<std::logic_error>(
      [&] { static_cast<void>(upload.ReserveDownload(64)); }));
  {
    auto canceled = upload.Map(64, 0);
    assert(canceled.Range().offset == 0 && canceled.Bytes().size() == 64);
  }
  auto mapping = upload.Map(96, 64);
  std::ranges::fill(mapping.Bytes(), std::byte{0x4E});
  auto upload_slice = upload.Commit(std::move(mapping), 64);
  assert(mapping.Bytes().empty() && upload_slice.Range().length == 64);
  auto reservation = download.ReserveDownload(64, 64);
  assert(reservation.Bytes().empty());
  auto readback = std::optional(download.Commit(std::move(reservation), 64));
  assert(Throws<std::logic_error>(
      [&] { static_cast<void>(download.ReadDownload(*readback)); }));
  auto source = upload.AcquireGPUAccess(upload_slice, BufferAccess::Read);
  auto target = download.AcquireGPUAccess(*readback, BufferAccess::Write);
  scheduler.SetQueue(QueueType::Transfer);
  scheduler.Record([source = std::move(source),
                    target = std::move(target)](EncodingContext &ctx) mutable {
    auto encoder = [ctx.GetCommandBuffer() computeCommandEncoder];
    const std::array uses{
        ResourceUse{source.GetBuffer(), BufferAccess::Read, MTLStageBlit,
                    source.Range()},
        ResourceUse{target.GetBuffer(), BufferAccess::Write, MTLStageBlit,
                    target.Range()},
    };
    ctx.Synchronize(encoder, uses);
    [encoder copyFromBuffer:source.GetBuffer()
               sourceOffset:source.Range().offset
                   toBuffer:target.GetBuffer()
          destinationOffset:target.Range().offset
                       size:target.Range().length];
    ctx.EndPass(encoder);
    ctx.Retain(std::move(source));
    ctx.Retain(std::move(target));
  });
  assert(Throws<std::overflow_error>(
      [&] { static_cast<void>(download.ReserveDownload(1, 0, deadline())); }));
  {
    auto bytes = download.ReadDownload(*readback, deadline());
    assert(std::ranges::all_of(
        bytes.Bytes(), [](auto byte) { return byte == std::byte{0x4E}; }));
  }
  readback.reset();
  {
    auto next_download = download.ReserveDownload(64, 64, deadline());
    assert(next_download.Wrapped() && next_download.Range().offset == 0);
  }
  const auto tick = scheduler.GetMasterSemaphore().Timeline().LastIssuedTick();
  assert(tick == 1 && scheduler.IsFree(tick));
  {
    auto next_upload = upload.Map(80, 64, deadline());
    assert(next_upload.Wrapped() && next_upload.Range().offset == 0);
  }
  assert(Throws<std::logic_error>([&] {
    static_cast<void>(
        upload.AcquireGPUAccess(upload_slice, BufferAccess::Read));
  }));
  assert(scheduler.Shutdown(deadline()));
}

void CheckPressure(id<MTLDevice> native) {
  DeviceContext device(native);
  Scheduler scheduler(device);
  MemoryBudget budget(32 * 1024);
  MemoryAllocator memory(native, budget);
  {
    StreamBuffer ring(memory, scheduler, 64 * 1024);
    assert(ring.Size() >= 8 * 1024 && ring.Size() <= 32 * 1024);
    assert(budget.GetUsage().physical_used >= ring.Size());
  }
  assert(budget.GetUsage().physical_used == 0);
}

} // Anonymous namespace

int main() {
  @autoreleasepool {
    CheckIntervals();
    CheckWatchGrowth();
    std::puts("PASS 20,000 watches, repeated overlap, stale generation, pins, "
              "partial and zero commits");
    id<MTLDevice> native = MTLCreateSystemDefaultDevice();
    if (!native || ![native supportsFamily:MTLGPUFamilyMetal4]) {
      std::puts("PENDING native stream upload/readback visibility and budget "
                "fallback");
      return 2;
    }
    CheckNative(native);
    CheckPressure(native);
    std::puts("PASS native stream upload, pinned readback, wrap, visibility "
              "and budget fallback");
  }
}
