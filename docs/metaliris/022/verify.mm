#include "video_core/renderer_metal/mtl_scheduler.h"
#include <cassert>
#include <limits>
#include <memory>

using namespace Metal;

#if defined(CHECK_NATIVE_CONSUMER)
void CheckConsumer(Scheduler &scheduler, id<MTLBuffer> buffer) {
  struct alignas(64) Capture {
    std::unique_ptr<int> value;
  };
  scheduler.SetQueue(QueueType::Transfer);
  scheduler.Record([capture = Capture{std::make_unique<int>(4)},
                    buffer](EncodingContext &ctx) {
    static_cast<void>(ctx.OwnsWork({}));
    static_cast<void>(ctx.GetResourcePool());
    const ResourceUse use{
        buffer, BufferAccess::Write, MTLStageBlit, {0, sizeof(int)}};
    auto encoder = [ctx.GetCommandBuffer() computeCommandEncoder];
    ctx.Synchronize(encoder, {&use, 1});
    [encoder fillBuffer:buffer
                  range:NSMakeRange(0, sizeof(int))
                  value:*capture.value];
    ctx.EndPass(encoder);
    ctx.Checkpoint("initialized");
  });
  const u64 producer = scheduler.Flush();
  scheduler.SetQueue(QueueType::Graphics);
  static_cast<void>(scheduler.Finish({.wait_for_tick = producer}));
}
#else
int main() {
  using enum BufferAccess;
  constexpr auto max = std::numeric_limits<std::size_t>::max();
  static_assert(!ResourceUse::Conflicts(Read, {0, 8}, Read, {0, 8}));
  static_assert(ResourceUse::Conflicts(Read, {0, 8}, Write, {7, 8}));
  static_assert(ResourceUse::Conflicts(Write, {7, 8}, Read, {0, 8}));
  static_assert(!ResourceUse::Conflicts(Write, {0, 8}, ReadWrite, {8, 8}));
  static_assert(
      !ResourceUse::Conflicts(Write, {max - 10, 1}, Read, {max - 5, 4}));
  static_assert(
      ResourceUse::Conflicts(Write, {max - 10, 9}, Read, {max - 5, 4}));
  static_assert(ResourceUse::Conflicts(ReadWrite, {}, Read, {8, 1}));
  static_assert(False(StateFlags::AllDirty));
  static_assert(True((StateFlags::Pipeline | StateFlags::ArgumentTables) &
                     StateFlags::Pipeline));
  static_assert(False(StateFlags::Pipeline & StateFlags::ArgumentTables));
  assert((StateFlags::Pipeline | StateFlags::ArgumentTables) ==
         static_cast<StateFlags>(3));
  puts("PASS native header consumer, range hazards and state flags");
}
#endif
