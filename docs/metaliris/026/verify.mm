#include "video_core/renderer_metal/mtl_descriptor_update_queue.h"
#include "video_core/renderer_metal/mtl_scheduler.h"
#include <cassert>
#include <cstdio>
#include <limits>

using namespace Metal;

#if defined(CHECK_NATIVE_CONSUMER)
void CheckNativeConsumer(DescriptorUpdateQueue &queue, Scheduler &scheduler,
                         id<MTLBuffer> buffer, id<MTLTexture> texture,
                         id<MTLSamplerState> sampler) {
  auto set = queue.CreateSet({1, 4, 4, true});
  queue.AddBuffer(set, 0, buffer, 0, 64, BufferAccess::Read, 16,
                  MTLStageDispatch);
  queue.AddImageSampler(set, 0, 2, texture, sampler, 1, MTLStageDispatch);
  const auto snapshot = queue.Capture(set);
  scheduler.Record([snapshot](EncodingContext &ctx) {
    auto encoder = [ctx.GetCommandBuffer() computeCommandEncoder];
    snapshot.BindCompute(ctx, encoder);
    ctx.EndPass(encoder);
  });
}
#else
int main() {
  using Queue = DescriptorUpdateQueue;
  constexpr auto max = std::numeric_limits<std::size_t>::max();
  static_assert(BindingLayout{31, 128, 16, true}.IsValid());
  static_assert(!BindingLayout{32, 128, 16, true}.IsValid());
  static_assert(!BindingLayout{31, 129, 16}.IsValid());
  static_assert(!BindingLayout{31, 128, 17}.IsValid());
  static_assert(Queue::Element(128, 100, 27) == 127);
  static_assert(!Queue::Element(128, 100, 28));
  static_assert(!Queue::Element(128, 100, std::numeric_limits<u32>::max()));
  static_assert(!Queue::Element(0, 0, 0));
  static_assert(!Queue::BufferBindingRange(64, 64, {}));
  static_assert(!Queue::BufferBindingRange(64, 16, 0));
  static_assert(!Queue::BufferBindingRange(64, 16, max));
  static_assert(Queue::BufferBindingRange(max, max - 16, {})->length == 16);
  static_assert(Queue::BufferBindingRange(64, 16, 48)->offset == 16);
  assert(!Queue::BufferBindingRange(16, 1, 16));
  std::puts("PASS native argument consumer and exact binding bounds");
}
#endif
