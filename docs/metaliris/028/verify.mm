#include "video_core/renderer_metal/mtl_render_manager.h"
#include <cassert>
#include <cstdio>
#include <limits>

using namespace Metal;

#if defined(CHECK_NATIVE_CONSUMER)
void CheckConstruction(const Instance &instance, Scheduler &scheduler,
                       const MemoryAllocator &allocator) {
  RenderManager manager(instance, scheduler, allocator);
}

void CheckConsumer(RenderManager &manager, id<MTLTexture> target,
                   id<MTLTexture> resolve,
                   id<MTLRenderPipelineState> pipeline) {
  RenderPass pass{
      .color = {.target = {target},
                .resolve = {resolve},
                .load_action = MTLLoadActionClear,
                .store_action = MTLStoreActionStoreAndMultisampleResolve},
      .area = {0, 0, 64, 64},
      .width = 64,
      .height = 64,
      .samples = 4,
  };
  manager.BeginRendering(pass);
  manager.RecordDraw(
      [pipeline](EncodingContext &ctx, id<MTL4RenderCommandEncoder> encoder) {
        [encoder setRenderPipelineState:pipeline];
        ctx.Retain(pipeline);
        [encoder drawPrimitives:MTLPrimitiveTypeTriangle
                    vertexStart:0
                    vertexCount:3];
      });
  manager.EndRendering();
}
#else
int main() {
  constexpr auto max = std::numeric_limits<u32>::max();
  static_assert(RenderArea{0, 0, 64, 64}.Covers(64, 64));
  static_assert(RenderArea{7, 6, 57, 58}.Fits(64, 64));
  static_assert(!RenderArea{7, 6, 57, 59}.Fits(64, 64));
  static_assert(!RenderArea{max, max, 1, 1}.Fits(max, max));
  static_assert(!RenderArea{0, 0, 0, 1}.Fits(1, 1));
  static_assert(RenderArea::FromRectangle(Common::Rectangle<u32>{
                    5, 90, 100, 70}) == RenderArea{5, 70, 95, 20});
  static_assert(
      !RenderArea::FromRectangle(Common::Rectangle<u32>{5, 70, 100, 90}));
  static_assert(
      !RenderArea::FromRectangle(Common::Rectangle<u32>{5, 90, 4, 70}));

  RenderPass original{.area = {0, 0, 32, 32}, .width = 32, .height = 32};
  auto next = original;
  assert(original.CanContinueWith(next));
  next.color.load_action = MTLLoadActionClear;
  assert(!original.CanContinueWith(next));
  assert(next.CanContinueWith(original));
  next = original;
  next.stencil.store_action = MTLStoreActionDontCare;
  assert(!original.CanContinueWith(next));
  next = original;
  ++next.area.x;
  assert(!original.CanContinueWith(next));
  next = original;
  next.tile_feedback = true;
  assert(!original.CanContinueWith(next));
  next = original;
  next.samples = 4;
  assert(!original.CanContinueWith(next));
  std::puts("PASS native render consumer, merge/split policy and checked "
            "render areas");
}
#endif
