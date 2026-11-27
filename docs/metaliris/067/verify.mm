#include "video_core/renderer_metal/mtl_pipeline_cache.h"
#include "video_core/renderer_metal/mtl_rasterizer.h"
#include <array>
#include <cassert>
#include <cstdio>
#include <type_traits>

static_assert(std::is_final_v<Metal::RasterizerMetal>);
static_assert(std::is_base_of_v<VideoCore::RasterizerAccelerated,
                                Metal::RasterizerMetal>);
static_assert(
    std::is_same_v<decltype(Metal::ScreenInfo::image), Metal::TextureResource>);
static_assert(std::is_constructible_v<
              Metal::RasterizerMetal, Memory::MemorySystem &, Pica::PicaCore &,
              VideoCore::CustomTexManager &, VideoCore::RendererBase &,
              Metal::Instance &, Metal::Scheduler &, Metal::RenderManager &,
              Metal::DescriptorUpdateQueue &, Metal::MemoryAllocator &,
              Metal::MemoryBudget &, Metal::PipelineCacheOptions>);
static_assert(requires(Metal::RasterizerMetal &raster,
                       const Pica::DisplayTransferConfig &display,
                       const Pica::MemoryFillConfig &fill,
                       const Pica::FramebufferConfig &buffer,
                       Metal::ScreenInfo &result, const std::atomic_bool &stop,
                       const VideoCore::DiskResourceLoadCallback &callback) {
  { raster.DrawTriangles() } -> std::same_as<void>;
  { raster.AccelerateDrawBatch(true) } -> std::same_as<bool>;
  { raster.FlushAll() } -> std::same_as<void>;
  { raster.FlushRegion(0, 1) } -> std::same_as<void>;
  { raster.InvalidateRegion(0, 1) } -> std::same_as<void>;
  { raster.FlushAndInvalidateRegion(0, 1) } -> std::same_as<void>;
  { raster.ClearAll(true) } -> std::same_as<void>;
  { raster.AccelerateDisplayTransfer(display) } -> std::same_as<bool>;
  { raster.AccelerateTextureCopy(display) } -> std::same_as<bool>;
  { raster.AccelerateFill(fill) } -> std::same_as<bool>;
  { raster.AccelerateDisplay(buffer, 0, 9, result) } -> std::same_as<bool>;
  { raster.LoadDefaultDiskResources(stop, callback) } -> std::same_as<void>;
  { raster.SwitchDiskResources(0) } -> std::same_as<void>;
  { raster.DiskResourcesLoadCrashed(0) } -> std::same_as<bool>;
#ifdef ENABLE_FREECAM
  { raster.SetFreeCam(std::array<f32, 16>{}, true) } -> std::same_as<void>;
#endif
});

int main() {
  for (u32 lo = 0; lo < 256; ++lo) {
    for (u32 hi = 0; hi < 256; ++hi) {
      const std::array<u8, 5> indices{5, 0, static_cast<u8>(lo),
                                      static_cast<u8>(hi), 17};
      assert(Metal::ReadVertexIndex(indices, 1, true) == hi * 256 + lo);
      assert(Metal::ReadVertexIndex(indices, 2, false) == lo);
      assert(!Metal::ReadVertexIndex(indices, 2, true));
      assert(!Metal::ReadVertexIndex(indices, 5, false));
    }
  }
  assert(!Metal::ReadVertexIndex({}, 0xffffffff, true));
  const auto screen = Metal::ScreenInfo::Coordinates({1, 5, 8, 2}, 9, 7);
  assert(screen && screen->left == 2.f / 7 && screen->right == 5.f / 7 &&
         screen->top == 1.f / 9 && screen->bottom == 8.f / 9);
  assert(!Metal::ScreenInfo::Coordinates({1, 5, 8, 2}, 7, 7));
  assert(!Metal::ScreenInfo::Coordinates({1, 5, 8, 2}, 9, 4));
  assert(!Metal::ScreenInfo::Coordinates({0, 0, 0, 0}, 0, 0));
  assert(!Metal::ScreenInfo::Coordinates({3, 1, 3, 1}, 3, 1));
  std::puts("PASS Azahar native rasterizer interface, all 65536 index values, "
            "truncation, "
            "rotated screen coordinates and clip bounds");
}
