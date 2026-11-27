#include "core/frontend/image_interface.h"
#include "video_core/custom_textures/material.h"
#include "video_core/renderer_metal/mtl_pipeline_cache.h"
#include "video_core/renderer_metal/mtl_texture_runtime.h"
#include <array>
#include <cassert>
#include <cstdio>
#include <cstring>
#include <ranges>
#include <vector>

using namespace Metal;
using namespace VideoCore;

namespace {

SurfaceParams Params(PixelFormat format, u32 width, u32 height,
                     u32 levels = 1) {
  SurfaceParams params;
  params.addr = 0x100000;
  params.pixel_format = format;
  params.width = width;
  params.height = height;
  params.levels = levels;
  params.UpdateParams();
  return params;
}

void Upload(TextureRuntime &runtime, TextureSurface &texture,
            Common::Rectangle<u32> rect, std::span<const u8> bytes,
            u32 level = 0) {
  const auto staging =
      runtime.FindStaging(static_cast<u32>(bytes.size()), true);
  std::ranges::copy(bytes, staging.mapped.begin());
  texture.Upload({staging.offset, staging.size, rect, level}, staging);
}

std::vector<u8> Download(TextureRuntime &runtime, TextureSurface &texture,
                         Common::Rectangle<u32> rect, u32 level = 0) {
  const auto dim = BlitBounds(rect);
  const auto staging = runtime.FindStaging(
      dim.width * dim.height * texture.GetInternalBytesPerPixel(), false);
  texture.Download({staging.offset, staging.size, rect, level}, staging);
  return {staging.mapped.begin(), staging.mapped.end()};
}

std::vector<u8> Words(u32 pixel, u32 bytes, u32 count) {
  std::vector<u8> result(count * bytes);
  for (u32 i = 0; i < count; ++i) {
    for (u32 byte = 0; byte < bytes; ++byte) {
      result[i * bytes + byte] = static_cast<u8>(pixel >> (byte * 8));
    }
  }
  return result;
}

void RoundTrip(TextureRuntime &runtime) {
  struct Case {
    PixelFormat format;
    u32 word;
    bool converts;
  };
  constexpr std::array corpus{
      Case{PixelFormat::RGBA8, 0xcc886644, true},
      Case{PixelFormat::RGB8, 0xffab6543, true},
      Case{PixelFormat::RGB565, 0xf091, false},
      Case{PixelFormat::RGB5A1, 0xd931, false},
      Case{PixelFormat::RGBA4, 0x1357, false},
      Case{PixelFormat::IA8, 0xbf444444, false},
      Case{PixelFormat::RG8, 0xff006655, false},
      Case{PixelFormat::I8, 0xff6f6f6f, false},
      Case{PixelFormat::D16, 0x832f, false},
      Case{PixelFormat::D24, 0x3f000000, true},
      Case{PixelFormat::D24S8, 0x8000002d, false},
  };
  constexpr Common::Rectangle<u32> region{1, 5, 8, 0};
  for (const auto &entry : corpus) {
    TextureSurface tex(runtime, Params(entry.format, 9, 7));
    assert(runtime.NeedsConversion(tex) == entry.converts);
    const auto input =
        Words(entry.word, tex.GetInternalBytesPerPixel(), region.GetArea());
    Upload(runtime, tex, region, input);
    assert(Download(runtime, tex, region) == input);
    assert(runtime.GetResourceFreeTick() != 0);
  }
  std::puts("PASS physical Metal texture roundtrip with odd pitches for "
            "color/depth/stencil formats");
}

void CopyFrameAndShadow(Instance &instance, TextureRuntime &runtime,
                        RenderManager &manager) {
  const auto params = Params(PixelFormat::RGBA8, 9, 7, 2);
  TextureSurface tex(runtime, params);
  const auto source = Words(0xeedd8855, 4, 9 * 7);
  Upload(runtime, tex, params.GetRect(), source);
  runtime.GenerateMipmaps(tex);
  assert(Download(runtime, tex, {0, 3, 4, 0}, 1) ==
         Words(0xeedd8855, 4, 4 * 3));
  const auto backup = tex.CopyImageView();
  assert(backup && backup != tex.Image() && backup.mipmapLevelCount == 2);
  const TextureCopy alias{0, 0, 0, 0, {0, 0}, {1, 1}, {8, 6}};
  assert(runtime.CopyTextures(tex, tex, alias));
  assert(Download(runtime, tex, params.GetRect()) == source);

  const bool supports_samples = instance.SupportsSampleCount(4);
  tex.ScaleUp(2, supports_samples ? 4 : 1);
  const auto params_depth = Params(PixelFormat::D24S8, 9, 7);
  TextureSurface depth(runtime, params_depth);
  Upload(runtime, depth, params_depth.GetRect(), Words(0x80000063, 4, 9 * 7));
  depth.ScaleUp(2, supports_samples ? 4 : 1);
  Framebuffer frame(runtime, FramebufferParams{}, &tex, &depth);
  assert(frame.IsValid() && frame.Scale() == 2 &&
         frame.Samples() == (supports_samples ? 4 : 1));
  const auto pass = frame.MakePass({0, 0, 18, 14});
  manager.BeginRendering(pass);
  manager.EndRendering();
  tex.ScaleUp(1, 1);
  depth.ScaleUp(1, 1);
  assert(frame.Scale() == 1 && frame.Samples() == 1);
  assert(Download(runtime, tex, params.GetRect()) == source);
  assert(Download(runtime, depth, params_depth.GetRect()) ==
         Words(0x80000063, 4, 9 * 7));

  const auto buffer = tex.PrepareShadowStorage(false);
  assert(buffer.buffer && !buffer.image && buffer.row_pitch == 9);
  tex.CommitShadowStorage(buffer);
  assert(Download(runtime, tex, params.GetRect()) == source);
  const auto sampled_shadow = tex.ShadowImage();
  assert(sampled_shadow &&
         sampled_shadow.Get().pixelFormat == MTLPixelFormatR32Uint &&
         !(sampled_shadow.Get().usage & MTLTextureUsageShaderAtomic));
  if (PipelineDeviceProfile::Detect(instance.GetContext()).texture_atomics) {
    const auto image = tex.PrepareShadowStorage(true);
    assert(image.image &&
           image.image.Get().pixelFormat == MTLPixelFormatR32Uint &&
           !image.buffer);
    tex.CommitShadowStorage(image);
    assert(Download(runtime, tex, params.GetRect()) == source);
  }

  TextureClear clear{0, 0, {2, 5, 7, 2}, {.color = {1, 0, 0, 1}}};
  assert(runtime.ClearTexture(tex, clear));
  const auto pixels = Download(runtime, tex, params.GetRect());
  for (u32 y = 0; y < 7; ++y) {
    for (u32 x = 0; x < 9; ++x) {
      u32 value{};
      std::memcpy(&value, pixels.data() + 4 * (9 * y + x), sizeof(value));
      const bool inside = x >= 2 && x < 7 && y >= 2 && y < 5;
      assert(value == (inside ? 0xff0000ff : 0xeedd8855));
    }
  }
  std::puts("PASS native alias copy, feedback view, all mips, multisample "
            "seeding, frame scale, "
            "shadow roundtrip and partial clear");
}

void Custom(TextureRuntime &runtime) {
  const bool bc = runtime.SupportsCustomFormat(CustomPixelFormat::BC1);
  const auto format = bc ? CustomPixelFormat::BC1 : CustomPixelFormat::RGBA8;
  Frontend::ImageInterface images;
  CustomTexture diffuse(images), normal(images);
  for (auto *texture : {&diffuse, &normal}) {
    texture->width = 7;
    texture->height = 5;
    texture->format = format;
    if (bc) {
      constexpr std::array<u8, 8> block{0, 0xf8, 0, 0, 0, 0, 0, 0};
      for (u32 i = 0; i < 4; ++i) {
        texture->data.insert(texture->data.end(), block.begin(), block.end());
      }
    } else {
      texture->data = Words(0xff0000ff, 4, 7 * 5);
    }
  }
  Material replacement{};
  replacement.width = 7;
  replacement.height = 5;
  replacement.format = format;
  replacement.textures = {&diffuse, &normal};
  replacement.state = DecodeState::Decoded;
  auto params = Params(PixelFormat::RGBA8, 7, 5);
  params.res_scale = 2;
  TextureSurface base(runtime, params);
  TextureSurface mapped(runtime, base, &replacement);
  mapped.UploadCustom(&replacement, 0);
  assert(mapped.Image(TextureRepresentation::Normal) &&
         mapped.Image(TextureRepresentation::Normal) !=
             mapped.Image(TextureRepresentation::Base));
  TextureSurface out(runtime, Params(PixelFormat::RGBA8, 7, 5));
  const TextureBlit blit{0, 0, 0, 0, params.GetRect(), params.GetRect()};
  assert(runtime.BlitTextures(mapped, out, blit));
  assert(Download(runtime, out, params.GetRect()) ==
         Words(0xff0000ff, 4, 7 * 5));
  std::puts("PASS actual custom color and normal allocation, padded native "
            "block upload and sample");
}

} // Anonymous namespace

int main() {
  @autoreleasepool {
    auto device = MTLCreateSystemDefaultDevice();
    if (!device || ![device supportsFamily:MTLGPUFamilyMetal4]) {
      std::puts(
          "PENDING real Metal 4 transfer roundtrips, native clears, mips, "
          "aliases, "
          "custom block textures, shadow storage and framebuffer resolves");
      return 2;
    }
    Instance instance(0, false);
    Scheduler scheduler(instance.GetContext());
    MemoryBudget budget(64 * 1024 * 1024);
    MemoryAllocator allocator(instance.GetDevice(), budget);
    RenderManager manager(instance, scheduler, allocator);
    TextureRuntime runtime(instance, scheduler, manager, allocator, budget);
    RoundTrip(runtime);
    CopyFrameAndShadow(instance, runtime, manager);
    Custom(runtime);
    runtime.Finish();
  }
}
