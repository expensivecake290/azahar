#include <cassert>
#include <cstdio>
#include <type_traits>
#include "video_core/renderer_metal/mtl_texture_runtime.h"

using namespace Metal;

static_assert(std::is_base_of_v<VideoCore::SurfaceBase, TextureSurface>);
static_assert(std::is_base_of_v<VideoCore::FramebufferParams, Framebuffer>);
static_assert(std::is_nothrow_move_constructible_v<TextureSurface>);
static_assert(!std::is_copy_constructible_v<TextureSurface>);
static_assert(std::is_same_v<Traits::Surface, Metal::TextureSurface>);
static_assert(std::is_same_v<Traits::Runtime, Metal::TextureRuntime>);
static_assert(requires(TextureRuntime& rt, TextureSurface& surface, Framebuffer& fb,
                       VideoCore::TextureClear clear, VideoCore::TextureCopy copy,
                       VideoCore::TextureBlit blit, VideoCore::BufferTextureCopy download,
                       VideoCore::StagingData staging) {
    rt.FindStaging(1, true);
    rt.ClearTexture(surface, clear);
    rt.CopyTextures(surface, surface, copy);
    rt.BlitTextures(surface, surface, blit);
    surface.Upload(download, staging);
    surface.Download(download, staging);
    surface.ScaleUp(2, 4);
    { surface.ShadowImage() } -> std::same_as<Metal::TextureResource>;
    fb.MakePass(Metal::RenderArea{0, 0, 1, 1});
    fb.Samples();
    fb.Scale();
});

int main() {
    @autoreleasepool {
        const auto tight = MakeTextureBufferLayout({13, 7}, 1, 1, 4, 1);
        assert((tight == TextureBufferLayout{52, 52, 7, 364}));
        const auto padded = MakeTextureBufferLayout({13, 7}, 1, 1, 4, 256);
        assert((padded == TextureBufferLayout{52, 256, 7, 1792}));
        const auto compressed = MakeTextureBufferLayout({13, 7}, 4, 4, 16, 256);
        assert((compressed == TextureBufferLayout{64, 256, 2, 512}));
        const auto astc = MakeTextureBufferLayout({7, 9}, 6, 6, 16, 64);
        assert((astc == TextureBufferLayout{32, 64, 2, 128}));
        const auto tiny = MakeTextureBufferLayout({1, 1}, 8, 8, 16, 1);
        assert((tiny == TextureBufferLayout{16, 16, 1, 16}));
        assert(!MakeTextureBufferLayout({13, 7}, 1, 0, 4, 256));
        assert(!MakeTextureBufferLayout({13, 7}, 1, 1, 4, 255));
        assert(!MakeTextureBufferLayout({0, 7}, 1, 1, 1, 16));
        assert(!MakeTextureBufferLayout({0xffffffff, 0xffffffff}, 1, 1, 0xffffffff, 256));
        using TC = VideoCore::SamplerParams::TextureConfig;
        VideoCore::SamplerParams params{};
        params.border_color = 0x7f0050ff;
        params.lod_min = 1;
        params.lod_max = 6;
        params.lod_bias = -127;
        for (u32 i = 0; i < 8; ++i) {
            params.wrap_s = static_cast<TC::WrapMode>(i);
            params.wrap_t = TC::Repeat;
            const auto meta = GetSamplerMetadata(params);
            assert(meta.UsesGuestBorder() == (i == 1 || i == 5));
            assert(meta.wrap_s == i && meta.wrap_t == TC::Repeat);
            assert(meta.lod_min == 1 && meta.lod_max == 6 && meta.lod_bias == -127);
            assert((meta.border_color == std::array{1.f, 80 / 255.f, 0.f, 127 / 255.f}));
        }
        auto native = [MTL4RenderPassDescriptor new];
        native.colorAttachments[0].loadAction = MTLLoadActionLoad;
        native.colorAttachments[0].storeAction = MTLStoreActionStoreAndMultisampleResolve;
        assert(native.colorAttachments[0].storeAction == MTLStoreActionStoreAndMultisampleResolve);
        std::puts("PASS native texture CRTP interfaces, staging row pitches, format blocks, "
                  "overflow and exact PICA sampler border/LOD metadata");
    }
}
