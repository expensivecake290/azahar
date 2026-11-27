#include <array>
#include <cassert>
#include <cstdio>
#include <type_traits>
#include "video_core/renderer_metal/mtl_blit_helper.h"

using namespace Metal;

static_assert(std::is_same_v<decltype(&BlitHelper::DepthToBuffer),
                             bool (BlitHelper::*)(id<MTLTexture>, MemoryAllocator::GPUBufferAccess,
                                                  const VideoCore::BufferTextureCopy&, u32)>);
static_assert(std::is_same_v<decltype(&BlitHelper::ResolveTexture),
                             bool (BlitHelper::*)(const RenderTexture&, const RenderTexture&,
                                                  TextureAspect)>);

int main() {
    @autoreleasepool {
        constexpr std::array routes{TextureFilterRoute::Fallback, TextureFilterRoute::Anime4K,
                                    TextureFilterRoute::Bicubic,  TextureFilterRoute::ScaleForce,
                                    TextureFilterRoute::xBRZ,     TextureFilterRoute::MMPX};
        for (u32 i = 0; i < routes.size(); ++i) {
            auto filter = static_cast<Settings::TextureFilter>(i);
            assert(SelectTextureFilter(filter, 0) == routes[i]);
            assert(SelectTextureFilter(filter, 1) ==
                   (i == 0 ? TextureFilterRoute::Fallback : TextureFilterRoute::MipChain));
        }
        assert(SelectTextureFilter(static_cast<Settings::TextureFilter>(255), 0) ==
               TextureFilterRoute::Fallback);
        for (u32 samples : {1, 2, 4, 8, 16}) {
            assert(SelectDepthCopyRoute(samples, samples) ==
                   (samples == 1 ? DepthCopyRoute::Compute : DepthCopyRoute::PerSampleRender));
        }
        assert(SelectDepthCopyRoute(2, 1) == DepthCopyRoute::Invalid);
        assert(SelectDepthCopyRoute(0, 0) == DepthCopyRoute::Invalid);
        assert(SelectDepthCopyRoute(3, 3) == DepthCopyRoute::Invalid);

        VideoCore::TextureBlit blit{};
        blit.src_rect = {2, 6, 7, 1};
        blit.dst_rect = {9, 3, 0, 18};
        const auto mapping = MakeBlitSampling(blit, 10, 8);
        assert(mapping && mapping->scale[0] == -.5f && mapping->scale[1] == -.625f &&
               mapping->offset[0] == .7f && mapping->offset[1] == .75f);
        assert((BlitBounds(blit.dst_rect) == RenderArea{0, 3, 9, 15}));
        assert(!MakeBlitSampling(blit, 6, 8));
        blit.dst_rect.left = blit.dst_rect.right;
        assert(!MakeBlitSampling(blit, 10, 8));

        const VideoCore::BufferTextureCopy tight{4, 5 * 4, {2, 3, 7, 2}, 0};
        auto readback = MakeDepthReadbackRegion(tight, 7, 3);
        assert((readback && readback->source_offset == std::array<u32, 2>{2, 2} &&
                readback->extent == std::array<u32, 2>{5, 1} && readback->row_stride == 5));
        auto with_pitch = tight;
        with_pitch.texture_rect.top = 4;
        with_pitch.buffer_size = 12 * 4;
        assert(MakeDepthReadbackRegion(with_pitch, 7, 4, 7)->capacity == 12);
        assert(!MakeDepthReadbackRegion(with_pitch, 7, 4, 8));
        assert(!MakeDepthReadbackRegion(with_pitch, 7, 4, 4));
        ++with_pitch.buffer_offset;
        assert(!MakeDepthReadbackRegion(with_pitch, 7, 4, 7));
        auto native = [MTL4RenderPassDescriptor new];
        native.defaultRasterSampleCount = 4;
        assert(native.defaultRasterSampleCount == 4);
        std::puts("PASS Metal 4 utility filter, mirrored geometry, readback and sample route ABI");
    }
}
