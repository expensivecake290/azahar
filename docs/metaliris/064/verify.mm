#include <array>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <vector>
#include "video_core/renderer_metal/mtl_blit_helper.h"
#include "video_core/renderer_metal/mtl_shader_util.h"

using namespace Metal;
using namespace std::chrono_literals;
using Pixel = std::array<u8, 4>;

namespace {

constexpr u32 Width = 9;
constexpr u32 Height = 7;
constexpr Pixel Paint{51, 102, 153, 204};
constexpr Pixel Guard{1, 2, 3, 255};

id<MTLTexture> Texture(id<MTLDevice> device, MTLPixelFormat format, u32 width, u32 height,
                       MTLTextureUsage usage, MTLStorageMode mode = MTLStorageModeShared,
                       u32 samples = 1) {
    auto desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format
                                                                   width:width
                                                                  height:height
                                                               mipmapped:NO];
    desc.usage = usage;
    desc.storageMode = mode;
    desc.hazardTrackingMode = MTLHazardTrackingModeUntracked;
    if (samples > 1) {
        desc.textureType = MTLTextureType2DMultisample;
        desc.sampleCount = samples;
    }
    return CheckResource([device newTextureWithDescriptor:desc],
                         "Allocate utility fixture texture");
}

void Fill(id<MTLTexture> texture, Pixel color) {
    const auto width = texture.width;
    const auto height = texture.height;
    std::vector<Pixel> pixels(width * height, color);
    [texture replaceRegion:MTLRegionMake2D(0, 0, width, height)
               mipmapLevel:0
                 withBytes:pixels.data()
               bytesPerRow:width * sizeof(Pixel)];
}

std::vector<Pixel> Read(id<MTLTexture> texture) {
    std::vector<Pixel> pixels(texture.width * texture.height);
    [texture getBytes:pixels.data()
          bytesPerRow:texture.width * sizeof(Pixel)
           fromRegion:MTLRegionMake2D(0, 0, texture.width, texture.height)
          mipmapLevel:0];
    return pixels;
}

void TestFilters(id<MTLDevice> device, Scheduler& scheduler, BlitHelper& helper) {
    auto source = Texture(device, MTLPixelFormatRGBA8Unorm, 5, 3,
                          MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget);
    Fill(source, Paint);
    const VideoCore::TextureBlit rect{0, 0, 0, 0, {4, 3, 1, 0}, {1, 6, 8, 1}};
    for (auto filter : {Settings::TextureFilter::Anime4K, Settings::TextureFilter::Bicubic,
                        Settings::TextureFilter::ScaleForce, Settings::TextureFilter::xBRZ,
                        Settings::TextureFilter::MMPX}) {
        auto target =
            Texture(device, MTLPixelFormatRGBA8Unorm, Width, Height, MTLTextureUsageRenderTarget);
        Fill(target, Guard);
        assert(!helper.Filter(source, target, rect, Settings::TextureFilter::NoFilter, 2.0f));
        assert(helper.Filter(source, target, rect, filter, 2.0f));
        assert(scheduler.Finish({}, std::chrono::steady_clock::now() + 10s) ==
               CompletionStatus::Complete);
        const auto pixels = Read(target);
        for (u32 y = 0; y < Height; ++y) {
            for (u32 x = 0; x < Width; ++x) {
                const bool inside = x >= 1 && x < 8 && y >= 1 && y < 6;
                const auto expected = inside ? Paint : Guard;
                for (u32 i = 0; i < 4; ++i) {
                    assert(std::abs(int(pixels[y * Width + x][i]) - int(expected[i])) <= 1);
                }
            }
        }
    }
    auto alias = Texture(device, MTLPixelFormatRGBA8Unorm, Width, Height,
                         MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget);
    Fill(alias, Paint);
    assert(helper.Filter(alias, alias, rect, Settings::TextureFilter::Anime4K, 2.0f));
    assert(scheduler.Finish({}, std::chrono::steady_clock::now() + 10s) ==
           CompletionStatus::Complete);
    for (const auto& pixel : Read(alias)) {
        assert(pixel == Paint);
    }
}

RenderPass DepthPass(id<MTLTexture> target, float depth, u32 stencil) {
    RenderPass pass;
    pass.depth = {.target = {target}, .load_action = MTLLoadActionClear};
    pass.stencil = pass.depth;
    pass.area = {0, 0, Width, Height};
    pass.width = Width;
    pass.height = Height;
    pass.samples = static_cast<u32>(target.sampleCount);
    pass.clear_depth = depth;
    pass.clear_stencil = stencil;
    return pass;
}

void TestDepth(id<MTLDevice> device, Scheduler& scheduler, const MemoryAllocator& allocator,
               RenderManager& passes, BlitHelper& helper) {
    const auto usage =
        MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget | MTLTextureUsagePixelFormatView;
    auto depth = Texture(device, MTLPixelFormatDepth32Float_Stencil8, Width, Height, usage,
                         MTLStorageModePrivate);
    auto blitted = Texture(device, MTLPixelFormatDepth32Float_Stencil8, Width, Height, usage,
                           MTLStorageModePrivate);
    auto color = Texture(device, MTLPixelFormatRGBA8Unorm, Width, Height,
                         MTLTextureUsageShaderWrite | MTLTextureUsageRenderTarget);
    Fill(color, Guard);
    passes.BeginRendering(DepthPass(depth, .25f, 0x2d));
    passes.EndRendering();
    passes.BeginRendering(DepthPass(blitted, 1.f, 0x18));
    passes.EndRendering();

    const VideoCore::TextureBlit scaled{0, 0, 0, 0, {1, 5, 3, 2}, {2, 5, 7, 2}};
    assert(helper.BlitDepthStencil(depth, blitted, scaled));
    const VideoCore::TextureCopy area{0, 0, 0, 0, {2, 2}, {1, 1}, {5, 3}};
    assert(helper.ConvertDS24S8ToRGBA8(blitted, color, area));

    const VideoCore::BufferTextureCopy copy{4, 23 * 4, {2, 5, 7, 2}, 0};
    auto output = allocator.AllocateBuffer(25 * sizeof(u32), MemoryPool::Readback);
    {
        auto map = output.MapWrite({0, output.Size()});
        std::memset(map.Bytes().data(), 0xef, map.Bytes().size());
    }
    assert(helper.DepthToBuffer(
        blitted, output.AcquireGPUAccess({0, output.Size()}, BufferAccess::Write), copy, 9));
    assert(scheduler.Finish({}, std::chrono::steady_clock::now() + 10s) ==
           CompletionStatus::Complete);
    {
        auto map = output.MapRead({0, output.Size()});
        std::array<u32, 25> words;
        std::memcpy(words.data(), map.Bytes().data(), map.Bytes().size());
        for (u32 i = 0; i < words.size(); ++i) {
            const bool written = i >= 1 && i <= 23 && (i - 1) % 9 < 5;
            assert(words[i] == (written ? 0x3fffff2d : 0xefefefef));
        }
    }
    const auto pixels = Read(color);
    for (u32 y = 0; y < Height; ++y) {
        for (u32 x = 0; x < Width; ++x) {
            const bool written = x >= 1 && x <= 5 && y >= 1 && y <= 3;
            const Pixel expected = written ? Pixel{0x2d, 0x40, 0x00, 0x00} : Guard;
            assert(pixels[y * Width + x] == expected);
        }
    }
}

void TestMSAA(Instance& instance, Scheduler& scheduler, RenderManager& passes, BlitHelper& helper,
              const char* fixture) {
    constexpr u32 samples = 4;
    if (!instance.SupportsSampleCount(samples)) {
        std::puts("PENDING native 4x samples unavailable on this Metal device");
        return;
    }

    auto device = instance.GetDevice();
    auto& context = instance.GetContext();
    const auto depth_usage =
        MTLTextureUsageShaderRead | MTLTextureUsagePixelFormatView | MTLTextureUsageRenderTarget;
    const auto color_usage = MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget;
    auto depth = Texture(device, MTLPixelFormatDepth32Float_Stencil8, Width, Height, depth_usage,
                         MTLStorageModePrivate, samples);
    auto color = Texture(device, MTLPixelFormatRGBA8Unorm, Width, Height, color_usage,
                         MTLStorageModePrivate, samples);
    auto resolved = Texture(device, MTLPixelFormatRGBA8Unorm, Width, Height, color_usage);
    auto resolved_again = Texture(device, MTLPixelFormatRGBA8Unorm, Width, Height, color_usage);
    const auto depth_resolved = Texture(device, MTLPixelFormatDepth32Float_Stencil8, Width, Height,
                                        depth_usage, MTLStorageModePrivate);

    std::ifstream input(fixture, std::ios::binary);
    assert(input);
    const std::vector<u8> bytes{std::istreambuf_iterator<char>{input}, {}};
    ShaderCompiler compiler(context);
    const auto loaded = compiler.LoadLibrary(bytes, "phase 064 sample verification");
    const auto library = loaded.GetLibrary(context);
    assert(library);
    const auto function = [&](NSString* entry) {
        auto descriptor = [MTL4LibraryFunctionDescriptor new];
        descriptor.library = library;
        descriptor.name = entry;
        return descriptor;
    };
    const auto compiler_native = compiler.GetNativeCompiler(context.AcquireWork());
    auto render_desc = [MTL4RenderPipelineDescriptor new];
    render_desc.vertexFunctionDescriptor = function(@"fill_depth_vertex");
    render_desc.fragmentFunctionDescriptor = function(@"fill_depth_samples");
    render_desc.rasterSampleCount = samples;
    NSError* error = nil;
    auto fill = [compiler_native newRenderPipelineStateWithDescriptor:render_desc
                                                  compilerTaskOptions:nil
                                                                error:&error];
    fill = CheckResource(fill, "Build depth/stencil sample fixture", error);
    auto depth_desc = [MTLDepthStencilDescriptor new];
    depth_desc.depthCompareFunction = MTLCompareFunctionAlways;
    depth_desc.depthWriteEnabled = YES;
    auto stencil_desc = [MTLStencilDescriptor new];
    stencil_desc.stencilCompareFunction = MTLCompareFunctionAlways;
    stencil_desc.depthStencilPassOperation = MTLStencilOperationReplace;
    depth_desc.frontFaceStencil = depth_desc.backFaceStencil = stencil_desc;
    auto depth_state = CheckResource([device newDepthStencilStateWithDescriptor:depth_desc],
                                     "Create multisample fixture depth state");
    passes.BeginRendering(DepthPass(depth, 1.f, 0));
    passes.RecordDraw([=](EncodingContext& ctx, id<MTL4RenderCommandEncoder> enc) {
        [enc setRenderPipelineState:fill];
        [enc setDepthStencilState:depth_state];
        [enc setViewport:MTLViewport{0, 0, Width, Height, 0, 1}];
        [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
        ctx.Retain(fill);
        ctx.Retain(depth_state);
    });
    passes.EndRendering();
    const VideoCore::TextureCopy copy{0, 0, 0, 0, {}, {}, {Width, Height}};
    assert(helper.ConvertDS24S8ToRGBA8(depth, color, copy, {resolved}));
    assert(helper.ResolveTexture({color}, {resolved_again}, TextureAspect::Color));
    assert(helper.ResolveTexture({depth}, {depth_resolved}, TextureAspect::DepthStencil));
    auto sample_zero =
        Texture(device, MTLPixelFormatRGBA8Unorm, Width, Height, MTLTextureUsageShaderWrite);
    assert(helper.ConvertDS24S8ToRGBA8(depth_resolved, sample_zero, copy));

    auto compute_desc = [MTL4ComputePipelineDescriptor new];
    compute_desc.computeFunctionDescriptor = function(@"inspect_samples");
    auto read = [compiler_native newComputePipelineStateWithDescriptor:compute_desc
                                                   compilerTaskOptions:nil
                                                                 error:&error];
    read = CheckResource(read, "Build individual sample readback", error);
    auto args = [MTL4ArgumentTableDescriptor new];
    args.maxBufferBindCount = args.maxTextureBindCount = 1;
    auto table = [device newArgumentTableWithDescriptor:args error:&error];
    table = CheckResource(table, "Create sample readback table", error);
    auto values = CheckResource([device newBufferWithLength:sizeof(u32) * samples
                                                    options:MTLResourceStorageModeShared |
                                                            MTLResourceHazardTrackingModeUntracked],
                                "Create sample readback buffer");
    [table setAddress:values.gpuAddress atIndex:0];
    [table setTexture:color.gpuResourceID atIndex:0];
    scheduler.Record([=](EncodingContext& ctx) {
        const std::array uses{ResourceUse{color, BufferAccess::Read, MTLStageDispatch},
                              ResourceUse{values, BufferAccess::Write, MTLStageDispatch}};
        auto enc = [ctx.GetCommandBuffer() computeCommandEncoder];
        ctx.Synchronize(enc, uses);
        [enc setComputePipelineState:read];
        [enc setArgumentTable:table];
        [enc dispatchThreads:MTLSizeMake(samples, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(samples, 1, 1)];
        ctx.Retain(read);
        ctx.Retain(table);
        ctx.EndPass(enc);
    });

    assert(scheduler.Finish({}, std::chrono::steady_clock::now() + 10s) ==
           CompletionStatus::Complete);
    const auto* observed = static_cast<const u32*>(values.contents);
    for (u32 i = 0; i < samples; ++i) {
        assert(observed[i] == ((i + 1) * 0x2000 | (i + 0x2d)));
    }
    assert(Read(resolved) == Read(resolved_again));
    for (const auto& pixel : Read(resolved)) {
        assert(std::abs(int(pixel[0]) - 47) <= 1 && pixel[1] == 0x50 && pixel[2] == 0 &&
               pixel[3] == 0);
    }
    const Pixel expected{0x2d, 0x20, 0, 0};
    for (const auto& pixel : Read(sample_zero)) {
        assert(pixel == expected);
    }
}

} // Anonymous namespace

int main(int argc, const char* argv[]) {
    @autoreleasepool {
        assert(argc == 2);
        auto device = MTLCreateSystemDefaultDevice();
        if (!device || ![device supportsFamily:MTLGPUFamilyMetal4]) {
            std::puts("PENDING real Metal 4 depth/stencil, packed readback, per-sample resolve, "
                      "alias and all texture-filter runs");
            return 2;
        }
        Instance instance(0, false);
        Scheduler scheduler(instance.GetContext());
        MemoryAllocator allocator(device, MemoryBudget(32 * 1024 * 1024));
        RenderManager passes(instance, scheduler, allocator);
        BlitHelper helper(instance, scheduler, passes, allocator);
        TestFilters(device, scheduler, helper);
        TestDepth(device, scheduler, allocator, passes, helper);
        TestMSAA(instance, scheduler, passes, helper, argv[1]);
        std::puts("PASS live Metal 4 utility known-answer output");
    }
}
