// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.txt file included.

#include <array>
#include <cmath>
#include <cstring>
#include <map>
#include <span>
#include <utility>
#include <vector>
#include "video_core/host_shaders/metal/bicubic_frag.h"
#include "video_core/host_shaders/metal/blit_depth_stencil_frag.h"
#include "video_core/host_shaders/metal/d24s8_to_rgba8_comp.h"
#include "video_core/host_shaders/metal/d24s8_to_rgba8_ms_comp.h"
#include "video_core/host_shaders/metal/depth_to_buffer_comp.h"
#include "video_core/host_shaders/metal/full_screen_triangle_vert.h"
#include "video_core/host_shaders/metal/mmpx_frag.h"
#include "video_core/host_shaders/metal/refine_frag.h"
#include "video_core/host_shaders/metal/scale_force_frag.h"
#include "video_core/host_shaders/metal/x_gradient_frag.h"
#include "video_core/host_shaders/metal/xbrz_freescale_frag.h"
#include "video_core/host_shaders/metal/y_gradient_frag.h"
#include "video_core/renderer_metal/mtl_blit_helper.h"
#include "video_core/renderer_metal/mtl_descriptor_update_queue.h"
#include "video_core/renderer_metal/mtl_shader_util.h"

namespace Metal {

namespace {

enum class Utility : u8 {
    FullScreen,
    SingleSamplePack,
    MultisamplePack,
    DepthReadback,
    DepthStencilBlit,
    Bicubic,
    ScaleForce,
    Xbrz,
    Mmpx,
    XGradient,
    YGradient,
    Refine,
};

struct Program {
    std::span<const u8> bytes;
    const char* entry;
};

constexpr Program GetProgram(Utility utility) {
    using namespace HostShaders::Metal;
    switch (utility) {
    case Utility::FullScreen:
        return {FULL_SCREEN_TRIANGLE_VERT, "full_screen_triangle_vert"};
    case Utility::SingleSamplePack:
        return {D24S8_TO_RGBA8_COMP, "d24s8_to_rgba8_comp"};
    case Utility::MultisamplePack:
        return {D24S8_TO_RGBA8_MS_COMP, "d24s8_to_rgba8_ms_fragment"};
    case Utility::DepthReadback:
        return {DEPTH_TO_BUFFER_COMP, "depth_to_buffer_comp"};
    case Utility::DepthStencilBlit:
        return {BLIT_DEPTH_STENCIL_FRAG, "blit_depth_stencil_frag"};
    case Utility::Bicubic:
        return {BICUBIC_FRAG, "bicubic_frag"};
    case Utility::ScaleForce:
        return {SCALE_FORCE_FRAG, "scale_force_frag"};
    case Utility::Xbrz:
        return {XBRZ_FREESCALE_FRAG, "xbrz_freescale_frag"};
    case Utility::Mmpx:
        return {MMPX_FRAG, "mmpx_frag"};
    case Utility::XGradient:
        return {X_GRADIENT_FRAG, "x_gradient_frag"};
    case Utility::YGradient:
        return {Y_GRADIENT_FRAG, "y_gradient_frag"};
    case Utility::Refine:
        return {REFINE_FRAG, "refine_frag"};
    }
    throw std::invalid_argument("Unknown native texture utility shader");
}

struct TextureRead {
    u32 slot;
    id<MTLTexture> texture;
};

struct alignas(8) XbrzUniform {
    BlitSampling sampling;
    float resolution_scale;
};
static_assert(offsetof(XbrzUniform, resolution_scale) == 16);

using BufferLease = MemoryAllocator::GPUBufferAccess;

u32 MipSize(NSUInteger dimension, u32 level) {
    return static_cast<u32>(std::max<NSUInteger>(1, dimension >> level));
}

bool DeviceTexture(id<MTLTexture> texture, id<MTLDevice> device) {
    return texture && texture.device == device &&
           texture.width <= std::numeric_limits<u32>::max() &&
           texture.height <= std::numeric_limits<u32>::max() && !texture.framebufferOnly &&
           texture.storageMode != MTLStorageModeMemoryless;
}

bool HasUsage(id<MTLTexture> texture, MTLTextureUsage usage) {
    return texture.usage == MTLTextureUsageUnknown || (texture.usage & usage) == usage;
}

bool InRange(const RenderTexture& subresource) {
    const auto texture = subresource.texture;
    if (!texture || subresource.depth_plane != 0 || subresource.level >= texture.mipmapLevelCount ||
        subresource.level >= 32) {
        return false;
    }
    switch (texture.textureType) {
    case MTLTextureType2D:
    case MTLTextureType2DMultisample:
        return subresource.slice == 0;
    case MTLTextureType2DArray:
    case MTLTextureType2DMultisampleArray:
        return subresource.slice < texture.arrayLength;
    case MTLTextureTypeCube:
    case MTLTextureTypeCubeArray:
        return subresource.slice < u64{texture.arrayLength} * 6;
    default:
        return false;
    }
}

id<MTLTexture> Backing(id<MTLTexture> texture) {
    while (texture.parentTexture) {
        texture = texture.parentTexture;
    }
    return texture;
}

RenderArea FullArea(RenderTexture texture) {
    return {0, 0, MipSize(texture.texture.width, texture.level),
            MipSize(texture.texture.height, texture.level)};
}

bool IsDepth(MTLPixelFormat format) {
    return format == MTLPixelFormatDepth16Unorm || format == MTLPixelFormatDepth32Float ||
           format == MTLPixelFormatDepth32Float_Stencil8;
}

bool IsStencil(MTLPixelFormat format) {
    return format == MTLPixelFormatStencil8 || format == MTLPixelFormatDepth32Float_Stencil8;
}

id<MTLTexture> ShaderView(const RenderTexture& source, MTLPixelFormat format) {
    const auto texture = source.texture;
    const auto view_type =
        texture.sampleCount == 1 ? MTLTextureType2D : MTLTextureType2DMultisample;
    if (source.level == 0 && texture.mipmapLevelCount == 1 && source.slice == 0 &&
        source.depth_plane == 0 && texture.pixelFormat == format &&
        texture.textureType == view_type) {
        return texture;
    }
    return [texture newTextureViewWithPixelFormat:format
                                      textureType:view_type
                                           levels:NSMakeRange(source.level, 1)
                                           slices:NSMakeRange(source.slice, 1)];
}

RenderPass MakePass(RenderTexture destination, RenderArea area, TextureAspect aspect,
                    RenderTexture resolve = {}) {
    const auto full = FullArea(destination);
    RenderAttachment attachment{.target = destination, .resolve = resolve};
    if (resolve.texture) {
        attachment.store_action = MTLStoreActionStoreAndMultisampleResolve;
    }
    RenderPass pass;
    pass.area = area;
    pass.width = full.width;
    pass.height = full.height;
    pass.samples = static_cast<u32>(destination.texture.sampleCount);
    if (aspect == TextureAspect::Color) {
        pass.color = attachment;
    } else {
        pass.depth = attachment;
        if (aspect == TextureAspect::DepthStencil) {
            pass.stencil = attachment;
        }
    }
    return pass;
}

} // Anonymous namespace

struct BlitHelper::State {
    struct PipelineKey {
        Utility shader;
        MTLPixelFormat format;
        u32 samples;
        auto operator<=>(const PipelineKey&) const = default;
    };

    State(Instance& instance, Scheduler& scheduler, RenderManager& passes,
          const MemoryAllocator& allocator)
        : context(instance.GetContext()), device(context.GetDevice()), scheduler(scheduler),
          passes(passes), allocator(allocator), shader_compiler(context, 0), descriptors(context) {}

    void Refresh() {
        if (context.GetLeaseStatus(work) == WorkLeaseStatus::Valid) {
            return;
        }
        work = context.AcquireWork();
        if (context.GetLeaseStatus(work) != WorkLeaseStatus::Valid) {
            throw std::logic_error("Metal utility work requires an active device generation");
        }
        libraries.clear();
        render_pipelines.clear();
        compute_pipelines.clear();
        depth_stencil = nil;
        linear = nil;
        x_gradient = nil;
        luminance = nil;
    }

    bool Valid(RenderTexture texture, MTLTextureUsage usage) const {
        return DeviceTexture(texture.texture, device) && InRange(texture) &&
               HasUsage(texture.texture, usage);
    }

    bool CanResolve(RenderTexture source, RenderTexture destination) const {
        return Valid(source, MTLTextureUsageRenderTarget) &&
               Valid(destination, MTLTextureUsageRenderTarget) && source.texture.sampleCount > 1 &&
               destination.texture.sampleCount == 1 &&
               source.texture.pixelFormat == destination.texture.pixelFormat &&
               Backing(source.texture) != Backing(destination.texture) &&
               FullArea(source) == FullArea(destination);
    }

    MTL4LibraryFunctionDescriptor* Function(Utility shader, const char* entry = nullptr) {
        const auto program = GetProgram(shader);
        auto& library = libraries[shader];
        if (!library) {
            const auto result = shader_compiler.LoadLibrary(program.bytes, program.entry);
            if (auto loaded = result.GetLibrary(context)) {
                library = loaded;
            } else {
                throw std::runtime_error("Unable to load native utility shader " +
                                         std::string(program.entry) + ": " +
                                         result.GetDiagnostic().message);
            }
        }
        auto descriptor = [MTL4LibraryFunctionDescriptor new];
        descriptor.library = library;
        descriptor.name = @(entry ? entry : program.entry);
        return descriptor;
    }

    id<MTLComputePipelineState> Compute(Utility shader) {
        auto& pipeline = compute_pipelines[shader];
        if (!pipeline) {
            auto compiler = CheckResource(shader_compiler.GetNativeCompiler(work),
                                          "Get Metal 4 utility compiler");
            auto descriptor = [MTL4ComputePipelineDescriptor new];
            descriptor.computeFunctionDescriptor = Function(shader);
            auto guard = context.AcquireSubmission(work);
            if (!guard) {
                throw std::logic_error("Stale Metal utility compute compilation");
            }
            NSError* error = nil;
            auto built = [compiler newComputePipelineStateWithDescriptor:descriptor
                                                     compilerTaskOptions:nil
                                                                   error:&error];
            pipeline = CheckResource(built, "Compile Metal 4 utility compute pipeline", error);
        }
        return pipeline;
    }

    id<MTLRenderPipelineState> Render(Utility shader, MTLPixelFormat format, u32 samples) {
        auto& pipeline = render_pipelines[{shader, format, samples}];
        if (!pipeline) {
            auto compiler = CheckResource(shader_compiler.GetNativeCompiler(work),
                                          "Get Metal 4 utility compiler");
            auto descriptor = [MTL4RenderPipelineDescriptor new];
            descriptor.vertexFunctionDescriptor = shader == Utility::MultisamplePack
                                                      ? Function(shader, "d24s8_to_rgba8_ms_vertex")
                                                      : Function(Utility::FullScreen);
            descriptor.fragmentFunctionDescriptor = Function(shader);
            descriptor.colorAttachments[0].pixelFormat = format;
            descriptor.rasterSampleCount = samples;
            descriptor.inputPrimitiveTopology = MTLPrimitiveTopologyClassTriangle;
            auto guard = context.AcquireSubmission(work);
            if (!guard) {
                throw std::logic_error("Stale Metal utility render compilation");
            }
            NSError* error = nil;
            auto built = [compiler newRenderPipelineStateWithDescriptor:descriptor
                                                    compilerTaskOptions:nil
                                                                  error:&error];
            pipeline = CheckResource(built, "Compile Metal 4 utility render pipeline", error);
        }
        return pipeline;
    }

    id<MTLDepthStencilState> DepthStencil() {
        if (!depth_stencil) {
            auto descriptor = [MTLDepthStencilDescriptor new];
            descriptor.depthCompareFunction = MTLCompareFunctionAlways;
            descriptor.depthWriteEnabled = YES;
            auto stencil = [MTLStencilDescriptor new];
            stencil.stencilCompareFunction = MTLCompareFunctionAlways;
            stencil.depthStencilPassOperation = MTLStencilOperationReplace;
            stencil.writeMask = 0xff;
            descriptor.frontFaceStencil = stencil;
            descriptor.backFaceStencil = stencil;
            depth_stencil = CheckResource([device newDepthStencilStateWithDescriptor:descriptor],
                                          "Create native depth/stencil replacement state");
        }
        return depth_stencil;
    }

    id<MTLSamplerState> Linear() {
        if (!linear) {
            auto descriptor = [MTLSamplerDescriptor new];
            descriptor.minFilter = MTLSamplerMinMagFilterLinear;
            descriptor.magFilter = MTLSamplerMinMagFilterLinear;
            descriptor.sAddressMode = MTLSamplerAddressModeClampToEdge;
            descriptor.tAddressMode = MTLSamplerAddressModeClampToEdge;
            descriptor.supportArgumentBuffers = YES;
            linear = CheckResource([device newSamplerStateWithDescriptor:descriptor],
                                   "Create utility Metal clamp sampler");
        }
        return linear;
    }

    id<MTLTexture> Temporary(MTLPixelFormat format, RenderArea area, MTLTextureUsage usage) const {
        auto descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format
                                                                             width:area.width
                                                                            height:area.height
                                                                         mipmapped:NO];
        descriptor.storageMode = MTLStorageModePrivate;
        descriptor.hazardTrackingMode = MTLHazardTrackingModeUntracked;
        descriptor.usage = usage;
        return CheckResource([device newTextureWithDescriptor:descriptor],
                             "Allocate private Metal texture utility storage");
    }

    RenderTexture SnapshotIfAliased(RenderTexture source, id<MTLTexture> destination) {
        if (Backing(source.texture) != Backing(destination)) {
            return source;
        }
        // Filter and scaled depth/stencil operations read a one-sample source. Copy its complete
        // selected subresource first so neighborhood taps and native depth flags remain readable.
        const auto area = FullArea(source);
        const auto temporary = Temporary(source.texture.pixelFormat, area, source.texture.usage);
        passes.EndRendering();
        scheduler.Record([source, temporary, area](EncodingContext& context) {
            const std::array uses{ResourceUse{source.texture, BufferAccess::Read, MTLStageBlit},
                                  ResourceUse{temporary, BufferAccess::Write, MTLStageBlit}};
            auto encoder = CheckResource([context.GetCommandBuffer() computeCommandEncoder],
                                         "Encode aliased Metal utility snapshot");
            context.Synchronize(encoder, uses);
            [encoder copyFromTexture:source.texture
                         sourceSlice:source.slice
                         sourceLevel:source.level
                        sourceOrigin:MTLOriginMake(0, 0, 0)
                          sourceSize:MTLSizeMake(area.width, area.height, 1)
                           toTexture:temporary
                    destinationSlice:0
                    destinationLevel:0
                   destinationOrigin:MTLOriginMake(0, 0, 0)];
            context.EndPass(encoder);
        });
        return {temporary};
    }

    BufferLease Uniform(const DescriptorUpdateQueue::Set& set, u32 index,
                        std::span<const std::byte> bytes) {
        auto storage = allocator.AllocateBuffer(bytes.size(), MemoryPool::Transient);
        const BufferRange range{0, bytes.size()};
        {
            const auto mapped = storage.MapWrite(range);
            std::memcpy(mapped.Bytes().data(), bytes.data(), bytes.size());
        }
        auto access = storage.AcquireGPUAccess(range, BufferAccess::Read);
        descriptors.AddBuffer(set, index, access.GetBuffer(), 0, bytes.size());
        return access;
    }

    void Draw(Utility shader, const RenderPass& pass, std::optional<BlitSampling> sampling,
              std::span<const TextureRead> reads, bool linear_sample,
              std::span<const std::byte> fragment_uniform = {}) {
        const auto pipeline = Render(shader, pass.Formats().color, pass.samples);
        const auto stencil = shader == Utility::DepthStencilBlit ? DepthStencil() : nil;

        std::vector<BufferLease> parameters;
        std::optional<DescriptorUpdateQueue::Snapshot> vertex;
        if (sampling) {
            const auto set = descriptors.CreateSet({.buffers = 1});
            parameters.push_back(Uniform(set, 0, std::as_bytes(std::span{&*sampling, 1})));
            vertex.emplace(descriptors.Capture(set));
        }

        u32 texture_slots = 0;
        for (const auto& read : reads) {
            texture_slots = std::max(texture_slots, read.slot + 1);
        }
        const auto set = descriptors.CreateSet({.buffers = fragment_uniform.empty() ? 0u : 1u,
                                                .textures = texture_slots,
                                                .samplers = linear_sample ? 1u : 0u});
        bool sampler_bound = false;
        for (const auto& read : reads) {
            if (linear_sample && !sampler_bound) {
                descriptors.AddImageSampler(set, read.slot, 0, read.texture, Linear(), 0);
                sampler_bound = true;
            } else {
                descriptors.AddStorageImage(set, read.slot, read.texture, BufferAccess::Read);
            }
        }
        if (!fragment_uniform.empty()) {
            parameters.push_back(Uniform(set, 0, fragment_uniform));
        }

        auto fragment = descriptors.Capture(set);
        passes.BeginRendering(pass);
        passes.RecordDraw(
            [pipeline, stencil, vertex, fragment, area = pass.area,
             parameters = std::move(parameters)](EncodingContext& context,
                                                 id<MTL4RenderCommandEncoder> encoder) mutable {
                if (vertex) {
                    vertex->BindRender(context, encoder, MTLRenderStageVertex);
                }
                fragment.BindRender(context, encoder, MTLRenderStageFragment);
                [encoder setRenderPipelineState:pipeline];
                [encoder setDepthStencilState:stencil];
                [encoder setCullMode:MTLCullModeNone];
                [encoder setViewport:MTLViewport{double(area.x), double(area.y), double(area.width),
                                                 double(area.height), 0, 1}];
                [encoder setScissorRect:MTLScissorRect{area.x, area.y, area.width, area.height}];
                [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
                context.Retain(pipeline);
                context.Retain(stencil);
                for (auto& access : parameters) {
                    context.Retain(std::move(access));
                }
            });
        passes.EndRendering();
    }

    void Dispatch(id<MTLComputePipelineState> pipeline, DescriptorUpdateQueue::Snapshot bindings,
                  u32 width, u32 height, std::vector<BufferLease> accesses) {
        passes.EndRendering();
        scheduler.Record([pipeline, bindings, width, height,
                          accesses = std::move(accesses)](EncodingContext& context) mutable {
            auto encoder = CheckResource([context.GetCommandBuffer() computeCommandEncoder],
                                         "Encode native texture utility compute pass");
            bindings.BindCompute(context, encoder);
            [encoder setComputePipelineState:pipeline];
            const NSUInteger thread_width = std::max<NSUInteger>(1, pipeline.threadExecutionWidth);
            const NSUInteger thread_height = std::max<NSUInteger>(
                1, std::min<NSUInteger>(8, pipeline.maxTotalThreadsPerThreadgroup / thread_width));
            [encoder dispatchThreads:MTLSizeMake(width, height, 1)
                threadsPerThreadgroup:MTLSizeMake(thread_width, thread_height, 1)];
            context.Retain(pipeline);
            for (auto& access : accesses) {
                context.Retain(std::move(access));
            }
            context.EndPass(encoder);
        });
    }

    DeviceContext& context;
    const id<MTLDevice> device;
    Scheduler& scheduler;
    RenderManager& passes;
    const MemoryAllocator& allocator;
    ShaderCompiler shader_compiler;
    DescriptorUpdateQueue descriptors;
    DeviceContext::WorkLease work;
    std::map<Utility, id<MTLLibrary>> libraries;
    std::map<Utility, id<MTLComputePipelineState>> compute_pipelines;
    std::map<PipelineKey, id<MTLRenderPipelineState>> render_pipelines;
    id<MTLDepthStencilState> depth_stencil = nil;
    id<MTLSamplerState> linear = nil;
    id<MTLTexture> x_gradient = nil;
    id<MTLTexture> luminance = nil;
};

BlitHelper::BlitHelper(Instance& instance, Scheduler& scheduler, RenderManager& passes,
                       const MemoryAllocator& allocator)
    : state(std::make_unique<State>(instance, scheduler, passes, allocator)) {}

BlitHelper::~BlitHelper() = default;

bool BlitHelper::Filter(id<MTLTexture> source, id<MTLTexture> destination,
                        const VideoCore::TextureBlit& blit, Settings::TextureFilter filter,
                        float resolution_scale) {
    const auto route = SelectTextureFilter(filter, blit.src_level);
    if (route == TextureFilterRoute::Fallback) {
        return false;
    }
    if (route == TextureFilterRoute::MipChain) {
        return true;
    }

    RenderTexture src{source, blit.src_level, blit.src_layer};
    const RenderTexture dst{destination, blit.dst_level, blit.dst_layer};
    if (!state->Valid(src, MTLTextureUsageShaderRead) ||
        !state->Valid(dst, MTLTextureUsageRenderTarget) || source.sampleCount != 1 ||
        destination.sampleCount != 1 || IsDepth(destination.pixelFormat) ||
        IsStencil(destination.pixelFormat) || !std::isfinite(resolution_scale) ||
        resolution_scale <= 0.0f) {
        return false;
    }

    const auto source_area = FullArea(src);
    const auto target = FullArea(dst);
    const auto area = BlitBounds(blit.dst_rect);
    const auto sampling = MakeBlitSampling(blit, source_area.width, source_area.height);
    if (!sampling || !area.Fits(target.width, target.height)) {
        return false;
    }

    state->Refresh();
    src = state->SnapshotIfAliased(src, destination);
    const auto image = CheckResource(ShaderView(src, src.texture.pixelFormat),
                                     "Create native texture-filter source view");
    const auto pass = MakePass(dst, area, TextureAspect::Color);
    const std::array input{TextureRead{0, image}};
    switch (route) {
    case TextureFilterRoute::Anime4K: {
        if (state->x_gradient.width != source_area.width ||
            state->x_gradient.height != source_area.height) {
            const auto usage = MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget;
            auto x = state->Temporary(MTLPixelFormatRG16Float, source_area, usage);
            auto y = state->Temporary(MTLPixelFormatR16Float, source_area, usage);
            state->x_gradient = x;
            state->luminance = y;
        }

        constexpr BlitSampling identity{{1, 1}, {0, 0}};
        const auto first = MakePass({state->x_gradient}, source_area, TextureAspect::Color);
        const auto second = MakePass({state->luminance}, source_area, TextureAspect::Color);
        const std::array gradient{TextureRead{2, state->x_gradient}};
        const std::array refine{TextureRead{0, image}, TextureRead{1, state->luminance}};
        state->Draw(Utility::XGradient, first, identity, input, true);
        state->Draw(Utility::YGradient, second, identity, gradient, true);
        state->Draw(Utility::Refine, pass, sampling, refine, true);
        break;
    }
    case TextureFilterRoute::Bicubic:
        state->Draw(Utility::Bicubic, pass, sampling, input, true);
        break;
    case TextureFilterRoute::ScaleForce:
        state->Draw(Utility::ScaleForce, pass, sampling, input, true);
        break;
    case TextureFilterRoute::xBRZ: {
        const XbrzUniform uniform{*sampling, resolution_scale};
        state->Draw(Utility::Xbrz, pass, sampling, input, false,
                    std::as_bytes(std::span{&uniform, 1}));
        break;
    }
    case TextureFilterRoute::MMPX:
        state->Draw(Utility::Mmpx, pass, sampling, input, false);
        break;
    default:
        return false;
    }
    return true;
}

bool BlitHelper::BlitDepthStencil(id<MTLTexture> source, id<MTLTexture> destination,
                                  const VideoCore::TextureBlit& blit) {
    RenderTexture src{source, blit.src_level, blit.src_layer};
    const RenderTexture dst{destination, blit.dst_level, blit.dst_layer};
    if (!state->Valid(src, MTLTextureUsageShaderRead | MTLTextureUsagePixelFormatView) ||
        !state->Valid(dst, MTLTextureUsageRenderTarget) || source.sampleCount != 1 ||
        destination.sampleCount != 1 || source.pixelFormat != MTLPixelFormatDepth32Float_Stencil8 ||
        destination.pixelFormat != MTLPixelFormatDepth32Float_Stencil8) {
        return false;
    }
    const auto full_src = FullArea(src);
    const auto full_dst = FullArea(dst);
    const auto area = BlitBounds(blit.dst_rect);
    const auto sampling = MakeBlitSampling(blit, full_src.width, full_src.height);
    if (!sampling || !area.Fits(full_dst.width, full_dst.height)) {
        return false;
    }

    state->Refresh();
    src = state->SnapshotIfAliased(src, destination);
    const auto depth =
        CheckResource(ShaderView(src, src.texture.pixelFormat), "Create native depth blit input");
    const auto stencil = CheckResource(ShaderView(src, MTLPixelFormatX32_Stencil8),
                                       "Create native stencil blit input");
    const std::array textures{TextureRead{0, depth}, TextureRead{1, stencil}};
    state->Draw(Utility::DepthStencilBlit, MakePass(dst, area, TextureAspect::DepthStencil),
                sampling, textures, false);
    return true;
}

bool BlitHelper::ConvertDS24S8ToRGBA8(id<MTLTexture> source, id<MTLTexture> destination,
                                      const VideoCore::TextureCopy& copy, RenderTexture resolve) {
    const RenderTexture src{source, copy.src_level, copy.src_layer};
    const RenderTexture dst{destination, copy.dst_level, copy.dst_layer};
    const auto route = SelectDepthCopyRoute(static_cast<u32>(source.sampleCount),
                                            static_cast<u32>(destination.sampleCount));
    if (route == DepthCopyRoute::Invalid ||
        !state->Valid(src, MTLTextureUsageShaderRead | MTLTextureUsagePixelFormatView) ||
        source.pixelFormat != MTLPixelFormatDepth32Float_Stencil8 ||
        destination.pixelFormat != MTLPixelFormatRGBA8Unorm ||
        !state->Valid(dst, route == DepthCopyRoute::Compute ? MTLTextureUsageShaderWrite
                                                            : MTLTextureUsageRenderTarget) ||
        (resolve.texture && !state->CanResolve(dst, resolve))) {
        return false;
    }
    const RenderArea source_area{copy.src_offset.x, copy.src_offset.y, copy.extent.width,
                                 copy.extent.height};
    const RenderArea destination_area{copy.dst_offset.x, copy.dst_offset.y, copy.extent.width,
                                      copy.extent.height};
    const auto source_full = FullArea(src);
    const auto destination_full = FullArea(dst);
    if (!source_area.Fits(source_full.width, source_full.height) ||
        !destination_area.Fits(destination_full.width, destination_full.height)) {
        return false;
    }

    state->Refresh();
    const auto depth =
        CheckResource(ShaderView(src, source.pixelFormat), "Read packed Metal depth");
    const auto stencil =
        CheckResource(ShaderView(src, MTLPixelFormatX32_Stencil8), "Read packed Metal stencil");
    const DepthStencilCopyRegion parameters{{source_area.x, source_area.y},
                                            {destination_area.x, destination_area.y},
                                            {source_area.width, source_area.height}};
    const auto uniform = std::as_bytes(std::span{&parameters, 1});
    if (route == DepthCopyRoute::PerSampleRender) {
        const std::array textures{TextureRead{0, depth}, TextureRead{1, stencil}};
        state->Draw(Utility::MultisamplePack,
                    MakePass(dst, destination_area, TextureAspect::Color, resolve), {}, textures,
                    false, uniform);
        return true;
    }
    const auto color = CheckResource(ShaderView(dst, destination.pixelFormat),
                                     "Create packed Metal color output view");
    const auto set = state->descriptors.CreateSet({.buffers = 1, .textures = 3});
    state->descriptors.AddStorageImage(set, 0, depth, BufferAccess::Read);
    state->descriptors.AddStorageImage(set, 1, stencil, BufferAccess::Read);
    state->descriptors.AddStorageImage(set, 2, color, BufferAccess::Write);
    std::vector<BufferLease> accesses;
    accesses.push_back(state->Uniform(set, 0, uniform));
    state->Dispatch(state->Compute(Utility::SingleSamplePack), state->descriptors.Capture(set),
                    source_area.width, source_area.height, std::move(accesses));
    return true;
}

bool BlitHelper::DepthToBuffer(id<MTLTexture> source, BufferLease buffer,
                               const VideoCore::BufferTextureCopy& copy, u32 row_stride_words) {
    const RenderTexture src{source, copy.texture_level};
    const auto native_buffer = buffer.GetBuffer();
    const auto range = buffer.Range();
    if (!state->Valid(src, MTLTextureUsageShaderRead | MTLTextureUsagePixelFormatView) ||
        source.sampleCount != 1 || source.pixelFormat != MTLPixelFormatDepth32Float_Stencil8 ||
        native_buffer.device != state->device || copy.buffer_offset < range.offset) {
        return false;
    }
    const BufferRange relative{copy.buffer_offset - range.offset, copy.buffer_size};
    const auto full = FullArea(src);
    const auto readback = MakeDepthReadbackRegion(copy, full.width, full.height, row_stride_words);
    if (!readback || !relative.Fits(range.length)) {
        return false;
    }
    state->Refresh();
    const auto depth = CheckResource(ShaderView(src, source.pixelFormat), "Read Metal depth");
    const auto stencil =
        CheckResource(ShaderView(src, MTLPixelFormatX32_Stencil8), "Read Metal stencil");
    const auto set = state->descriptors.CreateSet({.buffers = 2, .textures = 2});
    state->descriptors.AddStorageImage(set, 0, depth, BufferAccess::Read);
    state->descriptors.AddStorageImage(set, 1, stencil, BufferAccess::Read);
    state->descriptors.AddBuffer(set, 1, native_buffer, copy.buffer_offset, copy.buffer_size,
                                 BufferAccess::Write);
    std::vector<BufferLease> accesses;
    accesses.push_back(state->Uniform(set, 0, std::as_bytes(std::span{&*readback, 1})));
    accesses.push_back(std::move(buffer));
    state->Dispatch(state->Compute(Utility::DepthReadback), state->descriptors.Capture(set),
                    readback->extent[0], readback->extent[1], std::move(accesses));
    return true;
}

bool BlitHelper::ResolveTexture(const RenderTexture& source, const RenderTexture& destination,
                                TextureAspect aspect) {
    if (!state->CanResolve(source, destination)) {
        return false;
    }
    const auto format = source.texture.pixelFormat;
    switch (aspect) {
    case TextureAspect::Color:
        if (IsDepth(format) || IsStencil(format)) {
            return false;
        }
        break;
    case TextureAspect::Depth:
        if (!IsDepth(format)) {
            return false;
        }
        break;
    case TextureAspect::DepthStencil:
        if (!IsDepth(format) || !IsStencil(format)) {
            return false;
        }
        break;
    default:
        return false;
    }
    state->Refresh();
    state->passes.BeginRendering(MakePass(source, FullArea(source), aspect, destination));
    state->passes.EndRendering();
    return true;
}

} // namespace Metal
