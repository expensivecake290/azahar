// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.txt file included.

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <map>
#include <tuple>
#include <utility>
#include <vector>
#include "common/logging/log.h"
#include "video_core/custom_textures/material.h"
#include "video_core/renderer_metal/mtl_descriptor_update_queue.h"
#include "video_core/renderer_metal/mtl_shader_util.h"
#include "video_core/renderer_metal/mtl_texture_runtime.h"

namespace Metal {

struct TextureAllocation {
    id<MTLTexture> texture;
    MemoryBudget::Reservation reservation;
};

struct SurfaceStorage {
    struct ViewKey {
        TextureRepresentation representation;
        TextureView view;
        auto operator<=>(const ViewKey&) const = default;
    };

    FormatTraits format;
    TextureResource base;
    TextureResource scaled;
    TextureResource multisample;
    TextureResource normal;
    TextureResource feedback;
    TextureResource shadow;
    TextureResource shadow_sample;
    std::optional<MemoryAllocator::Buffer> shadow_buffer;
    std::map<ViewKey, id<MTLTexture>> views;
    bool multisample_valid{};
    u32 resolution_scale{1};

    TextureResource Current() const noexcept {
        return scaled ? scaled : base;
    }
};

namespace {
using VideoCore::Extent;
using VideoCore::PixelFormat;
using VideoCore::SurfaceType;
using VideoCore::TextureType;
using Buffer = MemoryAllocator::Buffer;
using BufferLease = MemoryAllocator::GPUBufferAccess;

enum class TransferProgram : u8 {
    Color,
    Depth,
    DepthStencil,
    TextureToShadowImage,
    ShadowImageToTexture,
    TextureToShadowBuffer,
    ShadowBufferToTexture,
};

constexpr std::string_view transfer_source = R"MSL(
#include <metal_stdlib>
using namespace metal;

struct Varying {
    float4 position [[position]];
    float2 uv [[user(locn0)]];
};

struct Transform {
    float2 scale;
    float2 offset;
};

vertex Varying transfer_vertex(uint vertex_index [[vertex_id]],
                              constant Transform& st [[buffer(0)]]) {
    const float2 uv(float((vertex_index & 1u) << 1u), float(vertex_index & 2u));
    return {float4(fma(uv, float2(2, -2), float2(-1, 1)), 0, 1), fma(uv, st.scale, st.offset)};
}

fragment float4 transfer_color(Varying in [[stage_in]], texture2d<float> src [[texture(0)]],
                               sampler source_sampler [[sampler(0)]]) {
    return src.sample(source_sampler, in.uv, level(0.0f));
}

static uint2 Coord(float2 uv, uint2 size) {
    return uint2(clamp(floor(uv * float2(size)), float2(0), float2(size - 1)));
}

struct DepthOutput { float depth [[depth(any)]]; };
struct DepthStencilOutput { float depth [[depth(any)]]; uint stencil [[stencil]]; };

fragment DepthOutput transfer_depth(Varying in [[stage_in]],
                                    depth2d<float, access::read> src [[texture(0)]]) {
    return {src.read(Coord(in.uv, uint2(src.get_width(), src.get_height())))};
}

fragment DepthStencilOutput transfer_depth_stencil(
    Varying in [[stage_in]], depth2d<float, access::read> depth [[texture(0)]],
    texture2d<uint, access::read> stencil [[texture(1)]]) {
    const uint2 at = Coord(in.uv, uint2(depth.get_width(), depth.get_height()));
    return {depth.read(at), stencil.read(at).r & 255u};
}

static uint Pack(float4 source) {
    const uint4 bytes = uint4(round(saturate(source) * 255.f));
    return bytes.r | (bytes.g << 8u) | (bytes.b << 16u) | (bytes.a << 24u);
}

static float4 Unpack(uint source) {
    return float4(source & 255u, (source >> 8u) & 255u, (source >> 16u) & 255u, source >> 24u) /
           255.f;
}

kernel void texture_to_shadow_image(texture2d<float, access::read> src [[texture(0)]],
                                   texture2d<uint, access::write> dst [[texture(1)]],
                                   uint2 at [[thread_position_in_grid]]) {
    dst.write(uint4(Pack(src.read(at))), at);
}

kernel void shadow_image_to_texture(texture2d<float, access::write> dst [[texture(0)]],
                                   texture2d<uint, access::read> src [[texture(1)]],
                                   uint2 at [[thread_position_in_grid]]) {
    dst.write(Unpack(src.read(at).r), at);
}

kernel void texture_to_shadow_buffer(texture2d<float, access::read> src [[texture(0)]],
                                    device uint* dst [[buffer(0)]],
                                    uint2 at [[thread_position_in_grid]]) {
    dst[at.y * src.get_width() + at.x] = Pack(src.read(at));
}

kernel void shadow_buffer_to_texture(texture2d<float, access::write> dst [[texture(0)]],
                                    device const uint* src [[buffer(0)]],
                                    uint2 at [[thread_position_in_grid]]) {
    dst.write(Unpack(src[at.y * dst.get_width() + at.x]), at);
}
)MSL";

constexpr const char* Entry(TransferProgram program) {
    switch (program) {
    case TransferProgram::Color:
        return "transfer_color";
    case TransferProgram::Depth:
        return "transfer_depth";
    case TransferProgram::DepthStencil:
        return "transfer_depth_stencil";
    case TransferProgram::TextureToShadowImage:
        return "texture_to_shadow_image";
    case TransferProgram::ShadowImageToTexture:
        return "shadow_image_to_texture";
    case TransferProgram::TextureToShadowBuffer:
        return "texture_to_shadow_buffer";
    case TransferProgram::ShadowBufferToTexture:
        return "shadow_buffer_to_texture";
    }
    throw std::invalid_argument("Unknown Metal texture utility");
}

constexpr u16 RepackColor(u16 color, PixelFormat format) {
    switch (format) {
    case PixelFormat::RGB565:
        return static_cast<u16>(((color & 0xf800) >> 11) | (color & 0x07e0) |
                                ((color & 0x001f) << 11));
    case PixelFormat::RGB5A1:
        return static_cast<u16>(((color & 0xf800) >> 10) | (color & 0x07c1) |
                                ((color & 0x003e) << 10));
    case PixelFormat::RGBA4:
        return static_cast<u16>(((color & 0xf000) >> 12) | ((color & 0x0f00) >> 4) |
                                ((color & 0x00f0) << 4) | ((color & 0x000f) << 12));
    default:
        return color;
    }
}

static_assert(RepackColor(0xf800, PixelFormat::RGB565) == 0x001f);
static_assert(RepackColor(0xf801, PixelFormat::RGB5A1) == 0x003f);
static_assert(RepackColor(0x1234, PixelFormat::RGBA4) == 0x4321);

template <typename T>
T ReadScalar(const void* bytes) {
    T value;
    std::memcpy(&value, bytes, sizeof(value));
    return value;
}

template <typename T>
void WriteScalar(void* bytes, T value) {
    std::memcpy(bytes, &value, sizeof(value));
}

void RepackRow(std::byte* destination, const std::byte* source, std::size_t length,
               PixelFormat format) {
    for (std::size_t i = 0; i < length; i += sizeof(u16)) {
        WriteScalar(destination + i, RepackColor(ReadScalar<u16>(source + i), format));
    }
}

Extent Dimensions(id<MTLTexture> image, u32 level) {
    if (!image || level >= image.mipmapLevelCount || level >= 32) {
        return {0, 0};
    }
    return {std::max(1u, static_cast<u32>(image.width) >> level),
            std::max(1u, static_cast<u32>(image.height) >> level)};
}

u32 Layers(id<MTLTexture> image) {
    return image.textureType == MTLTextureTypeCube ? static_cast<u32>(image.arrayLength) * 6
                                                   : static_cast<u32>(image.arrayLength);
}

bool AreaFits(id<MTLTexture> image, u32 level, u32 layer, RenderArea area) {
    const auto extent = Dimensions(image, level);
    return layer < Layers(image) && area.Fits(extent.width, extent.height);
}

bool HasRenderUsage(id<MTLTexture> image) {
    return image.usage == MTLTextureUsageUnknown || (image.usage & MTLTextureUsageRenderTarget);
}

RenderArea Full(Extent extent) {
    return {0, 0, extent.width, extent.height};
}

Common::Rectangle<u32> Rect(Extent extent) {
    return {0, extent.height, extent.width, 0};
}

id<MTLTexture> View(id<MTLTexture> source, MTLPixelFormat format, u32 level, u32 slice) {
    if (source.textureType == MTLTextureType2D && source.pixelFormat == format &&
        source.mipmapLevelCount == 1 && level == 0 && slice == 0) {
        return source;
    }
    return [source newTextureViewWithPixelFormat:format
                                     textureType:MTLTextureType2D
                                          levels:NSMakeRange(level, 1)
                                          slices:NSMakeRange(slice, 1)];
}

RenderAttachment Attachment(TextureResource resource, TextureResource resolve, u32 level, u32 slice,
                            MTLLoadAction load) {
    RenderAttachment attachment{.target = {resource.Get(), level, slice}, .load_action = load};
    if (resolve) {
        attachment.resolve = {resolve.Get(), level, slice};
        attachment.store_action = MTLStoreActionStoreAndMultisampleResolve;
    }
    return attachment;
}

RenderPass Pass(TextureResource resource, TextureResource resolve, u32 level, u32 slice,
                RenderArea area, TextureAspect aspect, MTLLoadAction load) {
    const auto extent = Dimensions(resource.Get(), level);
    const auto attachment = Attachment(resource, resolve, level, slice, load);
    RenderPass pass;
    pass.area = area;
    pass.width = extent.width;
    pass.height = extent.height;
    pass.samples = static_cast<u32>(resource.Get().sampleCount);
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

struct TextureRuntime::State {
    struct Plane {
        std::size_t offset;
        TextureBufferLayout layout;
        MTLBlitOption aspect;
    };

    State(Instance& instance, Scheduler& scheduler, RenderManager& passes,
          const MemoryAllocator& allocator, MemoryBudget budget, u32 anisotropy)
        : instance(instance), scheduler(scheduler), passes(passes), allocator(allocator),
          budget(std::move(budget)), anisotropy(std::clamp(anisotropy, 1u, 16u)),
          device(instance.GetDevice()), blitter(instance, scheduler, passes, allocator),
          compiler(instance.GetContext(), 0), descriptors(instance.GetContext()) {}

    TextureResource Allocate(const FormatTraits& traits, Extent size, u32 levels, TextureType type,
                             u32 samples, std::string_view label = {}) {
        if (!size.width || !size.height || !traits.IsSupported() ||
            !instance.SupportsSampleCount(samples) || !levels ||
            (samples > 1 && (type == TextureType::CubeMap || levels != 1)) ||
            (type == TextureType::CubeMap && size.width != size.height) ||
            levels > static_cast<u32>(std::bit_width(std::max(size.width, size.height)))) {
            throw std::invalid_argument("Invalid Metal surface size, format or representation");
        }
        auto desc = [MTLTextureDescriptor new];
        desc.pixelFormat = traits.native;
        desc.textureType = type == TextureType::CubeMap ? MTLTextureTypeCube
                           : samples > 1                ? MTLTextureType2DMultisample
                                                        : MTLTextureType2D;
        desc.width = size.width;
        desc.height = size.height;
        desc.mipmapLevelCount = levels;
        desc.sampleCount = samples;
        desc.storageMode = MTLStorageModePrivate;
        desc.hazardTrackingMode = MTLHazardTrackingModeUntracked;
        desc.usage =
            samples > 1 ? traits.usage & ~(MTLTextureUsageShaderWrite | MTLTextureUsageShaderAtomic)
                        : traits.usage;
        const auto footprint = [device heapTextureSizeAndAlignWithDescriptor:desc];
        auto reservation = budget.TryReserve(MemoryPool::Texture, footprint.size, footprint.size);
        if (!reservation) {
            passes.EndRendering();
            const auto completed = scheduler.Finish();
            if (completed == CompletionStatus::Complete) {
                reservation =
                    budget.TryReserve(MemoryPool::Texture, footprint.size, footprint.size);
            }
            if (!reservation) {
                throw std::runtime_error("Metal texture allocation exceeds its physical budget");
            }
        }
        auto texture =
            CheckResource([device newTextureWithDescriptor:desc], "Allocate private Metal texture");
        if (!label.empty()) {
            texture.label = [[NSString alloc] initWithBytes:label.data()
                                                     length:label.size()
                                                   encoding:NSUTF8StringEncoding];
        }
        if (!reservation->AdjustPhysical(texture.allocatedSize)) {
            throw std::runtime_error("Metal texture allocation exceeded its budget reservation");
        }
        return TextureResource(std::make_shared<TextureAllocation>(
            TextureAllocation{texture, std::move(*reservation)}));
    }

    static FormatTraits CurrentFormat(const SurfaceStorage& surface) {
        if (surface.scaled && surface.scaled.Get().pixelFormat != surface.format.native) {
            return {.native = surface.scaled.Get().pixelFormat,
                    .aspect = TextureAspect::Color,
                    .usage = surface.scaled.Get().usage,
                    .guest_format = surface.format.guest_format,
                    .bytes_per_block = 4,
                    .filterable = true};
        }
        return surface.format;
    }

    void Keep(TextureResource resource) {
        scheduler.Record([resource](EncodingContext& context) { resource.Retain(context); });
    }

    void BufferCopy(TextureResource resource, BufferLease access, std::span<const Plane> planes,
                    u32 level, u32 layer, RenderArea area, bool upload) {
        passes.EndRendering();
        scheduler.Record([resource, access = std::move(access), level, layer, area, upload,
                          planes = std::vector<Plane>{planes.begin(), planes.end()}](
                             EncodingContext& context) mutable {
            const auto buffer = access.GetBuffer();
            const auto texture = resource.Get();
            const std::array uses{
                ResourceUse{buffer, upload ? BufferAccess::Read : BufferAccess::Write, MTLStageBlit,
                            access.Range()},
                ResourceUse{texture, upload ? BufferAccess::Write : BufferAccess::Read,
                            MTLStageBlit},
            };
            auto encoder = CheckResource([context.GetCommandBuffer() computeCommandEncoder],
                                         "Begin native Metal texture transfer");
            context.Synchronize(encoder, uses);
            const auto origin = MTLOriginMake(area.x, area.y, 0);
            const auto extent = MTLSizeMake(area.width, area.height, 1);
            for (const auto& plane : planes) {
                if (upload) {
                    [encoder copyFromBuffer:buffer
                               sourceOffset:plane.offset
                          sourceBytesPerRow:plane.layout.row_pitch
                        sourceBytesPerImage:0
                                 sourceSize:extent
                                  toTexture:texture
                           destinationSlice:layer
                           destinationLevel:level
                          destinationOrigin:origin
                                    options:plane.aspect];
                } else {
                    [encoder copyFromTexture:texture
                                     sourceSlice:layer
                                     sourceLevel:level
                                    sourceOrigin:origin
                                      sourceSize:extent
                                        toBuffer:buffer
                               destinationOffset:plane.offset
                          destinationBytesPerRow:plane.layout.row_pitch
                        destinationBytesPerImage:0
                                         options:plane.aspect];
                }
            }
            context.Retain(std::move(access));
            resource.Retain(context);
            context.EndPass(encoder);
        });
    }

    void Copy(TextureResource source, TextureResource destination,
              std::span<const VideoCore::TextureCopy> copies) {
        passes.EndRendering();
        scheduler.Record([source, destination,
                          copies = std::vector<VideoCore::TextureCopy>{
                              copies.begin(), copies.end()}](EncodingContext& context) {
            auto encoder = CheckResource([context.GetCommandBuffer() computeCommandEncoder],
                                         "Begin native Metal texture transfer");
            const std::array uses{
                ResourceUse{source.Get(), BufferAccess::Read, MTLStageBlit},
                ResourceUse{destination.Get(), BufferAccess::Write, MTLStageBlit}};
            context.Synchronize(encoder, uses);
            for (const auto& copy : copies) {
                [encoder copyFromTexture:source.Get()
                             sourceSlice:copy.src_layer
                             sourceLevel:copy.src_level
                            sourceOrigin:MTLOriginMake(copy.src_offset.x, copy.src_offset.y, 0)
                              sourceSize:MTLSizeMake(copy.extent.width, copy.extent.height, 1)
                               toTexture:destination.Get()
                        destinationSlice:copy.dst_layer
                        destinationLevel:copy.dst_level
                       destinationOrigin:MTLOriginMake(copy.dst_offset.x, copy.dst_offset.y, 0)];
            }
            source.Retain(context);
            destination.Retain(context);
            context.EndPass(encoder);
        });
    }

    std::vector<VideoCore::TextureCopy> Subresources(TextureResource texture) const {
        std::vector<VideoCore::TextureCopy> copies;
        const auto image = texture.Get();
        for (u32 level = 0; level < image.mipmapLevelCount; ++level) {
            for (u32 slice = 0; slice < Layers(image); ++slice) {
                copies.push_back({level, level, slice, slice, {}, {}, Dimensions(image, level)});
            }
        }
        return copies;
    }

    TextureResource Snapshot(TextureResource source, FormatTraits traits) {
        const auto texture = source.Get();
        traits.native = texture.pixelFormat;
        traits.usage = texture.usage;
        const auto type = texture.textureType == MTLTextureTypeCube ? TextureType::CubeMap
                                                                    : TextureType::Texture2D;
        auto temporary =
            Allocate(traits, Dimensions(texture, 0), static_cast<u32>(texture.mipmapLevelCount),
                     type, static_cast<u32>(texture.sampleCount));
        Copy(source, temporary, Subresources(source));
        return temporary;
    }

    MTL4LibraryFunctionDescriptor* Function(const char* entry) {
        auto& context = instance.GetContext();
        if (!library.GetLibrary(context)) {
            render.clear();
            compute.clear();
            library = compiler.CompileLibrary(
                {std::string(transfer_source), "Metaliris texture transfers", {}, {}});
        }
        const auto native = library.GetLibrary(context);
        if (!native) {
            throw std::runtime_error("Metal texture MSL compilation failed: " +
                                     library.GetDiagnostic().message);
        }
        auto descriptor = [MTL4LibraryFunctionDescriptor new];
        descriptor.library = native;
        descriptor.name = @(entry);
        return descriptor;
    }

    id<MTLRenderPipelineState> Render(TransferProgram program, MTLPixelFormat color, u32 samples) {
        // Resolve function descriptors first so reinitialization cannot invalidate a map reference.
        auto fragment = Function(Entry(program));
        auto key = std::tuple{program, color, samples};
        if (auto found = render.find(key); found != render.end()) {
            return found->second;
        }
        auto descriptor = [MTL4RenderPipelineDescriptor new];
        descriptor.vertexFunctionDescriptor = Function("transfer_vertex");
        descriptor.fragmentFunctionDescriptor = fragment;
        descriptor.colorAttachments[0].pixelFormat = color;
        descriptor.rasterSampleCount = samples;
        descriptor.inputPrimitiveTopology = MTLPrimitiveTopologyClassTriangle;
        auto& context = instance.GetContext();
        auto lease = context.AcquireWork();
        auto native =
            CheckResource(compiler.GetNativeCompiler(lease), "Access Metal 4 texture compiler");
        auto guard = context.AcquireSubmission(lease);
        if (!guard) {
            throw std::logic_error("Stale Metal texture render compiler");
        }
        NSError* error = nil;
        auto result = [native newRenderPipelineStateWithDescriptor:descriptor
                                               compilerTaskOptions:nil
                                                             error:&error];
        result = CheckResource(result, "Compile Metal 4 texture transfer pipeline", error);
        render.emplace(key, result);
        return result;
    }

    id<MTLComputePipelineState> Compute(TransferProgram program) {
        auto function = Function(Entry(program));
        if (auto found = compute.find(program); found != compute.end()) {
            return found->second;
        }
        auto descriptor = [MTL4ComputePipelineDescriptor new];
        descriptor.computeFunctionDescriptor = function;
        auto& context = instance.GetContext();
        auto lease = context.AcquireWork();
        auto native =
            CheckResource(compiler.GetNativeCompiler(lease), "Access Metal 4 texture compiler");
        auto guard = context.AcquireSubmission(lease);
        if (!guard) {
            throw std::logic_error("Stale Metal texture compute compiler");
        }
        NSError* error = nil;
        auto result = [native newComputePipelineStateWithDescriptor:descriptor
                                                compilerTaskOptions:nil
                                                              error:&error];
        result = CheckResource(result, "Compile Metal 4 shadow transfer pipeline", error);
        compute.emplace(program, result);
        return result;
    }

    id<MTLDepthStencilState> DepthState(bool stencil) {
        auto& value = depth_states[stencil];
        if (!value) {
            auto descriptor = [MTLDepthStencilDescriptor new];
            descriptor.depthWriteEnabled = YES;
            descriptor.depthCompareFunction = MTLCompareFunctionAlways;
            if (stencil) {
                auto stencil_state = [MTLStencilDescriptor new];
                stencil_state.stencilCompareFunction = MTLCompareFunctionAlways;
                stencil_state.depthStencilPassOperation = MTLStencilOperationReplace;
                stencil_state.writeMask = 0xff;
                descriptor.frontFaceStencil = descriptor.backFaceStencil = stencil_state;
            }
            value = CheckResource([device newDepthStencilStateWithDescriptor:descriptor],
                                  "Create Metal texture-transfer depth state");
        }
        return value;
    }

    id<MTLSamplerState> Filter(bool linear) {
        auto& value = transfer_samplers[linear];
        if (!value) {
            auto descriptor = [MTLSamplerDescriptor new];
            descriptor.minFilter = descriptor.magFilter =
                linear ? MTLSamplerMinMagFilterLinear : MTLSamplerMinMagFilterNearest;
            descriptor.sAddressMode = descriptor.tAddressMode = MTLSamplerAddressModeClampToEdge;
            descriptor.supportArgumentBuffers = YES;
            value = CheckResource([device newSamplerStateWithDescriptor:descriptor],
                                  "Create native Metal texture-transfer sampler");
        }
        return value;
    }

    bool Blit(TextureResource source, TextureResource dest, FormatTraits traits,
              const VideoCore::TextureBlit& blit, bool linear,
              MTLLoadAction load = MTLLoadActionLoad) {
        if (!source || !dest || source.Get().sampleCount != 1 || !HasRenderUsage(dest.Get())) {
            return false;
        }
        const auto source_size = Dimensions(source.Get(), blit.src_level);
        const auto area = BlitBounds(blit.dst_rect);
        const auto sampling = MakeBlitSampling(blit, source_size.width, source_size.height);
        if (!sampling || blit.src_layer >= Layers(source.Get()) ||
            !AreaFits(dest.Get(), blit.dst_level, blit.dst_layer, area)) {
            return false;
        }
        const auto aspect = traits.aspect;
        if (source.Get() == dest.Get()) {
            source = Snapshot(source, traits);
        }
        auto source_view = CheckResource(
            View(source.Get(), source.Get().pixelFormat, blit.src_level, blit.src_layer),
            "View Metal transfer source texture");
        const bool is_stencil = aspect == TextureAspect::DepthStencil;
        const bool is_depth = is_stencil || aspect == TextureAspect::Depth;
        const auto program = is_stencil ? TransferProgram::DepthStencil
                             : is_depth ? TransferProgram::Depth
                                        : TransferProgram::Color;
        const auto pass = Pass(dest, {}, blit.dst_level, blit.dst_layer, area, aspect, load);
        const auto pipeline = Render(program, pass.Formats().color, pass.samples);
        const auto depth = is_depth ? DepthState(is_stencil) : nil;
        const auto storage = allocator.AllocateBuffer(sizeof(BlitSampling), MemoryPool::Transient);
        const BufferRange range{0, sizeof(BlitSampling)};
        {
            auto mapped = storage.MapWrite(range);
            std::memcpy(mapped.Bytes().data(), &*sampling, sizeof(BlitSampling));
        }
        auto access = storage.AcquireGPUAccess(range, BufferAccess::Read);
        const auto vs = descriptors.CreateSet({.buffers = 1});
        descriptors.AddBuffer(vs, 0, access.GetBuffer(), 0, sizeof(BlitSampling));
        const auto fs = descriptors.CreateSet(
            {.textures = is_stencil ? 2u : 1u, .samplers = is_depth ? 0u : 1u});
        if (is_depth) {
            descriptors.AddStorageImage(fs, 0, source_view, BufferAccess::Read);
            if (is_stencil) {
                auto stencil = CheckResource(
                    View(source.Get(), MTLPixelFormatX32_Stencil8, blit.src_level, blit.src_layer),
                    "View Metal transfer stencil source");
                descriptors.AddStorageImage(fs, 1, stencil, BufferAccess::Read);
            }
        } else {
            descriptors.AddImageSampler(fs, 0, 0, source_view, Filter(linear));
        }

        passes.BeginRendering(pass);
        passes.RecordDraw(
            [vs = descriptors.Capture(vs), fs = descriptors.Capture(fs), source, dest, area,
             pipeline, depth, access = std::move(access)](
                EncodingContext& context, id<MTL4RenderCommandEncoder> encoder) mutable {
                [encoder setRenderPipelineState:pipeline];
                [encoder setDepthStencilState:depth];
                [encoder setCullMode:MTLCullModeNone];
                [encoder setViewport:MTLViewport{double(area.x), double(area.y), double(area.width),
                                                 double(area.height), 0, 1}];
                [encoder setScissorRect:MTLScissorRect{area.x, area.y, area.width, area.height}];
                vs.BindRender(context, encoder, MTLRenderStageVertex);
                fs.BindRender(context, encoder, MTLRenderStageFragment);
                [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
                context.Retain(pipeline);
                context.Retain(depth);
                context.Retain(std::move(access));
                source.Retain(context);
                dest.Retain(context);
            });
        passes.EndRendering();
        return true;
    }

    void ShadowTransfer(TextureResource color, const ShadowTextureBinding& binding, bool upload) {
        const auto program = binding.buffer ? (upload ? TransferProgram::TextureToShadowBuffer
                                                      : TransferProgram::ShadowBufferToTexture)
                                            : (upload ? TransferProgram::TextureToShadowImage
                                                      : TransferProgram::ShadowImageToTexture);
        const auto set = descriptors.CreateSet(
            {.buffers = binding.buffer ? 1u : 0u, .textures = binding.buffer ? 1u : 2u});
        const auto access = [&](bool read) {
            return read ? BufferAccess::Read : BufferAccess::Write;
        };
        descriptors.AddStorageImage(set, 0, View(color.Get(), color.Get().pixelFormat, 0, 0),
                                    access(upload));
        std::optional<BufferLease> use;
        if (binding.buffer) {
            const BufferRange range{0, binding.buffer->Size()};
            use.emplace(binding.buffer->AcquireGPUAccess(range, access(!upload)));
            descriptors.AddBuffer(set, 0, use->GetBuffer(), 0, range.length, access(!upload));
        } else {
            descriptors.AddStorageImage(set, 1, binding.image.Get(), access(!upload));
        }

        const auto pipeline = Compute(program);
        const auto size = Dimensions(color.Get(), 0);
        passes.EndRendering();
        scheduler.Record([snapshot = descriptors.Capture(set), color, binding, pipeline, size,
                          use = std::move(use)](EncodingContext& context) mutable {
            auto encoder = CheckResource([context.GetCommandBuffer() computeCommandEncoder],
                                         "Begin native Metal texture transfer");
            snapshot.BindCompute(context, encoder);
            [encoder setComputePipelineState:pipeline];
            const auto threads = MTLSizeMake(pipeline.threadExecutionWidth, 1, 1);
            [encoder dispatchThreads:MTLSizeMake(size.width, size.height, 1)
                threadsPerThreadgroup:threads];
            if (use) {
                context.Retain(std::move(*use));
            }
            color.Retain(context);
            binding.image.Retain(context);
            context.Retain(pipeline);
            context.EndPass(encoder);
        });
    }

    Instance& instance;
    Scheduler& scheduler;
    RenderManager& passes;
    const MemoryAllocator& allocator;
    MemoryBudget budget;
    const u32 anisotropy;
    const id<MTLDevice> device;
    BlitHelper blitter;
    ShaderCompiler compiler;
    DescriptorUpdateQueue descriptors;
    ShaderLibrary library;
    std::map<std::tuple<TransferProgram, MTLPixelFormat, u32>, id<MTLRenderPipelineState>> render;
    std::map<TransferProgram, id<MTLComputePipelineState>> compute;
    std::array<id<MTLDepthStencilState>, 2> depth_states{};
    std::array<id<MTLSamplerState>, 2> transfer_samplers{};
    std::vector<u8> upload_staging;
    std::vector<u8> download_staging;
};

id<MTLTexture> TextureResource::Get() const noexcept {
    return allocation ? allocation->texture : nil;
}

void TextureResource::Retain(EncodingContext& context) const {
    if (allocation) {
        context.Retain(allocation);
    }
}

TextureRuntime::TextureRuntime(Instance& instance, Scheduler& scheduler, RenderManager& passes,
                               const MemoryAllocator& allocator, MemoryBudget budget,
                               u32 max_anisotropy)
    : state(std::make_unique<State>(instance, scheduler, passes, allocator, std::move(budget),
                                    max_anisotropy)) {}

TextureRuntime::~TextureRuntime() = default;

const Instance& TextureRuntime::GetInstance() const noexcept {
    return state->instance;
}

Scheduler& TextureRuntime::GetScheduler() const noexcept {
    return state->scheduler;
}

RenderManager& TextureRuntime::GetRenderManager() const noexcept {
    return state->passes;
}

u32 TextureRuntime::MaxAnisotropy() const noexcept {
    return state->anisotropy;
}

bool TextureRuntime::SupportsCustomFormat(VideoCore::CustomPixelFormat format) const {
    return GetInstance().GetFormats().GetTraits(format).IsSupported();
}

u64 TextureRuntime::GetResourceTick() {
    return GetScheduler().CurrentTick();
}

u64 TextureRuntime::GetResourceFreeTick() {
    return GetScheduler().GetMasterSemaphore().Timeline().RetiredTick();
}

void TextureRuntime::Finish() {
    state->passes.EndRendering();
    if (state->scheduler.Finish() != CompletionStatus::Complete) {
        throw std::runtime_error("Metal texture work did not complete");
    }
}

VideoCore::StagingData TextureRuntime::FindStaging(u32 size, bool upload) {
    auto& staging = upload ? state->upload_staging : state->download_staging;
    staging.resize(size);
    return {size, 0, staging};
}

bool TextureRuntime::NeedsConversion(const TextureSurface& surface) const {
    using enum TextureConversion;
    switch (surface.Format().conversion) {
    case GuestColorToRGBA8:
    case Depth24ToFloat:
        return true;
    default:
        return false;
    }
}

bool TextureRuntime::ClearTexture(TextureSurface& surface, const VideoCore::TextureClear& clear) {
    const auto resource = surface.Resource();
    const auto area = BlitBounds(clear.texture_rect);
    if (!resource || !HasRenderUsage(resource.Get()) ||
        !AreaFits(resource.Get(), clear.texture_level, clear.texture_layer, area)) {
        return false;
    }
    auto pass = Pass(resource, {}, clear.texture_level, clear.texture_layer, area, surface.Aspect(),
                     MTLLoadActionClear);
    if (surface.Aspect() == TextureAspect::Color) {
        const auto& value = clear.value.color;
        pass.clear_color = MTLClearColorMake(value.x, value.y, value.z, value.w);
    } else {
        pass.clear_depth = clear.value.depth;
        pass.clear_stencil = clear.value.stencil;
    }
    state->Keep(resource);
    state->passes.BeginRendering(pass);
    state->passes.EndRendering();
    surface.storage->multisample_valid &= clear.texture_level != 0;
    return true;
}

bool TextureRuntime::CopyTextures(TextureSurface& source, TextureSurface& destination,
                                  std::span<const VideoCore::TextureCopy> copies) {
    if (copies.empty()) {
        return true;
    }
    auto from = source.Resource();
    auto to = destination.Resource();
    if (!from || !to || from.Get().pixelFormat != to.Get().pixelFormat) {
        return false;
    }
    for (const auto& copy : copies) {
        const RenderArea src{copy.src_offset.x, copy.src_offset.y, copy.extent.width,
                             copy.extent.height};
        const RenderArea dst{copy.dst_offset.x, copy.dst_offset.y, copy.extent.width,
                             copy.extent.height};
        if (!AreaFits(from.Get(), copy.src_level, copy.src_layer, src) ||
            !AreaFits(to.Get(), copy.dst_level, copy.dst_layer, dst)) {
            return false;
        }
    }
    const bool touches_primary =
        std::ranges::any_of(copies, [](const auto& copy) { return copy.dst_level == 0; });
    const bool multisample = source.storage->multisample_valid &&
                             destination.storage->multisample_valid && source.sample_count > 1 &&
                             source.sample_count == destination.sample_count &&
                             std::ranges::all_of(copies, [](const auto& copy) {
                                 return copy.src_level == 0 && copy.dst_level == 0;
                             });
    if (multisample) {
        auto from_samples = source.Resource(TextureRepresentation::Multisample);
        auto to_samples = destination.Resource(TextureRepresentation::Multisample);
        if (from_samples.Get() == to_samples.Get()) {
            from_samples = state->Snapshot(from_samples, State::CurrentFormat(*source.storage));
        }
        state->Copy(from_samples, to_samples, copies);
    } else if (touches_primary) {
        destination.storage->multisample_valid = false;
    }
    if (from.Get() == to.Get()) {
        from = state->Snapshot(from, State::CurrentFormat(*source.storage));
    }
    state->Copy(from, to, copies);
    return true;
}

bool TextureRuntime::Reinterpret(TextureSurface& source, TextureSurface& destination,
                                 const VideoCore::TextureCopy& copy) {
    if (source.pixel_format == destination.pixel_format) {
        return false;
    }
    if (source.Image() && destination.Image() &&
        source.Image().pixelFormat == destination.Image().pixelFormat && !NeedsConversion(source) &&
        !NeedsConversion(destination) && source.Format().conversion == TextureConversion::None &&
        destination.Format().conversion == TextureConversion::None) {
        return CopyTextures(source, destination, copy);
    }
    if (source.pixel_format != PixelFormat::D24S8 ||
        destination.pixel_format != PixelFormat::RGBA8) {
        // The shared cache flushes the source guest bytes and reloads the requested format.
        return false;
    }
    auto from = source.Resource();
    auto to = destination.Resource();
    if (!from || !to) {
        return false;
    }
    if (source.storage->multisample_valid && destination.storage->multisample_valid &&
        source.sample_count == destination.sample_count && copy.src_level == 0 &&
        copy.dst_level == 0) {
        const auto src_samples = source.Resource(TextureRepresentation::Multisample);
        const auto dst_samples = destination.Resource(TextureRepresentation::Multisample);
        state->Keep(src_samples);
        state->Keep(dst_samples);
        state->Keep(to);
        return state->blitter.ConvertDS24S8ToRGBA8(src_samples.Get(), dst_samples.Get(), copy,
                                                   {to.Get(), copy.dst_level, copy.dst_layer});
    }
    state->Keep(from);
    state->Keep(to);
    if (!state->blitter.ConvertDS24S8ToRGBA8(from.Get(), to.Get(), copy)) {
        return false;
    }
    if (copy.dst_level == 0) {
        destination.storage->multisample_valid = false;
    }
    return true;
}

bool TextureRuntime::BlitTextures(TextureSurface& source, TextureSurface& destination,
                                  const VideoCore::TextureBlit& blit) {
    if (source.Aspect() != destination.Aspect() || source.Aspect() == TextureAspect::None) {
        return false;
    }
    const auto from = source.Resource();
    const auto to = destination.Resource();
    const bool linear = source.Aspect() == TextureAspect::Color && source.Format().filterable;
    if (!state->Blit(from, to, State::CurrentFormat(*destination.storage), blit, linear)) {
        return false;
    }
    if (blit.dst_level == 0) {
        destination.storage->multisample_valid = false;
    }
    return true;
}

void TextureRuntime::GenerateMipmaps(TextureSurface& surface) {
    const auto resource = surface.Resource();
    if (!resource || resource.Get().mipmapLevelCount < 2) {
        return;
    }
    if (surface.Aspect() == TextureAspect::Color) {
        state->passes.EndRendering();
        state->scheduler.Record([resource](EncodingContext& context) {
            auto encoder = CheckResource([context.GetCommandBuffer() computeCommandEncoder],
                                         "Begin native Metal texture transfer");
            const std::array uses{
                ResourceUse{resource.Get(), BufferAccess::ReadWrite, MTLStageBlit}};
            context.Synchronize(encoder, uses);
            [encoder generateMipmapsForTexture:resource.Get()];
            resource.Retain(context);
            context.EndPass(encoder);
        });
        return;
    }
    const auto image = resource.Get();
    // Keep one original level as the source of each operation. Avoid a read/write alias between
    // subresources on the same attachment by copying a level into a distinct texture.
    const auto traits = State::CurrentFormat(*surface.storage);
    for (u32 level = 1; level < image.mipmapLevelCount; ++level) {
        const auto src_size = Dimensions(image, level - 1);
        auto temporary = state->Allocate(traits, src_size, 1, TextureType::Texture2D, 1);
        for (u32 layer = 0; layer < Layers(image); ++layer) {
            const auto dst_size = Dimensions(image, level);
            const std::array copies{
                VideoCore::TextureCopy{level - 1, 0, layer, 0, {}, {}, src_size}};
            state->Copy(resource, temporary, copies);
            const VideoCore::TextureBlit blit{0, level, 0, layer, Rect(src_size), Rect(dst_size)};
            if (!state->Blit(temporary, resource, traits, blit, false)) {
                throw std::logic_error("Metal depth mip generation has incompatible subresources");
            }
        }
    }
}

TextureSurface::TextureSurface(TextureRuntime& runtime_, const VideoCore::SurfaceParams& params,
                               const VideoCore::SurfaceFlagBits& initial_flags)
    : SurfaceBase(params, initial_flags), runtime(&runtime_),
      storage(std::make_shared<SurfaceStorage>()) {
    auto& state = *runtime->state;
    auto traits = state.instance.GetFormats().GetTraits(pixel_format);
    if (!traits.IsSupported() || type == SurfaceType::Fill) {
        return;
    }
    if (True(flags & VideoCore::SurfaceFlagBits::ShadowSource) &&
        traits.aspect == TextureAspect::Color) {
        const auto guest = traits.guest_format;
        traits = state.instance.GetFormats().GetTraits(PixelFormat::RGBA8);
        traits.guest_format = guest;
        if (type == SurfaceType::Texture) {
            traits.conversion = TextureConversion::DecodeGuestTexture;
        }
    }
    traits.usage |= MTLTextureUsageRenderTarget;
    storage->format = traits;
    storage->resolution_scale = res_scale;
    storage->base =
        state.Allocate(traits, {width, height}, levels, texture_type, 1, DebugName(false));
    if (!res_scale || width > std::numeric_limits<u32>::max() / res_scale ||
        height > std::numeric_limits<u32>::max() / res_scale) {
        throw std::invalid_argument("Metal surface scale exceeds its dimensions");
    }
    const Extent scaled{GetScaledWidth(), GetScaledHeight()};
    if (res_scale > 1) {
        storage->scaled = state.Allocate(traits, scaled, levels, texture_type, 1, DebugName(true));
    }
    if (sample_count > 1 && texture_type == TextureType::Texture2D) {
        storage->multisample = state.Allocate(traits, scaled, 1, texture_type, sample_count,
                                              DebugName(res_scale > 1, false, sample_count));
    } else {
        sample_count = 1;
    }
}

TextureSurface::TextureSurface(TextureRuntime& runtime_, const VideoCore::SurfaceBase& original,
                               const VideoCore::Material* replacement)
    : SurfaceBase(original), runtime(&runtime_), storage(std::make_shared<SurfaceStorage>()) {
    if (!replacement) {
        return;
    }
    auto& state = *runtime->state;
    auto traits = state.instance.GetFormats().GetTraits(replacement->format);
    if (!traits.IsSupported()) {
        return;
    }
    if (traits.block_width == 1 && traits.block_height == 1) {
        traits.usage |= MTLTextureUsageRenderTarget;
    }
    storage->format = traits;
    storage->resolution_scale = res_scale;
    material = replacement;
    const Extent dimensions{replacement->width, replacement->height};
    const auto mip_count = std::min(
        levels, static_cast<u32>(std::bit_width(std::max(dimensions.width, dimensions.height))));
    storage->base =
        state.Allocate(traits, dimensions, mip_count, texture_type, 1, DebugName(false, true));
    if (res_scale > 1 && (traits.block_width > 1 || traits.block_height > 1)) {
        const auto rgba = state.instance.GetFormats().GetTraits(PixelFormat::RGBA8);
        storage->scaled =
            state.Allocate(rgba, dimensions, mip_count, texture_type, 1, DebugName(true, true));
    }
    if (const auto* normal = replacement->Map(VideoCore::MapType::Normal)) {
        const auto normal_traits = state.instance.GetFormats().GetTraits(normal->format);
        if (normal_traits.IsSupported()) {
            storage->normal = state.Allocate(normal_traits, dimensions, mip_count, texture_type, 1,
                                             DebugName(false, true));
        }
    }
    sample_count = 1;
}

TextureSurface::~TextureSurface() = default;
TextureSurface::TextureSurface(TextureSurface&&) noexcept = default;
TextureSurface& TextureSurface::operator=(TextureSurface&&) noexcept = default;

TextureAspect TextureSurface::Aspect() const noexcept {
    return storage->format.aspect;
}

const FormatTraits& TextureSurface::Format() const noexcept {
    return storage->format;
}

TextureResource TextureSurface::Resource(TextureRepresentation representation) const noexcept {
    using enum TextureRepresentation;
    switch (representation) {
    case Current:
        return storage->Current();
    case Base:
        return storage->base;
    case Scaled:
        return storage->scaled;
    case Multisample:
        return storage->multisample;
    case Normal:
        return storage->normal;
    case Feedback:
        return storage->feedback;
    }
    return {};
}

id<MTLTexture> TextureSurface::Image(TextureRepresentation representation) const noexcept {
    return Resource(representation).Get();
}

id<MTLTexture> TextureSurface::ImageView(TextureView view, TextureRepresentation representation) {
    if (view == TextureView::Storage) {
        return StorageView();
    }
    auto source = Image(representation);
    if (!source) {
        return nil;
    }
    if (view == TextureView::Sample || view == TextureView::Depth ||
        view == TextureView::Attachment) {
        return source;
    }
    if (view != TextureView::Stencil || Aspect() != TextureAspect::DepthStencil) {
        return nil;
    }
    auto& saved = storage->views[{representation, view}];
    if (!saved) {
        saved = CheckResource([source newTextureViewWithPixelFormat:MTLPixelFormatX32_Stencil8],
                              "Create native stencil aspect view");
    }
    return saved;
}

id<MTLTexture> TextureSurface::CopyImageView(TextureRepresentation representation) {
    const auto source = Resource(representation);
    if (!source) {
        return nil;
    }
    storage->feedback =
        runtime->state->Snapshot(source, TextureRuntime::State::CurrentFormat(*storage));
    return storage->feedback.Get();
}

id<MTLTexture> TextureSurface::StorageView() {
    return ShadowImage().Get();
}

TextureResource TextureSurface::ShadowImage() {
    auto& state = *runtime->state;
    const auto source = Resource();
    if (!source || source.Get().pixelFormat != MTLPixelFormatRGBA8Unorm ||
        source.Get().textureType != MTLTextureType2D) {
        throw std::invalid_argument("Metal shadow sampling needs a 2D RGBA8 guest surface");
    }
    const auto dimensions = Dimensions(source.Get(), 0);
    if (!storage->shadow_sample) {
        const FormatTraits traits{.native = MTLPixelFormatR32Uint,
                                  .aspect = TextureAspect::Color,
                                  .usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite,
                                  .bytes_per_block = 4};
        storage->shadow_sample = state.Allocate(traits, dimensions, 1, TextureType::Texture2D, 1);
    }
    state.ShadowTransfer(source, {.image = storage->shadow_sample, .row_pitch = dimensions.width},
                         true);
    return storage->shadow_sample;
}

ShadowTextureBinding TextureSurface::PrepareShadowStorage(bool texture_atomics) {
    auto& state = *runtime->state;
    const auto source = Resource();
    if (!source || source.Get().pixelFormat != MTLPixelFormatRGBA8Unorm ||
        source.Get().textureType != MTLTextureType2D) {
        throw std::invalid_argument("Metal shadow storage needs a 2D RGBA8 guest surface");
    }
    const auto dimensions = Dimensions(source.Get(), 0);
    ShadowTextureBinding binding{.row_pitch = dimensions.width};
    if (texture_atomics) {
        if (!storage->shadow) {
            const FormatTraits traits{.native = MTLPixelFormatR32Uint,
                                      .aspect = TextureAspect::Color,
                                      .usage = MTLTextureUsageShaderRead |
                                               MTLTextureUsageShaderWrite |
                                               MTLTextureUsageShaderAtomic,
                                      .bytes_per_block = 4};
            storage->shadow = state.Allocate(traits, dimensions, 1, TextureType::Texture2D, 1);
        }
        binding.image = storage->shadow;
    } else {
        if (!storage->shadow_buffer) {
            const auto layout = MakeTextureBufferLayout(dimensions, 1, 1, 4, 1);
            if (!layout) {
                throw std::overflow_error("Metal shadow word storage exceeds buffer limits");
            }
            storage->shadow_buffer = state.allocator.AllocateBuffer(
                layout->length, MemoryPool::Texture, MTLStorageModePrivate);
        }
        binding.buffer = storage->shadow_buffer;
    }
    state.ShadowTransfer(source, binding, true);
    return binding;
}

void TextureSurface::CommitShadowStorage(const ShadowTextureBinding& binding) {
    if (binding.buffer.has_value() == static_cast<bool>(binding.image) ||
        binding.row_pitch != Image().width ||
        (binding.image && binding.image.Get() != storage->shadow.Get()) ||
        (binding.buffer && !storage->shadow_buffer)) {
        throw std::invalid_argument("Metal shadow commit does not belong to this surface");
    }
    if (binding.buffer) {
        const auto& owner = *storage->shadow_buffer;
        const auto& candidate = *binding.buffer;
        const auto left = owner.AcquireGPUAccess({0, owner.Size()}, BufferAccess::Read);
        const auto right = candidate.AcquireGPUAccess({0, candidate.Size()}, BufferAccess::Read);
        if (left.GetBuffer() != right.GetBuffer()) {
            throw std::invalid_argument("Metal shadow commit has a different storage buffer");
        }
    }
    runtime->state->ShadowTransfer(Resource(), binding, false);
    storage->multisample_valid = false;
}

u32 TextureSurface::GetInternalBytesPerPixel() const {
    return Aspect() == TextureAspect::DepthStencil ? 4 : Format().bytes_per_block;
}

void TextureSurface::Upload(const VideoCore::BufferTextureCopy& copy,
                            const VideoCore::StagingData& staging) {
    auto& state = *runtime->state;
    const auto target = Resource(TextureRepresentation::Base);
    const auto area = BlitBounds(copy.texture_rect);
    const auto extent = Extent{area.width, area.height};
    const auto& traits = Format();
    const auto decoded = MakeTextureBufferLayout(extent, 1, 1, GetInternalBytesPerPixel(), 1);
    if (!target || !AreaFits(target.Get(), copy.texture_level, 0, area) || !decoded ||
        copy.buffer_offset != staging.offset || copy.buffer_size > staging.size ||
        copy.buffer_size > staging.mapped.size() || decoded->length > copy.buffer_size) {
        throw std::invalid_argument("Metal texture upload rectangle or shared staging is invalid");
    }
    const bool combined = Aspect() == TextureAspect::DepthStencil;
    const auto layout =
        MakeTextureBufferLayout(extent, 1, 1, GetInternalBytesPerPixel(), 256).value();
    std::vector<TextureRuntime::State::Plane> planes{
        {0, layout, combined ? MTLBlitOptionDepthFromDepthStencil : MTLBlitOptionNone}};
    auto capacity = layout.length;
    if (combined) {
        const auto stencil = MakeTextureBufferLayout(extent, 1, 1, 1, 256).value();
        if (stencil.length > std::numeric_limits<std::size_t>::max() - capacity) {
            throw std::overflow_error("Metal depth/stencil upload staging overflow");
        }
        planes.push_back({capacity, stencil, MTLBlitOptionStencilFromDepthStencil});
        capacity += stencil.length;
    }

    const auto upload = state.allocator.AllocateBuffer(capacity, MemoryPool::Stream);
    const BufferRange range{0, capacity};
    {
        const auto map = upload.MapWrite(range);
        const auto* input = reinterpret_cast<const std::byte*>(staging.mapped.data());
        auto* output = map.Bytes().data();
        for (std::size_t y = 0; y < decoded->rows; ++y) {
            const auto* src = input + y * decoded->row_pitch;
            auto* dst = output + y * layout.row_pitch;
            if (combined) {
                auto* stencil = output + planes[1].offset + y * planes[1].layout.row_pitch;
                for (u32 x = 0; x < area.width; ++x) {
                    const u32 packed = ReadScalar<u32>(src + 4 * x);
                    const auto value = static_cast<float>(packed >> 8) / 16777215.f;
                    WriteScalar(dst + 4 * x, value);
                    stencil[x] = static_cast<std::byte>(packed & 255);
                }
            } else if (traits.conversion == TextureConversion::RepackGuestColor16) {
                RepackRow(dst, src, decoded->row_bytes, pixel_format);
            } else {
                std::memcpy(dst, src, decoded->row_bytes);
            }
        }
    }
    state.BufferCopy(target, upload.AcquireGPUAccess(range, BufferAccess::Read), planes,
                     copy.texture_level, 0, area, true);

    if (storage->scaled) {
        const auto destination = storage->scaled;
        const auto dst_extent = Dimensions(destination.Get(), copy.texture_level);
        const auto dst_x = std::min(u64{area.x} * res_scale, u64{dst_extent.width});
        const auto dst_y = std::min(u64{area.y} * res_scale, u64{dst_extent.height});
        const auto dst_right =
            std::min(u64{area.x + area.width} * res_scale, u64{dst_extent.width});
        const auto dst_top =
            std::min(u64{area.y + area.height} * res_scale, u64{dst_extent.height});
        const VideoCore::TextureBlit blit{copy.texture_level,
                                          copy.texture_level,
                                          0,
                                          0,
                                          copy.texture_rect,
                                          {static_cast<u32>(dst_x), static_cast<u32>(dst_top),
                                           static_cast<u32>(dst_right), static_cast<u32>(dst_y)}};
        if (dst_x < dst_right && dst_y < dst_top) {
            state.Keep(target);
            state.Keep(destination);
            const bool handled = Aspect() == TextureAspect::Color &&
                                 state.blitter.Filter(target.Get(), destination.Get(), blit,
                                                      Settings::values.texture_filter.GetValue(),
                                                      static_cast<float>(res_scale));
            if (!handled &&
                !state.Blit(target, destination, TextureRuntime::State::CurrentFormat(*storage),
                            blit, false)) {
                throw std::logic_error("Unable to initialize the scaled Metal upload");
            }
        }
    }
    storage->multisample_valid &= copy.texture_level != 0;
}

void TextureSurface::UploadCustom(const VideoCore::Material* replacement, u32 level) {
    auto& state = *runtime->state;
    if (!replacement || !replacement->IsDecoded()) {
        return;
    }
    const auto upload = [&](TextureResource destination, const VideoCore::CustomTexture* texture) {
        if (!destination || !texture) {
            return;
        }
        const auto extent = Dimensions(destination.Get(), level);
        const auto& format = state.instance.GetFormats().GetTraits(texture->format);
        const auto tight = MakeTextureBufferLayout(extent, format.block_width, format.block_height,
                                                   format.bytes_per_block, 1);
        const auto native = MakeTextureBufferLayout(extent, format.block_width, format.block_height,
                                                    format.bytes_per_block, 256);
        if (!tight || !native || !format.IsSupported() ||
            destination.Get().pixelFormat != format.native ||
            texture->data.size() < tight->length) {
            throw std::invalid_argument("Custom Metal texture data has an invalid layout");
        }
        const auto staging = state.allocator.AllocateBuffer(native->length, MemoryPool::Stream);
        const BufferRange range{0, staging.Size()};
        {
            const auto map = staging.MapWrite(range);
            for (std::size_t row = 0; row < native->rows; ++row) {
                std::memcpy(map.Bytes().data() + row * native->row_pitch,
                            texture->data.data() + row * tight->row_pitch, tight->row_bytes);
            }
        }
        const std::array planes{TextureRuntime::State::Plane{0, *native, MTLBlitOptionNone}};
        state.BufferCopy(destination, staging.AcquireGPUAccess(range, BufferAccess::Read), planes,
                         level, 0, Full(extent), true);
    };
    upload(storage->base, replacement->Map(VideoCore::MapType::Color));
    upload(storage->normal, replacement->Map(VideoCore::MapType::Normal));
    if (storage->scaled) {
        const auto source_extent = Dimensions(storage->base.Get(), level);
        const auto destination_extent = Dimensions(storage->scaled.Get(), level);
        const VideoCore::TextureBlit blit{
            level, level, 0, 0, Rect(source_extent), Rect(destination_extent)};
        if (!state.Blit(storage->base, storage->scaled,
                        TextureRuntime::State::CurrentFormat(*storage), blit, false)) {
            throw std::logic_error("Unable to transfer native custom texture to scaled storage");
        }
    }
    storage->multisample_valid &= level != 0;
}

void TextureSurface::Download(const VideoCore::BufferTextureCopy& copy,
                              const VideoCore::StagingData& staging) {
    auto& state = *runtime->state;
    const auto target = storage->base;
    const auto area = BlitBounds(copy.texture_rect);
    const Extent extent{area.width, area.height};
    const auto tight = MakeTextureBufferLayout(extent, 1, 1, GetInternalBytesPerPixel(), 1);
    if (!target || !AreaFits(target.Get(), copy.texture_level, 0, area) || !tight ||
        copy.buffer_offset != staging.offset || copy.buffer_size > staging.size ||
        copy.buffer_size > staging.mapped.size() || tight->length > copy.buffer_size) {
        throw std::invalid_argument(
            "Metal texture download rectangle or shared staging is invalid");
    }
    if (storage->scaled) {
        const auto scaled_extent = Dimensions(storage->scaled.Get(), copy.texture_level);
        const auto sx = std::min(u64{area.x} * res_scale, u64{scaled_extent.width});
        const auto sy = std::min(u64{area.y} * res_scale, u64{scaled_extent.height});
        const auto sx2 = std::min(u64{area.x + area.width} * res_scale, u64{scaled_extent.width});
        const auto sy2 = std::min(u64{area.y + area.height} * res_scale, u64{scaled_extent.height});
        const VideoCore::TextureBlit blit{copy.texture_level,
                                          copy.texture_level,
                                          0,
                                          0,
                                          {static_cast<u32>(sx), static_cast<u32>(sy2),
                                           static_cast<u32>(sx2), static_cast<u32>(sy)},
                                          copy.texture_rect};
        if (!state.Blit(storage->scaled, target, Format(), blit, false)) {
            throw std::logic_error("Unable to downscale native Metal texture for readback");
        }
    }

    const bool combined = Aspect() == TextureAspect::DepthStencil;
    const auto layout =
        MakeTextureBufferLayout(extent, 1, 1, GetInternalBytesPerPixel(), combined ? 1 : 256)
            .value();
    const auto output = state.allocator.AllocateBuffer(layout.length, MemoryPool::Readback);
    const BufferRange range{0, layout.length};
    if (combined) {
        if (layout.length > std::numeric_limits<u32>::max()) {
            throw std::overflow_error(
                "Packed Metal depth/stencil download exceeds transfer layout");
        }
        auto native_copy = copy;
        native_copy.buffer_offset = 0;
        native_copy.buffer_size = static_cast<u32>(layout.length);
        state.Keep(target);
        if (!state.blitter.DepthToBuffer(
                target.Get(), output.AcquireGPUAccess(range, BufferAccess::Write), native_copy)) {
            throw std::logic_error("Unable to read packed Metal depth/stencil");
        }
    } else {
        const std::array planes{TextureRuntime::State::Plane{0, layout, MTLBlitOptionNone}};
        state.BufferCopy(target, output.AcquireGPUAccess(range, BufferAccess::Write), planes,
                         copy.texture_level, 0, area, false);
    }
    runtime->Finish();
    const auto mapped = output.MapRead(range);
    auto* destination = reinterpret_cast<std::byte*>(staging.mapped.data());
    for (std::size_t y = 0; y < tight->rows; ++y) {
        auto* dst = destination + y * tight->row_pitch;
        const auto* src = mapped.Bytes().data() + y * layout.row_pitch;
        if (Format().conversion == TextureConversion::RepackGuestColor16) {
            RepackRow(dst, src, tight->row_bytes, pixel_format);
        } else {
            std::memcpy(dst, src, tight->row_bytes);
        }
    }
}

void TextureSurface::ScaleUp(u32 new_scale, u8 new_sample_count) {
    auto& state = *runtime->state;
    const auto existing = Resource();
    if (!existing) {
        return;
    }
    if (!new_scale || width > std::numeric_limits<u32>::max() / new_scale ||
        height > std::numeric_limits<u32>::max() / new_scale ||
        !state.instance.SupportsSampleCount(new_sample_count)) {
        throw std::invalid_argument("Unsupported Metal resolution or sample count change");
    }
    if (material || texture_type == TextureType::CubeMap) {
        new_sample_count = 1;
    }
    if (new_scale == res_scale && new_sample_count == sample_count) {
        return;
    }

    const auto traits = Format();
    TextureResource next_scaled;
    TextureResource next_multisample;
    const Extent scaled_size =
        material ? Dimensions(existing.Get(), 0) : Extent{width * new_scale, height * new_scale};
    if (new_scale != res_scale) {
        if (new_scale > 1) {
            auto scaled_format = traits;
            if (material && (traits.block_width != 1 || traits.block_height != 1)) {
                scaled_format = state.instance.GetFormats().GetTraits(PixelFormat::RGBA8);
            }
            next_scaled = state.Allocate(scaled_format, scaled_size,
                                         static_cast<u32>(existing.Get().mipmapLevelCount),
                                         texture_type, 1, DebugName(true, material != nullptr));
        }
        const auto destination = next_scaled ? next_scaled : storage->base;
        if (existing.Get() != destination.Get() && HasRenderUsage(destination.Get())) {
            for (u32 level = 0; level < existing.Get().mipmapLevelCount; ++level) {
                const auto from = Dimensions(existing.Get(), level);
                const auto to = Dimensions(destination.Get(), level);
                for (u32 layer = 0; layer < Layers(existing.Get()); ++layer) {
                    const VideoCore::TextureBlit blit{level, level,      layer,
                                                      layer, Rect(from), Rect(to)};
                    if (!state.Blit(existing, destination, traits, blit, false)) {
                        throw std::logic_error(
                            "Metal rescale could not preserve a valid subresource");
                    }
                }
            }
        }
    } else {
        next_scaled = storage->scaled;
    }

    if (new_sample_count > 1) {
        next_multisample = state.Allocate(traits, scaled_size, 1, texture_type, new_sample_count,
                                          DebugName(new_scale > 1, false, new_sample_count));
    }
    storage->scaled = std::move(next_scaled);
    storage->multisample = std::move(next_multisample);
    storage->multisample_valid = false;
    storage->views.clear();
    storage->feedback = {};
    storage->shadow = {};
    storage->shadow_sample = {};
    storage->shadow_buffer.reset();
    res_scale = new_scale;
    storage->resolution_scale = new_scale;
    sample_count = new_sample_count;
}

struct Framebuffer::State {
    State(TextureRuntime& runtime, TextureSurface* color, TextureSurface* depth)
        : runtime(runtime), color(color ? color->storage : nullptr),
          depth(depth ? depth->storage : nullptr),
          color_format(color ? color->pixel_format : PixelFormat::Invalid),
          depth_format(depth ? depth->pixel_format : PixelFormat::Invalid) {}

    TextureResource Active(const std::shared_ptr<SurfaceStorage>& surface) const {
        return surface ? surface->Current() : TextureResource{};
    }

    RenderAttachment Make(const std::shared_ptr<SurfaceStorage>& surface, u32 level, u32 samples) {
        const auto current = Active(surface);
        if (!current) {
            return {};
        }
        if (samples == 1) {
            return Attachment(current, {}, level, 0, MTLLoadActionLoad);
        }
        if (!surface->multisample_valid) {
            const auto rect = Rect(Dimensions(current.Get(), 0));
            const VideoCore::TextureBlit seed{0, 0, 0, 0, rect, rect};
            if (!runtime.state->Blit(current, surface->multisample,
                                     TextureRuntime::State::CurrentFormat(*surface), seed, false,
                                     MTLLoadActionDontCare)) {
                throw std::logic_error("Unable to seed native Metal multisample attachment");
            }
            surface->multisample_valid = true;
        }
        return Attachment(surface->multisample, current, 0, 0, MTLLoadActionLoad);
    }

    TextureRuntime& runtime;
    std::shared_ptr<SurfaceStorage> color;
    std::shared_ptr<SurfaceStorage> depth;
    PixelFormat color_format;
    PixelFormat depth_format;
};

Framebuffer::Framebuffer(TextureRuntime& runtime, const VideoCore::FramebufferParams& params,
                         TextureSurface* color, TextureSurface* depth)
    : FramebufferParams(params), state(std::make_unique<State>(runtime, color, depth)) {}

Framebuffer::~Framebuffer() = default;
Framebuffer::Framebuffer(Framebuffer&&) noexcept = default;
Framebuffer& Framebuffer::operator=(Framebuffer&&) noexcept = default;

bool Framebuffer::IsValid() const noexcept {
    if (shadow_rendering) {
        return static_cast<bool>(state->Active(state->color)) && color_level == 0;
    }
    const auto extent = Extent();
    return extent.width != 0 && extent.height != 0;
}

PixelFormat Framebuffer::Format(SurfaceType type) const noexcept {
    return type == SurfaceType::Color ? state->color_format : state->depth_format;
}

u32 Framebuffer::Scale() const noexcept {
    const auto& surface = state->color ? state->color : state->depth;
    return surface ? surface->resolution_scale : 1;
}

u8 Framebuffer::Samples() const noexcept {
    if (shadow_rendering) {
        return 1;
    }
    const auto count = [](const auto& surface, u32 level) -> u8 {
        return surface && surface->multisample && !level
                   ? static_cast<u8>(surface->multisample.Get().sampleCount)
                   : 1;
    };
    const u8 color = count(state->color, color_level);
    const u8 depth = count(state->depth, depth_level);
    if (state->Active(state->color) && state->Active(state->depth)) {
        return color == depth ? color : 1;
    }
    return state->Active(state->color) ? color : depth;
}

VideoCore::Extent Framebuffer::Extent() const noexcept {
    const auto color = state->Active(state->color);
    const auto depth = shadow_rendering ? TextureResource{} : state->Active(state->depth);
    const auto from_color = Dimensions(color.Get(), color_level);
    const auto from_depth = Dimensions(depth.Get(), depth_level);
    if (color && depth) {
        return {std::min(from_color.width, from_depth.width),
                std::min(from_color.height, from_depth.height)};
    }
    return color ? from_color : from_depth;
}

RenderPass Framebuffer::MakePass(RenderArea area) {
    const auto extent = Extent();
    if (!IsValid() || !area.Fits(extent.width, extent.height)) {
        throw std::invalid_argument("Metal framebuffer render area exceeds its attachments");
    }
    RenderPass pass;
    pass.area = area;
    pass.width = extent.width;
    pass.height = extent.height;
    pass.samples = Samples();
    if (shadow_rendering) {
        pass.shadow_storage = state->color->shadow.Get();
        state->runtime.state->Keep(state->color->shadow);
        return pass;
    }
    pass.color = state->Make(state->color, color_level, pass.samples);
    pass.depth = state->Make(state->depth, depth_level, pass.samples);
    if (state->depth && state->depth->format.aspect == TextureAspect::DepthStencil) {
        pass.stencil = pass.depth;
    }
    for (const auto& surface : {state->color, state->depth}) {
        if (surface) {
            state->runtime.state->Keep(surface->Current());
            if (pass.samples > 1) {
                state->runtime.state->Keep(surface->multisample);
            }
        }
    }
    if (pass.samples == 1) {
        if (state->color && color_level == 0) {
            state->color->multisample_valid = false;
        }
        if (state->depth && depth_level == 0) {
            state->depth->multisample_valid = false;
        }
    }
    return pass;
}

void Framebuffer::Retain(EncodingContext& context) const {
    for (const auto& surface : {state->color, state->depth}) {
        if (surface) {
            surface->Current().Retain(context);
            surface->multisample.Retain(context);
            surface->shadow.Retain(context);
        }
    }
}

Sampler::Sampler(TextureRuntime& runtime, const VideoCore::SamplerParams& params)
    : metadata(GetSamplerMetadata(params)) {
    using TC = VideoCore::SamplerParams::TextureConfig;
    const auto address = [](TC::WrapMode mode) {
        switch (mode) {
        case TC::Repeat:
        case TC::Repeat2:
        case TC::Repeat3:
            return MTLSamplerAddressModeRepeat;
        case TC::MirroredRepeat:
            return MTLSamplerAddressModeMirrorRepeat;
        default:
            // The PICA shader uses the exact guest wrap and border uniform at an edge.
            return MTLSamplerAddressModeClampToEdge;
        }
    };
    const auto filter = [](TC::TextureFilter request) {
        return request == TC::Linear ? MTLSamplerMinMagFilterLinear : MTLSamplerMinMagFilterNearest;
    };
    auto desc = [MTLSamplerDescriptor new];
    desc.minFilter = filter(params.min_filter);
    desc.magFilter = filter(params.mag_filter);
    desc.mipFilter =
        params.mip_filter == TC::Linear ? MTLSamplerMipFilterLinear : MTLSamplerMipFilterNearest;
    desc.sAddressMode = address(params.wrap_s);
    desc.tAddressMode = address(params.wrap_t);
    desc.rAddressMode = MTLSamplerAddressModeClampToEdge;
    desc.lodMinClamp = static_cast<float>(params.lod_min);
    desc.lodMaxClamp = static_cast<float>(std::max(params.lod_min, params.lod_max));
    desc.maxAnisotropy = params.min_filter == TC::Linear && params.mag_filter == TC::Linear
                             ? runtime.MaxAnisotropy()
                             : 1u;
    desc.supportArgumentBuffers = YES;
    sampler = CheckResource([runtime.GetInstance().GetDevice() newSamplerStateWithDescriptor:desc],
                            "Allocate native PICA sampler");
}

Sampler::~Sampler() = default;

DebugScope::DebugScope(TextureRuntime& runtime, Common::Vec4f color, std::string_view label) {
    if (!runtime.GetInstance().IsCaptureActive() || label.empty()) {
        return;
    }
    static_cast<void>(color);
    NSString* name = [[NSString alloc] initWithBytes:label.data()
                                              length:label.size()
                                            encoding:NSUTF8StringEncoding];
    if (name) {
        scheduler = &runtime.GetScheduler();
        scheduler->Record([name](EncodingContext& context) {
            [context.GetCommandBuffer() pushDebugGroup:name];
            [context.GetCommandBuffer() popDebugGroup];
        });
    }
}

DebugScope::~DebugScope() {
    if (scheduler) {
        try {
            scheduler->Record([](EncodingContext& context) {
                [context.GetCommandBuffer() pushDebugGroup:@"End texture operation"];
                [context.GetCommandBuffer() popDebugGroup];
            });
        } catch (const std::exception& error) {
            LOG_WARNING(Render, "Unable to end Metal texture debug group: {}", error.what());
        }
    }
}

} // namespace Metal
