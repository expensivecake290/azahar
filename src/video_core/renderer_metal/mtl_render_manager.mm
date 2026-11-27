// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.txt file included.

#import <Foundation/Foundation.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <stdexcept>
#include <utility>
#include <vector>
#include "video_core/renderer_metal/mtl_instance.h"
#include "video_core/renderer_metal/mtl_render_manager.h"

namespace Metal {
namespace {

constexpr char ClearSource[] = R"MSL(
#include <metal_stdlib>
using namespace metal;

struct ClearData {
    uint4 color;
    float depth;
};

struct ClearPosition {
    float4 position [[position]];
};

vertex ClearPosition metaliris_clear_vertex(uint vid [[vertex_id]],
                                             constant ClearData& data [[buffer(0)]]) {
    const float2 xy = float2(vid == 1 ? 3.0f : -1.0f, vid == 2 ? 3.0f : -1.0f);
    return {float4(xy, data.depth, 1.0f)};
}

fragment float4 metaliris_clear_float(constant ClearData& data [[buffer(0)]]) {
    return as_type<float4>(data.color);
}

fragment uint4 metaliris_clear_uint(constant ClearData& data [[buffer(0)]]) {
    return data.color;
}

fragment int4 metaliris_clear_int(constant ClearData& data [[buffer(0)]]) {
    return as_type<int4>(data.color);
}
)MSL";

enum class ColorType { Float, Unsigned, Signed };

ColorType ColorTypeFor(MTLPixelFormat format) {
    switch (format) {
    case MTLPixelFormatR8Uint:
    case MTLPixelFormatR16Uint:
    case MTLPixelFormatR32Uint:
    case MTLPixelFormatRG8Uint:
    case MTLPixelFormatRG16Uint:
    case MTLPixelFormatRG32Uint:
    case MTLPixelFormatRGBA8Uint:
    case MTLPixelFormatRGBA16Uint:
    case MTLPixelFormatRGBA32Uint:
    case MTLPixelFormatRGB10A2Uint:
        return ColorType::Unsigned;
    case MTLPixelFormatR8Sint:
    case MTLPixelFormatR16Sint:
    case MTLPixelFormatR32Sint:
    case MTLPixelFormatRG8Sint:
    case MTLPixelFormatRG16Sint:
    case MTLPixelFormatRG32Sint:
    case MTLPixelFormatRGBA8Sint:
    case MTLPixelFormatRGBA16Sint:
    case MTLPixelFormatRGBA32Sint:
        return ColorType::Signed;
    default:
        return ColorType::Float;
    }
}

bool IsDepth(MTLPixelFormat format) {
    return format == MTLPixelFormatDepth16Unorm || format == MTLPixelFormatDepth32Float ||
           format == MTLPixelFormatDepth32Float_Stencil8 ||
           format == MTLPixelFormatDepth24Unorm_Stencil8;
}

bool IsStencil(MTLPixelFormat format) {
    return format == MTLPixelFormatStencil8 || format == MTLPixelFormatDepth32Float_Stencil8 ||
           format == MTLPixelFormatDepth24Unorm_Stencil8 || format == MTLPixelFormatX32_Stencil8 ||
           format == MTLPixelFormatX24_Stencil8;
}

std::size_t LevelSize(std::size_t size, u32 level) {
    if (level >= std::numeric_limits<std::size_t>::digits) {
        return 1;
    }
    return std::max<std::size_t>(1, size >> level);
}

void ValidateTexture(const RenderTexture& view, id<MTLDevice> device, RenderArea region) {
    id<MTLTexture> texture = view.texture;
    if (!texture || texture.device != device || view.level >= texture.mipmapLevelCount ||
        region.x > LevelSize(texture.width, view.level) ||
        region.y > LevelSize(texture.height, view.level) ||
        region.width > LevelSize(texture.width, view.level) - region.x ||
        region.height > LevelSize(texture.height, view.level) - region.y) {
        throw std::invalid_argument("Metal render subresource or extent is invalid");
    }

    bool valid = view.slice == 0 && view.depth_plane == 0;
    switch (texture.textureType) {
    case MTLTextureType1DArray:
    case MTLTextureType2DArray:
    case MTLTextureType2DMultisampleArray:
        valid = view.slice < texture.arrayLength && view.depth_plane == 0;
        break;
    case MTLTextureTypeCube:
    case MTLTextureTypeCubeArray:
        valid = view.slice / 6 < texture.arrayLength && view.depth_plane == 0;
        break;
    case MTLTextureType3D:
        valid = view.slice == 0 && view.depth_plane < LevelSize(texture.depth, view.level);
        break;
    case MTLTextureTypeTextureBuffer:
        valid = false;
        break;
    default:
        break;
    }
    if (!valid) {
        throw std::invalid_argument("Metal render slice or depth plane is outside the texture");
    }
}

id<MTLResource> Backing(id<MTLTexture> texture) {
    while (texture.parentTexture) {
        texture = texture.parentTexture;
    }
    if (id<MTLBuffer> buffer = texture.buffer) {
        return buffer;
    }
    return texture;
}

bool Resolves(MTLStoreAction action) {
    return action == MTLStoreActionMultisampleResolve ||
           action == MTLStoreActionStoreAndMultisampleResolve;
}

bool StoresSamples(MTLStoreAction action) {
    return action == MTLStoreActionStore || action == MTLStoreActionStoreAndMultisampleResolve;
}

void ValidateAttachment(const RenderAttachment& attachment, id<MTLDevice> device, RenderArea frame,
                        u32 samples) {
    if (!attachment.target.texture) {
        if (attachment.resolve.texture || attachment.load_action == MTLLoadActionClear) {
            throw std::invalid_argument("Metal resolve or clear requires a render attachment");
        }
        return;
    }

    ValidateTexture(attachment.target, device, frame);
    id<MTLTexture> target = attachment.target.texture;
    const bool has_usage =
        target.usage == MTLTextureUsageUnknown || (target.usage & MTLTextureUsageRenderTarget);
    if (!has_usage || target.sampleCount != samples) {
        throw std::invalid_argument("Metal render attachment has incompatible usage or samples");
    }

    switch (attachment.load_action) {
    case MTLLoadActionLoad:
    case MTLLoadActionDontCare:
    case MTLLoadActionClear:
        break;
    default:
        throw std::invalid_argument("Invalid Metal render load action");
    }
    if (!StoresSamples(attachment.store_action) && !Resolves(attachment.store_action) &&
        attachment.store_action != MTLStoreActionDontCare) {
        throw std::invalid_argument("Invalid Metal render store action");
    }
    if (target.storageMode == MTLStorageModeMemoryless &&
        (attachment.load_action == MTLLoadActionLoad || StoresSamples(attachment.store_action))) {
        throw std::invalid_argument("Memoryless Metal attachment cannot load or store samples");
    }

    if (Resolves(attachment.store_action)) {
        ValidateTexture(attachment.resolve, device, frame);
        id<MTLTexture> resolved = attachment.resolve.texture;
        if (samples == 1 || resolved.sampleCount != 1 ||
            resolved.pixelFormat != target.pixelFormat || Backing(resolved) == Backing(target) ||
            resolved.storageMode == MTLStorageModeMemoryless ||
            (resolved.usage != MTLTextureUsageUnknown &&
             !(resolved.usage & MTLTextureUsageRenderTarget))) {
            throw std::invalid_argument("Metal multisample resolve target is incompatible");
        }
    } else if (attachment.resolve.texture) {
        throw std::invalid_argument("Metal resolve target requires a resolve store action");
    }
}

bool RequiresScissoredClear(const RenderAttachment& attachment, const RenderArea& area) {
    const auto& view = attachment.target;
    return view.texture && attachment.load_action == MTLLoadActionClear &&
           (area.x != 0 || area.y != 0 || area.width != LevelSize(view.texture.width, view.level) ||
            area.height != LevelSize(view.texture.height, view.level));
}

struct ClearMask {
    bool color{};
    bool depth{};
    bool stencil{};

    [[nodiscard]] bool Any() const noexcept {
        return color || depth || stencil;
    }

    auto operator<=>(const ClearMask&) const = default;
};

ClearMask ScissoredClears(const RenderPass& pass) {
    return {RequiresScissoredClear(pass.color, pass.area),
            RequiresScissoredClear(pass.depth, pass.area),
            RequiresScissoredClear(pass.stencil, pass.area)};
}

void ValidatePass(const Instance& instance, const RenderPass& pass) {
    const bool supported_samples = instance.SupportsSampleCount(pass.samples);
    if (!pass.area.Fits(pass.width, pass.height) || !supported_samples) {
        throw std::invalid_argument("Metal render pass has invalid bounds or sample count");
    }

    const auto device = instance.GetDevice();
    const RenderArea frame{0, 0, pass.width, pass.height};
    ValidateAttachment(pass.color, device, frame, pass.samples);
    ValidateAttachment(pass.depth, device, frame, pass.samples);
    ValidateAttachment(pass.stencil, device, frame, pass.samples);
    const auto formats = pass.Formats();
    if ((pass.color.target.texture && (IsDepth(formats.color) || IsStencil(formats.color))) ||
        (pass.depth.target.texture && !IsDepth(formats.depth)) ||
        (pass.stencil.target.texture && !IsStencil(formats.stencil))) {
        throw std::invalid_argument("Metal render attachment is incompatible with its aspect");
    }
    if (pass.depth.target.texture && pass.stencil.target.texture &&
        (IsStencil(formats.depth) || IsDepth(formats.stencil)) &&
        (pass.depth.target != pass.stencil.target || pass.depth.resolve != pass.stencil.resolve)) {
        throw std::invalid_argument("Packed Metal depth and stencil must share a subresource");
    }
    if (pass.shadow_storage) {
        ValidateTexture({pass.shadow_storage}, device, frame);
        if (pass.color.target.texture || pass.depth.target.texture || pass.stencil.target.texture ||
            pass.shadow_storage.sampleCount != 1 ||
            pass.shadow_storage.storageMode == MTLStorageModeMemoryless ||
            (pass.shadow_storage.usage != MTLTextureUsageUnknown &&
             !(pass.shadow_storage.usage & MTLTextureUsageShaderWrite))) {
            throw std::invalid_argument("Metal shadow target requires a writable storage texture");
        }
    }
    if (pass.tile_feedback &&
        (!pass.color.target.texture || !instance.GetCapabilities().raster_order_groups)) {
        throw std::invalid_argument(
            "Metal tile feedback requires raster order groups and a color target");
    }

    const ClearMask scissored = ScissoredClears(pass);
    if ((scissored.color && pass.color.target.texture.storageMode == MTLStorageModeMemoryless) ||
        (scissored.depth && pass.depth.target.texture.storageMode == MTLStorageModeMemoryless) ||
        (scissored.stencil &&
         pass.stencil.target.texture.storageMode == MTLStorageModeMemoryless)) {
        throw std::invalid_argument(
            "A partial Metal clear must preserve the rest of its attachment");
    }
    if (pass.depth.load_action == MTLLoadActionClear &&
        (!std::isfinite(pass.clear_depth) || pass.clear_depth < 0.0 || pass.clear_depth > 1.0)) {
        throw std::invalid_argument("Metal clear depth must be finite and within the depth range");
    }
    if (pass.stencil.load_action == MTLLoadActionClear && pass.clear_stencil > 0xFFu) {
        throw std::invalid_argument("Metal clear stencil exceeds eight bits");
    }
    if (pass.color.load_action == MTLLoadActionClear &&
        (!std::isfinite(pass.clear_color.red) || !std::isfinite(pass.clear_color.green) ||
         !std::isfinite(pass.clear_color.blue) || !std::isfinite(pass.clear_color.alpha))) {
        throw std::invalid_argument("Metal clear color components must be finite");
    }
}

void SetAttachment(MTLRenderPassAttachmentDescriptor* native, const RenderAttachment& attachment,
                   bool partial) {
    if (!attachment.target.texture) {
        return;
    }
    native.texture = attachment.target.texture;
    native.level = attachment.target.level;
    native.slice = attachment.target.slice;
    native.depthPlane = attachment.target.depth_plane;
    native.loadAction = partial ? MTLLoadActionLoad : attachment.load_action;
    native.storeAction = attachment.store_action;
    if (attachment.resolve.texture) {
        native.resolveTexture = attachment.resolve.texture;
        native.resolveLevel = attachment.resolve.level;
        native.resolveSlice = attachment.resolve.slice;
        native.resolveDepthPlane = attachment.resolve.depth_plane;
    }
}

void AddAttachmentUse(EncodingContext& context, std::vector<ResourceUse>& uses,
                      const RenderAttachment& attachment, bool partial) {
    if (id<MTLTexture> texture = attachment.target.texture) {
        if (texture.storageMode == MTLStorageModeMemoryless) {
            context.Retain(texture);
        } else {
            const auto access = partial || attachment.load_action == MTLLoadActionLoad
                                    ? BufferAccess::ReadWrite
                                    : BufferAccess::Write;
            uses.push_back({texture, access, MTLStageFragment});
        }
    }
    if (attachment.resolve.texture) {
        uses.push_back({attachment.resolve.texture, BufferAccess::Write, MTLStageFragment});
    }
}

MTL4LibraryFunctionDescriptor* MakeFunction(id<MTLLibrary> library, NSString* name) {
    auto descriptor = [MTL4LibraryFunctionDescriptor new];
    descriptor.library = library;
    descriptor.name = name;
    return descriptor;
}

NSString* ClearFunction(ColorType type) {
    switch (type) {
    case ColorType::Unsigned:
        return @"metaliris_clear_uint";
    case ColorType::Signed:
        return @"metaliris_clear_int";
    case ColorType::Float:
        return @"metaliris_clear_float";
    }
    throw std::logic_error("Invalid Metal clear component type");
}

struct alignas(16) ClearData {
    std::array<u32, 4> color{};
    f32 depth{};
};
static_assert(sizeof(ClearData) == 32);

u32 PackClearComponent(double value, ColorType type) {
    switch (type) {
    case ColorType::Unsigned:
        return static_cast<u32>(
            std::clamp(value, 0.0, static_cast<double>(std::numeric_limits<u32>::max())));
    case ColorType::Signed: {
        const auto clamped = std::clamp(value, static_cast<double>(std::numeric_limits<s32>::min()),
                                        static_cast<double>(std::numeric_limits<s32>::max()));
        return std::bit_cast<u32>(static_cast<s32>(clamped));
    }
    case ColorType::Float:
        return std::bit_cast<u32>(static_cast<f32>(value));
    }
    throw std::logic_error("Invalid Metal clear component type");
}

} // Anonymous namespace

struct RenderManager::State {
    explicit State(const MemoryAllocator& allocator) : allocator(allocator) {}

    struct ClearKey {
        RenderFormats formats;
        ClearMask mask;

        auto operator<=>(const ClearKey&) const = default;
    };

    struct ClearPipeline {
        id<MTLRenderPipelineState> pipeline;
        id<MTLDepthStencilState> depth_stencil;
    };

    ClearPipeline GetClearPipeline(id<MTLDevice> device, const ClearKey& key) {
        if (const auto item = clear_pipelines.find(key); item != clear_pipelines.end()) {
            return item->second;
        }
        NSError* error = nil;
        if (!clear_library) {
            auto options = [MTLCompileOptions new];
            options.languageVersion = MTLLanguageVersion4_0;
            auto compiled = [device newLibraryWithSource:[NSString stringWithUTF8String:ClearSource]
                                                 options:options
                                                   error:&error];
            clear_library = CheckResource(compiled, "Compile Metal partial clear MSL", error);
        }
        if (!compiler) {
            error = nil;
            auto created = [device newCompilerWithDescriptor:[MTL4CompilerDescriptor new]
                                                       error:&error];
            compiler = CheckResource(created, "Create Metal partial clear compiler", error);
        }

        auto descriptor = [MTL4RenderPipelineDescriptor new];
        descriptor.label = @"Metaliris partial clear";
        descriptor.vertexFunctionDescriptor =
            MakeFunction(clear_library, @"metaliris_clear_vertex");
        descriptor.rasterSampleCount = key.formats.samples;
        if (key.formats.color != MTLPixelFormatInvalid) {
            descriptor.colorAttachments[0].pixelFormat = key.formats.color;
            descriptor.colorAttachments[0].writeMask =
                key.mask.color ? MTLColorWriteMaskAll : MTLColorWriteMaskNone;
        }
        if (key.mask.color) {
            descriptor.fragmentFunctionDescriptor =
                MakeFunction(clear_library, ClearFunction(ColorTypeFor(key.formats.color)));
        }
        error = nil;
        auto pipeline = [compiler newRenderPipelineStateWithDescriptor:descriptor
                                                   compilerTaskOptions:nil
                                                                 error:&error];
        pipeline = CheckResource(pipeline, "Compile Metal partial clear pipeline", error);

        auto state = [MTLDepthStencilDescriptor new];
        state.depthCompareFunction = MTLCompareFunctionAlways;
        state.depthWriteEnabled = key.mask.depth;
        if (key.mask.stencil) {
            auto stencil = [MTLStencilDescriptor new];
            stencil.stencilCompareFunction = MTLCompareFunctionAlways;
            stencil.depthStencilPassOperation = MTLStencilOperationReplace;
            stencil.writeMask = 0xFF;
            state.frontFaceStencil = stencil;
            state.backFaceStencil = stencil;
        }
        auto depth_stencil = CheckResource([device newDepthStencilStateWithDescriptor:state],
                                           "Create Metal partial clear depth/stencil state");
        ClearPipeline result{pipeline, depth_stencil};
        clear_pipelines.emplace(key, result);
        return result;
    }

    void EncodePartialClear(EncodingContext& context, const RenderPass& pass, ClearMask mask) {
        const auto device = context.GetCommandBuffer().device;
        const auto formats = pass.Formats();
        const auto pipeline = GetClearPipeline(device, {formats, mask});
        const auto scalar = ColorTypeFor(formats.color);
        const ClearData constants{
            {PackClearComponent(pass.clear_color.red, scalar),
             PackClearComponent(pass.clear_color.green, scalar),
             PackClearComponent(pass.clear_color.blue, scalar),
             PackClearComponent(pass.clear_color.alpha, scalar)},
            mask.depth ? static_cast<f32>(pass.clear_depth) : 0.0f,
        };
        const BufferRange range{0, sizeof(constants)};
        auto buffer = allocator.AllocateBuffer(sizeof(constants), MemoryPool::Transient);
        {
            auto cpu = buffer.MapWrite(range);
            std::memcpy(cpu.Bytes().data(), &constants, sizeof(constants));
        }
        auto gpu = buffer.AcquireGPUAccess(range, BufferAccess::Read);
        auto descriptor = [MTL4ArgumentTableDescriptor new];
        descriptor.maxBufferBindCount = 1;
        descriptor.initializeBindings = YES;
        NSError* error = nil;
        auto table = [device newArgumentTableWithDescriptor:descriptor error:&error];
        table = CheckResource(table, "Allocate Metal clear argument table", error);
        [table setAddress:gpu.GetBuffer().gpuAddress atIndex:0];
        context.Retain(std::move(gpu));
        context.Retain(table);
        context.Retain(pipeline.pipeline);
        context.Retain(pipeline.depth_stencil);

        [encoder setRenderPipelineState:pipeline.pipeline];
        [encoder setDepthStencilState:pipeline.depth_stencil];
        [encoder setStencilReferenceValue:pass.clear_stencil];
        [encoder setViewport:MTLViewport{0.0, 0.0, static_cast<double>(pass.width),
                                         static_cast<double>(pass.height), 0.0, 1.0}];
        [encoder setScissorRect:MTLScissorRect{pass.area.x, pass.area.y, pass.area.width,
                                               pass.area.height}];
        [encoder setArgumentTable:table
                         atStages:mask.color ? MTLRenderStageVertex | MTLRenderStageFragment
                                             : MTLRenderStageVertex];
        [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
        [encoder setDepthStencilState:nil];
    }

    void Begin(EncodingContext& context, const RenderPass& pass) {
        if (encoder) {
            throw std::logic_error("Metal render worker already has an open pass");
        }
        const ClearMask scissored = ScissoredClears(pass);
        auto descriptor = [MTL4RenderPassDescriptor new];
        descriptor.renderTargetWidth = pass.width;
        descriptor.renderTargetHeight = pass.height;
        descriptor.renderTargetArrayLength = 1;
        descriptor.defaultRasterSampleCount = pass.samples;
        SetAttachment(descriptor.colorAttachments[0], pass.color, scissored.color);
        SetAttachment(descriptor.depthAttachment, pass.depth, scissored.depth);
        SetAttachment(descriptor.stencilAttachment, pass.stencil, scissored.stencil);
        descriptor.colorAttachments[0].clearColor = pass.clear_color;
        descriptor.depthAttachment.clearDepth = pass.clear_depth;
        descriptor.stencilAttachment.clearStencil = pass.clear_stencil;
        descriptor.depthAttachment.depthResolveFilter = MTLMultisampleDepthResolveFilterSample0;
        descriptor.stencilAttachment.stencilResolveFilter =
            MTLMultisampleStencilResolveFilterSample0;

        std::vector<ResourceUse> attachments;
        attachments.reserve(7);
        AddAttachmentUse(context, attachments, pass.color, scissored.color);
        AddAttachmentUse(context, attachments, pass.depth, scissored.depth);
        AddAttachmentUse(context, attachments, pass.stencil, scissored.stencil);
        if (pass.shadow_storage) {
            attachments.push_back({pass.shadow_storage, BufferAccess::ReadWrite, MTLStageFragment});
        }
        encoder = CheckResource(
            [context.GetCommandBuffer() renderCommandEncoderWithDescriptor:descriptor],
            "Begin Metal 4 render pass");
        context.Synchronize(encoder, attachments);
        worker_tick = context.CurrentTick();
        [encoder setScissorRect:MTLScissorRect{pass.area.x, pass.area.y, pass.area.width,
                                               pass.area.height}];
        if (scissored.Any()) {
            EncodePartialClear(context, pass, scissored);
        }
    }

    static void QueueEnd(Scheduler& scheduler, const std::shared_ptr<State>& state) {
        if (!state->planned) {
            return;
        }
        scheduler.Record([state](EncodingContext& context) {
            context.EndPass(RenderManager::RequireEncoder(*state, context));
            state->encoder = nil;
            state->worker_tick = 0;
        });
        state->planned.reset();
    }

    MemoryAllocator allocator;
    std::optional<RenderPass> planned;
    id<MTL4RenderCommandEncoder> encoder = nil;
    u64 worker_tick{};
    id<MTLLibrary> clear_library = nil;
    id<MTL4Compiler> compiler = nil;
    std::map<ClearKey, ClearPipeline> clear_pipelines;
};

RenderFormats RenderPass::Formats() const noexcept {
    return {color.target.texture.pixelFormat, depth.target.texture.pixelFormat,
            stencil.target.texture.pixelFormat, samples};
}

RenderManager::RenderManager(const Instance& instance, Scheduler& scheduler,
                             const MemoryAllocator& allocator)
    : instance(instance), scheduler(scheduler), state(std::make_shared<State>(allocator)) {
    if (instance.GetDevice() != scheduler.GetMasterSemaphore().GetDevice()) {
        throw std::invalid_argument("Metal render manager and scheduler devices differ");
    }
    scheduler.RegisterOnSubmit([weak = std::weak_ptr(state), &scheduler] {
        if (auto active = weak.lock()) {
            State::QueueEnd(scheduler, active);
        }
    });
}

RenderManager::~RenderManager() = default;

void RenderManager::BeginRendering(const RenderPass& pass) {
    ValidatePass(instance, pass);
    if (state->planned && state->planned->CanContinueWith(pass)) {
        return;
    }
    EndRendering();
    scheduler.Record([state = state, pass](EncodingContext& context) {
        context.Retain(state);
        state->Begin(context, pass);
    });
    state->planned = pass;
    scheduler.MakeDirty(StateFlags::Pipeline | StateFlags::ArgumentTables);
}

void RenderManager::BeginRendering(RenderPass pass, const Common::Rectangle<u32>& draw_rect) {
    const auto area = RenderArea::FromRectangle(draw_rect);
    if (!area) {
        throw std::invalid_argument("Metal render rectangle must have a positive extent");
    }
    pass.area = *area;
    BeginRendering(pass);
}

void RenderManager::EndRendering() {
    State::QueueEnd(scheduler, state);
}

void RenderManager::RequireActivePass() const {
    if (!state->planned) {
        throw std::logic_error("Metal draw requires an open render pass");
    }
}

id<MTL4RenderCommandEncoder> RenderManager::RequireEncoder(const State& state,
                                                           EncodingContext& context) {
    if (!state.encoder || state.worker_tick != context.CurrentTick()) {
        throw std::logic_error("Metal draw does not belong to the worker's active render pass");
    }
    return state.encoder;
}

RenderFormats RenderManager::GetRenderpass(VideoCore::PixelFormat color,
                                           VideoCore::PixelFormat depth, u32 sample_count) const {
    if (!instance.SupportsSampleCount(sample_count)) {
        throw std::invalid_argument("Unsupported Metal render sample count");
    }
    const auto& formats = instance.GetFormats();
    const auto& color_format = formats.GetTraits(color);
    const auto& depth_format = formats.GetTraits(depth);
    if ((color != VideoCore::PixelFormat::Invalid &&
         (color_format.aspect != TextureAspect::Color ||
          !(color_format.usage & MTLTextureUsageRenderTarget))) ||
        (depth != VideoCore::PixelFormat::Invalid && depth_format.aspect != TextureAspect::Depth &&
         depth_format.aspect != TextureAspect::DepthStencil)) {
        throw std::invalid_argument("Invalid PICA color or depth render format");
    }
    return {color_format.native, depth_format.native,
            depth_format.aspect == TextureAspect::DepthStencil ? depth_format.native
                                                               : MTLPixelFormatInvalid,
            sample_count};
}

void RenderManager::CopyForFeedback(const RenderTexture& source, const RenderTexture& destination,
                                    RenderArea area) {
    const auto device = instance.GetDevice();
    ValidateTexture(source, device, area);
    ValidateTexture(destination, device, area);
    if (!area.width || !area.height ||
        source.texture.sampleCount != destination.texture.sampleCount ||
        LevelSize(source.texture.width, source.level) !=
            LevelSize(destination.texture.width, destination.level) ||
        LevelSize(source.texture.height, source.level) !=
            LevelSize(destination.texture.height, destination.level) ||
        source.texture.pixelFormat != destination.texture.pixelFormat ||
        IsDepth(source.texture.pixelFormat) || IsStencil(source.texture.pixelFormat) ||
        Backing(source.texture) == Backing(destination.texture) || source.texture.framebufferOnly ||
        destination.texture.framebufferOnly ||
        source.texture.storageMode == MTLStorageModeMemoryless ||
        destination.texture.storageMode == MTLStorageModeMemoryless) {
        throw std::invalid_argument(
            "Metal feedback requires compatible color texture subresources");
    }
    if (state->planned && source == state->planned->color.target &&
        !StoresSamples(state->planned->color.store_action)) {
        throw std::logic_error("Metal feedback cannot copy a discarded render attachment");
    }
    EndRendering();
    scheduler.Record([source, destination, area](EncodingContext& context) {
        const std::array uses{
            ResourceUse{source.texture, BufferAccess::Read, MTLStageBlit},
            ResourceUse{destination.texture, BufferAccess::Write, MTLStageBlit},
        };
        auto encoder = CheckResource([context.GetCommandBuffer() computeCommandEncoder],
                                     "Encode Metal render feedback copy");
        context.Synchronize(encoder, uses);
        [encoder copyFromTexture:source.texture
                     sourceSlice:source.slice
                     sourceLevel:source.level
                    sourceOrigin:MTLOriginMake(area.x, area.y, source.depth_plane)
                      sourceSize:MTLSizeMake(area.width, area.height, 1)
                       toTexture:destination.texture
                destinationSlice:destination.slice
                destinationLevel:destination.level
               destinationOrigin:MTLOriginMake(area.x, area.y, destination.depth_plane)];
        context.EndPass(encoder);
    });
}

} // namespace Metal
