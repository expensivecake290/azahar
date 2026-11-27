#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <fstream>
#include <future>
#include <mutex>
#include <stdexcept>
#include "video_core/renderer_metal/mtl_graphics_pipeline.h"
#include "video_core/renderer_metal/mtl_instance.h"
#include "video_core/renderer_metal/mtl_render_manager.h"

using namespace Metal;
using namespace std::chrono_literals;

@interface PipelineFixtureObject : NSObject
@property(nonatomic) id<MTLDevice> device;
@property(nonatomic) NSArray<NSString*>* functionNames;
@end

@implementation PipelineFixtureObject
@end

@interface PipelineFixtureCompiler : NSObject {
@public
    std::mutex mutex;
    std::condition_variable queueChanged;
    std::condition_variable unlocked;
    bool blockRender;
    bool renderWaiting;
    std::atomic_uint renderSync;
    std::atomic_uint renderAsync;
    std::atomic_uint computeSync;
    std::atomic_uint computeAsync;
    std::atomic_uint sourceSync;
    std::atomic_uint sourceAsync;
}
@property(nonatomic, weak) id<MTLDevice> device;
@property(nonatomic) MTL4PipelineDescriptor* lastRender;
@property(nonatomic) MTLDepthStencilDescriptor* lastDepth;
@property(nonatomic) bool failRender;
- (void)waitForAsyncRender;
- (void)unblockRender;
@end

@implementation PipelineFixtureCompiler

- (PipelineFixtureObject*)nativeObject {
    auto output = [PipelineFixtureObject new];
    output.device = self.device;
    return output;
}

- (id<MTLLibrary>)makeLibrary {
    auto output = [self nativeObject];
    output.functionNames = @[
        @"triangle", @"paint", @"object_stage", @"mesh_stage", @"transform_stage", @"expand_stage"
    ];
    return (id<MTLLibrary>)output;
}

- (id<MTLLibrary>)newLibraryWithDescriptor:(MTL4LibraryDescriptor*)descriptor
                                     error:(NSError**)error {
    assert(descriptor && error);
    ++sourceSync;
    return [self makeLibrary];
}

- (id<MTL4CompilerTask>)newLibraryWithDescriptor:(MTL4LibraryDescriptor*)descriptor
                               completionHandler:(MTLNewLibraryCompletionHandler)completion {
    assert(descriptor);
    ++sourceAsync;
    completion([self makeLibrary], nil);
    return (id<MTL4CompilerTask>)[NSObject new];
}

- (id<MTLRenderPipelineState>)render:(MTL4PipelineDescriptor*)descriptor error:(NSError**)error {
    self.lastRender = descriptor;
    if (self.failRender) {
        *error = [NSError errorWithDomain:@"Fixture.Pipeline"
                                     code:82
                                 userInfo:@{NSLocalizedDescriptionKey : @"render rejected"}];
        return nil;
    }
    return (id<MTLRenderPipelineState>)[self nativeObject];
}

- (id<MTLRenderPipelineState>)newRenderPipelineStateWithDescriptor:
                                  (MTL4PipelineDescriptor*)descriptor
                                               compilerTaskOptions:(MTL4CompilerTaskOptions*)options
                                                             error:(NSError**)error {
    assert(options);
    ++renderSync;
    return [self render:descriptor error:error];
}

- (id<MTL4CompilerTask>)newRenderPipelineStateWithDescriptor:(MTL4PipelineDescriptor*)descriptor
                                         compilerTaskOptions:(MTL4CompilerTaskOptions*)options
                                           completionHandler:
                                               (MTLNewRenderPipelineStateCompletionHandler)done {
    assert(options);
    ++renderAsync;
    // Only the test callback is delayed. The SDK submission always returns before
    // its callback.
    dispatch_async(dispatch_get_global_queue(QOS_CLASS_DEFAULT, 0), ^{
      {
          std::unique_lock lock(mutex);
          renderWaiting = true;
          queueChanged.notify_all();
          unlocked.wait(lock, [self] { return !self->blockRender; });
          renderWaiting = false;
      }
      NSError* error = nil;
      auto result = [self render:descriptor error:&error];
      done(result, error);
    });
    return (id<MTL4CompilerTask>)[NSObject new];
}

- (id<MTLComputePipelineState>)
    newComputePipelineStateWithDescriptor:(MTL4ComputePipelineDescriptor*)descriptor
                      compilerTaskOptions:(MTL4CompilerTaskOptions*)options
                                    error:(NSError**)error {
    assert(descriptor.computeFunctionDescriptor && options && error);
    ++computeSync;
    return (id<MTLComputePipelineState>)[self nativeObject];
}

- (id<MTL4CompilerTask>)
    newComputePipelineStateWithDescriptor:(MTL4ComputePipelineDescriptor*)descriptor
                      compilerTaskOptions:(MTL4CompilerTaskOptions*)options
                        completionHandler:(MTLNewComputePipelineStateCompletionHandler)done {
    assert(descriptor.computeFunctionDescriptor && options);
    ++computeAsync;
    done((id<MTLComputePipelineState>)[self nativeObject], nil);
    return (id<MTL4CompilerTask>)[NSObject new];
}

- (void)waitForAsyncRender {
    std::unique_lock lock(mutex);
    assert(queueChanged.wait_for(lock, 10s, [self] { return self->renderWaiting; }));
}

- (void)unblockRender {
    std::lock_guard lock(mutex);
    blockRender = false;
    unlocked.notify_all();
}

@end

@interface PipelineFixtureDevice : NSObject
@property(nonatomic, readonly) PipelineFixtureCompiler* compiler;
@end

@implementation PipelineFixtureDevice

- (instancetype)init {
    if ((self = [super init])) {
        _compiler = [PipelineFixtureCompiler new];
        _compiler.device = (id<MTLDevice>)self;
    }
    return self;
}

- (uint64_t)registryID {
    return 46;
}

- (BOOL)supportsTextureSampleCount:(NSUInteger)samples {
    return samples == 1 || samples == 4;
}

- (id<MTL4Compiler>)newCompilerWithDescriptor:(MTL4CompilerDescriptor*)descriptor
                                        error:(NSError**)error {
    assert(descriptor && error);
    return (id<MTL4Compiler>)self.compiler;
}

- (id<MTLDepthStencilState>)newDepthStencilStateWithDescriptor:
    (MTLDepthStencilDescriptor*)descriptor {
    self.compiler.lastDepth = descriptor;
    auto output = [PipelineFixtureObject new];
    output.device = (id<MTLDevice>)self;
    return (id<MTLDepthStencilState>)output;
}
@end

@interface PipelineFixtureArchive : NSObject
@property(nonatomic) id<MTLDevice> device;
@property(nonatomic) bool hit;
@property(nonatomic) NSUInteger reads;
@end

@implementation PipelineFixtureArchive

- (id<MTLRenderPipelineState>)newRenderPipelineStateWithDescriptor:
                                  (MTL4PipelineDescriptor*)descriptor
                                                             error:(NSError**)error {
    assert(descriptor && !error);
    self.reads += 1;
    if (!self.hit) {
        return nil;
    }
    auto output = [PipelineFixtureObject new];
    output.device = self.device;
    return (id<MTLRenderPipelineState>)output;
}
@end

@interface PipelineFixtureEncoder : NSObject {
@public
    MTLWinding winding;
    MTLCullMode cull;
    MTLViewport viewport;
    MTLScissorRect scissor;
    std::array<float, 4> constants;
    u32 reference;
    unsigned calls;
}
@end

@implementation PipelineFixtureEncoder

- (void)setRenderPipelineState:(id<MTLRenderPipelineState>)state {
    assert(state);
    ++calls;
}

- (void)setDepthStencilState:(id<MTLDepthStencilState>)state {
    assert(state);
    ++calls;
}

- (void)setDepthClipMode:(MTLDepthClipMode)mode {
    assert(mode == MTLDepthClipModeClip);
    ++calls;
}

- (void)setFrontFacingWinding:(MTLWinding)value {
    winding = value;
}

- (void)setCullMode:(MTLCullMode)value {
    cull = value;
}

- (void)setViewport:(MTLViewport)value {
    viewport = value;
}

- (void)setScissorRect:(MTLScissorRect)value {
    scissor = value;
}

- (void)setStencilReferenceValue:(u32)value {
    reference = value;
}

- (void)setBlendColorRed:(float)red green:(float)green blue:(float)blue alpha:(float)alpha {
    constants = {red, green, blue, alpha};
}

@end

namespace {

using Clock = std::chrono::steady_clock;
using F = Pica::FramebufferRegs;
using P = Pica::PipelineRegs;
using R = Pica::RasterizerRegs;

PipelineBuild Resolve(const PipelineCompilation& compilation) {
    auto result = compilation.WaitUntil(Clock::now() + 10s);
    assert(result);
    return std::move(*result);
}

GraphicsPipelineKey MakeKey(const decltype(PipelineFunction::artifact)& artifact,
                            PipelineRoute route) {
    const auto shader = [&](std::string name) {
        return PipelineFunction{artifact, std::move(name)};
    };
    GraphicsPipelineKey key;
    key.route = route;
    if (route == PipelineRoute::Compute) {
        key.shaders.transform = shader("transform_stage");
        return key;
    }
    key.attachments = {MTLPixelFormatRGBA8Unorm, MTLPixelFormatDepth32Float_Stencil8,
                       MTLPixelFormatDepth32Float_Stencil8, 1};
    key.shaders.fragment = shader("paint");
    key.fragment.writes_color = true;
    key.fragment.writes_depth = true;
    if (route == PipelineRoute::Mesh) {
        key.shaders.object = shader("object_stage");
        key.shaders.mesh = shader("mesh_stage");
        key.mesh.payload_bytes = sizeof(u32);
    } else {
        key.shaders.vertex = shader("triangle");
    }
    if (route != PipelineRoute::Vertex) {
        key.shaders.transform = shader("transform_stage");
    }
    if (route == PipelineRoute::ComputeExpansion) {
        key.shaders.expand = shader("expand_stage");
    }
    return key;
}

void VerifyHost() {
    auto device = [PipelineFixtureDevice new];
    auto sdk = device.compiler;
    DeviceContext context((id<MTLDevice>)device);
    DeviceContext foreign((id<MTLDevice>)device);
    ShaderCompiler shaders(context);
    PipelineCompiler compiler(context, shaders, 1, 1);
    const ShaderSource source{"test-only fixture compiler input", "fixture", {}, {}};
    auto key = MakeKey(source, PipelineRoute::Vertex);
    key.vertex_layout.bindings = {{2, 12, 1, MTLVertexStepFunctionPerVertex},
                                  {3, 0, 0, MTLVertexStepFunctionConstant}};
    key.vertex_layout.attributes = {
        {0, 2, 4, MTLVertexFormatFloat2, P::VertexAttributeFormat::FLOAT, 2, false},
        {1, 3, 0, MTLVertexFormatChar4, P::VertexAttributeFormat::BYTE, 4, true},
    };
    key.depth_stencil.depth_test = true;
    key.depth_stencil.depth_write = true;
    key.depth_stencil.depth_compare = F::CompareFunc::LessThan;
    key.depth_stencil.stencil_test = true;
    key.depth_stencil.stencil_compare = F::CompareFunc::GreaterThanOrEqual;
    key.depth_stencil.stencil_pass = F::StencilAction::Replace;
    key.depth_stencil.stencil_fail = F::StencilAction::IncrementWrap;
    key.depth_stencil.stencil_depth_fail = F::StencilAction::Decrement;
    key.depth_stencil.stencil_write_mask = 0xa;
    key.depth_stencil.stencil_read_mask = 0xb5;
    key.blend.write_mask = 5;
    key.blend.enabled = true;
    key.blend.rgb = {F::BlendFactor::SourceAlpha, F::BlendFactor::OneMinusDestAlpha,
                     F::BlendEquation::Max, true};
    key.blend.alpha = {F::BlendFactor::ConstantAlpha, F::BlendFactor::OneMinusSourceAlpha,
                       F::BlendEquation::ReverseSubtract, false};
    key.attachments.samples = 4;
    key.raster.cull_mode = R::CullMode::KeepClockWise;
    key.raster.flip_viewport = true;

    {
        std::lock_guard lock(sdk->mutex);
        sdk->blockRender = true;
    }
    auto initial = compiler.Queue(key);
    [sdk waitForAsyncRender];
    assert(!initial.IsReady());
    auto duplicate = compiler.Queue(key);
    assert(!duplicate.IsReady() && sdk->renderAsync == 1 && sdk->sourceAsync == 1);

    auto variant = key;
    variant.blend.write_mask = 7;
    assert(Resolve(compiler.Queue(variant)).GetStatus(context) == PipelineBuildStatus::QueueFull);
    const auto synchronous = compiler.Build(variant);
    assert(synchronous.GetStatus(context) == PipelineBuildStatus::Success);
    assert(synchronous.GetStatus(foreign) == PipelineBuildStatus::ForeignDevice);
    assert(!synchronous.GetStates(foreign) && sdk->renderSync == 1 && sdk->sourceSync == 1);
    [sdk unblockRender];
    const auto info = Resolve(initial);
    const auto settings = info.GetStates(context);
    assert(settings && settings->render == Resolve(duplicate).GetStates(context)->render);
    const auto render = (MTL4RenderPipelineDescriptor*)sdk.lastRender;
    assert(render.rasterSampleCount == 4 &&
           render.inputPrimitiveTopology == MTLPrimitiveTopologyClassTriangle);
    assert(render.vertexDescriptor.layouts[2].stride == 12);
    assert(render.vertexDescriptor.layouts[3].stepFunction == MTLVertexStepFunctionConstant);
    assert(render.vertexDescriptor.layouts[3].stepRate == 0);
    assert(render.vertexDescriptor.attributes[0].offset == 4);
    assert(render.vertexDescriptor.attributes[1].format == MTLVertexFormatChar4);
    const auto blend = render.colorAttachments[0];
    assert(blend.writeMask == (MTLColorWriteMaskRed | MTLColorWriteMaskBlue));
    assert(blend.blendingState == MTL4BlendStateEnabled);
    assert(blend.rgbBlendOperation == MTLBlendOperationAdd &&
           blend.sourceRGBBlendFactor == MTLBlendFactorOne &&
           blend.destinationRGBBlendFactor == MTLBlendFactorZero);
    assert(blend.alphaBlendOperation == MTLBlendOperationReverseSubtract &&
           blend.sourceAlphaBlendFactor == MTLBlendFactorBlendAlpha &&
           blend.destinationAlphaBlendFactor == MTLBlendFactorOneMinusSourceAlpha);

    const auto depth = sdk.lastDepth;
    assert(depth.depthWriteEnabled && depth.depthCompareFunction == MTLCompareFunctionLess);
    for (MTLStencilDescriptor* face in @[ depth.frontFaceStencil, depth.backFaceStencil ]) {
        assert(face.stencilCompareFunction == MTLCompareFunctionGreaterEqual);
        assert(face.stencilFailureOperation == MTLStencilOperationIncrementWrap);
        assert(face.depthFailureOperation == MTLStencilOperationDecrementClamp);
        assert(face.depthStencilPassOperation == MTLStencilOperationReplace);
        assert(face.readMask == 0xb5 && face.writeMask == 0xa);
    }

    auto calls = [PipelineFixtureEncoder new];
    const PipelineDrawState draw{{2, 3, 10, 11, 0, 1}, {1, 2, 3, 4}, 0x008040ff, 0x91};
    BindGraphicsPipeline((id<MTL4RenderCommandEncoder>)calls, *settings, draw);
    assert(calls->calls == 3 && calls->winding == MTLWindingCounterClockwise &&
           calls->cull == MTLCullModeFront && calls->reference == draw.stencil_reference);
    assert(calls->viewport.width == 10 && calls->scissor.height == 4);
    assert((calls->constants == std::array<float, 4>{1.0f, 64.0f / 255, 128.0f / 255, 0.0f}));
    auto reversed = *settings;
    reversed.raster.cull_mode = R::CullMode::KeepCounterClockWise;
    reversed.raster.flip_viewport = false;
    BindGraphicsPipeline((id<MTL4RenderCommandEncoder>)calls, reversed, draw);
    assert(calls->winding == MTLWindingClockwise && calls->cull == MTLCullModeBack);

    auto archive = [PipelineFixtureArchive new];
    archive.device = (id<MTLDevice>)device;
    const auto prior = sdk->renderSync.load();
    assert(compiler.Build(key, @[ (id<MTL4Archive>)archive ]).GetStates(context) &&
           archive.reads == 1);
    archive.hit = true;
    assert(compiler.Build(key, @[ (id<MTL4Archive>)archive ]).GetStates(context) &&
           archive.reads == 2);
    assert(sdk->renderSync == prior + 1);

    variant = key;
    variant.shaders.fragment->entry = "missing";
    assert(compiler.Build(variant).GetStatus(context) == PipelineBuildStatus::ShaderFailed);
    variant = key;
    variant.attachments.samples = 3;
    assert(compiler.Build(variant).GetStatus(context) == PipelineBuildStatus::InvalidState);
    variant = key;
    variant.raster.primitive = MTLPrimitiveTypeTriangleStrip;
    variant.raster.index_type = PipelineIndexType::UInt16;
    assert(compiler.Build(variant).GetStatus(context) == PipelineBuildStatus::InvalidState);
    variant = key;
    variant.vertex_layout.bindings[1].step_rate = 1;
    assert(compiler.Build(variant).GetStatus(context) == PipelineBuildStatus::InvalidState);
    sdk.failRender = true;
    const auto rejected = compiler.Build(MakeKey(source, PipelineRoute::ComputeExpansion));
    assert(!rejected.GetStates(context) &&
           rejected.GetStatus(context) == PipelineBuildStatus::CompileFailed);
    assert(rejected.GetDiagnostic().domain == "Fixture.Pipeline" &&
           rejected.GetDiagnostic().message == "render rejected");
    assert(sdk->computeSync == 2);
    sdk.failRender = false;

    const auto compute = MakeKey(source, PipelineRoute::Compute);
    const auto compute_states = Resolve(compiler.Queue(compute)).GetStates(context);
    assert(compute_states && !compute_states->render && !compute_states->expand &&
           compute_states->transform && sdk->computeAsync == 1);
    const auto mesh = compiler.Build(MakeKey(source, PipelineRoute::Mesh)).GetStates(context);
    assert(mesh && mesh->render && mesh->transform && !mesh->expand);
    const auto mesh_desc = (MTL4MeshRenderPipelineDescriptor*)sdk.lastRender;
    assert(mesh_desc.objectFunctionDescriptor && mesh_desc.meshFunctionDescriptor &&
           mesh_desc.payloadMemoryLength == sizeof(u32));

    {
        std::lock_guard lock(sdk->mutex);
        sdk->blockRender = true;
    }
    const auto stale = compiler.Queue(key);
    [sdk waitForAsyncRender];
    compiler.InvalidateTitle();
    assert(Resolve(stale).GetStatus(context) == PipelineBuildStatus::Canceled);
    assert(!info.GetStates(context));
    std::promise<void> entered;
    auto idle = std::async(std::launch::async, [&] {
        entered.set_value();
        compiler.WaitForIdle();
    });
    entered.get_future().wait();
    assert(idle.wait_for(20ms) == std::future_status::timeout);
    [sdk unblockRender];
    assert(idle.wait_for(10s) == std::future_status::ready);
    idle.get();
    std::puts("PASS native pipeline worker idle boundary with an in-flight SDK callback");
    PipelineCompiler disabled(context, shaders, 0, 8);
    assert(Resolve(disabled.Queue(key)).GetStatus(context) == PipelineBuildStatus::QueueFull);
    const auto old_shader = disabled.Build(key);
    assert(old_shader.GetStates(context));
    shaders.InvalidateTitle();
    assert(old_shader.GetStatus(context) == PipelineBuildStatus::Canceled);
    const auto old_device = disabled.Build(key);
    assert(old_device.GetStates(context) && context.AdvanceGeneration());
    assert(old_device.GetStatus(context) == PipelineBuildStatus::Canceled);
    std::puts("PASS real descriptors and test-only callbacks for archive "
              "hit/miss, independent sync, "
              "async worker deduplication, bounded work, title/device "
              "cancellation, complete mesh/compute "
              "bundle, typed failures, draw controls and independent RGB/stencil "
              "mapping");
}

constexpr u32 Size = 4;

id<MTLTexture> Texture(id<MTLDevice> device, MTLPixelFormat format, MTLStorageMode storage,
                       u32 samples) {
    auto desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format
                                                                   width:Size
                                                                  height:Size
                                                               mipmapped:NO];
    desc.storageMode = storage;
    desc.hazardTrackingMode = MTLHazardTrackingModeUntracked;
    desc.usage = MTLTextureUsageRenderTarget;
    if (samples > 1) {
        desc.textureType = MTLTextureType2DMultisample;
        desc.sampleCount = samples;
    }
    return CheckResource([device newTextureWithDescriptor:desc], "Create graphics pipeline target");
}

void VerifyRender(Instance& instance, const ShaderSource& source, std::span<const u8> library,
                  PipelineRoute route, u32 samples, bool async) {
    auto device = instance.GetDevice();
    auto& context = instance.GetContext();
    ShaderCompiler shaders(context);
    PipelineCompiler compiler(context, shaders);
    auto key = MakeKey(async ? decltype(PipelineFunction::artifact){source}
                             : decltype(PipelineFunction::artifact){std::vector<u8>{library.begin(),
                                                                                    library.end()}},
                       route);
    key.attachments.samples = samples;
    key.blend.enabled = true;
    key.blend.write_mask = 5;
    key.blend.rgb = {F::BlendFactor::SourceAlpha, F::BlendFactor::OneMinusSourceAlpha,
                     F::BlendEquation::Add, false};
    key.depth_stencil.depth_test = true;
    key.depth_stencil.depth_write = true;
    key.depth_stencil.depth_compare = F::CompareFunc::LessThan;
    key.depth_stencil.stencil_test = true;
    key.depth_stencil.stencil_pass = F::StencilAction::Replace;
    key.depth_stencil.stencil_write_mask = 0x0f;

    const auto build = [&](const GraphicsPipelineKey& request) {
        auto result = async ? Resolve(compiler.Queue(request)) : compiler.Build(request);
        auto states = result.GetStates(context);
        if (!states) {
            std::fprintf(stderr, "Metal pipeline %u %s\n", unsigned(result.GetStatus(context)),
                         result.GetDiagnostic().message.c_str());
        }
        assert(states);
        return *states;
    };
    const auto first = build(key);
    key.blend.enabled = false;
    key.blend.write_mask = 0xa;
    key.depth_stencil.depth_compare = F::CompareFunc::Equal;
    key.depth_stencil.depth_write = false;
    key.depth_stencil.stencil_compare = F::CompareFunc::Equal;
    key.depth_stencil.stencil_pass = F::StencilAction::Keep;
    key.depth_stencil.stencil_write_mask = 0;
    const auto second = build(key);
    const auto target = Texture(device, key.attachments.color, MTLStorageModeShared, 1);
    const auto multisample =
        samples > 1 ? Texture(device, key.attachments.color, MTLStorageModePrivate, samples) : nil;
    const auto depth = Texture(device, key.attachments.depth, MTLStorageModePrivate, samples);

    Scheduler scheduler(context);
    MemoryAllocator allocator(device, MemoryBudget(8 * 1024 * 1024));
    RenderManager pass(instance, scheduler, allocator);
    RenderPass render;
    render.color.target = {multisample ?: target};
    if (multisample) {
        render.color.resolve = {target};
        render.color.store_action = MTLStoreActionStoreAndMultisampleResolve;
    }
    render.color.load_action = MTLLoadActionClear;
    render.depth = {.target = {depth}, .load_action = MTLLoadActionClear};
    render.stencil = render.depth;
    render.area = {0, 0, Size, Size};
    render.width = render.height = Size;
    render.samples = samples;
    render.clear_color = MTLClearColorMake(.25, .5, .75, 1);
    render.clear_stencil = 0x30;
    pass.BeginRendering(render);
    const auto record = [&](const NativePipelineStates& states, u8 reference, u32 scissor) {
        pass.RecordDraw([=](EncodingContext& ctx, id<MTL4RenderCommandEncoder> encoder) {
            BindGraphicsPipeline(encoder, states,
                                 {{0, 0, Size, Size, 0, 1}, {0, 0, scissor, Size}, 0, reference});
            ctx.Retain(states.render);
            ctx.Retain(states.depth_stencil);
            if (route == PipelineRoute::Mesh) {
                [encoder drawMeshThreadgroups:MTLSizeMake(1, 1, 1)
                    threadsPerObjectThreadgroup:MTLSizeMake(1, 1, 1)
                      threadsPerMeshThreadgroup:MTLSizeMake(1, 1, 1)];
            } else {
                [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
            }
        });
    };
    record(first, 0x9a, Size);
    render.color.load_action = MTLLoadActionLoad;
    render.depth.load_action = MTLLoadActionLoad;
    render.stencil.load_action = MTLLoadActionLoad;
    pass.EndRendering();
    pass.BeginRendering(render);
    record(second, 0x3a, Size / 2);
    pass.EndRendering();
    assert(scheduler.Finish({}, Clock::now() + 15s) == CompletionStatus::Complete);
    std::array<std::array<u8, 4>, Size * Size> pixels{};
    [target getBytes:pixels.data()
         bytesPerRow:Size * 4
          fromRegion:MTLRegionMake2D(0, 0, Size, Size)
         mipmapLevel:0];
    for (std::size_t i = 0; i < pixels.size(); ++i) {
        const bool painted = i % Size < Size / 2;
        const std::array<int, 4> expected{128, painted ? 64 : 128, 159, painted ? 128 : 255};
        for (std::size_t component = 0; component < expected.size(); ++component) {
            assert(std::abs(int(pixels[i][component]) - expected[component]) <= 1);
        }
    }
}

void VerifyCompute(Instance& instance, const ShaderSource& source) {
    auto device = instance.GetDevice();
    auto& context = instance.GetContext();
    ShaderCompiler shaders(context);
    PipelineCompiler compiler(context, shaders);
    auto renderless = compiler.Build(MakeKey(source, PipelineRoute::Compute)).GetStates(context);
    auto expanded = Resolve(compiler.Queue(MakeKey(source, PipelineRoute::ComputeExpansion)))
                        .GetStates(context);
    assert(renderless && renderless->transform && !renderless->render && expanded &&
           expanded->render && expanded->transform && expanded->expand);
    auto args = [MTL4ArgumentTableDescriptor new];
    args.maxBufferBindCount = 1;
    NSError* error = nil;
    auto table = [device newArgumentTableWithDescriptor:args error:&error];
    table = CheckResource(table, "Allocate compute expansion fixture argument table", error);
    const u32 initial{};
    auto target = CheckResource([device newBufferWithBytes:&initial
                                                    length:sizeof(initial)
                                                   options:MTLResourceStorageModeShared |
                                                           MTLResourceHazardTrackingModeUntracked],
                                "Allocate compute expansion fixture storage");
    [table setAddress:target.gpuAddress atIndex:0];
    Scheduler scheduler(context);
    for (const auto& pipeline : {renderless->transform, expanded->transform, expanded->expand}) {
        scheduler.Record([=](EncodingContext& ctx) {
            const std::array uses{ResourceUse{target, BufferAccess::ReadWrite, MTLStageDispatch}};
            auto encoder = [ctx.GetCommandBuffer() computeCommandEncoder];
            ctx.Synchronize(encoder, uses);
            [encoder setComputePipelineState:pipeline];
            [encoder setArgumentTable:table];
            [encoder dispatchThreads:MTLSizeMake(1, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
            ctx.Retain(table);
            ctx.Retain(pipeline);
            ctx.EndPass(encoder);
        });
    }
    assert(scheduler.Finish({}, Clock::now() + 15s) == CompletionStatus::Complete);
    assert(*static_cast<const u32*>(target.contents) == 42);
}

} // Anonymous namespace

int main(int argc, const char* argv[]) {
    @autoreleasepool {
        assert(argc == 3);
        VerifyHost();
        auto device = MTLCreateSystemDefaultDevice();
        if (!device || ![device supportsFamily:MTLGPUFamilyMetal4]) {
            std::puts("PENDING live Metal 4 vertex, object/mesh, compute-expanded "
                      "and renderless "
                      "pipeline synchronization with blended color, depth/stencil "
                      "and MSAA readback");
            return 2;
        }
        std::ifstream source_file(argv[1]);
        std::ifstream binary_file(argv[2], std::ios::binary);
        assert(source_file && binary_file);
        const ShaderSource source{std::string{std::istreambuf_iterator<char>{source_file}, {}},
                                  "native pipeline fixture",
                                  {},
                                  {}};
        const std::vector<u8> binary{std::istreambuf_iterator<char>{binary_file}, {}};
        assert(HasMetalLibrarySignature(binary));
        Instance instance(0, false);
        VerifyRender(instance, source, binary, PipelineRoute::Vertex, 1, false);
        VerifyRender(instance, source, binary, PipelineRoute::ComputeExpansion, 1, true);
        if ([device supportsTextureSampleCount:4]) {
            VerifyRender(instance, source, binary, PipelineRoute::Mesh, 4, true);
        } else {
            VerifyRender(instance, source, binary, PipelineRoute::Mesh, 1, true);
            std::puts("PENDING 4x MSAA device support");
        }
        VerifyCompute(instance, source);
        std::puts("PASS native graphics and compute pipeline fixture readback");
        return 0;
    }
}
