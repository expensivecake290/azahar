// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.txt file included.

#include <algorithm>
#include <array>
#include <atomic>
#include <limits>
#include <mutex>
#include <string_view>
#include <type_traits>
#include <utility>
#include "common/thread_worker.h"
#include "video_core/renderer_metal/mtl_graphics_pipeline.h"

namespace Metal {

struct PipelineCompilationEpoch {
    std::atomic<u64> generation{1};
};

namespace {

using Clock = std::chrono::steady_clock;
using FramebufferRegs = Pica::FramebufferRegs;
using ShaderBytecode = decltype(PipelineFunction::artifact);
using FragmentFeedback = Pica::Shader::Generator::MSL::FragmentFeedback;

constexpr auto RenderCompiler = [](id<MTL4Compiler> compiler, MTL4PipelineDescriptor* descriptor,
                                   MTL4CompilerTaskOptions* options, NSError** error) {
    return [compiler newRenderPipelineStateWithDescriptor:descriptor
                                      compilerTaskOptions:options
                                                    error:error];
};
constexpr auto RenderCompilerAsync =
    [](id<MTL4Compiler> compiler, MTL4PipelineDescriptor* descriptor,
       MTL4CompilerTaskOptions* options, MTLNewRenderPipelineStateCompletionHandler completion) {
        return [compiler newRenderPipelineStateWithDescriptor:descriptor
                                          compilerTaskOptions:options
                                            completionHandler:completion];
    };
constexpr auto ComputeCompiler = [](id<MTL4Compiler> compiler,
                                    MTL4ComputePipelineDescriptor* descriptor,
                                    MTL4CompilerTaskOptions* options, NSError** error) {
    return [compiler newComputePipelineStateWithDescriptor:descriptor
                                       compilerTaskOptions:options
                                                     error:error];
};
constexpr auto ComputeCompilerAsync =
    [](id<MTL4Compiler> compiler, MTL4ComputePipelineDescriptor* descriptor,
       MTL4CompilerTaskOptions* options, MTLNewComputePipelineStateCompletionHandler completion) {
        return [compiler newComputePipelineStateWithDescriptor:descriptor
                                           compilerTaskOptions:options
                                             completionHandler:completion];
    };

ShaderDiagnostic DescribeError(NSError* error, std::string_view fallback, bool warning = false) {
    if (!error) {
        return ShaderDiagnostic{std::string(fallback), {}, 0, warning};
    }
    const char* message = error.localizedDescription.UTF8String;
    const char* domain = error.domain.UTF8String;
    return ShaderDiagnostic{message ? std::string(message) : std::string(fallback),
                            domain ? domain : "", error.code, warning};
}

MTLBlendFactor BlendFactor(FramebufferRegs::BlendFactor value) {
    static constexpr std::array factors{
        MTLBlendFactorZero,
        MTLBlendFactorOne,
        MTLBlendFactorSourceColor,
        MTLBlendFactorOneMinusSourceColor,
        MTLBlendFactorDestinationColor,
        MTLBlendFactorOneMinusDestinationColor,
        MTLBlendFactorSourceAlpha,
        MTLBlendFactorOneMinusSourceAlpha,
        MTLBlendFactorDestinationAlpha,
        MTLBlendFactorOneMinusDestinationAlpha,
        MTLBlendFactorBlendColor,
        MTLBlendFactorOneMinusBlendColor,
        MTLBlendFactorBlendAlpha,
        MTLBlendFactorOneMinusBlendAlpha,
        MTLBlendFactorSourceAlphaSaturated,
    };
    const auto index = static_cast<u32>(value);
    return index < factors.size() ? factors[index] : MTLBlendFactorOne;
}

MTLBlendOperation BlendEquation(FramebufferRegs::BlendEquation value) {
    static constexpr std::array equations{
        MTLBlendOperationAdd, MTLBlendOperationSubtract, MTLBlendOperationReverseSubtract,
        MTLBlendOperationMin, MTLBlendOperationMax,
    };
    const auto index = static_cast<u32>(value);
    return index < equations.size() ? equations[index] : MTLBlendOperationAdd;
}

MTLCompareFunction CompareFunction(FramebufferRegs::CompareFunc value) {
    static constexpr std::array functions{
        MTLCompareFunctionNever,    MTLCompareFunctionAlways,       MTLCompareFunctionEqual,
        MTLCompareFunctionNotEqual, MTLCompareFunctionLess,         MTLCompareFunctionLessEqual,
        MTLCompareFunctionGreater,  MTLCompareFunctionGreaterEqual,
    };
    return functions.at(static_cast<u32>(value));
}

MTLStencilOperation StencilOperation(FramebufferRegs::StencilAction value) {
    static constexpr std::array operations{
        MTLStencilOperationKeep,           MTLStencilOperationZero,
        MTLStencilOperationReplace,        MTLStencilOperationIncrementClamp,
        MTLStencilOperationDecrementClamp, MTLStencilOperationInvert,
        MTLStencilOperationIncrementWrap,  MTLStencilOperationDecrementWrap,
    };
    return operations.at(static_cast<u32>(value));
}

std::optional<MTLPrimitiveTopologyClass> Topology(MTLPrimitiveType primitive) {
    switch (primitive) {
    case MTLPrimitiveTypePoint:
        return MTLPrimitiveTopologyClassPoint;
    case MTLPrimitiveTypeLine:
    case MTLPrimitiveTypeLineStrip:
        return MTLPrimitiveTopologyClassLine;
    case MTLPrimitiveTypeTriangle:
    case MTLPrimitiveTypeTriangleStrip:
        return MTLPrimitiveTopologyClassTriangle;
    default:
        return std::nullopt;
    }
}

bool ValidateVertexLayout(const PipelineVertexLayout& layout) {
    if (layout.bindings.size() > layout.MaxBindings ||
        layout.attributes.size() > layout.MaxAttributes) {
        return false;
    }
    std::array<const PipelineVertexBinding*, 31> buffers{};
    u32 previous_slot{};
    bool first = true;
    for (const auto& binding : layout.bindings) {
        const bool valid_step =
            (binding.step == MTLVertexStepFunctionConstant && binding.step_rate == 0) ||
            (binding.step == MTLVertexStepFunctionPerVertex && binding.step_rate == 1) ||
            (binding.step == MTLVertexStepFunctionPerInstance && binding.step_rate > 0);
        if (!valid_step || binding.buffer_index >= buffers.size() ||
            (!first && binding.buffer_index <= previous_slot) ||
            (binding.step != MTLVertexStepFunctionConstant && binding.stride == 0)) {
            return false;
        }
        buffers[binding.buffer_index] = &binding;
        first = false;
        previous_slot = binding.buffer_index;
    }

    first = true;
    for (const auto& attribute : layout.attributes) {
        if (attribute.location >= layout.MaxAttributes ||
            (!first && attribute.location <= previous_slot) ||
            attribute.buffer_index >= buffers.size() || !buffers[attribute.buffer_index] ||
            attribute.format == MTLVertexFormatInvalid || attribute.components == 0 ||
            attribute.components > 4) {
            return false;
        }
        const auto& binding = *buffers[attribute.buffer_index];
        if (binding.step != MTLVertexStepFunctionConstant && attribute.offset >= binding.stride) {
            return false;
        }
        previous_slot = attribute.location;
        first = false;
    }
    return true;
}

std::string_view ValidateKey(const GraphicsPipelineKey& key, id<MTLDevice> device) {
    const auto& stages = key.shaders;
    const auto& layout = key.vertex_layout;
    switch (key.route) {
    case PipelineRoute::Vertex:
        if (!stages.vertex || stages.object || stages.mesh || stages.transform || stages.expand ||
            !ValidateVertexLayout(layout)) {
            return "Invalid vertex pipeline stages or input layout";
        }
        break;
    case PipelineRoute::Mesh:
        if (stages.vertex || !stages.object || !stages.mesh || !stages.transform || stages.expand ||
            key.mesh.object_threads == 0 || key.mesh.mesh_threads == 0 ||
            key.mesh.max_mesh_threadgroups == 0 || key.mesh.payload_bytes == 0 ||
            key.raster.primitive != MTLPrimitiveTypeTriangle) {
            return "Invalid PICA object/mesh pipeline";
        }
        break;
    case PipelineRoute::ComputeExpansion:
        if (!stages.vertex || stages.object || stages.mesh || !stages.transform || !stages.expand) {
            return "Invalid compute expansion pipeline stages";
        }
        break;
    case PipelineRoute::Compute:
        return stages.transform && !stages.vertex && !stages.fragment && !stages.object &&
                       !stages.mesh && !stages.expand && key.attachments == PipelineAttachments{} &&
                       layout.bindings.empty() && layout.attributes.empty()
                   ? ""
                   : "Invalid renderless compute pipeline";
    default:
        return "Unknown graphics pipeline route";
    }

    if (key.route != PipelineRoute::Vertex &&
        (!layout.bindings.empty() || !layout.attributes.empty())) {
        return "Expanded geometry uses shader fetches rather than a native vertex layout";
    }
    if (!Topology(key.raster.primitive) || key.attachments.samples == 0 ||
        ![device supportsTextureSampleCount:key.attachments.samples]) {
        return "Unsupported graphics topology or render sample count";
    }
    using enum Pica::PipelineRegs::TriangleTopology;
    if (key.route == PipelineRoute::Vertex && key.raster.guest_topology == Fan) {
        return "PICA triangle fans require native mesh or compute expansion";
    }
    if (key.raster.primitive == MTLPrimitiveTypeTriangleStrip &&
        (key.raster.index_type != PipelineIndexType::None || key.route != PipelineRoute::Vertex)) {
        return "Indexed guest triangle strips require explicit restart-free expansion";
    }

    const auto& depth = key.depth_stencil;
    const bool valid_depth =
        static_cast<u32>(depth.depth_compare) < 8 && static_cast<u32>(depth.stencil_compare) < 8 &&
        static_cast<u32>(depth.stencil_fail) < 8 && static_cast<u32>(depth.stencil_pass) < 8 &&
        static_cast<u32>(depth.stencil_depth_fail) < 8;
    if (!valid_depth || key.blend.write_mask > 0xf || static_cast<u32>(key.blend.logic_op) > 15 ||
        static_cast<u32>(key.raster.cull_mode) > 3) {
        return "Invalid PICA color, depth, stencil or cull state";
    }
    if (!stages.fragment &&
        (key.fragment.writes_color || key.fragment.writes_depth ||
         key.fragment.feedback != FragmentFeedback::None ||
         key.fragment.shadow != Pica::Shader::Generator::MSL::ShadowStorage::None)) {
        return "Fragment effects require a fragment function";
    }
    if (key.fragment.feedback != FragmentFeedback::None &&
        key.attachments.color == MTLPixelFormatInvalid) {
        return "Fragment feedback requires a native color attachment";
    }
    return "";
}

MTL4LibraryFunctionDescriptor* FunctionDescriptor(id<MTLLibrary> library, std::string_view entry) {
    if (!library || !IsShaderIdentifier(entry)) {
        return nil;
    }
    auto name = [[NSString alloc] initWithBytes:entry.data()
                                         length:entry.size()
                                       encoding:NSUTF8StringEncoding];
    if (![library.functionNames containsObject:name]) {
        return nil;
    }
    auto descriptor = [MTL4LibraryFunctionDescriptor new];
    descriptor.library = library;
    descriptor.name = name;
    return descriptor;
}

MTLVertexDescriptor* VertexDescriptor(const PipelineVertexLayout& layout) {
    if (layout.attributes.empty()) {
        return nil;
    }
    auto descriptor = [MTLVertexDescriptor new];
    for (const auto& binding : layout.bindings) {
        auto native = descriptor.layouts[binding.buffer_index];
        native.stride = binding.stride;
        native.stepFunction = binding.step;
        native.stepRate = binding.step_rate;
    }
    for (const auto& attr : layout.attributes) {
        auto native = descriptor.attributes[attr.location];
        native.bufferIndex = attr.buffer_index;
        native.offset = attr.offset;
        native.format = attr.format;
    }
    return descriptor;
}

void SetColorAttachment(MTL4RenderPipelineColorAttachmentDescriptor* color,
                        const GraphicsPipelineKey& key) {
    const auto& blend = key.blend;
    const PipelineBlendChannel copy;
    const auto& rgb = blend.rgb.shader_emulation ? copy : blend.rgb;
    const auto& alpha = blend.alpha.shader_emulation ? copy : blend.alpha;
    color.pixelFormat = key.attachments.color;
    color.writeMask = key.attachments.color == MTLPixelFormatInvalid ? MTLColorWriteMaskNone
                                                                     : blend.ColorWriteMask();
    color.blendingState = blend.enabled && key.attachments.color != MTLPixelFormatInvalid
                              ? MTL4BlendStateEnabled
                              : MTL4BlendStateDisabled;
    color.sourceRGBBlendFactor = BlendFactor(rgb.source);
    color.destinationRGBBlendFactor = BlendFactor(rgb.destination);
    color.rgbBlendOperation = BlendEquation(rgb.equation);
    color.sourceAlphaBlendFactor = BlendFactor(alpha.source);
    color.destinationAlphaBlendFactor = BlendFactor(alpha.destination);
    color.alphaBlendOperation = BlendEquation(alpha.equation);
}

MTLDepthStencilDescriptor* DepthStencilDescriptor(const GraphicsPipelineKey& key) {
    const auto& depth = key.depth_stencil;
    const bool test_depth = depth.depth_test && key.attachments.depth != MTLPixelFormatInvalid;
    const bool test_stencil =
        depth.stencil_test && key.attachments.stencil != MTLPixelFormatInvalid;
    auto descriptor = [MTLDepthStencilDescriptor new];
    descriptor.depthCompareFunction =
        test_depth ? CompareFunction(depth.depth_compare) : MTLCompareFunctionAlways;
    descriptor.depthWriteEnabled = test_depth && depth.depth_write;
    if (test_stencil) {
        auto stencil = [MTLStencilDescriptor new];
        stencil.stencilCompareFunction = CompareFunction(depth.stencil_compare);
        stencil.stencilFailureOperation = StencilOperation(depth.stencil_fail);
        stencil.depthFailureOperation = StencilOperation(depth.stencil_depth_fail);
        stencil.depthStencilPassOperation = StencilOperation(depth.stencil_pass);
        stencil.readMask = depth.stencil_read_mask;
        stencil.writeMask = depth.stencil_write_mask;
        descriptor.frontFaceStencil = stencil;
        descriptor.backFaceStencil = stencil;
    }
    return descriptor;
}

template <typename State>
struct NativeCompilation {
    State pipeline = nil;
    NSError* error = nil;
};

struct WorkScope {
    const DeviceContext::WorkLease lease;
    const u64 compiler_generation;
    const u64 title_generation;
};

template <typename Descriptor, typename Sync, typename Async>
auto CompileNative(const DeviceContext& context, const DeviceContext::WorkLease& lease,
                   id<MTL4Compiler> compiler, Descriptor descriptor, MTL4CompilerTaskOptions* opts,
                   bool async, Sync sync, Async queue) {
    using State = decltype(sync(compiler, descriptor, opts, nullptr));
    using Result = NativeCompilation<State>;
    auto guard = context.AcquireSubmission(lease);
    if (!guard) {
        return Result{};
    }
    for (id<MTL4Archive> archive in opts.lookupArchives) {
        State found = nil;
        if constexpr (std::is_same_v<Descriptor, MTL4ComputePipelineDescriptor*>) {
            found = [archive newComputePipelineStateWithDescriptor:descriptor error:nil];
        } else {
            found = [archive newRenderPipelineStateWithDescriptor:descriptor error:nil];
        }
        if (found && found.device == context.GetDevice()) {
            return Result{found, nil};
        }
    }
    if (!async) {
        NSError* error = nil;
        State pipeline = sync(compiler, descriptor, opts, &error);
        return Result{pipeline, error};
    }

    auto completion = std::make_shared<std::promise<Result>>();
    auto finished = completion->get_future();
    auto task = queue(compiler, descriptor, opts, ^(State pipeline, NSError* error) {
      completion->set_value(Result{pipeline, error});
    });
    guard.reset();
    if (!task) {
        return Result{};
    }
    return finished.get();
}

} // Anonymous namespace

struct PipelineCompiler::State {
    struct Key {
        u64 compiler_generation;
        u64 title_generation;
        u64 device_generation;
        GraphicsPipelineKey pipeline;
        auto operator<=>(const Key&) const = default;
    };

    struct Job {
        std::promise<PipelineBuild> promise;
        std::shared_future<PipelineBuild> future{promise.get_future().share()};
        bool published{};
    };

    State(DeviceContext& context, ShaderCompiler& shaders, std::size_t worker_count,
          std::size_t max_async_jobs)
        : context(context), shaders(shaders), limit(worker_count ? max_async_jobs : 0),
          worker(std::min(worker_count, limit), "Metaliris pipeline compiler") {}

    WorkScope AcquireWork() const {
        return {context.AcquireWork(), epoch->generation.load(), shaders.TitleGeneration()};
    }

    bool IsCurrent(const WorkScope& scope) const {
        return scope.compiler_generation != 0 && scope.title_generation != 0 &&
               scope.compiler_generation == epoch->generation.load() &&
               scope.title_generation == shaders.TitleGeneration() &&
               context.GetLeaseStatus(scope.lease) == WorkLeaseStatus::Valid;
    }

    static PipelineBuild Failure(PipelineBuildStatus status, std::string_view message = {}) {
        PipelineBuild result;
        result.status = status;
        result.diagnostic.message = message;
        return result;
    }

    static PipelineCompilation Ready(PipelineBuild result) {
        std::promise<PipelineBuild> promise;
        auto future = promise.get_future().share();
        promise.set_value(std::move(result));
        return PipelineCompilation(std::move(future));
    }

    void Invalidate(bool close) {
        std::lock_guard lock(mutex);
        const u64 generation = epoch->generation.load();
        if (generation == 0) {
            return;
        }

        epoch->generation.store(
            close || generation == std::numeric_limits<u64>::max() ? 0 : generation + 1);
        for (const auto& [_, job] : active) {
            if (!job->published) {
                job->promise.set_value(PipelineBuild{});
                job->published = true;
            }
        }
    }

    ShaderLibrary Resolve(const ShaderBytecode& binary, const WorkScope& scope, bool async) const {
        if (const auto* source = std::get_if<ShaderSource>(&binary)) {
            if (async) {
                const auto job = shaders.QueueLibrary(*source);
                while (IsCurrent(scope)) {
                    const auto result = job.WaitUntil(Clock::now() + std::chrono::milliseconds(20));
                    if (!result) {
                        continue;
                    }
                    if (result->GetStatus(context) != ShaderCompileStatus::QueueFull) {
                        return *result;
                    }
                    break;
                }
            }
            return IsCurrent(scope) ? shaders.CompileLibrary(*source) : ShaderLibrary{};
        }
        return shaders.LoadLibrary(std::get<std::vector<u8>>(binary), "");
    }

    PipelineBuild Build(const GraphicsPipelineKey& key, const WorkScope& scope,
                        NSArray<id<MTL4Archive>>* archives, bool async) const {
        @autoreleasepool {
            if (!IsCurrent(scope)) {
                return {};
            }
            auto device = context.GetDevice();
            if (const auto invalid = ValidateKey(key, device); !invalid.empty()) {
                return Failure(PipelineBuildStatus::InvalidState, invalid);
            }

            PipelineBuild result;
            result.states.raster = key.raster;
            result.epoch = epoch;
            result.compiler_generation = scope.compiler_generation;
            result.lease = scope.lease;
            auto native = shaders.GetNativeCompiler(scope.lease);
            if (!native) {
                return {};
            }
            auto options = [MTL4CompilerTaskOptions new];
            options.lookupArchives = archives;

            std::map<ShaderBytecode, ShaderLibrary> libraries;
            auto lookup = [&](const std::optional<PipelineFunction>& shader,
                              MTL4LibraryFunctionDescriptor* __strong& out) {
                if (!shader) {
                    return true;
                }
                if (!IsCurrent(scope)) {
                    result.status = PipelineBuildStatus::Canceled;
                    return false;
                }
                auto [it, inserted] = libraries.try_emplace(shader->artifact);
                if (inserted) {
                    it->second = Resolve(shader->artifact, scope, async);
                }
                const auto& library = it->second;
                switch (library.GetStatus(context)) {
                case ShaderCompileStatus::Success:
                    break;
                case ShaderCompileStatus::Canceled:
                    result.status = PipelineBuildStatus::Canceled;
                    return false;
                case ShaderCompileStatus::ForeignDevice:
                    result.status = PipelineBuildStatus::ForeignDevice;
                    return false;
                default:
                    result.status = PipelineBuildStatus::ShaderFailed;
                    result.diagnostic = library.GetDiagnostic();
                    return false;
                }
                if (!result.library.GetLibrary(context)) {
                    result.library = library;
                }
                if (library.GetDiagnostic().warning && result.diagnostic.message.empty()) {
                    result.diagnostic = library.GetDiagnostic();
                }
                out = FunctionDescriptor(library.GetLibrary(context), shader->entry);
                if (!out) {
                    result.status = PipelineBuildStatus::ShaderFailed;
                    result.diagnostic = DescribeError(nil, "Missing requested Metal shader entry");
                }
                return out != nil;
            };

            MTL4LibraryFunctionDescriptor* vertex = nil;
            MTL4LibraryFunctionDescriptor* fragment = nil;
            MTL4LibraryFunctionDescriptor* object = nil;
            MTL4LibraryFunctionDescriptor* mesh = nil;
            MTL4LibraryFunctionDescriptor* transform = nil;
            MTL4LibraryFunctionDescriptor* expand = nil;
            const bool all_stages =
                lookup(key.shaders.vertex, vertex) && lookup(key.shaders.fragment, fragment) &&
                lookup(key.shaders.object, object) && lookup(key.shaders.mesh, mesh) &&
                lookup(key.shaders.transform, transform) && lookup(key.shaders.expand, expand);
            if (!IsCurrent(scope)) {
                return {};
            }
            if (!all_stages) {
                return result;
            }

            auto make_compute = [&](MTL4LibraryFunctionDescriptor* function,
                                    id<MTLComputePipelineState> __strong& out) {
                if (!function) {
                    return true;
                }
                if (!IsCurrent(scope)) {
                    result.status = PipelineBuildStatus::Canceled;
                    return false;
                }
                auto descriptor = [MTL4ComputePipelineDescriptor new];
                descriptor.computeFunctionDescriptor = function;
                const auto compiled =
                    CompileNative(context, scope.lease, native, descriptor, options, async,
                                  ComputeCompiler, ComputeCompilerAsync);
                return Assign(result, out, compiled, "Metal compute pipeline compilation failed");
            };

            if (!make_compute(transform, result.states.transform) ||
                !make_compute(expand, result.states.expand)) {
                return IsCurrent(scope) ? result : PipelineBuild{};
            }
            if (key.route != PipelineRoute::Compute) {
                if (!IsCurrent(scope)) {
                    return {};
                }

                MTL4PipelineDescriptor* descriptor = nil;
                if (key.route == PipelineRoute::Mesh) {
                    auto mesh_descriptor = [MTL4MeshRenderPipelineDescriptor new];
                    mesh_descriptor.objectFunctionDescriptor = object;
                    mesh_descriptor.meshFunctionDescriptor = mesh;
                    mesh_descriptor.fragmentFunctionDescriptor = fragment;
                    mesh_descriptor.rasterSampleCount = key.attachments.samples;
                    mesh_descriptor.rasterizationEnabled = key.raster.rasterization_enabled;
                    mesh_descriptor.maxTotalThreadsPerObjectThreadgroup = key.mesh.object_threads;
                    mesh_descriptor.maxTotalThreadsPerMeshThreadgroup = key.mesh.mesh_threads;
                    mesh_descriptor.requiredThreadsPerObjectThreadgroup =
                        MTLSizeMake(key.mesh.object_threads, 1, 1);
                    mesh_descriptor.requiredThreadsPerMeshThreadgroup =
                        MTLSizeMake(key.mesh.mesh_threads, 1, 1);
                    mesh_descriptor.maxTotalThreadgroupsPerMeshGrid =
                        key.mesh.max_mesh_threadgroups;
                    mesh_descriptor.payloadMemoryLength = key.mesh.payload_bytes;
                    SetColorAttachment(mesh_descriptor.colorAttachments[0], key);
                    descriptor = mesh_descriptor;
                } else {
                    auto render = [MTL4RenderPipelineDescriptor new];
                    render.vertexFunctionDescriptor = vertex;
                    render.fragmentFunctionDescriptor = fragment;
                    render.rasterSampleCount = key.attachments.samples;
                    render.rasterizationEnabled = key.raster.rasterization_enabled;
                    render.inputPrimitiveTopology = *Topology(key.raster.primitive);
                    render.vertexDescriptor = VertexDescriptor(key.vertex_layout);
                    SetColorAttachment(render.colorAttachments[0], key);
                    descriptor = render;
                }
                const auto compiled =
                    CompileNative(context, scope.lease, native, descriptor, options, async,
                                  RenderCompiler, RenderCompilerAsync);
                if (!Assign(result, result.states.render, compiled,
                            "Metal render pipeline compilation failed")) {
                    return IsCurrent(scope) ? result : PipelineBuild{};
                }

                auto guard = context.AcquireSubmission(scope.lease);
                if (!guard) {
                    return {};
                }
                result.states.depth_stencil =
                    [device newDepthStencilStateWithDescriptor:DepthStencilDescriptor(key)];
                if (!result.states.depth_stencil) {
                    return Failure(PipelineBuildStatus::CompileFailed,
                                   "Metal depth/stencil state creation failed");
                }
                if (result.states.depth_stencil.device != device) {
                    return Failure(PipelineBuildStatus::ForeignDevice,
                                   "Metal depth/stencil state belongs to another device");
                }
            }
            if (!IsCurrent(scope) || !result.library.GetLibrary(context)) {
                return {};
            }
            result.status = PipelineBuildStatus::Success;
            return result;
        }
    }

    template <typename Native>
    bool Assign(PipelineBuild& result, Native& target, const NativeCompilation<Native>& native,
                std::string_view error) const {
        if (!native.pipeline) {
            result.diagnostic = DescribeError(native.error, error);
            result.status = PipelineBuildStatus::CompileFailed;
            return false;
        }
        if (native.pipeline.device != context.GetDevice()) {
            result.diagnostic = DescribeError(nil, "Metal pipeline belongs to another device");
            result.status = PipelineBuildStatus::ForeignDevice;
            return false;
        }
        if (native.error && result.diagnostic.message.empty()) {
            result.diagnostic = DescribeError(native.error, {}, true);
        }
        target = native.pipeline;
        return true;
    }

    DeviceContext& context;
    ShaderCompiler& shaders;
    const std::size_t limit;
    const std::shared_ptr<PipelineCompilationEpoch> epoch =
        std::make_shared<PipelineCompilationEpoch>();
    std::mutex mutex;
    std::map<Key, std::shared_ptr<Job>> active;
    Common::ThreadWorker worker;
};

PipelineBuildStatus PipelineBuild::GetStatus(const DeviceContext& context) const {
    if (!epoch) {
        return status;
    }
    if (epoch->generation.load() != compiler_generation) {
        return PipelineBuildStatus::Canceled;
    }
    switch (context.GetLeaseStatus(lease)) {
    case WorkLeaseStatus::Valid:
        if (status == PipelineBuildStatus::Success && !library.GetLibrary(context)) {
            return PipelineBuildStatus::Canceled;
        }
        return status;
    case WorkLeaseStatus::ForeignContext:
        return PipelineBuildStatus::ForeignDevice;
    default:
        return PipelineBuildStatus::Canceled;
    }
}

std::optional<NativePipelineStates> PipelineBuild::GetStates(const DeviceContext& context) const {
    return GetStatus(context) == PipelineBuildStatus::Success ? std::optional{states}
                                                              : std::nullopt;
}

PipelineCompiler::PipelineCompiler(DeviceContext& context, ShaderCompiler& shaders,
                                   std::size_t worker_count, std::size_t max_async_jobs)
    : state(std::make_unique<State>(context, shaders, worker_count, max_async_jobs)) {}

PipelineCompiler::~PipelineCompiler() {
    Close();
    WaitForIdle();
}

PipelineBuild PipelineCompiler::Build(const GraphicsPipelineKey& key,
                                      NSArray<id<MTL4Archive>>* archives) {
    return state->Build(key, state->AcquireWork(), archives, false);
}

PipelineCompilation PipelineCompiler::Queue(GraphicsPipelineKey key,
                                            NSArray<id<MTL4Archive>>* archives) {
    @autoreleasepool {
        const auto scope = state->AcquireWork();
        std::lock_guard lock(state->mutex);
        if (!state->IsCurrent(scope)) {
            return State::Ready({});
        }
        State::Key request{scope.compiler_generation, scope.title_generation,
                           scope.lease.Generation(), std::move(key)};
        if (const auto it = state->active.find(request); it != state->active.end()) {
            return PipelineCompilation(it->second->future);
        }
        if (state->active.size() >= state->limit) {
            return State::Ready(State::Failure(PipelineBuildStatus::QueueFull,
                                               "Async Metal pipeline compiler is full"));
        }

        auto job = std::make_shared<State::Job>();
        const auto [it, inserted] = state->active.emplace(std::move(request), job);
        try {
            state->worker.QueueWork(
                [owner = state.get(), it, scope, job, archives = [archives copy]] {
                    try {
                        auto result = owner->Build(it->first.pipeline, scope, archives, true);
                        std::lock_guard lock(owner->mutex);
                        if (!job->published) {
                            job->promise.set_value(std::move(result));
                        }
                        owner->active.erase(it);
                    } catch (...) {
                        std::lock_guard lock(owner->mutex);
                        if (!job->published) {
                            job->promise.set_exception(std::current_exception());
                        }
                        owner->active.erase(it);
                    }
                });
        } catch (...) {
            state->active.erase(it);
            throw;
        }
        return PipelineCompilation(job->future);
    }
}

void PipelineCompiler::WaitForIdle() {
    state->worker.WaitForRequests();
}

void PipelineCompiler::InvalidateTitle() {
    state->Invalidate(false);
}

void PipelineCompiler::Close() {
    state->Invalidate(true);
}

void BindGraphicsPipeline(id<MTL4RenderCommandEncoder> encoder, const NativePipelineStates& states,
                          const PipelineDrawState& draw) {
    if (!encoder || !states.render || !states.depth_stencil) {
        throw std::invalid_argument("Binding an incomplete Metal render pipeline");
    }
    using enum Pica::RasterizerRegs::CullMode;
    const bool cull =
        states.raster.cull_mode == KeepClockWise || states.raster.cull_mode == KeepCounterClockWise;
    const MTLWinding winding = states.raster.cull_mode == KeepCounterClockWise
                                   ? MTLWindingClockwise
                                   : MTLWindingCounterClockwise;
    [encoder setRenderPipelineState:states.render];
    [encoder setDepthStencilState:states.depth_stencil];
    [encoder setDepthClipMode:MTLDepthClipModeClip];
    [encoder setFrontFacingWinding:winding];
    [encoder setCullMode:!cull                         ? MTLCullModeNone
                         : states.raster.flip_viewport ? MTLCullModeFront
                                                       : MTLCullModeBack];
    [encoder setViewport:draw.viewport];
    [encoder setScissorRect:draw.scissor];
    [encoder setStencilReferenceValue:draw.stencil_reference];
    const auto component = [&](unsigned shift) {
        return static_cast<float>((draw.blend_color >> shift) & 0xff) / 255.0f;
    };
    [encoder setBlendColorRed:component(0)
                        green:component(8)
                         blue:component(16)
                        alpha:component(24)];
}

} // namespace Metal
