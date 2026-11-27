// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.txt file included.

#pragma once

#include <chrono>
#include <compare>
#include <future>
#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>
#include "video_core/pica/regs_framebuffer.h"
#include "video_core/pica/regs_pipeline.h"
#include "video_core/pica/regs_rasterizer.h"
#include "video_core/renderer_metal/mtl_shader_util.h"
#include "video_core/renderer_metal/shader/msl_fs_shader_gen.h"

namespace Metal {

enum class PipelineRoute : u8 {
    Vertex,
    Mesh,
    ComputeExpansion,
    Compute,
};

enum class PipelineIndexType : u8 {
    None,
    UInt16,
    UInt32,
};

struct PipelineRasterState {
    Pica::PipelineRegs::TriangleTopology guest_topology{Pica::PipelineRegs::TriangleTopology::List};
    Pica::RasterizerRegs::CullMode cull_mode{Pica::RasterizerRegs::CullMode::KeepAll};
    MTLPrimitiveType primitive{MTLPrimitiveTypeTriangle};
    PipelineIndexType index_type{PipelineIndexType::None};
    bool flip_viewport{};
    bool rasterization_enabled{true};

    auto operator<=>(const PipelineRasterState&) const = default;
};

struct PipelineBlendChannel {
    Pica::FramebufferRegs::BlendFactor source{Pica::FramebufferRegs::BlendFactor::One};
    Pica::FramebufferRegs::BlendFactor destination{Pica::FramebufferRegs::BlendFactor::Zero};
    Pica::FramebufferRegs::BlendEquation equation{Pica::FramebufferRegs::BlendEquation::Add};
    bool shader_emulation{};

    auto operator<=>(const PipelineBlendChannel&) const = default;
};

struct PipelineBlendState {
    PipelineBlendChannel rgb;
    PipelineBlendChannel alpha;
    Pica::FramebufferRegs::LogicOp logic_op{Pica::FramebufferRegs::LogicOp::Copy};
    /// Guest RGBA bits, starting from red in bit zero.
    u8 write_mask{0xf};
    bool enabled{};

    [[nodiscard]] constexpr MTLColorWriteMask ColorWriteMask() const noexcept {
        if (!enabled && logic_op == Pica::FramebufferRegs::LogicOp::NoOp) {
            return MTLColorWriteMaskNone;
        }
        return static_cast<MTLColorWriteMask>(((write_mask & 1) ? MTLColorWriteMaskRed : 0) |
                                              ((write_mask & 2) ? MTLColorWriteMaskGreen : 0) |
                                              ((write_mask & 4) ? MTLColorWriteMaskBlue : 0) |
                                              ((write_mask & 8) ? MTLColorWriteMaskAlpha : 0));
    }

    auto operator<=>(const PipelineBlendState&) const = default;
};

struct PipelineDepthStencilState {
    Pica::FramebufferRegs::CompareFunc depth_compare{Pica::FramebufferRegs::CompareFunc::Always};
    Pica::FramebufferRegs::CompareFunc stencil_compare{Pica::FramebufferRegs::CompareFunc::Always};
    Pica::FramebufferRegs::StencilAction stencil_fail{Pica::FramebufferRegs::StencilAction::Keep};
    Pica::FramebufferRegs::StencilAction stencil_pass{Pica::FramebufferRegs::StencilAction::Keep};
    Pica::FramebufferRegs::StencilAction stencil_depth_fail{
        Pica::FramebufferRegs::StencilAction::Keep};
    u8 stencil_read_mask{0xff};
    u8 stencil_write_mask{0xff};
    bool depth_test{};
    bool depth_write{};
    bool stencil_test{};

    auto operator<=>(const PipelineDepthStencilState&) const = default;
};

struct PipelineVertexBinding {
    u32 buffer_index{};
    u32 stride{};
    u32 step_rate{1};
    MTLVertexStepFunction step{MTLVertexStepFunctionPerVertex};

    auto operator<=>(const PipelineVertexBinding&) const = default;
};

struct PipelineVertexAttribute {
    u32 location{};
    u32 buffer_index{};
    u32 offset{};
    MTLVertexFormat format{MTLVertexFormatInvalid};
    Pica::PipelineRegs::VertexAttributeFormat guest_format{};
    u8 components{};
    bool convert_integer_to_float{};

    auto operator<=>(const PipelineVertexAttribute&) const = default;
};

struct PipelineVertexLayout {
    static constexpr u32 MaxAttributes = 16;
    static constexpr u32 MaxBindings = 13;
    /// Active records in ascending slot order. A fixed attribute uses
    /// MTLVertexStepFunctionConstant.
    std::vector<PipelineVertexBinding> bindings;
    std::vector<PipelineVertexAttribute> attributes;

    auto operator<=>(const PipelineVertexLayout&) const = default;
};

struct PipelineAttachments {
    MTLPixelFormat color{MTLPixelFormatInvalid};
    MTLPixelFormat depth{MTLPixelFormatInvalid};
    MTLPixelFormat stencil{MTLPixelFormatInvalid};
    u32 samples{1};

    auto operator<=>(const PipelineAttachments&) const = default;
};

struct PipelineFragmentState {
    Pica::Shader::Generator::MSL::FragmentFeedback feedback{};
    Pica::Shader::Generator::MSL::ShadowStorage shadow{};
    bool barycentrics{};
    bool writes_color{};
    bool writes_depth{};

    auto operator<=>(const PipelineFragmentState&) const = default;
};

struct PipelineFunction {
    /// Full direct MSL source and options or byte-exact executable Metal library.
    std::variant<ShaderSource, std::vector<u8>> artifact;
    std::string entry;

    auto operator<=>(const PipelineFunction&) const = default;
};

struct PipelineShaderStages {
    std::optional<PipelineFunction> vertex;
    std::optional<PipelineFunction> fragment;
    std::optional<PipelineFunction> object;
    std::optional<PipelineFunction> mesh;
    std::optional<PipelineFunction> transform;
    std::optional<PipelineFunction> expand;

    auto operator<=>(const PipelineShaderStages&) const = default;
};

struct PipelineMeshState {
    u32 object_threads{1};
    u32 mesh_threads{1};
    u32 max_mesh_threadgroups{1};
    u32 payload_bytes{};

    auto operator<=>(const PipelineMeshState&) const = default;
};

struct GraphicsPipelineKey {
    static constexpr u32 Version = 1;

    PipelineRoute route{PipelineRoute::Vertex};
    PipelineShaderStages shaders;
    PipelineVertexLayout vertex_layout;
    PipelineAttachments attachments;
    PipelineRasterState raster;
    PipelineBlendState blend;
    PipelineDepthStencilState depth_stencil;
    PipelineFragmentState fragment;
    PipelineMeshState mesh;

    [[nodiscard]] constexpr bool WritesDepthOrStencil() const noexcept {
        const bool depth = attachments.depth != MTLPixelFormatInvalid && depth_stencil.depth_test &&
                           depth_stencil.depth_write;
        const bool stencil = attachments.stencil != MTLPixelFormatInvalid &&
                             depth_stencil.stencil_test && depth_stencil.stencil_write_mask != 0;
        return depth || stencil;
    }

    /// Full structural comparison. Never use a persistent digest by itself to resolve a hit.
    auto operator<=>(const GraphicsPipelineKey&) const = default;
};

/// The caller maps guest coordinates into the target pass and supplies a clipped Metal scissor.
struct PipelineDrawState {
    MTLViewport viewport{};
    MTLScissorRect scissor{};
    u32 blend_color{};
    u8 stencil_reference{};
};

enum class PipelineBuildStatus : u8 {
    Success,
    InvalidState,
    ShaderFailed,
    CompileFailed,
    QueueFull,
    Canceled,
    ForeignDevice,
};

struct NativePipelineStates {
    id<MTLRenderPipelineState> render = nil;
    id<MTLDepthStencilState> depth_stencil = nil;
    id<MTLComputePipelineState> transform = nil;
    id<MTLComputePipelineState> expand = nil;
    PipelineRasterState raster;
};

struct PipelineCompilationEpoch;

class PipelineBuild {
public:
    [[nodiscard]] PipelineBuildStatus GetStatus(const DeviceContext& context) const;

    [[nodiscard]] std::optional<NativePipelineStates> GetStates(const DeviceContext& context) const;

    [[nodiscard]] const ShaderDiagnostic& GetDiagnostic() const noexcept {
        return diagnostic;
    }

private:
    friend class PipelineCompiler;
    PipelineBuildStatus status{PipelineBuildStatus::Canceled};
    ShaderDiagnostic diagnostic;
    DeviceContext::WorkLease lease;
    std::shared_ptr<const PipelineCompilationEpoch> epoch;
    u64 compiler_generation{};
    ShaderLibrary library;
    NativePipelineStates states;
};

class PipelineCompilation {
public:
    [[nodiscard]] bool IsReady() const {
        return future.valid() &&
               future.wait_for(std::chrono::seconds::zero()) == std::future_status::ready;
    }

    [[nodiscard]] std::optional<PipelineBuild> WaitUntil(
        std::chrono::steady_clock::time_point deadline) const {
        if (!future.valid() || future.wait_until(deadline) != std::future_status::ready) {
            return std::nullopt;
        }
        return future.get();
    }

private:
    friend class PipelineCompiler;
    explicit PipelineCompilation(std::shared_future<PipelineBuild> future)
        : future(std::move(future)) {}

    std::shared_future<PipelineBuild> future;
};

/// The context and shader compiler must outlive this builder. Destruction cancels publication and
/// joins its native work before either dependency can be destroyed.
class PipelineCompiler {
public:
    PipelineCompiler(DeviceContext& context, ShaderCompiler& shader_compiler,
                     std::size_t worker_count = 2, std::size_t max_async_jobs = 16);
    ~PipelineCompiler();

    PipelineCompiler(const PipelineCompiler&) = delete;
    PipelineCompiler& operator=(const PipelineCompiler&) = delete;

    [[nodiscard]] PipelineBuild Build(const GraphicsPipelineKey& key,
                                      NSArray<id<MTL4Archive>>* archives = nil);

    [[nodiscard]] PipelineCompilation Queue(GraphicsPipelineKey key,
                                            NSArray<id<MTL4Archive>>* archives = nil);

    /// Quiesce producers before waiting for every queued native build to retire.
    void WaitForIdle();
    void InvalidateTitle();
    void Close();

private:
    struct State;
    std::unique_ptr<State> state;
};

/// Apply a complete, context-validated render bundle and all per-draw state to an active pass.
/// Retain the bundle's native objects through completion using the scheduler's encoding context.
void BindGraphicsPipeline(id<MTL4RenderCommandEncoder> encoder, const NativePipelineStates& states,
                          const PipelineDrawState& draw);

} // namespace Metal
