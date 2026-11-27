// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.txt file included.

#pragma once

#include <atomic>
#include <cstddef>
#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include "video_core/renderer_metal/mtl_descriptor_update_queue.h"
#include "video_core/renderer_metal/mtl_shader_disk_cache.h"
#include "video_core/renderer_metal/shader/msl_shader_gen.h"

namespace Metal {

struct PipelineDeviceProfile {
    bool raster_order_groups{};
    bool fragment_barycentrics{};
    bool mesh_shaders{};
    bool texture_atomics{};

    [[nodiscard]] static PipelineDeviceProfile Detect(const DeviceContext& context);

    bool operator==(const PipelineDeviceProfile&) const = default;
};

struct PipelineCacheOptions {
    std::filesystem::path directory;
    bool disk_cache{};
    bool async_shaders{};
    std::size_t shader_workers{4};
    std::size_t pipeline_workers{2};
    std::size_t max_pipeline_jobs{16};
};

struct PipelineTitle {
    u64 program_id{};
    std::string content_id;

    [[nodiscard]] bool IsIdentified() const noexcept {
        return program_id != 0 || !content_id.empty();
    }

    [[nodiscard]] constexpr bool IsApplet() const noexcept {
        return (program_id >> 32) == 0x00040030;
    }

    bool operator==(const PipelineTitle&) const = default;
};

struct PipelineWarmup {
    ShaderCacheReadStatus disk_status{ShaderCacheReadStatus::Missing};
    std::size_t completed{};
    std::size_t failed{};
    bool previous_load_interrupted{};
    bool canceled{};
};

struct PipelineShaderRequest {
    const Pica::RegsInternal& regs;
    const Pica::Shader::UserConfig& user;
    GraphicsPipelineKey fixed_state;
    VideoCore::PixelFormat guest_color{VideoCore::PixelFormat::Invalid};

    /// Null means a batch of already interpreted HardwareVertex triangles.
    Pica::ShaderSetup* vertex_setup{};
    const Pica::ShaderSetup* geometry_setup{};
    std::optional<Pica::Shader::Generator::MSL::ProgrammableGeometryConfig> geometry_config;
    u32 vertex_count{};
    /// Set when the front end proves inter-primitive ordering cannot affect any observable output.
    bool order_independent{};
    /// Packed PICA input cannot always be bound as native stage-in at its original alignment.
    bool force_vertex_expansion{};
};

[[nodiscard]] constexpr Pica::Shader::Generator::MSL::GeometryPath SelectPipelineGeometry(
    const PipelineRasterState& raster, const PipelineDeviceProfile& device, u32 vertex_count,
    bool programmable_geometry, bool lighting, bool serial_fragment_feedback,
    bool order_independent, bool force_vertex_expansion) noexcept {
    using namespace Pica::Shader::Generator::MSL;
    if (force_vertex_expansion ||
        (raster.guest_topology == Pica::PipelineRegs::TriangleTopology::Strip &&
         raster.index_type != PipelineIndexType::None)) {
        // Native indexed strips reserve their largest index for restart. PICA accepts it as data.
        return GeometryPath::Compute;
    }
    const bool mesh_order_equivalent =
        order_independent || PrimitiveCount(raster.guest_topology, vertex_count) <= 1;
    return SelectGeometryPath(
        raster.guest_topology,
        {device.mesh_shaders, mesh_order_equivalent, device.fragment_barycentrics},
        programmable_geometry, lighting, serial_fragment_feedback);
}

enum class PipelineSelectionStatus : u8 {
    Ready,
    InvalidVertexProgram,
    InvalidGeometryProgram,
    InvalidFragmentState,
};

struct PipelineSelection {
    PipelineSelectionStatus status{PipelineSelectionStatus::InvalidVertexProgram};
    GraphicsPipelineKey key;
    Pica::Shader::Generator::MSL::FragmentRequirements fragment;
    bool cacheable{};
    std::string diagnostic;

    [[nodiscard]] explicit operator bool() const noexcept {
        return status == PipelineSelectionStatus::Ready;
    }
};

struct PipelineArguments {
    MTLRenderStages stages;
    DescriptorUpdateQueue::Snapshot snapshot;
};

/// Cache selection and compilation retain a complete device and title scope. Callers commit or
/// drain the previous title's render work before switching; pending builds observe cancellation.
class PipelineCache {
public:
    PipelineCache(DeviceContext& context, PipelineDeviceProfile profile,
                  PipelineCacheOptions options);
    ~PipelineCache();
    PipelineCache(const PipelineCache&) = delete;
    PipelineCache& operator=(const PipelineCache&) = delete;

    [[nodiscard]] PipelineSelection Select(const PipelineShaderRequest& request) const;

    /// Return the correct ready pipeline, share an existing in-flight request, or return a typed
    /// failure. A full or disabled asynchronous queue takes the independent synchronous path.
    [[nodiscard]] PipelineBuild Require(const GraphicsPipelineKey& key, bool cacheable,
                                        const std::atomic_bool* stop = nullptr);

    [[nodiscard]] PipelineWarmup SwitchTitle(
        PipelineTitle title, const std::atomic_bool& stop,
        const VideoCore::DiskResourceLoadCallback& callback = {});

    [[nodiscard]] PipelineTitle CurrentTitle() const;
    [[nodiscard]] bool PreviousLoadInterrupted(const PipelineTitle& title) const;

    /// Retire requests and persist completed native captures for the active title.
    [[nodiscard]] ShaderCacheWriteStatus Save();
    void Close();

    /// Use immutable per-stage bind snapshots and retain native states through GPU completion.
    static void Bind(EncodingContext& encoding, const DeviceContext& context,
                     id<MTL4RenderCommandEncoder> encoder, const PipelineBuild& pipeline,
                     const PipelineDrawState& draw, std::span<const PipelineArguments> arguments);

private:
    struct State;
    std::unique_ptr<State> state;
};

} // namespace Metal
