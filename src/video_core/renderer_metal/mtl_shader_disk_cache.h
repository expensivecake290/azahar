// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.txt file included.

#pragma once

#include <array>
#include <atomic>
#include <compare>
#include <cstddef>
#include <filesystem>
#include <memory>
#include <string>
#include <variant>
#include <vector>
#include "video_core/rasterizer_interface.h"
#include "video_core/renderer_metal/mtl_graphics_pipeline.h"

namespace Metal {

struct ShaderCacheIdentity {
    u64 title_id{};
    /// A stable content identifier can identify titles that do not provide a program ID.
    std::string content_id;
    std::string source_version;
    std::string device_architecture;
    std::string system_build;
    std::string profile;

    [[nodiscard]] bool IsValid() const noexcept {
        return (title_id != 0 || !content_id.empty()) && !source_version.empty() &&
               !device_architecture.empty() && !system_build.empty();
    }

    auto operator<=>(const ShaderCacheIdentity&) const = default;
};

/// Include the native shader schema/build and live SDK device and OS compatibility.
[[nodiscard]] ShaderCacheIdentity MakeShaderCacheIdentity(const DeviceContext& context, u64 title,
                                                          std::string content_id,
                                                          std::string profile);

enum class CachedShaderStage : u8 {
    Vertex,
    Fragment,
    Object,
    Mesh,
    ComputeTransform,
    ComputeExpansion,
};

struct CachedShaderFunction {
    CachedShaderStage stage{};
    PipelineFunction function;

    auto operator<=>(const CachedShaderFunction&) const = default;
};

using ShaderCacheRecord = std::variant<CachedShaderFunction, GraphicsPipelineKey>;

enum class ShaderCacheReadStatus : u8 {
    Ready,
    Missing,
    Recovered,
    Incompatible,
    Canceled,
    Disabled,
    IOError,
};

struct ShaderCacheContents {
    ShaderCacheReadStatus status = ShaderCacheReadStatus::Missing;
    std::vector<ShaderCacheRecord> records;
    std::size_t rejected_records{};
};

struct ShaderCacheArchives {
    ShaderCacheReadStatus status = ShaderCacheReadStatus::Missing;
    NSArray<id<MTL4Archive>>* archives = nil;
    std::size_t rejected_archives{};
};

enum class ShaderCacheWriteStatus : u8 {
    Saved,
    Duplicate,
    Removed,
    Missing,
    InvalidRecord,
    Disabled,
    IOError,
};

/// Disk records are independently validated transactions scoped to one immutable title identity.
/// Callers can dispatch filesystem operations to a worker, and may publish only a completed load
/// for the current title. Persistent records contain native shader recipes, native executable
/// Metal libraries and pipeline keys. Device-specific binary archives share the same identity.
class ShaderDiskCache {
public:
    static constexpr std::array<u8, 8> Magic{'M', 'T', 'L', 'I', 'R', 'I', 'S', 'C'};
    static constexpr u32 FormatVersion = 1;
    static constexpr std::size_t MaxRecordSize = 32 * 1024 * 1024;

    ShaderDiskCache(std::filesystem::path directory, ShaderCacheIdentity identity, bool enabled);
    ~ShaderDiskCache();

    ShaderDiskCache(const ShaderDiskCache&) = delete;
    ShaderDiskCache& operator=(const ShaderDiskCache&) = delete;

    [[nodiscard]] const ShaderCacheIdentity& Identity() const noexcept;
    [[nodiscard]] bool IsEnabled() const noexcept;

    /// Counts disk records inspected. Native compiler work has a separate Build callback.
    [[nodiscard]] ShaderCacheContents Load(
        const std::atomic_bool& stop,
        const VideoCore::DiskResourceLoadCallback& callback = {}) const;

    /// The caller supplies a cacheable recipe after verifying its stage or pipeline compilation.
    [[nodiscard]] ShaderCacheWriteStatus Store(const ShaderCacheRecord& record);

    /// Discard only the exact record that failed native rehydration.
    [[nodiscard]] ShaderCacheWriteStatus Remove(const ShaderCacheRecord& record);

    [[nodiscard]] ShaderCacheArchives LoadArchives(const DeviceContext& context,
                                                   const std::atomic_bool& stop) const;

    /// The serializer is device- and title-scoped by its caller. Only completed native pipeline
    /// builds may be captured and the serializer must be finished with that generation.
    [[nodiscard]] ShaderCacheWriteStatus CommitArchive(
        const DeviceContext& context, id<MTL4PipelineDataSetSerializer> serializer);

private:
    struct State;
    std::unique_ptr<State> state;
};

} // namespace Metal
