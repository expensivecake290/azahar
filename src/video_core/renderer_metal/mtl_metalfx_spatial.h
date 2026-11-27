// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.txt file included.

#pragma once

#import <MetalFX/MTL4FXSpatialScaler.h>

#include <memory>
#include <span>
#include <vector>
#include "video_core/renderer_metal/mtl_texture_runtime.h"

namespace Metal {

enum class SpatialStatus : u8 {
    Eligible,
    Queued,
    Disabled,
    UnsupportedDevice,
    InvalidInput,
    NotUpscaling,
    UnsupportedFormat,
    MissingInputUsage,
    InvalidDependency,
    NativeCreationFailed,
    OutOfMemory,
};

struct SpatialConfiguration {
    VideoCore::Extent input;
    VideoCore::Extent content;
    VideoCore::Extent output;
    MTLPixelFormat input_format;
    MTLPixelFormat output_format;
    MTLFXSpatialScalerColorProcessingMode color_mode;
};

[[nodiscard]] constexpr SpatialStatus CheckSpatialConfiguration(const SpatialConfiguration& config,
                                                                bool enabled,
                                                                bool supported) noexcept {
    using enum SpatialStatus;
    if (!enabled) {
        return Disabled;
    }
    if (!supported) {
        return UnsupportedDevice;
    }
    constexpr u32 max_dimension = 16384;
    const auto valid = [](VideoCore::Extent extent) {
        return extent.width && extent.height && extent.width <= max_dimension &&
               extent.height <= max_dimension;
    };
    if (!valid(config.input) || !valid(config.content) || !valid(config.output) ||
        config.content.width > config.input.width || config.content.height > config.input.height) {
        return InvalidInput;
    }
    if (config.output.width < config.content.width ||
        config.output.height < config.content.height ||
        (config.output.width == config.content.width &&
         config.output.height == config.content.height)) {
        return NotUpscaling;
    }
    const auto format_matches = [&](MTLPixelFormat format) {
        const bool unorm = format == MTLPixelFormatBGRA8Unorm || format == MTLPixelFormatRGBA8Unorm;
        const bool float16 = format == MTLPixelFormatRGBA16Float;
        switch (config.color_mode) {
        case MTLFXSpatialScalerColorProcessingModePerceptual:
            return unorm;
        case MTLFXSpatialScalerColorProcessingModeLinear:
            return unorm || float16;
        case MTLFXSpatialScalerColorProcessingModeHDR:
            return float16;
        }
        return false;
    };
    if (!format_matches(config.input_format) || !format_matches(config.output_format)) {
        return UnsupportedFormat;
    }
    return Eligible;
}

struct SpatialRequest {
    /// The presentation producer provides a single-sample texture. Content starts at its origin.
    TextureResource source;
    VideoCore::Extent content;
    VideoCore::Extent output;
    MTLPixelFormat output_format = MTLPixelFormatRGBA8Unorm;
    MTLFXSpatialScalerColorProcessingMode color_mode =
        MTLFXSpatialScalerColorProcessingModePerceptual;
};

class MetalFXSpatial final {
    struct Allocation;
    struct State;

public:
    class Image {
    public:
        Image() = default;
        [[nodiscard]] explicit operator bool() const noexcept {
            return static_cast<bool>(allocation);
        }
        [[nodiscard]] id<MTLTexture> Texture() const noexcept;
        void Retain(Scheduler& scheduler) const;

    private:
        friend struct State;
        explicit Image(std::shared_ptr<Allocation> allocation);
        std::shared_ptr<Allocation> allocation;
    };

    struct Result {
        SpatialStatus status;
        Image image;
    };

    struct Batch {
        std::vector<Result> results;
        /// Wait on this ticket in the same scheduler before reading any queued output.
        u64 completion{};
    };

    MetalFXSpatial(DeviceContext& context, Scheduler& scheduler, MemoryBudget& budget);
    ~MetalFXSpatial();
    MetalFXSpatial(const MetalFXSpatial&) = delete;
    MetalFXSpatial& operator=(const MetalFXSpatial&) = delete;

    [[nodiscard]] bool IsSupported() const;

    /// Run independent Spatial jobs after an issued producer ticket. Presentation consumes only
    /// queued results and retains each output through its own exact submission.
    [[nodiscard]] Batch Apply(std::span<const SpatialRequest> requests, bool enabled,
                              u64 produced_ticket, const DeviceContext::WorkLease& producer);

private:
    std::unique_ptr<State> state;
};

} // namespace Metal
