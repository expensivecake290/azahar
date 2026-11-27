// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.txt file included.

#include <algorithm>
#include <stdexcept>
#include <utility>
#include "video_core/renderer_metal/mtl_metalfx_spatial.h"
#include "video_core/renderer_metal/mtl_shader_util.h"

namespace Metal {
namespace {

constexpr std::size_t CachedScalers = 8;

bool SameRecipe(const SpatialConfiguration& a, const SpatialConfiguration& b) {
    return a.input.width == b.input.width && a.input.height == b.input.height &&
           a.output.width == b.output.width && a.output.height == b.output.height &&
           a.input_format == b.input_format && a.output_format == b.output_format &&
           a.color_mode == b.color_mode;
}

} // Anonymous namespace

struct MetalFXSpatial::Allocation {
    id<MTLTexture> texture;
    MemoryBudget::Reservation charge;
    DeviceContext::WorkLease device;
};

struct MetalFXSpatial::State {
    struct Scaler {
        SpatialConfiguration recipe;
        id<MTL4FXSpatialScaler> effect;
        u64 last_use{};
    };

    struct Work {
        TextureResource source;
        std::shared_ptr<Allocation> destination;
        id<MTL4FXSpatialScaler> scaler;
        VideoCore::Extent content;
    };

    State(DeviceContext& context, Scheduler& scheduler, MemoryBudget& budget)
        : context(context), scheduler(scheduler), budget(budget), compiler(context) {
        scalers.reserve(CachedScalers);
    }

    bool SupportsSpatial() const {
        const auto device = context.GetDevice();
        return device && [device supportsFamily:MTLGPUFamilyMetal4] &&
               [MTLFXSpatialScalerDescriptor supportsMetal4FX:device];
    }

    id<MTL4FXSpatialScaler> CreateScaler(const SpatialConfiguration& recipe,
                                         const DeviceContext::WorkLease& lease) {
        auto descriptor = [MTLFXSpatialScalerDescriptor new];
        descriptor.inputWidth = recipe.input.width;
        descriptor.inputHeight = recipe.input.height;
        descriptor.outputWidth = recipe.output.width;
        descriptor.outputHeight = recipe.output.height;
        descriptor.colorTextureFormat = recipe.input_format;
        descriptor.outputTextureFormat = recipe.output_format;
        descriptor.colorProcessingMode = recipe.color_mode;
        const auto native = compiler.GetNativeCompiler(lease);
        auto guard = context.AcquireSubmission(lease);
        if (!native || !guard) {
            return nil;
        }
        return [descriptor newSpatialScalerWithDevice:context.GetDevice() compiler:native];
    }

    id<MTL4FXSpatialScaler> AcquireScaler(const SpatialConfiguration& recipe,
                                          const DeviceContext::WorkLease& lease) {
        const auto retired_match = std::ranges::find_if(scalers, [&](const Scaler& entry) {
            return scheduler.IsFree(entry.last_use) && SameRecipe(entry.recipe, recipe);
        });
        if (retired_match != scalers.end()) {
            return retired_match->effect;
        }
        const auto scaler = CreateScaler(recipe, lease);
        if (!scaler) {
            return nil;
        }
        if (scalers.size() < CachedScalers) {
            scalers.push_back({recipe, scaler});
        } else {
            const auto available = std::ranges::find_if(scalers, [&](const Scaler& candidate) {
                return scheduler.IsFree(candidate.last_use);
            });
            if (available != scalers.end()) {
                *available = {recipe, scaler};
            }
        }
        return scaler;
    }

    void NoteUse(id<MTL4FXSpatialScaler> effect, u64 ticket) {
        const auto slot = std::ranges::find(scalers, effect, &Scaler::effect);
        if (slot != scalers.end()) {
            slot->last_use = ticket;
        }
    }

    std::shared_ptr<Allocation> Allocate(const SpatialConfiguration& recipe,
                                         id<MTL4FXSpatialScaler> effect,
                                         const DeviceContext::WorkLease& lease) {
        auto descriptor = [MTLTextureDescriptor new];
        descriptor.pixelFormat = recipe.output_format;
        descriptor.width = recipe.output.width;
        descriptor.height = recipe.output.height;
        descriptor.textureType = MTLTextureType2D;
        descriptor.storageMode = MTLStorageModePrivate;
        descriptor.hazardTrackingMode = MTLHazardTrackingModeUntracked;
        descriptor.usage = effect.outputTextureUsage | MTLTextureUsageShaderRead;
        auto device = context.GetDevice();
        const auto footprint = [device heapTextureSizeAndAlignWithDescriptor:descriptor];
        auto charge = budget.TryReserve(MemoryPool::Texture, footprint.size, footprint.size);
        if (!charge) {
            return {};
        }
        auto texture = [device newTextureWithDescriptor:descriptor];
        if (!texture || !charge->AdjustPhysical(texture.allocatedSize)) {
            return {};
        }
        texture.label = @"Metaliris Spatial presentation output";
        return std::make_shared<Allocation>(Allocation{texture, std::move(*charge), lease});
    }

    Batch Apply(std::span<const SpatialRequest> requests, bool enabled, u64 produced_ticket,
                const DeviceContext::WorkLease& producer) {
        using enum SpatialStatus;
        Batch batch;
        if (requests.empty()) {
            return batch;
        }
        const auto bypass = [&](SpatialStatus reason) {
            return Batch{std::vector<Result>(requests.size(), Result{reason, {}}), 0};
        };
        if (!enabled) {
            return bypass(Disabled);
        }
        if (!SupportsSpatial()) {
            return bypass(UnsupportedDevice);
        }
        if (!produced_ticket ||
            produced_ticket > scheduler.GetMasterSemaphore().Timeline().LastIssuedTick() ||
            context.GetLeaseStatus(producer) != WorkLeaseStatus::Valid) {
            return bypass(InvalidDependency);
        }
        if (generation != producer.Generation()) {
            scalers.clear();
            generation = producer.Generation();
        }

        const auto pending_tick = scheduler.CurrentTick();
        auto device = context.GetDevice();
        std::vector<Work> work;
        work.reserve(requests.size());
        batch.results.reserve(requests.size());
        for (const auto& request : requests) {
            const auto source = request.source.Get();
            if (!source || source.device != device || source.textureType != MTLTextureType2D ||
                source.sampleCount != 1 || source.framebufferOnly ||
                source.storageMode == MTLStorageModeMemoryless) {
                batch.results.push_back({InvalidInput, {}});
                continue;
            }
            const SpatialConfiguration recipe{
                {static_cast<u32>(source.width), static_cast<u32>(source.height)},
                request.content,
                request.output,
                source.pixelFormat,
                request.output_format,
                request.color_mode,
            };
            const auto status = CheckSpatialConfiguration(recipe, true, true);
            if (status != Eligible) {
                batch.results.push_back({status, {}});
                continue;
            }
            auto effect = AcquireScaler(recipe, producer);
            if (!effect) {
                batch.results.push_back({NativeCreationFailed, {}});
                continue;
            }
            if ((source.usage & effect.colorTextureUsage) != effect.colorTextureUsage) {
                batch.results.push_back({MissingInputUsage, {}});
                continue;
            }
            auto output = Allocate(recipe, effect, producer);
            if (!output) {
                batch.results.push_back({OutOfMemory, {}});
                continue;
            }
            NoteUse(effect, pending_tick);
            batch.results.push_back({Queued, Image(output)});
            work.push_back({request.source, std::move(output), effect, request.content});
        }
        if (work.empty()) {
            return batch;
        }

        scheduler.Record([work = std::move(work), producer](EncodingContext& encoding) {
            if (!encoding.OwnsWork(producer)) {
                throw std::runtime_error(
                    "MetalFX Spatial work belongs to a retired device generation");
            }
            for (const auto& job : work) {
                auto source = job.source.Get();
                auto destination = job.destination->texture;
                encoding.MakeResident(source);
                encoding.MakeResident(destination);
                job.source.Retain(encoding);
                encoding.Retain(job.destination);
                encoding.Retain(job.scaler);
                job.scaler.colorTexture = source;
                job.scaler.outputTexture = destination;
                job.scaler.inputContentWidth = job.content.width;
                job.scaler.inputContentHeight = job.content.height;
                [job.scaler encodeToCommandBuffer:encoding.GetCommandBuffer()];
                job.scaler.colorTexture = nil;
                job.scaler.outputTexture = nil;
            }
        });
        // MetalFX owns the encoders in this packet. The exact source and consumer tickets
        // order those opaque stages without forcing unrelated resources into a barrier.
        batch.completion = scheduler.Flush({.wait_for_tick = produced_ticket});
        return batch;
    }

    DeviceContext& context;
    Scheduler& scheduler;
    MemoryBudget budget;
    ShaderCompiler compiler;
    std::vector<Scaler> scalers;
    u64 generation{};
};

MetalFXSpatial::Image::Image(std::shared_ptr<Allocation> allocation)
    : allocation(std::move(allocation)) {}

id<MTLTexture> MetalFXSpatial::Image::Texture() const noexcept {
    return allocation ? allocation->texture : nil;
}

void MetalFXSpatial::Image::Retain(Scheduler& scheduler) const {
    if (!allocation) {
        throw std::invalid_argument("Cannot retain an empty MetalFX Spatial output");
    }
    scheduler.Record([output = allocation](EncodingContext& encoding) {
        if (!encoding.OwnsWork(output->device)) {
            throw std::invalid_argument("MetalFX Spatial image belongs to a retired device");
        }
        encoding.Retain(output);
    });
}

MetalFXSpatial::MetalFXSpatial(DeviceContext& context, Scheduler& scheduler, MemoryBudget& budget)
    : state(std::make_unique<State>(context, scheduler, budget)) {}

MetalFXSpatial::~MetalFXSpatial() = default;

bool MetalFXSpatial::IsSupported() const {
    return state->SupportsSpatial();
}

MetalFXSpatial::Batch MetalFXSpatial::Apply(std::span<const SpatialRequest> requests, bool enabled,
                                            u64 produced_ticket,
                                            const DeviceContext::WorkLease& producer) {
    @autoreleasepool {
        return state->Apply(requests, enabled, produced_ticket, producer);
    }
}

} // namespace Metal
