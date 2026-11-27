// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.txt file included.

#include <algorithm>
#include <atomic>
#include <limits>
#include <mutex>
#include <utility>
#include "video_core/renderer_metal/mtl_shader_util.h"

namespace Metal {

struct ShaderCompilationEpoch {
    /// Zero denotes a closed compiler. Every other value identifies a distinct title generation.
    std::atomic<u64> generation{1};
};

namespace {

NSString* ToString(std::string_view text) {
    if (text.empty()) {
        return @"";
    }
    if (text.find('\0') != std::string_view::npos) {
        return nil;
    }
    return [[NSString alloc] initWithBytes:text.data()
                                    length:text.size()
                                  encoding:NSUTF8StringEncoding];
}

std::string FromString(NSString* text) {
    const char* bytes = text.UTF8String;
    return bytes ? bytes : "";
}

MTL4LibraryDescriptor* MakeDescriptor(const ShaderSource& source) {
    if (source.text.find_first_not_of(" \t\r\n") == std::string::npos ||
        (source.options.math != ShaderMath::Precise && source.options.math != ShaderMath::Fast) ||
        (source.options.optimization != ShaderOptimization::Default &&
         source.options.optimization != ShaderOptimization::Size)) {
        return nil;
    }

    const auto label = ToString(source.name);
    const auto text = source.preamble.empty() ? ToString(source.text)
                                              : ToString(source.preamble + "\n" + source.text);
    if (!label || !text) {
        return nil;
    }

    auto options = [MTLCompileOptions new];
    options.languageVersion = MTLLanguageVersion4_0;
    const bool precise = source.options.math == ShaderMath::Precise;
    options.mathMode = precise ? MTLMathModeSafe : MTLMathModeFast;
    options.mathFloatingPointFunctions =
        precise ? MTLMathFloatingPointFunctionsPrecise : MTLMathFloatingPointFunctionsFast;
    options.preserveInvariance = source.options.preserve_invariance;
    options.optimizationLevel = source.options.optimization == ShaderOptimization::Size
                                    ? MTLLibraryOptimizationLevelSize
                                    : MTLLibraryOptimizationLevelDefault;
    options.libraryType = MTLLibraryTypeExecutable;
    if (!source.options.definitions.empty()) {
        auto definitions =
            [NSMutableDictionary dictionaryWithCapacity:source.options.definitions.size()];
        for (const auto& [name, value] : source.options.definitions) {
            const auto string_value = ToString(value);
            if (!IsShaderIdentifier(name) || !string_value) {
                return nil;
            }
            definitions[ToString(name)] = string_value;
        }
        options.preprocessorMacros = definitions;
    }

    auto descriptor = [MTL4LibraryDescriptor new];
    descriptor.source = text;
    descriptor.options = options;
    descriptor.name = label.length ? label : nil;
    return descriptor;
}

} // Anonymous namespace

struct ShaderCompiler::State {
    struct Key {
        u64 title_generation;
        u64 device_generation;
        ShaderSource source;
        auto operator<=>(const Key&) const = default;
    };

    struct Job {
        Job(DeviceContext::WorkLease lease, u64 generation, MTL4LibraryDescriptor* descriptor)
            : result(promise.get_future().share()), lease(std::move(lease)), generation(generation),
              descriptor(descriptor) {}

        std::promise<ShaderLibrary> promise;
        const std::shared_future<ShaderLibrary> result;
        const DeviceContext::WorkLease lease;
        const u64 generation;
        MTL4LibraryDescriptor* descriptor;
        id<MTL4CompilerTask> task = nil;
        bool published{};
        bool finished{};
    };

    State(id<MTLDevice> device, std::size_t limit, id<MTL4PipelineDataSetSerializer> serializer)
        : device(device), limit(limit), epoch(std::make_shared<ShaderCompilationEpoch>()) {
        auto descriptor = [MTL4CompilerDescriptor new];
        descriptor.label = @"Metaliris shader compiler";
        descriptor.pipelineDataSetSerializer = serializer;
        NSError* error = nil;
        auto native = [device newCompilerWithDescriptor:descriptor error:&error];
        compiler = CheckResource(native, "Create Metal 4 shader compiler", error);
    }

    static ShaderLibrary Failure(ShaderCompileStatus status, std::string_view message) {
        ShaderLibrary result;
        result.status = status;
        result.diagnostic.message = message;
        return result;
    }

    static ShaderCompilation Ready(ShaderLibrary result) {
        std::promise<ShaderLibrary> promise;
        auto future = promise.get_future().share();
        promise.set_value(std::move(result));
        return ShaderCompilation(std::move(future));
    }

    ShaderLibrary MakeResult(id<MTLLibrary> native, NSError* error, ShaderCompileStatus failure,
                             const DeviceContext::WorkLease& lease, u64 generation) const {
        ShaderLibrary result;
        result.lease = lease;
        result.epoch = epoch;
        result.title_generation = generation;
        if (native && native.device != device) {
            result.status = ShaderCompileStatus::ForeignDevice;
            result.diagnostic.message = "Metal compiler returned a library for another device";
            return result;
        }

        result.library = native;
        result.status = native ? ShaderCompileStatus::Success : failure;
        if (error) {
            result.diagnostic.message = FromString(error.localizedDescription);
            result.diagnostic.domain = FromString(error.domain);
            result.diagnostic.code = error.code;
            result.diagnostic.warning = native != nil;
        } else if (!native) {
            result.diagnostic.message = "Metal returned no executable shader library";
        }
        return result;
    }

    void Publish(Job& job, ShaderLibrary result) {
        if (!job.published) {
            job.promise.set_value(std::move(result));
            job.published = true;
        }
    }

    void Complete(const std::shared_ptr<Job>& job, id<MTLLibrary> native, NSError* error) {
        std::lock_guard lock(mutex);
        if (job->finished) {
            return;
        }
        job->finished = true;
        if (!job->published) {
            try {
                Publish(*job, MakeResult(native, error, ShaderCompileStatus::CompileFailed,
                                         job->lease, job->generation));
            } catch (...) {
                job->promise.set_exception(std::current_exception());
                job->published = true;
            }
        }
        job->task = nil;
        job->descriptor = nil;
        std::erase_if(active, [&](const auto& item) { return item.second == job; });
    }

    void Invalidate(bool close) {
        std::lock_guard submit_lock(admission);
        std::lock_guard lock(mutex);
        const u64 previous = epoch->generation.load();
        if (previous == 0) {
            return;
        }
        epoch->generation.store(
            close || previous == std::numeric_limits<u64>::max() ? 0 : previous + 1);
        for (const auto& [_, job] : active) {
            Publish(*job, ShaderLibrary{});
        }
    }

    const id<MTLDevice> device;
    const std::size_t limit;
    const std::shared_ptr<ShaderCompilationEpoch> epoch;
    id<MTL4Compiler> compiler = nil;
    /// Serialize native submissions against title invalidation, leaving completion callbacks free.
    std::mutex admission;
    std::mutex mutex;
    std::map<Key, std::shared_ptr<Job>> active;
};

ShaderCompileStatus ShaderLibrary::GetStatus(const DeviceContext& device) const {
    if (!epoch) {
        return status;
    }
    if (epoch->generation.load() != title_generation) {
        return ShaderCompileStatus::Canceled;
    }
    switch (device.GetLeaseStatus(lease)) {
    case WorkLeaseStatus::Valid:
        return status;
    case WorkLeaseStatus::ForeignContext:
        return ShaderCompileStatus::ForeignDevice;
    default:
        return ShaderCompileStatus::Canceled;
    }
}

id<MTLLibrary> ShaderLibrary::GetLibrary(const DeviceContext& device) const {
    return GetStatus(device) == ShaderCompileStatus::Success ? library : nil;
}

ShaderCompiler::ShaderCompiler(DeviceContext& context, std::size_t max_async_jobs)
    : ShaderCompiler(context, max_async_jobs, nil) {}

ShaderCompiler::ShaderCompiler(DeviceContext& context, std::size_t max_async_jobs,
                               id<MTL4PipelineDataSetSerializer> serializer)
    : context(context),
      state(std::make_shared<State>(context.GetDevice(), max_async_jobs, serializer)) {}

ShaderCompiler::~ShaderCompiler() {
    Close();
}

ShaderLibrary ShaderCompiler::CompileLibrary(const ShaderSource& source) {
    @autoreleasepool {
        const auto lease = context.AcquireWork();
        auto guard = context.AcquireSubmission(lease);
        const u64 generation = TitleGeneration();
        if (!guard || generation == 0) {
            return State::Failure(ShaderCompileStatus::Canceled, "Shader compiler is closed");
        }

        auto descriptor = MakeDescriptor(source);
        if (!descriptor) {
            return State::Failure(ShaderCompileStatus::InvalidSource,
                                  "Invalid MSL source, label or compile options");
        }
        NSError* error = nil;
        auto library = [state->compiler newLibraryWithDescriptor:descriptor error:&error];
        return state->MakeResult(library, error, ShaderCompileStatus::CompileFailed, lease,
                                 generation);
    }
}

ShaderCompilation ShaderCompiler::QueueLibrary(ShaderSource source) {
    @autoreleasepool {
        const auto retained = state;
        const auto lease = context.AcquireWork();
        auto guard = context.AcquireSubmission(lease);
        if (!guard) {
            return State::Ready(
                State::Failure(ShaderCompileStatus::Canceled, "Shader device context is closed"));
        }

        auto descriptor = MakeDescriptor(source);
        if (!descriptor) {
            return State::Ready(State::Failure(ShaderCompileStatus::InvalidSource,
                                               "Invalid MSL source, label or compile options"));
        }

        std::lock_guard submit_lock(retained->admission);
        const u64 generation = retained->epoch->generation.load();
        if (generation == 0) {
            return State::Ready(
                State::Failure(ShaderCompileStatus::Canceled, "Shader compiler is closed"));
        }

        std::shared_ptr<State::Job> job;
        {
            std::lock_guard lock(retained->mutex);
            State::Key key{generation, lease.Generation(), std::move(source)};
            if (const auto previous = retained->active.find(key);
                previous != retained->active.end()) {
                return ShaderCompilation(previous->second->result);
            }
            if (retained->active.size() >= retained->limit) {
                return State::Ready(State::Failure(ShaderCompileStatus::QueueFull,
                                                   "Async shader compiler is full"));
            }
            job = std::make_shared<State::Job>(lease, generation, descriptor);
            retained->active.emplace(std::move(key), job);
        }

        auto task =
            [retained->compiler newLibraryWithDescriptor:descriptor
                                       completionHandler:^(id<MTLLibrary> library, NSError* error) {
                                         @autoreleasepool {
                                             retained->Complete(job, library, error);
                                         }
                                       }];
        if (task) {
            std::lock_guard lock(retained->mutex);
            if (!job->finished) {
                job->task = task;
            }
        } else {
            retained->Complete(job, nil, nil);
        }
        return ShaderCompilation(job->result);
    }
}

ShaderLibrary ShaderCompiler::LoadLibrary(std::span<const u8> bytes, std::string_view label) {
    @autoreleasepool {
        const auto lease = context.AcquireWork();
        auto guard = context.AcquireSubmission(lease);
        const u64 generation = TitleGeneration();
        if (!guard || generation == 0) {
            return State::Failure(ShaderCompileStatus::Canceled, "Shader compiler is closed");
        }
        if (!HasMetalLibrarySignature(bytes)) {
            return State::Failure(ShaderCompileStatus::InvalidBinary,
                                  "Invalid executable Metal library signature");
        }
        const auto native_label = ToString(label);
        if (!native_label) {
            return State::Failure(ShaderCompileStatus::InvalidBinary,
                                  "Invalid Metal library label");
        }
        // The default dispatch data destructor requests an owning copy of the supplied byte span.
        auto data =
            dispatch_data_create(bytes.data(), bytes.size(), nil, DISPATCH_DATA_DESTRUCTOR_DEFAULT);
        if (!data) {
            return State::Failure(ShaderCompileStatus::LoadFailed,
                                  "Unable to allocate native Metal library data");
        }
        NSError* error = nil;
        auto library = [state->device newLibraryWithData:data error:&error];
        if (library && native_label.length) {
            library.label = native_label;
        }
        return state->MakeResult(library, error, ShaderCompileStatus::LoadFailed, lease,
                                 generation);
    }
}

void ShaderCompiler::InvalidateTitle() {
    state->Invalidate(false);
}

void ShaderCompiler::Close() {
    state->Invalidate(true);
}

u64 ShaderCompiler::TitleGeneration() const {
    return state->epoch->generation.load();
}

id<MTL4Compiler> ShaderCompiler::GetNativeCompiler(const DeviceContext::WorkLease& lease) const {
    return TitleGeneration() != 0 && context.GetLeaseStatus(lease) == WorkLeaseStatus::Valid
               ? state->compiler
               : nil;
}

} // namespace Metal
