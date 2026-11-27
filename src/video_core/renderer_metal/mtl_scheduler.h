// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.txt file included.

#pragma once

#include <concepts>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include "common/common_funcs.h"
#include "video_core/renderer_metal/mtl_memory_util.h"
#include "video_core/renderer_metal/mtl_resource_pool.h"

namespace Metal {

enum class StateFlags : u8 {
    AllDirty = 0,
    Pipeline = 1 << 0,
    ArgumentTables = 1 << 1,
};
DECLARE_ENUM_FLAG_OPERATORS(StateFlags)

enum class QueueType : u8 {
    Graphics,
    Transfer,
};

struct ResourceUse {
    id<MTLResource> resource;
    BufferAccess access;
    MTLStages stages;
    /// Length zero denotes the entire allocation. Buffer ranges are relative to backing.
    BufferRange range{};
    /// Optional shared backing identity for resources that alias the same allocation.
    id<MTLResource> backing = nil;

    [[nodiscard]] static constexpr bool Conflicts(BufferAccess first, BufferRange first_range,
                                                   BufferAccess second, BufferRange second_range) {
        return (first != BufferAccess::Read || second != BufferAccess::Read) &&
               (first_range.length == 0 || second_range.length == 0 ||
                first_range.Overlaps(second_range));
    }
};

struct SubmissionOptions {
    u64 wait_for_tick{};
    id<MTLDrawable> drawable = nil;
    /// A non-null drawable is waited on, signalled and presented on the submission queue.
    std::optional<double> minimum_present_duration;
};

enum class SubmissionStage : u8 {
    Allocation,
    Encoding,
    Dependency,
    DeviceGeneration,
    Commit,
    Execution,
    Canceled,
};

struct SubmissionCheckpoint {
    std::string label;
    std::optional<u64> gpu_timestamp;
};

struct SubmissionDiagnostic {
    u64 tick;
    QueueType queue;
    u64 device_generation;
    SubmissionStage stage;
    NSError* error;
    std::optional<SubmissionCheckpoint> checkpoint;
};

class Scheduler;

/// Worker-thread access to one native submission. Commands can share an open render pass.
class EncodingContext {
    struct Impl;

public:
    ~EncodingContext();
    EncodingContext(const EncodingContext&) = delete;
    EncodingContext& operator=(const EncodingContext&) = delete;

    [[nodiscard]] id<MTL4CommandBuffer> GetCommandBuffer() const noexcept;
    [[nodiscard]] u64 CurrentTick() const noexcept;
    [[nodiscard]] QueueType GetQueue() const noexcept;
    [[nodiscard]] bool OwnsWork(const DeviceContext::WorkLease& lease) const;
    [[nodiscard]] ResourcePool& GetResourcePool() const noexcept;

    /// Declare resource accesses before issuing the corresponding commands. An encoder stays
    /// current until EndPass. Register all attachment load/store accesses at pass start.
    void Synchronize(id<MTL4CommandEncoder> encoder, std::span<const ResourceUse> uses);
    void EndPass(id<MTL4CommandEncoder> encoder);

    /// Resources that are only accessed indirectly can require separate residency registration.
    void MakeResident(id<MTLResource> resource);
    void Retain(id object);
    void Retain(MemoryAllocator::GPUBufferAccess access);
    void Retain(ResourcePool::BindingLease table);
    void Retain(std::shared_ptr<void> owner);

    /// Record a GPU checkpoint between encoders. Its value is observable after commit feedback.
    void Checkpoint(std::string_view label);

private:
    friend class Scheduler;
    explicit EncodingContext(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl;
};

/// A single recording producer publishes work to an ordered native Metal encoding worker.
/// DeviceContext must outlive the worker. Completion observers may run on other threads.
class Scheduler {
    struct State;

    class Command {
    public:
        virtual ~Command() = default;
        virtual void Execute(EncodingContext& context) = 0;
    };

    template <typename T>
    class TypedCommand final : public Command {
    public:
        template <typename U>
        explicit TypedCommand(U&& command) : command(std::forward<U>(command)) {}

        void Execute(EncodingContext& context) override {
            std::invoke(command, context);
        }

    private:
        T command;
    };

public:
    explicit Scheduler(DeviceContext& device, std::size_t command_capacity = 8,
                       std::size_t binding_capacity = 256);
    ~Scheduler();
    Scheduler(const Scheduler&) = delete;
    Scheduler& operator=(const Scheduler&) = delete;

    /// Select a queue before recording any commands for the current submission.
    void SetQueue(QueueType queue);

    template <typename T>
        requires std::invocable<std::decay_t<T>&, EncodingContext&>
    void Record(T&& command) {
        Enqueue(std::make_unique<TypedCommand<std::decay_t<T>>>(std::forward<T>(command)));
    }

    void DispatchWork();
    void WaitWorker();

    [[nodiscard]] u64 Flush(const SubmissionOptions& options = {});
    [[nodiscard]] CompletionStatus Finish(const SubmissionOptions& options = {},
                                          CompletionTimeline::Deadline deadline = {});
    /// A producer waiting for its current tick flushes it. Observers wait on already issued ticks.
    [[nodiscard]] CompletionStatus Wait(u64 tick, CompletionTimeline::Deadline deadline = {});

    [[nodiscard]] u64 CurrentTick() const;
    [[nodiscard]] bool IsFree(u64 tick) const;
    [[nodiscard]] const MasterSemaphore& GetMasterSemaphore() const noexcept;
    [[nodiscard]] ResourcePool& GetResourcePool() noexcept;
    [[nodiscard]] std::optional<SubmissionDiagnostic> GetFault() const;
    [[nodiscard]] std::optional<SubmissionDiagnostic> GetLastCompletion() const;

    /// Cancel work that was not submitted and wait until committed work actually retires.
    [[nodiscard]] bool Shutdown(CompletionTimeline::Deadline deadline = {});

    void RegisterOnSubmit(std::function<void()> callback);
    void RegisterOnDispatch(std::function<void()> callback);

    void MarkStateNonDirty(StateFlags flags) noexcept {
        cached_state |= flags;
    }

    void MakeDirty(StateFlags flags) noexcept {
        cached_state &= ~flags;
    }

    [[nodiscard]] bool IsStateDirty(StateFlags flags) const noexcept {
        return False(cached_state & flags);
    }

private:
    void Enqueue(std::unique_ptr<Command> command);
    void Stop();

    std::unique_ptr<State> state;
    StateFlags cached_state{};
};

} // namespace Metal
