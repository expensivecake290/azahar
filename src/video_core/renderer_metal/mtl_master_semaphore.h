// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.txt file included.

#pragma once

#include <chrono>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include "video_core/renderer_metal/mtl_common.h"

namespace Metal {

enum class CompletionStatus : u8 {
    Complete,
    TimedOut,
    Faulted,
    Closed,
    UnknownTicket,
};

struct CompletionFault {
    u64 ticket;
    NSError* error;
};

/// Tracks issued work independently of the order in which queues report completion.
class CompletionTimeline {
public:
    using Deadline = std::optional<std::chrono::steady_clock::time_point>;

    explicit CompletionTimeline(u64 initial_tick = 0);

    [[nodiscard]] u64 Issue();
    [[nodiscard]] u64 LastIssuedTick() const;
    [[nodiscard]] u64 RetiredTick() const;
    [[nodiscard]] u64 SuccessfulTick() const;
    [[nodiscard]] std::optional<CompletionFault> FirstFault() const;

    /// Report a terminal result only after the driver has finished using the work's resources.
    /// A nil error reports success. Repeated reports and unissued tickets are rejected.
    [[nodiscard]] bool ReportCompletion(u64 ticket, NSError* error = nil);

    [[nodiscard]] CompletionStatus Wait(u64 ticket, Deadline deadline = {}) const;

    /// Retirement remains meaningful after close or fault and is separate from success.
    [[nodiscard]] bool WaitRetired(u64 ticket, Deadline deadline = {}) const;
    void Close();

private:
    mutable std::mutex mutex;
    mutable std::condition_variable changed;
    std::deque<bool> pending;
    u64 issued;
    u64 retired;
    std::optional<CompletionFault> fault;
    bool closed{};
};

class MasterSemaphore {
    struct State;

public:
    explicit MasterSemaphore(id<MTLDevice> device);
    ~MasterSemaphore();
    MasterSemaphore(const MasterSemaphore&) = delete;
    MasterSemaphore& operator=(const MasterSemaphore&) = delete;

    [[nodiscard]] id<MTLDevice> GetDevice() const noexcept;
    [[nodiscard]] u64 Issue();
    [[nodiscard]] const CompletionTimeline& Timeline() const noexcept;
    [[nodiscard]] bool ReportFailure(u64 ticket, NSError* error);

    /// Bind to one commit's options. Retain resources through completion feedback.
    [[nodiscard]] MTL4CommitFeedbackHandler MakeCompletionHandler(
        u64 ticket, std::shared_ptr<void> retain_until_completion = {}) const;

    /// Orders the queue after a successful producer. Faulted or closed work never enqueues a wait.
    [[nodiscard]] CompletionStatus WaitOnQueue(id<MTL4CommandQueue> queue, u64 ticket,
                                               CompletionTimeline::Deadline deadline = {}) const;
    void Close();

private:
    std::shared_ptr<State> state;
};

} // namespace Metal
