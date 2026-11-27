// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.txt file included.

#import <Foundation/Foundation.h>

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>
#include "video_core/renderer_metal/mtl_master_semaphore.h"

namespace Metal {

CompletionTimeline::CompletionTimeline(u64 initial_tick)
    : issued(initial_tick), retired(initial_tick) {}

u64 CompletionTimeline::Issue() {
    std::scoped_lock lock(mutex);
    if (closed || fault) {
        throw std::logic_error("Cannot issue work on a closed or faulted Metal timeline");
    }
    if (issued == std::numeric_limits<u64>::max()) {
        throw std::overflow_error("Metal completion timeline exhausted its tickets");
    }
    pending.push_back(false);
    return ++issued;
}

u64 CompletionTimeline::LastIssuedTick() const {
    std::scoped_lock lock(mutex);
    return issued;
}

u64 CompletionTimeline::RetiredTick() const {
    std::scoped_lock lock(mutex);
    return retired;
}

u64 CompletionTimeline::SuccessfulTick() const {
    std::scoped_lock lock(mutex);
    return fault ? std::min(retired, fault->ticket - 1) : retired;
}

std::optional<CompletionFault> CompletionTimeline::FirstFault() const {
    std::scoped_lock lock(mutex);
    return fault;
}

bool CompletionTimeline::ReportCompletion(u64 ticket, NSError* error) {
    {
        std::scoped_lock lock(mutex);
        if (ticket <= retired || ticket > issued) {
            return false;
        }
        const auto index = ticket - retired - 1;
        if (pending[index]) {
            return false;
        }
        if (error && (!fault || ticket < fault->ticket)) {
            fault = CompletionFault{ticket, error};
        }
        pending[index] = true;
        while (!pending.empty() && pending.front()) {
            pending.pop_front();
            ++retired;
        }
    }
    changed.notify_all();
    return true;
}

CompletionStatus CompletionTimeline::Wait(u64 ticket, Deadline deadline) const {
    std::unique_lock lock(mutex);
    if (ticket > issued) {
        return CompletionStatus::UnknownTicket;
    }
    const auto ready = [&] {
        return ticket <= retired || (fault && fault->ticket <= ticket) || closed;
    };
    if (deadline) {
        if (!changed.wait_until(lock, *deadline, ready)) {
            return CompletionStatus::TimedOut;
        }
    } else {
        changed.wait(lock, ready);
    }
    if (fault && fault->ticket <= ticket) {
        return CompletionStatus::Faulted;
    }
    return ticket <= retired ? CompletionStatus::Complete : CompletionStatus::Closed;
}

bool CompletionTimeline::WaitRetired(u64 ticket, Deadline deadline) const {
    std::unique_lock lock(mutex);
    if (ticket > issued) {
        return false;
    }
    const auto ready = [&] { return ticket <= retired; };
    if (deadline) {
        return changed.wait_until(lock, *deadline, ready);
    }
    changed.wait(lock, ready);
    return true;
}

void CompletionTimeline::Close() {
    {
        std::scoped_lock lock(mutex);
        closed = true;
    }
    changed.notify_all();
}

struct MasterSemaphore::State {
    explicit State(id<MTLDevice> device)
        : device(CheckResource(device, "Create Metal completion timeline")),
          event(CheckResource([device newSharedEvent], "Create Metal completion event")) {
        event.label = @"Metaliris completion";
    }

    void PublishSuccess() {
        std::scoped_lock lock(event_mutex);
        const auto successful = timeline.SuccessfulTick();
        if (successful > event.signaledValue) {
            event.signaledValue = successful;
        }
    }

    id<MTLDevice> device;
    id<MTLSharedEvent> event;
    CompletionTimeline timeline;
    std::mutex event_mutex;
};

MasterSemaphore::MasterSemaphore(id<MTLDevice> device) : state(std::make_shared<State>(device)) {}

MasterSemaphore::~MasterSemaphore() {
    Close();
}

id<MTLDevice> MasterSemaphore::GetDevice() const noexcept {
    return state->device;
}

u64 MasterSemaphore::Issue() {
    return state->timeline.Issue();
}

const CompletionTimeline& MasterSemaphore::Timeline() const noexcept {
    return state->timeline;
}

bool MasterSemaphore::ReportFailure(u64 ticket, NSError* error) {
    if (!error) {
        throw std::invalid_argument("A failed Metal submission must report its error");
    }
    const bool recorded = state->timeline.ReportCompletion(ticket, error);
    if (recorded) {
        state->PublishSuccess();
    }
    return recorded;
}

MTL4CommitFeedbackHandler MasterSemaphore::MakeCompletionHandler(
    u64 ticket, std::shared_ptr<void> retain_until_completion) const {
    if (ticket <= state->timeline.RetiredTick() || ticket > state->timeline.LastIssuedTick()) {
        throw std::out_of_range("Metal feedback needs an outstanding completion ticket");
    }
    auto shared = state;
    __block std::shared_ptr<void> resources = std::move(retain_until_completion);
    return ^(id<MTL4CommitFeedback> feedback) {
        @autoreleasepool {
            NSError* error = feedback.error;
            if (!feedback) {
                error = [NSError errorWithDomain:MTL4CommandQueueErrorDomain
                                            code:MTL4CommandQueueErrorInternal
                                        userInfo:nil];
            }
            resources.reset();
            if (shared->timeline.ReportCompletion(ticket, error)) {
                shared->PublishSuccess();
            }
        }
    };
}

CompletionStatus MasterSemaphore::WaitOnQueue(id<MTL4CommandQueue> queue, u64 ticket,
                                               CompletionTimeline::Deadline deadline) const {
    if (!queue || queue.device != state->device) {
        throw std::invalid_argument("Metal dependency queue belongs to another device");
    }
    const auto status = state->timeline.Wait(ticket, deadline);
    if (status == CompletionStatus::Complete && ticket != 0) {
        state->PublishSuccess();
        [queue waitForEvent:state->event value:ticket];
    }
    return status;
}

void MasterSemaphore::Close() {
    state->timeline.Close();
}

} // namespace Metal
