// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.txt file included.

#include <limits>
#include <mutex>
#include <utility>
#include "video_core/renderer_metal/mtl_common.h"

namespace Metal {

namespace {

std::string FromNSString(NSString* string) {
    const char* utf8 = string.UTF8String;
    return utf8 ? utf8 : "";
}

std::string ErrorMessage(std::string_view operation, NSError* error) {
    std::string message{operation};
    auto description = FromNSString(error.localizedDescription);
    message += ": ";
    message += description.empty() ? "Metal returned no resource" : description;
    return message;
}

} // namespace

MetalError::MetalError(std::string_view operation, NSError* error)
    : std::runtime_error(ErrorMessage(operation, error)), domain(FromNSString(error.domain)),
      code(static_cast<s64>(error.code)) {}

struct DeviceContext::State {
    explicit State(id<MTLDevice> device) : device(device), registry_id(device.registryID) {}

    const id<MTLDevice> device;
    const u64 registry_id;
    std::shared_mutex mutex;
    u64 generation{1}; ///< Zero denotes a closed context.
};

DeviceContext::WorkLease::WorkLease(std::shared_ptr<State> state_, u64 generation_)
    : state(std::move(state_)), generation(generation_) {}

DeviceContext::SubmissionGuard::SubmissionGuard(std::shared_ptr<State> state_,
                                                std::shared_lock<std::shared_mutex> lock_)
    : state(std::move(state_)), lock(std::move(lock_)) {}

DeviceContext::DeviceContext(id<MTLDevice> device)
    : state(std::make_shared<State>(CheckResource(device, "Create Metal device context"))) {}

DeviceContext::~DeviceContext() = default;

id<MTLDevice> DeviceContext::GetDevice() const noexcept {
    return state->device;
}

u64 DeviceContext::RegistryID() const noexcept {
    return state->registry_id;
}

DeviceContext::WorkLease DeviceContext::AcquireWork() const {
    std::shared_lock lock(state->mutex);
    if (state->generation == 0) {
        return {};
    }
    return WorkLease{state, state->generation};
}

WorkLeaseStatus DeviceContext::GetLeaseStatusLocked(const WorkLease& lease) const noexcept {
    if (!lease.state) {
        return WorkLeaseStatus::Empty;
    }
    if (lease.state != state) {
        return WorkLeaseStatus::ForeignContext;
    }
    if (state->generation == 0) {
        return WorkLeaseStatus::Closed;
    }
    if (lease.generation != state->generation) {
        return WorkLeaseStatus::StaleGeneration;
    }
    return WorkLeaseStatus::Valid;
}

WorkLeaseStatus DeviceContext::GetLeaseStatus(const WorkLease& lease) const {
    std::shared_lock lock(state->mutex);
    return GetLeaseStatusLocked(lease);
}

std::optional<DeviceContext::SubmissionGuard> DeviceContext::AcquireSubmission(
    const WorkLease& lease) const {
    std::shared_lock lock(state->mutex);
    if (GetLeaseStatusLocked(lease) != WorkLeaseStatus::Valid) {
        return std::nullopt;
    }
    return SubmissionGuard{state, std::move(lock)};
}

bool DeviceContext::AdvanceGeneration() {
    std::unique_lock lock(state->mutex);
    if (state->generation == 0) {
        return false;
    }
    if (state->generation == std::numeric_limits<u64>::max()) {
        state->generation = 0;
        return false;
    }
    ++state->generation;
    return true;
}

void DeviceContext::Close() {
    std::unique_lock lock(state->mutex);
    state->generation = 0;
}

} // namespace Metal
