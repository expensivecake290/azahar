// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.txt file included.

#pragma once

#import <Metal/Metal.h>

#include <memory>
#include <optional>
#include <shared_mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include "common/common_types.h"

namespace Metal {

class MetalError final : public std::runtime_error {
public:
    MetalError(std::string_view operation, NSError* error);

    [[nodiscard]] const std::string& Domain() const noexcept {
        return domain;
    }

    [[nodiscard]] s64 Code() const noexcept {
        return code;
    }

private:
    std::string domain;
    s64 code{};
};

template <typename T>
[[nodiscard]] T CheckResource(T resource, std::string_view operation, NSError* error = nil) {
    if (!resource) {
        throw MetalError(operation, error);
    }
    return resource;
}

enum class WorkLeaseStatus : u8 {
    Valid,
    Empty,
    ForeignContext,
    StaleGeneration,
    Closed,
};

class DeviceContext {
    struct State;

public:
    /// Keeps the originating Metal device alive through asynchronous work and completion.
    class WorkLease {
    public:
        WorkLease() = default;

        [[nodiscard]] u64 Generation() const noexcept {
            return generation;
        }

    private:
        friend class DeviceContext;
        WorkLease(std::shared_ptr<State> state, u64 generation);

        std::shared_ptr<State> state;
        u64 generation{};
    };

    /// Hold across queue submission so device generation changes cannot invalidate a checked lease.
    class SubmissionGuard {
    public:
        SubmissionGuard(SubmissionGuard&&) noexcept = default;
        SubmissionGuard& operator=(SubmissionGuard&&) = delete;

    private:
        friend class DeviceContext;
        SubmissionGuard(std::shared_ptr<State> state, std::shared_lock<std::shared_mutex> lock);

        std::shared_ptr<State> state;
        std::shared_lock<std::shared_mutex> lock;
    };

    explicit DeviceContext(id<MTLDevice> device);
    ~DeviceContext();
    DeviceContext(const DeviceContext&) = delete;
    DeviceContext& operator=(const DeviceContext&) = delete;

    [[nodiscard]] id<MTLDevice> GetDevice() const noexcept;
    [[nodiscard]] u64 RegistryID() const noexcept;
    [[nodiscard]] WorkLease AcquireWork() const;
    [[nodiscard]] WorkLeaseStatus GetLeaseStatus(const WorkLease& lease) const;
    [[nodiscard]] std::optional<SubmissionGuard> AcquireSubmission(const WorkLease& lease) const;

    /// Invalidates existing leases and opens a new generation of work.
    [[nodiscard]] bool AdvanceGeneration();

    /// Stops new work and invalidates existing leases once active submissions leave their guards.
    void Close();

private:
    [[nodiscard]] WorkLeaseStatus GetLeaseStatusLocked(const WorkLease& lease) const noexcept;

    std::shared_ptr<State> state;
};

} // namespace Metal
