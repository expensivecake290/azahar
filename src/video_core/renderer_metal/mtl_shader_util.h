// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.txt file included.

#pragma once

#include <chrono>
#include <compare>
#include <cstddef>
#include <future>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include "video_core/renderer_metal/mtl_common.h"

namespace Metal {

enum class ShaderMath : u8 {
    Precise,
    Fast,
};

enum class ShaderOptimization : u8 {
    Default,
    Size,
};

struct ShaderCompileOptions {
    ShaderMath math = ShaderMath::Precise;
    ShaderOptimization optimization = ShaderOptimization::Default;
    bool preserve_invariance = true;
    std::map<std::string, std::string, std::less<>> definitions;

    auto operator<=>(const ShaderCompileOptions&) const = default;
};

struct ShaderSource {
    std::string text;
    std::string name;
    std::string preamble;
    ShaderCompileOptions options;

    auto operator<=>(const ShaderSource&) const = default;
};

enum class ShaderCompileStatus : u8 {
    Success,
    InvalidSource,
    InvalidBinary,
    CompileFailed,
    LoadFailed,
    QueueFull,
    Canceled,
    ForeignDevice,
};

struct ShaderDiagnostic {
    std::string message;
    std::string domain;
    s64 code{};
    bool warning{};
};

[[nodiscard]] constexpr bool HasMetalLibrarySignature(std::span<const u8> bytes) noexcept {
    return bytes.size() > 4 && bytes[0] == 'M' && bytes[1] == 'T' && bytes[2] == 'L' &&
           bytes[3] == 'B';
}

[[nodiscard]] constexpr bool IsShaderIdentifier(std::string_view name) noexcept {
    const auto alpha = [](char ch) {
        return (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') || ch == '_';
    };
    if (name.empty() || !alpha(name.front())) {
        return false;
    }
    for (const char ch : name.substr(1)) {
        if (!alpha(ch) && (ch < '0' || ch > '9')) {
            return false;
        }
    }
    return true;
}

struct ShaderCompilationEpoch;

class ShaderLibrary {
public:
    /// A resolved library is usable only by its original device and live device and title
    /// generations.
    [[nodiscard]] ShaderCompileStatus GetStatus(const DeviceContext& device) const;
    [[nodiscard]] id<MTLLibrary> GetLibrary(const DeviceContext& device) const;

    [[nodiscard]] const ShaderDiagnostic& GetDiagnostic() const noexcept {
        return diagnostic;
    }

private:
    friend class ShaderCompiler;
    ShaderCompileStatus status{ShaderCompileStatus::Canceled};
    ShaderDiagnostic diagnostic;
    id<MTLLibrary> library = nil;
    DeviceContext::WorkLease lease;
    std::shared_ptr<const ShaderCompilationEpoch> epoch;
    u64 title_generation{};
};

class ShaderCompilation {
public:
    [[nodiscard]] bool IsReady() const {
        return future.valid() &&
               future.wait_for(std::chrono::seconds::zero()) == std::future_status::ready;
    }

    [[nodiscard]] std::optional<ShaderLibrary> WaitUntil(
        std::chrono::steady_clock::time_point deadline) const {
        if (!future.valid() || future.wait_until(deadline) != std::future_status::ready) {
            return std::nullopt;
        }
        return future.get();
    }

private:
    friend class ShaderCompiler;
    explicit ShaderCompilation(std::shared_future<ShaderLibrary> future)
        : future(std::move(future)) {}

    std::shared_future<ShaderLibrary> future;
};

/// Compiler and in-flight tasks share retained SDK resources. The originating context must outlive
/// ShaderCompiler method calls; Metal callbacks retain a lease and never access the context object.
class ShaderCompiler {
public:
    explicit ShaderCompiler(DeviceContext& context, std::size_t max_async_jobs = 4);
    ShaderCompiler(DeviceContext& context, std::size_t max_async_jobs,
                   id<MTL4PipelineDataSetSerializer> serializer);
    ~ShaderCompiler();

    ShaderCompiler(const ShaderCompiler&) = delete;
    ShaderCompiler& operator=(const ShaderCompiler&) = delete;

    /// Uses the native synchronous compiler independently of the async queue and its capacity.
    [[nodiscard]] ShaderLibrary CompileLibrary(const ShaderSource& source);

    /// Identical in-flight requests from the same device/title generation share a result. A full or
    /// disabled queue publishes QueueFull so the caller can take the native synchronous path.
    [[nodiscard]] ShaderCompilation QueueLibrary(ShaderSource source);

    /// Load an already compiled executable Metal library, including CMake-embedded native assets.
    [[nodiscard]] ShaderLibrary LoadLibrary(std::span<const u8> bytes, std::string_view label);

    /// Discard all results from the previous title and signal canceled futures. Native compiler
    /// tasks may still finish and retain their own devices until the callback releases them.
    void InvalidateTitle();
    void Close();

    [[nodiscard]] u64 TitleGeneration() const;
    [[nodiscard]] id<MTL4Compiler> GetNativeCompiler(const DeviceContext::WorkLease& lease) const;

private:
    struct State;
    DeviceContext& context;
    std::shared_ptr<State> state;
};

} // namespace Metal
