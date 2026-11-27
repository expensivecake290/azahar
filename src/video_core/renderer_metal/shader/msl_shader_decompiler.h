// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.txt file included.

#pragma once

#include <array>
#include <cstddef>
#include <string>
#include <string_view>
#include <vector>
#include "video_core/pica/shader_setup.h"
#include "video_core/shader/generator/shader_uniforms.h"

namespace Pica::Shader::Generator::MSL {

enum class ProgramKind : u8 {
    Vertex,
    Geometry,
};

enum class UniformStorage : u8 {
    Constant,
    Thread,
};

struct ProgramOptions {
    ProgramKind kind = ProgramKind::Vertex;
    std::string_view function_name = "ExecutePica";
    UniformStorage uniform_storage = UniformStorage::Constant;
};

enum class RunStatus : u32 {
    Complete,
    InvalidProgramCounter,
    InvalidGeometryOperation,
    EmitOverflow,
    NonTerminatingControlFlow,
};

enum class DecompileIssue : u8 {
    InvalidEntry,
    InvalidFunctionName,
    InvalidProgramKind,
    UnsupportedInstruction,
    InvalidComparison,
    InvalidGeometryOperation,
    InvalidUniformStorage,
};

struct DecompileDiagnostic {
    DecompileIssue issue;
    u32 offset;
    u32 instruction;
};

struct DecompiledProgram {
    std::string source;
    std::vector<DecompileDiagnostic> diagnostics;

    [[nodiscard]] explicit operator bool() const noexcept {
        return !source.empty();
    }
};

/// Shared with MSL for register transfer and differential testing. Host and shader arithmetic use
/// the float32 representation of PICA f24, as in the interpreter.
struct alignas(16) RegisterState {
    using Vector = std::array<f32, 4>;
    using RegisterBank = std::array<Vector, 16>;

    RegisterBank input{};
    RegisterBank temporary{};
    std::array<RegisterBank, 2> output{};
    std::array<s32, 4> address{};
    /// Comparison X/Y and the active output bank.
    std::array<u32, 4> condition{};
    /// Geometry vertex slot, primitive emission flag and winding reversal flag.
    std::array<u32, 4> emission{};
};

static_assert(sizeof(RegisterState) == 0x430);
static_assert(offsetof(RegisterState, temporary) == 0x100);
static_assert(offsetof(RegisterState, output) == 0x200);
static_assert(offsetof(RegisterState, address) == 0x400);
static_assert(offsetof(RegisterState, condition) == 0x410);
static_assert(offsetof(RegisterState, emission) == 0x420);
static_assert(offsetof(VSPicaUniformData, b) == 0);
static_assert(offsetof(VSPicaUniformData, i) == 16);
static_assert(offsetof(VSPicaUniformData, f) == 80);

/// Include once per MSL source library before any generated PICA function. It declares native
/// register state, uniforms and exact arithmetic helpers in the MetalirisPica namespace.
[[nodiscard]] std::string_view DecompilerPreamble() noexcept;

/// Emit a directly callable MSL PICA program. The caller supplies RegisterState and the shared
/// VSPicaUniformData layout in the supplied Metal address space. Geometry functions additionally
/// take an emitter with
/// bool Emit(thread const float4*, uint vertex, bool primitive, bool winding).
[[nodiscard]] DecompiledProgram DecompileProgram(const ProgramCode& code,
                                                 const SwizzleData& swizzles, u32 entry,
                                                 ProgramOptions options = {});

} // namespace Pica::Shader::Generator::MSL
