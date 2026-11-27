// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.txt file included.

#include <algorithm>
#include <bitset>
#include <iterator>
#include <utility>
#include <fmt/format.h>
#include <nihstro/shader_bytecode.h>
#include "video_core/renderer_metal/shader/msl_shader_decompiler.h"

namespace Pica::Shader::Generator::MSL {
namespace {

using nihstro::DestRegister;
using nihstro::Instruction;
using nihstro::OpCode;
using nihstro::RegisterType;
using nihstro::SourceRegister;
using nihstro::SwizzlePattern;

constexpr u32 ImplicitEnd = MAX_PROGRAM_CODE_LENGTH - 1;

constexpr std::string_view Preamble = R"MSL(
#include <metal_stdlib>

namespace MetalirisPica {
using namespace metal;

enum class RunStatus : uint {
    Complete,
    InvalidProgramCounter,
    InvalidGeometryOperation,
    EmitOverflow,
    NonTerminatingControlFlow,
};

struct RegisterState {
    float4 input[16];
    float4 temporary[16];
    float4 output[2][16];
    int4 address;
    uint4 condition;
    uint4 emission;
};

struct Uniforms {
    uint b;
    uint4 i[4];
    float4 f[96];
};

static_assert(sizeof(RegisterState) == 0x430, "PICA register layout");
static_assert(sizeof(Uniforms) == 1616, "PICA uniform layout");

inline float DecodeFloat24(uint raw) {
    const uint sign = (raw & 0x800000u) << 8u;
    uint exp = (raw >> 16u) & 127u;
    uint mantissa = raw & 0xFFFFu;
    if (exp == 0u) {
        if (mantissa == 0u) {
            return as_type<float>(sign);
        }
        exp = 65u;
        while ((mantissa & 0x10000u) == 0u) {
            mantissa <<= 1u;
            --exp;
        }
        mantissa &= 0xFFFFu;
    } else {
        exp = exp == 127u ? 255u : exp + 64u;
    }
    return as_type<float>(sign | (exp << 23u) | (mantissa << 7u));
}

inline float4 Multiply(float4 lhs, float4 rhs) {
    const float4 product = lhs * rhs;
    return select(product, float4(0.0f), isnan(product) & !isnan(lhs) & !isnan(rhs));
}

inline float OrderedDot(float4 lhs, float4 rhs, bool use_w, bool homogeneous) {
    #pragma clang fp contract(off)
    if (homogeneous) {
        lhs.w = 1.0f;
    }
    const float4 terms = Multiply(lhs, rhs);
    float sum = 0.0f + terms.x;
    sum += terms.y;
    sum += terms.z;
    if (use_w) {
        sum += terms.w;
    }
    return sum;
}

inline int ConvertAddress(float value) {
    if (isnan(value)) {
        return 0;
    }
    if (value >= 2147483648.0f) {
        return 2147483647;
    }
    if (value <= -2147483648.0f) {
        return -2147483647 - 1;
    }
    return int(value);
}

inline uint RelativeUniformIndex(int base, int offset) {
    if (offset < -128 || offset > 127) {
        offset = 0;
    }
    return uint((base + offset) & 127);
}

inline float4 RelativeUniform(constant Uniforms& uniforms, int base, int offset) {
    const uint index = RelativeUniformIndex(base, offset);
    return index < 96u ? uniforms.f[index] : float4(1.0f);
}

inline float4 RelativeUniform(thread const Uniforms& uniforms, int base, int offset) {
    const uint index = RelativeUniformIndex(base, offset);
    return index < 96u ? uniforms.f[index] : float4(1.0f);
}

struct CallScope {
    uint end;
    uint resume;
};

struct IfScope {
    uint else_address;
    uint end;
};

struct LoopScope {
    uint entry;
    uint end;
    uint remaining;
    uint increment;
    uint previous_address;
};

template <typename T, uint Capacity>
inline void Push(thread T (&stack)[Capacity], thread uint& depth, T scope) {
    if (depth == Capacity) {
        for (uint i = 1; i < Capacity; ++i) {
            stack[i - 1] = stack[i];
        }
    } else {
        ++depth;
    }
    stack[depth - 1] = scope;
}

} // namespace MetalirisPica
)MSL";

Instruction ReadInstruction(const ProgramCode& code, u32 address) {
    if (address < ImplicitEnd) {
        return Instruction{code[address]};
    }
    Instruction instruction{};
    instruction.opcode.Assign(OpCode::Id::END);
    return instruction;
}

struct ProgramLayout {
    std::bitset<MAX_PROGRAM_CODE_LENGTH> reachable;
    bool has_control_flow{};
};

ProgramLayout AnalyzeProgram(const ProgramCode& code, u32 entry) {
    ProgramLayout layout;
    std::vector<u32> pending;
    auto mark = [&](u32 pc) {
        if (pc <= ImplicitEnd && !layout.reachable.test(pc)) {
            layout.reachable.set(pc);
            pending.push_back(pc);
        }
    };
    mark(entry);
    for (std::size_t i = 0; i < pending.size(); ++i) {
        const u32 pc = pending[i];
        const Instruction instruction = ReadInstruction(code, pc);
        const auto opcode = instruction.opcode.Value().EffectiveOpCode();
        if (opcode == OpCode::Id::END) {
            continue;
        }
        mark(pc + 1);
        const u32 target = instruction.flow_control.dest_offset;
        switch (opcode) {
        case OpCode::Id::CALL:
        case OpCode::Id::CALLC:
        case OpCode::Id::CALLU:
        case OpCode::Id::JMPC:
        case OpCode::Id::JMPU:
            layout.has_control_flow = true;
            mark(target);
            break;
        case OpCode::Id::IFC:
        case OpCode::Id::IFU:
            layout.has_control_flow = true;
            mark(target);
            mark(target + instruction.flow_control.num_instructions);
            break;
        case OpCode::Id::LOOP:
            layout.has_control_flow = true;
            mark(target + 1);
            break;
        case OpCode::Id::BREAK:
        case OpCode::Id::BREAKC:
            layout.has_control_flow = true;
            break;
        default:
            break;
        }
    }
    return layout;
}

bool IsFunctionName(std::string_view name) {
    auto identifier_start = [](char c) {
        return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || c == '_';
    };
    return !name.empty() && identifier_start(name.front()) &&
           std::ranges::all_of(
               name, [&](char c) { return identifier_start(c) || (c >= '0' && c <= '9'); });
}

constexpr std::string_view Components = "xyzw";

std::string SourceExpression(SourceRegister source, u32 address) {
    const auto index = source.GetIndex();
    switch (source.GetRegisterType()) {
    case RegisterType::Input:
        return fmt::format("registers.input[{}]", index);
    case RegisterType::Temporary:
        return fmt::format("registers.temporary[{}]", index);
    case RegisterType::FloatUniform:
        if (address) {
            return fmt::format("RelativeUniform(uniforms, {}, registers.address.{})", index,
                               Components[address - 1]);
        }
        return fmt::format("uniforms.f[{}]", index);
    default:
        return "float4(0.0f)";
    }
}

std::optional<std::string> DestinationExpression(DestRegister destination) {
    if (destination < 0x10) {
        return fmt::format("registers.output[registers.condition.z & 1u][{}]",
                           destination.GetIndex());
    }
    if (destination < 0x20) {
        return fmt::format("registers.temporary[{}]", destination.GetIndex());
    }
    return {};
}

std::string Operand(SourceRegister source, u32 address, SwizzlePattern swizzle, u32 operand) {
    using Selector = SwizzlePattern::Selector (SwizzlePattern::*)(int) const;
    constexpr std::array<Selector, 3> select{&SwizzlePattern::GetSelectorSrc1,
                                             &SwizzlePattern::GetSelectorSrc2,
                                             &SwizzlePattern::GetSelectorSrc3};
    const std::array<bool, 3> negate{swizzle.negate_src1.Value() != 0,
                                     swizzle.negate_src2.Value() != 0,
                                     swizzle.negate_src3.Value() != 0};
    std::array<char, 4> lanes{};
    for (int i = 0; i < 4; ++i) {
        lanes[i] = Components[static_cast<u32>((swizzle.*select[operand])(i))];
    }
    return fmt::format("{}({}).{}", negate[operand] ? "-" : "", SourceExpression(source, address),
                       std::string_view(lanes.data(), lanes.size()));
}

std::string Condition(Instruction::FlowControlType flow) {
    using Op = Instruction::FlowControlType::Op;
    const auto x = fmt::format("(registers.condition.x {} 0u)", flow.refx ? "!=" : "==");
    const auto y = fmt::format("(registers.condition.y {} 0u)", flow.refy ? "!=" : "==");
    switch (flow.op) {
    case Op::Or:
        return fmt::format("({} || {})", x, y);
    case Op::And:
        return fmt::format("({} && {})", x, y);
    case Op::JustX:
        return x;
    case Op::JustY:
        return y;
    }
    return "false";
}

std::string BooleanCondition(u32 index, bool inverted = false) {
    return fmt::format("((uniforms.b & {}u) {} 0u)", 1u << index, inverted ? "==" : "!=");
}

std::string_view Comparison(Instruction::Common::CompareOpType::Op mode) {
    using Modes = Instruction::Common::CompareOpType;
    switch (mode) {
    case Modes::Equal:
        return "==";
    case Modes::NotEqual:
        return "!=";
    case Modes::LessThan:
        return "<";
    case Modes::LessEqual:
        return "<=";
    case Modes::GreaterThan:
        return ">";
    case Modes::GreaterEqual:
        return ">=";
    default:
        return {};
    }
}

class Emitter {
public:
    Emitter(const ProgramCode& code, const SwizzleData& swizzles, ProgramOptions options)
        : code(code), swizzles(swizzles), options(options) {}

    DecompiledProgram Generate(u32 entry) {
        const auto layout = AnalyzeProgram(code, entry);
        result.source = "namespace MetalirisPica {\n";
        if (options.kind == ProgramKind::Geometry) {
            Emit("template <typename GeometryEmitter>\n");
        }
        Emit("inline RunStatus {}(thread RegisterState& registers, {} Uniforms& uniforms",
             options.function_name,
             options.uniform_storage == UniformStorage::Thread ? "thread const" : "constant");
        if (options.kind == ProgramKind::Geometry) {
            Emit(", thread GeometryEmitter& emitter");
        }
        Emit(") {{\n    #pragma clang fp contract(off)\n");

        if (layout.has_control_flow) {
            EmitControlFlow(entry, layout);
        } else {
            for (u32 address = entry; address <= ImplicitEnd && layout.reachable.test(address);
                 ++address) {
                const auto instruction = ReadInstruction(code, address);
                if (instruction.opcode.Value() == OpCode::Id::END) {
                    break;
                }
                Emit("    {{\n");
                EmitInstruction(instruction, address);
                Emit("    }}\n");
            }
            Emit("    return RunStatus::Complete;\n");
        }
        Emit("}}\n}} // namespace MetalirisPica\n");
        return std::move(result);
    }

private:
    template <typename... Args>
    void Emit(fmt::format_string<Args...> text, Args&&... args) {
        fmt::format_to(std::back_inserter(result.source), text, std::forward<Args>(args)...);
    }

    void Diagnose(DecompileIssue issue, u32 address, Instruction instruction) {
        result.diagnostics.push_back({issue, address, instruction.hex});
    }

    void WriteDestination(DestRegister destination, SwizzlePattern swizzle,
                          std::string_view value) {
        const auto reg = DestinationExpression(destination);
        if (!reg) {
            return;
        }
        std::string lanes;
        for (u32 i = 0; i < 4; ++i) {
            if (swizzle.DestComponentEnabled(i)) {
                lanes += Components[i];
            }
        }
        if (!lanes.empty()) {
            Emit("        {}.{} = ({}).{};\n", *reg, lanes, value, lanes);
        }
    }

    void EmitArithmetic(Instruction instruction, u32 pc) {
        const OpCode opcode = instruction.opcode.Value();
        const auto id = opcode.EffectiveOpCode();
        const SwizzlePattern swizzle{swizzles[instruction.common.operand_desc_id]};
        const bool inverted = (opcode.GetInfo().subtype & OpCode::Info::SrcInversed) != 0;
        const u32 address = instruction.common.address_register_index;
        const auto source1 =
            Operand(instruction.common.GetSrc1(inverted), inverted ? 0 : address, swizzle, 0);
        Emit("        const float4 src1 = {};\n", source1);
        if (opcode.GetInfo().subtype & OpCode::Info::Src2) {
            const auto source2 =
                Operand(instruction.common.GetSrc2(inverted), inverted ? address : 0, swizzle, 1);
            Emit("        const float4 src2 = {};\n", source2);
        }

        std::string_view value;
        switch (id) {
        case OpCode::Id::ADD:
            value = "src1 + src2";
            break;
        case OpCode::Id::MUL:
            value = "Multiply(src1, src2)";
            break;
        case OpCode::Id::FLR:
            value = "floor(src1)";
            break;
        case OpCode::Id::MAX:
            value = "select(src2, src1, src1 > src2)";
            break;
        case OpCode::Id::MIN:
            value = "select(src2, src1, src1 < src2)";
            break;
        case OpCode::Id::DP3:
            value = "float4(OrderedDot(src1, src2, false, false))";
            break;
        case OpCode::Id::DP4:
            value = "float4(OrderedDot(src1, src2, true, false))";
            break;
        case OpCode::Id::DPH:
        case OpCode::Id::DPHI:
            value = "float4(OrderedDot(src1, src2, true, true))";
            break;
        case OpCode::Id::RCP:
            value = "float4(1.0f / src1.x)";
            break;
        case OpCode::Id::RSQ:
            value = "float4(1.0f / sqrt(src1.x))";
            break;
        case OpCode::Id::EX2:
            value = "float4(exp2(src1.x))";
            break;
        case OpCode::Id::LG2:
            value = "float4(log2(src1.x))";
            break;
        case OpCode::Id::MOV:
            value = "src1";
            break;
        case OpCode::Id::SGE:
        case OpCode::Id::SGEI:
            value = "select(float4(0.0f), float4(1.0f), src1 >= src2)";
            break;
        case OpCode::Id::SLT:
        case OpCode::Id::SLTI:
            value = "select(float4(0.0f), float4(1.0f), src1 < src2)";
            break;
        case OpCode::Id::MOVA:
            for (u32 i = 0; i < 2; ++i) {
                if (swizzle.DestComponentEnabled(i)) {
                    Emit("        registers.address.{0} = ConvertAddress(src1.{0});\n",
                         Components[i]);
                }
            }
            return;
        case OpCode::Id::CMP: {
            const std::array operations{instruction.common.compare_op.x.Value(),
                                        instruction.common.compare_op.y.Value()};
            for (u32 i = 0; i < operations.size(); ++i) {
                const auto operation = Comparison(operations[i]);
                if (operation.empty()) {
                    Diagnose(DecompileIssue::InvalidComparison, pc, instruction);
                    continue;
                }
                Emit("        registers.condition.{0} = uint(src1.{0} {1} src2.{0});\n",
                     Components[i], operation);
            }
            return;
        }
        default:
            Diagnose(DecompileIssue::UnsupportedInstruction, pc, instruction);
            return;
        }
        WriteDestination(instruction.common.dest.Value(), swizzle, value);
    }

    void EmitMultiplyAdd(Instruction instruction, u32 pc) {
        const auto id = instruction.opcode.Value().EffectiveOpCode();
        const bool inverted = id == OpCode::Id::MADI;
        if (id != OpCode::Id::MAD && !inverted) {
            Diagnose(DecompileIssue::UnsupportedInstruction, pc, instruction);
            return;
        }
        const SwizzlePattern swizzle{swizzles[instruction.mad.operand_desc_id]};
        const u32 address = instruction.mad.address_register_index;
        Emit("        const float4 src1 = {};\n",
             Operand(instruction.mad.GetSrc1(inverted), 0, swizzle, 0));
        Emit("        const float4 src2 = {};\n",
             Operand(instruction.mad.GetSrc2(inverted), inverted ? 0 : address, swizzle, 1));
        Emit("        const float4 src3 = {};\n",
             Operand(instruction.mad.GetSrc3(inverted), inverted ? address : 0, swizzle, 2));
        Emit("        const float4 product = Multiply(src1, src2);\n");
        WriteDestination(instruction.mad.dest.Value(), swizzle, "product + src3");
    }

    void EmitGeometry(Instruction instruction, u32 pc) {
        const auto opcode = instruction.opcode.Value().EffectiveOpCode();
        if (options.kind != ProgramKind::Geometry ||
            (opcode == OpCode::Id::SETEMIT && instruction.setemit.vertex_id >= 3)) {
            Diagnose(DecompileIssue::InvalidGeometryOperation, pc, instruction);
            Emit("        return RunStatus::InvalidGeometryOperation;\n");
            return;
        }
        if (opcode == OpCode::Id::SETEMIT) {
            Emit("        registers.emission = uint4({}u, {}u, {}u, 0u);\n",
                 instruction.setemit.vertex_id.Value(), instruction.setemit.prim_emit.Value(),
                 instruction.setemit.winding.Value());
        } else {
            Emit(R"MSL(
        if (registers.emission.x >= 3u) {{
            return RunStatus::InvalidGeometryOperation;
        }}
        if (!emitter.Emit(registers.output[registers.condition.z & 1u], registers.emission.x,
                          registers.emission.y != 0u, registers.emission.z != 0u)) {{
            return RunStatus::EmitOverflow;
        }}
        registers.condition.z ^= 1u;
)MSL");
        }
    }

    void EmitInstruction(Instruction instruction, u32 pc) {
        const OpCode opcode = instruction.opcode.Value();
        if (opcode.GetInfo().type == OpCode::Type::Arithmetic) {
            EmitArithmetic(instruction, pc);
            return;
        }
        if (opcode.GetInfo().type == OpCode::Type::MultiplyAdd) {
            EmitMultiplyAdd(instruction, pc);
            return;
        }

        const auto flow = instruction.flow_control;
        const u32 target = flow.dest_offset;
        const u32 end = target + flow.num_instructions;
        const auto id = opcode.EffectiveOpCode();
        switch (id) {
        case OpCode::Id::END:
            Emit("        done = true;\n");
            break;
        case OpCode::Id::NOP:
            break;
        case OpCode::Id::JMPC:
        case OpCode::Id::JMPU:
            Emit("        if ({}) {{ pc = {}u; }}\n",
                 id == OpCode::Id::JMPC
                     ? Condition(flow)
                     : BooleanCondition(flow.bool_uniform_id, (flow.num_instructions & 1) != 0),
                 target);
            break;
        case OpCode::Id::CALL:
        case OpCode::Id::CALLC:
        case OpCode::Id::CALLU: {
            const auto condition = id == OpCode::Id::CALLC ? Condition(flow)
                                   : id == OpCode::Id::CALLU
                                       ? BooleanCondition(flow.bool_uniform_id)
                                       : "true";
            Emit("        if ({}) {{ Push(calls, call_depth, CallScope{{{}u, {}u}}); pc = {}u; "
                 "}}\n",
                 condition, end, pc + 1, target);
            break;
        }
        case OpCode::Id::IFC:
        case OpCode::Id::IFU:
            Emit("        if ({}) {{ Push(ifs, if_depth, IfScope{{{}u, {}u}}); }}"
                 " else {{ pc = {}u; }}\n",
                 id == OpCode::Id::IFC ? Condition(flow) : BooleanCondition(flow.bool_uniform_id),
                 target, end, target);
            break;
        case OpCode::Id::LOOP:
            // The interpreter assigns aL before saving the value for the inner scope.
            Emit(
                "        const uint4 args = uniforms.i[{}] & uint4(255u);\n"
                "        registers.address.z = int(args.y);\n"
                "        Push(loops, loop_depth, LoopScope{{{}u, {}u, args.x, args.z, args.y}});\n",
                flow.int_uniform_id.Value(), pc + 1, target + 1);
            break;
        case OpCode::Id::BREAK:
            Emit("        requested_break = true;\n");
            break;
        case OpCode::Id::BREAKC:
            Emit("        requested_break = {};\n", Condition(flow));
            break;
        case OpCode::Id::EMIT:
        case OpCode::Id::SETEMIT:
            EmitGeometry(instruction, pc);
            break;
        default:
            Diagnose(DecompileIssue::UnsupportedInstruction, pc, instruction);
            break;
        }
    }

    void EmitControlFlow(u32 entry, const ProgramLayout& layout) {
        Emit(R"MSL(
    CallScope calls[4];
    IfScope ifs[8];
    LoopScope loops[4];
    uint call_depth = 0u;
    uint if_depth = 0u;
    uint loop_depth = 0u;
    uint backward_transfers = 0u;
    uint pc = {}u;
    while (true) {{
        const uint old_pc = pc++;
        bool requested_break = false;
        bool done = old_pc >= {}u;
        if (!done) {{
            switch (old_pc) {{
)MSL",
             entry, ImplicitEnd);

        for (u32 address = 0; address < ImplicitEnd; ++address) {
            if (!layout.reachable.test(address)) {
                continue;
            }
            Emit("    case {}u: {{\n", address);
            EmitInstruction(ReadInstruction(code, address), address);
            Emit("        break;\n    }}\n");
        }
        Emit(R"MSL(
    default:
        return RunStatus::InvalidProgramCounter;
            }}
        }}

        uint next_pc = old_pc + 1u;
        for (uint i = 0u; i < 4u && call_depth; ++i) {{
            const CallScope call = calls[call_depth - 1u];
            if (call.end != next_pc) {{
                break;
            }}
            if (i < 3u) {{
                pc = call.resume;
                next_pc = pc;
            }}
            --call_depth;
        }}
        if (if_depth && ifs[if_depth - 1u].else_address == old_pc + 1u) {{
            pc = ifs[--if_depth].end;
        }}

        bool counted_repeat = false;
        if (loop_depth && (loops[loop_depth - 1u].end == old_pc + 1u || requested_break)) {{
            thread LoopScope& loop = loops[loop_depth - 1u];
            registers.address.z += int(loop.increment);
            if (!requested_break && loop.remaining) {{
                --loop.remaining;
                pc = loop.entry;
                counted_repeat = true;
            }} else {{
                pc = loop.end;
                if (loop_depth > 1u) {{
                    registers.address.z = int(loop.previous_address);
                }}
                --loop_depth;
            }}
        }}
        if (done) {{
            return RunStatus::Complete;
        }}
        // Count only unstructured backward transfers. Hardware-counted LOOPs keep their full range.
        if (!counted_repeat && pc <= old_pc && ++backward_transfers == (1u << 20u)) {{
            return RunStatus::NonTerminatingControlFlow;
        }}
    }}
)MSL");
    }

    const ProgramCode& code;
    const SwizzleData& swizzles;
    ProgramOptions options;
    DecompiledProgram result;
};

} // Anonymous namespace

std::string_view DecompilerPreamble() noexcept {
    return Preamble;
}

DecompiledProgram DecompileProgram(const ProgramCode& code, const SwizzleData& swizzles, u32 entry,
                                   ProgramOptions options) {
    if (entry >= MAX_PROGRAM_CODE_LENGTH) {
        return {{}, {{DecompileIssue::InvalidEntry, entry, 0}}};
    }
    if (!IsFunctionName(options.function_name)) {
        return {{}, {{DecompileIssue::InvalidFunctionName, entry, code[entry]}}};
    }
    if (options.kind != ProgramKind::Vertex && options.kind != ProgramKind::Geometry) {
        return {{}, {{DecompileIssue::InvalidProgramKind, entry, code[entry]}}};
    }
    if (options.uniform_storage != UniformStorage::Constant &&
        options.uniform_storage != UniformStorage::Thread) {
        return {{}, {{DecompileIssue::InvalidUniformStorage, entry, code[entry]}}};
    }
    return Emitter(code, swizzles, options).Generate(entry);
}

} // namespace Pica::Shader::Generator::MSL
