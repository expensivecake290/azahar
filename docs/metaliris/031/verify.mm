#import <Foundation/Foundation.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <optional>
#include <string>
#include <vector>
#include <nihstro/shader_bytecode.h>
#include "video_core/pica/shader_unit.h"
#include "video_core/renderer_metal/mtl_scheduler.h"
#include "video_core/renderer_metal/shader/msl_shader_decompiler.h"
#include "video_core/shader/shader_interpreter.h"

using namespace Pica::Shader::Generator::MSL;
using nihstro::DestRegister;
using nihstro::Instruction;
using nihstro::OpCode;
using nihstro::SourceRegister;
using nihstro::SwizzlePattern;
using Pica::Shader::Generator::VSPicaUniformData;
using namespace Metal;
using namespace std::chrono_literals;

namespace {

Instruction Op(OpCode::Id opcode) {
    Instruction result{};
    result.opcode.Assign(opcode);
    return result;
}

Instruction ALU(OpCode::Id opcode, u32 destination, u32 a, u32 b = 0, u32 swizzle = 0,
                u32 address = 0) {
    auto result = Op(opcode);
    result.common.dest = DestRegister(destination);
    if (OpCode(opcode).GetInfo().subtype & OpCode::Info::SrcInversed) {
        result.common.src1i = SourceRegister(a);
        result.common.src2i = SourceRegister(b);
    } else {
        result.common.src1 = SourceRegister(a);
        result.common.src2 = SourceRegister(b);
    }
    result.common.operand_desc_id = swizzle;
    result.common.address_register_index = address;
    return result;
}

Instruction Mad(OpCode::Id opcode, u32 destination, u32 a, u32 b, u32 c, u32 swizzle) {
    auto result = Op(opcode);
    result.mad.dest = DestRegister(destination);
    result.mad.src1 = SourceRegister(a);
    if (opcode == OpCode::Id::MADI) {
        result.mad.src2i = SourceRegister(b);
        result.mad.src3i = SourceRegister(c);
    } else {
        result.mad.src2 = SourceRegister(b);
        result.mad.src3 = SourceRegister(c);
    }
    result.mad.operand_desc_id = swizzle;
    return result;
}

Instruction Flow(OpCode::Id opcode, u32 target, u32 length = 0, u32 uniform = 0) {
    auto result = Op(opcode);
    result.flow_control.dest_offset = target;
    result.flow_control.num_instructions = length;
    if (opcode == OpCode::Id::LOOP) {
        result.flow_control.int_uniform_id = uniform;
    } else if (opcode == OpCode::Id::CALLU || opcode == OpCode::Id::IFU ||
               opcode == OpCode::Id::JMPU) {
        result.flow_control.bool_uniform_id = uniform;
    } else {
        result.flow_control.op = Instruction::FlowControlType::Op::JustX;
        result.flow_control.refx = 1;
    }
    return result;
}

Instruction EmitVertex(u32 slot, bool emit = false, bool winding = false) {
    auto result = Op(OpCode::Id::SETEMIT);
    result.setemit.vertex_id = slot;
    result.setemit.prim_emit = emit;
    result.setemit.winding = winding;
    return result;
}

struct Fixture {
    explicit Fixture(std::string name) : name(std::move(name)) {
        SwizzlePattern identity{};
        identity.dest_mask = 15;
        for (u32 i = 0; i < 4; ++i) {
            const auto lane = static_cast<SwizzlePattern::Selector>(i);
            identity.SetSelectorSrc1(i, lane);
            identity.SetSelectorSrc2(i, lane);
            identity.SetSelectorSrc3(i, lane);
        }
        swizzles[0] = identity.hex;
        auto masked = identity;
        masked.dest_mask = 10;
        masked.negate_src1 = 1;
        for (u32 i = 0; i < 4; ++i) {
            masked.SetSelectorSrc1(i, static_cast<SwizzlePattern::Selector>(3 - i));
        }
        swizzles[1] = masked.hex;
        auto negate_last = identity;
        negate_last.negate_src3 = 1;
        swizzles[2] = negate_last.hex;
    }

    void Set(std::initializer_list<Instruction> instructions, u32 start = 0) {
        for (const auto instruction : instructions) {
            code[start++] = instruction.hex;
        }
    }

    std::string name;
    Pica::ProgramCode code{};
    Pica::SwizzleData swizzles{};
    Pica::Uniforms uniforms{};
    RegisterState initial{};
    u32 entry{};
    ProgramKind kind{ProgramKind::Vertex};
    RunStatus expected_status{RunStatus::Complete};
};

std::vector<Fixture> Corpus() {
    using enum OpCode::Id;
    Fixture arithmetic("Arithmetic");
    arithmetic.initial.input[0] = {4, 2, 8, 16};
    arithmetic.initial.input[1] = {2, 4, 16, 8};
    arithmetic.initial.input[2] = {std::numeric_limits<f32>::infinity(),
                                   std::numeric_limits<f32>::quiet_NaN(), 0, -0.0f};
    arithmetic.initial.input[3] = {0, 0, 0, 0};
    arithmetic.initial.input[4] = {1.9f, 128.0f, 0, 0};
    arithmetic.initial.output[0][15] = {99, 99, 99, 99};
    arithmetic.initial.condition[0] = 1;
    arithmetic.uniforms.f[95] = {Pica::f24::FromFloat32(2), Pica::f24::FromFloat32(3),
                                 Pica::f24::FromFloat32(4), Pica::f24::FromFloat32(5)};
    auto compare = ALU(CMP, 0, 0, 1);
    using Compare = Instruction::Common::CompareOpType;
    compare.common.compare_op.x = Compare::Unk6;
    compare.common.compare_op.y = Compare::GreaterThan;
    arithmetic.Set({
        ALU(ADD, 0, 0, 1),
        ALU(MUL, 1, 2, 3),
        ALU(FLR, 2, 4),
        ALU(MAX, 3, 3, 2),
        ALU(MIN, 4, 2, 3),
        ALU(DP3, 5, 0, 1),
        ALU(DP4, 6, 0, 1),
        ALU(DPH, 7, 0, 1),
        ALU(DPHI, 8, 1, 32 + 95),
        ALU(RCP, 9, 0),
        ALU(RSQ, 10, 0),
        ALU(EX2, 11, 0),
        ALU(LG2, 12, 0),
        ALU(SGE, 13, 0, 1),
        ALU(SGEI, 14, 0, 32 + 95),
        ALU(MOV, 15, 0, 0, 1),
        ALU(SLT, 0x10, 0, 1),
        ALU(SLTI, 0x11, 0, 32 + 95),
        ALU(MOVA, 0, 4),
        ALU(MOV, 0x12, 32 + 95, 0, 0, 1),
        ALU(MOV, 0x13, 32 + 95, 0, 0, 2),
        Mad(MAD, 0x14, 0, 1, 4, 2),
        Mad(MADI, 0x15, 0, 1, 32 + 95, 2),
        compare,
        Instruction{0x11u << 26},
        Op(END),
    });

    Fixture scope("NestedScopes");
    scope.initial.input[0] = {4, 2, 8, 16};
    scope.initial.input[1] = {2, 4, 16, 8};
    scope.initial.input[2] = {3, 6, 9, 12};
    scope.initial.condition[0] = 1;
    scope.uniforms.b[0] = true;
    scope.uniforms.i[0] = {1, 5, 2, 0};
    scope.uniforms.i[1] = {2, 9, 1, 0};
    auto flags = ALU(CMP, 0, 1, 1);
    flags.common.compare_op.x = Compare::Equal;
    flags.common.compare_op.y = Compare::Equal;
    scope.Set({ALU(MOV, 0, 0), Flow(CALLU, 12, 2), flags, Flow(IFC, 5, 1), ALU(ADD, 1, 0, 1),
               ALU(MOV, 1, 1), Flow(LOOP, 10, 0, 0), Flow(LOOP, 8, 0, 1), ALU(ADD, 0x10, 0x10, 0),
               Flow(BREAKC, 0), Op(NOP), Op(END), ALU(MOV, 2, 1), Flow(CALLC, 16, 1), Op(END),
               Op(NOP), ALU(MOV, 3, 2), Op(END)});

    Fixture branch("Branches");
    branch.uniforms.b[0] = true;
    branch.initial.input[0] = {7, 11, 13, 17};
    branch.initial.input[1] = {2, 3, 5, 7};
    branch.initial.condition[0] = 1;
    branch.uniforms.i[0] = {4, 9, 2, 0};
    branch.Set({Flow(JMPU, 2), Op(END), Flow(JMPC, 4), Op(END), Flow(IFU, 6, 1, 1), ALU(MOV, 1, 0),
                ALU(MOV, 1, 1), Flow(CALL, 12, 1), Flow(LOOP, 10), Op(BREAK), ALU(MOV, 2, 1),
                Op(END), ALU(MOV, 0, 0), Op(END)});

    Fixture geometry("GeometryBanks");
    geometry.kind = ProgramKind::Geometry;
    geometry.initial.input[0] = {10, 20, 30, 40};
    geometry.initial.input[1] = {11, 21, 31, 41};
    geometry.Set({EmitVertex(0), ALU(MOV, 0, 0), Op(EMIT), EmitVertex(1), ALU(MOV, 0, 1), Op(EMIT),
                  EmitVertex(2, true, true), Op(EMIT), Op(END)});

    Fixture implicit("ImplicitEnd");
    implicit.entry = Pica::MAX_PROGRAM_CODE_LENGTH - 2;
    implicit.initial.input[0] = {3, 5, 7, 11};
    implicit.Set({ALU(MOV, 0, 0), Op(EMIT)}, implicit.entry);

    Fixture malformed("InvalidGeometry");
    malformed.Set({Op(EMIT), Op(END)});
    malformed.expected_status = RunStatus::InvalidGeometryOperation;
    Fixture runaway("RunawayControl");
    runaway.Set({Flow(JMPU, 0, 1), Op(END)});
    runaway.expected_status = RunStatus::NonTerminatingControlFlow;
    Fixture overflow = geometry;
    overflow.name = "GeometryOverflow";
    overflow.Set({Op(EMIT), Op(END)}, 8);
    overflow.expected_status = RunStatus::EmitOverflow;
    return {std::move(arithmetic), std::move(scope),     std::move(branch),  std::move(geometry),
            std::move(implicit),   std::move(malformed), std::move(runaway), std::move(overflow)};
}

struct Reference {
    RegisterState regs{};
    std::vector<RegisterState::Vector> emitted;
    bool winding{};
};

Reference Interpret(const Fixture& fixture) {
    Reference result;
    Pica::ShaderSetup setup;
    setup.UpdateProgramCode(fixture.code);
    setup.UpdateSwizzleData(fixture.swizzles);
    setup.uniforms = fixture.uniforms;
    Pica::GeometryEmitter emitter{};
    Pica::Handlers handlers{};
    if (fixture.kind == ProgramKind::Geometry) {
        emitter.output_mask = 1;
        emitter.handlers = &handlers;
        handlers.winding_setter = [&] { result.winding = true; };
        handlers.vertex_handler = [&](const Pica::AttributeBuffer& output) {
            const auto& value = output[0];
            result.emitted.push_back({value.x.ToFloat32(), value.y.ToFloat32(), value.z.ToFloat32(),
                                      value.w.ToFloat32()});
        };
    }
    Pica::ShaderUnit state(fixture.kind == ProgramKind::Geometry ? &emitter : nullptr);
    auto import = [](auto& dest, const RegisterState::RegisterBank& source) {
        for (std::size_t reg = 0; reg < dest.size(); ++reg) {
            for (u32 lane = 0; lane < 4; ++lane) {
                dest[reg][lane] = Pica::f24::FromFloat32(source[reg][lane]);
            }
        }
    };
    import(state.input, fixture.initial.input);
    import(state.temporary, fixture.initial.temporary);
    import(state.output[0], fixture.initial.output[0]);
    import(state.output[1], fixture.initial.output[1]);
    for (u32 i = 0; i < 2; ++i) {
        state.conditional_code[i] = fixture.initial.condition[i] != 0;
    }
    for (u32 i = 0; i < 3; ++i) {
        state.address_registers[i] = fixture.initial.address[i];
    }
    Pica::Shader::InterpreterEngine engine;
    engine.SetupBatch(setup, fixture.entry);
    engine.Run(setup, state);
    auto export_bank = [](const auto& src, RegisterState::RegisterBank& dest) {
        for (std::size_t reg = 0; reg < dest.size(); ++reg) {
            for (u32 lane = 0; lane < 4; ++lane) {
                dest[reg][lane] = src[reg][lane].ToFloat32();
            }
        }
    };
    export_bank(state.input, result.regs.input);
    export_bank(state.temporary, result.regs.temporary);
    export_bank(state.output[0], result.regs.output[0]);
    export_bank(state.output[1], result.regs.output[1]);
    result.regs.condition[0] = state.conditional_code[0];
    result.regs.condition[1] = state.conditional_code[1];
    result.regs.condition[2] = state.output_bank;
    std::copy_n(state.address_registers, 3, result.regs.address.begin());
    return result;
}

void Generate(const std::vector<Fixture>& corpus, const char* file) {
    std::ofstream out(file);
    assert(out);
    out << DecompilerPreamble();
    for (const auto& test : corpus) {
        const auto generated =
            DecompileProgram(test.code, test.swizzles, test.entry, {test.kind, test.name});
        assert(generated && generated.source.find(test.name) != std::string::npos);
        if (test.name == "Arithmetic") {
            const auto ref = Interpret(test);
            assert(ref.regs.output[0][0][0] == 6.0f);
            assert(ref.regs.output[0][1][0] == 0.0f && std::isnan(ref.regs.output[0][1][1]));
            assert(ref.regs.output[0][15][0] == -16.0f && ref.regs.output[0][15][1] == 99.0f);
            assert(ref.regs.temporary[2][0] == 1.0f && ref.regs.temporary[3][0] == 2.0f);
            assert(ref.regs.condition[0] == 1 && ref.regs.condition[1] == 0);
            assert(generated.diagnostics.size() == 2);
            assert(generated.diagnostics[0].issue == DecompileIssue::InvalidComparison);
            assert(generated.diagnostics[1].issue == DecompileIssue::UnsupportedInstruction);
        } else if (test.name == "NestedScopes") {
            const auto ref = Interpret(test);
            assert(ref.regs.output[0][1][0] == 6.0f);
            assert(ref.regs.output[0][3][3] == 12.0f);
            assert(ref.regs.temporary[0][0] == 12.0f);
            assert(ref.regs.address[2] == 11);
        } else if (test.name == "Branches") {
            const auto ref = Interpret(test);
            assert(ref.regs.output[0][0][0] == 7.0f && ref.regs.output[0][1][0] == 2.0f);
            assert(ref.regs.output[0][2][0] == 0.0f && ref.regs.address[2] == 11);
        } else if (test.name == "GeometryBanks") {
            const auto ref = Interpret(test);
            assert(ref.regs.condition[2] == 1 && ref.emitted.size() == 3 && ref.winding);
            assert(ref.emitted[0][0] == 10 && ref.emitted[1][0] == 11 && ref.emitted[2][0] == 10);
        } else if (test.name == "ImplicitEnd") {
            assert(Interpret(test).regs.output[0][0][0] == 3.0f);
        } else if (test.name == "InvalidGeometry") {
            assert(generated.diagnostics.size() == 1 &&
                   generated.diagnostics.front().issue == DecompileIssue::InvalidGeometryOperation);
        }
        out << generated.source;
    }
    out << R"MSL(
using namespace metal;
using namespace MetalirisPica;
struct TestEmitter {
    float4 vertices[3];
    uint primitive;
    uint winding;
    uint count;
    bool Emit(thread const float4* output, uint slot, bool emit, bool reverse) {
        if (count == 3u) {
            return false;
        }
        ++count;
        vertices[slot] = output[0];
        primitive += uint(emit);
        winding += uint(reverse);
        return true;
    }
};
)MSL";
    for (const auto& test : corpus) {
        out << "kernel void test_" << test.name
            << "(device RegisterState* regs [[buffer(0)]], "
               "constant Uniforms& uniforms [[buffer(1)]], device uint4* result "
               "[[buffer(2)]], "
               "device float4* vertices [[buffer(3)]]) {\n"
               "    RegisterState local = regs[0];\n";
        if (test.kind == ProgramKind::Geometry) {
            out << "    TestEmitter emitter{};\n"
                   "    RunStatus status = "
                << test.name
                << "(local, uniforms, emitter);\n"
                   "    for (uint i = 0u; i < 3u; ++i) vertices[i] = "
                   "emitter.vertices[i];\n"
                   "    result[0] = uint4(uint(status), emitter.primitive, "
                   "emitter.winding, 0u);\n";
        } else {
            out << "    result[0] = uint4(uint(" << test.name
                << "(local, uniforms)), 0u, 0u, 0u);\n";
        }
        out << "    regs[0] = local;\n}\n";
    }
    assert(out.good());
    const auto& test = corpus.front();
    assert(!DecompileProgram(test.code, test.swizzles, Pica::MAX_PROGRAM_CODE_LENGTH));
    assert(!DecompileProgram(test.code, test.swizzles, 0, {ProgramKind::Vertex, "bad name"}));
    std::puts("PASS direct emission, precise PICA register oracle and "
              "diagnostics for eight programs");
}

template <typename T>
id<MTLBuffer> Shared(id<MTLDevice> device, const T& value) {
    return CheckResource([device newBufferWithBytes:&value
                                             length:sizeof(value)
                                            options:MTLResourceStorageModeShared |
                                                    MTLResourceHazardTrackingModeUntracked],
                         "Create decompiler verification storage");
}

void CompareGPU(const Fixture& fixture, id<MTLDevice> device, id<MTLLibrary> library) {
    NSError* error = nil;
    auto compiler = [device newCompilerWithDescriptor:[MTL4CompilerDescriptor new] error:&error];
    compiler = CheckResource(compiler, "Create decompiler verification compiler", error);
    auto function = [MTL4LibraryFunctionDescriptor new];
    function.library = library;
    function.name = [NSString stringWithFormat:@"test_%s", fixture.name.c_str()];
    auto descriptor = [MTL4ComputePipelineDescriptor new];
    descriptor.computeFunctionDescriptor = function;
    auto pipeline = [compiler newComputePipelineStateWithDescriptor:descriptor
                                                compilerTaskOptions:nil
                                                              error:&error];
    pipeline = CheckResource(pipeline, "Create decompiler verification pipeline", error);

    Pica::ShaderSetup setup;
    setup.uniforms = fixture.uniforms;
    VSPicaUniformData uniforms{};
    uniforms.SetFromRegs(setup);
    auto registers = Shared(device, fixture.initial);
    auto uniform_buffer = Shared(device, uniforms);
    auto status = Shared(device, std::array<u32, 4>{});
    auto emitter = Shared(device, std::array<RegisterState::Vector, 3>{});
    DeviceContext context(device);
    Scheduler scheduler(context);
    scheduler.Record([=](EncodingContext& ctx) {
        auto pipeline_arguments = [MTL4ArgumentTableDescriptor new];
        pipeline_arguments.maxBufferBindCount = 4;
        NSError* error = nil;
        auto table = [device newArgumentTableWithDescriptor:pipeline_arguments error:&error];
        table = CheckResource(table, "Create decompiler verification arguments", error);
        const std::array uses{
            ResourceUse{registers, BufferAccess::ReadWrite, MTLStageDispatch},
            ResourceUse{uniform_buffer, BufferAccess::Read, MTLStageDispatch},
            ResourceUse{status, BufferAccess::Write, MTLStageDispatch},
            ResourceUse{emitter, BufferAccess::Write, MTLStageDispatch},
        };
        auto encoder = [ctx.GetCommandBuffer() computeCommandEncoder];
        ctx.Synchronize(encoder, uses);
        for (std::size_t i = 0; i < uses.size(); ++i) {
            id<MTLBuffer> buffer = static_cast<id<MTLBuffer>>(uses[i].resource);
            [table setAddress:buffer.gpuAddress atIndex:i];
        }
        [encoder setComputePipelineState:pipeline];
        [encoder setArgumentTable:table];
        [encoder dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
        ctx.Retain(table);
        ctx.Retain(pipeline);
        ctx.EndPass(encoder);
    });
    assert(scheduler.Finish({}, std::chrono::steady_clock::now() + 10s) ==
           CompletionStatus::Complete);
    const auto* bits = static_cast<const u32*>(status.contents);
    assert(bits[0] == static_cast<u32>(fixture.expected_status));
    if (fixture.expected_status != RunStatus::Complete) {
        return;
    }
    const auto expected = Interpret(fixture);
    const auto& actual = *static_cast<const RegisterState*>(registers.contents);
    const auto compare = [](const auto& a, const auto& b) {
        for (std::size_t reg = 0; reg < a.size(); ++reg) {
            for (u32 i = 0; i < 4; ++i) {
                assert(std::isnan(a[reg][i])
                           ? std::isnan(b[reg][i])
                           : std::bit_cast<u32>(a[reg][i]) == std::bit_cast<u32>(b[reg][i]));
            }
        }
    };
    compare(expected.regs.output[0], actual.output[0]);
    compare(expected.regs.output[1], actual.output[1]);
    compare(expected.regs.temporary, actual.temporary);
    assert(expected.regs.address == actual.address);
    assert(expected.regs.condition == actual.condition);
    if (fixture.kind == ProgramKind::Geometry) {
        const auto& vertices =
            *static_cast<const std::array<RegisterState::Vector, 3>*>(emitter.contents);
        compare(expected.emitted, vertices);
        assert(bits[1] == 1 && bits[2] == 1);
    }
}

} // namespace

int main(int argc, char** argv) {
    @autoreleasepool {
        assert(argc == 3);
        const auto corpus = Corpus();
        if (std::string_view(argv[1]) == "generate") {
            Generate(corpus, argv[2]);
            return 0;
        }
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device || ![device supportsFamily:MTLGPUFamilyMetal4]) {
            std::puts("PENDING native PICA arithmetic, control flow and "
                      "geometry-bank differential");
            return 2;
        }
        NSError* error = nil;
        auto library = [device newLibraryWithURL:[NSURL fileURLWithPath:@(argv[2])] error:&error];
        library = CheckResource(library, "Load generated Metal decompiler oracle", error);
        for (const auto& test : corpus) {
            CompareGPU(test, device, library);
        }
        std::puts("PASS live PICA interpreter versus emitted Metal 4 functions");
    }
}
