#include <array>
#include <cassert>
#include <cstdio>
#include "video_core/renderer_metal/mtl_shader_util.h"

using namespace Metal;
using namespace std::chrono_literals;

#if defined(CHECK_SDK_CONSUMER)
bool CompileExample(DeviceContext& context, std::span<const u8> embedded) {
    ShaderCompiler compiler(context, 2);
    const auto compiled = compiler.CompileLibrary({
        .text = "#include <metal_stdlib>\nkernel void example() {}",
        .name = "native-stage-consumer",
        .preamble = "#define KNOWN_NATIVE_CONSUMER 1",
        .options = {.definitions = {{"SECOND_FLAG", "1"}}},
    });
    if (compiled.GetStatus(context) != ShaderCompileStatus::Success) {
        return false;
    }
    const auto loaded = compiler.LoadLibrary(embedded, "embedded consumer");
    if (!loaded.GetLibrary(context)) {
        return false;
    }
    const auto task = compiler.QueueLibrary({.text = "kernel void async_consumer() {}"});
    const auto result = task.WaitUntil(std::chrono::steady_clock::now() + 10s);
    return result && result->GetLibrary(context);
}

id<MTL4CompilerTask> BuildSDKContract(id<MTL4Compiler> compiler) {
    auto options = [MTLCompileOptions new];
    options.languageVersion = MTLLanguageVersion4_0;
    options.mathMode = MTLMathModeSafe;
    options.mathFloatingPointFunctions = MTLMathFloatingPointFunctionsPrecise;
    options.optimizationLevel = MTLLibraryOptimizationLevelSize;
    options.preserveInvariance = YES;
    options.preprocessorMacros = @{@"FLAG" : @"1"};
    auto descriptor = [MTL4LibraryDescriptor new];
    descriptor.source = @"kernel void foo() {}";
    descriptor.options = options;
    return [compiler newLibraryWithDescriptor:descriptor
                            completionHandler:^(id<MTLLibrary>, NSError*){
                            }];
}
#else
int main() {
    static_assert(IsShaderIdentifier("x"));
    static_assert(IsShaderIdentifier("_Work2"));
    static_assert(!IsShaderIdentifier(""));
    static_assert(!IsShaderIdentifier("1x"));
    static_assert(!IsShaderIdentifier("x'"));
    static_assert(!IsShaderIdentifier(std::string_view("x\0y", 3)));
    static_assert(!HasMetalLibrarySignature({}));
    static_assert(!HasMetalLibrarySignature(std::array<u8, 4>{'M', 'T', 'L', 'B'}));
    static_assert(!HasMetalLibrarySignature(std::array<u8, 6>{0, 'T', 'L', 'B', 0, 0}));
    static_assert(HasMetalLibrarySignature(std::array<u8, 5>{'M', 'T', 'L', 'B', 0}));
    ShaderSource first{.text = "kernel void main0() {}", .name = "shader"};
    first.options.definitions.emplace("B", "2");
    first.options.definitions.emplace("A", "1");
    auto reordered = first;
    reordered.options.definitions.clear();
    reordered.options.definitions.emplace("A", "1");
    reordered.options.definitions.emplace("B", "2");
    assert(first == reordered);
    auto different = first;
    different.options.preserve_invariance = false;
    assert(first != different);
    different = first;
    different.options.optimization = ShaderOptimization::Size;
    assert(first != different);
    different = first;
    different.options.math = ShaderMath::Fast;
    assert(first != different);
    different = first;
    different.options.definitions["A"] = "9";
    assert(first != different);
    different = first;
    different.preamble = "// exact new preamble";
    assert(first != different);
    different = first;
    different.text += "\n";
    assert(first != different);
    std::puts(
        "PASS: native compiler API, binary signature, macro identifier and complete option key");
}
#endif
