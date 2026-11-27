#include <array>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <deque>
#include <fstream>
#include <future>
#include <mutex>
#include <vector>
#include "video_core/renderer_metal/mtl_scheduler.h"
#include "video_core/renderer_metal/mtl_shader_util.h"

using namespace Metal;
using namespace std::chrono_literals;

@class ControlledCompiler;

@interface ControlledDevice : NSObject
@property(nonatomic, readonly) ControlledCompiler* compiler;
@property(nonatomic) id<MTL4PipelineDataSetSerializer> recordedSerializer;
@end

@interface ControlledLibrary : NSObject
@property(nonatomic) id<MTLDevice> device;
@property(nonatomic, copy) NSString* label;
@end

@implementation ControlledLibrary
@end

struct PendingRequest {
    MTL4LibraryDescriptor* descriptor;
    MTLNewLibraryCompletionHandler completion;
};

@interface ControlledCompiler : NSObject {
@public
    std::deque<PendingRequest> pending;
    std::mutex mutex;
    std::size_t asynchronous;
    std::size_t synchronous;
    bool inlineCompletion;
}
@property(nonatomic, weak) ControlledDevice* owner;
@property(nonatomic) MTL4LibraryDescriptor* lastDescriptor;
- (void)completeNext;
@end

@implementation ControlledCompiler

- (id<MTLLibrary>)makeLibrary:(MTL4LibraryDescriptor*)descriptor error:(NSError**)error {
    const bool fail = [descriptor.name isEqualToString:@"fail"];
    const bool warn = [descriptor.name isEqualToString:@"warn"];
    if (fail || warn) {
        *error =
            [NSError errorWithDomain:@"Fixture.Compiler"
                                code:fail ? 17 : 19
                            userInfo:@{NSLocalizedDescriptionKey : fail ? @"error" : @"warning"}];
    }
    if (fail) {
        return nil;
    }
    ControlledLibrary* library = [ControlledLibrary new];
    library.device = (id<MTLDevice>)self.owner;
    library.label = descriptor.name;
    return (id<MTLLibrary>)library;
}

- (id<MTLLibrary>)newLibraryWithDescriptor:(MTL4LibraryDescriptor*)descriptor
                                     error:(NSError**)error {
    std::lock_guard lock(mutex);
    ++synchronous;
    self.lastDescriptor = descriptor;
    return [self makeLibrary:descriptor error:error];
}

- (id<MTL4CompilerTask>)newLibraryWithDescriptor:(MTL4LibraryDescriptor*)descriptor
                               completionHandler:(MTLNewLibraryCompletionHandler)completion {
    {
        std::lock_guard lock(mutex);
        ++asynchronous;
        self.lastDescriptor = descriptor;
        if (!inlineCompletion) {
            pending.push_back({descriptor, [completion copy]});
            return (id<MTL4CompilerTask>)[NSObject new];
        }
    }

    NSError* error = nil;
    auto library = [self makeLibrary:descriptor error:&error];
    completion(library, error);
    return (id<MTL4CompilerTask>)[NSObject new];
}

- (void)completeNext {
    PendingRequest request;
    {
        std::lock_guard lock(mutex);
        assert(!pending.empty());
        request = std::move(pending.front());
        pending.pop_front();
    }

    NSError* error = nil;
    const auto library = [self makeLibrary:request.descriptor error:&error];
    request.completion(library, error);
}
@end

@implementation ControlledDevice

- (instancetype)init {
    if ((self = [super init])) {
        _compiler = [ControlledCompiler new];
        _compiler.owner = self;
    }
    return self;
}

- (uint64_t)registryID {
    return 123;
}

- (id<MTL4Compiler>)newCompilerWithDescriptor:(MTL4CompilerDescriptor*)descriptor
                                        error:(NSError**)error {
    assert(descriptor.label.length != 0 && error != nullptr);
    self.recordedSerializer = descriptor.pipelineDataSetSerializer;
    return (id<MTL4Compiler>)self.compiler;
}

- (id<MTLLibrary>)newLibraryWithData:(dispatch_data_t)data error:(NSError**)error {
    assert(error != nullptr);
    const void* payload = nullptr;
    std::size_t size{};
    auto mapped = dispatch_data_create_map(data, &payload, &size);
    assert(mapped != nil && HasMetalLibrarySignature({static_cast<const u8*>(payload), size}));
    if (size != 5 || static_cast<const u8*>(payload)[4] != 42) {
        *error = [NSError errorWithDomain:@"Fixture.Binary" code:23 userInfo:nil];
        return nil;
    }
    ControlledLibrary* library = [ControlledLibrary new];
    library.device = (id<MTLDevice>)self;
    return (id<MTLLibrary>)library;
}
@end

namespace {

using Clock = std::chrono::steady_clock;

ShaderLibrary Resolve(const ShaderCompilation& compilation) {
    auto result = compilation.WaitUntil(Clock::now() + 5s);
    assert(result);
    return *result;
}

constexpr std::string_view stamp_source = R"(
#include <metal_stdlib>
using namespace metal;
kernel void stamp(device uint* output [[buffer(0)]]) {
    output[0] = 0x4d534c34u;
}
)";

void VerifyHost() {
    auto holder = [ControlledDevice new];
    auto native = holder.compiler;
    DeviceContext context((id<MTLDevice>)holder);
    DeviceContext foreign((id<MTLDevice>)holder);
    const auto capture = (id<MTL4PipelineDataSetSerializer>)[NSObject new];
    {
        ShaderCompiler capturing(context, 1, capture);
        assert(holder.recordedSerializer == capture);
    }
    ShaderCompiler compiler(context, 1);
    assert(holder.recordedSerializer == nil);
    std::puts("PASS native compiler descriptor serializer attachment and default ABI");
    ShaderSource shader{std::string{stamp_source}, "stamp", "#define STAMP 42", {}};
    shader.options.math = ShaderMath::Fast;
    shader.options.optimization = ShaderOptimization::Size;
    shader.options.preserve_invariance = false;
    shader.options.definitions.emplace("VARIANT", "17");

    const auto initial = compiler.QueueLibrary(shader);
    assert(!initial.IsReady() && !initial.WaitUntil(Clock::now()));
    assert(native->asynchronous == 1);
    const auto descriptor = native.lastDescriptor;
    assert([descriptor.name isEqualToString:@"stamp"]);
    assert([descriptor.source hasPrefix:@"#define STAMP 42\n"]);
    assert(descriptor.options.languageVersion == MTLLanguageVersion4_0);
    assert(descriptor.options.mathMode == MTLMathModeFast);
    assert(descriptor.options.mathFloatingPointFunctions == MTLMathFloatingPointFunctionsFast);
    assert(descriptor.options.optimizationLevel == MTLLibraryOptimizationLevelSize);
    assert(!descriptor.options.preserveInvariance);
    assert([descriptor.options.preprocessorMacros[@"VARIANT"] isEqual:@"17"]);

    std::vector<std::future<ShaderCompilation>> requests;
    for (std::size_t i = 0; i < 12; ++i) {
        requests.push_back(
            std::async(std::launch::async, [&] { return compiler.QueueLibrary(shader); }));
    }
    std::vector<ShaderCompilation> duplicates;
    for (auto& request : requests) {
        duplicates.push_back(request.get());
    }
    assert(native->asynchronous == 1);

    auto variant = shader;
    variant.options.definitions["VARIANT"] = "18";
    assert(Resolve(compiler.QueueLibrary(variant)).GetStatus(context) ==
           ShaderCompileStatus::QueueFull);
    variant.options.definitions["bad name"] = "1";
    assert(Resolve(compiler.QueueLibrary(variant)).GetStatus(context) ==
           ShaderCompileStatus::InvalidSource);
    variant = shader;
    variant.preamble = std::string{"A\0B", 3};
    assert(compiler.CompileLibrary(variant).GetStatus(context) ==
           ShaderCompileStatus::InvalidSource);
    variant = shader;
    variant.name = "warn";
    variant.options = {};
    const auto independent = compiler.CompileLibrary(variant);
    assert(independent.GetStatus(context) == ShaderCompileStatus::Success);
    assert(independent.GetStatus(foreign) == ShaderCompileStatus::ForeignDevice);
    assert(independent.GetLibrary(foreign) == nil);
    assert(independent.GetDiagnostic().warning && independent.GetDiagnostic().code == 19);
    assert(native.lastDescriptor.options.mathMode == MTLMathModeSafe);
    assert(native.lastDescriptor.options.preserveInvariance);

    [native completeNext];
    const auto done = Resolve(initial);
    assert(done.GetLibrary(context) != nil);
    for (const auto& duplicate : duplicates) {
        assert(Resolve(duplicate).GetLibrary(context) == done.GetLibrary(context));
    }

    variant.name = "fail";
    const auto failed = compiler.CompileLibrary(variant);
    assert(failed.GetStatus(context) == ShaderCompileStatus::CompileFailed);
    assert(failed.GetDiagnostic().domain == "Fixture.Compiler");
    assert(failed.GetDiagnostic().message == "error");
    assert(!failed.GetDiagnostic().warning && failed.GetDiagnostic().code == 17);
    native->inlineCompletion = true;
    assert(Resolve(compiler.QueueLibrary(variant)).GetDiagnostic().message == "error");
    assert(Resolve(compiler.QueueLibrary(shader)).GetLibrary(context) != nil);
    native->inlineCompletion = false;

    const auto title = compiler.TitleGeneration();
    const auto old_job = compiler.QueueLibrary(shader);
    compiler.InvalidateTitle();
    assert(compiler.TitleGeneration() == title + 1);
    assert(Resolve(old_job).GetStatus(context) == ShaderCompileStatus::Canceled);
    assert(done.GetLibrary(context) == nil);
    assert(Resolve(compiler.QueueLibrary(shader)).GetStatus(context) ==
           ShaderCompileStatus::QueueFull);
    [native completeNext];
    const auto generation_job = compiler.QueueLibrary(shader);
    const auto valid = context.AcquireWork();
    assert(compiler.GetNativeCompiler(valid) != nil);
    assert(context.AdvanceGeneration() && compiler.GetNativeCompiler(valid) == nil);
    [native completeNext];
    assert(Resolve(generation_job).GetStatus(context) == ShaderCompileStatus::Canceled);

    ShaderCompiler disabled(context, 0);
    assert(Resolve(disabled.QueueLibrary(shader)).GetStatus(context) ==
           ShaderCompileStatus::QueueFull);
    assert(disabled.CompileLibrary(shader).GetLibrary(context) != nil);
    const std::array<u8, 5> bytes{'M', 'T', 'L', 'B', 42};
    const auto binary = compiler.LoadLibrary(bytes, "fixture");
    assert([binary.GetLibrary(context).label isEqualToString:@"fixture"]);
    assert(compiler.LoadLibrary(std::span{bytes}.first(4), "").GetStatus(context) ==
           ShaderCompileStatus::InvalidBinary);
    auto malformed = bytes;
    malformed.back() = 0;
    const auto signature_only = compiler.LoadLibrary(malformed, "");
    assert(signature_only.GetStatus(context) == ShaderCompileStatus::LoadFailed);
    assert(signature_only.GetDiagnostic().domain == "Fixture.Binary");

    std::optional<ShaderCompilation> orphan;
    {
        ShaderCompiler temporary(context);
        orphan.emplace(temporary.QueueLibrary(shader));
    }
    assert(Resolve(*orphan).GetStatus(context) == ShaderCompileStatus::Canceled);
    [native completeNext];
    compiler.Close();
    compiler.Close();
    assert(compiler.TitleGeneration() == 0 && binary.GetLibrary(context) == nil);
    assert(compiler.CompileLibrary(shader).GetStatus(context) == ShaderCompileStatus::Canceled);
    std::puts("PASS isolated host queue and SDK contracts using test-only "
              "completion controls");
}

void Dispatch(ShaderCompiler& compiler, DeviceContext& context, id<MTLLibrary> library) {
    assert(library && [library.functionNames containsObject:@"stamp"]);
    auto device = context.GetDevice();
    auto function = [MTL4LibraryFunctionDescriptor new];
    function.library = library;
    function.name = @"stamp";
    auto descriptor = [MTL4ComputePipelineDescriptor new];
    descriptor.computeFunctionDescriptor = function;
    const auto native = compiler.GetNativeCompiler(context.AcquireWork());
    assert(native);
    NSError* error = nil;
    auto pipeline = [native newComputePipelineStateWithDescriptor:descriptor
                                              compilerTaskOptions:nil
                                                            error:&error];
    pipeline = CheckResource(pipeline, "Compile shader compiler verification pipeline", error);
    auto args = [MTL4ArgumentTableDescriptor new];
    args.maxBufferBindCount = 1;
    auto table = [device newArgumentTableWithDescriptor:args error:&error];
    table = CheckResource(table, "Create shader compiler verification argument table", error);
    const u32 zero{};
    auto buffer = CheckResource([device newBufferWithBytes:&zero
                                                    length:sizeof(zero)
                                                   options:MTLResourceStorageModeShared |
                                                           MTLResourceHazardTrackingModeUntracked],
                                "Create shader compiler verification output");
    [table setAddress:buffer.gpuAddress atIndex:0];
    Scheduler scheduler(context);
    scheduler.Record([=](EncodingContext& ctx) {
        const std::array uses{ResourceUse{buffer, BufferAccess::Write, MTLStageDispatch}};
        auto encoder = [ctx.GetCommandBuffer() computeCommandEncoder];
        ctx.Synchronize(encoder, uses);
        [encoder setComputePipelineState:pipeline];
        [encoder setArgumentTable:table];
        [encoder dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
        ctx.Retain(table);
        ctx.Retain(pipeline);
        ctx.EndPass(encoder);
    });
    assert(scheduler.Finish({}, Clock::now() + 10s) == CompletionStatus::Complete);
    assert(*static_cast<const u32*>(buffer.contents) == 0x4d534c34u);
}

} // Anonymous namespace

int main(int argc, const char* argv[]) {
    @autoreleasepool {
        assert(argc >= 2);
        if (std::string_view{argv[1]} == "generate") {
            assert(argc == 3);
            std::ofstream file(argv[2]);
            file << stamp_source;
            assert(file.good());
            return 0;
        }
        assert(argc == 3 && std::string_view{argv[1]} == "run");
        VerifyHost();
        auto device = MTLCreateSystemDefaultDevice();
        if (!device || ![device supportsFamily:MTLGPUFamilyMetal4]) {
            std::puts("PENDING live Metal 4 synchronous and asynchronous compilation, "
                      "diagnostics, "
                      "native metallib load and three known-answer compute dispatches");
            return 2;
        }
        DeviceContext context(device);
        ShaderCompiler compiler(context);
        const ShaderSource shader{std::string(stamp_source), "runtime stamp", {}, {}};
        Dispatch(compiler, context, compiler.CompileLibrary(shader).GetLibrary(context));
        Dispatch(compiler, context, Resolve(compiler.QueueLibrary(shader)).GetLibrary(context));
        std::ifstream input(argv[2], std::ios::binary);
        assert(input);
        const std::vector<u8> blob{std::istreambuf_iterator<char>{input}, {}};
        Dispatch(compiler, context,
                 compiler.LoadLibrary(blob, "offline stamp").GetLibrary(context));
        auto invalid = shader;
        invalid.text = "kernel void stamp( %";
        const auto result = compiler.CompileLibrary(invalid);
        assert(result.GetStatus(context) == ShaderCompileStatus::CompileFailed);
        assert(!result.GetDiagnostic().domain.empty() && !result.GetDiagnostic().message.empty());
        assert(Resolve(compiler.QueueLibrary(invalid)).GetStatus(context) ==
               ShaderCompileStatus::CompileFailed);
        std::puts("PASS live native shader compiler and embedded binary execution");
    }
}
