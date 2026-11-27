#include <array>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include "video_core/renderer_metal/mtl_instance.h"
#include "video_core/renderer_metal/mtl_render_manager.h"

using namespace Metal;
using namespace std::chrono_literals;

namespace {

constexpr u32 TestSize = 8;

auto Deadline() {
    return std::chrono::steady_clock::now() + 10s;
}

template <typename Exception, typename T>
bool Throws(T&& callback) {
    try {
        callback();
        return false;
    } catch (const Exception&) {
        return true;
    }
}

id<MTLTexture> Texture(id<MTLDevice> device, MTLPixelFormat format, MTLStorageMode storage,
                       u32 samples = 1) {
    auto descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format
                                                                         width:TestSize
                                                                        height:TestSize
                                                                     mipmapped:NO];
    if (samples > 1) {
        descriptor.textureType = MTLTextureType2DMultisample;
        descriptor.sampleCount = samples;
    }
    descriptor.storageMode = storage;
    descriptor.hazardTrackingMode = MTLHazardTrackingModeUntracked;
    descriptor.usage = MTLTextureUsageRenderTarget;
    if (storage != MTLStorageModeMemoryless) {
        descriptor.usage |= MTLTextureUsageShaderRead;
    }
    return CheckResource([device newTextureWithDescriptor:descriptor], "Test render attachment");
}

id<MTLRenderPipelineState> BuildPipeline(id<MTLDevice> device, id<MTLLibrary> library,
                                         NSString* fragment, MTLPixelFormat color, u32 samples) {
    NSError* error = nil;
    auto compiler = [device newCompilerWithDescriptor:[MTL4CompilerDescriptor new] error:&error];
    compiler = CheckResource(compiler, "Create test render compiler", error);
    auto vertex = [MTL4LibraryFunctionDescriptor new];
    vertex.library = library;
    vertex.name = @"full_triangle";
    auto pixel = [MTL4LibraryFunctionDescriptor new];
    pixel.library = library;
    pixel.name = fragment;
    auto descriptor = [MTL4RenderPipelineDescriptor new];
    descriptor.vertexFunctionDescriptor = vertex;
    descriptor.fragmentFunctionDescriptor = pixel;
    descriptor.rasterSampleCount = samples;
    if (color != MTLPixelFormatInvalid) {
        descriptor.colorAttachments[0].pixelFormat = color;
    }
    auto pipeline = [compiler newRenderPipelineStateWithDescriptor:descriptor
                                               compilerTaskOptions:nil
                                                             error:&error];
    return CheckResource(pipeline, "Compile test render pipeline", error);
}

std::array<u8, 4> Pixel(id<MTLTexture> texture, u32 x, u32 y) {
    std::array<u8, 4> result{};
    [texture getBytes:result.data()
          bytesPerRow:result.size()
           fromRegion:MTLRegionMake2D(x, y, 1, 1)
          mipmapLevel:0];
    return result;
}

void Triangle(EncodingContext& ctx, id<MTL4RenderCommandEncoder> encoder,
              id<MTLRenderPipelineState> pipeline) {
    [encoder setRenderPipelineState:pipeline];
    [encoder setViewport:MTLViewport{0, 0, TestSize, TestSize, 0, 1}];
    ctx.Retain(pipeline);
    [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
}

void CheckColor(Instance& instance, id<MTLLibrary> library) {
    id<MTLDevice> device = instance.GetDevice();
    Scheduler scheduler(instance.GetContext());
    MemoryBudget budget(8 * 1024 * 1024);
    MemoryAllocator allocator(device, budget);
    RenderManager manager(instance, scheduler, allocator);
    auto target = Texture(device, MTLPixelFormatRGBA8Unorm, MTLStorageModeShared);
    auto copied = Texture(device, MTLPixelFormatRGBA8Unorm, MTLStorageModeShared);
    RenderPass full{
        .color = {.target = {target}, .load_action = MTLLoadActionClear},
        .area = {0, 0, TestSize, TestSize},
        .width = TestSize,
        .height = TestSize,
        .clear_color = MTLClearColorMake(1, 0, 0, 1),
    };
    id<MTL4RenderCommandEncoder> previous = nil;
    manager.BeginRendering(full);
    manager.RecordDraw(
        [&](EncodingContext&, id<MTL4RenderCommandEncoder> encoder) { previous = encoder; });
    auto same = full;
    same.color.load_action = MTLLoadActionLoad;
    manager.BeginRendering(same);
    manager.RecordDraw([&](EncodingContext&, id<MTL4RenderCommandEncoder> encoder) {
        assert(previous && previous == encoder);
        previous = nil;
    });
    assert(Throws<std::invalid_argument>(
        [&] { manager.CopyForFeedback({target}, {target}, full.area); }));
    manager.CopyForFeedback({target}, {copied}, full.area);
    auto partial = full;
    partial.area = {2, 3, 4, 2};
    partial.clear_color = MTLClearColorMake(0, 1, 0, 1);
    manager.BeginRendering(partial);
    assert(scheduler.Finish({}, Deadline()) == CompletionStatus::Complete);
    assert(budget.GetUsage().physical_used == 0);
    const std::array<u8, 4> green = {0, 255, 0, 255};
    const std::array<u8, 4> red = {255, 0, 0, 255};
    for (u32 row = 0; row < TestSize; ++row) {
        for (u32 column = 0; column < TestSize; ++column) {
            const bool area = row >= 3 && row < 5 && column >= 2 && column < 6;
            assert(Pixel(target, column, row) == (area ? green : red));
            assert(Pixel(copied, column, row) == red);
        }
    }

    if (instance.GetCapabilities().raster_order_groups) {
        auto pipeline = BuildPipeline(device, library, @"ordered_increment", target.pixelFormat, 1);
        full.tile_feedback = true;
        full.clear_color = MTLClearColorMake(0, 0, 0, 1);
        manager.BeginRendering(full);
        manager.RecordDraw([pipeline](EncodingContext& ctx, id<MTL4RenderCommandEncoder> encoder) {
            Triangle(ctx, encoder, pipeline);
            Triangle(ctx, encoder, pipeline);
        });
        assert(scheduler.Finish({}, Deadline()) == CompletionStatus::Complete);
        assert((Pixel(target, 4, 4) == std::array<u8, 4>{2, 0, 0, 255}));
    }
    assert(scheduler.Shutdown(Deadline()));
}

void CheckColorResolve(Instance& instance, u32 samples) {
    const auto device = instance.GetDevice();
    Scheduler scheduler(instance.GetContext());
    MemoryAllocator allocator(device, MemoryBudget(8 * 1024 * 1024));
    RenderManager manager(instance, scheduler, allocator);
    auto samples_tex = Texture(device, MTLPixelFormatRGBA8Unorm, MTLStorageModePrivate, samples);
    auto resolved = Texture(device, MTLPixelFormatRGBA8Unorm, MTLStorageModeShared);
    RenderPass pass{
        .color = {.target = {samples_tex},
                  .resolve = {resolved},
                  .load_action = MTLLoadActionClear,
                  .store_action = MTLStoreActionStoreAndMultisampleResolve},
        .area = {0, 0, TestSize, TestSize},
        .width = TestSize,
        .height = TestSize,
        .samples = samples,
        .clear_color = MTLClearColorMake(0, 0, 1, 1),
    };
    manager.BeginRendering(pass);
    assert(scheduler.Finish({}, Deadline()) == CompletionStatus::Complete);
    assert((Pixel(resolved, 4, 4) == std::array<u8, 4>{0, 0, 255, 255}));

    auto another = Texture(device, MTLPixelFormatRGBA8Unorm, MTLStorageModeShared);
    pass.color.resolve.texture = another;
    pass.color.load_action = MTLLoadActionLoad;
    manager.BeginRendering(pass);
    assert(scheduler.Finish({}, Deadline()) == CompletionStatus::Complete);
    assert((Pixel(another, 4, 4) == std::array<u8, 4>{0, 0, 255, 255}));

    pass.color.target.texture =
        Texture(device, MTLPixelFormatRGBA8Unorm, MTLStorageModeMemoryless, samples);
    assert(Throws<std::invalid_argument>([&] { manager.BeginRendering(pass); }));
    pass.color.load_action = MTLLoadActionClear;
    pass.color.store_action = MTLStoreActionMultisampleResolve;
    pass.clear_color = MTLClearColorMake(1, 0, 0, 1);
    manager.BeginRendering(pass);
    assert(scheduler.Finish({}, Deadline()) == CompletionStatus::Complete);
    assert((Pixel(another, 4, 4) == std::array<u8, 4>{255, 0, 0, 255}));
}

void CheckSampleFeedback(Instance& instance, id<MTLLibrary> library, u32 samples) {
    auto device = instance.GetDevice();
    Scheduler scheduler(instance.GetContext());
    MemoryAllocator allocator(device, MemoryBudget(8 * 1024 * 1024));
    RenderManager manager(instance, scheduler, allocator);
    auto source = Texture(device, MTLPixelFormatRGBA8Unorm, MTLStorageModePrivate, samples);
    auto copy = Texture(device, MTLPixelFormatRGBA8Unorm, MTLStorageModePrivate, samples);
    auto single = Texture(device, MTLPixelFormatRGBA8Unorm, MTLStorageModePrivate);
    RenderPass pass{
        .color = {.target = {source}, .load_action = MTLLoadActionClear},
        .area = {0, 0, TestSize, TestSize},
        .width = TestSize,
        .height = TestSize,
        .samples = samples,
    };
    assert(Throws<std::invalid_argument>(
        [&] { manager.CopyForFeedback({source}, {single}, pass.area); }));
    auto pipeline = BuildPipeline(device, library, @"distinct_color", source.pixelFormat, samples);
    manager.BeginRendering(pass);
    manager.RecordDraw([pipeline](EncodingContext& ctx, id<MTL4RenderCommandEncoder> encoder) {
        Triangle(ctx, encoder, pipeline);
    });
    manager.CopyForFeedback({source}, {copy}, pass.area);

    NSError* error = nil;
    auto compiler = [device newCompilerWithDescriptor:[MTL4CompilerDescriptor new] error:&error];
    compiler = CheckResource(compiler, "Create MSAA snapshot fixture compiler", error);
    auto function = [MTL4LibraryFunctionDescriptor new];
    function.library = library;
    function.name = @"read_samples";
    auto descriptor = [MTL4ComputePipelineDescriptor new];
    descriptor.computeFunctionDescriptor = function;
    auto reader = [compiler newComputePipelineStateWithDescriptor:descriptor
                                              compilerTaskOptions:nil
                                                            error:&error];
    reader = CheckResource(reader, "Compile MSAA snapshot readback", error);
    const BufferRange range{0, samples * sizeof(u32)};
    auto output = allocator.AllocateBuffer(range.length, MemoryPool::Readback);
    auto lease = output.AcquireGPUAccess(range, BufferAccess::Write);
    scheduler.Record(
        [copy, reader, samples, access = std::move(lease)](EncodingContext& ctx) mutable {
            auto encoder = [ctx.GetCommandBuffer() computeCommandEncoder];
            const std::array uses{ResourceUse{copy, BufferAccess::Read, MTLStageDispatch},
                                  ResourceUse{access.GetBuffer(), BufferAccess::Write,
                                              MTLStageDispatch, access.Range()}};
            ctx.Synchronize(encoder, uses);
            auto arguments = [MTL4ArgumentTableDescriptor new];
            arguments.maxTextureBindCount = 1;
            arguments.maxBufferBindCount = 1;
            NSError* error = nil;
            auto table = [ctx.GetCommandBuffer().device newArgumentTableWithDescriptor:arguments
                                                                                 error:&error];
            table = CheckResource(table, "Create MSAA snapshot readback arguments", error);
            [table setTexture:copy.gpuResourceID atIndex:0];
            [table setAddress:access.GetBuffer().gpuAddress atIndex:0];
            [encoder setComputePipelineState:reader];
            [encoder setArgumentTable:table];
            [encoder dispatchThreads:MTLSizeMake(samples, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(samples, 1, 1)];
            ctx.Retain(std::move(access));
            ctx.Retain(table);
            ctx.Retain(reader);
            ctx.EndPass(encoder);
        });
    assert(scheduler.Finish({}, Deadline()) == CompletionStatus::Complete);
    auto read = output.MapRead(range);
    for (u32 sample = 0; sample < samples; ++sample) {
        u32 value;
        std::memcpy(&value, read.Bytes().data() + sample * sizeof(value), sizeof(value));
        assert(value == sample + 1);
    }
}

void CheckDepthStencil(Instance& instance, id<MTLLibrary> library, u32 samples) {
    constexpr auto Format = MTLPixelFormatDepth32Float_Stencil8;
    const auto device = instance.GetDevice();
    Scheduler scheduler(instance.GetContext());
    MemoryAllocator allocator(device, MemoryBudget(8 * 1024 * 1024));
    RenderManager manager(instance, scheduler, allocator);
    auto backing = Texture(device, Format, MTLStorageModePrivate, samples);
    auto resolved = Texture(device, Format, MTLStorageModePrivate);
    const RenderAttachment attachment{
        .target = {backing},
        .resolve = {resolved},
        .load_action = MTLLoadActionClear,
        .store_action = MTLStoreActionStoreAndMultisampleResolve,
    };
    RenderPass pass{
        .depth = attachment,
        .stencil = attachment,
        .area = {0, 0, TestSize, TestSize},
        .width = TestSize,
        .height = TestSize,
        .samples = samples,
        .clear_depth = 1,
    };
    auto pipeline =
        BuildPipeline(device, library, @"distinct_samples", MTLPixelFormatInvalid, samples);
    auto state = [MTLDepthStencilDescriptor new];
    state.depthWriteEnabled = YES;
    state.depthCompareFunction = MTLCompareFunctionAlways;
    auto stencil = [MTLStencilDescriptor new];
    stencil.stencilCompareFunction = MTLCompareFunctionAlways;
    stencil.depthStencilPassOperation = MTLStencilOperationReplace;
    state.frontFaceStencil = stencil;
    state.backFaceStencil = stencil;
    auto depth_stencil =
        CheckResource([device newDepthStencilStateWithDescriptor:state], "test depth state");
    manager.BeginRendering(pass);
    manager.RecordDraw(
        [pipeline, depth_stencil](EncodingContext& ctx, id<MTL4RenderCommandEncoder> encoder) {
            [encoder setDepthStencilState:depth_stencil];
            ctx.Retain(depth_stencil);
            Triangle(ctx, encoder, pipeline);
        });
    manager.EndRendering();

    pass.area = {1, 2, 3, 4};
    pass.clear_depth = 0.5;
    pass.depth.load_action = MTLLoadActionClear;
    pass.stencil.load_action = MTLLoadActionLoad;
    manager.BeginRendering(pass);
    manager.EndRendering();

    constexpr NSUInteger Stride = 256;
    constexpr NSUInteger Plane = Stride * TestSize;
    auto data = allocator.AllocateBuffer(2 * Plane, MemoryPool::Readback);
    auto write = data.AcquireGPUAccess({0, 2 * Plane}, BufferAccess::Write);
    scheduler.Record([resolved, write = std::move(write)](EncodingContext& ctx) mutable {
        auto encoder = [ctx.GetCommandBuffer() computeCommandEncoder];
        const auto buffer = write.GetBuffer();
        const std::array uses{
            ResourceUse{resolved, BufferAccess::Read, MTLStageBlit},
            ResourceUse{buffer, BufferAccess::Write, MTLStageBlit, write.Range()},
        };
        ctx.Synchronize(encoder, uses);
        const std::array<MTLBlitOption, 2> aspects = {MTLBlitOptionDepthFromDepthStencil,
                                                      MTLBlitOptionStencilFromDepthStencil};
        for (NSUInteger aspect = 0; aspect < aspects.size(); ++aspect) {
            [encoder copyFromTexture:resolved
                             sourceSlice:0
                             sourceLevel:0
                            sourceOrigin:MTLOriginMake(0, 0, 0)
                              sourceSize:MTLSizeMake(TestSize, TestSize, 1)
                                toBuffer:buffer
                       destinationOffset:aspect * Plane
                  destinationBytesPerRow:Stride
                destinationBytesPerImage:0
                                 options:aspects[aspect]];
        }
        ctx.Retain(std::move(write));
        ctx.EndPass(encoder);
    });
    assert(scheduler.Finish({}, Deadline()) == CompletionStatus::Complete);
    const auto bytes = data.MapRead({0, 2 * Plane});
    for (u32 y = 0; y < TestSize; ++y) {
        for (u32 x = 0; x < TestSize; ++x) {
            f32 depth;
            std::memcpy(&depth, bytes.Bytes().data() + y * Stride + x * sizeof(depth),
                        sizeof(depth));
            const bool cleared = x >= 1 && x < 4 && y >= 2 && y < 6;
            assert(depth == (cleared ? 0.5f : 0.25f));
            assert(bytes.Bytes()[Plane + y * Stride + x] == std::byte{0x55});
        }
    }
}

} // Anonymous namespace

int main(int argc, char** argv) {
    @autoreleasepool {
        assert(argc == 2);
        id<MTLDevice> native = MTLCreateSystemDefaultDevice();
        if (!native || ![native supportsFamily:MTLGPUFamilyMetal4]) {
            std::puts("PENDING Metal 4 merge/split, full and partial clears, "
                      "feedback and color/depth/stencil resolves");
            return 2;
        }
        Instance instance(0, false);
        NSError* error = nil;
        auto library = [instance.GetDevice() newLibraryWithURL:[NSURL fileURLWithPath:@(argv[1])]
                                                         error:&error];
        library = CheckResource(library, "Load native render verification library", error);
        CheckColor(instance, library);
        const u32 samples = instance.SupportsSampleCount(4) ? 4 : 2;
        assert(instance.SupportsSampleCount(samples));
        CheckColorResolve(instance, samples);
        CheckSampleFeedback(instance, library, samples);
        CheckDepthStencil(instance, library, samples);
        std::puts("PASS Metal 4 render passes, feedback, rectangular clear and "
                  "MSAA resolve");
    }
}
