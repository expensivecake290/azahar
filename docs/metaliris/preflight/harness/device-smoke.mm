#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalFX/MetalFX.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>

namespace {
constexpr uint32_t ExpectedComputeResult = 0x4d544c34u;

int Failed(const char* operation, NSError* error = nil) {
    std::fprintf(stderr, "%s failed: %s\n", operation,
                 error ? error.localizedDescription.UTF8String : "no error detail");
    return 1;
}

MTL4LibraryFunctionDescriptor* Function(id<MTLLibrary> library, NSString* name) {
    auto* descriptor = [MTL4LibraryFunctionDescriptor new];
    descriptor.library = library;
    descriptor.name = name;
    return descriptor;
}

bool SubmitAndWait(id<MTL4CommandQueue> queue, id<MTL4CommandBuffer> command) {
    auto completion = dispatch_semaphore_create(0);
    __block NSError* execution_error = nil;
    auto* options = [MTL4CommitOptions new];
    [options addFeedbackHandler:^(id<MTL4CommitFeedback> feedback) {
        execution_error = feedback.error;
        dispatch_semaphore_signal(completion);
    }];
    const id<MTL4CommandBuffer> commands[] = {command};
    [queue commit:commands count:1 options:options];
    if (dispatch_semaphore_wait(completion, dispatch_time(DISPATCH_TIME_NOW, 10 * NSEC_PER_SEC))) {
        Failed("Metal 4 completion timed out");
        return false;
    }
    if (execution_error) {
        Failed("Metal 4 execution", execution_error);
        return false;
    }
    return true;
}

int RunWorkload(id<MTLDevice> device, NSString* library_path, NSString* trace_path) {
    NSError* error = nil;
    id<MTL4CommandQueue> queue = [device newMTL4CommandQueue];
    id<MTL4CommandAllocator> allocator = [device newCommandAllocator];
    id<MTL4CommandBuffer> command = [device newCommandBuffer];
    id<MTL4Compiler> compiler = [device newCompilerWithDescriptor:[MTL4CompilerDescriptor new]
                                                          error:&error];
    if (!queue || !allocator || !command || !compiler) {
        return Failed("Metal 4 command or compiler creation", error);
    }

    id<MTLLibrary> library = [device newLibraryWithURL:[NSURL fileURLWithPath:library_path]
                                               error:&error];
    if (!library) {
        return Failed("Load compiled MSL library", error);
    }
    auto* compute = [MTL4ComputePipelineDescriptor new];
    compute.computeFunctionDescriptor = Function(library, @"gate0_write");
    id<MTLComputePipelineState> compute_pipeline =
        [compiler newComputePipelineStateWithDescriptor:compute compilerTaskOptions:nil error:&error];
    if (!compute_pipeline) {
        return Failed("Metal 4 compute pipeline", error);
    }

    id<MTLBuffer> result = [device newBufferWithLength:sizeof(uint32_t)
                                              options:MTLResourceStorageModeShared];
    auto* arguments = [MTL4ArgumentTableDescriptor new];
    arguments.maxBufferBindCount = 1;
    id<MTL4ArgumentTable> table = [device newArgumentTableWithDescriptor:arguments error:&error];
    id<MTLResidencySet> residency =
        [device newResidencySetWithDescriptor:[MTLResidencySetDescriptor new] error:&error];
    if (!result || !table || !residency) {
        return Failed("Shared result, argument table, or residency set", error);
    }
    auto* result_value = static_cast<uint32_t*>(result.contents);
    *result_value = 0;
    [table setAddress:result.gpuAddress atIndex:0];
    [residency addAllocation:result];

    id<MTLRenderPipelineState> mesh_pipeline = nil;
    id<MTLTexture> mesh_output = nil;
    if ([device supportsFamily:MTLGPUFamilyApple7]) {
        auto* mesh = [MTL4MeshRenderPipelineDescriptor new];
        mesh.objectFunctionDescriptor = Function(library, @"gate0_object");
        mesh.meshFunctionDescriptor = Function(library, @"gate0_mesh");
        mesh.fragmentFunctionDescriptor = Function(library, @"gate0_fragment");
        mesh.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
        mesh_pipeline = [compiler newRenderPipelineStateWithDescriptor:mesh
                                                    compilerTaskOptions:nil error:&error];
        auto* target = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                                         width:4 height:4 mipmapped:NO];
        target.storageMode = MTLStorageModeShared;
        target.usage = MTLTextureUsageRenderTarget;
        mesh_output = [device newTextureWithDescriptor:target];
        if (!mesh_pipeline || !mesh_output) {
            return Failed("Metal 4 object/mesh pipeline or target", error);
        }
        [residency addAllocation:mesh_output];
    } else {
        std::puts("Object/mesh draw skipped because the Apple7 family gate is false.");
    }

    if (![MTLFXSpatialScalerDescriptor supportsMetal4FX:device]) {
        return Failed("Native MetalFX Spatial device capability");
    }
    auto* spatial = [MTLFXSpatialScalerDescriptor new];
    spatial.inputWidth = spatial.inputHeight = 32;
    spatial.outputWidth = spatial.outputHeight = 64;
    spatial.colorTextureFormat = spatial.outputTextureFormat = MTLPixelFormatRGBA8Unorm;
    id<MTL4FXSpatialScaler> scaler = [spatial newSpatialScalerWithDevice:device compiler:compiler];
    if (!scaler) {
        return Failed("Native MetalFX Spatial creation");
    }
    auto* input_desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:spatial.colorTextureFormat
                                                                          width:32 height:32 mipmapped:NO];
    input_desc.storageMode = MTLStorageModeShared;
    input_desc.usage = scaler.colorTextureUsage;
    auto* output_desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:spatial.outputTextureFormat
                                                                           width:64 height:64 mipmapped:NO];
    output_desc.storageMode = MTLStorageModePrivate;
    output_desc.usage = scaler.outputTextureUsage;
    id<MTLTexture> input_texture = [device newTextureWithDescriptor:input_desc];
    id<MTLTexture> output_texture = [device newTextureWithDescriptor:output_desc];
    id<MTLBuffer> spatial_result = [device newBufferWithLength:256
                                                      options:MTLResourceStorageModeShared];
    if (!input_texture || !output_texture || !spatial_result) {
        return Failed("MetalFX Spatial texture or readback creation");
    }
    std::array<uint32_t, 32 * 32> input_pixels;
    input_pixels.fill(0xff0000ffu);
    [input_texture replaceRegion:MTLRegionMake2D(0, 0, 32, 32) mipmapLevel:0
                       withBytes:input_pixels.data() bytesPerRow:32 * sizeof(uint32_t)];
    scaler.colorTexture = input_texture;
    scaler.outputTexture = output_texture;
    scaler.inputContentWidth = scaler.inputContentHeight = 32;
    [residency addAllocation:input_texture];
    [residency addAllocation:output_texture];
    [residency addAllocation:spatial_result];
    [residency commit];

    auto* capture = [MTLCaptureManager sharedCaptureManager];
    auto* capture_desc = [MTLCaptureDescriptor new];
    capture_desc.captureObject = device;
    capture_desc.destination = MTLCaptureDestinationGPUTraceDocument;
    capture_desc.outputURL = [NSURL fileURLWithPath:trace_path];
    if (![capture startCaptureWithDescriptor:capture_desc error:&error]) {
        return Failed("GPU capture start", error);
    }

    [command beginCommandBufferWithAllocator:allocator];
    [command useResidencySet:residency];
    id<MTL4ComputeCommandEncoder> encoder = [command computeCommandEncoder];
    if (!encoder) {
        [command endCommandBuffer];
        [capture stopCapture];
        return Failed("Metal 4 compute encoder");
    }
    [encoder setComputePipelineState:compute_pipeline];
    [encoder setArgumentTable:table];
    [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [encoder endEncoding];

    if (mesh_pipeline) {
        auto* pass = [MTL4RenderPassDescriptor new];
        pass.colorAttachments[0].texture = mesh_output;
        pass.colorAttachments[0].loadAction = MTLLoadActionClear;
        pass.colorAttachments[0].storeAction = MTLStoreActionStore;
        id<MTL4RenderCommandEncoder> render = [command renderCommandEncoderWithDescriptor:pass];
        if (!render) {
            [command endCommandBuffer];
            [capture stopCapture];
            return Failed("Metal 4 mesh render encoder");
        }
        [render setRenderPipelineState:mesh_pipeline];
        [render drawMeshThreadgroups:MTLSizeMake(1, 1, 1)
          threadsPerObjectThreadgroup:MTLSizeMake(1, 1, 1)
            threadsPerMeshThreadgroup:MTLSizeMake(1, 1, 1)];
        [render endEncoding];
    }

    [scaler encodeToCommandBuffer:command];
    [command endCommandBuffer];
    if (!SubmitAndWait(queue, command)) {
        [capture stopCapture];
        return 1;
    }

    // The smoke test waits for the first submission before reading private Spatial output.
    [command beginCommandBufferWithAllocator:allocator];
    [command useResidencySet:residency];
    id<MTL4ComputeCommandEncoder> readback = [command computeCommandEncoder];
    if (!readback) {
        [command endCommandBuffer];
        [capture stopCapture];
        return Failed("Metal 4 Spatial readback encoder");
    }
    [readback copyFromTexture:output_texture sourceSlice:0 sourceLevel:0
                  sourceOrigin:MTLOriginMake(31, 31, 0) sourceSize:MTLSizeMake(1, 1, 1)
                      toBuffer:spatial_result destinationOffset:0 destinationBytesPerRow:256
          destinationBytesPerImage:0];
    [readback endEncoding];
    [command endCommandBuffer];
    const bool finished = SubmitAndWait(queue, command);
    [capture stopCapture];
    if (!finished) {
        return 1;
    }
    std::printf("Metal 4 GPU result=0x%08x expected=0x%08x\n", *result_value, ExpectedComputeResult);
    if (*result_value != ExpectedComputeResult) {
        return Failed("Compute result comparison");
    }
    const auto* scaled_pixel = static_cast<const uint8_t*>(spatial_result.contents);
    constexpr std::array<uint8_t, 4> ExpectedPixel = {255, 0, 0, 255};
    std::printf("MetalFX Spatial pixel=%u,%u,%u,%u\n",
                scaled_pixel[0], scaled_pixel[1], scaled_pixel[2], scaled_pixel[3]);
    if (!std::equal(ExpectedPixel.begin(), ExpectedPixel.end(), scaled_pixel)) {
        return Failed("MetalFX Spatial pixel comparison");
    }

    if (mesh_output) {
        uint8_t pixel[4] = {};
        [mesh_output getBytes:pixel bytesPerRow:sizeof(pixel)
                   fromRegion:MTLRegionMake2D(1, 1, 1, 1) mipmapLevel:0];
        std::printf("Object/mesh pixel=%u,%u,%u,%u\n", pixel[0], pixel[1], pixel[2], pixel[3]);
        if (pixel[0] != 255 || pixel[1] != 0 || pixel[2] != 0 || pixel[3] != 255) {
            return Failed("Object/mesh pixel comparison");
        }
    }

    return 0;
}
} // namespace

int main(int argc, char** argv) {
    @autoreleasepool {
        NSArray<id<MTLDevice>>* devices = MTLCopyAllDevices();
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        std::printf("MTLCopyAllDevices=%lu MTLCreateSystemDefaultDevice=%s\n",
                    static_cast<unsigned long>(devices.count), device ? "present" : "nil");
        if (!device) {
            return 2;
        }
        if (![device supportsFamily:MTLGPUFamilyMetal4]) {
            return Failed("Metal 4 GPU family capability");
        }
        std::printf("device=%s unified=%d apple7=%d nativeSpatial=%d\n",
                    device.name.UTF8String, device.hasUnifiedMemory,
                    [device supportsFamily:MTLGPUFamilyApple7],
                    [MTLFXSpatialScalerDescriptor supportsMetal4FX:device]);
        if (argc != 3) {
            return Failed("Expected a compiled MSL library and GPU trace destination");
        }
        return RunWorkload(device, [NSString stringWithUTF8String:argv[1]],
                           [NSString stringWithUTF8String:argv[2]]);
    }
}
