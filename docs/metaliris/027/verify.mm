#include "video_core/renderer_metal/mtl_descriptor_update_queue.h"
#include "video_core/renderer_metal/mtl_scheduler.h"
#include <array>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <limits>
#include <stdexcept>

using namespace Metal;
using namespace std::chrono_literals;

namespace {

auto Deadline() { return std::chrono::steady_clock::now() + 10s; }

template <typename Exception, typename T> bool Throws(T &&callback) {
  try {
    callback();
    return false;
  } catch (const Exception &) {
    return true;
  }
}

id<MTLComputePipelineState> CreatePipeline(id<MTLDevice> device,
                                           const char *library_path) {
  NSError *error = nil;
  auto library =
      [device newLibraryWithURL:[NSURL fileURLWithPath:@(library_path)]
                          error:&error];
  library = CheckResource(library, "Load descriptor test shader", error);
  auto function = [MTL4LibraryFunctionDescriptor new];
  function.library = library;
  function.name = @"verify_generation";
  auto compiler = [device newCompilerWithDescriptor:[MTL4CompilerDescriptor new]
                                              error:&error];
  compiler =
      CheckResource(compiler, "Create Metal descriptor test compiler", error);
  auto descriptor = [MTL4ComputePipelineDescriptor new];
  descriptor.computeFunctionDescriptor = function;
  auto pipeline = [compiler newComputePipelineStateWithDescriptor:descriptor
                                              compilerTaskOptions:nil
                                                            error:&error];
  return CheckResource(pipeline, "Create Metal descriptor test pipeline",
                       error);
}

void CheckGenerations(id<MTLDevice> native,
                      id<MTLComputePipelineState> pipeline) {
  DeviceContext device(native);
  Scheduler scheduler(device, 1, 1);
  assert(Throws<std::invalid_argument>(
      [&] { DescriptorUpdateQueue invalid(device, 0); }));
  DescriptorUpdateQueue queue(device, 1);
  auto target = queue.CreateSet({2, 3, 2, false});
  auto empty = queue.Capture(target);
  queue.Flush();
  assert(queue.Capture(target).Generation() == empty.Generation());
  assert(empty.Layout() == target.Layout());

  const auto options =
      MTLResourceStorageModeShared | MTLResourceHazardTrackingModeUntracked;
  auto src = CheckResource([native newBufferWithLength:32 options:options],
                           "descriptor src");
  auto dst = CheckResource([native newBufferWithLength:32 options:options],
                           "descriptor dst");
  auto *words = static_cast<u32 *>(src.contents);
  words[0] = 40;
  words[4] = 150;
  static_cast<u32 *>(dst.contents)[0] = static_cast<u32 *>(dst.contents)[4] = 0;

  auto sampler_desc = [MTLSamplerDescriptor new];
  sampler_desc.minFilter = MTLSamplerMinMagFilterNearest;
  sampler_desc.magFilter = MTLSamplerMinMagFilterNearest;
  sampler_desc.supportArgumentBuffers = YES;
  auto sampler = CheckResource(
      [native newSamplerStateWithDescriptor:sampler_desc], "test sampler");
  auto *texture_desc = [MTLTextureDescriptor
      texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                   width:1
                                  height:1
                               mipmapped:NO];
  texture_desc.storageMode = MTLStorageModeShared;
  texture_desc.usage = MTLTextureUsageShaderRead;
  auto image = CheckResource([native newTextureWithDescriptor:texture_desc],
                             "test sampled texture");
  const std::array<u8, 4> red = {255, 0, 0, 255};
  [image replaceRegion:MTLRegionMake2D(0, 0, 1, 1)
           mipmapLevel:0
             withBytes:red.data()
           bytesPerRow:red.size()];

  texture_desc.pixelFormat = MTLPixelFormatRGBA8Uint;
  texture_desc.usage = MTLTextureUsageShaderWrite;
  auto storage = CheckResource([native newTextureWithDescriptor:texture_desc],
                               "test storage image");
  const auto format = MTLPixelFormatR32Uint;
  const auto row = [native minimumTextureBufferAlignmentForPixelFormat:format];
  assert(row >= sizeof(u32));
  auto backing = CheckResource([native newBufferWithLength:row options:options],
                               "texel storage");
  *static_cast<u32 *>(backing.contents) = 7;
  auto *lookup_desc = [MTLTextureDescriptor
      textureBufferDescriptorWithPixelFormat:format
                                       width:1
                             resourceOptions:options
                                       usage:MTLTextureUsageShaderRead];
  auto lookup = CheckResource([backing newTextureWithDescriptor:lookup_desc
                                                         offset:0
                                                    bytesPerRow:row],
                              "typed texel view");

  assert(Throws<std::out_of_range>(
      [&] { queue.AddBuffer(target, 0, src, 30, 4); }));
  assert(
      Throws<std::out_of_range>([&] { queue.AddBuffer(target, 2, src, 0); }));
  assert(Throws<std::invalid_argument>(
      [&] { queue.AddBuffer(target, 0, nil, 1); }));
  assert(Throws<std::invalid_argument>(
      [&] { queue.AddTexelBuffer(target, 1, image); }));
  assert(Throws<std::out_of_range>([&] {
    queue.AddImageSampler(target, 0, std::numeric_limits<u32>::max(), image,
                          sampler);
  }));
  queue.AddBuffer(target, 0, src, 0, 4);
  queue.AddBuffer(target, 1, dst, 0, 4, BufferAccess::Write);
  queue.AddImageSampler(target, 0, 0, image, sampler, 1);
  queue.AddTexelBuffer(target, 1, lookup);
  queue.AddStorageImage(target, 2, storage, BufferAccess::Write);
  auto first = queue.Capture(target);
  queue.AddBuffer(target, 0, src, 16, 4);
  queue.AddBuffer(target, 1, dst, 16, 4, BufferAccess::Write);
  auto second = queue.Capture(target);
  assert(first.Generation() != second.Generation());

  scheduler.Record([first, second, pipeline](EncodingContext &context) {
    auto encoder = [context.GetCommandBuffer() computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    first.BindCompute(context, encoder);
    [encoder dispatchThreads:MTLSizeMake(1, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    second.BindCompute(context, encoder);
    [encoder dispatchThreads:MTLSizeMake(1, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    context.Retain(pipeline);
    context.EndPass(encoder);
  });
  const auto issued = scheduler.Flush();
  queue.AddBuffer(target, 0, nil, 0);
  queue.AddImageSampler(target, 0, 0, nil, nil, 1);
  queue.AddTexelBuffer(target, 1, nil);
  queue.AddStorageImage(target, 2, nil);
  assert(queue.Capture(target).Generation() > second.Generation());
  assert(scheduler.Wait(issued, Deadline()) == CompletionStatus::Complete);
  const auto *result = static_cast<const u32 *>(dst.contents);
  assert(result[0] == 63 && result[4] == 173);
  std::array<u8, 4> texture_result{};
  [storage getBytes:texture_result.data()
        bytesPerRow:texture_result.size()
         fromRegion:MTLRegionMake2D(0, 0, 1, 1)
        mipmapLevel:0];
  assert((texture_result == std::array<u8, 4>{173, 173, 173, 173}));
  assert(scheduler.Shutdown(Deadline()));
}

void CheckDeviceIdentity(id<MTLDevice> native) {
  DeviceContext source(native);
  DeviceContext destination(native);
  DescriptorUpdateQueue queue(source);
  auto set = queue.CreateSet({1, 0, 0});
  const auto prior = queue.Capture(set);
  DescriptorUpdateQueue foreign(destination);
  assert(Throws<std::invalid_argument>(
      [&] { static_cast<void>(foreign.Capture(set)); }));
  {
    Scheduler scheduler(destination);
    scheduler.Record([prior](EncodingContext &context) {
      auto encoder = [context.GetCommandBuffer() computeCommandEncoder];
      assert(Throws<std::invalid_argument>(
          [&] { prior.BindCompute(context, encoder); }));
      context.EndPass(encoder);
    });
    assert(scheduler.Finish({}, Deadline()) == CompletionStatus::Complete);
  }
  assert(source.AdvanceGeneration());
  assert(Throws<std::invalid_argument>(
      [&] { static_cast<void>(queue.Capture(set)); }));
  Scheduler scheduler(source);
  scheduler.Record([prior](EncodingContext &context) {
    auto encoder = [context.GetCommandBuffer() computeCommandEncoder];
    assert(Throws<std::invalid_argument>(
        [&] { prior.BindCompute(context, encoder); }));
    context.EndPass(encoder);
  });
  assert(scheduler.Finish({}, Deadline()) == CompletionStatus::Complete);
}

} // Anonymous namespace

int main(int argc, char **argv) {
  @autoreleasepool {
    assert(argc == 2);
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device || ![device supportsFamily:MTLGPUFamilyMetal4]) {
      std::puts("PENDING descriptor generation mutation, cache exhaustion, and "
                "context checks on Metal 4 GPU");
      return 2;
    }
    CheckGenerations(device, CreatePipeline(device, argv[1]));
    CheckDeviceIdentity(device);
    std::puts("PASS exact immutable descriptor generations, native table "
              "fallback and device identity");
  }
}
