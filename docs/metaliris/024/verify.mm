#include "video_core/renderer_metal/mtl_scheduler.h"
#include "video_core/renderer_metal/mtl_stream_buffer.h"
#include <cassert>
#include <cstdio>
#include <limits>
#include <type_traits>

using namespace Metal;

#if defined(CHECK_NATIVE_CONSUMER)
void CheckUpload(StreamBuffer &stream, Scheduler &scheduler) {
  auto cpu = stream.Map(128, 48);
  cpu.Bytes()[0] = std::byte{0x4E};
  auto slice = stream.Commit(std::move(cpu), 64);
  auto gpu = stream.AcquireGPUAccess(slice, BufferAccess::Read);
  scheduler.Record([gpu = std::move(gpu)](EncodingContext &ctx) mutable {
    assert(gpu.Range().length == 64);
    ctx.Retain(std::move(gpu));
  });
}
#else
int main() {
  static_assert(!std::is_copy_constructible_v<StreamBuffer::Mapping>);
  static_assert(!std::is_copy_constructible_v<StreamBuffer>);
  static_assert(StreamRing::Locate(96, 1, 16, 48)->range.offset == 48);
  static_assert(StreamRing::Locate(96, 1, 16, 1)->range.offset == 1);
  static_assert(StreamRing::Locate(96, 32, 64, 0)->range.offset == 32);
  static_assert(StreamRing::Locate(96, 96, 0, 0)->range.offset == 96);
  static_assert(StreamRing::Locate(96, 96, 1, 0)->wrapped);
  static_assert(
      StreamRing::Locate(96, 1, 16, std::numeric_limits<u64>::max())->wrapped);
  static_assert(!StreamRing::Locate(96, 96, 97, 0));
  static_assert(!StreamRing::Locate(0, 0, 0, 0));
  static_assert(!StreamRing::Locate(96, 97, 0, 0));
  const auto aligned = StreamRing::Locate(4096, 4001, 8, 32);
  assert(aligned && !aligned->wrapped && aligned->range.offset == 4032);
  std::puts("PASS real stream consumer types, exact alignment and placement "
            "boundaries");
}
#endif
