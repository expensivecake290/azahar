#include <concepts>
#include <limits>
#include <type_traits>
#include "video_core/renderer_metal/mtl_memory_util.h"

using namespace Metal;

static_assert(BufferRange{10, 3}.Fits(13));
static_assert(!BufferRange{10, 3}.Fits(12));
static_assert(!BufferRange{std::numeric_limits<std::size_t>::max(), 2}.Fits(4));
static_assert(!BufferRange{0, 0}.Fits(0));
static_assert(BufferRange{5, 10}.Overlaps({14, 2}));
static_assert(!BufferRange{5, 10}.Overlaps({15, 2}));
static_assert(!BufferRange{5, 10}.Overlaps({7, 0}));
static_assert(BufferRange{std::numeric_limits<std::size_t>::max(), 2}.Overlaps(
    {std::numeric_limits<std::size_t>::max() - 1, 3}));
static_assert(!std::is_copy_constructible_v<MemoryBudget::Reservation>);
static_assert(!std::is_copy_constructible_v<MemoryAllocator::GPUBufferAccess>);
static_assert(requires(MemoryAllocator::ReadMap& read, MemoryAllocator::WriteMap& write) {
    { read.Bytes() } -> std::same_as<std::span<const std::byte>>;
    { write.Bytes() } -> std::same_as<std::span<std::byte>>;
});
int main() {}
