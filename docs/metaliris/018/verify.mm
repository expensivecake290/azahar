#include <concepts>
#include <type_traits>
#include "video_core/renderer_metal/mtl_master_semaphore.h"

using namespace Metal;
static_assert(!std::is_copy_constructible_v<CompletionTimeline>);
static_assert(!std::is_default_constructible_v<MasterSemaphore>);
static_assert(std::is_constructible_v<MasterSemaphore, id<MTLDevice>>);
static_assert(requires(MasterSemaphore& semaphore, id<MTL4CommandQueue> queue) {
    { semaphore.GetDevice() } -> std::same_as<id<MTLDevice>>;
    { semaphore.Issue() } -> std::same_as<u64>;
    { semaphore.MakeCompletionHandler(1) } -> std::same_as<MTL4CommitFeedbackHandler>;
    { semaphore.WaitOnQueue(queue, 1) } -> std::same_as<CompletionStatus>;
});
int main() {
    return CompletionStatus::Complete == CompletionStatus::Faulted;
}
