#import <Metal/Metal.h>
#include <concepts>
#include <type_traits>
#include "video_core/renderer_metal/mtl_common.h"

#if !__has_feature(objc_arc)
#error Objective-C ARC is required
#endif

using Context = Metal::DeviceContext;
static_assert(std::is_constructible_v<Context, id<MTLDevice>>);
static_assert(!std::is_copy_constructible_v<Context>);
static_assert(std::is_copy_constructible_v<Context::WorkLease>);
static_assert(std::is_move_constructible_v<Context::SubmissionGuard>);
static_assert(!std::is_copy_constructible_v<Context::SubmissionGuard>);
static_assert(!std::is_move_assignable_v<Context::SubmissionGuard>);
static_assert(requires(Context& context, const Context::WorkLease& lease) {
    { context.AcquireWork() } -> std::same_as<Context::WorkLease>;
    { context.GetLeaseStatus(lease) } -> std::same_as<Metal::WorkLeaseStatus>;
    { context.AcquireSubmission(lease) } -> std::same_as<std::optional<Context::SubmissionGuard>>;
    { context.AdvanceGeneration() } -> std::same_as<bool>;
    { context.Close() } -> std::same_as<void>;
});
static_assert(std::same_as<decltype(Metal::CheckResource(id<MTL4CommandQueue>{}, "", nil)),
                           id<MTL4CommandQueue>>);

int main() {}
