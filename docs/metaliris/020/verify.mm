#include <concepts>
#include <type_traits>
#include "video_core/renderer_metal/mtl_resource_pool.h"
using namespace Metal;
static_assert(BindingLayout{31, 128, 16, true}.IsValid());
static_assert(!BindingLayout{32, 128, 16, false}.IsValid());
static_assert(!BindingLayout{31, 129, 16, false}.IsValid());
static_assert(!BindingLayout{31, 128, 17, false}.IsValid());
static_assert(!std::is_copy_constructible_v<ResourcePool::CommandLease>);
static_assert(!std::is_copy_constructible_v<ResourcePool::BindingLease>);
static_assert(requires(ResourcePool::CommandLease& command, ResourcePool::BindingLease& binding) {
    { command.Get() } -> std::same_as<id<MTL4CommandAllocator>>;
    { binding.Get() } -> std::same_as<id<MTL4ArgumentTable>>;
    { command.RetireWith(1) } -> std::same_as<std::shared_ptr<void>>;
    { binding.RetireWith(1) } -> std::same_as<std::shared_ptr<void>>;
});
int main() {}
