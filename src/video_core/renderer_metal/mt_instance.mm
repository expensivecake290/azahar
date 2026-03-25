// Copyright Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include "video_core/renderer_metal/mt_instance.h"

#if defined(__APPLE__)

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <TargetConditionals.h>

#include "common/logging/log.h"
#include "core/frontend/emu_window.h"

namespace Metal {

struct Instance::Impl {
    id<MTLDevice> device = nil;
    id<MTL4CommandQueue> command_queue = nil;
    id<MTL4CommandAllocator> command_allocator = nil;
    id<MTL4Compiler> compiler = nil;
    bool supports_metal4 = false;
};

namespace {

static NSArray<id<MTLDevice>>* CopyAllDevices() {
#if TARGET_OS_OSX
    if (@available(macOS 10.13, *)) {
        return MTLCopyAllDevices();
    }
#endif
    return nil;
}

} // Anonymous namespace

Instance::Instance(Frontend::EmuWindow& window, u32 physical_device_index) : impl{} {
    (void)window;
    impl = std::make_unique<Impl>();

    NSArray<id<MTLDevice>>* devices = CopyAllDevices();
    if (devices != nil && physical_device_index < devices.count) {
        impl->device = [devices[physical_device_index] retain];
    } else {
        impl->device = [MTLCreateSystemDefaultDevice() retain];
    }

    if (devices != nil) {
        [devices release];
    }

    if (impl->device == nil) {
        LOG_CRITICAL(Render_Metal, "Failed to create Metal device");
        return;
    }

    if (@available(macOS 26.0, iOS 26.0, *)) {
        NSError* error = nil;

        MTL4CommandQueueDescriptor* queue_descriptor = [[MTL4CommandQueueDescriptor alloc] init];
        queue_descriptor.label = @"Azahar Metal Queue";
        impl->command_queue =
            [[impl->device newMTL4CommandQueueWithDescriptor:queue_descriptor error:&error] retain];
        [queue_descriptor release];
        if (impl->command_queue == nil) {
            LOG_CRITICAL(Render_Metal, "Failed to create Metal 4 command queue: {}",
                         error != nil ? error.localizedDescription.UTF8String : "unknown error");
            return;
        }

        MTL4CommandAllocatorDescriptor* allocator_descriptor =
            [[MTL4CommandAllocatorDescriptor alloc] init];
        impl->command_allocator =
            [[impl->device newCommandAllocatorWithDescriptor:allocator_descriptor error:&error]
                retain];
        [allocator_descriptor release];
        if (impl->command_allocator == nil) {
            LOG_CRITICAL(Render_Metal, "Failed to create Metal 4 command allocator: {}",
                         error != nil ? error.localizedDescription.UTF8String : "unknown error");
            return;
        }

        MTL4CompilerDescriptor* compiler_descriptor = [[MTL4CompilerDescriptor alloc] init];
        compiler_descriptor.label = @"Azahar Metal Compiler";
        impl->compiler =
            [[impl->device newCompilerWithDescriptor:compiler_descriptor error:&error] retain];
        [compiler_descriptor release];
        if (impl->compiler == nil) {
            LOG_CRITICAL(Render_Metal, "Failed to create Metal 4 compiler: {}",
                         error != nil ? error.localizedDescription.UTF8String : "unknown error");
            return;
        }

        impl->supports_metal4 = true;
    }

    if (!impl->supports_metal4) {
        LOG_CRITICAL(Render_Metal, "Metal 4 APIs are unavailable on this Apple platform");
    }
}

Instance::~Instance() {
    if (!impl) {
        return;
    }
    [impl->compiler release];
    [impl->command_allocator release];
    [impl->command_queue release];
    [impl->device release];
}

bool Instance::IsValid() const {
    return impl && impl->device != nil && impl->supports_metal4 && impl->command_queue != nil &&
           impl->command_allocator != nil && impl->compiler != nil;
}

bool Instance::SupportsMetal4() const {
    return impl && impl->supports_metal4;
}

void* Instance::GetDevice() const {
    return impl ? (__bridge void*)impl->device : nullptr;
}

void* Instance::GetCommandQueue() const {
    return impl ? (__bridge void*)impl->command_queue : nullptr;
}

void* Instance::GetCommandAllocator() const {
    return impl ? (__bridge void*)impl->command_allocator : nullptr;
}

void* Instance::GetCompiler() const {
    return impl ? (__bridge void*)impl->compiler : nullptr;
}

std::string Instance::GetDeviceName() const {
    if (!impl || impl->device == nil) {
        return "Unavailable";
    }
    return impl->device.name != nil ? impl->device.name.UTF8String : "Unnamed Metal Device";
}

} // namespace Metal

#endif
