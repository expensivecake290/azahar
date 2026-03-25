// Copyright Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include "video_core/renderer_metal/mt_texture_runtime.h"

#if defined(__APPLE__)

#import <Metal/Metal.h>

#include <array>

#include "common/common_paths.h"
#include "common/file_util.h"
#include "common/hash.h"
#include "video_core/renderer_metal/mt_instance.h"

namespace Metal {

namespace {

struct DisplaySurfaceDesc {
    PAddr addr = 0;
    u32 size = 0;
    u32 width = 0;
    u32 height = 0;
    u32 pixel_stride = 0;
    Pica::PixelFormat pixel_format = Pica::PixelFormat::RGBA8;

    bool operator==(const DisplaySurfaceDesc& other) const = default;
};

struct DisplayBufferSlot {
    id<MTLBuffer> buffer = nil;
    DisplaySurfaceDesc desc{};
    u64 content_hash = 0;
};

static void ReleaseBuffer(DisplayBufferSlot& slot) {
    [slot.buffer release];
    slot.buffer = nil;
    slot.desc = {};
    slot.content_hash = 0;
}

static bool EnsureBuffer(DisplayBufferSlot& slot, id<MTLDevice> device, u32 size) {
    if (slot.buffer != nil && [slot.buffer length] >= size) {
        return true;
    }

    [slot.buffer release];
    slot.buffer = [device newBufferWithLength:size options:MTLResourceStorageModeShared];
    return slot.buffer != nil;
}

static u64 ComputeContentHash(const u8* pixels, u32 size) {
    return Common::ComputeHash64<Common::HashAlgo64::CityHash>(pixels, size);
}

} // Anonymous namespace

struct TextureRuntime::Impl {
    explicit Impl(const Instance& instance_)
        : device{(__bridge id<MTLDevice>)instance_.GetDevice()},
          cache_dir{FileUtil::GetUserPath(FileUtil::UserPath::ShaderDir) + "metal" + DIR_SEP} {}

    id<MTLDevice> device = nil;
    std::string cache_dir;
    u64 current_title_id = 0;
    std::array<DisplayBufferSlot, 3> display_buffers{};
};

TextureRuntime::TextureRuntime(const Instance& instance) : impl{std::make_unique<Impl>(instance)} {
    FileUtil::CreateDir(FileUtil::GetUserPath(FileUtil::UserPath::ShaderDir));
    FileUtil::CreateDir(impl->cache_dir);
}

TextureRuntime::~TextureRuntime() {
    Finish();
}

void TextureRuntime::Finish() {
    if (!impl) {
        return;
    }
    for (auto& buffer : impl->display_buffers) {
        ReleaseBuffer(buffer);
    }
}

void TextureRuntime::SwitchDiskResources(u64 title_id) {
    impl->current_title_id = title_id;
}

DisplayTextureInfo TextureRuntime::SynchronizeDisplaySurface(u32 index, PAddr addr, const u8* pixels,
                                                             u32 size, u32 width, u32 height,
                                                             u32 pixel_stride,
                                                             Pica::PixelFormat pixel_format) {
    if (!impl || impl->device == nil || pixels == nullptr || size == 0 || width == 0 ||
        height == 0 || index >= impl->display_buffers.size()) {
        return {};
    }

    auto& slot = impl->display_buffers[index];
    const DisplaySurfaceDesc new_desc{
        .addr = addr,
        .size = size,
        .width = width,
        .height = height,
        .pixel_stride = pixel_stride,
        .pixel_format = pixel_format,
    };

    if (!EnsureBuffer(slot, impl->device, size)) {
        return {};
    }

    const u64 new_hash = ComputeContentHash(pixels, size);
    if (!(slot.desc == new_desc) || slot.content_hash != new_hash) {
        std::memcpy([slot.buffer contents], pixels, size);
        slot.desc = new_desc;
        slot.content_hash = new_hash;
    }

    return {
        .buffer = (__bridge void*)slot.buffer,
        .width = width,
        .height = height,
        .pixel_stride = pixel_stride,
        .pixel_format = pixel_format,
        .texcoords = {0.0f, 0.0f, 1.0f, 1.0f},
        .valid = true,
    };
}

void TextureRuntime::ClearDisplayTextures() {
    if (!impl) {
        return;
    }
    for (auto& buffer : impl->display_buffers) {
        ReleaseBuffer(buffer);
    }
}

} // namespace Metal

#endif
