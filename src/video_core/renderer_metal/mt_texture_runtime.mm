// Copyright Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include "video_core/renderer_metal/mt_texture_runtime.h"

#include "common/common_paths.h"
#include "common/file_util.h"

namespace Metal {

struct TextureRuntime::Impl {
    std::string cache_dir = FileUtil::GetUserPath(FileUtil::UserPath::ShaderDir) + "metal" + DIR_SEP;
    u64 current_title_id = 0;
};

TextureRuntime::TextureRuntime(const Instance& instance) : impl{std::make_unique<Impl>()} {
    (void)instance;
    FileUtil::CreateDir(FileUtil::GetUserPath(FileUtil::UserPath::ShaderDir));
    FileUtil::CreateDir(impl->cache_dir);
}

TextureRuntime::~TextureRuntime() = default;

void TextureRuntime::Finish() {}

void TextureRuntime::SwitchDiskResources(u64 title_id) {
    impl->current_title_id = title_id;
}

} // namespace Metal
