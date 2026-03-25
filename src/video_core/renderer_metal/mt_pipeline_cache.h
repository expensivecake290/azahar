// Copyright Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <memory>
#include <string>

namespace Metal {

class Instance;

class PipelineCache {
public:
    explicit PipelineCache(const Instance& instance);
    ~PipelineCache();

    [[nodiscard]] bool IsValid() const;
    [[nodiscard]] void* GetPresentPipeline() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

} // namespace Metal
