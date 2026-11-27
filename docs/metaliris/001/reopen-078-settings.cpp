#include <cassert>
#include <cstdio>
#include "common/settings.h"

#ifndef ENABLE_METAL
#error The real native CMake configuration must enable the Metal settings
#endif

int main() {
    using Settings::GraphicsAPI;
    auto& values = Settings::values;
    assert(values.graphics_api.GetDefault() == GraphicsAPI::Metal);
    assert(Settings::GetWorkingGraphicsAPI() == GraphicsAPI::Metal);
    assert(!values.use_metalfx_spatial.GetValue());
    values.graphics_api.SetValue(GraphicsAPI::Vulkan);
    assert(Settings::GetWorkingGraphicsAPI() == GraphicsAPI::Metal);
    values.graphics_api.SetValue(GraphicsAPI::Software);
    assert(Settings::GetWorkingGraphicsAPI() == GraphicsAPI::Software);
    std::puts("PASS native CMake settings default, independent Spatial flag and existing Software selection");
}
