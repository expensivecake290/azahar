#include <cassert>
#include <cstdio>
#include <QApplication>
#include <QSettings>
#include <QTemporaryDir>
#include "citra_qt/configuration/config.h"
#include "common/file_util.h"
#include "common/logging/backend.h"

int main(int argc, char** argv) {
    QApplication app(argc, argv);
    QTemporaryDir directory;
    assert(directory.isValid());
    FileUtil::SetUserPath(directory.path().toStdString() + "/");
    Common::Log::Initialize();
    const auto config_dir = QString::fromStdString(FileUtil::GetUserPath(FileUtil::UserPath::ConfigDir));
    const auto global_path = config_dir + QStringLiteral("legacy.ini");
    {
        QSettings q(global_path, QSettings::IniFormat);
        q.setValue(QStringLiteral("Renderer/graphics_api"), static_cast<int>(Settings::GraphicsAPI::Vulkan));
        q.setValue(QStringLiteral("Renderer/physical_device"), 7);
        q.setValue(QStringLiteral("Renderer/resolution_factor"), 3);
        q.setValue(QStringLiteral("Renderer/async_shader_compilation"), true);
        q.setValue(QStringLiteral("Renderer/texture_sampling"), 2);
    }
    {
        auto& v = Settings::values;
        QtConfig config{"legacy"};
        assert(v.graphics_api.GetValue() == Settings::GraphicsAPI::Metal);
        assert(v.physical_device.GetValue() == 0);
        assert(!v.use_metalfx_spatial.GetValue());
        assert(v.resolution_factor.GetValue() == 3);
        assert(v.async_shader_compilation.GetValue());
        assert(static_cast<int>(v.texture_sampling.GetValue()) == 2);
        v.use_metalfx_spatial.SetValue(true);
        v.renderer_debug.SetValue(true);
        v.dump_command_buffers.SetValue(true);
        v.physical_device.SetValue(3);
        config.Save();
    }
    {
        QSettings saved(global_path, QSettings::IniFormat);
        assert(saved.value(QStringLiteral("Renderer/use_metalfx_spatial")).toBool());
        assert(saved.value(QStringLiteral("Debugging/dump_command_buffers")).toBool());
    }
    {
        auto& v = Settings::values;
        QtConfig verified{"legacy"};
        assert(v.graphics_api.GetValue() == Settings::GraphicsAPI::Metal);
        assert(v.physical_device.GetValue() == 3);
        assert(v.use_metalfx_spatial.GetValue() && v.renderer_debug.GetValue());
        assert(v.dump_command_buffers.GetValue());
        v.dump_command_buffers.SetValue(false);
        verified.Save();
    }
    {
        QtConfig matching{"legacy"};
        auto& v = Settings::values;
        assert(!v.dump_command_buffers.GetValue());
        {
            QSettings q(config_dir + QStringLiteral("custom/game.ini"), QSettings::IniFormat);
            q.setValue(QStringLiteral("Renderer/graphics_api"), static_cast<int>(Settings::GraphicsAPI::Vulkan));
            q.setValue(QStringLiteral("Renderer/graphics_api/use_global"), false);
            q.setValue(QStringLiteral("Renderer/physical_device/use_global"), true);
        }
        {
            QtConfig per_game{"game", QtConfig::ConfigType::PerGameConfig};
            assert(Settings::values.graphics_api.GetValue() == Settings::GraphicsAPI::Metal);
            assert(!Settings::values.graphics_api.UsingGlobal());
            assert(v.physical_device.GetValue() == 3 && v.physical_device.UsingGlobal());
            assert(v.use_metalfx_spatial.GetValue() && v.use_metalfx_spatial.UsingGlobal());
            v.use_metalfx_spatial.SetGlobal(false);
            v.use_metalfx_spatial.SetValue(false);
            per_game.Save();
            matching.Save();
            QSettings global_check(global_path, QSettings::IniFormat);
            assert(global_check.value(QStringLiteral("Renderer/use_metalfx_spatial")).toBool());
        }
        {
            QtConfig per_game{"game", QtConfig::ConfigType::PerGameConfig};
            assert(!v.use_metalfx_spatial.GetValue() && !v.use_metalfx_spatial.UsingGlobal());
            assert(v.use_metalfx_spatial.GetValue(true));
            assert(v.resolution_factor.GetValue() == 3);
            assert(v.async_shader_compilation.GetValue());
            assert(static_cast<int>(v.texture_sampling.GetValue()) == 2);
            v.use_metalfx_spatial.SetGlobal(true);
            per_game.Save();
        }
        {
            QtConfig per_game{"game", QtConfig::ConfigType::PerGameConfig};
            assert(v.use_metalfx_spatial.GetValue() && v.use_metalfx_spatial.UsingGlobal());
        }
        v.graphics_api.SetGlobal(true);
    }
    std::puts("PASS native Qt INI migration, global/per-game Spatial, independent texture sampling and debugger trace settings");
}
