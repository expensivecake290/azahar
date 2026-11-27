#include <cassert>
#include <cstdio>
#include <QApplication>
#include <QCheckBox>
#include <QComboBox>
#include <QStandardItemModel>
#include "citra_qt/configuration/configure_graphics.h"
#include "citra_qt/util/metal_util.h"
#include "common/settings.h"

#ifdef METALIRIS_DETERMINISTIC_DEVICES
namespace MetalUtil {
std::vector<DeviceInfo> EnumerateDevicesForGraphicsFixture() {
    return {{"Primary", true, true}, {"Unsupported", false, false}, {"Other", true, false}};
}
} // namespace MetalUtil
#endif

int main(int argc, char** argv) {
    QApplication app(argc, argv);
    Settings::SetConfiguringGlobal(true);
    Settings::values.graphics_api.SetValue(Settings::GraphicsAPI::Metal);
    Settings::values.physical_device.SetValue(0);
    Settings::values.use_metalfx_spatial.SetValue(false);
    const auto filter = Settings::values.texture_sampling.GetValue();
    const auto internal_scale = Settings::values.resolution_factor.GetValue();
    const auto async = Settings::values.async_shader_compilation.GetValue();
    ConfigureGraphics global({}, {}, false);
    const auto api = global.findChild<QComboBox*>(QStringLiteral("graphics_api_combo"));
    const auto device = global.findChild<QComboBox*>(QStringLiteral("physical_device_combo"));
    const auto spatial = global.findChild<QCheckBox*>(QStringLiteral("toggle_metalfx_spatial"));
    const auto spirv = global.findChild<QCheckBox*>(QStringLiteral("spirv_shader_gen"));
    assert(api && device && spatial && spirv);
    assert(global.findChildren<QCheckBox*>(QStringLiteral("toggle_metalfx_spatial")).size() == 1);
    assert(api->count() == 4 && api->itemText(3) == QStringLiteral("Metal"));
    assert(api->currentText() == QStringLiteral("Metal") && spirv->isHidden());
    assert(!spatial->isChecked());
#ifdef METALIRIS_DETERMINISTIC_DEVICES
    assert(device->count() == 3);
    const auto* model = qobject_cast<QStandardItemModel*>(device->model());
    assert(model->item(0)->isEnabled() && !model->item(1)->isEnabled() && model->item(2)->isEnabled());
    assert(!spatial->isHidden() && spatial->isEnabled());
    device->setCurrentIndex(2);
    assert(!spatial->isEnabled());
    device->setCurrentIndex(0);
    spatial->click();
    global.ApplyConfiguration();
    assert(Settings::values.use_metalfx_spatial.GetValue());
    device->setCurrentIndex(2);
    assert(spatial->isChecked() && spatial->isEnabled());
    spatial->click();
    global.ApplyConfiguration();
    assert(!Settings::values.use_metalfx_spatial.GetValue() && !spatial->isEnabled());
    device->setCurrentIndex(0);
    spatial->click();
    global.ApplyConfiguration();
    assert(Settings::values.use_metalfx_spatial.GetValue());
    api->setCurrentIndex(static_cast<int>(Settings::GraphicsAPI::Software));
    assert(spatial->isHidden() && !spatial->isEnabled());
    global.ApplyConfiguration();
    assert(Settings::values.use_metalfx_spatial.GetValue());
    Settings::values.graphics_api.SetValue(Settings::GraphicsAPI::Metal);
    Settings::SetConfiguringGlobal(false);
    ConfigureGraphics per_game({}, {}, false);
    auto* inherited_api = per_game.findChild<QComboBox*>(QStringLiteral("graphics_api_combo"));
    auto* inherited_device = per_game.findChild<QComboBox*>(QStringLiteral("physical_device_combo"));
    auto* override_spatial = per_game.findChild<QCheckBox*>(QStringLiteral("toggle_metalfx_spatial"));
    assert(inherited_api->currentIndex() == 0 && inherited_device->currentIndex() == 0);
    const auto inherited_label = inherited_api->itemText(0);
    per_game.RetranslateUI();
    assert(inherited_api->itemText(0) == inherited_label);
    assert(inherited_api->itemText(3) == QStringLiteral("OpenGL"));
    assert(inherited_api->itemText(4) == QStringLiteral("Vulkan"));
    assert(inherited_api->itemText(5) == QStringLiteral("Metal"));
    assert(override_spatial->isEnabled() && override_spatial->isChecked());
    override_spatial->click();
    per_game.ApplyConfiguration();
    assert(!Settings::values.use_metalfx_spatial.UsingGlobal());
    assert(Settings::values.use_metalfx_spatial.GetValue(true));
    assert(!Settings::values.use_metalfx_spatial.GetValue());
    std::puts("PASS Qt device ordering and support, independent Spatial and per-game override");
#else
    if (device->count() == 0) {
        assert(!spatial->isEnabled());
        const auto* model = qobject_cast<QStandardItemModel*>(api->model());
        assert(!model->item(static_cast<int>(Settings::GraphicsAPI::Metal))->isEnabled());
        Settings::values.physical_device.SetValue(17);
        global.ApplyConfiguration();
        assert(Settings::values.physical_device.GetValue() == 17);
    }
    std::printf("PASS real Qt device availability and saved selection with %d native devices\n",
                device->count());
#endif
    assert(Settings::values.texture_sampling.GetValue() == filter);
    assert(Settings::values.resolution_factor.GetValue() == internal_scale);
    assert(Settings::values.async_shader_compilation.GetValue() == async);
}
