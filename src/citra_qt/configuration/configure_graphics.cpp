// Copyright 2016-2026 Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.txt file included.

#include <QColorDialog>
#include <QStandardItemModel>
#include "citra_qt/configuration/configuration_shared.h"
#include "citra_qt/configuration/configure_graphics.h"
#include "common/settings.h"
#include "ui_configure_graphics.h"
#ifdef ENABLE_METAL
#include "citra_qt/util/metal_util.h"
#endif
#ifdef ENABLE_VULKAN
#include "video_core/renderer_vulkan/vk_instance.h"
#endif

namespace {

int PerGameOffset() {
    return Settings::IsConfiguringGlobal() ? 0 : ConfigurationShared::USE_GLOBAL_OFFSET;
}

Settings::GraphicsAPI EffectiveGraphicsAPI(int index) {
    if (!Settings::IsConfiguringGlobal() && index == ConfigurationShared::USE_GLOBAL_INDEX) {
        return Settings::values.graphics_api.GetValue(true);
    }
    return static_cast<Settings::GraphicsAPI>(index - PerGameOffset());
}

} // Anonymous namespace

ConfigureGraphics::ConfigureGraphics(QString gl_renderer, std::span<const QString> physical_devices,
                                     bool is_powered_on, QWidget* parent)
    : QWidget(parent), ui(std::make_unique<Ui::ConfigureGraphics>()) {
    ui->setupUi(this);

#ifdef ENABLE_METAL
    const auto devices = MetalUtil::EnumerateDevices();
    for (const auto& device : devices) {
        ui->physical_device_combo->addItem(QString::fromStdString(device.name),
                                           device.spatial_scaling);
    }
#else
    ui->graphics_api_combo->removeItem(static_cast<int>(Settings::GraphicsAPI::Metal));
    ui->toggle_metalfx_spatial->hide();
#ifdef ENABLE_VULKAN
    for (const QString& name : physical_devices) {
        ui->physical_device_combo->addItem(name);
    }
#endif
#endif

    SetupPerGameUI();

    ui->graphics_api_combo->setEnabled(!is_powered_on && ui->graphics_api_combo->isEnabled());
    ui->physical_device_combo->setEnabled(!is_powered_on && ui->physical_device_combo->isEnabled());
    ui->toggle_accurate_mul->setEnabled(!is_powered_on && ui->toggle_accurate_mul->isEnabled());
    ui->toggle_async_shaders->setEnabled(!is_powered_on && ui->toggle_async_shaders->isEnabled());
    ui->toggle_async_present->setEnabled(!is_powered_on && ui->toggle_async_present->isEnabled());
    ui->toggle_display_refresh_rate_detection->setEnabled(!is_powered_on);
    // Set the index to -1 to ensure the below lambda is called with setCurrentIndex
    ui->graphics_api_combo->setCurrentIndex(-1);

    const auto width = static_cast<int>(QString::fromStdString("000000000").size() * 6);
    ui->delay_render_display_label->setMinimumWidth(width);
    ui->delay_render_combo->setVisible(!Settings::IsConfiguringGlobal());

    auto graphics_api_combo_model =
        qobject_cast<QStandardItemModel*>(ui->graphics_api_combo->model());
    const auto disable_api = [graphics_api_combo_model](Settings::GraphicsAPI api) {
        auto* item = graphics_api_combo_model->item(static_cast<int>(api) + PerGameOffset());
        item->setEnabled(false);
    };
#ifndef ENABLE_SOFTWARE_RENDERER
    disable_api(Settings::GraphicsAPI::Software);
#endif

#ifndef ENABLE_OPENGL
    disable_api(Settings::GraphicsAPI::OpenGL);
#else
    ui->opengl_renderer_name_label->setText(gl_renderer);
#endif

#ifndef ENABLE_VULKAN
    disable_api(Settings::GraphicsAPI::Vulkan);
#else
    if (physical_devices.empty()) {
        disable_api(Settings::GraphicsAPI::Vulkan);
    }
#endif
#ifdef ENABLE_METAL
    auto* device_model = qobject_cast<QStandardItemModel*>(ui->physical_device_combo->model());
    bool has_compatible_device = false;
    for (int i = 0; i < static_cast<int>(devices.size()); ++i) {
        auto* item = device_model->item(i + PerGameOffset());
        item->setEnabled(devices[i].supported);
        has_compatible_device |= devices[i].supported;
    }
    if (!has_compatible_device) {
        disable_api(Settings::GraphicsAPI::Metal);
    }
    connect(ui->physical_device_combo, &QComboBox::currentIndexChanged, this,
            [this] { SetPhysicalDeviceComboVisibility(ui->graphics_api_combo->currentIndex()); });
    connect(ui->toggle_metalfx_spatial, &QCheckBox::toggled, this, [this] {
        UpdateMetalFXSpatialAvailability(
            EffectiveGraphicsAPI(ui->graphics_api_combo->currentIndex()) ==
            Settings::GraphicsAPI::Metal);
    });
#endif

    connect(ui->graphics_api_combo, qOverload<int>(&QComboBox::currentIndexChanged), this,
            [this](int index) {
                const auto graphics_api = EffectiveGraphicsAPI(index);
                const bool is_software = graphics_api == Settings::GraphicsAPI::Software;

                ui->hw_renderer_group->setEnabled(!is_software);
                ui->toggle_disk_shader_cache->setEnabled(!is_software &&
                                                         ui->toggle_hw_shader->isChecked());
            });

    connect(ui->toggle_hw_shader, &QCheckBox::toggled, this, [this] {
        const bool enabled = ui->toggle_hw_shader->isEnabled();
        const bool checked = ui->toggle_hw_shader->isChecked();
        ui->hw_shader_group->setEnabled(checked && enabled);
        ui->toggle_disk_shader_cache->setEnabled(checked && enabled);
    });

    connect(ui->graphics_api_combo, qOverload<int>(&QComboBox::currentIndexChanged), this,
            &ConfigureGraphics::SetPhysicalDeviceComboVisibility);

    connect(ui->delay_render_slider, &QSlider::valueChanged, this, [&](int value) {
        ui->delay_render_display_label->setText(
            QStringLiteral("%1 ms")
                .arg(((double)value) / 1000.f, 0, 'f', 3)
                .rightJustified(QString::fromStdString("000000000").size()));
    });

    SetConfiguration();
}

ConfigureGraphics::~ConfigureGraphics() = default;

void ConfigureGraphics::SetConfiguration() {
    ui->delay_render_slider->setValue(Settings::values.delay_game_render_thread_us.GetValue());
    ui->delay_render_display_label->setText(
        QStringLiteral("%1 ms")
            .arg(((double)ui->delay_render_slider->value()) / 1000, 0, 'f', 3)
            .rightJustified(QString::fromStdString("000000000").size()));

    if (!Settings::IsConfiguringGlobal()) {
        ConfigurationShared::SetHighlight(ui->graphics_api_group,
                                          !Settings::values.graphics_api.UsingGlobal());
        ConfigurationShared::SetPerGameSetting(ui->graphics_api_combo,
                                               &Settings::values.graphics_api);
        ConfigurationShared::SetHighlight(ui->physical_device_group,
                                          !Settings::values.physical_device.UsingGlobal());
        ConfigurationShared::SetPerGameSetting(ui->physical_device_combo,
                                               &Settings::values.physical_device);
        ConfigurationShared::SetPerGameSetting(ui->texture_sampling_combobox,
                                               &Settings::values.texture_sampling);
        ConfigurationShared::SetHighlight(ui->widget_texture_sampling,
                                          !Settings::values.texture_sampling.UsingGlobal());
        ConfigurationShared::SetHighlight(
            ui->delay_render_layout, !Settings::values.delay_game_render_thread_us.UsingGlobal());

        if (Settings::values.delay_game_render_thread_us.UsingGlobal()) {
            ui->delay_render_combo->setCurrentIndex(0);
            ui->delay_render_slider->setEnabled(false);
        } else {
            ui->delay_render_combo->setCurrentIndex(1);
            ui->delay_render_slider->setEnabled(true);
        }
    } else {
        ui->graphics_api_combo->setCurrentIndex(
            static_cast<int>(Settings::values.graphics_api.GetValue()));
        ui->physical_device_combo->setCurrentIndex(
            static_cast<int>(Settings::values.physical_device.GetValue()));
        ui->texture_sampling_combobox->setCurrentIndex(
            static_cast<int>(Settings::values.texture_sampling.GetValue()));
    }

    ui->toggle_hw_shader->setChecked(Settings::values.use_hw_shader.GetValue());
    ui->toggle_accurate_mul->setChecked(Settings::values.shaders_accurate_mul.GetValue());
    ui->toggle_disk_shader_cache->setChecked(Settings::values.use_disk_shader_cache.GetValue());
    ui->toggle_vsync->setChecked(Settings::values.use_vsync.GetValue());
    ui->toggle_limit_frames_ahead_of_gpu->setChecked(
        Settings::values.use_limit_frames_ahead_of_gpu.GetValue());
    ui->toggle_skip_duplicate_frames->setChecked(
        Settings::values.use_skip_duplicate_frames.GetValue());
    ui->spirv_shader_gen->setChecked(Settings::values.spirv_shader_gen.GetValue());
    ui->disable_spirv_optimizer->setChecked(Settings::values.disable_spirv_optimizer.GetValue());
    ui->toggle_async_shaders->setChecked(Settings::values.async_shader_compilation.GetValue());
    ui->toggle_async_present->setChecked(Settings::values.async_presentation.GetValue());

#ifdef ENABLE_METAL
    ui->toggle_metalfx_spatial->setChecked(Settings::values.use_metalfx_spatial.GetValue());
    SetPhysicalDeviceComboVisibility(ui->graphics_api_combo->currentIndex());
#endif

    if (Settings::IsConfiguringGlobal()) {
        ui->toggle_shader_jit->setChecked(Settings::values.use_shader_jit.GetValue());
        ui->toggle_display_refresh_rate_detection->setChecked(
            Settings::values.use_display_refresh_rate_detection.GetValue());
    }
    ui->simulate_3ds_gpu_timings->setChecked(Settings::values.simulate_3ds_gpu_timings.GetValue());
}

void ConfigureGraphics::ApplyConfiguration() {
    ConfigurationShared::ApplyPerGameSetting(&Settings::values.graphics_api,
                                             ui->graphics_api_combo);
#ifdef ENABLE_METAL
    if (ui->physical_device_combo->count() > PerGameOffset() &&
        ui->physical_device_combo->currentIndex() >= 0) {
        ConfigurationShared::ApplyPerGameSetting(&Settings::values.physical_device,
                                                 ui->physical_device_combo);
    }
    const auto* spatial = ui->toggle_metalfx_spatial;
    if ((!spatial->isChecked() || spatial->isEnabled()) &&
        EffectiveGraphicsAPI(ui->graphics_api_combo->currentIndex()) ==
            Settings::GraphicsAPI::Metal) {
        ConfigurationShared::ApplyPerGameSetting(&Settings::values.use_metalfx_spatial, spatial,
                                                 use_metalfx_spatial);
    }
#else
    ConfigurationShared::ApplyPerGameSetting(&Settings::values.physical_device,
                                             ui->physical_device_combo);
#endif
    ConfigurationShared::ApplyPerGameSetting(&Settings::values.async_shader_compilation,
                                             ui->toggle_async_shaders, async_shader_compilation);
    ConfigurationShared::ApplyPerGameSetting(&Settings::values.async_presentation,
                                             ui->toggle_async_present, async_presentation);
    ConfigurationShared::ApplyPerGameSetting(&Settings::values.spirv_shader_gen,
                                             ui->spirv_shader_gen, spirv_shader_gen);
    ConfigurationShared::ApplyPerGameSetting(&Settings::values.disable_spirv_optimizer,
                                             ui->disable_spirv_optimizer, disable_spirv_optimizer);
    ConfigurationShared::ApplyPerGameSetting(&Settings::values.use_hw_shader, ui->toggle_hw_shader,
                                             use_hw_shader);
    ConfigurationShared::ApplyPerGameSetting(&Settings::values.shaders_accurate_mul,
                                             ui->toggle_accurate_mul, shaders_accurate_mul);
    ConfigurationShared::ApplyPerGameSetting(&Settings::values.texture_sampling,
                                             ui->texture_sampling_combobox);
    ConfigurationShared::ApplyPerGameSetting(&Settings::values.use_disk_shader_cache,
                                             ui->toggle_disk_shader_cache, use_disk_shader_cache);
    ConfigurationShared::ApplyPerGameSetting(&Settings::values.use_vsync, ui->toggle_vsync,
                                             use_vsync);
    ConfigurationShared::ApplyPerGameSetting(&Settings::values.use_skip_duplicate_frames,
                                             ui->toggle_skip_duplicate_frames,
                                             use_skip_duplicate_frames);
    ConfigurationShared::ApplyPerGameSetting(&Settings::values.use_limit_frames_ahead_of_gpu,
                                             ui->toggle_limit_frames_ahead_of_gpu,
                                             use_limit_frames_ahead_of_gpu);
    ConfigurationShared::ApplyPerGameSetting(
        &Settings::values.delay_game_render_thread_us, ui->delay_render_combo,
        [this](s32) { return ui->delay_render_slider->value(); });
    ConfigurationShared::ApplyPerGameSetting(&Settings::values.simulate_3ds_gpu_timings,
                                             ui->simulate_3ds_gpu_timings,
                                             simulate_3ds_gpu_timings);

    if (Settings::IsConfiguringGlobal()) {
        Settings::values.use_shader_jit = ui->toggle_shader_jit->isChecked();
        Settings::values.use_display_refresh_rate_detection =
            ui->toggle_display_refresh_rate_detection->isChecked();
    }
}

void ConfigureGraphics::RetranslateUI() {
    auto* api_combo = ui->graphics_api_combo;
    const int offset = PerGameOffset();
    const auto inherited_label = api_combo->itemText(ConfigurationShared::USE_GLOBAL_INDEX);
    ui->retranslateUi(this);
    if (offset) {
        // UIC translates Software. Backend names are constants in the form.
        api_combo->setItemText(offset, api_combo->itemText(0));
        api_combo->setItemText(ConfigurationShared::USE_GLOBAL_INDEX, inherited_label);
    }
}

void ConfigureGraphics::SetupPerGameUI() {
    // Block the global settings if a game is currently running that overrides them
    if (Settings::IsConfiguringGlobal()) {
        ui->graphics_api_group->setEnabled(Settings::values.graphics_api.UsingGlobal());
        ui->toggle_hw_shader->setEnabled(Settings::values.use_hw_shader.UsingGlobal());
        ui->toggle_accurate_mul->setEnabled(Settings::values.shaders_accurate_mul.UsingGlobal());
        ui->toggle_disk_shader_cache->setEnabled(
            Settings::values.use_disk_shader_cache.UsingGlobal());
        ui->toggle_vsync->setEnabled(ui->toggle_vsync->isEnabled() &&
                                     Settings::values.use_vsync.UsingGlobal());
        ui->toggle_skip_duplicate_frames->setEnabled(
            ui->toggle_skip_duplicate_frames->isEnabled() &&
            Settings::values.use_skip_duplicate_frames.UsingGlobal());
        ui->toggle_limit_frames_ahead_of_gpu->setEnabled(
            ui->toggle_limit_frames_ahead_of_gpu->isEnabled() &&
            Settings::values.use_limit_frames_ahead_of_gpu.UsingGlobal());
        ui->toggle_async_shaders->setEnabled(
            Settings::values.async_shader_compilation.UsingGlobal());
        ui->widget_texture_sampling->setEnabled(Settings::values.texture_sampling.UsingGlobal());
        ui->toggle_async_present->setEnabled(Settings::values.async_presentation.UsingGlobal());
        ui->graphics_api_combo->setEnabled(Settings::values.graphics_api.UsingGlobal());
        ui->physical_device_combo->setEnabled(Settings::values.physical_device.UsingGlobal());
        ui->delay_render_combo->setEnabled(
            Settings::values.delay_game_render_thread_us.UsingGlobal());
        ui->simulate_3ds_gpu_timings->setEnabled(
            Settings::values.simulate_3ds_gpu_timings.UsingGlobal());
        return;
    }

    connect(ui->delay_render_combo, qOverload<int>(&QComboBox::activated), this, [this](int index) {
        ui->delay_render_slider->setEnabled(index == 1);
        ConfigurationShared::SetHighlight(ui->delay_render_layout, index == 1);
    });

    ui->toggle_shader_jit->setVisible(false);
    ui->toggle_display_refresh_rate_detection->setVisible(false);

    ConfigurationShared::SetColoredComboBox(
        ui->graphics_api_combo, ui->graphics_api_group,
        static_cast<u32>(Settings::values.graphics_api.GetValue(true)));

    ConfigurationShared::SetColoredComboBox(
        ui->physical_device_combo, ui->physical_device_group,
        static_cast<u32>(Settings::values.physical_device.GetValue(true)));

    ConfigurationShared::SetColoredComboBox(
        ui->texture_sampling_combobox, ui->widget_texture_sampling,
        static_cast<int>(Settings::values.texture_sampling.GetValue(true)));

    ConfigurationShared::SetColoredTristate(ui->toggle_hw_shader, Settings::values.use_hw_shader,
                                            use_hw_shader);
    ConfigurationShared::SetColoredTristate(
        ui->toggle_accurate_mul, Settings::values.shaders_accurate_mul, shaders_accurate_mul);
    ConfigurationShared::SetColoredTristate(ui->toggle_disk_shader_cache,
                                            Settings::values.use_disk_shader_cache,
                                            use_disk_shader_cache);
    ConfigurationShared::SetColoredTristate(ui->toggle_vsync, Settings::values.use_vsync,
                                            use_vsync);
    ConfigurationShared::SetColoredTristate(ui->toggle_skip_duplicate_frames,
                                            Settings::values.use_skip_duplicate_frames,
                                            use_skip_duplicate_frames);
    ConfigurationShared::SetColoredTristate(ui->toggle_limit_frames_ahead_of_gpu,
                                            Settings::values.use_limit_frames_ahead_of_gpu,
                                            use_limit_frames_ahead_of_gpu);
    ConfigurationShared::SetColoredTristate(ui->toggle_async_shaders,
                                            Settings::values.async_shader_compilation,
                                            async_shader_compilation);
    ConfigurationShared::SetColoredTristate(
        ui->toggle_async_present, Settings::values.async_presentation, async_presentation);
    ConfigurationShared::SetColoredTristate(ui->spirv_shader_gen, Settings::values.spirv_shader_gen,
                                            spirv_shader_gen);
    ConfigurationShared::SetColoredTristate(ui->disable_spirv_optimizer,
                                            Settings::values.disable_spirv_optimizer,
                                            disable_spirv_optimizer);
    ConfigurationShared::SetColoredTristate(ui->simulate_3ds_gpu_timings,
                                            Settings::values.simulate_3ds_gpu_timings,
                                            simulate_3ds_gpu_timings);
#ifdef ENABLE_METAL
    ConfigurationShared::SetColoredTristate(
        ui->toggle_metalfx_spatial, Settings::values.use_metalfx_spatial, use_metalfx_spatial);
#endif
}

#ifdef ENABLE_METAL
void ConfigureGraphics::UpdateMetalFXSpatialAvailability(bool is_metal) {
    auto* spatial = ui->toggle_metalfx_spatial;
    int device_index = ui->physical_device_combo->currentIndex();
    if (!Settings::IsConfiguringGlobal() && device_index == ConfigurationShared::USE_GLOBAL_INDEX) {
        const auto global = Settings::values.physical_device.GetValue(true);
        if (global < static_cast<u32>(ui->physical_device_combo->count() - PerGameOffset())) {
            device_index = static_cast<int>(global) + PerGameOffset();
        }
    }
    const bool supports_spatial = ui->physical_device_combo->itemData(device_index).toBool();
    const bool setting_editable =
        !Settings::IsConfiguringGlobal() || Settings::values.use_metalfx_spatial.UsingGlobal();
    spatial->setVisible(is_metal);
    spatial->setEnabled(is_metal && (supports_spatial || spatial->isChecked()) && setting_editable);
}
#endif

void ConfigureGraphics::SetPhysicalDeviceComboVisibility(int index) {
    const auto effective_api = EffectiveGraphicsAPI(index);
    bool is_vulkan = false;
#ifdef ENABLE_VULKAN
    is_vulkan = effective_api == Settings::GraphicsAPI::Vulkan &&
                ui->physical_device_combo->count() > PerGameOffset();
#endif
    bool has_devices = is_vulkan;
#ifdef ENABLE_METAL
    const bool is_metal = effective_api == Settings::GraphicsAPI::Metal;
    has_devices |= is_metal && ui->physical_device_combo->count() > PerGameOffset();
    UpdateMetalFXSpatialAvailability(is_metal);
#endif

    ui->physical_device_group->setVisible(has_devices);
    ui->spirv_shader_gen->setVisible(is_vulkan);
    ui->disable_spirv_optimizer->setVisible(is_vulkan);
    ui->opengl_renderer_group->setVisible(effective_api == Settings::GraphicsAPI::OpenGL);
}
