#include <cassert>
#include <cstdio>
#include <QApplication>
#include <QCheckBox>
#include <QTemporaryDir>
#include "citra_qt/configuration/configure_debug.h"
#include "common/file_util.h"
#include "common/logging/backend.h"
#include "common/settings.h"

int main(int argc, char** argv) {
    qputenv("MTL_DEBUG_LAYER", "1");
    QApplication app(argc, argv);
    QTemporaryDir directory;
    assert(directory.isValid());
    FileUtil::SetUserPath(directory.path().toStdString() + "/");
    Common::Log::Initialize();
    Settings::SetConfiguringGlobal(true);
    Settings::values.renderer_debug.SetValue(false);
    Settings::values.dump_command_buffers.SetValue(false);
    const auto pica_debug = Settings::values.pica_debugging.GetValue();
    ConfigureDebug configuration(false);
    auto* validation = configuration.findChild<QCheckBox*>(QStringLiteral("toggle_renderer_debug"));
    auto* trace = configuration.findChild<QCheckBox*>(QStringLiteral("toggle_dump_command_buffers"));
    assert(validation && trace);
    assert(validation->text().contains(QStringLiteral("Metal API Validation")));
    assert(trace->text().contains(QStringLiteral("Metal GPU Trace")));
    assert(validation->isEnabled() && trace->isEnabled());
    assert(!validation->isChecked() && !trace->isChecked());
    validation->click();
    configuration.ApplyConfiguration();
    assert(Settings::values.renderer_debug.GetValue());
    assert(!Settings::values.dump_command_buffers.GetValue());
    trace->setChecked(true);
    validation->click();
    configuration.ApplyConfiguration();
    assert(!Settings::values.renderer_debug.GetValue());
    assert(Settings::values.dump_command_buffers.GetValue());
    assert(Settings::values.pica_debugging.GetValue() == pica_debug);
    configuration.RetranslateUI();
    assert(validation->text().contains(QStringLiteral("Metal API Validation")));
    assert(trace->toolTip().contains(QStringLiteral("first rendered frame")));

    ConfigureDebug running(true);
    auto* locked_validation = running.findChild<QCheckBox*>(QStringLiteral("toggle_renderer_debug"));
    auto* locked_trace = running.findChild<QCheckBox*>(QStringLiteral("toggle_dump_command_buffers"));
    assert(!locked_validation->isEnabled() && !locked_trace->isEnabled());
    assert(!locked_validation->isChecked() && locked_trace->isChecked());
    std::puts("PASS native Qt validation and trace preferences are separate, translated and locked while running");
}
