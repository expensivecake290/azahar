#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <memory>
#include <QApplication>
#include <QFile>
#include <QFileOpenEvent>
#include <QMessageBox>
#include <QPushButton>
#include <QSettings>
#include <QTemporaryDir>
#include <QTimer>
#include "citra_qt/bootmanager.h"
#include "citra_qt/citra_qt.h"
#include "citra_qt/util/metal_util.h"
#include "common/file_util.h"
#include "common/settings.h"

int main(int argc, char** argv) {
    const bool preserve_environment = qEnvironmentVariableIsSet("METALIRIS_FIXTURE_KEEP_EXTERNAL");
    if (preserve_environment) {
        qputenv("MTL_DEBUG_LAYER", "0");
        qputenv("MTL_CAPTURE_ENABLED", "0");
    } else {
        qunsetenv("MTL_DEBUG_LAYER");
        qunsetenv("MTL_CAPTURE_ENABLED");
    }
    QApplication app(argc, argv);
    QCoreApplication::setOrganizationName(QStringLiteral("Azahar Fixture"));
    QCoreApplication::setApplicationName(QStringLiteral("Metaliris Phase 086"));
    Q_INIT_RESOURCE(compatibility_list);
    Q_INIT_RESOURCE(theme_colorful);
    Q_INIT_RESOURCE(theme_colorful_dark);
    Q_INIT_RESOURCE(theme_colorful_midnight_blue);
    Q_INIT_RESOURCE(theme_default);
    Q_INIT_RESOURCE(theme_qdarkstyle);
    Q_INIT_RESOURCE(theme_qdarkstyle_midnight_blue);
    QTemporaryDir directory;
    assert(directory.isValid());
    FileUtil::SetUserPath(directory.path().toStdString() + "/");
    {
        const auto config_dir = QString::fromStdString(FileUtil::GetUserPath(FileUtil::UserPath::ConfigDir));
        QSettings config(config_dir + QStringLiteral("qt-config.ini"), QSettings::IniFormat);
        config.setValue(QStringLiteral("Debugging/renderer_debug"), true);
        config.setValue(QStringLiteral("Debugging/dump_command_buffers"), true);
    }
    std::unique_ptr<GMainWindow> window;
    QTimer startup_dialog;
    startup_dialog.setInterval(10);
    QObject::connect(&startup_dialog, &QTimer::timeout, [&] {
        if (auto* dialog = qobject_cast<QMessageBox*>(app.activeModalWidget());
            dialog && dialog->text().contains(
                          QStringLiteral("being run directly rather than via the Azahar.app"))) {
            std::puts("PASS acknowledged existing macOS terminal-launch warning");
            dialog->accept();
        }
    });
    startup_dialog.start();
    QTimer::singleShot(15000, &app, [] {
        std::fputs("Qt startup fixture did not complete within 15 seconds\n", stderr);
        std::abort();
    });
    try {
        window = std::make_unique<GMainWindow>(Core::System::GetInstance());
    } catch (const std::exception& error) {
        if (std::string_view(error.what()).find("bind: Operation not permitted") ==
            std::string_view::npos) {
            throw;
        }
        std::puts("PENDING Qt main-window status and ROM rejection: process cannot bind the existing UDP input socket");
    }
    startup_dialog.stop();
    assert(qgetenv("MTL_DEBUG_LAYER") == (preserve_environment ? "0" : "1"));
    assert(qgetenv("MTL_CAPTURE_ENABLED") == (preserve_environment ? "0" : "1"));
    std::puts("PASS persisted Qt launch diagnostics and explicit environment precedence");
    if (!window) {
        return 0;
    }
    auto* button = window->findChild<QPushButton*>(QStringLiteral("GraphicsAPIStatusBarButton"));
    assert(button && button->text() == QStringLiteral("METAL 4"));
    const auto devices = MetalUtil::EnumerateDevices();
    if (devices.empty()) {
        QFile file(directory.filePath(QStringLiteral("device-unavailable.3dsx")));
        assert(file.open(QIODevice::WriteOnly));
        file.close();
        QFileOpenEvent open(file.fileName());
        QString message;
        QTimer closer;
        closer.setInterval(10);
        QObject::connect(&closer, &QTimer::timeout, [&] {
            if (auto* dialog = qobject_cast<QMessageBox*>(app.activeModalWidget())) {
                message = dialog->text();
                std::fprintf(stderr, "Observed ROM launch dialog [%s]: %s\n",
                             qPrintable(dialog->windowTitle()), qPrintable(message));
                dialog->accept();
            }
        });
        closer.start();
        window->OnFileOpen(&open);
        closer.stop();
        assert(message == QStringLiteral("The selected device does not support Metal 4 with "
                                         "unified memory or is unavailable to Azahar."));
        assert(!Core::System::GetInstance().IsPoweredOn());
        int primary_surfaces = 0;
        int secondary_surfaces = 0;
        for (auto* widget : QApplication::allWidgets()) {
            if (auto* render_window = qobject_cast<GRenderWindow*>(widget)) {
                assert(render_window->GetNativeSurface().surface.IsHeadless());
                if (render_window->isSecondary()) {
                    ++secondary_surfaces;
                } else {
                    ++primary_surfaces;
                }
            }
        }
        assert(primary_surfaces == 1 && secondary_surfaces == 1);
        std::puts("PASS no-device launch leaves both Qt render targets headless");
#ifdef ENABLE_SOFTWARE_RENDERER
        button->click();
        assert(button->text() == QStringLiteral("SOFTWARE"));
        button->click();
        assert(button->text() == QStringLiteral("SOFTWARE"));
#endif
    }
    std::printf("PASS Qt Metal 4 startup/status, launch diagnostics and adapter gate with %zu native devices\n", devices.size());
}
