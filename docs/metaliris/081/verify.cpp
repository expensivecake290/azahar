#include "citra_qt/bootmanager.h"
#include "common/settings.h"
#include "video_core/renderer_metal/mtl_platform.h"
#include <QApplication>
#include <QWindow>
#include <cassert>
#include <cstdio>

int main(int argc, char **argv) {
  QApplication app(argc, argv);
  Settings::values.graphics_api.SetValue(Settings::GraphicsAPI::Metal);
  Settings::values.layout_option.SetValue(
      Settings::LayoutOption::SeparateWindows);
  if (app.platformName() == QStringLiteral("cocoa") && app.screens().isEmpty()) {
    std::puts("PENDING Qt Cocoa exposes no screens to this process");
    return 77;
  }
  GRenderWindow primary(nullptr, nullptr, Core::System::GetInstance(), false);
  if (app.platformName() != QStringLiteral("cocoa")) {
    assert(!primary.InitRenderTarget() && primary.GetNativeSurface().surface.IsHeadless());
    std::puts("PASS Qt rejects non-Cocoa native render targets without publishing a surface");
    return 0;
  }
  GRenderWindow secondary(nullptr, nullptr, Core::System::GetInstance(), true);
  assert(primary.InitRenderTarget());
  assert(secondary.InitRenderTarget());
  const auto initial = primary.GetNativeSurface();
  const auto other = secondary.GetNativeSurface();
  assert(initial.surface.handle && other.surface.handle);
  assert(initial.surface.handle != other.surface.handle);
  assert(initial.surface.suspended && other.surface.suspended);
  assert(initial.layout.top_screen_enabled && !other.layout.top_screen_enabled);
  assert(!primary.GetWindowInfo().render_surface);
  assert(Metal::Surface{initial}.GetLayer());

  primary.show();
  app.processEvents();
  const auto shown = primary.GetNativeSurface();
  assert(shown.surface.IsDrawableAvailable());
  assert(shown.surface.handle == initial.surface.handle);
  assert(secondary.GetNativeSurface().surface.suspended);

  primary.resize(840, 650);
  app.processEvents();
  const auto resized = primary.GetNativeSurface();
  assert(resized.generation > shown.generation);
  assert(resized.layout.width == resized.surface.width);
  assert(resized.layout.height == resized.surface.height);
  assert(primary.ScaleTouch(QPointF(12, 12)).first ==
         static_cast<u32>(std::round(12 * resized.surface.scale)));

  auto *child =
      primary.findChild<QWidget *>(QString(), Qt::FindDirectChildrenOnly);
  assert(child && child->windowHandle());
  auto *window = child->windowHandle();
  window->destroy();
  app.processEvents();
  const auto retired = primary.GetNativeSurface();
  assert(retired.surface.IsHeadless());
  assert(Metal::Surface{resized}.GetLayer());
  window->create();
  app.processEvents();
  const auto restored = primary.GetNativeSurface();
  assert(restored.generation > retired.generation);
  assert(restored.surface.handle &&
         restored.surface.handle != resized.surface.handle);
  primary.hide();
  app.processEvents();
  assert(primary.GetNativeSurface().surface.suspended);

  primary.ReleaseRenderTarget();
  secondary.ReleaseRenderTarget();
  app.processEvents();
  assert(primary.GetNativeSurface().surface.IsHeadless());
  assert(secondary.GetNativeSurface().surface.IsHeadless());
  assert(Metal::Surface{restored}.GetLayer());
  assert(Metal::Surface{other}.GetLayer());
  std::puts("PASS real Cocoa Qt render targets, independent layers, resize, "
            "surface recreation and retirement");
}
