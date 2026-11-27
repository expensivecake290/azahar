#include <cassert>

#include "common/settings.h"
#include "core/frontend/emu_window.h"

class Window final : public Frontend::EmuWindow {
public:
  explicit Window(bool secondary = false) : EmuWindow(secondary) {}
  void PollEvents() override {}
  using EmuWindow::UpdateNativeSurface;
};

int main() {
  Settings::values.layout_option.SetValue(Settings::LayoutOption::Default);
  Settings::values.render_3d.SetValue(Settings::StereoRenderOption::Off);
  Window primary;
  Window secondary{true};
  auto layer = std::make_shared<int>(17);
  assert(primary.UpdateNativeSurface(
      {.handle = layer, .width = 832, .height = 642}, 20, 20));
  const auto initial = primary.GetNativeSurface();
  assert(initial.layout.width == 832 && initial.layout.height == 642);
  assert(initial.generation == 1 && initial.surface.IsDrawableAvailable());
  Settings::values.layout_option.SetValue(
      Settings::LayoutOption::SeparateWindows);
  primary.UpdateCurrentFramebufferLayout(832, 642);
  const auto top = primary.GetNativeSurface();
  assert(top.layout.top_screen_enabled && !top.layout.bottom_screen_enabled);
  assert(top.generation == initial.generation && top.surface.handle == layer);
  assert(secondary.UpdateNativeSurface({.handle = std::make_shared<int>(29),
                                        .width = 640,
                                        .height = 480,
                                        .scale = 2.0f},
                                       10, 11));
  const auto bottom = secondary.GetNativeSurface();
  assert(!bottom.layout.top_screen_enabled &&
         bottom.layout.bottom_screen_enabled);
  assert(bottom.layout.width == 640 && bottom.layout.height == 480);
  assert(bottom.generation == 1 && bottom.surface.handle != layer);
  assert(primary.UpdateNativeSurface({.suspended = true}, 960, 720));
  const auto headless = primary.GetNativeSurface();
  assert(headless.surface.IsHeadless() && headless.generation == 2);
  assert(headless.layout.width == 960 && headless.layout.height == 720);
  assert(initial.surface.handle == layer);
  assert(
      !secondary.UpdateNativeSurface({.handle = layer, .width = 500}, 10, 11));
  assert(secondary.GetNativeSurface().generation == 1);
}
