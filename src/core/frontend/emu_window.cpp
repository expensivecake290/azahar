// Copyright 2014-2026 Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.txt file included.

#include <cmath>
#include <limits>
#include <mutex>
#include "common/settings.h"
#include "core/3ds.h"
#include "core/frontend/emu_window.h"
#include "core/frontend/input.h"

namespace Frontend {
/// We need a global touch state that is shared across the different window instances
static std::weak_ptr<EmuWindow::TouchState> global_touch_state;

GraphicsContext::~GraphicsContext() = default;

class EmuWindow::TouchState : public Input::Factory<Input::TouchDevice>,
                              public std::enable_shared_from_this<TouchState> {
public:
    std::unique_ptr<Input::TouchDevice> Create(const Common::ParamPackage&) override {
        return std::make_unique<Device>(shared_from_this());
    }

    std::mutex mutex;

    bool touch_pressed = false; ///< True if touchpad area is currently pressed, otherwise false
    float touch_x = 0.0f;       ///< Touchpad X-position
    float touch_y = 0.0f;       ///< Touchpad Y-position

    u64 press_count = 0;       ///< Number of touches that have started
    float last_press_x = 0.0f; ///< X-position of the most recent touch before its release
    float last_press_y = 0.0f; ///< Y-position of the most recent touch before its release

private:
    class Device : public Input::TouchDevice {
    public:
        explicit Device(std::weak_ptr<TouchState>&& touch_state) : touch_state(touch_state) {}
        std::tuple<float, float, bool> GetStatus() const override {
            if (auto state = touch_state.lock()) {
                std::scoped_lock guard{state->mutex};
                // Make sure touch events that are shorter than the read period are also seen.
                if (!state->touch_pressed && seen_press_count != state->press_count) {
                    seen_press_count = state->press_count;
                    return std::make_tuple(state->last_press_x, state->last_press_y, true);
                }
                seen_press_count = state->press_count;
                return std::make_tuple(state->touch_x, state->touch_y, state->touch_pressed);
            }
            return std::make_tuple(0.0f, 0.0f, false);
        }

    private:
        std::weak_ptr<TouchState> touch_state;
        mutable u64 seen_press_count = 0;
    };
};

EmuWindow::EmuWindow() {
    CreateTouchState();
};

EmuWindow::EmuWindow(bool is_secondary_) : is_secondary{is_secondary_} {
    CreateTouchState();
}

EmuWindow::~EmuWindow() = default;

EmuWindow::NativeSurfaceSnapshot EmuWindow::GetNativeSurface() const {
    std::scoped_lock lock(native_surface_mutex);
    return native_surface;
}

bool EmuWindow::PublishNativeSurface(NativeSurface surface, Layout::FramebufferLayout layout) {
    if (!std::isfinite(surface.scale) || surface.scale <= 0.0f ||
        !std::isfinite(surface.refresh_rate) || surface.refresh_rate < 0.0 ||
        (surface.width == 0) != (surface.height == 0) ||
        (surface.handle && surface.handle.use_count() == 0)) {
        return false;
    }
    if (!surface.handle) {
        surface.handle.reset();
    }

    // Release the old owner outside the lock. Its deleter can call into the frontend.
    std::shared_ptr<void> retired_handle;
    {
        std::scoped_lock lock(native_surface_mutex);
        const auto& current = native_surface.surface;
        const bool ownership_changed = current.handle.owner_before(surface.handle) ||
                                       surface.handle.owner_before(current.handle);
        if (ownership_changed || current != surface) {
            if (native_surface.generation == std::numeric_limits<u64>::max()) {
                return false;
            }
            retired_handle = std::move(native_surface.surface.handle);
            native_surface.surface = std::move(surface);
            ++native_surface.generation;
        }
        native_surface.layout = layout;
        framebuffer_layout = layout;
    }
    return true;
}

bool EmuWindow::UpdateNativeSurface(NativeSurface surface, u32 fallback_width, u32 fallback_height) {
    const auto [layout, min_size] =
        CalculateFramebufferLayout(surface.width ? surface.width : fallback_width,
                                   surface.height ? surface.height : fallback_height,
                                   surface.portrait);
    if (!PublishNativeSurface(std::move(surface), layout)) {
        return false;
    }
    UpdateMinimumWindowSize(min_size);
    return true;
}

Settings::StereoRenderOption EmuWindow::get3DMode() const {
    Settings::StereoRenderOption render_3d_mode = Settings::values.render_3d.GetValue();
#ifndef ANDROID
    // on desktop, if separate windows and this is the bottom screen, then no stereo
    if (Settings::values.layout_option.GetValue() == Settings::LayoutOption::SeparateWindows &&
        ((!is_secondary && Settings::values.swap_screen.GetValue()) ||
         (is_secondary && !Settings::values.swap_screen.GetValue()))) {
        render_3d_mode = Settings::StereoRenderOption::Off;
    }
#else
    // adjust the StereoRenderOption setting to Off if appropriate on mobile
    Settings::StereoWhichDisplay whichDisplay = Settings::values.render_3d_which_display.GetValue();
    if (whichDisplay == Settings::StereoWhichDisplay::None ||
        whichDisplay == Settings::StereoWhichDisplay::PrimaryOnly && is_secondary ||
        whichDisplay == Settings::StereoWhichDisplay::SecondaryOnly && !is_secondary) {
        render_3d_mode = Settings::StereoRenderOption::Off;
    }
#endif
    return render_3d_mode;
}

std::tuple<unsigned, unsigned> EmuWindow::ClipToTouchScreen(unsigned new_x, unsigned new_y) const {

    Settings::StereoRenderOption render_3d_mode = get3DMode();

    if (new_x >= framebuffer_layout.width / 2) {
        if (render_3d_mode == Settings::StereoRenderOption::SideBySide ||
            render_3d_mode == Settings::StereoRenderOption::SideBySideFull)
            new_x -= framebuffer_layout.width / 2;
        else if (render_3d_mode == Settings::StereoRenderOption::CardboardVR)
            new_x -=
                (framebuffer_layout.width / 2) - (framebuffer_layout.cardboard.user_x_shift * 2);
    }

    if (render_3d_mode == Settings::StereoRenderOption::SideBySide) {
        new_x = std::max(new_x, framebuffer_layout.bottom_screen.left / 2);
        new_x = std::min(new_x, framebuffer_layout.bottom_screen.right / 2 - 1);
    } else {
        new_x = std::max(new_x, framebuffer_layout.bottom_screen.left);
        new_x = std::min(new_x, framebuffer_layout.bottom_screen.right - 1);
    }

    new_y = std::max(new_y, framebuffer_layout.bottom_screen.top);
    new_y = std::min(new_y, framebuffer_layout.bottom_screen.bottom - 1);

    return std::make_tuple(new_x, new_y);
}

void EmuWindow::CreateTouchState() {
    touch_state = global_touch_state.lock();
    if (touch_state) {
        return;
    }
    touch_state = std::make_shared<TouchState>();
    Input::RegisterFactory<Input::TouchDevice>("emu_window", touch_state);
    global_touch_state = touch_state;
}

bool EmuWindow::TouchPressed(unsigned framebuffer_x, unsigned framebuffer_y) {
    std::scoped_lock layout_lock(native_surface_mutex);
    if (!framebuffer_layout.IsWithinTouchscreen(framebuffer_x, framebuffer_y))
        return false;
    Settings::StereoRenderOption render_3d_mode = get3DMode();

    if (framebuffer_x >= framebuffer_layout.width / 2) {
        if (render_3d_mode == Settings::StereoRenderOption::SideBySide ||
            render_3d_mode == Settings::StereoRenderOption::SideBySideFull)
            framebuffer_x -= framebuffer_layout.width / 2;
        else if (render_3d_mode == Settings::StereoRenderOption::CardboardVR)
            framebuffer_x -=
                (framebuffer_layout.width / 2) - (framebuffer_layout.cardboard.user_x_shift * 2);
    }

    std::scoped_lock guard(touch_state->mutex);

    if (render_3d_mode == Settings::StereoRenderOption::SideBySide) {
        touch_state->touch_x =
            static_cast<float>(framebuffer_x - framebuffer_layout.bottom_screen.left / 2) /
            (framebuffer_layout.bottom_screen.right / 2 -
             framebuffer_layout.bottom_screen.left / 2);
    } else {
        touch_state->touch_x =
            static_cast<float>(framebuffer_x - framebuffer_layout.bottom_screen.left) /
            (framebuffer_layout.bottom_screen.right - framebuffer_layout.bottom_screen.left);
    }
    touch_state->touch_y =
        static_cast<float>(framebuffer_y - framebuffer_layout.bottom_screen.top) /
        (framebuffer_layout.bottom_screen.bottom - framebuffer_layout.bottom_screen.top);

    if (!framebuffer_layout.is_rotated) {
        std::swap(touch_state->touch_x, touch_state->touch_y);
        touch_state->touch_x = 1.f - touch_state->touch_x;
    }

    if (!touch_state->touch_pressed) {
        touch_state->press_count++;
    }
    touch_state->touch_pressed = true;
    return true;
}

void EmuWindow::TouchReleased() {
    std::scoped_lock guard{touch_state->mutex};
    if (touch_state->touch_pressed) {
        touch_state->last_press_x = touch_state->touch_x;
        touch_state->last_press_y = touch_state->touch_y;
    }
    touch_state->touch_pressed = false;
    touch_state->touch_x = 0;
    touch_state->touch_y = 0;
}

void EmuWindow::TouchMoved(unsigned framebuffer_x, unsigned framebuffer_y) {
    {
        std::scoped_lock guard(touch_state->mutex);
        if (!touch_state->touch_pressed) {
            return;
        }
    }
    {
        std::scoped_lock layout_lock(native_surface_mutex);
        if (!framebuffer_layout.IsWithinTouchscreen(framebuffer_x, framebuffer_y)) {
            std::tie(framebuffer_x, framebuffer_y) = ClipToTouchScreen(framebuffer_x, framebuffer_y);
        }
    }

    TouchPressed(framebuffer_x, framebuffer_y);
}

void EmuWindow::UpdateCurrentFramebufferLayout(u32 width, u32 height, bool is_portrait_mode) {
    const auto [layout, min_size] = CalculateFramebufferLayout(width, height, is_portrait_mode);
    UpdateMinimumWindowSize(min_size);
    NotifyFramebufferLayoutChanged(layout);
}

EmuWindow::WindowLayout EmuWindow::CalculateFramebufferLayout(u32 width, u32 height,
                                                              bool is_portrait_mode) const {
    Layout::FramebufferLayout layout;
    const Settings::LayoutOption layout_option = Settings::values.layout_option.GetValue();
    const Settings::StereoRenderOption stereo_option = get3DMode();
    bool render_full_stereo = (stereo_option == Settings::StereoRenderOption::SideBySideFull);
    bool is_bottom = is_secondary;
    if (Settings::values.swap_screen.GetValue())
        is_bottom = !is_bottom;

    const Settings::PortraitLayoutOption portrait_layout_option =
        Settings::values.portrait_layout_option.GetValue();
    const auto min_size = is_portrait_mode
                              ? Layout::GetMinimumSizeFromPortraitLayout()
                              : Layout::GetMinimumSizeFromLayout(
                                    layout_option, Settings::values.upright_screen.GetValue());

    width = std::max(width, min_size.first);
    height = std::max(height, min_size.second);
    if (render_full_stereo) {
        width = width / 2;
    }
    if (is_portrait_mode) {
        switch (portrait_layout_option) {
        case Settings::PortraitLayoutOption::PortraitTopFullWidth:
            layout = Layout::PortraitTopFullFrameLayout(width, height,
                                                        Settings::values.swap_screen.GetValue(),
                                                        Settings::values.upright_screen.GetValue());
            break;
        case Settings::PortraitLayoutOption::PortraitCustomLayout:
            layout = Layout::CustomFrameLayout(
                width, height, Settings::values.swap_screen.GetValue(), is_portrait_mode);
            break;
        case Settings::PortraitLayoutOption::PortraitOriginal:
            layout = Layout::PortraitOriginalLayout(width, height,
                                                    Settings::values.swap_screen.GetValue(),
                                                    Settings::values.upright_screen.GetValue());
            break;
        }
    } else {
        switch (layout_option) {
        case Settings::LayoutOption::CustomLayout:
            layout = Layout::CustomFrameLayout(
                width, height, Settings::values.swap_screen.GetValue(), is_portrait_mode);
            break;
        case Settings::LayoutOption::SingleScreen:
            layout =
                Layout::SingleFrameLayout(width, height, Settings::values.swap_screen.GetValue(),
                                          Settings::values.upright_screen.GetValue());
            break;
        case Settings::LayoutOption::LargeScreen:
            layout =
                Layout::LargeFrameLayout(width, height, Settings::values.swap_screen.GetValue(),
                                         Settings::values.upright_screen.GetValue(),
                                         Settings::values.large_screen_proportion.GetValue(),
                                         Settings::values.small_screen_position.GetValue());
            break;
        case Settings::LayoutOption::HybridScreen:
            layout =
                Layout::HybridScreenLayout(width, height, Settings::values.swap_screen.GetValue(),
                                           Settings::values.upright_screen.GetValue());
            break;
        case Settings::LayoutOption::SideScreen:
            layout =
                Layout::LargeFrameLayout(width, height, Settings::values.swap_screen.GetValue(),
                                         Settings::values.upright_screen.GetValue(), 1.0f,
                                         Settings::SmallScreenPosition::MiddleRight);
            break;
#ifndef ANDROID
        case Settings::LayoutOption::SeparateWindows:
            layout = Layout::SeparateWindowsLayout(width, height, is_secondary,
                                                   Settings::values.upright_screen.GetValue());
            break;
#endif
        case Settings::LayoutOption::Default:
        default:
            layout =
                Layout::DefaultFrameLayout(width, height, Settings::values.swap_screen.GetValue(),
                                           Settings::values.upright_screen.GetValue());
            break;
        }
    }
#ifdef ANDROID
    if (is_secondary) {
        layout = Layout::AndroidSecondaryLayout(width, height);
    }
#endif

    if (render_full_stereo) {
        layout.width = width * 2;
    }
    if (Settings::values.render_3d.GetValue() == Settings::StereoRenderOption::CardboardVR) {
        layout = Layout::GetCardboardSettings(layout);
    }
    layout.render_3d_mode = stereo_option;
    return {layout, min_size};
}

void EmuWindow::UpdateMinimumWindowSize(std::pair<unsigned, unsigned> min_size) {
    WindowConfig new_config = config;
    new_config.min_client_area_size = min_size;
    SetConfig(new_config);
    ProcessConfigurationChanges();
}

} // namespace Frontend
