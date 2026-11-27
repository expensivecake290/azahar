"""Build and run the shared native surface lifecycle test with Azahar libraries."""

import argparse
import importlib.util
from pathlib import Path
import shlex
import subprocess
import sys

sys.dont_write_bytecode = True
SPEC = importlib.util.spec_from_file_location("phase006", Path(__file__).resolve().parents[1] / "006/verify.py")
phase006 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(phase006)

FIXTURE = r"""
#include <atomic>
#include <cassert>
#include <limits>
#include <thread>
#include "core/frontend/emu_window.h"

class Window final : public Frontend::EmuWindow {
public:
    explicit Window(bool secondary = false) : EmuWindow(secondary) {}
    void PollEvents() override {}
    using EmuWindow::PublishNativeSurface;
};

int main() {
    Window primary;
    Window secondary(true);
    auto layout = Layout::DefaultFrameLayout(640, 480, false, false);
    assert(primary.GetNativeSurface().generation == 0);
    assert(primary.GetNativeSurface().surface.IsHeadless());
    assert(primary.GetFramebufferLayout().width == 0 && primary.GetFramebufferLayout().height == 0);
    assert(secondary.isSecondary());

    int released = 0;
    Window::NativeSurface initial{
        .handle = std::shared_ptr<void>(new int{42}, [&released](void* value) {
            delete static_cast<int*>(value);
            ++released;
        }),
        .width = 640,
        .height = 480,
        .scale = 2.0f,
        .refresh_rate = 120.0,
    };
    assert(primary.PublishNativeSurface(initial, layout));
    auto retained = primary.GetNativeSurface();
    assert(retained.generation == 1 && retained.surface.IsDrawableAvailable());
    assert(retained.layout.width == 640 && retained.surface.scale == 2.0f);
    assert(retained.surface.refresh_rate == 120.0);
    assert(primary.PublishNativeSurface(initial, layout));
    assert(primary.GetNativeSurface().generation == 1);
    assert(secondary.GetNativeSurface().generation == 0);

    const auto bottom = primary.GetFramebufferLayout().bottom_screen;
    assert(primary.TouchPressed(bottom.left, bottom.top));
    primary.TouchMoved(640, 480);
    primary.TouchReleased();

    auto invalid = initial;
    invalid.width = 0;
    assert(!primary.PublishNativeSurface(invalid, layout));
    invalid = initial;
    invalid.scale = std::numeric_limits<float>::quiet_NaN();
    assert(!primary.PublishNativeSurface(invalid, layout));
    invalid = initial;
    invalid.refresh_rate = -60;
    assert(!primary.PublishNativeSurface(invalid, layout));
    invalid = initial;
    int borrowed = 1;
    invalid.handle = std::shared_ptr<void>(std::shared_ptr<void>{}, &borrowed);
    assert(!primary.PublishNativeSurface(invalid, layout));
    assert(primary.GetNativeSurface().generation == 1);
    assert(primary.GetNativeSurface().surface.handle == initial.handle);
    invalid.handle.reset();

    initial.suspended = true;
    assert(primary.PublishNativeSurface(initial, layout));
    assert(primary.GetNativeSurface().generation == 2);
    assert(!primary.GetNativeSurface().surface.IsDrawableAvailable());
    initial.suspended = false;
    initial.fullscreen = true;
    initial.portrait = true;
    initial.width = 480;
    initial.height = 640;
    auto resized = Layout::DefaultFrameLayout(480, 640, false, true);
    assert(primary.PublishNativeSurface(initial, resized));
    assert(primary.GetNativeSurface().generation == 3);
    assert(primary.GetNativeSurface().surface.fullscreen);
    assert(primary.GetNativeSurface().surface.portrait);
    auto layout_only = resized;
    layout_only.top_screen_enabled = false;
    assert(primary.PublishNativeSurface(initial, layout_only));
    assert(primary.GetNativeSurface().generation == 3);
    assert(!primary.GetNativeSurface().layout.top_screen_enabled);

    assert(primary.PublishNativeSurface({}, layout));
    assert(primary.GetNativeSurface().generation == 4);
    assert(primary.GetNativeSurface().surface.IsHeadless());
    initial.handle.reset();
    assert(released == 0 && retained.surface.IsDrawableAvailable());
    assert(*static_cast<int*>(retained.surface.handle.get()) == 42);
    retained.surface.handle.reset();
    assert(released == 1);

    bool deleter_saw_headless = false;
    Window::NativeSurface reentrant{
        .handle = std::shared_ptr<void>(new int{1}, [&](void* value) {
            deleter_saw_headless = primary.GetNativeSurface().surface.IsHeadless();
            delete static_cast<int*>(value);
        }),
        .width = 640,
        .height = 480,
    };
    assert(primary.PublishNativeSurface(std::move(reentrant), layout));
    assert(primary.PublishNativeSurface({}, layout));
    assert(deleter_saw_headless);

    std::atomic<bool> running{true};
    std::thread reader([&] {
        u64 last = 0;
        while (running.load()) {
            const auto legacy_layout = secondary.GetFramebufferLayout();
            if (legacy_layout.width) {
                assert(legacy_layout.width >= 640 && legacy_layout.width < 1640);
                assert(legacy_layout.height == 480);
            }
            const auto frame = secondary.GetNativeSurface();
            assert(frame.generation >= last);
            last = frame.generation;
            if (frame.surface.IsDrawableAvailable()) {
                assert(frame.layout.width == frame.surface.width);
                assert(frame.layout.height == frame.surface.height);
                assert(*static_cast<int*>(frame.surface.handle.get()) == 99);
            }
        }
    });
    auto handle = std::make_shared<int>(99);
    for (u32 i = 0; i < 1000; ++i) {
        const u32 width = 640 + i;
        const auto frame = Layout::DefaultFrameLayout(width, 480, false, false);
        assert(secondary.PublishNativeSurface({.handle = handle, .width = width, .height = 480}, frame));
    }
    running.store(false);
    reader.join();
    assert(secondary.GetNativeSurface().generation == 1000);
    assert(primary.GetNativeSurface().generation == 6);
    return 0;
}
"""


def link_command(build, objects, binary):
    text = subprocess.check_output(["ninja", "-C", str(build), "-t", "commands", "bin/Release/tests"], text=True)
    recipe = text.splitlines()[-1]
    tokens = shlex.split(recipe)
    start = tokens.index("&&") + 1
    end = tokens.index("&&", start)
    args = [arg for arg in tokens[start:end] if not arg.endswith(".o")]
    args[1:1] = [str(obj) for obj in objects]
    args[args.index("-o") + 1] = str(binary)
    return args


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("build", type=Path)
    build = parser.parse_args().build.resolve()
    command, cwd = phase006.flags(build)
    source = build / "phase007_surface.cpp"
    obj = source.with_suffix(".o")
    binary = build / "phase007_surface"
    source.write_text(FIXTURE)
    phase006.run([*command, "-UNDEBUG", "-c", source, "-o", obj], cwd)
    implementation = phase006.ROOT / "src/core/frontend/emu_window.cpp"
    implementation_obj = build / "phase007_emu_window.o"
    phase006.run([*command, "-c", implementation, "-o", implementation_obj], cwd)
    phase006.run(link_command(build, [obj, implementation_obj], binary), build)
    phase006.run([binary], build)
    print("PASS native lifecycle, layout, touch, retention, rejection and concurrent snapshots")


if __name__ == "__main__":
    main()
