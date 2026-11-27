"""Verify the native surface contract against real Azahar C++ build inputs."""

import argparse
import json
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[3]
BASELINE = "c621eafbf316699c55f4c9c0c689d4e49498d360"

HEADER_FIXTURE = r"""
#include <cassert>
#include <type_traits>
#include "core/frontend/emu_window.h"

using Window = Frontend::EmuWindow;
static_assert(std::is_same_v<decltype(std::declval<const Window&>().GetNativeSurface()),
                             Window::NativeSurfaceSnapshot>);
static_assert(static_cast<int>(Frontend::WindowSystemType::MacOS) == 3);
static_assert(static_cast<int>(Frontend::WindowSystemType::LibRetro) == 6);

int main() {
    Window::NativeSurface surface{};
    assert(surface.IsHeadless());
    assert(!surface.IsDrawableAvailable());
    assert(surface.scale == 1.0f && surface.refresh_rate == 0.0);
    surface.handle = std::make_shared<int>(42);
    assert(!surface.IsHeadless());
    assert(!surface.IsDrawableAvailable());
    surface.width = 640;
    surface.height = 480;
    assert(surface.IsDrawableAvailable());
    surface.suspended = true;
    assert(!surface.IsDrawableAvailable());
    auto retained = surface;
    surface.handle.reset();
    assert(*static_cast<int*>(retained.handle.get()) == 42);
    return 0;
}
"""


def flags(build):
    items = json.loads((build / "compile_commands.json").read_text())
    entry = next(e for e in items if e["file"].endswith("/core/frontend/emu_window.cpp"))
    command = shlex.split(entry["command"])
    at = command.index("-o")
    del command[at:at + 2]
    at = command.index("-c")
    del command[at:at + 2]
    command[1:1] = ["-I", str(ROOT / "src")]
    return command, entry["directory"]


def run(command, directory):
    print("+", shlex.join([str(x) for x in command]), flush=True)
    subprocess.run(command, cwd=directory, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("build", type=Path)
    build = parser.parse_args().build.resolve()
    command, cwd = flags(build)
    source = build / "phase006_header.cpp"
    source.write_text(HEADER_FIXTURE)
    binary = build / "phase006_header"
    depfile = build / "phase006_header.d"
    run([*command, "-UNDEBUG", "-MD", "-MF", depfile, source, "-o", binary], cwd)
    run([binary], cwd)
    dep_text = depfile.read_text()
    assert "frontend/emu_window.h" in dep_text
    for forbidden in ("renderer_vulkan", "MoltenVK", "/Metal.framework/", "/QtCore.framework/"):
        assert forbidden not in dep_text, forbidden
    print("PASS portable native surface interface and transitive header isolation")


if __name__ == "__main__":
    main()
