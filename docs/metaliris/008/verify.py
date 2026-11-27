"""Compile the backend-independent RendererMetal interface as a real C++ consumer."""

import argparse
import importlib.util
from pathlib import Path
import sys

sys.dont_write_bytecode = True
SPEC = importlib.util.spec_from_file_location("phase006", Path(__file__).resolve().parents[1] / "006/verify.py")
phase006 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(phase006)

FIXTURE = r"""
#include <concepts>
#include <type_traits>
#include "video_core/renderer_metal/renderer_metal.h"

using Renderer = Metal::RendererMetal;
static_assert(std::derived_from<Renderer, VideoCore::RendererBase>);
static_assert(std::is_final_v<Renderer>);
static_assert(std::is_constructible_v<Renderer, Core::System&, Pica::PicaCore&,
                                      Frontend::EmuWindow&, Frontend::EmuWindow*>);
static_assert(!std::is_abstract_v<Renderer> && std::has_virtual_destructor_v<Renderer>);
static_assert(!std::is_copy_constructible_v<Renderer> && !std::is_copy_assignable_v<Renderer>);
static_assert(requires(Renderer& renderer) {
    { renderer.Rasterizer() } -> std::same_as<VideoCore::RasterizerInterface*>;
    { renderer.SwapBuffers() } -> std::same_as<void>;
    { renderer.TryPresent(15) } -> std::same_as<void>;
    { renderer.TryPresent(15, true) } -> std::same_as<void>;
    { renderer.NotifySurfaceChanged(true) } -> std::same_as<void>;
    { renderer.IsScreenshotPending() } -> std::same_as<bool>;
});

int main() {}
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("build", type=Path)
    build = parser.parse_args().build.resolve()
    source = build / "phase008_header.cpp"
    source.write_text(FIXTURE)
    flags, cwd = phase006.flags(build)
    depfile = build / "phase008_header.d"
    binary = build / "phase008_header"
    phase006.run([*flags, "-MD", "-MF", depfile, source, "-o", binary], cwd)
    phase006.run([binary], cwd)
    deps = depfile.read_text()
    assert "renderer_metal/renderer_metal.h" in deps
    assert "renderer_base.h" in deps and "rasterizer_interface.h" in deps
    for item in ("renderer_vulkan", "MoltenVK", "citra_libretro/libretro_vk", "/Metal.framework/", "/QtCore.framework/"):
        assert item not in deps, item
    print("PASS C++ renderer interface and transitive dependency isolation")


if __name__ == "__main__":
    main()
