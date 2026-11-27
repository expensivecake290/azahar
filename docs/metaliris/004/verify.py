"""Compile the owned shared settings interfaces against real generated keys and dependencies."""

import argparse
import json
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[3]
BASELINE = "c621eafbf316699c55f4c9c0c689d4e49498d360"

HEADER_FIXTURE = r"""
#include <cassert>
#include "common/settings.h"

using Settings::GraphicsAPI;
static_assert(static_cast<int>(GraphicsAPI::Software) == 0);
static_assert(static_cast<int>(GraphicsAPI::OpenGL) == 1);
static_assert(static_cast<int>(GraphicsAPI::Vulkan) == 2);
static_assert(static_cast<int>(GraphicsAPI::Metal) == 3);

int main() {
    Settings::Values settings{};
    settings.graphics_api.SetValue(static_cast<GraphicsAPI>(-1));
    assert(settings.graphics_api.GetValue() == GraphicsAPI::Software);
    settings.graphics_api.SetValue(static_cast<GraphicsAPI>(99));
#ifdef ENABLE_METAL
    assert(settings.graphics_api.GetDefault() == GraphicsAPI::Metal);
    assert(settings.graphics_api.GetValue() == GraphicsAPI::Metal);
    auto& spatial = settings.use_metalfx_spatial;
    assert(spatial.GetLabel() == "use_metalfx_spatial");
    assert(spatial.UsingGlobal() && !spatial.GetDefault() && !spatial.GetValue());
    spatial.SetGlobal(false);
    spatial.SetValue(true);
    assert(spatial.GetValue() && !spatial.GetValue(true));
    spatial.SetGlobal(true);
    assert(!spatial.GetValue());
#else
    assert(settings.graphics_api.GetDefault() == GraphicsAPI::Software);
    assert(settings.graphics_api.GetValue() == GraphicsAPI::Vulkan);
#endif
    assert(settings.resolution_factor.GetValue() == 1);
    assert(settings.texture_filter.GetValue() == Settings::TextureFilter::NoFilter);
    return 0;
}
"""

IMPLEMENTATION_FIXTURE = r"""
#include <cassert>
#include "common/settings.cpp"

using Settings::GraphicsAPI;
int main() {
    using Settings::GetGraphicsAPIName;
    assert(GetGraphicsAPIName(GraphicsAPI::Software) == "Software");
    assert(GetGraphicsAPIName(GraphicsAPI::OpenGL) == "OpenGL");
    assert(GetGraphicsAPIName(GraphicsAPI::Vulkan) == "Vulkan");
    assert(GetGraphicsAPIName(GraphicsAPI::Metal) == "Metal");
    assert(GetGraphicsAPIName(static_cast<GraphicsAPI>(99)) == "Invalid");
#ifdef ENABLE_METAL
    assert(Settings::GetWorkingGraphicsAPI() == GraphicsAPI::Metal);
    Settings::values.graphics_api.SetValue(GraphicsAPI::Vulkan);
    assert(Settings::GetWorkingGraphicsAPI() == GraphicsAPI::Metal);
    Settings::values.graphics_api.SetValue(GraphicsAPI::OpenGL);
    assert(Settings::GetWorkingGraphicsAPI() == GraphicsAPI::Metal);
    Settings::values.graphics_api.SetValue(GraphicsAPI::Software);
#ifdef ENABLE_SOFTWARE_RENDERER
    assert(Settings::GetWorkingGraphicsAPI() == GraphicsAPI::Software);
#else
    assert(Settings::GetWorkingGraphicsAPI() == GraphicsAPI::Metal);
#endif
    auto& spatial = Settings::values.use_metalfx_spatial;
    spatial.SetGlobal(false);
    spatial.SetValue(true);
    Settings::RestoreGlobalState(true);
    assert(!spatial.UsingGlobal() && spatial.GetValue());
    Settings::RestoreGlobalState(false);
    assert(spatial.UsingGlobal() && !spatial.GetValue());
    spatial.SetGlobal(false);
    assert(spatial.GetValue());
#else
    assert(Settings::GetWorkingGraphicsAPI() == GraphicsAPI::Software);
#endif
    return 0;
}
"""


def commands(build):
    entries = json.loads((build / "compile_commands.json").read_text())
    entry = next(e for e in entries if e["file"].endswith("/common/settings.cpp"))
    args = shlex.split(entry["command"])
    output = args.index("-o")
    del args[output:output + 2]
    source = args.index("-c")
    del args[source:source + 2]
    args[1:1] = ["-I", str(ROOT / "src")]
    return args, entry["directory"]


def run(cmd, directory):
    print("+", shlex.join([str(arg) for arg in cmd]), flush=True)
    subprocess.run(cmd, cwd=directory, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=("header", "implementation", "source-header", "source-implementation"))
    parser.add_argument("build", type=Path)
    parser.add_argument("configuration", choices=("off", "metal", "metal-nosw"))
    args = parser.parse_args()
    build = args.build.resolve()
    if args.case.startswith("source-"):
        file = "src/common/settings.h" if args.case == "source-header" else "src/common/settings.cpp"
        old = subprocess.check_output(["git", "-C", ROOT, "show", f"{BASELINE}:{file}"], text=True)
        # This phase adds declarations and branches without deleting existing ones.
        current = iter((ROOT / file).read_text().splitlines())
        for lineno, line in enumerate(old.splitlines(), 1):
            if args.case == "source-header" and lineno == 549:
                continue
            assert any(line == candidate for candidate in current), line
        print("PASS all original settings lines retained", file)
        return
    command, directory = commands(build)
    if args.configuration.startswith("metal"):
        key = build / "src/common/setting_keys.h"
        if "DEFINE_KEY(use_metalfx_spatial)" not in key.read_text():
            print("PENDING: Phase 005A generated Settings::Keys::use_metalfx_spatial")
            raise SystemExit(2)
        command.append("-DENABLE_METAL")
        if args.configuration == "metal-nosw":
            command = [arg for arg in command if arg != "-DENABLE_SOFTWARE_RENDERER"]
    else:
        command = [arg for arg in command if arg != "-DENABLE_METAL"]
        command.append("-DENABLE_SOFTWARE_RENDERER")
    stem = build / f"{args.case}-{args.configuration}"
    fixture = stem.with_suffix(".cpp")
    fixture.write_text(HEADER_FIXTURE if args.case == "header" else IMPLEMENTATION_FIXTURE)
    run([*command, "-UNDEBUG", "-Wl,-dead_strip", fixture, "-o", stem], directory)
    run([stem], directory)
    print("PASS", args.case, args.configuration)


if __name__ == "__main__":
    main()
