"""Check native build graph, language flags and source isolation for Phase 002."""

import argparse
import importlib.util
import json
from pathlib import Path

SPEC = importlib.util.spec_from_file_location("phase001", Path(__file__).parent.parent / "001/verify.py")
phase001 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(phase001)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work", type=Path)
    args = parser.parse_args()
    work = args.work.resolve()
    source = work / "source"
    build = work / "native"

    # Inject a test-only source into the real target in the disposable source copy.
    fixture = work / "phase_002_fixture.mm"
    fixture.write_text("""
#import <Metal/Metal.h>

#if !__has_feature(objc_arc) || __cplusplus < 202002L
#error Objective-C++20 with ARC required
#endif

id<MTL4CommandQueue> Phase002Queue(id<MTLDevice> device) {
    return [device newMTL4CommandQueue];
}
""")
    include = work / "fixture.cmake"
    include.write_text(f'cmake_language(DEFER CALL target_sources video_core PRIVATE [[{fixture}]])\n')
    phase001.configure(source, build, "-DENABLE_METAL=ON", f"-DCMAKE_PROJECT_citra_INCLUDE={include}")
    entries = phase001.inspect_graph(build)
    by_file = {Path(entry["file"]).name: entry for entry in entries}
    command = by_file[fixture.name]
    assert "-fobjc-arc" in command["command"]
    assert "-fobjc-arc" not in by_file["gpu.cpp"]["command"]
    assert not {"glsl_shader_gen.cpp", "glsl_shader_decompiler.cpp", "glsl_fs_shader_gen.cpp"} & by_file.keys()
    assert {"shader_interpreter.cpp", "sw_blitter.cpp", "shader_gen.cpp"} <= by_file.keys()
    fixture_output = Path(command["output"])
    if fixture_output.is_absolute():
        fixture_output = fixture_output.relative_to(build)
    phase001.run("ninja", "-C", build, fixture_output)
    phase001.run("xcrun", "nm", "-g", build / fixture_output)

    link = phase001.run("ninja", "-C", build, "-t", "commands", "bin/Release/tests").splitlines()[-1]
    assert all(f"-framework {framework}" in link for framework in ("Metal", "MetalFX", "QuartzCore"))
    assert all(word not in link for word in ("MoltenVK", "glslang", "SPIRV", "sirit"))

    baseline = work / "off"
    phase001.configure(source, baseline, "-DENABLE_METAL=OFF")
    off_db = phase001.inspect_graph(baseline)
    assert {"glsl_shader_gen.cpp", "glsl_shader_decompiler.cpp", "glsl_fs_shader_gen.cpp"} <= {
        Path(entry["file"]).name for entry in off_db
    }
    assert "METALFX_LIBRARY" not in (baseline / "CMakeCache.txt").read_text()
    print("PASS Phase 002 compiler flags, framework link, native dependency isolation and existing graph")


if __name__ == "__main__":
    main()
