"""Exercise real MSL4 compilation, embedding and dependency tracking in an isolated source copy."""

import argparse
import importlib.util
from pathlib import Path
import re

SPEC = importlib.util.spec_from_file_location("phase001", Path(__file__).parent.parent / "001/verify.py")
phase001 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(phase001)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work", type=Path)
    work = parser.parse_args().work.resolve()
    source = work / "source"
    host = source / "src/video_core/host_shaders"
    native = work / "native"
    fixture = host / "metal"
    fixture.mkdir(exist_ok=True)
    (fixture / "phase003_fixture.metal").write_text(
        '#include <metal_stdlib>\n#include "phase003_value.h"\n'
        'kernel void phase003_fixture(device uint* result [[buffer(0)]]) {\n'
        '    *result = Phase003Value;\n}\n'
    )
    value = fixture / "phase003_value.h"
    initial = "constant uint Phase003Value = 0x52454333;\n"
    value.write_text(initial)

    owner = host / "CMakeLists.txt"
    code = (phase001.ROOT / owner.relative_to(source)).read_text()
    anchor = "    # Register completed Metal shader sources here."
    assert code.count(anchor) == 1
    owner.write_text(code.replace(anchor, anchor + "\n    add_metal_host_shader(phase003_fixture)"))

    phase001.configure(source, native, "-DENABLE_METAL=ON")
    graph = (native / "build.ninja").read_text()
    assert "phase003_fixture.metallib" in graph
    assert "vulkan_present_frag.h" not in graph and "opengl_present_frag.h" not in graph
    compile_shader = ("cmake", "--build", native, "--target", "host_shaders", "--verbose")
    phase001.run(*compile_shader)

    outputs = native / "src/video_core/host_shaders"
    library = outputs / "metal/phase003_fixture.metallib"
    include = outputs / "include"
    header = include / "video_core/host_shaders/metal/phase003_fixture.h"
    import_path = header.relative_to(include)
    original_library = library.read_bytes()
    original_header = header.read_bytes()
    assert original_library[:4] == b"MTLB"
    assert bytes(int(byte, 16) for byte in re.findall(rb"0x([0-9a-f]{2})", original_header)) == original_library
    phase001.run("file", library)
    assert "no work to do" in phase001.run(*compile_shader)
    library.unlink()
    phase001.run(*compile_shader)
    assert library.read_bytes() == original_library
    assert header.read_bytes() == original_header

    value.write_text("constant uint Phase003Value = 0x52454334;\n")
    phase001.run(*compile_shader)
    assert library.read_bytes() != original_library and header.read_bytes() != original_header
    value.write_text("#error phase003_invalid_fixture_include\n")
    phase001.run(*compile_shader, expect="phase003_invalid_fixture_include")
    value.write_text(initial)
    phase001.run(*compile_shader)
    assert library.read_bytes() == original_library

    consumer = work / "phase003_consumer.mm"
    consumer.write_text(f"""
#include "{import_path}"
static_assert(sizeof(HostShaders::Metal::PHASE003_FIXTURE) == {len(original_library)});
int main() {{
    return HostShaders::Metal::PHASE003_FIXTURE[0] != 'M';
}}
""")
    app = work / "phase003_consumer"
    phase001.run("xcrun", "--sdk", "macosx", "clang++", "-std=c++20", "-fobjc-arc",
                 "-I", include, consumer, "-o", app)
    phase001.run(app)

    baseline = work / "off"
    phase001.configure(source, baseline, "-DENABLE_METAL=OFF")
    graph = (baseline / "build.ninja").read_text()
    assert "vulkan_present_frag.h" in graph and "opengl_present_frag.h" in graph
    assert "phase003_fixture.metallib" not in graph
    print("PASS Phase 003 matching embedded data, dependency rebuild, failed build recovery, and Metal-off graph")


if __name__ == "__main__":
    main()
