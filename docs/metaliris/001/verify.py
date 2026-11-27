"""Phase 001 verification against the real source tree staged in a writable directory."""

import argparse
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[3]
BASELINE = "c621eafbf316699c55f4c9c0c689d4e49498d360"


def run(*command, expect=None):
    print("+", subprocess.list2cmdline([str(arg) for arg in command]), flush=True)
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(result.stdout, end="")
    if expect is None:
        assert result.returncode == 0, result.returncode
    else:
        assert result.returncode and expect in result.stdout, result.stdout
    return result.stdout


def configure(source, build, *flags, expect=None, explicit_arch=True):
    sdk = subprocess.check_output(["xcrun", "--sdk", "macosx", "--show-sdk-path"], text=True).strip()
    arch = ("-DCMAKE_OSX_ARCHITECTURES=arm64",) if explicit_arch else ()
    return run(
        "cmake", "-S", source, "-B", build, "-G", "Ninja",
        *arch, f"-DCMAKE_OSX_SYSROOT={sdk}", "-DCMAKE_BUILD_TYPE=Release",
        "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON", "-DCMAKE_PREFIX_PATH=/opt/homebrew",
        "-DENABLE_TESTS=ON", "-DENABLE_LTO=OFF", "-DENABLE_FFMPEG=OFF",
        "-DENABLE_QT=OFF", "-DENABLE_VULKAN=OFF",
        "-DUSE_SYSTEM_OPENSSL=ON", "-DUSE_SYSTEM_SDL2=ON",
        "-DCITRA_USE_PRECOMPILED_HEADERS=OFF", *flags, expect=expect,
    )


def baseline_source():
    return subprocess.check_output(["git", "-C", ROOT, "show", f"{BASELINE}:CMakeLists.txt"], text=True)


def inspect_graph(build):
    compile_db = json.loads((build / "compile_commands.json").read_text())
    commands = "\n".join(entry["command"] for entry in compile_db)
    ninja = (build / "build.ninja").read_text()
    forbidden = ("/renderer_vulkan/", "MoltenVK", "/vulkan-headers/", "/vma/include",
                 "/spirv-tools/", "/spirv-headers/", "/glslang/", "/sirit/", "-DENABLE_VULKAN")
    for item in forbidden:
        assert item not in commands and item not in ninja, item
    assert "CMakeFiles/tests.dir" in ninja
    return compile_db


def validate_options(source, work):
    for cmake_source, original in ((baseline_source(), True), (source.read_text(), False)):
        declaration = re.search(r"^CMAKE_DEPENDENT_OPTION\(ENABLE_VULKAN [^\n]+", cmake_source, re.M)[0]
        fixture = work / ("baseline-options.cmake" if original else "options.cmake")
        fixture.write_text(
            "cmake_minimum_required(VERSION 3.25)\ninclude(CMakeDependentOption)\n"
            f"{declaration}\n"
            'if(NOT ENABLE_VULKAN STREQUAL EXPECTED)\n'
            '    message(FATAL_ERROR "The Vulkan option evaluated to ${ENABLE_VULKAN}")\nendif()\n'
        )
        for bsd, metal, expected in (("", "OFF", "ON"), ("NetBSD", "OFF", "OFF"),
                                     ("", "ON", "ON" if original else "OFF")):
            run("cmake", f"-DBSD={bsd}", f"-DENABLE_METAL={metal}", f"-DEXPECTED={expected}", "-P", fixture)
        run("cmake", "-DENABLE_VULKAN=OFF", "-DEXPECTED=OFF", "-P", fixture)
        run("cmake", "-DENABLE_VULKAN=ON", "-DENABLE_METAL=ON",
            f"-DEXPECTED={'ON' if original else 'OFF'}", "-P", fixture)

    cache_project = work / "option-cache-fixture"
    cache_project.mkdir(exist_ok=True)
    (cache_project / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.25)\nproject(OptionCache NONE)\n"
        f"include(\"{fixture}\")\n"
    )
    cache_build = work / "option-cache"
    run("cmake", "-S", cache_project, "-B", cache_build, "-DENABLE_VULKAN=ON",
        "-DENABLE_METAL=ON", "-DEXPECTED=OFF")
    run("cmake", "-S", cache_project, "-B", cache_build, "-DENABLE_METAL=OFF", "-DEXPECTED=ON")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=("source", "options", "metal", "off", "qt", "architecture",
                                         "platform", "deployment", "failure", "native", "pica"))
    parser.add_argument("work", type=Path)
    args = parser.parse_args()
    work = args.work.resolve()
    source = work / "source"
    baseline = work / "baseline-source"
    metal = "-DENABLE_METAL=ON"
    build = work / args.case

    if args.case == "source":
        old = baseline_source().splitlines()
        new = (source / "CMakeLists.txt").read_text().splitlines()
        retained = iter(new)
        changed = {64, 65, 151, 152}
        for line, statement in enumerate(old, 1):
            if line not in changed:
                assert any(statement == candidate for candidate in retained), (line, statement)
        print(f"All {len(old) - len(changed)} original lines outside the owned changes are retained.")
    elif args.case == "options":
        validate_options(source / "CMakeLists.txt", work)
    elif args.case == "metal":
        output = configure(source, build, metal, "-DENABLE_VULKAN=ON")
        assert "AZAHAR_METAL4_SDK - Success" in output
        compile_db = inspect_graph(build)
        assert any("-mmacosx-version-min=26.0" in entry["command"] for entry in compile_db)
    elif args.case == "off":
        flags = ("-DENABLE_METAL=OFF",)
        configure(baseline, work / "baseline-build")
        output = configure(source, build, *flags)
        assert "AZAHAR_METAL4_SDK" not in output
        old = inspect_graph(work / "baseline-build")
        new = inspect_graph(build)
        def normalized(entries, root, binary):
            return sorted(entry["command"].replace(str(root), "{source}").replace(str(binary), "{build}")
                          for entry in entries)
        assert normalized(old, baseline, work / "baseline-build") == normalized(new, source, build)
        assert any("-mmacosx-version-min=13.4" in entry["command"] for entry in new)
    elif args.case == "qt":
        configure(source, build, metal, "-DENABLE_QT=ON", "-DUSE_SYSTEM_QT=ON",
                  "-DENABLE_QT_TRANSLATION=OFF",
                  "-DCMAKE_PREFIX_PATH=/opt/homebrew/opt/qtbase;/opt/homebrew/opt/qtmultimedia;/opt/homebrew",
                  "-DQt6_DIR=/opt/homebrew/opt/qtbase/lib/cmake/Qt6",
                  "-DQt6Multimedia_DIR=/opt/homebrew/opt/qtmultimedia/lib/cmake/Qt6Multimedia")
        inspect_graph(build)
        assert "CMakeFiles/citra_qt.dir" in (build / "build.ninja").read_text()
    elif args.case == "architecture":
        for directory, arch in (("intel", "x86_64"), ("universal", "arm64;x86_64")):
            configure(source, work / directory, metal, f"-DCMAKE_OSX_ARCHITECTURES={arch}",
                      expect="ENABLE_METAL requires an arm64 macOS target")
        configure(source, work / "implicit", metal, explicit_arch=False)
        inspect_graph(work / "implicit")
        configure(source, work / "implicit-intel", metal, "-DCMAKE_OBJCXX_FLAGS=-arch x86_64",
                  explicit_arch=False, expect="Metal-enabled Apple Silicon SDK")
    elif args.case == "platform":
        diagnostic = "ENABLE_METAL is supported only for desktop macOS"
        configure(source, work / "libretro", metal, "-DENABLE_LIBRETRO=ON", expect=diagnostic)
        configure(source, work / "ios", metal, "-DIOS=ON", expect=diagnostic)
        configure(source, work / "generic-platform", metal, "-DCMAKE_SYSTEM_NAME=Generic",
                  expect=diagnostic)
    elif args.case == "deployment":
        configure(source, work / "old-os", metal, "-DCMAKE_OSX_DEPLOYMENT_TARGET=15.0",
                  expect="ENABLE_METAL requires a macOS deployment target of 26.0 or later")
        for version in ("26.0", "26.1"):
            directory = work / f"target-{version}"
            configure(source, directory, metal, f"-DCMAKE_OSX_DEPLOYMENT_TARGET={version}")
            assert any(f"-mmacosx-version-min={version}" in entry["command"]
                       for entry in inspect_graph(directory))
    elif args.case == "failure":
        bad_header = work / "unusable-sdk.h"
        bad_header.write_text("#ifdef AZAHAR_METAL4_SDK\n#error injected_sdk_failure\n#endif\n")
        flags = ("-DCMAKE_OBJCXX_FLAGS=-include " + str(bad_header),)
        configure(source, work / "fresh-failure", metal, *flags, expect="Metal-enabled Apple Silicon SDK")
        configure(source, work / "cached-failure", metal)
        configure(source, work / "cached-failure", metal, *flags, expect="Metal-enabled Apple Silicon SDK")
        configure(source, work / "cached-failure", metal, "-DCMAKE_OBJCXX_FLAGS=")
    elif args.case == "native":
        script = ROOT / "docs/metaliris/preflight/harness/run-native.sh"
        result = subprocess.run(["zsh", script, build], check=False)
        assert (build / "smoke.air").stat().st_size
        assert (build / "smoke.metallib").stat().st_size
        assert (build / "device-smoke").stat().st_size
        run("otool", "-L", build / "device-smoke")
        if result.returncode == 2:
            print("PENDING: native command submission, object/mesh output, Spatial output and GPU capture")
            raise SystemExit(2)
        assert result.returncode == 0, result.returncode
        assert (build / "workload.gputrace").exists()
    elif args.case == "pica":
        metal_build = work / "metal"
        # The preceding case configured this real Ninja tree with Vulkan disabled.
        run("cmake", "--build", metal_build, "--target", "tests", "--parallel", "8")
        run("ctest", "--test-dir", metal_build, "--output-on-failure", "-j", "4")
        output = run(metal_build / "bin/Release/tests", "[video_core]", "--rng-seed", "225818237")
        assert "801 assertions in 48 test cases" in output

    print(f"PASS {args.case}")


if __name__ == "__main__":
    main()
