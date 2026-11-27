#!/usr/bin/env python3
"""Capture actual compiler includes and the CMake File API target inventory."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
import shlex
import subprocess


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def classify(path, source_root, build_root):
    identity = str(path)
    if any(token in identity for token in ("/renderer_vulkan/", "/MoltenVK/", "/libretro_vulkan")):
        return "vulkan-private"
    if any(token in identity for token in ("/glslang/", "/spirv-tools/", "/spirv-headers/",
                                           "/vulkan-headers/", "/vma/", "/sirit/", "/include/vulkan/",
                                           "/shader/generator/spv_", "/host_shaders/vulkan_",
                                           "/format_reinterpreter/vulkan_")):
        return "vulkan-private"
    if "/host_shaders/" in identity or "/shader/generator/glsl_" in identity:
        return "excluded-shader-reference"
    if "/include/c++/" in identity:
        return "standard-library"
    if ".sdk/" in identity or "/usr/lib/clang/" in identity:
        return "apple-sdk"
    if "/src/citra_qt/" in identity or "/include/Qt" in identity or re.search(r"/Qt\w+\.framework/", identity):
        return "excluded-frontend"
    if "/src/citra_libretro/" in identity or "/src/android/" in identity:
        return "excluded-platform"
    if (path.is_relative_to(source_root / "externals") or path.is_relative_to(build_root / "externals")
            or "/opt/homebrew/" in identity):
        return "external-candidate"
    if path.is_relative_to(source_root / "src") or path.is_relative_to(build_root / "src"):
        return "azahar-candidate"
    return "unresolved"


def scan(entry, output, source_root, build_root):
    path = Path(entry["file"]).resolve()
    name = str(path.relative_to(source_root)).replace("/", "__")
    args = shlex.split(entry["command"])
    args.remove("-c")
    index = args.index("-o")
    del args[index:index + 2]
    args.extend(("-M", "-H", "-MT", "translation-unit"))
    result = subprocess.run(args, cwd=entry["directory"], capture_output=True, text=True)
    (output / f"{name}.d").write_text(result.stdout)
    (output / f"{name}.includes").write_text(result.stderr)

    stack = [path.resolve()]
    edges = []
    cycles = []
    for line in result.stderr.splitlines():
        match = re.match(r"^(\.+) (.+)$", line)
        if not match:
            continue
        depth = len(match[1])
        include = Path(match[2]).resolve()
        stack = stack[:depth]
        if stack:
            edges.append([str(stack[-1]), str(include)])
        if include in stack:
            cycles.append([str(item) for item in stack[stack.index(include):] + [include]])
        stack.append(include)

    paths = {Path(include) for _, include in edges}
    return {
        "file": str(path.relative_to(source_root)),
        "argv": args,
        "cwd": entry["directory"],
        "exit_code": result.returncode,
        "dependency_evidence": f"{name}.d",
        "include_trace": f"{name}.includes",
        "resolved_paths": {str(item): classify(item, source_root, build_root) for item in sorted(paths)},
        "edges": edges,
        "repeated_include_paths": cycles,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_root", type=Path)
    parser.add_argument("build_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--all-project", action="store_true")
    args = parser.parse_args()
    source_root, build_root, output = (path.resolve() for path in (
        args.source_root, args.build_root, args.output,
    ))
    output.mkdir(parents=True, exist_ok=True)

    reply = build_root / ".cmake/api/v1/reply"
    model = json.loads(next(reply.glob("codemodel-v2-*.json")).read_bytes())
    targets = []
    for configuration in model["configurations"]:
        names = {target["id"]: target["name"] for target in configuration["targets"]}
        for target in configuration["targets"]:
            data = json.loads((reply / target["jsonFile"]).read_bytes())
            targets.append({
                "name": target["name"],
                "type": data["type"],
                "dependencies": sorted(names.get(dep["id"], dep["id"]) for dep in data.get("dependencies", [])),
                "sources": [item["path"] for item in data.get("sources", [])],
            })
    write_json(output / "targets.json", targets)

    candidates = set("""
        src/common/settings.cpp src/core/frontend/emu_window.cpp
        src/video_core/video_core.cpp src/video_core/renderer_base.cpp src/video_core/gpu.cpp
        src/video_core/pica/pica_core.cpp src/video_core/pica/geometry_pipeline.cpp
        src/video_core/shader/shader_interpreter.cpp src/video_core/shader/generator/shader_gen.cpp
        src/video_core/shader/generator/pica_fs_config.cpp src/video_core/rasterizer_cache/rasterizer_cache.cpp
        src/tests/video_core/shader.cpp src/tests/video_core/pica_types.cpp
        src/citra_qt/bootmanager.cpp src/citra_qt/citra_qt.cpp src/citra_qt/util/metal_util.mm
        src/citra_qt/util/graphics_device_info.cpp src/citra_qt/configuration/config.cpp
        src/citra_qt/configuration/configure_graphics.cpp src/citra_qt/configuration/configure_debug.cpp
    """.split())
    commands = json.loads((build_root / "compile_commands.json").read_bytes())
    selected = {}
    for entry in commands:
        path = Path(entry["file"]).resolve()
        if not path.is_relative_to(source_root):
            continue
        relative = str(path.relative_to(source_root))
        if (relative in candidates or relative.startswith("src/video_core/renderer_vulkan/")
                or (args.all_project and relative.startswith("src/"))):
            selected.setdefault(relative, entry)

    with ThreadPoolExecutor(max_workers=3) as pool:
        rows = list(pool.map(lambda entry: scan(entry, output, source_root, build_root), selected.values()))
    registry = {}
    for row in rows:
        registry.update(row.pop("resolved_paths"))
    write_json(output / "include-registry.json", registry)
    with (output / "translation-units.jsonl").open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    print(f"{len(targets)} targets, {len(rows)} scans, "
          f"{sum(row['exit_code'] != 0 for row in rows)} failed, {len(registry)} resolved paths")
    return int(any(row["exit_code"] for row in rows))


if __name__ == "__main__":
    raise SystemExit(main())
