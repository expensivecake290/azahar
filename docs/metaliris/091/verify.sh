#!/bin/zsh
set -eux
core="$1"
qt="$2"
output="$3"
here="$(dirname "$0")"

cmake --build "$core" --target host_shaders tests --parallel 8
cmake --build "$qt" --target citra_meta --parallel 8
python3 "$here/verify.py" "$core" "$output"
ctest --test-dir "$core" -C Release --output-on-failure -j 4
"$core/bin/Release/tests" '[video_core]' --rng-seed 225818237
python3 - "$core" "$qt" <<'PY'
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys

core, qt = map(Path, sys.argv[1:])
for build in (core, qt):
    entries = json.loads((build / "compile_commands.json").read_text())
    regex = re.compile(r"renderer_vulkan/|moltenvk|spirv|shaderc", re.I)
    assert not any(regex.search(entry["file"]) for entry in entries)
    assert not any("-DENABLE_VULKAN" in shlex.split(entry["command"]) for entry in entries)
    native = [entry for entry in entries if "/video_core/renderer_metal/" in entry["file"]]
    assert native
    seen = set()
    for entry in native:
        name = Path(entry["file"]).resolve()
        assert name not in seen
        seen.add(name)
        options = shlex.split(entry["command"])
        assert "-DENABLE_METAL" in options
        assert "QtCore.framework" not in entry["command"]
        if name.suffix == ".mm":
            assert options.count("-fobjc-arc") == 1
    print(f"PASS {build.name}: {len(native)} unique native source compile entries with Vulkan excluded")

shaders = core / "src/video_core/host_shaders"
payloads = sorted((shaders / "metal").glob("*.metallib"))
assert len(payloads) == 19
for payload in payloads:
    data = payload.read_bytes()
    assert data.startswith(b"MTLB")
    header = shaders / "include/video_core/host_shaders/metal" / f"{payload.stem}.h"
    packed = bytes(int(n, 16) for n in re.findall(r"0x([0-9a-fA-F]{2})", header.read_text()))
    assert packed == data, payload.stem
print("PASS all 19 production MSL4 libraries match the exact generated headers")

app = qt / "bin/Release/azahar.app/Contents/MacOS/azahar"
libraries = subprocess.check_output(["otool", "-L", app], text=True)
symbols = subprocess.check_output(["nm", "-gU", app], text=True)
undefined = subprocess.check_output(["nm", "-u", app], text=True)
assert "_ZN5Metal13RendererMetal" in symbols
assert re.search(r"T .*AttachMetalSurface", symbols)
assert not re.search(r"moltenvk|libvulkan|_vk(Create|GetInstanceProcAddr)|RendererVulkan",
                     libraries + symbols + undefined, re.I)
for framework in ("Metal.framework", "MetalFX.framework", "QuartzCore.framework", "AppKit.framework"):
    assert framework in libraries
print("PASS final arm64 Qt link contains actual native renderer and frameworks, without Vulkan runtime symbols")

archive = core / "src/video_core/libvideo_core.a"
native_references = subprocess.check_output(["nm", "-m", "-o", archive], text=True)
selectors = set(re.findall(r"_objc_msgSend\$([^\s]+)", native_references))
legacy = {"newCommandQueue", "blitCommandEncoder", "commandBufferWithUnretainedReferences",
          "newComputePipelineStateWithFunction:error:"}
assert not selectors.intersection(legacy)
assert {"newMTL4CommandQueueWithDescriptor:error:", "beginCommandBufferWithAllocator:"} <= selectors
ambiguous = [line for line in native_references.splitlines()
             if "_objc_msgSend$newRenderPipelineStateWithDescriptor:error:" in line]
assert len(ambiguous) == 1 and ":mtl_graphics_pipeline.mm.o:" in ambiguous[0]
entry = next(e for e in entries if e["file"].endswith("/mtl_graphics_pipeline.mm"))
pipeline = Path(entry["file"]).read_text()
assert "for (id<MTL4Archive> archive in" in pipeline
assert "[archive newRenderPipelineStateWithDescriptor:descriptor error:nil]" in pipeline
print("PASS native archive references Metal4 queue creation and allocator-bound encoding without audited legacy selectors")
PY
