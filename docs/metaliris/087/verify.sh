#!/bin/zsh
set -eux
source_mirror="${1:?provide isolated source mirror}"
qt="${2:?provide native Qt build}"
off="${3:?provide native-off Qt build path}"
core="${4:?provide native core build}"
cmake -S "$source_mirror" -B "$qt"
qt_base_dir="$(sed -n 's/^Qt6_DIR:[^=]*=//p' "$qt/CMakeCache.txt")"
qt_multimedia_dir="$(sed -n 's/^Qt6Multimedia_DIR:[^=]*=//p' "$qt/CMakeCache.txt")"
[[ -d "$qt_base_dir" && -d "$qt_multimedia_dir" ]]
cmake -S "$source_mirror" -B "$off" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_OSX_ARCHITECTURES=arm64 \
    -DCMAKE_OSX_DEPLOYMENT_TARGET=26.0 -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DENABLE_METAL=OFF -DENABLE_VULKAN=OFF -DENABLE_OPENGL=OFF \
    -DENABLE_SOFTWARE_RENDERER=ON -DENABLE_QT=ON -DUSE_SYSTEM_QT=ON \
    -DENABLE_QT_UPDATE_CHECKER=OFF -DENABLE_QT_TRANSLATION=OFF -DENABLE_FFMPEG=OFF \
    -DENABLE_TESTS=ON -DUSE_SYSTEM_OPENSSL=ON -DUSE_SYSTEM_SDL2=ON \
    "-DCMAKE_OSX_SYSROOT=$(xcrun --sdk macosx --show-sdk-path)" \
    "-DQt6_DIR=$qt_base_dir" "-DQt6Multimedia_DIR=$qt_multimedia_dir"
python3 - "$qt" "$off" <<'PY'
import json
from pathlib import Path
import shlex
import subprocess
import sys

def entries(directory):
    contents = json.loads((Path(directory)/"compile_commands.json").read_text())
    assert all("/renderer_vulkan/" not in e["file"] and
               "-DENABLE_VULKAN" not in shlex.split(e["command"]) for e in contents)
    return contents

native, off = (entries(path) for path in sys.argv[1:])
bridge, = [e for e in native if e["file"].endswith("/citra_qt/util/metal_util.mm")]
assert "/citra_qt.dir/" in bridge["command"]
assert shlex.split(bridge["command"]).count("-fobjc-arc") == 1
assert "-DENABLE_METAL" in shlex.split(bridge["command"])
legacy = [e["command"] for e in off if e["file"].endswith("/citra_qt/util/metal_util.mm")]
assert len(legacy) == 2 and any("citra_meta.dir/" in command for command in legacy)
assert all("-fobjc-arc" not in shlex.split(command) and "-DENABLE_METAL" not in shlex.split(command)
           for command in legacy)
assert any("/renderer_software/" in e["file"] for e in off)
assert all("/renderer_metal/" not in e["file"] and
           "-DENABLE_METAL" not in shlex.split(e["command"]) for e in off)
assert sum(e["file"].endswith("/citra_qt/qt_swizzle.mm") for e in native) == 2
surface, = [e["command"] for e in native if e["file"].endswith("/renderer_metal/mtl_surface_adapter.mm")]
assert "QtWidgets.framework" not in surface and "QtCore.framework" not in surface
qt_link = subprocess.check_output(["ninja", "-C", sys.argv[1], "-t", "commands",
    "bin/Release/azahar.app/Contents/MacOS/azahar"], text=True).splitlines()
link, = [line for line in qt_link if "-o bin/Release/azahar.app/Contents/MacOS/azahar " in line]
assert "citra_meta.dir/__/citra_qt/util/metal_util.mm.o" not in link
assert "MoltenVK" not in link and "vulkan" not in link
for framework in ("AppKit", "QuartzCore", "Metal", "MetalFX"):
    assert f"-framework {framework}" in link, framework
print("PASS one Qt ARC bridge, original native-off registration and neutral renderer framework closure")
PY
cmake --build "$qt" --target citra_meta --parallel 10
cmake --build "$off" --target tests citra_meta --parallel 8
app="$qt/bin/Release/azahar.app/Contents/MacOS/azahar"
otool -L "$app"
if nm -m "$app" | rg -i 'MoltenVK|VkGetInstanceProcAddr|Vulkan::|_vkCreateInstance'; then
    exit 1
fi
[[ "$(nm -gU "$app" | rg -c 'T .*AttachMetalSurface')" == 1 ]]
off_app="$off/bin/Release/azahar.app/Contents/MacOS/azahar"
off_frameworks="$(otool -L "$off_app")"
print -r -- "$off_frameworks"
[[ "$off_frameworks" == */QuartzCore.framework/* ]]
# QtGui declares Metal.framework in its interface even for a Software-only application.
if print -r -- "$off_frameworks" | rg -i '/MetalFX\.framework/|MoltenVK|Vulkan'; then
    exit 1
fi
if nm -m "$off_app" | rg -i 'AttachMetalSurface|RendererMetal|MTL4Command|MTL4Compiler|MTLCreateSystemDefaultDevice|MTLCopyAllDevices|MoltenVK|VkGetInstanceProcAddr|Vulkan::|_vkCreateInstance'; then
    exit 1
fi
print "PASS native and Software-only Qt app links and framework closure"
"$core/bin/Release/tests" '[video_core]' --rng-seed 225818237
