#!/bin/zsh
set -eu
here="${0:a:h}"
source_root="${here:h:h:h}"
build_dir="${1:?provide the native CMake build directory}"
output="${2:?provide a writable output directory}"
mkdir -p "$output"
output="${output:a}"
sdk="$(xcrun --sdk macosx --show-sdk-path)"
profile=0
if rg -q '^ENABLE_MICROPROFILE:BOOL=ON$' "$build_dir/CMakeCache.txt"; then
    profile=1
fi
flags=(-arch arm64 -isysroot "$sdk" -mmacosx-version-min=26.0 -std=c++20 -fobjc-arc
       -DENABLE_METAL -DENABLE_SOFTWARE_RENDERER "-DMICROPROFILE_ENABLED=$profile" -I "$source_root/src" -I "$build_dir/src" -I "$build_dir/src/video_core/host_shaders/include"
       -isystem "$source_root/externals/boost" -isystem "$source_root/externals/fmt/include" -isystem "$source_root/externals/microprofile" -isystem "$source_root/externals/nihstro/include" -isystem "$source_root/externals/xxHash" -isystem "$source_root/externals/dds-ktx" -Wall -Wextra -Wno-ignored-qualifiers -Werror -Wno-unused-parameter)

set -x
xcrun --sdk macosx clang++ "${flags[@]}" -MD -MF "$output/present-window.d" \
    -c "$source_root/src/video_core/renderer_metal/mtl_present_window.mm" -o "$output/present-window.o"
xcrun --sdk macosx clang++ "${flags[@]}" -c "$here/verify.mm" -o "$output/present-window-test.o"
python3 - "$here" "$build_dir" "$output" <<'PY'
import importlib.util
from pathlib import Path
import shlex
import subprocess
import sys

sys.dont_write_bytecode = True
here, build, output = map(Path, sys.argv[1:])
spec = importlib.util.spec_from_file_location("phase007", here.parent / "007/verify.py")
phase007 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(phase007)
command = phase007.link_command(build, [output / "present-window-test.o", output / "present-window.o"],
                                output / "metal-present-window-tests")
print("+", shlex.join(command), flush=True)
subprocess.run(command, cwd=build, check=True)
PY

python3 - "$source_root/src/video_core/renderer_metal/mtl_present_window.mm" "$output/present-scale.metal" <<'PY2'
from pathlib import Path
import re
import sys
source, shader = map(Path, sys.argv[1:])
match = re.search(r'R"MSL\((.*?)\)MSL"', source.read_text(), re.S)
assert match
shader.write_text(match[1])
PY2
xcrun --sdk macosx metal -std=metal4.0 -mmacosx-version-min=26.0 -fno-fast-math \
    "-fmodules-cache-path=$output/modules" -Werror -MD -MF "$output/present-scale.d" \
    "$output/present-scale.metal" -o "$output/present-scale.metallib"
file "$output/present-scale.metallib"
printf '%s\n' 'PASS direct Metal 4 MSL presentation scale program and linked window fixture'
if rg -i 'renderer_vulkan|moltenvk|vulkan\.h|shaderc|spirv' "$output"/*.d; then
    exit 1
fi
env MTL_DEBUG_LAYER=1 "$output/metal-present-window-tests"
