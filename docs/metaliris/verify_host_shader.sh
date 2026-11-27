#!/bin/zsh
set -eu
doc_root="${0:a:h}"
source_root="${doc_root:h:h}"
phase="${1:?provide the production phase}"
build_dir="${2:?provide the native CMake build directory}"
output="${3:?provide a writable output directory}"
shift 3
here="$doc_root/$phase"
mkdir -p "$output"
output="${output:a}"
shader_path="$(python3 - "$doc_root/progress.json" "$phase" <<'PY'
import json
import sys
with open(sys.argv[1]) as stream:
    phase, = (entry for entry in json.load(stream)["phases"] if entry["id"] == sys.argv[2])
name = phase["owner"]
assert name.startswith("src/video_core/host_shaders/metal/") and name.endswith(".metal")
print(name)
PY
)"
shader="${shader_path:t:r}"
sdk="$(xcrun --sdk macosx --show-sdk-path)"
profile=0
if rg -q '^ENABLE_MICROPROFILE:BOOL=ON$' "$build_dir/CMakeCache.txt"; then
    profile=1
fi
flags=(-arch arm64 -isysroot "$sdk" -mmacosx-version-min=26.0 -std=c++20 -fobjc-arc
       -DENABLE_SOFTWARE_RENDERER "-DMICROPROFILE_ENABLED=$profile" -I "$source_root/src" -I "$build_dir/src"
       -isystem "$source_root/externals/boost" -isystem "$source_root/externals/fmt/include"
       -isystem "$source_root/externals/microprofile" -isystem "$source_root/externals/nihstro/include"
       -isystem "$source_root/externals/xxHash" -Wall -Wextra -Werror -Wno-unused-parameter)

set -x
xcrun --sdk macosx metal -std=metal4.0 -mmacosx-version-min=26.0 -Werror \
    "-fmodules-cache-path=$output/modules" -MD -MF "$output/native-msl.d" \
    "$source_root/$shader_path" -o "$output/$shader.metallib"
file "$output/$shader.metallib"
fixture_libraries=()
if [[ -f "$here/verify.metal" ]]; then
    xcrun --sdk macosx metal -std=metal4.0 -mmacosx-version-min=26.0 -Werror \
        "-fmodules-cache-path=$output/modules" -MD -MF "$output/fixture-msl.d" \
        "$here/verify.metal" -o "$output/fixture.metallib"
    fixture_libraries+=("$output/fixture.metallib")
fi
xcrun --sdk macosx clang++ "${flags[@]}" -MD -MF "$output/fixture.d" \
    -c "$here/verify.mm" -o "$output/fixture.o"
python3 - "$doc_root" "$build_dir" "$output" <<'PY'
import importlib.util
from pathlib import Path
import shlex
import subprocess
import sys

sys.dont_write_bytecode = True
docs, build, output = map(Path, sys.argv[1:])
spec = importlib.util.spec_from_file_location("phase007", docs / "007/verify.py")
phase007 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(phase007)
command = phase007.link_command(build, [output / "fixture.o"], output / "native-shader-test")
print("+", shlex.join(command), flush=True)
subprocess.run(command, cwd=build, check=True)
PY
env MTL_DEBUG_LAYER=1 "$output/native-shader-test" "$output/$shader.metallib" "${fixture_libraries[@]}" "$@"
