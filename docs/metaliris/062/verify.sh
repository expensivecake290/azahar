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
       -DENABLE_SOFTWARE_RENDERER "-DMICROPROFILE_ENABLED=$profile" -I "$source_root/src" -I "$build_dir/src"
       -isystem "$source_root/externals/boost" -isystem "$source_root/externals/fmt/include" -isystem "$source_root/externals/microprofile" -isystem "$source_root/externals/nihstro/include" -isystem "$source_root/externals/xxHash" -Wall -Wextra -Wno-ignored-qualifiers -Werror -Wno-unused-parameter)

set -x
xcrun --sdk macosx clang++ "${flags[@]}" -MD -MF "$output/pipeline-cache.d" \
    -c "$source_root/src/video_core/renderer_metal/mtl_pipeline_cache.mm" -o "$output/pipeline-cache.o"
xcrun --sdk macosx clang++ "${flags[@]}" -c "$here/verify.mm" -o "$output/pipeline-cache-test.o"
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
command = phase007.link_command(build, [output / "pipeline-cache-test.o", output / "pipeline-cache.o"],
                                output / "metal-pipeline-cache-tests")
print("+", shlex.join(command), flush=True)
subprocess.run(command, cwd=build, check=True)
PY

rm -f "$output"/*.metal(N)
fixture_exit=0
env MTL_DEBUG_LAYER=1 "$output/metal-pipeline-cache-tests" "$output" "$here/../058/fixture.metal" || fixture_exit=$?
if (( fixture_exit != 0 && fixture_exit != 2 )); then
    exit "$fixture_exit"
fi
for source in "$output"/*.metal; do
    xcrun --sdk macosx metal -std=metal4.0 -mmacosx-version-min=26.0 -fno-fast-math \
        "-fmodules-cache-path=$output/modules" -Werror -Wno-unused-variable -Wno-unused-function -MD -MF "${source:r}.d" "$source" -o "${source:r}.metallib"
done
printf '%s\n' 'PASS offline native generated PICA Metal 4 corpus'
exit "$fixture_exit"
