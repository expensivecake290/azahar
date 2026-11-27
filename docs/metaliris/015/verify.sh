#!/bin/zsh
set -eu
here="${0:a:h}"
source_root="${here:h:h:h}"
build_dir="${1:?provide the native CMake build directory}"
output="${2:?provide a writable output directory}"
mkdir -p "$output"
output="${output:a}"
sdk="$(xcrun --sdk macosx --show-sdk-path)"
flags=(-arch arm64 -isysroot "$sdk" -mmacosx-version-min=26.0 -std=c++20 -fobjc-arc
       -DENABLE_SOFTWARE_RENDERER -I "$source_root/src" -I "$build_dir/src"
       -isystem "$source_root/externals/boost" -isystem "$source_root/externals/fmt/include" -Wall -Wextra -Werror -Wno-unused-parameter)

set -x
xcrun --sdk macosx clang++ "${flags[@]}" -MD -MF "$output/pica-to-metal.d" \
    -c "$here/verify.mm" -o "$output/mapping-test.o"
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
command = phase007.link_command(build, [output / "mapping-test.o"],
                                output / "pica-to-metal-tests")
print("+", shlex.join(command), flush=True)
subprocess.run(command, cwd=build, check=True)
PY
env MTL_DEBUG_LAYER=1 "$output/pica-to-metal-tests"
