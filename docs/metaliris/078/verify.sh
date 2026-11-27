#!/bin/zsh
set -eu
here="${0:a:h}"
source_root="${here:h:h:h}"
build_dir="${1:?provide the native Qt CMake build directory}"
output="${2:?provide a writable output directory}"
mkdir -p "$output"
output="${output:a}"

python3 - "$here" "$source_root" "$build_dir" "$output" <<'PY'
import json
from pathlib import Path
import shlex
import subprocess
import sys

here, source, build, output = map(lambda arg: Path(arg).resolve(), sys.argv[1:])
entries = json.loads((build / "compile_commands.json").read_text())
entry = next(e for e in entries if e["file"].endswith("/citra_qt/util/metal_util.mm")
             and "CMakeFiles/citra_qt.dir/" in e["command"])
base = shlex.split(entry["command"])
assert "-DENABLE_METAL" in base and "-DENABLE_VULKAN" not in base
base.insert(1, f"-I{source / 'src'}")
base[base.index("-x")+1] = "c++"
if "-fobjc-arc" in base:
    base.remove("-fobjc-arc")

def run(command):
    print("+", shlex.join(command), flush=True)
    subprocess.run(command, cwd=build, check=True)

for mode, flags in (("native", []), ("off", ["-UENABLE_METAL"])):
    command = base.copy()
    command[command.index("-c")+1] = str(here / "verify.cpp")
    object_file = output / (mode + ".o")
    command[command.index("-o")+1] = str(object_file)
    run(command + ["-MD", "-MF", str(object_file.with_suffix(".d")), *flags])
    framework_dirs = [arg for arg in base if arg.startswith("-F")]
    framework_dirs += ["-F" + base[index + 1] for index, arg in enumerate(base)
                       if arg == "-iframework"]
    run([base[0], "-arch", "arm64", "-mmacosx-version-min=26.0", *framework_dirs,
         str(object_file), "-framework", "QtCore", "-o", str(output / mode)])
    run([str(output / mode)])
PY
if rg -i 'renderer_vulkan|moltenvk|vulkan\.h|shaderc|spirv' "$output"/*.d; then
    exit 1
fi
