#!/bin/zsh
set -eu
here="${0:a:h}"
source_root="${here:h:h:h}"
qt_dir="${1:?provide the native Qt CMake build directory}"
core_dir="${2:?provide the native core CMake build directory}"
output="${3:?provide a writable output directory}"
mkdir -p "$output"
output="${output:a}"

python3 - "$here" "$source_root" "$qt_dir" "$core_dir" "$output" <<'PY'
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess
import sys

sys.dont_write_bytecode=True
here, source, qt, core, output = map(lambda arg: Path(arg).resolve(), sys.argv[1:])
database = json.loads((qt / "compile_commands.json").read_text())
entry, = [e for e in database if e["file"].endswith("/citra_qt/util/metal_util.mm")]
assert "CMakeFiles/citra_qt.dir/" in entry["command"]

def run(command, cwd=qt):
    print("+", shlex.join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)

def compile_tu(base, path, stem, flags=()):
    command=base.copy()
    command.insert(1,f"-I{source / 'src'}")
    obj=output/(stem+".o")
    command[command.index("-c")+1] = str(path)
    command[command.index("-o")+1] = str(obj)
    run(command + ["-MD","-MF",str(obj.with_suffix(".d")), *flags])
    return obj

owner=source/"src/citra_qt/util/metal_util.mm"
qt_command=shlex.split(entry["command"])
assert qt_command.count("-fobjc-arc") == 1 and "-DENABLE_METAL" in qt_command
produced=compile_tu(qt_command,owner,"qt-bridge")
compile_tu(qt_command,owner,"off-bridge",["-UENABLE_METAL","-fno-objc-arc"])
fixture=compile_tu(qt_command,here/"verify.mm","consumer",["-UNDEBUG"])

spec=importlib.util.spec_from_file_location("phase007",here.parent/"007/verify.py")
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
exe=output/"qt-metal-bridge"
link=module.link_command(core,[fixture,produced],exe)
link += ["-F"+qt_command[i+1] for i,a in enumerate(qt_command) if a=="-iframework"]
link += ["-framework","QtCore","-framework","QtGui"]
run(link,core)
run([str(exe)],core)
PY
if rg -i 'renderer_vulkan|moltenvk|vulkan\.h|shaderc|spirv' "$output"/*.d; then
    exit 1
fi
