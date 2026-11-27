#!/bin/zsh
set -eu
here="${0:a:h}"
source_root="${here:h:h:h}"
build_dir="${1:?provide the native Qt CMake build directory}"
output="${2:?provide a writable output directory}"
mkdir -p "$output"
output="${output:a}"
python3 - "$source_root" "$build_dir" "$output" <<'PY'
import json
from pathlib import Path
import shlex
import subprocess
import sys

source, build, output = map(lambda arg: Path(arg).resolve(), sys.argv[1:])
entries = json.loads((build / "compile_commands.json").read_text())
entry = next(e for e in entries if e["file"].endswith("/citra_qt/bootmanager.cpp"))
base = shlex.split(entry["command"])
assert "-DENABLE_METAL" in base and "-DENABLE_VULKAN" not in base
base.insert(1, f"-I{source / 'src'}")
unit = source / "src/citra_qt/bootmanager.cpp"
header = unit.with_suffix(".h")
moc = Path("/opt/homebrew/opt/qtbase/share/qt/libexec/moc")

def run(command):
    print("+", shlex.join(command), flush=True)
    subprocess.run(command, cwd=build, check=True)

def compile_file(source_file, object_file, flags):
    command = base.copy()
    command[command.index("-c")+1] = str(source_file)
    command[command.index("-o")+1] = str(object_file)
    run(command + ["-MD", "-MF", str(object_file.with_suffix(".d")), *flags])

for mode, flags in (("native", []), ("off", ["-UENABLE_METAL"])):
    moc_unit = output / f"moc-{mode}.cpp"
    moc_options = [a for a in base if (a.startswith("-I") or a.startswith("-D")) and a != "-DENABLE_METAL"]
    if mode == "native":
        moc_options.append("-DENABLE_METAL")
    run([str(moc), *moc_options, str(header), "-o", str(moc_unit)])
    assert "GRenderWindow::qt_metacall" in moc_unit.read_text()
    compile_file(moc_unit, output / f"moc-{mode}.o", flags)
    compile_file(unit, output / f"bootmanager-{mode}.o", flags)
PY
if rg -i 'renderer_vulkan|moltenvk|vulkan\.h|shaderc|spirv' "$output"/*.d; then
    exit 1
fi
