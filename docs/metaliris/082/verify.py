import argparse
import importlib.util
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SPEC = importlib.util.spec_from_file_location("phase007", HERE.parent / "007" / "verify.py")
CORE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CORE)


def run(command, build):
    command = [str(path) for path in command]
    print("+", shlex.join(command), flush=True)
    subprocess.run(command, cwd=build, check=True)


def compile_flags(build, source):
    entries = json.loads((build / "compile_commands.json").read_text())
    entry = next(e for e in entries if e["file"].endswith("/" + source))
    command = shlex.split(entry["command"])
    assert "-DENABLE_METAL" in command and "-DENABLE_VULKAN" not in command
    command.insert(1, f"-I{ROOT / 'src'}")
    return command


def compile_unit(base, source, obj, build, flags=()):
    command = base.copy()
    command[command.index("-c")+1] = str(source)
    command[command.index("-o")+1] = str(obj)
    dependency = obj.with_suffix(".d")
    run([*command, "-MD", "-MF", dependency, *flags], build)
    assert not re.search(r"renderer_vulkan|moltenvk|vulkan\.h|shaderc|spirv", dependency.read_text(), re.I)


def frameworks(base):
    return [arg for arg in base if arg.startswith("-F")] + [
        "-F" + base[i + 1] for i, arg in enumerate(base) if arg == "-iframework"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("qt", type=Path)
    parser.add_argument("core", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    qt, core, output = [p.resolve() for p in (args.qt, args.core, args.output)]
    output.mkdir(exist_ok=True, parents=True)
    unit = "citra_qt/util/graphics_device_info.cpp"
    base = compile_flags(qt, unit)
    production = output / "graphics_device_info.o"
    compile_unit(base, ROOT / "src" / unit, production, qt)
    compile_unit(base, ROOT / "src" / unit, output / "off.o", qt, ["-UENABLE_METAL"])
    bridge_unit = "citra_qt/util/metal_util.mm"
    bridge = output / "bridge.o"
    compile_unit(compile_flags(qt, bridge_unit), ROOT / "src" / bridge_unit, bridge, qt)
    for name, flags, extra in [("real", [], [bridge]), ("deterministic", ["-DDETERMINISTIC_DEVICES"], [])]:
        obj = output / f"{name}.o"
        exe = output / name
        compile_unit(base, HERE / "verify.cpp", obj, qt, ["-UNDEBUG", *flags])
        linker = CORE.link_command(core, [obj, production, *extra], exe)
        run([*linker, *frameworks(base), "-framework", "QtCore"], core)
        run([exe], core)


if __name__ == "__main__":
    main()
