import argparse
import importlib.util
from pathlib import Path
import shlex
import subprocess
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("phase082", HERE.parent / "082" / "verify.py")
TOOLS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TOOLS)


def link_command(build, output, objects):
    target = "bin/Release/azahar.app/Contents/MacOS/azahar"
    recipe = subprocess.check_output(["ninja", "-C", str(build), "-t", "commands", target], text=True)
    link = next(line for line in recipe.splitlines() if f"-o {target} " in line)
    args = shlex.split(link)
    start = args.index("&&") + 1
    end = args.index("&&", start) if "&&" in args[start:] else len(args)
    args = args[start:end]
    args = [a for a in args if not a.endswith("/citra_meta.dir/main.cpp.o")]
    args[1:1] = [str(obj) for obj in objects]
    args[args.index("-o") + 1] = str(output)
    assert not any("vulkan" in a.lower() for a in args)
    return args


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("qt", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    qt, output = args.qt.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    base = TOOLS.compile_flags(qt, "citra_qt/bootmanager.cpp")
    base.insert(1, "-xobjective-c++")
    obj = output / "qt-surfaces.o"
    binary = output / "qt-surfaces"
    TOOLS.compile_unit(base, HERE / "verify.cpp", obj, qt, ["-UNDEBUG", "-fobjc-arc"])
    TOOLS.run(link_command(qt, binary, [obj]), qt)
    TOOLS.run([binary, "-platform", "offscreen"], qt)
    command = [str(binary), "-platform", "cocoa"]
    print("+", shlex.join(command), flush=True)
    result = subprocess.run(command, cwd=qt)
    if result.returncode != 77:
        result.check_returncode()


if __name__ == "__main__":
    main()
