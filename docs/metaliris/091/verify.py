import argparse
import importlib.util
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("phase082", HERE.parent / "082" / "verify.py")
TOOLS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TOOLS)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("core", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    core, output = args.core.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    base = TOOLS.compile_flags(core, "video_core/renderer_metal/renderer_metal.mm")
    assert "-fobjc-arc" in base
    unit = "video_core/renderer_metal/renderer_metal.mm"
    TOOLS.compile_unit(base, TOOLS.ROOT / "src" / unit, output / "renderer-audit.o", core)
    obj = output / "compositor-verify.o"
    binary = output / "compositor-verify"
    TOOLS.compile_unit(base, HERE / "verify.mm", obj, core, ["-UNDEBUG"])
    TOOLS.run(TOOLS.CORE.link_command(core, [obj], binary), core)
    TOOLS.run([binary], core)


if __name__ == "__main__":
    main()
