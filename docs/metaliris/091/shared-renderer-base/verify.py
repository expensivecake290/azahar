import argparse
import importlib.util
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("phase082", HERE.parents[1] / "082" / "verify.py")
TOOLS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TOOLS)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("core", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    core, output = args.core.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    base = TOOLS.compile_flags(core, "video_core/renderer_base.cpp")
    obj = output / "request-fixture.o"
    native = output / "screenshot-owner.o"
    unit = TOOLS.ROOT / "src/video_core/renderer_base.cpp"
    TOOLS.compile_unit(base, unit, output / "feature-off.o", core, ["-UENABLE_METAL"])
    TOOLS.compile_unit(base, unit, native, core)
    TOOLS.compile_unit(base, HERE / "verify.cpp", obj, core, ["-UNDEBUG"])
    binary = output / "screenshot-request"
    TOOLS.run(TOOLS.CORE.link_command(core, [obj, native], binary), core)
    TOOLS.run([binary, output], core)


if __name__ == "__main__":
    main()
