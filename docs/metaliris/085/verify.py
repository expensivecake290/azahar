import argparse
import importlib.util
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("phase081", HERE.parent / "081" / "verify.py")
FRONTEND = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FRONTEND)
TOOLS = FRONTEND.TOOLS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("qt", type=Path)
    parser.add_argument("core", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    qt, core, output = [p.resolve() for p in (args.qt, args.core, args.output)]
    output.mkdir(parents=True, exist_ok=True)
    name = "citra_qt/configuration/config.cpp"
    unit = TOOLS.ROOT / "src" / name
    base = TOOLS.compile_flags(qt, name)
    production = output / "native-owner.o"
    TOOLS.compile_unit(base, unit, output / "off.o", qt, ["-UENABLE_METAL"])
    TOOLS.compile_unit(base, unit, production, qt)
    fixture = output / "fixture.o"
    TOOLS.compile_unit(base, HERE / "verify.cpp", fixture, qt, ["-UNDEBUG"])
    binary = output / "settings-roundtrip"
    TOOLS.run(FRONTEND.link_command(qt, binary, [fixture, production]), qt)
    TOOLS.run([binary, "-platform", "offscreen"], qt)
    TOOLS.run([core / "bin/Release/tests", "[video_core]", "--rng-seed", "225818237"], core)


if __name__ == "__main__":
    main()
