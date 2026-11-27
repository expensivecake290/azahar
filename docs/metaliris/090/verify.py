import argparse
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("phase082", HERE.parent / "082" / "verify.py")
TOOLS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TOOLS)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("native", type=Path)
    parser.add_argument("off", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    native, off, output = [p.resolve() for p in (args.native, args.off, args.output)]
    output.mkdir(parents=True, exist_ok=True)
    unit = "video_core/video_core.cpp"
    for mode, build in [("native", native), ("off", off)]:
        entries = json.loads((build / "compile_commands.json").read_text())
        entry, = [e for e in entries if e["file"].endswith("/" + unit)]
        base = shlex.split(entry["command"])
        assert ("-DENABLE_METAL" in base) == (mode == "native")
        assert not {"-DENABLE_VULKAN", "-DENABLE_OPENGL"}.intersection(base)
        base.insert(1, f"-I{TOOLS.ROOT / 'src'}")
        obj = output / (mode + ".o")
        TOOLS.compile_unit(base, TOOLS.ROOT / "src" / unit, obj, build)
        symbols = subprocess.check_output(["nm", "-u", str(obj)], text=True)
        print(symbols)
        assert ("RendererMetal" in symbols) == (mode == "native")
        assert "RendererSoftware" in symbols and "RendererVulkan" not in symbols and "OpenGL" not in symbols
        assert "SDL_Init" not in symbols
    print("PASS actual native and Software-only dispatcher objects and symbol isolation")
    TOOLS.run([native / "bin/Release/tests", "[video_core]", "--rng-seed", "225818237"], native)


if __name__ == "__main__":
    main()
