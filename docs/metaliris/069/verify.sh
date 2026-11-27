#!/bin/zsh
set -eu
here="${0:a:h}"
source_root="${here:h:h:h}"
build_dir="${1:?provide native CMake build directory}"
output="${2:?provide a writable output directory}"
mkdir -p "$output"
output="${output:a}"

python3 - "$here" "$source_root" "$build_dir" "$output" <<'PY'
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess
import sys

sys.dont_write_bytecode = True
here, source, build, output = map(lambda arg: Path(arg).resolve(), sys.argv[1:])
database = json.loads((build / "compile_commands.json").read_text())
entry = next(e for e in database if e["file"].endswith("/mtl_rasterizer_cache.mm"))
command = shlex.split(entry["command"])
command.insert(1, f"-I{source / 'src'}")
assert "-fobjc-arc" in command

def run(command):
    print("+", shlex.join(command), flush=True)
    subprocess.run(command, cwd=build, check=True)

production = Path(entry["file"]).read_bytes()
owner = source / "src/video_core/renderer_metal/mtl_rasterizer_cache.mm"
assert production == owner.read_bytes()
fixture_object = output / "cache-test.o"
compile_check = command.copy()
compile_check[compile_check.index("-c") + 1] = str(owner)
compile_check[compile_check.index("-o") + 1] = str(output / "cache.o")
run(compile_check + ["-MD", "-MF", str(output / "cache.d")])
command[command.index("-c") + 1] = str(here.parent / "068/verify.mm")
command[command.index("-o") + 1] = str(fixture_object)
run(command + ["-UNDEBUG", "-DMETALIRIS_USE_PRODUCTION_CACHE", "-MD", "-MF",
               str(output / "consumer.d")])

native = build / "src/video_core/libvideo_core.a"
symbols = subprocess.check_output(["xcrun", "nm", "-gUj", str(output / "cache.o")],
                                  cwd=build, text=True)
wanted = ("AccelerateTextureCopy", "AccelerateDisplayTransfer", "AccelerateFill",
          "GetSurfaceSubRect", "GetFramebufferSurfaces", "GetTextureSurface", "TickFrame",
          "InvalidateRegion", "FlushRegion", "FlushAll", "ClearAll", "GetTextureCube")
for method in wanted:
    assert any("RasterizerCacheIN5Metal6Traits" in line and method in line
               for line in symbols.splitlines()), method
members = subprocess.check_output(["ar", "-t", str(native)], text=True)
assert members.splitlines().count("mtl_rasterizer_cache.mm.o") == 1
assert all("vk_" not in entry for entry in members.splitlines())

spec = importlib.util.spec_from_file_location("phase007", here.parent / "007/verify.py")
phase007 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(phase007)
link = phase007.link_command(build, [fixture_object], output / "metal-rasterizer-cache-tests")
run(link)
print("PASS 12 native shared cache entry points, single CMake archive owner and real consumer link")
PY
if rg -i 'renderer_vulkan|moltenvk|vulkan\.h|shaderc|spirv' "$output"/*.d; then
    exit 1
fi
MTL_DEBUG_LAYER=1 "$output/metal-rasterizer-cache-tests"
