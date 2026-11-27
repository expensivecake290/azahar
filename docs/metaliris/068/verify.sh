#!/bin/zsh
set -eu
here="${0:a:h}"
source_root="${here:h:h:h}"
build_dir="${1:?provide native CMake build directory}"
output="${2:?provide writable output directory}"
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
baseline = next(e for e in database if e["file"].endswith("/mtl_texture_runtime.mm"))
base_command = shlex.split(baseline["command"])
base_command.insert(1, f"-I{source / 'src'}")
assert "-fobjc-arc" in base_command

def run(command):
    print("+", shlex.join(command), flush=True)
    subprocess.run(command, cwd=build, check=True)

def compile_tu(source, stem, *flags):
    command = base_command.copy()
    command[command.index("-c") + 1] = str(source)
    destination = output / (stem + ".o")
    command[command.index("-o") + 1] = str(destination)
    run(command + ["-MD", "-MF", str(destination.with_suffix(".d")), *flags])
    return destination

production = source / "src/video_core/renderer_metal/mtl_rasterizer.mm"
compile_tu(production, "rasterizer")
compile_tu(production, "freecam", "-DENABLE_FREECAM")
verification = compile_tu(here / "verify.mm", "rasterizer-test", "-UNDEBUG")

# Instantiate the shared consumer in the fixture to check the dependency before its own
# production-file phase. The shipped specialization belongs to Phase 069.
cache = output / "cache-consumer.mm"
cache.write_text('''#include "video_core/renderer_metal/mtl_texture_runtime.h"
#include "video_core/rasterizer_cache/rasterizer_cache.h"
namespace VideoCore { template class RasterizerCache<Metal::Traits>; }
''')
cache_object = compile_tu(cache, "cache-consumer")
spec = importlib.util.spec_from_file_location("phase007", here.parent / "007/verify.py")
phase007 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(phase007)
link = phase007.link_command(build, [verification, cache_object], output / "metal-rasterizer-tests")
run(link)
print("PASS arm64 native rasterizer and FreeCam SDK compilation, shared cache consumer and link")
PY
if rg -i 'renderer_vulkan|moltenvk|vulkan\.h|shaderc|spirv' "$output"/*.d; then
    exit 1
fi
MTL_DEBUG_LAYER=1 "$output/metal-rasterizer-tests"
