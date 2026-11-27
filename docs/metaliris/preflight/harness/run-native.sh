#!/bin/zsh
set -eu

here="${0:a:h}"
output="${1:?provide a writable output directory}"
mkdir -p "$output"
output="${output:a}"
sdk="$(xcrun --sdk macosx --show-sdk-path)"

set -x
xcrun --sdk macosx metal -isysroot "$sdk" -mmacosx-version-min=26.0 -std=metal4.0 \
    -fmodules-cache-path="$output/metal-module-cache" -Wall -Werror \
    -MD -MF "$output/smoke-metal.d" -c "$here/smoke.metal" -o "$output/smoke.air"
xcrun --sdk macosx metal -isysroot "$sdk" -mmacosx-version-min=26.0 -std=metal4.0 \
    -fmodules-cache-path="$output/metal-module-cache" -Wall -Werror \
    "$here/smoke.metal" -o "$output/smoke.metallib"
xcrun --sdk macosx clang++ -std=c++20 -isysroot "$sdk" -arch arm64 \
    -mmacosx-version-min=26.0 -fobjc-arc -Wall -Wextra -Werror \
    -MD -MF "$output/device-smoke.d" -framework Foundation -framework Metal -framework MetalFX \
    "$here/device-smoke.mm" -o "$output/device-smoke"
env MTL_DEBUG_LAYER=1 MTL_CAPTURE_ENABLED=1 "$output/device-smoke" "$output/smoke.metallib" "$output/workload.gputrace"
