#!/bin/zsh
set -eu
here="${0:a:h}"
source_root="${here:h:h:h}"
build_dir="${1:?provide the native CMake build directory}"
output="${2:?provide a writable output directory}"
mkdir -p "$output"
sdk="$(xcrun --sdk macosx --show-sdk-path)"
flags=(-arch arm64 -isysroot "$sdk" -mmacosx-version-min=26.0 -std=c++20 -fobjc-arc
       -DENABLE_SOFTWARE_RENDERER -DENABLE_METAL -I "$source_root/src" -I "$build_dir/src" -isystem "$source_root/externals/boost" -isystem "$source_root/externals/fmt/include" -isystem "$source_root/externals/xxHash" -isystem "$source_root/externals/nihstro/include" -Wall -Wextra -Wno-ignored-qualifiers -Werror -Wno-unused-parameter)
set -x
xcrun --sdk macosx clang++ "${flags[@]}" -MD -MF "$output/consumer.d" \
    -c "$here/verify.mm" -o "$output/consumer.o"
xcrun --sdk macosx clang++ -arch arm64 -mmacosx-version-min=26.0 "$output/consumer.o" -framework Metal -framework Foundation -o "$output/rasterizer-contract"
"$output/rasterizer-contract"
xcrun --sdk macosx clang++ "${flags[@]}" -DENABLE_FREECAM -MD -MF "$output/freecam.d" \
    -c "$here/verify.mm" -o "$output/freecam.o"
if rg -i 'renderer_vulkan|moltenvk|vulkan\.h|shaderc|spirv' "$output/consumer.d" "$output/freecam.d"; then
    exit 1
fi
