#!/bin/zsh
set -eu
here="${0:a:h}"
source_root="${here:h:h:h}"
build_dir="${1:?provide the native CMake build directory}"
output="${2:?provide a writable output directory}"
mkdir -p "$output"
sdk="$(xcrun --sdk macosx --show-sdk-path)"
flags=(-arch arm64 -isysroot "$sdk" -mmacosx-version-min=26.0 -std=c++20
       -DENABLE_SOFTWARE_RENDERER -I "$source_root/src" -I "$build_dir/src" -isystem "$source_root/externals/boost" -isystem "$source_root/externals/fmt/include" -isystem "$source_root/externals/xxHash" -isystem "$source_root/externals/nihstro/include" -Wall -Wextra -Wno-ignored-qualifiers -Werror -Wno-unused-parameter)
set -x
xcrun --sdk macosx clang++ "${flags[@]}" -MD -MF "$output/consumer.d" \
    -c "$here/verify.cpp" -o "$output/consumer.o"
xcrun --sdk macosx clang++ -arch arm64 -mmacosx-version-min=26.0 "$output/consumer.o"  -o "$output/surface-adapter-contract"
"$output/surface-adapter-contract"
xcrun --sdk macosx clang++ "${flags[@]}" -x objective-c++ -fobjc-arc -MD -MF "$output/bridge.d" \
    -c "$here/verify.cpp" -o "$output/bridge.o"
if rg -i 'renderer_vulkan|moltenvk|vulkan\.h|shaderc|spirv' "$output/consumer.d" "$output/bridge.d"; then
    exit 1
fi
