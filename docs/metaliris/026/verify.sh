#!/bin/zsh
set -eu
here="${0:a:h}"
source_root="${here:h:h:h}"
build_dir="${1:?provide the native CMake build directory}"
output="${2:?provide a writable output directory}"
mkdir -p "$output"
sdk="$(xcrun --sdk macosx --show-sdk-path)"
flags=(-arch arm64 -isysroot "$sdk" -mmacosx-version-min=26.0 -std=c++20 -fobjc-arc
       -DENABLE_SOFTWARE_RENDERER -I "$source_root/src" -I "$build_dir/src" -Wall -Wextra -Werror)
set -x
xcrun --sdk macosx clang++ "${flags[@]}" -DCHECK_NATIVE_CONSUMER -MD -MF "$output/consumer.d" \
    -c "$here/verify.mm" -o "$output/consumer.o"
xcrun --sdk macosx clang++ "${flags[@]}" "$here/verify.mm" -o "$output/descriptor-contract"
"$output/descriptor-contract"
if rg -i 'renderer_vulkan|moltenvk|vulkan\.h|shaderc|spirv' "$output/consumer.d"; then
    exit 1
fi
