#!/bin/zsh
set -eu
here="${0:a:h}"
source_root="${here:h:h:h}"
output="${1:?provide a writable output directory}"
mkdir -p "$output"
output="${output:a}"
sdk="$(xcrun --sdk macosx --show-sdk-path)"

set -x
xcrun --sdk macosx clang++ -arch arm64 -isysroot "$sdk" -mmacosx-version-min=26.0 \
    -std=c++20 -fobjc-arc -I "$source_root/src" -Wall -Wextra -Werror \
    -MD -MF "$output/mtl-common.d" -c "$source_root/src/video_core/renderer_metal/mtl_common.mm" \
    -o "$output/mtl-common.o"
xcrun --sdk macosx clang++ -arch arm64 -isysroot "$sdk" -mmacosx-version-min=26.0 \
    -std=c++20 -fobjc-arc -I "$source_root/src" -Wall -Wextra -Werror \
    "$here/verify.mm" "$output/mtl-common.o" -framework Foundation -framework Metal \
    -o "$output/metal-context-tests"
env MTL_DEBUG_LAYER=1 "$output/metal-context-tests"
