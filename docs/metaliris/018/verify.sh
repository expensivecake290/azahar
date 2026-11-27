#!/bin/zsh
set -eu
here="${0:a:h}"
root="${here:h:h:h}"
build="${1:?pass native CMake build directory}"
out="${2:?pass writable scratch directory}"
mkdir -p "$out"
sdk="$(xcrun --sdk macosx --show-sdk-path)"
set -x
xcrun --sdk macosx clang++ -std=c++20 -arch arm64 -isysroot "$sdk" -mmacosx-version-min=26.0 \
  -fobjc-arc -DENABLE_METAL -Wall -Wextra -Werror -I "$root/src" -I "$build/src" \
  -isystem "$root/externals/boost" -isystem "$root/externals/fmt/include" -MD -MF "$out/timeline-header.d" "$here/verify.mm" \
  -framework Foundation -o "$out/timeline-header"
"$out/timeline-header"
