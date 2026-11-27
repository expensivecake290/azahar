#!/bin/zsh
set -eu
mode="${1:?provide vulkan or pica}"
source_dir="${2:?provide the checkout or source snapshot}"
out="${3:?provide a writable build directory}"
mkdir -p "$out/.cmake/api/v1/query"
touch "$out/.cmake/api/v1/query/codemodel-v2"
sdk="$(xcrun --sdk macosx --show-sdk-path)"
flags=(-DCMAKE_OSX_ARCHITECTURES=arm64 "-DCMAKE_OSX_SYSROOT=$sdk" -DCMAKE_BUILD_TYPE=Release
       -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DENABLE_TESTS=ON -DENABLE_LTO=OFF
       -DENABLE_FFMPEG=OFF -DUSE_SYSTEM_SDL2=ON -DUSE_SYSTEM_OPENSSL=ON
       -DCITRA_USE_PRECOMPILED_HEADERS=OFF)
case "$mode" in
    vulkan)
        flags+=(-DCMAKE_PREFIX_PATH="/opt/homebrew/opt/qtbase;/opt/homebrew/opt/qtmultimedia;/opt/homebrew"
                -DQt6_DIR=/opt/homebrew/opt/qtbase/lib/cmake/Qt6
                -DQt6Multimedia_DIR=/opt/homebrew/opt/qtmultimedia/lib/cmake/Qt6Multimedia
                -DMOLTENVK_LIBRARY=/opt/homebrew/lib/libMoltenVK.dylib
                -DCMAKE_EXE_LINKER_FLAGS="-framework QuartzCore"
                -DUSE_SYSTEM_QT=ON -DUSE_SYSTEM_MOLTENVK=ON -DENABLE_VULKAN=ON
                -DENABLE_QT=ON -DENABLE_QT_TRANSLATION=OFF -DENABLE_QT_UPDATE_CHECKER=OFF)
        ;;
    pica)
        flags+=(-DCMAKE_PREFIX_PATH=/opt/homebrew -DENABLE_QT=OFF -DENABLE_VULKAN=OFF)
        ;;
    *)
        exit 64
        ;;
esac
set -x
cmake -S "$source_dir" -B "$out" -G Ninja "${flags[@]}"
