# Phase 001

**Owner** `CMakeLists.txt`  
**Baseline** `c621eafbf316699c55f4c9c0c689d4e49498d360`  
**Interface** `ENABLE_METAL` and `METAL_LIBRARIES`, version 1

The source matches the 664-line fingerprint in Corrected Revision 12. The three
JSON ledgers were recorded before the owner file was changed. Metal is opt-in.
The native target is macOS 26.0 or newer on arm64, with an actual ARC
Objective-C++20 compile/link probe against Metal, MetalFX and QuartzCore. The
CMake probe is independent of the frontend and of the machine's GPU. The
dependent Vulkan option disables Vulkan in Metal configurations and preserves
the previous cache choice for non-Metal configurations.

The installed toolchain is AppleClang 21.0.0, Xcode 27.0 (27A5194q), SDK 27.0
and CMake 4.4.2. The host is arm64 macOS 27.2 (26B5086k). CMake verification
covers selection, prior graph equivalence, Qt and headless configurations,
explicit and implicit target architectures, deployment floors, unsupported
platforms and compiler failure followed by cache recovery. The SDK probe and
native fixture compiled and linked. The include registry records paths
resolved by Clang. The root CMake file is GPL-2.0-or-later project code.

Direct configuration hit an existing Git-hook copy into a read-only directory.
Tests configure writable source snapshots from the same checkout.
`verify.py` accepts a scratch directory containing `source` (the implementation
snapshot) and `baseline-source` (the baseline with its dependency sources).
Create either tree with `../preflight/harness/snapshot-reference.py` from
the repo root. Restore only the baseline root CMake file in the baseline
snapshot using `git show c621eafbf316699c55f4c9c0c689d4e49498d360:CMakeLists.txt`.
The log files show the actual configurations and commands.

The native workload enumerated zero devices. Reopen on an arm64 macOS 26+
environment with an accessible Metal 4 GPU and run
`python3 docs/metaliris/001/verify.py native <scratch-dir>`. Confirm the
compute sentinel, optional object/mesh color, MetalFX Spatial color and
readable GPU capture with Metal API validation enabled. Phase 002 consumes
the native framework list. Runtime renderer and shader consumers are owned by
their later phases.

The new PICA200 build and link passed 48 cases and 801 assertions. CTest
passed 64 cases and skipped five existing audio fixtures. The historical
Revision 11 success is retained separately.
