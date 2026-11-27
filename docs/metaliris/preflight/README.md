# Metaliris Gate 0

**Status: blocked.** Production phase entry is closed. The scope of this work is Revision 11's preflight. All 91 production phases are unopened.

## Checkout and toolchain

| Check | Observed result |
| --- | --- |
| Source revision | `c621eafbf316699c55f4c9c0c689d4e49498d360` on `mtl-4-backend` |
| Repository inventory | 2,359 top-level Git index entries and 21,026 tracked paths with initialized submodules. Full SHA-256 is recorded for each tracked file or symlink target. The pre-existing deletion of `AI-POLICY.md` is separately recorded. |
| Revision 11 audit | 91 unique ordered owners. All 75 distinct printed source fingerprints match the checkout, including all 42 Vulkan files. The PDF records a different total for its archive. Its file count and the checkout's Git index count measure different inventories. |
| Specification input | All 103 pages were read from the attachment. The byte download failed DNS, so the attachment's binary was not independently hashed. |
| Host | Apple M1 MacBook Pro (`MacBookPro17,1`), macOS 27.2 build `26B5086k` |
| Tools | Xcode 27.0 build `27A5194q`, macOS SDK 27.0 build `26A5353p`, Apple Clang 21.0.0, Apple Metal compiler 32023.917, CMake 4.4.2, Ninja 1.13.2 |
| Licenses | Repository root carries GPLv3. Inspected files carry individual GPLv2-or-later or GPLv3-or-later notices. Preserve the relevant file notices and audit any added dependency before use. HEAD's project AI policy governs upstream submissions. This preflight concerns the user's Metaliris fork and includes no submission. |

See [baseline](evidence/baseline.json), [source fingerprints](evidence/revision11-fingerprints.json), [toolchain](evidence/toolchain.log) and [ownership](evidence/ownership.json).

## Verified checks and remaining work

| Requirement | Result |
| --- | --- |
| SDK and host compiler | The isolated ARC Objective-C++ test links Foundation, Metal and MetalFX against the installed SDK with warnings treated as errors. Native declarations for Metal 4 queue, allocator, command buffer, compiler, argument table, residency, barriers, counters, object/mesh and Spatial are captured. |
| MSL compiler | Metal 4.0 compiled compute, object, mesh and fragment functions into AIR and linked a macOS metallib. The compiler needed a writable modules cache. |
| GPU workload and API validation | The diagnostic layer reported enabled. Device enumeration returned zero and the default device was nil, with and without validation enabled. The test could not create GPU objects or check outputs. |
| GPU debugging | Instruments could not create its cache directory. Programmatic capture in the native harness could not start because no Metal device was available. |
| Vulkan reference | The ARM64 Vulkan configuration generated all 26 baseline host shader headers and built the Vulkan-enabled `video_core` library. The desktop executable linked with the Homebrew MoltenVK dylib and ran `--version` with matched Qt frameworks and plugins. The installed MoltenVK driver returned `VK_ERROR_INCOMPATIBLE_DRIVER`. No rendered reference output was captured. |
| PICA200 oracle | The Vulkan-disabled tests built and ran independently. The `[video_core]` filter passed 48 cases and 801 assertions, including interpreter cases and number-format checks. The full CTest run produced 64 passes, 5 existing audio skips and 0 failures. Physical reference fixtures and broader guest rendering oracles are pending. |
| Build and dependency audit | CMake File API recorded 81 targets for Vulkan/Qt and 44 for PICA. Compiler-resolved include traces cover all 486 and 388 configured source-tree translation units respectively, with zero final scan failures. Each scan uses one compiler command per source path. The provisional phase graph contains 162 edges, nine to later phases, and no cycles. Full project interpretation of include classifications and the native dependency plan remain pending. |
| Exclusive ownership | The generator for the persisted Spatial setting has no owner. See the controlled correction record for the missing owner and other unresolved integration decisions. |

The device result describes the managed session. It does not establish that the M1 hardware lacks Metal 4. No GPU capabilities were accepted based on family names or SDK declarations alone.

## Reproduction and evidence

Run the following from the checkout on a GPU-visible macOS session. Use a new writable output directory for the GPU capture.

```sh
zsh docs/metaliris/preflight/harness/run-native.sh /tmp/metaliris-native-gate0
VK_DRIVER_FILES=/opt/homebrew/etc/vulkan/icd.d/MoltenVK_icd.json vulkaninfo --summary
```

The harness checks a GPU-computed sentinel, a mesh/object triangle pixel when the GPU supports the capability, and a native Spatial output pixel. It records GPU completion feedback and requests an Xcode GPU trace. It builds only its own test files.

The reference builds can be recreated from the checkout with these commands.

```sh
python3 docs/metaliris/preflight/harness/snapshot-reference.py /tmp/metaliris-baseline-source
zsh docs/metaliris/preflight/harness/configure-reference.sh pica /tmp/metaliris-baseline-source /tmp/metaliris-pica-build
cmake --build /tmp/metaliris-pica-build --target tests
ctest --test-dir /tmp/metaliris-pica-build --output-on-failure
zsh docs/metaliris/preflight/harness/configure-reference.sh vulkan /tmp/metaliris-baseline-source /tmp/metaliris-vulkan-build
cmake --build /tmp/metaliris-vulkan-build --target citra_meta
```

Reference configurations used an isolated copy of the tracked working tree and initialized submodule contents. The copy was configured as an archive tree after the direct configuration tried to write a Git hook into the read-only Git metadata. Installed Qt, SDL and OpenSSL were selected. The builds used Release, ARM64, no PCH or LTO, tests enabled, and FFmpeg disabled. The configure logs record the exact flags, and the link warnings identify installed dependencies built for newer macOS targets. These results do not verify compatibility with the older Azahar deployment floor.

- [Native compile and device probe](evidence/native-harness.log) and [SDK declarations](evidence/sdk-declarations.log)
- [Vulkan configuration](evidence/vulkan-configure-final.log), [core build](evidence/vulkan-build.log), [desktop link and startup](evidence/vulkan-desktop-matched-qt.log), and [runtime failure](evidence/vulkan-runtime.log)
- [PICA configuration](evidence/pica-configure-final.log), [build](evidence/pica-build.log), and [test results](evidence/pica-tests.log)
- [Dependency summary](evidence/dependency-summary.json) and [controlled corrections](controlled-corrections.md)
- [Progress state](../progress.json)

Gate 0 requires actual GPU-backed Metal 4 and MetalFX results, GPU debugging evidence, independent Vulkan reference outputs, ownership reconciliation and completion of the remaining repository and dependency audit. The runtime tests cannot advance further in this managed session.
