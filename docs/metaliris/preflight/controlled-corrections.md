# Gate 0 controlled correction record

Revision 11 assigns 91 unique production files. This record preserves their order and records discrepancies before any phase opens. No existing owner has been reopened and no downstream evidence has been invalidated because no production phase has started.

## G0-OWN-001 · Generated settings key has no owner

**Evidence.** Phase 004 owns `src/common/settings.h` and requires a MetalFX Spatial setting. Phase 085 owns Qt persistence in `src/citra_qt/configuration/config.cpp`. That code reads and writes typed settings, and `src/common/settings.h:16` imports the generated key declarations. `src/CMakeLists.txt:4` includes `CMakeModules/GenerateSettingKeys.cmake`. The generator documents itself as the source of settings keys at lines 1–3 and generates `common/setting_keys.h` at line 355. The generated header and existing source template take their key definitions from this generator. There is no existing Spatial key and no owner for the generator in Revision 11.

**Controlled correction pending.** Reconcile the generator as a distinct single-file owner before Phase 004 depends on it. Record its exact ordered position and keep all 91 existing owners in relative order. Acceptance requires a real generated header in each relevant configuration, compilation of the settings consumers, and Qt persistence tests for the default and a stored override. A header template edit or literal duplicated in Qt would bypass the current key-generation contract. There is no implementation or ownership change in this record.

## G0-ARCH-001 · Backend enum can affect excluded targets

**Evidence.** `src/common/settings.h:22` declares the shared graphics enum. Several libretro switches enumerate current values without a default, including `src/citra_libretro/input/mouse_tracker.cpp:74` and `src/citra_libretro/emu_window/libretro_window.cpp:74`. `src/CMakeLists.txt:132` applies warnings-as-errors on relevant builds. The Android renderer switch already has a default branch. Revision 11 excludes non-Metal platform work and requires existing builds to remain valid.

**Scope review pending.** Decide and compile-check the build-scoped type and dispatch boundaries before declaring any additional ownership necessary. Preserve the existing stored numeric values and compile the relevant no-Metal consumers. No libretro or Android source change has been assigned.

## G0-SPEC-001 · Spatial bypass versus release acceptance

**Evidence.** Revision 11 pages 1 and 3 treat missing essential native MetalFX Spatial as a blocker for the affected configuration. Phases 074 and 075, pages 81–84, also require an explicit unsupported-capability bypass. SDK declarations expose a native Metal 4 capability probe and native scaler encoding overload.

**Controlled interpretation pending.** Keep the overall release requirement and the runtime bypass contract explicit in the same acceptance record. An unsupported configuration cannot count toward native Spatial acceptance. Resolve the expected behavior of the affected configuration in the specification before using the bypass in code. Device support is still unmeasured here.

## G0-ARCH-002 · Existing shader target includes both renderer families

**Evidence.** `src/video_core/CMakeLists.txt:71–76` lists GLSL generators in the common static library. Its Vulkan branch at line 205 adds Vulkan headers, VMA, Sirit, SPIR-V and glslang linkage. `src/video_core/host_shaders/CMakeLists.txt` generates GLSL source headers for both existing backends. The current Vulkan definition is set at directory scope in `src/CMakeLists.txt:180–182`.

**Controlled verification pending.** Phases 001–003 and 090 must establish the native execution link and symbol closure. Compile/link an independent Vulkan baseline and check the eventual native target for prohibited dependencies. Additional ownership is conditional on the measured target closure and the design of the build gate.

## G0-ARCH-003 · Qt device-info declaration may need an owner

**Evidence.** Phase 082 owns only `src/citra_qt/util/graphics_device_info.cpp`, which implements backend-specific frontend device queries. `src/citra_qt/util/graphics_device_info.h:12–17` exposes the current queries, and `src/citra_qt/citra_qt.cpp:551–555` calls them. The header has no production owner in Revision 11.

**Scope review pending.** Identify the existing adapter interface that can serve the Qt consumer before confirming whether Phase 082 is a no-change audit or whether this header needs distinct ownership. Do not add a shared declaration to the implementation file to avoid owner reconciliation.

## G0-BUILD-001 · Vulkan desktop link selected a static MoltenVK XCFramework

**Evidence.** The Vulkan/Qt configure selected the installed `/usr/local/lib/MoltenVK.xcframework`. The generated final link line used the arm64 archive slice. Linking the desktop target failed on QuartzCore symbols including `CAMetalLayer` and `CAEDRMetadata`. The same configuration had already built the static Azahar `video_core` target.

**Preflight-only correction.** Reconfigure the snapshot to select the installed Homebrew MoltenVK dylib explicitly and pass the SDK's QuartzCore framework to the executable linker. This correction is confined to command-line configuration of the reference build. The stable macOS build and packaging contract must be audited when the designated build owners open.

The revised link exited zero and the linked executable returned its version in the offscreen Qt configuration. The first startup exposed a QtBase 6.11.2 and QtMultimedia 6.11.1 private-symbol mismatch. Selecting the installed QtBase 6.11.1 framework and plugin directories at runtime resolved that startup failure. These are local reference-build adjustments. The Vulkan GPU oracle remains blocked.

- [Original link diagnostics](evidence/vulkan-frontend-link-attempt.log)
- [Reference link and Qt startup evidence](evidence/vulkan-desktop-matched-qt.log)
- [Vulkan driver evidence](evidence/vulkan-runtime.log)
