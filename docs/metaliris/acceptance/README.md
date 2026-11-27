# Metaliris release verification

2026-11-27 · Corrected Revision 12

Release acceptance remains pending. The numbered phases and their required
outputs, dependencies, and reopening triggers are tracked in
[progress.json](../progress.json). The [fixture run index](fixture-runs.json)
records commands, exits, and explicit skips from the existing phase fixtures.
An exit of zero can include an explicitly skipped device or display test.

## Reproduced checks

| Check | Evidence |
| --- | --- |
| Native Metal 4 core and arm64 Qt application link, 19 exact compiled MSL 4 payloads, CPU compositor checks, zero-device factory rejection | [Native build and link audit](native.log) |
| Qt main-window startup, local UDP bind, graphics API status and cycling, persisted diagnostic settings, and rejection before Core startup with both render targets headless | [Offscreen Qt fixture](qt-offscreen.log) |
| Canonical disk-cache codec, interrupted writes, eight concurrent writers, and host pipeline publication controls | [Disk-cache fixture](sweep/060.log), [Pipeline fixture](sweep/058.log) |
| Shared screenshot storage and request synchronization with two concurrent callers | [Shared screenshot fixture](sweep/091-shared.log) |
| Qt lifecycle host rejection and physical display probe | [Qt surface fixture](sweep/081.log) |

The native run passed 48 PICA200 cases with 801 assertions and 64 CTest
tests. Five audio tests skipped. DSP LLE vs HLE and DSP LLE Sanity need
`<SDMC>/3ds/dspaudio.cdc`. Both SourceStatus cases and AudioTest-BiquadFilter
need `<SDMC>/3ds/dspfirm.cdc`. Supply the appropriate firmware to the test
SDMC and repeat the existing audio tests. [Audio investigation](audio-investigation.log)

Phase 087 was reopened after the Software-only Qt link failed on
`_OBJC_CLASS_$_CAMetalLayer`. Its existing Apple helper now links QuartzCore
directly when the Metaliris backend is disabled. The system QtGui target
also declares Metal as a framework dependency. The Software-only checks
verify that the Azahar build graph and executable omit the Metaliris
renderer, native device enumeration and Metal 4 command and compiler
references. The native check requires its own complete framework closure
and one ARC Qt bridge. [Original link failure](../087/reopen-release-software-link-failure.log)

## Outstanding runtime requirements

The macOS 27.2 arm64 host identified an eight-core Apple M1 GPU. The
process-visible Metal inventory was empty, the default device was nil,
and `xctrace` listed no recording devices. The Cocoa fixture exposed zero
screens. The direct Qt Cocoa startup exited on SIGABRT with invalid
HIServices connections. LaunchServices returned `kLSNoExecutableErr` for
the compiled app and the standalone smoke bundle, whose executables
were present. The earlier local UDP restriction no longer reproduced.
[Host probe](native-environment.log) · [LaunchServices app probe](launcher-qt.log)

| Outstanding gate | Required environment and trigger |
| --- | --- |
| Device capabilities, command submission, barriers, residency, argument tables, and lifetime | An arm64 macOS 26+ desktop session exposing an Apple Silicon Metal 4 device. Enable Metal API Validation, run the phase-local fixtures, and compare completion, readback, and lifetime assertions. Reopen the phase that owns a failed invariant. |
| Guest shader and geometry parity, texture and framebuffer behavior, native cache and compilation | Run the recorded command, shader, rasterizer, and cache fixtures on that device. Exercise synchronous and asynchronous work, mesh/object and compute routes, both cache states, and independent PICA200 expected results. Reopen on a wrong result, missing draw, invalid resource use, or stale cache reuse. |
| Cocoa startup, CAMetalLayer presentation, dual screens, Spatial, screenshots, and shutdown | Launch the native Qt app from a logged-in desktop with a screen, matching Qt Base and Multimedia, a Metal 4 GPU, and an available UDP bind. Use available display scale and refresh modes and a device supporting Metal 4 MetalFX Spatial for the Spatial output test. Reopen on a device or surface selection, render, pacing, or lifetime mismatch. |
| Optional Qt video dumping | Configure with FFmpeg 7 and libavutil 59.x. This run found libavutil 61.1.101 and CMake stopped at the version requirement. With the matching SDK and a valid Metal 4 title, test menu and queued startup rejection and confirm no output file is created. [Configure attempt](ffmpeg-configure.log) |

## Gameplay and measurements

A local Dream Team cartridge-image candidate was identified. Paper Jam
was absent from the searched local locations. Use legally obtained
images and repeat a recorded save and input sequence through exploration
and movement, then battle and transitions. Preserve the title revision,
save checkpoint, input sequence, OS and SDK builds, GPU model, app
revision, and configuration alongside the captures. Start with Dream
Team at 1×, then use higher internal resolution, Spatial off and on,
empty and warm caches, synchronous and asynchronous compilation, and
the dual-screen, split-window and stereoscopic modes. Repeat applicable
cases for Paper Jam and additional available PICA200 workloads. Compare
representative captures with deterministic fixtures, independently
derived output, or the software renderer.

Record average FPS and p50, p95, and p99 frame times for matched input
segments. Include CPU and GPU frame time, stutters, memory footprint and
peak, allocated and resident GPU resources, pipeline latency and cache
hits, submission overhead and synchronization stalls, pass breaks,
bandwidth, energy and thermal behavior, and the Spatial pass cost where
Instruments or GPU counters expose them. Keep pre- and post-correction
captures on the same hardware and verify output parity after any change.
Gameplay and performance measurements are pending physical execution.
Vulkan serves as the read-only source reference.
