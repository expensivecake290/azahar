# Phase 002

**Owner** `src/video_core/CMakeLists.txt`  
**Baseline** `c621eafbf316699c55f4c9c0c689d4e49498d360`

The three ledgers precede the production change. The native configuration
compiles video_core Objective-C++ sources as C++20 under ARC and takes the
three SDK frameworks from Phase 001. In the same configuration it excludes
the GLSL generators. It retains backend-neutral shader and PICA behavior,
the software blitter, dependencies, IDE grouping, and the host shader target.
Non-Metal configurations keep their GLSL sources and frontend choices.

The test fixture attaches a native Metal 4 Objective-C++ source to the real
video_core target in a disposable copy, then compiles that object and inspects
the resulting dependency and target link graph. The fixture is test-only.
The real PICA executable separately built and linked, passed 48 test cases
with 801 assertions, and has no Vulkan runtime dependency.

Version 1 is the build interface for later Metal owners. Reopen this phase
after each actual native source is complete, register the path explicitly, and
repeat the compiler and target link checks. Final renderer link evidence
depends on the completed later renderer phases. Phase 003 owns asset
packaging and can use the existing host_shaders order-only dependency.
