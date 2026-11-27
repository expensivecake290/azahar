# Phase 012

The native platform layer enumerates Metal devices, prioritizes the system
default, exposes stable registry identities, and distinguishes unavailable,
missing Metal 4, and missing unified-memory states. Mesh and native MetalFX
Spatial are queried from the real SDK. The renderer can retain a platform
surface along with its layout and generation without involving Qt.

The ARC fixture verified creation and release of a real CAMetalLayer,
nil-device behavior, headless operation, suspension, and rejection of an
aliased or foreign opaque handle. The layer release check runs after the
Core Animation transaction flush, because the framework can retain an
otherwise abandoned layer until that transaction is processed. The process
could not enumerate a Metal device, so the live capability matrix remains
pending. The fixture commands and trigger are in the phase ledgers.
