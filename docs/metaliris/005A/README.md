# Phase 005A

The sole persisted name is `use_metalfx_spatial`. The generator emits the
common and optional Qt declarations only for the native build configuration.
The old key definitions and ordering are preserved. The Android and LibRetro
key outputs are unchanged. No generated file was edited in the checkout.

Generation was tested against the actual baseline templates across six
variants, including repeated generation and a Qt C++ consumer. Phases 004
and 005 reopened after the real key became available. Their native and
non-Metal fixtures passed the selection and setting checks. The Metal
enum is 3, the legacy enum values remain 0–2, and Spatial is disabled
by default. Builds without a renderer integration use the existing
application dispatch until Phases 090 and 091 complete.

Phase 085 owns the remaining real Qt configuration round-trip tests.
