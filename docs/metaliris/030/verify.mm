#include "video_core/renderer_metal/shader/msl_shader_decompiler.h"
#include <bit>
#include <cassert>
#include <cstdio>

using namespace Pica::Shader::Generator::MSL;

#if defined(CHECK_NATIVE_CONSUMER)
std::string BuildLibrary(const Pica::ShaderSetup &setup, u32 entry) {
  const auto geometry =
      DecompileProgram(setup.GetProgramCode(), setup.GetSwizzleData(), entry,
                       {ProgramKind::Geometry, "TestGeometry"});
  assert(geometry);
  return std::string(DecompilerPreamble()) + geometry.source;
}
#else
int main() {
  constexpr RegisterState state{};
  static_assert(state.address[2] == 0 && state.condition[1] == 0 &&
                state.emission[0] == 0);
  static_assert(state.output[0][0][3] == 0.0f &&
                state.temporary[15][3] == 0.0f);
  static_assert(sizeof(Pica::Shader::Generator::VSPicaUniformData) == 1616);
  constexpr auto to_bits = [](u32 raw) {
    return std::bit_cast<u32>(Pica::f24::FromRaw(raw).ToFloat32());
  };
  static_assert(to_bits(0x800000u) == 0x80000000u);
  static_assert(to_bits(0x3F0000u) == 0x3F800000u);
  static_assert(to_bits(0x7F0000u) == 0x7F800000u);
  static_assert(to_bits(0x7F8000u) == 0x7FC00000u);
  static_assert(to_bits(0x000001u) == 0x18800000u);
  assert(!DecompiledProgram{});
  std::puts("PASS native decompiler consumer, register/uniform layout and f24 "
            "conversion boundaries");
}
#endif
