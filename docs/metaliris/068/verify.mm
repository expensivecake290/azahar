#include "core/core.h"
#include "core/frontend/emu_window.h"
#include "core/memory.h"
#include "video_core/custom_textures/custom_tex_manager.h"
#include "video_core/pica/pica_core.h"
#include "video_core/renderer_base.h"
#include "video_core/renderer_metal/mtl_pipeline_cache.h"
#include "video_core/renderer_metal/mtl_rasterizer.h"
#ifndef METALIRIS_USE_PRODUCTION_CACHE
#include "video_core/renderer_metal/mtl_rasterizer.mm"
#endif
#include "video_core/utils.h"
#include <cassert>
#include <cstdio>

namespace {

#ifndef METALIRIS_USE_PRODUCTION_CACHE
void VerifyHostState() {
  using Field = Pica::PipelineRegs::VertexAttributeFormat;
  auto bytes = std::make_shared<BufferMem>(40);
  const s16 negative = -3310;
  const f32 fraction = 0.125f;
  const std::array<u8, 3> components{0x81, 0xff, 0x12};
  std::memcpy(bytes->GetPtr() + 1, components.data(), components.size());
  std::memcpy(bytes->GetPtr() + 5, &negative, sizeof(negative));
  std::memcpy(bytes->GetPtr() + 9, &fraction, sizeof(fraction));
  const MemoryRef base{bytes};
  Metal::GuestVertices input;
  input.attributes[0] = {base + 1, 16, 1, 3, 7, Field::BYTE};
  input.attributes[1] = {base + 2, 16, 1, 1, 5, Field::UBYTE};
  input.attributes[2] = {base + 5, 16, 2, 1, 3, Field::SHORT};
  input.attributes[3] = {base + 9, 16, 4, 1, 1, Field::FLOAT};
  input.attributes[4] = {{}, 0, 0, 4, 12, Field::FLOAT};
  input.defaults[4] = {42.f, -42.f, 5.f, 9.f};
  const auto packed = Metal::LoadVertex(input, 0);
  assert((packed[7] == Metal::VertexVector{-127, -1, 18, 1}));
  assert((packed[5] == Metal::VertexVector{255, 0, 0, 1}));
  assert((packed[3] == Metal::VertexVector{-3310, 0, 0, 1}));
  assert((packed[1] == Metal::VertexVector{.125, 0, 0, 1}));
  assert(packed[12] == input.defaults[4]);
  assert(packed[9] == Metal::DefaultAttribute);

  using F = Pica::FramebufferRegs;
  Pica::RegsInternal regs{};
  regs.pipeline.triangle_topology.Assign(
      Pica::PipelineRegs::TriangleTopology::Strip);
  regs.framebuffer.framebuffer.flip.Assign(1);
  regs.framebuffer.framebuffer.allow_color_write.Assign(15);
  regs.framebuffer.framebuffer.depth_format.Assign(F::DepthFormat::D24S8);
  auto &output = regs.framebuffer.output_merger;
  output.alphablend_enable.Assign(1);
  output.alpha_blending.blend_equation_rgb.Assign(F::BlendEquation::Max);
  output.alpha_blending.factor_source_rgb.Assign(F::BlendFactor::SourceAlpha);
  output.alpha_blending.factor_dest_rgb.Assign(
      F::BlendFactor::OneMinusSourceAlpha);
  output.depth_color_mask = 0xb00;
  output.depth_write_enable.Assign(1);
  output.stencil_test.enable.Assign(1);
  output.stencil_test.input_mask.Assign(0x5a);
  output.stencil_test.write_mask.Assign(0x3a);
  output.stencil_test.action_depth_pass.Assign(F::StencilAction::IncrementWrap);
  const auto masked = Metal::DrawState(regs, true, true);
  assert(masked.raster.guest_topology ==
             Pica::PipelineRegs::TriangleTopology::Strip &&
         masked.raster.index_type == Metal::PipelineIndexType::UInt32 &&
         !masked.raster.flip_viewport);
  assert(
      masked.blend.ColorWriteMask() ==
      (MTLColorWriteMaskRed | MTLColorWriteMaskGreen | MTLColorWriteMaskAlpha));
  assert(masked.blend.rgb.equation == F::BlendEquation::Max);
  assert(masked.depth_stencil.depth_test && !masked.depth_stencil.depth_write &&
         masked.depth_stencil.depth_compare == F::CompareFunc::Always);
  assert(masked.depth_stencil.stencil_test &&
         masked.depth_stencil.stencil_read_mask == 0x5a &&
         masked.depth_stencil.stencil_write_mask == 0);
  regs.framebuffer.framebuffer.allow_depth_stencil_write.Assign(3);
  const auto write = Metal::DrawState(regs, false, false);
  assert(write.depth_stencil.depth_write &&
         write.depth_stencil.stencil_write_mask == 0x3a &&
         write.depth_stencil.stencil_pass == F::StencilAction::IncrementWrap);
  assert(write.raster.guest_topology ==
         Pica::PipelineRegs::TriangleTopology::List);
  std::puts("PASS packed and fixed vertex inputs, exact PICA "
            "blend/mask/stencil state, "
            "depth write gating and topology selection");
}

#endif

class HeadlessWindow final : public Frontend::EmuWindow {
public:
  void PollEvents() override {}
};

class HeadlessRenderer final : public VideoCore::RendererBase {
public:
  using RendererBase::RendererBase;

  VideoCore::RasterizerInterface *Rasterizer() override { return rasterizer; }

  void SwapBuffers() override {}
  void TryPresent(int, bool) override {}
  VideoCore::RasterizerInterface *rasterizer{};
};

Pica::OutputVertex Vertex(f32 x, f32 y) {
  const auto f24 = [](f32 n) { return Pica::f24::FromFloat32(n); };
  Pica::OutputVertex vertex{};
  vertex.pos = {f24(x), f24(y), f24(-0.5f), f24(1)};
  vertex.color = {f24(1), f24(0), f24(0), f24(1)};
  vertex.quat.w = f24(1);
  return vertex;
}

#ifdef METALIRIS_USE_PRODUCTION_CACHE
void VerifyCacheRanges(Metal::RasterizerMetal& raster, Memory::MemorySystem& memory) {
  constexpr PAddr address = Memory::VRAM_PADDR + 0x6000;
  auto* raw = memory.GetPhysicalPointer(address);
  std::fill_n(raw, 48, 0xa5);
  Pica::MemoryFillConfig fill{};
  fill.address_start = (address + 8) / 8;
  fill.address_end = (address + 40) / 8;
  fill.fill_24bit.Assign(1);
  fill.value_32bit = 0x1e66a2;
  assert(raster.AccelerateFill(fill));
  raster.FlushAndInvalidateRegion(address + 8, 32);
  constexpr std::array<u8, 3> pattern{0xa2, 0x66, 0x1e};
  for (u32 i = 0; i < 48; ++i) {
    assert(raw[i] == (i < 8 || i >= 40 ? 0xa5 : pattern[(i - 8) % 3]));
  }
  Pica::DisplayTransferConfig copy{};
  copy.texture_copy.size = 15;
  assert(!raster.AccelerateTextureCopy(copy));
  std::puts("PASS shared native cache dirty fill, exact guard bytes and short transfer fallback");
}
#endif

void VerifyDevice(bool async) {
  auto &system = Core::System::GetInstance();
  Memory::MemorySystem memory(system);
  Pica::PicaCore pica(memory, nullptr);
  VideoCore::CustomTexManager custom(system);
  HeadlessWindow window;
  HeadlessRenderer renderer(system, window, nullptr);
  Metal::Instance instance(0, false);
  Metal::Scheduler scheduler(instance.GetContext());
  Metal::MemoryBudget budget(128 * 1024 * 1024);
  Metal::MemoryAllocator allocator(instance.GetDevice(), budget);
  Metal::RenderManager manager(instance, scheduler, allocator);
  Metal::DescriptorUpdateQueue queue(instance.GetContext());
  Metal::RasterizerMetal raster(memory, pica, custom, renderer, instance,
                                scheduler, manager, queue, allocator, budget,
                                {.async_shaders = async});
  pica.BindRasterizer(&raster);
  renderer.rasterizer = &raster;

  constexpr PAddr color_address = Memory::VRAM_PADDR + 0x4000;
  using F = Pica::FramebufferRegs;
  auto &regs = pica.regs.internal;
  auto &fb = regs.framebuffer.framebuffer;
  fb.color_buffer_address.Assign(color_address / 8);
  fb.color_format.Assign(F::ColorFormat::RGBA8);
  fb.allow_color_write.Assign(15);
  fb.width.Assign(8);
  fb.height.Assign(7);
  fb.flip.Assign(1);
  regs.framebuffer.output_merger.logic_op.Assign(F::LogicOp::Copy);
  regs.framebuffer.output_merger.depth_color_mask = 0xf00;
  regs.lighting.disable.Assign(1);
  regs.rasterizer.viewport_size_x.Assign(0x410000);
  regs.rasterizer.viewport_size_y.Assign(0x410000);
  regs.rasterizer.viewport_depth_range.Assign(0x3f0000);
  auto &scissor = regs.rasterizer.scissor_test;
  scissor.mode.Assign(Pica::RasterizerRegs::ScissorMode::Include);
  scissor.x1.Assign(2);
  scissor.x2.Assign(5);
  scissor.y1.Assign(2);
  scissor.y2.Assign(5);
  pica.dirty_regs.SetAllDirty();

  auto *raw = memory.GetPhysicalPointer(color_address);
  constexpr std::array<u8, 4> background{0xff, 0x80, 0x40, 0x20};
  constexpr std::array<u8, 4> covered{0xff, 0, 0, 0xff};
  for (u32 i = 0; i < 8 * 8; ++i) {
    std::memcpy(raw + 4 * i, background.data(), 4);
  }
  raster.AddTriangle(Vertex(-1, 1), Vertex(3, 1), Vertex(-1, -3));
  raster.DrawTriangles();
  raster.FlushRegion(color_address, 8 * 8 * 4);
  for (u32 y = 0; y < 8; ++y) {
    for (u32 x = 0; x < 8; ++x) {
      const auto expected =
          x >= 2 && x <= 5 && y >= 2 && y <= 5 ? covered : background;
      const auto offset = VideoCore::GetMortonOffset(x, y, 4);
      assert(std::memcmp(raw + offset, expected.data(), 4) == 0);
    }
  }
#ifdef METALIRIS_USE_PRODUCTION_CACHE
  VerifyCacheRanges(raster, memory);
#endif
  raster.ClearAll(false);
  assert(scheduler.Finish() == Metal::CompletionStatus::Complete);
  assert(!scheduler.GetFault());
  std::printf("PASS physical PICA color and scissor guest readback with %s "
              "compilation\n",
              async ? "asynchronous" : "synchronous");
}

} // Anonymous namespace

int main() {
  @autoreleasepool {
#ifndef METALIRIS_USE_PRODUCTION_CACHE
    VerifyHostState();
#endif
    auto device = MTLCreateSystemDefaultDevice();
    if (!device || ![device supportsFamily:MTLGPUFamilyMetal4]) {
      std::puts("PENDING actual Metal 4 rasterizer draw, submission and guest "
                "readback");
      return 2;
    }
    VerifyDevice(false);
    VerifyDevice(true);
  }
}
