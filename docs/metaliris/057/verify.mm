#include "video_core/renderer_metal/mtl_graphics_pipeline.h"
#include <array>
#include <cassert>
#include <cstdio>
#include <map>

using namespace Metal;

static_assert(std::is_same_v<decltype(&PipelineCompiler::WaitForIdle),
                             void (PipelineCompiler::*)()>);

int main() {
  @autoreleasepool {
    using F = Pica::FramebufferRegs;
    using P = Pica::PipelineRegs;
    using R = Pica::RasterizerRegs;
    namespace MSL = Pica::Shader::Generator::MSL;

    GraphicsPipelineKey original;
    const PipelineFunction source{ShaderSource{"MSL", "fixture", "", {}},
                                  "entry"};
    const PipelineFunction binary{std::vector<u8>{'M', 'T', 'L', 'B', 0},
                                  "entry"};
    original.shaders.vertex = source;
    original.shaders.fragment = binary;
    original.vertex_layout.bindings.push_back(
        {8, 4, 1, MTLVertexStepFunctionPerVertex});
    original.vertex_layout.attributes.push_back({0, 8, 0, MTLVertexFormatChar4,
                                                 P::VertexAttributeFormat::BYTE,
                                                 4, true});
    original.attachments = {MTLPixelFormatRGBA8Unorm,
                            MTLPixelFormatDepth32Float_Stencil8,
                            MTLPixelFormatDepth32Float_Stencil8, 1};
    std::map<GraphicsPipelineKey, int> keys{{original, 0}};
    const auto check = [&](auto mutate) {
      auto variant = original;
      mutate(variant);
      assert(variant != original);
      const int size = static_cast<int>(keys.size());
      assert(keys.emplace(variant, size).second);
      assert(keys.at(variant) == size);
      assert(keys.at(original) == 0);
    };

    check([](auto &k) { k.route = PipelineRoute::Mesh; });
    check([](auto &k) { k.route = PipelineRoute::ComputeExpansion; });
    check([](auto &k) { k.route = PipelineRoute::Compute; });
    check([](auto &k) { k.shaders.vertex.reset(); });
    check([&](auto &k) { k.shaders.object = source; });
    check([&](auto &k) { k.shaders.mesh = source; });
    check([&](auto &k) { k.shaders.transform = source; });
    check([&](auto &k) { k.shaders.expand = source; });
    check([](auto &k) { k.shaders.vertex->entry += 'A'; });
    check([](auto &k) {
      std::get<ShaderSource>(k.shaders.vertex->artifact).name += 'A';
    });
    check([](auto &k) {
      std::get<ShaderSource>(k.shaders.vertex->artifact).text += 'A';
    });
    check([](auto &k) {
      std::get<ShaderSource>(k.shaders.vertex->artifact).preamble += 'A';
    });
    check([](auto &k) {
      std::get<ShaderSource>(k.shaders.vertex->artifact).options.math =
          ShaderMath::Fast;
    });
    check([](auto &k) {
      std::get<ShaderSource>(k.shaders.vertex->artifact).options.optimization =
          ShaderOptimization::Size;
    });
    check([](auto &k) {
      std::get<ShaderSource>(k.shaders.vertex->artifact)
          .options.preserve_invariance = false;
    });
    check([](auto &k) {
      std::get<ShaderSource>(k.shaders.vertex->artifact)
          .options.definitions["GPU"] = "1";
    });
    check([](auto &k) {
      std::get<std::vector<u8>>(k.shaders.fragment->artifact).back() = 1;
    });
    check([&](auto &k) { k.shaders.fragment = source; });

    check([](auto &k) { k.vertex_layout.bindings.clear(); });
    check([](auto &k) { k.vertex_layout.attributes.clear(); });
    check([](auto &k) { ++k.vertex_layout.bindings[0].buffer_index; });
    check([](auto &k) { ++k.vertex_layout.bindings[0].stride; });
    check([](auto &k) { ++k.vertex_layout.bindings[0].step_rate; });
    check([](auto &k) {
      k.vertex_layout.bindings[0].step = MTLVertexStepFunctionConstant;
    });
    check([](auto &k) { ++k.vertex_layout.attributes[0].buffer_index; });
    check([](auto &k) { ++k.vertex_layout.attributes[0].location; });
    check([](auto &k) { ++k.vertex_layout.attributes[0].offset; });
    check([](auto &k) {
      k.vertex_layout.attributes[0].format = MTLVertexFormatUChar4;
    });
    check([](auto &k) {
      k.vertex_layout.attributes[0].guest_format =
          P::VertexAttributeFormat::UBYTE;
    });
    check([](auto &k) { --k.vertex_layout.attributes[0].components; });
    check([](auto &k) {
      k.vertex_layout.attributes[0].convert_integer_to_float = false;
    });

    check([](auto &k) { k.attachments.color = MTLPixelFormatBGRA8Unorm; });
    check([](auto &k) { k.attachments.depth = MTLPixelFormatDepth32Float; });
    check([](auto &k) { k.attachments.stencil = MTLPixelFormatStencil8; });
    for (const u32 samples : {2u, 4u, 8u}) {
      check([&](auto &k) { k.attachments.samples = samples; });
    }
    check([](auto &k) { k.raster.guest_topology = P::TriangleTopology::Fan; });
    check([](auto &k) { k.raster.cull_mode = R::CullMode::KeepClockWise; });
    check([](auto &k) { k.raster.primitive = MTLPrimitiveTypeTriangleStrip; });
    check([](auto &k) { k.raster.index_type = PipelineIndexType::UInt16; });
    check([](auto &k) { k.raster.index_type = PipelineIndexType::UInt32; });
    check([](auto &k) { k.raster.flip_viewport = true; });
    check([](auto &k) { k.raster.rasterization_enabled = false; });

    for (const auto channel :
         {&PipelineBlendState::rgb, &PipelineBlendState::alpha}) {
      check([&](auto &k) {
        (k.blend.*channel).source = F::BlendFactor::ConstantColor;
      });
      check([&](auto &k) {
        (k.blend.*channel).destination = F::BlendFactor::DestAlpha;
      });
      check([&](auto &k) {
        (k.blend.*channel).equation = F::BlendEquation::Max;
      });
      check([&](auto &k) { (k.blend.*channel).shader_emulation = true; });
    }
    check([](auto &k) { k.blend.logic_op = F::LogicOp::NoOp; });
    check([](auto &k) { k.blend.enabled = true; });
    check([](auto &k) { k.blend.write_mask = 1; });

    check([](auto &k) {
      k.depth_stencil.depth_compare = F::CompareFunc::LessThan;
    });
    check([](auto &k) {
      k.depth_stencil.stencil_compare = F::CompareFunc::GreaterThan;
    });
    check(
        [](auto &k) { k.depth_stencil.stencil_fail = F::StencilAction::Zero; });
    check([](auto &k) {
      k.depth_stencil.stencil_pass = F::StencilAction::Replace;
    });
    check([](auto &k) {
      k.depth_stencil.stencil_depth_fail = F::StencilAction::IncrementWrap;
    });
    check([](auto &k) { k.depth_stencil.stencil_read_mask = 3; });
    check([](auto &k) { k.depth_stencil.stencil_write_mask = 0; });
    check([](auto &k) { k.depth_stencil.depth_test = true; });
    check([](auto &k) { k.depth_stencil.depth_write = true; });
    check([](auto &k) { k.depth_stencil.stencil_test = true; });

    check([](auto &k) {
      k.fragment.feedback = MSL::FragmentFeedback::RasterOrderGroup;
    });
    check([](auto &k) {
      k.fragment.feedback = MSL::FragmentFeedback::TextureCopy;
    });
    check(
        [](auto &k) { k.fragment.shadow = MSL::ShadowStorage::AtomicBuffer; });
    check([](auto &k) { k.fragment.barycentrics = true; });
    check([](auto &k) { k.fragment.writes_color = true; });
    check([](auto &k) { k.fragment.writes_depth = true; });
    check([](auto &k) { ++k.mesh.object_threads; });
    check([](auto &k) { ++k.mesh.mesh_threads; });
    check([](auto &k) { ++k.mesh.max_mesh_threadgroups; });
    check([](auto &k) { ++k.mesh.payload_bytes; });

    // Metal enumerates the color bits in the opposite order to PICA.
    constexpr std::array<NSUInteger, 16> expected{0, 8, 4, 12, 2, 10, 6, 14,
                                                  1, 9, 5, 13, 3, 11, 7, 15};
    auto blend = original.blend;
    for (std::size_t i = 0; i < expected.size(); ++i) {
      blend.write_mask = static_cast<u8>(i);
      assert(blend.ColorWriteMask() == expected[i]);
    }
    blend.logic_op = F::LogicOp::NoOp;
    assert(blend.ColorWriteMask() == MTLColorWriteMaskNone);
    blend.enabled = true;
    assert(blend.ColorWriteMask() == MTLColorWriteMaskAll);
    auto depth = original;
    depth.depth_stencil.depth_write = true;
    assert(!depth.WritesDepthOrStencil());
    depth.depth_stencil.depth_test = true;
    assert(depth.WritesDepthOrStencil());
    depth.attachments.depth = MTLPixelFormatInvalid;
    assert(!depth.WritesDepthOrStencil());
    depth.depth_stencil.stencil_test = true;
    assert(depth.WritesDepthOrStencil());
    depth.depth_stencil.stencil_write_mask = 0;
    assert(!depth.WritesDepthOrStencil());

    auto render = [MTL4RenderPipelineDescriptor new];
    auto mesh = [MTL4MeshRenderPipelineDescriptor new];
    auto compute = [MTL4ComputePipelineDescriptor new];
    auto stencil = [MTLDepthStencilDescriptor new];
    render.rasterSampleCount = 4;
    render.colorAttachments[0].writeMask = MTLColorWriteMaskRed;
    mesh.requiredThreadsPerMeshThreadgroup = MTLSizeMake(32, 1, 1);
    mesh.maxTotalThreadgroupsPerMeshGrid = 1;
    compute.computeFunctionDescriptor = [MTL4LibraryFunctionDescriptor new];
    stencil.frontFaceStencil.readMask = 0xc3;
    assert(render.rasterSampleCount == 4 &&
           render.colorAttachments[0].writeMask == 8);
    assert(mesh.requiredThreadsPerMeshThreadgroup.width == 32 &&
           mesh.maxTotalThreadgroupsPerMeshGrid == 1);
    assert(compute.computeFunctionDescriptor);
    std::printf("PASS structural equality across %zu graphics pipeline keys, "
                "native colors and depth/stencil predicates, and live SDK "
                "descriptor declarations\n",
                keys.size());
  }
}
