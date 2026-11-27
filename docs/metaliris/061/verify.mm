#include <array>
#include <cassert>
#include <cstdio>
#include <type_traits>
#include "video_core/renderer_metal/mtl_pipeline_cache.h"

using namespace Metal;
using Path = Pica::Shader::Generator::MSL::GeometryPath;
using Topology = Pica::PipelineRegs::TriangleTopology;

static_assert(std::is_same_v<decltype(&PipelineCache::Require),
                             PipelineBuild (PipelineCache::*)(const GraphicsPipelineKey&, bool,
                                                              const std::atomic_bool*)>);
static_assert(
    std::is_same_v<decltype(&PipelineCache::Bind),
                   void (*)(EncodingContext&, const DeviceContext&, id<MTL4RenderCommandEncoder>,
                            const PipelineBuild&, const PipelineDrawState&,
                            std::span<const PipelineArguments>)>);

int main() {
    std::size_t cases{};
    for (const auto topology : {Topology::List, Topology::Strip, Topology::Fan, Topology::Shader}) {
        for (const auto index :
             {PipelineIndexType::None, PipelineIndexType::UInt16, PipelineIndexType::UInt32}) {
            for (const u32 vertices : {3, 6}) {
                for (unsigned flags = 0; flags < 128; ++flags) {
                    const bool mesh = (flags & 1) != 0;
                    const bool barycentric = (flags & 2) != 0;
                    const bool programmable = (flags & 4) != 0;
                    const bool lighting = (flags & 8) != 0;
                    const bool serial = (flags & 16) != 0;
                    const bool independent = (flags & 32) != 0;
                    const bool packed = (flags & 64) != 0;
                    PipelineRasterState raster;
                    raster.guest_topology = topology;
                    raster.index_type = index;
                    PipelineDeviceProfile device;
                    device.mesh_shaders = mesh;
                    device.fragment_barycentrics = barycentric;
                    const auto result =
                        SelectPipelineGeometry(raster, device, vertices, programmable, lighting,
                                               serial, independent, packed);
                    Path expected = Path::Vertex;
                    if (packed || programmable || serial ||
                        (topology == Topology::Strip && index != PipelineIndexType::None)) {
                        expected = Path::Compute;
                    } else if (topology == Topology::Fan || (lighting && !barycentric)) {
                        expected =
                            mesh && (vertices == 3 || independent) ? Path::Mesh : Path::Compute;
                    }
                    assert(result == expected);
                    ++cases;
                }
            }
        }
    }
    assert(cases == 3072);
    const PipelineTitle original{0x0004000001234567, ""};
    const PipelineTitle applet{0x0004003001234567, ""};
    const PipelineTitle homebrew{0, "fixture-title"};
    assert(original.IsIdentified() && !original.IsApplet());
    assert(applet.IsIdentified() && applet.IsApplet());
    assert(homebrew.IsIdentified() && !homebrew.IsApplet() && !PipelineTitle{}.IsIdentified());
    assert(!static_cast<bool>(PipelineSelection{}));
    std::printf("PASS %zu Metal 4 geometry routes, indexed strip restart, title identity, "
                "immutable argument ABI and required pipeline signatures\n",
                cases);
}
