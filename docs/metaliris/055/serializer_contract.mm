#include <type_traits>
#include "video_core/renderer_metal/mtl_shader_util.h"

static_assert(std::is_constructible_v<Metal::ShaderCompiler, Metal::DeviceContext&>);
static_assert(std::is_constructible_v<Metal::ShaderCompiler, Metal::DeviceContext&, std::size_t,
                                      id<MTL4PipelineDataSetSerializer>>);
