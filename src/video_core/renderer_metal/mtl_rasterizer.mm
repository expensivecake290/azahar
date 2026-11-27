// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.txt file included.

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>
#include "common/alignment.h"
#include "common/logging/log.h"
#include "core/core.h"
#include "core/loader/loader.h"
#include "core/memory.h"
#include "video_core/pica/pica_core.h"
#include "video_core/rasterizer_cache/rasterizer_cache_base.h"
#include "video_core/renderer_metal/mtl_pipeline_cache.h"
#include "video_core/renderer_metal/mtl_rasterizer.h"
#include "video_core/renderer_metal/mtl_stream_buffer.h"
#include "video_core/texture/texture_decode.h"

namespace Metal {
namespace {

namespace MSL = Pica::Shader::Generator::MSL;
using Pica::Shader::Generator::FSUniformData;
using Pica::Shader::Generator::VSPicaUniformData;
using Pica::Shader::Generator::VSUniformData;
using Topology = Pica::PipelineRegs::TriangleTopology;
using AttributeFormat = Pica::PipelineRegs::VertexAttributeFormat;
using TextureType = Pica::TexturingRegs::TextureConfig::TextureType;
using Buffer = MemoryAllocator::Buffer;
using BufferLease = MemoryAllocator::GPUBufferAccess;
using RegisterBank = MSL::RegisterState::RegisterBank;
using VertexVector = MSL::RegisterState::Vector;

constexpr u32 VertexBufferSlot = 12;
constexpr u32 FragmentTextureCount =
    MSL::FragmentBindings::FirstShadowCubeFace + MSL::FragmentBindings::ShadowCubeFaceCount;
constexpr u32 UniformAlignment = 256;
constexpr u32 UniformVertexOffset =
    Common::AlignUp<u32>(sizeof(VSPicaUniformData), UniformAlignment);
constexpr u32 UniformFragmentOffset =
    UniformVertexOffset + Common::AlignUp<u32>(sizeof(VSUniformData), UniformAlignment);
constexpr u32 UniformBytes = UniformFragmentOffset + sizeof(FSUniformData);
constexpr u32 LightFogEntries = Pica::LightingRegs::NumLightingSampler * 256 + 128;
constexpr u32 ProcValueEntries = 3 * 128;
constexpr u32 ProcColorEntries = 2 * 256;
constexpr VertexVector DefaultAttribute{0.f, 0.f, 0.f, 1.f};

struct VertexSource {
    MemoryRef memory;
    u32 stride{};
    u32 width{};
    u32 count{};
    u32 input_register{};
    AttributeFormat format{};
};

struct GuestVertices {
    std::array<VertexSource, 16> attributes{};
    std::array<VertexVector, 16> defaults{};
    std::vector<u32> indices;
    u32 first{};
    u32 count{};
};

struct FragmentResources {
    std::array<TextureResource, FragmentTextureCount> textures;
    std::array<id<MTLSamplerState>, 4> samplers{};
    ShadowTextureBinding shadow;
    std::optional<VideoCore::SurfaceId> shadow_surface;
};

struct LutView {
    Buffer owner;
    id<MTLTexture> view;
};

struct DrawGeometry {
    PipelineRoute route;
    MTLPrimitiveType primitive;
    u32 count;
    std::optional<Buffer> vertices;
    std::optional<Buffer> indices;
    u32 index_count{};
    bool uses_pica_uniforms{};
};

template <typename T>
void CopyBytes(std::span<std::byte> destination, std::size_t offset, const T& value) {
    std::memcpy(destination.data() + offset, std::addressof(value), sizeof(value));
}

template <typename T>
T ReadValue(const u8* data) {
    T value;
    std::memcpy(std::addressof(value), data, sizeof(T));
    return value;
}

VertexVector ReadAttribute(const VertexSource& source, u32 vertex) {
    auto result = DefaultAttribute;
    const auto* element = source.memory.GetPtr() + std::size_t{source.stride} * vertex;
    for (u32 i = 0; i < source.count; ++i, element += source.width) {
        switch (source.format) {
        case AttributeFormat::BYTE:
            result[i] = ReadValue<s8>(element);
            break;
        case AttributeFormat::UBYTE:
            result[i] = *element;
            break;
        case AttributeFormat::SHORT:
            result[i] = ReadValue<s16>(element);
            break;
        case AttributeFormat::FLOAT:
            result[i] = ReadValue<f32>(element);
            break;
        }
    }
    return result;
}

RegisterBank LoadVertex(const GuestVertices& input, u32 vertex) {
    RegisterBank registers;
    registers.fill(DefaultAttribute);
    for (u32 i = 0; i < input.attributes.size(); ++i) {
        const auto& attribute = input.attributes[i];
        if (attribute.memory) {
            registers[attribute.input_register] = ReadAttribute(attribute, vertex);
        }
    }
    for (u32 i = 0; i < input.attributes.size(); ++i) {
        const auto& attribute = input.attributes[i];
        if (!attribute.memory && attribute.count) {
            registers[attribute.input_register] = input.defaults[i];
        }
    }
    return registers;
}

PipelineVertexLayout ProgrammableLayout() {
    PipelineVertexLayout layout;
    layout.bindings.push_back({VertexBufferSlot, sizeof(RegisterBank)});
    for (u32 index = 0; index < 16; ++index) {
        layout.attributes.push_back({index, VertexBufferSlot,
                                     static_cast<u32>(index * sizeof(VertexVector)),
                                     MTLVertexFormatFloat4, AttributeFormat::FLOAT, 4, false});
    }
    return layout;
}

PipelineAttachments Attachments(const Framebuffer& framebuffer, const Instance& instance) {
    if (framebuffer.shadow_rendering) {
        return {};
    }
    const auto& formats = instance.GetFormats();
    const auto color = formats.GetTraits(framebuffer.Format(VideoCore::SurfaceType::Color));
    const auto depth = formats.GetTraits(framebuffer.Format(VideoCore::SurfaceType::Depth));
    return {color.native, depth.native,
            depth.aspect == TextureAspect::DepthStencil ? depth.native : MTLPixelFormatInvalid,
            framebuffer.Samples()};
}

GraphicsPipelineKey DrawState(const Pica::RegsInternal& regs, bool accelerated, bool indexed) {
    GraphicsPipelineKey key;
    const auto& fb = regs.framebuffer.framebuffer;
    const auto& output = regs.framebuffer.output_merger;
    key.raster.guest_topology =
        accelerated ? regs.pipeline.triangle_topology.Value() : Topology::List;
    key.raster.index_type = indexed ? PipelineIndexType::UInt32 : PipelineIndexType::None;
    key.raster.cull_mode = regs.rasterizer.cull_mode;
    key.raster.flip_viewport = fb.IsFlipped();
    const auto blend = output.alpha_blending;
    key.blend = {
        .rgb = {blend.factor_source_rgb, blend.factor_dest_rgb, blend.blend_equation_rgb},
        .alpha = {blend.factor_source_a, blend.factor_dest_a, blend.blend_equation_a},
        .logic_op = output.logic_op,
        .write_mask =
            static_cast<u8>(fb.allow_color_write ? (output.depth_color_mask >> 8) & 15 : 0),
        .enabled = static_cast<bool>(output.alphablend_enable),
    };
    const auto stencil = output.stencil_test;
    key.depth_stencil = {
        .depth_compare = output.depth_test_enable ? output.depth_test_func.Value()
                                                  : Pica::FramebufferRegs::CompareFunc::Always,
        .stencil_compare = stencil.func,
        .stencil_fail = stencil.action_stencil_fail,
        .stencil_pass = stencil.action_depth_pass,
        .stencil_depth_fail = stencil.action_depth_fail,
        .stencil_read_mask = static_cast<u8>(stencil.input_mask),
        .stencil_write_mask =
            static_cast<u8>(fb.allow_depth_stencil_write ? stencil.write_mask.Value() : 0),
        .depth_test = output.depth_test_enable || output.depth_write_enable,
        .depth_write = fb.allow_depth_stencil_write && output.depth_write_enable,
        .stencil_test = regs.framebuffer.HasStencil() && stencil.enable,
    };
    return key;
}

} // Anonymous namespace

struct RasterizerMetal::State {
    State(RasterizerMetal& owner, VideoCore::CustomTexManager& custom_tex_manager,
          VideoCore::RendererBase& renderer, Instance& instance, Scheduler& scheduler,
          RenderManager& render_manager, DescriptorUpdateQueue& descriptors,
          MemoryAllocator& allocator, MemoryBudget& budget, PipelineCacheOptions options)
        : owner(owner), instance(instance), scheduler(scheduler), render_manager(render_manager),
          descriptors(descriptors), allocator(allocator),
          profile(PipelineDeviceProfile::Detect(instance.GetContext())),
          pipelines(instance.GetContext(), profile, std::move(options)),
          runtime(instance, scheduler, render_manager, allocator, budget, 16),
          cache(owner.memory, custom_tex_manager, runtime, owner.regs, renderer),
          uniforms(allocator, scheduler, 2 * 1024 * 1024) {}

    PipelineVertexLayout SoftwareLayout() const {
        constexpr std::array offsets{
            offsetof(HardwareVertex, position),   offsetof(HardwareVertex, color),
            offsetof(HardwareVertex, tex_coord0), offsetof(HardwareVertex, tex_coord1),
            offsetof(HardwareVertex, tex_coord2), offsetof(HardwareVertex, tex_coord0_w),
            offsetof(HardwareVertex, normquat),   offsetof(HardwareVertex, view),
        };
        constexpr std::array<u8, 8> counts{4, 4, 2, 2, 2, 1, 4, 3};
        constexpr std::array formats{MTLVertexFormatFloat, MTLVertexFormatFloat2,
                                     MTLVertexFormatFloat3, MTLVertexFormatFloat4};
        PipelineVertexLayout layout;
        layout.bindings.push_back({VertexBufferSlot, sizeof(HardwareVertex)});
        for (u32 index = 0; index < counts.size(); ++index) {
            layout.attributes.push_back({index, VertexBufferSlot, static_cast<u32>(offsets[index]),
                                         formats[counts[index] - 1], AttributeFormat::FLOAT,
                                         counts[index], false});
        }
        return layout;
    }

    std::optional<GuestVertices> PrepareGuestVertices(bool indexed) {
        const auto& regs = owner.regs;
        const auto& pipeline = regs.pipeline;
        const auto& attributes = pipeline.vertex_attributes;
        const PAddr base = attributes.GetPhysicalBaseAddress();
        GuestVertices input;
        input.count = pipeline.num_vertices;
        if (!input.count) {
            return input;
        }

        u32 last{};
        if (indexed) {
            const bool wide = pipeline.index_array.format != 0;
            const u64 size = u64{input.count} * (wide ? 2 : 1);
            const u64 address = u64{base} + pipeline.index_array.offset;
            if (address + size > std::numeric_limits<PAddr>::max() ||
                size > std::numeric_limits<u32>::max()) {
                return {};
            }
            cache.FlushRegion(static_cast<PAddr>(address), static_cast<u32>(size));
            const auto memory = owner.memory.GetPhysicalRef(static_cast<PAddr>(address));
            if (!memory.GetPtr() || size > memory.GetSize()) {
                return {};
            }
            const std::span bytes{memory.GetPtr(), static_cast<std::size_t>(size)};
            input.indices.resize(input.count);
            for (u32 i = 0; i < input.count; ++i) {
                input.indices[i] = ReadVertexIndex(bytes, i, wide).value();
            }
            const auto [first_it, last_it] =
                std::minmax_element(input.indices.begin(), input.indices.end());
            input.first = *first_it;
            last = *last_it;
        } else {
            input.first = pipeline.vertex_offset;
            const auto end = u64{input.first} + input.count - 1;
            if (end > std::numeric_limits<u32>::max()) {
                return {};
            }
            last = static_cast<u32>(end);
        }

        for (const auto& loader : attributes.attribute_loaders) {
            if (!loader.byte_count) {
                continue;
            }
            u32 offset = 0;
            for (u32 i = 0; i < std::min<u32>(loader.component_count, 12); ++i) {
                const u32 attribute = loader.GetComponent(i);
                if (attribute >= 12) {
                    offset = Common::AlignUp(offset, 4u) + (attribute - 11) * 4;
                    continue;
                }
                const auto width = attributes.GetElementSizeInBytes(attribute);
                const auto stride = attributes.GetStride(attribute);
                offset = Common::AlignUp(offset, width);
                if (attribute <= regs.vs.max_input_attribute_index) {
                    const u64 address = u64{base} + loader.data_offset + offset +
                                        u64{input.first} * loader.byte_count;
                    const u64 length = u64{last - input.first} * loader.byte_count + stride;
                    if (address + length > std::numeric_limits<PAddr>::max() ||
                        length > std::numeric_limits<u32>::max()) {
                        return {};
                    }
                    cache.FlushRegion(static_cast<PAddr>(address), static_cast<u32>(length));
                    const auto memory = owner.memory.GetPhysicalRef(static_cast<PAddr>(address));
                    if (!memory.GetPtr() || length > memory.GetSize()) {
                        return {};
                    }
                    input.attributes[attribute] = {memory,
                                                   loader.byte_count,
                                                   width,
                                                   attributes.GetNumElements(attribute),
                                                   regs.vs.GetRegisterForAttribute(attribute),
                                                   attributes.GetFormat(attribute)};
                }
                offset += stride;
            }
        }
        for (u32 i = 0; i <= regs.vs.max_input_attribute_index; ++i) {
            if (!attributes.IsDefaultAttribute(i) || input.attributes[i].memory) {
                continue;
            }
            input.attributes[i].count = 4;
            input.attributes[i].input_register = regs.vs.GetRegisterForAttribute(i);
            const auto& value = owner.pica.input_default_attributes[i];
            input.defaults[i] = {value.x.ToFloat32(), value.y.ToFloat32(), value.z.ToFloat32(),
                                 value.w.ToFloat32()};
        }
        return input;
    }

    template <typename T>
    Buffer Upload(std::span<const T> data) const {
        const auto buffer = allocator.AllocateBuffer(data.size_bytes(), MemoryPool::Stream);
        const auto mapping = buffer.MapWrite({0, data.size_bytes()});
        std::memcpy(mapping.Bytes().data(), data.data(), data.size_bytes());
        return buffer;
    }

    template <typename T>
    LutView MakeLut(std::span<const T> source, MTLPixelFormat format) const {
        const auto buffer = Upload(source);
        const auto use = buffer.AcquireGPUAccess({0, buffer.Size()}, BufferAccess::Read);
        auto descriptor = [MTLTextureDescriptor
            textureBufferDescriptorWithPixelFormat:format
                                             width:source.size()
                                   resourceOptions:MTLResourceStorageModeShared |
                                                   MTLResourceHazardTrackingModeUntracked
                                             usage:MTLTextureUsageShaderRead];
        const auto view = CheckResource([use.GetBuffer() newTextureWithDescriptor:descriptor
                                                                           offset:0
                                                                      bytesPerRow:buffer.Size()],
                                        "Create native PICA LUT texel buffer view");
        return {buffer, view};
    }

    void SyncLuts() {
        auto& pica = owner.pica;
        auto& fs = owner.fs_data;
        if (!light_fog || pica.lighting.lut_dirty || pica.fog.lut_dirty) {
            std::array<Common::Vec2f, LightFogEntries> values;
            for (u32 i = 0; i < pica.lighting.luts.size(); ++i) {
                fs.lighting_lut_offset[i / 4][i % 4] = static_cast<int>(i * 256);
                for (u32 j = 0; j < pica.lighting.luts[i].size(); ++j) {
                    const auto& entry = pica.lighting.luts[i][j];
                    values[i * 256 + j] = {entry.ToFloat(), entry.DiffToFloat()};
                }
            }
            constexpr u32 fog_offset = LightFogEntries - 128;
            fs.fog_lut_offset = fog_offset;
            for (u32 i = 0; i < pica.fog.lut.size(); ++i) {
                values[fog_offset + i] = {pica.fog.lut[i].ToFloat(), pica.fog.lut[i].DiffToFloat()};
            }
            light_fog = MakeLut(std::span<const Common::Vec2f>{values}, MTLPixelFormatRG32Float);
            pica.lighting.lut_dirty = 0;
            pica.fog.lut_dirty = false;
        }
        auto& proc = pica.proctex;
        if (!proc_values || !proc_colors || proc.table_dirty) {
            std::array<Common::Vec2f, ProcValueEntries> values;
            std::array<Common::Vec4f, ProcColorEntries> colors;
            fs.proctex_noise_lut_offset = 0;
            fs.proctex_color_map_offset = 128;
            fs.proctex_alpha_map_offset = 256;
            fs.proctex_lut_offset = 0;
            fs.proctex_diff_lut_offset = 256;
            const auto copy_values = [&](const auto& lut, u32 offset) {
                for (u32 i = 0; i < lut.size(); ++i) {
                    values[offset + i] = {lut[i].ToFloat(), lut[i].DiffToFloat()};
                }
            };
            copy_values(proc.noise_table, 0);
            copy_values(proc.color_map_table, 128);
            copy_values(proc.alpha_map_table, 256);
            for (u32 i = 0; i < 256; ++i) {
                colors[i] = proc.color_table[i].ToVector() / 255.f;
                colors[256 + i] = proc.color_diff_table[i].ToVector() / 255.f;
            }
            proc_values = MakeLut(std::span<const Common::Vec2f>{values}, MTLPixelFormatRG32Float);
            proc_colors =
                MakeLut(std::span<const Common::Vec4f>{colors}, MTLPixelFormatRGBA32Float);
            proc.table_dirty = 0;
        }
    }

    BufferLease UploadUniforms(bool accelerated) {
        auto mapping = uniforms.Map(UniformBytes, UniformAlignment);
        if (accelerated) {
            VSPicaUniformData values{};
            values.SetFromRegs(owner.pica.vs_setup);
            CopyBytes(mapping.Bytes(), 0, values);
            owner.pica.vs_setup.uniforms_dirty = false;
        }
        CopyBytes(mapping.Bytes(), UniformVertexOffset, owner.vs_data);
        CopyBytes(mapping.Bytes(), UniformFragmentOffset, owner.fs_data);
        owner.vs_data_dirty = false;
        owner.fs_data_dirty = false;
        const auto slice = uniforms.Commit(std::move(mapping), UniformBytes);
        return uniforms.AcquireGPUAccess(slice, BufferAccess::Read);
    }

    FragmentResources PrepareTextures(Framebuffer& framebuffer) {
        FragmentResources result;
        const auto& texturing = owner.regs.texturing;
        const auto textures = texturing.GetTextures();
        owner.user_config.raw = 0;
        for (u32 unit = 0; unit < textures.size(); ++unit) {
            const auto& config = textures[unit];
            const auto type = unit == 0 ? config.config.type.Value() : TextureType::Texture2D;
            if (!config.enabled) {
                const auto id = type == TextureType::TextureCube ? VideoCore::NULL_SURFACE_CUBE_ID
                                                                 : VideoCore::NULL_SURFACE_ID;
                result.textures[unit] = cache.GetSurface(id).Resource();
                result.samplers[unit] = cache.GetSampler(VideoCore::NULL_SAMPLER_ID).Handle();
                continue;
            }
            auto& sampler = cache.GetSampler(config.config);
            result.samplers[unit] = sampler.Handle();
            using CubeFace = Pica::TexturingRegs::CubeFace;
            if (type == TextureType::ShadowCube) {
                constexpr std::array faces{CubeFace::PositiveX, CubeFace::NegativeX,
                                           CubeFace::PositiveY, CubeFace::NegativeY,
                                           CubeFace::PositiveZ, CubeFace::NegativeZ};
                auto info =
                    Pica::Texture::TextureInfo::FromPicaRegister(config.config, config.format);
                for (u32 face = 0; face < faces.size(); ++face) {
                    info.physical_address = texturing.GetCubePhysicalAddress(faces[face]);
                    auto& surface = cache.GetSurface(cache.GetTextureSurface(info));
                    surface.flags |= VideoCore::SurfaceFlagBits::ShadowSource;
                    result.textures[MSL::FragmentBindings::FirstShadowCubeFace + face] =
                        surface.ShadowImage();
                }
                continue;
            }
            if (type == TextureType::TextureCube) {
                const VideoCore::TextureCubeConfig cube{
                    .px = texturing.GetCubePhysicalAddress(CubeFace::PositiveX),
                    .nx = texturing.GetCubePhysicalAddress(CubeFace::NegativeX),
                    .py = texturing.GetCubePhysicalAddress(CubeFace::PositiveY),
                    .ny = texturing.GetCubePhysicalAddress(CubeFace::NegativeY),
                    .pz = texturing.GetCubePhysicalAddress(CubeFace::PositiveZ),
                    .nz = texturing.GetCubePhysicalAddress(CubeFace::NegativeZ),
                    .width = config.config.width,
                    .levels = config.config.lod.max_level + 1,
                    .format = config.format,
                };
                result.textures[unit] = cache.GetTextureCube(cube).Resource();
                continue;
            }
            auto& surface = cache.GetTextureSurface(config);
            if (type == TextureType::Shadow2D) {
                surface.flags |= VideoCore::SurfaceFlagBits::ShadowSource;
                result.textures[unit] = surface.ShadowImage();
                continue;
            }
            if ((framebuffer.color_id &&
                 surface.Image() == cache.GetSurface(framebuffer.color_id).Image()) ||
                (framebuffer.depth_id &&
                 surface.Image() == cache.GetSurface(framebuffer.depth_id).Image())) {
                (void)surface.CopyImageView();
                result.textures[unit] = surface.Resource(TextureRepresentation::Feedback);
            } else {
                result.textures[unit] = surface.Resource();
            }
            if (unit == 0) {
                auto normal = surface.Resource(TextureRepresentation::Normal);
                if (normal) {
                    result.textures[MSL::FragmentBindings::NormalMap] = std::move(normal);
                    result.samplers[MSL::FragmentBindings::NormalMapSampler] = sampler.Handle();
                    owner.user_config.use_custom_normal.Assign(1);
                }
            }
        }
        if (framebuffer.shadow_rendering) {
            auto& surface = cache.GetSurface(framebuffer.color_id);
            result.shadow = surface.PrepareShadowStorage(profile.texture_atomics);
            result.shadow_surface = framebuffer.color_id;
        }
        return result;
    }

    void BindRange(const DescriptorUpdateQueue::Set& set, u32 slot, const BufferLease& use,
                   std::size_t offset, std::size_t size, BufferAccess access = BufferAccess::Read) {
        descriptors.AddBuffer(set, slot, use.GetBuffer(), use.Range().offset + offset, size,
                              access);
    }

    static BufferLease Access(const Buffer& buffer, BufferAccess access = BufferAccess::Read) {
        return buffer.AcquireGPUAccess({0, buffer.Size()}, access);
    }

    void RecordCompute(id<MTLComputePipelineState> pipeline,
                       DescriptorUpdateQueue::Snapshot arguments, u32 count,
                       std::vector<BufferLease> uses, std::shared_ptr<BufferLease> uniform = {}) {
        render_manager.EndRendering();
        scheduler.Record([pipeline, arguments = std::move(arguments), count, uses = std::move(uses),
                          uniform = std::move(uniform)](EncodingContext& encoding) mutable {
            auto encoder = CheckResource([encoding.GetCommandBuffer() computeCommandEncoder],
                                         "Encode PICA geometry in Metal 4");
            [encoder setComputePipelineState:pipeline];
            arguments.BindCompute(encoding, encoder);
            const auto width = std::max<NSUInteger>(
                1, std::min(count, static_cast<u32>(pipeline.threadExecutionWidth)));
            [encoder dispatchThreads:MTLSizeMake(count, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
            encoding.Retain(pipeline);
            if (uniform) {
                encoding.Retain(std::move(uniform));
            }
            for (auto& use : uses) {
                encoding.Retain(std::move(use));
            }
            encoding.EndPass(encoder);
        });
    }

    DrawGeometry PrepareGeometry(const GuestVertices* guest, const GraphicsPipelineKey& key,
                                 const NativePipelineStates& native,
                                 const std::shared_ptr<BufferLease>& uniform, u32 count) {
        DrawGeometry geometry{key.route, key.raster.primitive, count};
        if (!guest) {
            geometry.vertices =
                Upload(std::span<const HardwareVertex>{owner.vertex_batch.data(), count});
            return geometry;
        }
        const bool indexed = !guest->indices.empty();
        if (key.route == PipelineRoute::Vertex) {
            geometry.uses_pica_uniforms = true;
            const u32 vertices =
                indexed ? *std::max_element(guest->indices.begin(), guest->indices.end()) -
                              guest->first + 1
                        : count;
            auto buffer = allocator.AllocateBuffer(std::size_t{vertices} * sizeof(RegisterBank),
                                                   MemoryPool::Stream);
            {
                auto mapping = buffer.MapWrite({0, buffer.Size()});
                for (u32 i = 0; i < vertices; ++i) {
                    CopyBytes(mapping.Bytes(), i * sizeof(RegisterBank), LoadVertex(*guest, i));
                }
            }
            geometry.vertices = std::move(buffer);
            if (indexed) {
                auto indices = guest->indices;
                for (auto& index : indices) {
                    index -= guest->first;
                }
                geometry.indices = Upload(std::span<const u32>{indices});
                geometry.index_count = count;
            }
            return geometry;
        }
        auto packed =
            allocator.AllocateBuffer(std::size_t{count} * sizeof(RegisterBank), MemoryPool::Stream);
        {
            auto mapping = packed.MapWrite({0, packed.Size()});
            for (u32 i = 0; i < count; ++i) {
                const u32 source = indexed ? guest->indices[i] - guest->first : i;
                CopyBytes(mapping.Bytes(), i * sizeof(RegisterBank), LoadVertex(*guest, source));
            }
        }
        const auto transformed =
            allocator.AllocateBuffer(std::size_t{count} * sizeof(MSL::TransformedVertex),
                                     MemoryPool::Transient, MTLStorageModePrivate);
        const MSL::VertexTransformParams data{count};
        const auto params = Upload(std::span<const MSL::VertexTransformParams>{&data, 1});
        const auto set = descriptors.CreateSet({.buffers = MSL::VertexBindings::Parameters + 1});
        std::vector<BufferLease> uses;
        uses.push_back(Access(packed));
        uses.push_back(Access(transformed, BufferAccess::Write));
        uses.push_back(Access(params));
        BindRange(set, MSL::VertexBindings::InputVertices, uses[0], 0, packed.Size());
        BindRange(set, MSL::VertexBindings::OutputVertices, uses[1], 0, transformed.Size(),
                  BufferAccess::Write);
        BindRange(set, MSL::VertexBindings::Parameters, uses[2], 0, params.Size());
        BindRange(set, MSL::VertexBindings::PicaUniforms, *uniform, 0, sizeof(VSPicaUniformData));
        RecordCompute(native.transform, descriptors.Capture(set), count, std::move(uses), uniform);

        const u32 primitives = MSL::PrimitiveCount(key.raster.guest_topology, count);
        const MSL::FixedGeometryParams expand{count, 0, primitives, primitives * 3};
        const auto expand_params = Upload(std::span<const MSL::FixedGeometryParams>{&expand, 1});
        if (key.route == PipelineRoute::Mesh) {
            geometry.vertices = transformed;
            geometry.indices = expand_params;
            geometry.count = primitives;
            return geometry;
        }
        auto expanded = allocator.AllocateBuffer(std::size_t{expand.output_vertex_capacity} *
                                                     sizeof(MSL::ExpandedVertex),
                                                 MemoryPool::Transient, MTLStorageModePrivate);
        const auto expand_set =
            descriptors.CreateSet({.buffers = MSL::VertexBindings::Parameters + 1});
        std::vector<BufferLease> expand_uses;
        expand_uses.push_back(Access(transformed));
        expand_uses.push_back(Access(expanded, BufferAccess::Write));
        expand_uses.push_back(Access(expand_params));
        BindRange(expand_set, MSL::VertexBindings::InputVertices, expand_uses[0], 0,
                  transformed.Size());
        BindRange(expand_set, MSL::VertexBindings::OutputVertices, expand_uses[1], 0,
                  expanded.Size(), BufferAccess::Write);
        BindRange(expand_set, MSL::VertexBindings::Parameters, expand_uses[2], 0,
                  expand_params.Size());
        RecordCompute(native.expand, descriptors.Capture(expand_set), primitives,
                      std::move(expand_uses));
        geometry.vertices = std::move(expanded);
        geometry.count = expand.output_vertex_capacity;
        return geometry;
    }

    void RecordDraw(const PipelineSelection& selection, const PipelineBuild& pipeline,
                    const PipelineDrawState& draw, const DrawGeometry& geometry,
                    const FragmentResources& textures, const TextureResource& feedback,
                    id<MTLTexture> feedback_view, std::shared_ptr<BufferLease> uniform, u32 first,
                    u32 count, u32 width, u32 height) {
        const auto fragments =
            descriptors.CreateSet({.buffers = MSL::FragmentBindings::ShadowBufferInfo + 1,
                                   .textures = FragmentTextureCount,
                                   .samplers = 4});
        BindRange(fragments, MSL::FragmentBindings::Uniforms, *uniform, UniformFragmentOffset,
                  sizeof(FSUniformData));
        for (u32 i = 0; i < 3; ++i) {
            descriptors.AddImageSampler(fragments, i, 0, textures.textures[i].Get(),
                                        textures.samplers[i]);
        }
        for (u32 i = MSL::FragmentBindings::FirstShadowCubeFace; i < FragmentTextureCount; ++i) {
            descriptors.AddStorageImage(fragments, i, textures.textures[i].Get(),
                                        BufferAccess::Read);
        }
        descriptors.AddTexelBuffer(fragments, MSL::FragmentBindings::LightingFogLut,
                                   light_fog->view);
        descriptors.AddTexelBuffer(fragments, MSL::FragmentBindings::ProceduralLut,
                                   proc_values->view);
        descriptors.AddTexelBuffer(fragments, MSL::FragmentBindings::ProceduralColorLut,
                                   proc_colors->view);
        descriptors.AddImageSampler(fragments, MSL::FragmentBindings::NormalMap, 0,
                                    textures.textures[MSL::FragmentBindings::NormalMap].Get(),
                                    textures.samplers[MSL::FragmentBindings::NormalMapSampler],
                                    MSL::FragmentBindings::NormalMapSampler);
        if (feedback_view) {
            descriptors.AddStorageImage(fragments, MSL::FragmentBindings::FeedbackTexture,
                                        feedback_view, BufferAccess::Read);
        }
        std::vector<BufferLease> uses;
        uses.push_back(Access(light_fog->owner));
        uses.push_back(Access(proc_values->owner));
        uses.push_back(Access(proc_colors->owner));
        if (textures.shadow.image) {
            descriptors.AddStorageImage(fragments, MSL::FragmentBindings::ShadowTarget,
                                        textures.shadow.image.Get(), BufferAccess::ReadWrite);
        } else if (textures.shadow.buffer) {
            const MSL::FragmentShadowBufferInfo info{width, height, textures.shadow.row_pitch};
            const auto params = Upload(std::span<const MSL::FragmentShadowBufferInfo>{&info, 1});
            uses.push_back(Access(*textures.shadow.buffer, BufferAccess::ReadWrite));
            BindRange(fragments, MSL::FragmentBindings::ShadowBuffer, uses.back(), 0,
                      textures.shadow.buffer->Size(), BufferAccess::ReadWrite);
            uses.push_back(Access(params));
            BindRange(fragments, MSL::FragmentBindings::ShadowBufferInfo, uses.back(), 0,
                      params.Size());
        }

        const auto vertices = descriptors.CreateSet({.buffers = VertexBufferSlot + 1});
        BindRange(vertices, MSL::VertexBindings::RasterUniforms, *uniform, UniformVertexOffset,
                  sizeof(VSUniformData));
        uses.push_back(Access(*geometry.vertices));
        const auto slot = geometry.route == PipelineRoute::Vertex
                              ? VertexBufferSlot
                              : MSL::VertexBindings::InputVertices;
        BindRange(vertices, slot, uses.back(), 0, geometry.vertices->Size());
        MTLGPUAddress index_address{};
        std::size_t index_bytes{};
        auto index_buffer = (id<MTLBuffer>)nil;
        if (geometry.route == PipelineRoute::Mesh) {
            uses.push_back(Access(*geometry.indices));
            BindRange(vertices, MSL::VertexBindings::Parameters, uses.back(), 0,
                      geometry.indices->Size());
        } else if (geometry.index_count) {
            uses.push_back(Access(*geometry.indices));
            index_buffer = uses.back().GetBuffer();
            index_address = index_buffer.gpuAddress;
            index_bytes = geometry.indices->Size();
        }
        if (geometry.uses_pica_uniforms) {
            BindRange(vertices, MSL::VertexBindings::PicaUniforms, *uniform, 0,
                      sizeof(VSPicaUniformData));
        }

        const std::array arguments{
            PipelineArguments{MTLRenderStageFragment, descriptors.Capture(fragments)},
            PipelineArguments{geometry.route == PipelineRoute::Mesh
                                  ? MTLRenderStageObject | MTLRenderStageMesh
                                  : MTLRenderStageVertex,
                              descriptors.Capture(vertices)}};
        const auto route = geometry.route;
        const auto primitive = geometry.primitive;
        auto* context = &instance.GetContext();
        render_manager.RecordDraw(
            [pipeline, draw, arguments, uses = std::move(uses), route, primitive, first, count,
             index_buffer, index_address, index_bytes, context, uniform = std::move(uniform),
             textures, feedback, mesh = selection.key.mesh](
                EncodingContext& encoding, id<MTL4RenderCommandEncoder> encoder) mutable {
                PipelineCache::Bind(encoding, *context, encoder, pipeline, draw, arguments);
                for (const auto& texture : textures.textures) {
                    texture.Retain(encoding);
                }
                textures.shadow.image.Retain(encoding);
                feedback.Retain(encoding);
                encoding.Retain(std::move(uniform));
                if (route == PipelineRoute::Mesh) {
                    [encoder drawMeshThreadgroups:MTLSizeMake(count, 1, 1)
                        threadsPerObjectThreadgroup:MTLSizeMake(mesh.object_threads, 1, 1)
                          threadsPerMeshThreadgroup:MTLSizeMake(mesh.mesh_threads, 1, 1)];
                } else if (index_buffer) {
                    const ResourceUse index{
                        index_buffer, BufferAccess::Read, MTLStageVertex, {0, index_bytes}};
                    encoding.Synchronize(encoder, std::span{&index, 1});
                    [encoder drawIndexedPrimitives:primitive
                                        indexCount:count
                                         indexType:MTLIndexTypeUInt32
                                       indexBuffer:index_address
                                 indexBufferLength:index_bytes];
                } else {
                    [encoder drawPrimitives:primitive vertexStart:first vertexCount:count];
                }
                for (auto& use : uses) {
                    encoding.Retain(std::move(use));
                }
            });
    }

    bool Draw(const GuestVertices* guest, bool indexed) {
        auto& regs = owner.regs;
        owner.SyncDrawUniforms();
        auto key = DrawState(regs, guest != nullptr, indexed);
        const u32 count = guest ? guest->count : static_cast<u32>(owner.vertex_batch.size());
        const auto primitives = MSL::PrimitiveCount(key.raster.guest_topology, count);
        if (!primitives) {
            return true;
        }
        if (primitives > std::numeric_limits<u32>::max() / 3 ||
            u64{count} * sizeof(MSL::TransformedVertex) >
                instance.GetCapabilities().max_buffer_length) {
            if (guest) {
                return false;
            }
            throw std::overflow_error("PICA interpreted draw exceeds Metal vertex buffer limits");
        }
        key.vertex_layout = guest ? ProgrammableLayout() : SoftwareLayout();
        const bool shadow = regs.framebuffer.IsShadowRendering();
        const bool use_color = regs.framebuffer.framebuffer.GetColorBufferPhysicalAddress() != 0 &&
                               (shadow || key.blend.ColorWriteMask());
        const bool use_depth = !shadow &&
                               regs.framebuffer.framebuffer.GetDepthBufferPhysicalAddress() != 0 &&
                               (key.depth_stencil.depth_test || key.depth_stencil.stencil_test);
        const auto framebuffer = cache.GetFramebufferSurfaces(use_color, use_depth);
        auto& target = *framebuffer.Framebuffer();
        const auto area = RenderArea::FromRectangle(framebuffer.DrawRect());
        if (!target.IsValid() || !area) {
            return true;
        }
        const auto viewport = framebuffer.Viewport();
        if (viewport.width <= 0 || viewport.height <= 0) {
            return true;
        }
        const auto [sx1, sy2, sx2, sy1] = framebuffer.Scissor();
        owner.fs_data.scissor_x1 = sx1;
        owner.fs_data.scissor_x2 = sx2;
        owner.fs_data.scissor_y1 = sy1;
        owner.fs_data.scissor_y2 = sy2;
        owner.fs_data.framebuffer_scale = static_cast<int>(target.Scale());
        auto textures = PrepareTextures(target);
        key.attachments = Attachments(target, instance);
        const auto selection =
            pipelines.Select({.regs = regs,
                              .user = owner.user_config,
                              .fixed_state = std::move(key),
                              .guest_color = target.Format(VideoCore::SurfaceType::Color),
                              .vertex_setup = guest ? &owner.pica.vs_setup : nullptr,
                              .vertex_count = count});
        if (!selection) {
            LOG_ERROR(Render, "Unable to select Metal PICA pipeline: {}", selection.diagnostic);
            if (guest && selection.status == PipelineSelectionStatus::InvalidVertexProgram) {
                return false;
            }
            throw std::runtime_error(selection.diagnostic);
        }
        auto pipeline = pipelines.Require(selection.key, selection.cacheable);
        auto native = pipeline.GetStates(instance.GetContext());
        if (!native) {
            LOG_ERROR(Render, "Unable to compile Metal PICA pipeline: {}",
                      pipeline.GetDiagnostic().message);
            if (guest &&
                pipeline.GetStatus(instance.GetContext()) == PipelineBuildStatus::ShaderFailed) {
                return false;
            }
            throw std::runtime_error("Native Metal PICA pipeline creation failed");
        }

        SyncLuts();
        const auto uniform = std::make_shared<BufferLease>(UploadUniforms(guest != nullptr));
        const auto geometry = PrepareGeometry(guest, selection.key, *native, uniform, count);
        auto pass = target.MakePass(*area);
        pass.tile_feedback = selection.fragment.feedback == MSL::FragmentFeedback::RasterOrderGroup;
        const PipelineDrawState draw{
            {static_cast<double>(viewport.x), static_cast<double>(viewport.y),
             static_cast<double>(viewport.width), static_cast<double>(viewport.height), 0., 1.},
            {area->x, area->y, area->width, area->height},
            regs.framebuffer.output_merger.blend_const.raw,
            static_cast<u8>(regs.framebuffer.output_merger.stencil_test.reference_value),
        };
        if (selection.fragment.feedback == MSL::FragmentFeedback::TextureCopy) {
            auto& color = cache.GetSurface(target.color_id);
            const auto representation = pass.samples > 1 ? TextureRepresentation::Multisample
                                                         : TextureRepresentation::Current;
            (void)color.CopyImageView(representation);
            const auto feedback = color.Resource(TextureRepresentation::Feedback);
            const RenderTexture feedback_target{feedback.Get(), pass.color.target.level,
                                                pass.color.target.slice};
            auto feedback_view = feedback.Get();
            if (feedback_target.level > 0) {
                feedback_view = CheckResource(
                    [feedback_view
                        newTextureViewWithPixelFormat:feedback_view.pixelFormat
                                          textureType:MTLTextureType2D
                                               levels:NSMakeRange(feedback_target.level, 1)
                                               slices:NSMakeRange(feedback_target.slice, 1)],
                    "Create Metal destination feedback subresource view");
            }
            for (u32 vertex = 0; vertex < geometry.count; vertex += 3) {
                if (vertex > 0) {
                    render_manager.CopyForFeedback(pass.color.target, feedback_target, *area);
                }
                render_manager.BeginRendering(pass);
                RecordDraw(selection, pipeline, draw, geometry, textures, feedback, feedback_view,
                           uniform, vertex, 3, pass.width, pass.height);
            }
        } else {
            render_manager.BeginRendering(pass);
            RecordDraw(selection, pipeline, draw, geometry, textures, {}, nil, uniform, 0,
                       geometry.count, pass.width, pass.height);
        }
        if (textures.shadow_surface) {
            cache.GetSurface(*textures.shadow_surface).CommitShadowStorage(textures.shadow);
        }
        return true;
    }

    void SwitchTitle(u64 title_id, const std::atomic_bool& stop,
                     const VideoCore::DiskResourceLoadCallback& callback) {
        if (stop) {
            return;
        }
        render_manager.EndRendering();
        if (scheduler.Finish() != CompletionStatus::Complete) {
            throw std::runtime_error("Unable to retire Metal draw work before switching titles");
        }
        using VideoCore::LoadCallbackStage;
        VideoCore::DiskResourceLoadCallback progress;
        if (callback) {
            callback(LoadCallbackStage::Prepare, 0, 0, "");
            progress = [&callback](LoadCallbackStage stage, std::size_t completed,
                                   std::size_t total, const std::string& detail) {
                if (stage != LoadCallbackStage::Prepare && stage != LoadCallbackStage::Complete) {
                    callback(stage, completed, total, detail);
                }
            };
        }
        const auto result = pipelines.SwitchTitle({.program_id = title_id}, stop, progress);
        if (result.previous_load_interrupted) {
            LOG_WARNING(Render, "Skipped interrupted Metal shader warmup for title {:016X}",
                        title_id);
        }
        if (callback && !result.canceled) {
            callback(LoadCallbackStage::Complete, 0, 0, "");
        }
    }

    RasterizerMetal& owner;
    Instance& instance;
    Scheduler& scheduler;
    RenderManager& render_manager;
    DescriptorUpdateQueue& descriptors;
    MemoryAllocator& allocator;
    const PipelineDeviceProfile profile;
    PipelineCache pipelines;
    TextureRuntime runtime;
    RasterizerCache cache;
    StreamBuffer uniforms;
    std::optional<LutView> light_fog;
    std::optional<LutView> proc_values;
    std::optional<LutView> proc_colors;
};

RasterizerMetal::RasterizerMetal(Memory::MemorySystem& memory, Pica::PicaCore& pica,
                                 VideoCore::CustomTexManager& custom_tex_manager,
                                 VideoCore::RendererBase& renderer, Instance& instance,
                                 Scheduler& scheduler, RenderManager& render_manager,
                                 DescriptorUpdateQueue& update_queue, MemoryAllocator& allocator,
                                 MemoryBudget& budget, PipelineCacheOptions options)
    : RasterizerAccelerated(memory, pica),
      state(std::make_unique<State>(*this, custom_tex_manager, renderer, instance, scheduler,
                                    render_manager, update_queue, allocator, budget,
                                    std::move(options))) {}

RasterizerMetal::~RasterizerMetal() {
    state->render_manager.EndRendering();
    (void)state->scheduler.Finish();
}

void RasterizerMetal::TickFrame() {
    state->scheduler.WaitWorker();
    state->cache.TickFrame();
}

void RasterizerMetal::DrawTriangles() {
    if (!vertex_batch.empty()) {
        (void)state->Draw(nullptr, false);
        vertex_batch.clear();
    }
}

bool RasterizerMetal::AccelerateDrawBatch(bool is_indexed) {
    // PicaCore owns the continuation of the programmable geometry shader and emitter state.
    if (regs.pipeline.use_gs != Pica::PipelineRegs::UseGS::No) {
        return false;
    }
    regs.rasterizer.ValidateSemantics();
    const auto vertices = state->PrepareGuestVertices(is_indexed);
    if (!vertices) {
        LOG_DEBUG(Render, "Skipping draw with an invalid PICA vertex or index range");
        return true;
    }
    return state->Draw(std::addressof(*vertices), is_indexed);
}

void RasterizerMetal::FlushAll() {
    state->cache.FlushAll();
}

void RasterizerMetal::FlushRegion(PAddr addr, u32 size) {
    state->cache.FlushRegion(addr, size);
}

void RasterizerMetal::InvalidateRegion(PAddr addr, u32 size) {
    state->cache.InvalidateRegion(addr, size);
}

void RasterizerMetal::FlushAndInvalidateRegion(PAddr addr, u32 size) {
    state->cache.FlushRegion(addr, size);
    state->cache.InvalidateRegion(addr, size);
}

void RasterizerMetal::ClearAll(bool flush) {
    state->cache.ClearAll(flush);
}

bool RasterizerMetal::AccelerateDisplayTransfer(const Pica::DisplayTransferConfig& config) {
    return state->cache.AccelerateDisplayTransfer(config);
}

bool RasterizerMetal::AccelerateTextureCopy(const Pica::DisplayTransferConfig& config) {
    return state->cache.AccelerateTextureCopy(config);
}

bool RasterizerMetal::AccelerateFill(const Pica::MemoryFillConfig& config) {
    return state->cache.AccelerateFill(config);
}

bool RasterizerMetal::AccelerateDisplay(const Pica::FramebufferConfig& config,
                                        PAddr framebuffer_addr, u32 pixel_stride,
                                        ScreenInfo& screen_info) {
    if (!framebuffer_addr || !pixel_stride || !config.width || !config.height) {
        return false;
    }
    VideoCore::SurfaceParams params;
    params.addr = framebuffer_addr;
    params.width = std::min(config.width.Value(), pixel_stride);
    params.height = config.height;
    params.stride = pixel_stride;
    params.is_tiled = false;
    params.pixel_format = VideoCore::PixelFormatFromGPUPixelFormat(config.color_format);
    if (params.pixel_format == VideoCore::PixelFormat::Invalid) {
        return false;
    }
    params.UpdateParams();
    const auto [surface_id, rect] =
        state->cache.GetSurfaceSubRect(params, VideoCore::ScaleMatch::Ignore, true);
    if (!surface_id) {
        return false;
    }
    const auto& surface = state->cache.GetSurface(surface_id);
    const auto coordinates =
        ScreenInfo::Coordinates(rect, surface.GetScaledWidth(), surface.GetScaledHeight());
    auto image = surface.Resource();
    if (!coordinates || !image) {
        return false;
    }
    screen_info = {std::move(image), *coordinates};
    return true;
}

void RasterizerMetal::LoadDefaultDiskResources(
    const std::atomic_bool& stop_loading, const VideoCore::DiskResourceLoadCallback& callback) {
    u64 title{};
    if (Core::System::GetInstance().GetAppLoader().ReadProgramId(title) !=
        Loader::ResultStatus::Success) {
        title = 0;
    }
    state->SwitchTitle(title, stop_loading, callback);
}

void RasterizerMetal::SwitchDiskResources(u64 title_id) {
    const std::atomic_bool stop{false};
    state->SwitchTitle(title_id, stop, switch_disk_resources_callback);
}

bool RasterizerMetal::DiskResourcesLoadCrashed(u64 title_id) {
    return state->pipelines.PreviousLoadInterrupted({.program_id = title_id});
}

} // namespace Metal
