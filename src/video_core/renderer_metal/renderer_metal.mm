// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.txt file included.

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>
#include "common/color.h"
#include "common/file_util.h"
#include "common/logging/log.h"
#include "common/memory_detect.h"
#include "common/scope_exit.h"
#include "common/settings.h"
#include "core/core.h"
#include "core/frontend/emu_window.h"
#include "core/memory.h"
#include "core/perf_stats.h"
#include "video_core/host_shaders/metal/cursor_frag.h"
#include "video_core/host_shaders/metal/cursor_vert.h"
#include "video_core/host_shaders/metal/present_anaglyph_frag.h"
#include "video_core/host_shaders/metal/present_frag.h"
#include "video_core/host_shaders/metal/present_interlaced_frag.h"
#include "video_core/host_shaders/metal/present_vert.h"
#include "video_core/pica/pica_core.h"
#include "video_core/renderer_metal/mtl_descriptor_update_queue.h"
#include "video_core/renderer_metal/mtl_instance.h"
#include "video_core/renderer_metal/mtl_metalfx_spatial.h"
#include "video_core/renderer_metal/mtl_pipeline_cache.h"
#include "video_core/renderer_metal/mtl_present_window.h"
#include "video_core/renderer_metal/mtl_rasterizer.h"
#include "video_core/renderer_metal/mtl_shader_util.h"
#include "video_core/renderer_metal/mtl_stream_buffer.h"
#include "video_core/renderer_metal/renderer_metal.h"

namespace Metal {
namespace {

constexpr u32 screen_count = 3;
constexpr u32 bottom_screen = 2;
constexpr u32 max_dimension = 16384;
constexpr auto full_uv = Common::Rectangle<f32>{0, 0, 1, 1};
using StereoMode = Settings::StereoRenderOption;

struct ScreenVertex {
    f32 x, y, u, v;
};

struct alignas(16) PresentUniforms {
    std::array<f32, 16> modelview;
    std::array<f32, 4> input_resolution;
    std::array<f32, 4> output_resolution;
    s32 left_screen{};
    s32 right_screen{};
    s32 layer{};
    s32 reverse_interlaced{};
};
static_assert(sizeof(PresentUniforms) == 112);
static_assert(offsetof(PresentUniforms, left_screen) == 96);

enum class CompositeShader : u8 {
    Screen,
    Anaglyph,
    Interlaced,
    Cursor,
    Normalize,
};

struct ScreenQuad {
    f32 x, y, width, height;
    u32 left, right;
    s32 layer{};
    f32 opacity{1.0f};
};

struct PresentationImage {
    ScreenInfo screen;
    MetalFXSpatial::Image scaled;

    id<MTLTexture> Get() const {
        return scaled ? scaled.Texture() : screen.image.Get();
    }
};

constexpr u32 DisplayBytesPerPixel(Pica::PixelFormat format) {
    switch (format) {
    case Pica::PixelFormat::RGBA8:
        return 4;
    case Pica::PixelFormat::RGB8:
        return 3;
    case Pica::PixelFormat::RGB565:
    case Pica::PixelFormat::RGB5A1:
    case Pica::PixelFormat::RGBA4:
        return 2;
    }
    return 0;
}

PAddr DisplayAddress(const Pica::FramebufferConfig& config, bool right_eye) {
    right_eye &= config.address_right1 != 0 && config.address_right2 != 0;
    if (config.active_fb == 0) {
        return right_eye ? config.address_right1 : config.address_left1;
    }
    return right_eye ? config.address_right2 : config.address_left2;
}

Common::Vec4<u8> DecodeDisplayPixel(Pica::PixelFormat format, const u8* pixel) {
    using namespace Common::Color;
    switch (format) {
    case Pica::PixelFormat::RGBA8:
        return DecodeRGBA8(pixel);
    case Pica::PixelFormat::RGB8:
        return DecodeRGB8(pixel);
    case Pica::PixelFormat::RGB565:
        return DecodeRGB565(pixel);
    case Pica::PixelFormat::RGB5A1:
        return DecodeRGB5A1(pixel);
    case Pica::PixelFormat::RGBA4:
        return DecodeRGBA4(pixel);
    }
    throw std::invalid_argument("Unsupported PICA scanout format");
}

struct DisplayMemory {
    VideoCore::Extent extent;
    u32 stride_pixels;
    u32 bytes;
};

std::optional<DisplayMemory> DescribeDisplay(const Pica::FramebufferConfig& config) {
    const u32 bpp = DisplayBytesPerPixel(config.color_format);
    if (!bpp || !config.stride || config.stride % bpp) {
        return {};
    }
    const u32 stride = config.stride / bpp;
    const u32 width = std::min(config.width.Value(), stride);
    const u32 height = config.height.Value();
    if (!width || !height || width > max_dimension || height > max_dimension) {
        return {};
    }
    const u64 bytes = u64{height - 1} * config.stride + u64{width} * bpp;
    if (bytes > std::numeric_limits<u32>::max()) {
        return {};
    }
    return DisplayMemory{{width, height}, stride, static_cast<u32>(bytes)};
}

CompositeShader ShaderForStereo(StereoMode mode) {
    switch (mode) {
    case StereoMode::Anaglyph:
        return CompositeShader::Anaglyph;
    case StereoMode::Interlaced:
    case StereoMode::ReverseInterlaced:
        return CompositeShader::Interlaced;
    default:
        return CompositeShader::Screen;
    }
}

std::vector<ScreenQuad> ScreenQuads(const Layout::FramebufferLayout& layout, bool swap_screens,
                                    bool swap_eyes, u32 mono_eye) {
    std::vector<ScreenQuad> quads;
    quads.reserve(6);
    const auto append = [&](bool bottom, const Common::Rectangle<u32>& rect, f32 opacity) {
        if ((bottom ? !layout.bottom_screen_enabled : !layout.top_screen_enabled) ||
            rect.right <= rect.left || rect.bottom <= rect.top) {
            return;
        }
        const auto left = bottom ? bottom_screen : u32{swap_eyes};
        const auto right = bottom ? bottom_screen : u32{!swap_eyes};
        const f32 x = static_cast<f32>(rect.left);
        const f32 y = static_cast<f32>(rect.top);
        const f32 w = static_cast<f32>(rect.GetWidth());
        const f32 h = static_cast<f32>(rect.GetHeight());
        const f32 half = static_cast<f32>(layout.width / 2);
        const auto add = [&](f32 draw_x, f32 width, u32 eye_l, u32 eye_r, s32 layer) {
            quads.push_back({draw_x, y, width, h, eye_l, eye_r, layer, opacity});
        };
        switch (layout.render_3d_mode) {
        case StereoMode::Off: {
            const auto eye = bottom ? bottom_screen : std::min(mono_eye, 1u);
            add(x, w, eye, eye, 0);
            break;
        }
        case StereoMode::SideBySide:
            add(x / 2, w / 2, left, left, 0);
            add(x / 2 + half, w / 2, right, right, 1);
            break;
        case StereoMode::SideBySideFull:
            add(x, w, left, left, 0);
            add(x + half, w, right, right, 1);
            break;
        case StereoMode::CardboardVR: {
            const auto shifted = bottom ? layout.cardboard.bottom_screen_right_eye
                                        : layout.cardboard.top_screen_right_eye;
            add(x, w, left, left, 0);
            add(static_cast<f32>(shifted) + half, w, right, right, 1);
            break;
        }
        case StereoMode::Anaglyph:
        case StereoMode::Interlaced:
        case StereoMode::ReverseInterlaced:
            add(x, w, left, right, 0);
            break;
        }
    };
    const auto second_opacity = swap_screens ? layout.top_opacity : layout.bottom_opacity;
    if (swap_screens) {
        append(true, layout.bottom_screen, 1.0f);
        append(false, layout.top_screen, second_opacity);
    } else {
        append(false, layout.top_screen, 1.0f);
        append(true, layout.bottom_screen, second_opacity);
    }
    if (layout.additional_screen_enabled) {
        append(layout.additional_screen_is_bottom, layout.additional_screen, second_opacity);
    }
    return quads;
}

std::array<ScreenVertex, 4> ScreenVertices(const ScreenQuad& quad, const Common::Rectangle<f32>& uv,
                                           Layout::DisplayOrientation orientation) {
    const f32 l = quad.x;
    const f32 t = quad.y;
    const f32 r = l + quad.width;
    const f32 b = t + quad.height;
    using enum Layout::DisplayOrientation;
    switch (orientation) {
    case Landscape:
        return {{{l, t, uv.bottom, uv.left},
                 {r, t, uv.bottom, uv.right},
                 {l, b, uv.top, uv.left},
                 {r, b, uv.top, uv.right}}};
    case Portrait:
        return {{{l, t, uv.bottom, uv.right},
                 {r, t, uv.top, uv.right},
                 {l, b, uv.bottom, uv.left},
                 {r, b, uv.top, uv.left}}};
    case LandscapeFlipped:
        return {{{l, t, uv.top, uv.right},
                 {r, t, uv.top, uv.left},
                 {l, b, uv.bottom, uv.right},
                 {r, b, uv.bottom, uv.left}}};
    case PortraitFlipped:
        return {{{l, t, uv.top, uv.left},
                 {r, t, uv.bottom, uv.left},
                 {l, b, uv.top, uv.right},
                 {r, b, uv.bottom, uv.right}}};
    }
    throw std::invalid_argument("Invalid native display orientation");
}

VideoCore::Extent TextureExtent(const PresentationImage& image) {
    const auto src = image.Get();
    const auto& uv = image.screen.texcoords;
    return {static_cast<u32>(std::lround(src.width * (uv.bottom - uv.top))),
            static_cast<u32>(std::lround(src.height * (uv.right - uv.left)))};
}

PresentUniforms MakeUniforms(VideoCore::Extent target, VideoCore::Extent input,
                             const ScreenQuad& quad, bool rotated, StereoMode mode) {
    const f32 in_w = static_cast<f32>(std::max(input.width, 1u));
    const f32 in_h = static_cast<f32>(std::max(input.height, 1u));
    const f32 out_w = rotated ? quad.height : quad.width;
    const f32 out_h = rotated ? quad.width : quad.height;
    return {
        .modelview = {2.f / target.width, 0, 0, -1, 0, 2.f / target.height, 0, -1, 0, 0, 1, 0, 0, 0,
                      0, 1},
        .input_resolution = {in_w, in_h, 1.f / in_w, 1.f / in_h},
        .output_resolution = {out_w, out_h, 1.f / out_w, 1.f / out_h},
        .left_screen = static_cast<s32>(quad.left),
        .right_screen = static_cast<s32>(quad.right),
        .layer = quad.layer,
        .reverse_interlaced = mode == StereoMode::ReverseInterlaced,
    };
}

using CursorVertices = std::array<std::array<f32, 2>, 12>;

std::optional<CursorVertices> Crosshair(const Layout::FramebufferLayout& layout,
                                        const Frontend::EmuWindow::CursorInfo& cursor) {
    if (!cursor.visible || !layout.bottom_screen_enabled || !layout.width || !layout.height) {
        return {};
    }
    const f32 w = static_cast<f32>(layout.width);
    const f32 h = static_cast<f32>(layout.height);
    const auto& rect = layout.bottom_screen;
    if (rect.right <= rect.left || rect.bottom <= rect.top || !std::isfinite(cursor.projected_x) ||
        !std::isfinite(cursor.projected_y)) {
        return {};
    }
    const f32 x = (rect.left + cursor.projected_x) / w * 2.f - 1.f;
    const f32 y = (rect.top + cursor.projected_y) / h * 2.f - 1.f;
    const f32 radius = rect.GetHeight() / 30.f;
    const f32 rx = radius / w;
    const f32 ry = radius / h;
    const f32 left = rect.left / w * 2.f - 1.f;
    const f32 top = rect.top / h * 2.f - 1.f;
    const f32 right = rect.right / w * 2.f - 1.f;
    const f32 bottom = rect.bottom / h * 2.f - 1.f;
    CursorVertices result{};
    const auto make = [&](u32 index, f32 dx, f32 dy) {
        const f32 l = std::clamp(x - dx, left, right);
        const f32 t = std::clamp(y - dy, top, bottom);
        const f32 r = std::clamp(x + dx, left, right);
        const f32 b = std::clamp(y + dy, top, bottom);
        const std::array<std::array<f32, 2>, 6> vertices{
            {{l, t}, {r, t}, {r, b}, {l, t}, {r, b}, {l, b}}};
        std::ranges::copy(vertices, result.begin() + index);
    };
    make(0, rx / 5.f, ry);
    make(6, rx, ry / 5.f);
    return result;
}

u64 PhysicalBudget(const Instance& instance) {
    const auto recommended = instance.GetCapabilities().recommended_working_set_size;
    const u64 physical = Common::GetMemInfo().total_physical_memory;
    const u64 app_budget = physical - physical / 4;
    if (recommended && app_budget) {
        return std::min(recommended, app_budget);
    }
    return recommended ? recommended : app_budget;
}

PipelineCacheOptions PipelineOptions() {
    return {.directory =
                std::filesystem::path(FileUtil::GetUserPath(FileUtil::UserPath::ShaderDir)) /
                "metal",
            .disk_cache = Settings::values.use_disk_shader_cache.GetValue(),
            .async_shaders = Settings::values.async_shader_compilation.GetValue()};
}

} // Anonymous namespace

class RendererMetal::Impl {
    using ImageArray = std::array<PresentationImage, screen_count>;
    using TextureArray = std::array<std::optional<TextureSurface>, screen_count>;
    using Access = MemoryAllocator::GPUBufferAccess;

    struct DrawCommand {
        DescriptorUpdateQueue::Snapshot vertex;
        std::optional<DescriptorUpdateQueue::Snapshot> fragment;
        std::vector<Access> uploads;
        f32 opacity;
        CompositeShader shader;
    };

public:
    Impl(RendererMetal& renderer, Core::System& system, Pica::PicaCore& pica,
         Frontend::EmuWindow& window, Frontend::EmuWindow* secondary_window)
        : instance(Settings::values.physical_device.GetValue(),
                   Settings::values.use_metalfx_spatial.GetValue()),
          memory(system.Memory()), pica(pica), budget(PhysicalBudget(instance)),
          allocator(instance.GetDevice(), budget), scheduler(instance.GetContext()),
          passes(instance, scheduler, allocator), descriptors(instance.GetContext()),
          compiler(instance.GetContext()), uploads(allocator, scheduler, 512 * 1024),
          display_runtime(instance, scheduler, passes, allocator, budget),
          primary(window, instance, scheduler, budget,
                  Settings::values.async_presentation.GetValue()),
          spatial(instance.GetContext(), scheduler, budget),
          rasterizer(memory, pica, system.CustomTexManager(), renderer, instance, scheduler, passes,
                     descriptors, allocator, budget, PipelineOptions()) {
        if (secondary_window) {
            secondary =
                std::make_unique<PresentWindow>(*secondary_window, instance, scheduler, budget,
                                                Settings::values.async_presentation.GetValue());
        }
        CompilePresentation();
    }

    ~Impl() {
        primary.Close();
        if (secondary) {
            secondary->Close();
        }
    }

    PresentWindow* Presentation(bool is_secondary) {
        return is_secondary ? secondary.get() : &primary;
    }

    RasterizerMetal& Rasterizer() {
        return rasterizer;
    }

    void Swap(VideoCore::RendererSettings& settings, Frontend::EmuWindow& window,
              Frontend::EmuWindow* second) {
        const bool separate = second && Settings::values.layout_option.GetValue() ==
                                            Settings::LayoutOption::SeparateWindows;
        const auto main_snapshot = window.GetNativeSurface();
        const auto second_snapshot =
            separate ? second->GetNativeSurface() : Frontend::EmuWindow::NativeSurfaceSnapshot{};
        const bool layout_changed = main_snapshot.generation != generations[0] ||
                                    second_snapshot.generation != generations[1] ||
                                    separate != was_separate;
        const bool background_changed = settings.bg_color_update_requested.exchange(false);
        const bool display_changed = settings.shader_update_requested.exchange(false);
        const bool wants_spatial = Settings::values.use_metalfx_spatial.GetValue();
        const bool draw_frame = !Settings::values.use_skip_duplicate_frames.GetValue() ||
                                Core::PerfStats::game_frames_updated || layout_changed ||
                                background_changed || display_changed ||
                                wants_spatial != was_spatial || !latest_tick;
        const bool screenshot = settings.screenshot_requested.load(std::memory_order_acquire);
        passes.EndRendering();

        const bool capturing = draw_frame && BeginCapture();
        SCOPE_EXIT({
            if (capturing) {
                [MTLCaptureManager.sharedCaptureManager stopCapture];
            }
        });

        if (draw_frame || screenshot) {
            PrepareScreens();
        }
        if (screenshot) {
            RenderScreenshot(settings, window.GetCursorInfo());
        }
        bool presented = false;
        if (draw_frame) {
            const auto first_layout = ChooseLayout(window, main_snapshot);
            presented = RenderToWindow(primary, first_layout, window.GetCursorInfo());
            if (separate) {
                const auto layout = ChooseLayout(*second, second_snapshot);
                presented =
                    RenderToWindow(*secondary, layout, second->GetCursorInfo()) && presented;
            }
            if (presented) {
                generations = {main_snapshot.generation, second_snapshot.generation};
                was_separate = separate;
                was_spatial = wants_spatial;
                Core::PerfStats::game_frames_updated = false;
            }
        }
        if (!draw_frame || !presented) {
            latest_tick = scheduler.Flush();
        }
        if (separate) {
            second->PollEvents();
        }
        if (capturing) {
            RequireCompletion(primary.WaitPresent(), "complete the primary GPU trace frame");
            if (separate) {
                RequireCompletion(secondary->WaitPresent(),
                                  "complete the secondary GPU trace frame");
            }
        }
    }

    void LimitFramesAhead() {
        if (previous_game_frame_tick) {
            RequireCompletion(scheduler.Wait(previous_game_frame_tick), "limit frames ahead");
        }
        previous_game_frame_tick = latest_tick;
    }

private:
    static Layout::FramebufferLayout ChooseLayout(
        const Frontend::EmuWindow& window,
        const Frontend::EmuWindow::NativeSurfaceSnapshot& snapshot) {
        return snapshot.layout.width && snapshot.layout.height
                   ? snapshot.layout
                   : Layout::FrameLayoutFromResolutionScale(1, window.isSecondary());
    }

    static void RequireCompletion(CompletionStatus status, const char* operation) {
        if (status != CompletionStatus::Complete) {
            throw std::runtime_error(fmt::format("Metal command did not {} ({})", operation,
                                                 static_cast<unsigned>(status)));
        }
    }

    bool BeginCapture() {
        if (capture_attempted || !Settings::values.dump_command_buffers.GetValue()) {
            return false;
        }
        capture_attempted = true;
        if (!instance.CanCaptureTrace() || instance.IsCaptureActive()) {
            LOG_WARNING(Render, "Metal GPU trace capture is unavailable for this session");
            return false;
        }

        const auto log_dir = FileUtil::GetUserPath(FileUtil::UserPath::LogDir);
        const auto timestamp = std::chrono::duration_cast<std::chrono::microseconds>(
                                   std::chrono::system_clock::now().time_since_epoch())
                                   .count();
        const auto path =
            std::filesystem::path(log_dir) / fmt::format("metal-{}.gputrace", timestamp);
        auto descriptor = [MTLCaptureDescriptor new];
        descriptor.captureObject = instance.GetDevice();
        descriptor.destination = MTLCaptureDestinationGPUTraceDocument;
        descriptor.outputURL = [NSURL fileURLWithPath:@(path.c_str())];
        NSError* error = nil;
        if (![MTLCaptureManager.sharedCaptureManager startCaptureWithDescriptor:descriptor
                                                                          error:&error]) {
            LOG_ERROR(Render, "Could not start Metal GPU trace: {}",
                      error.localizedDescription.UTF8String ?: "unknown error");
            return false;
        }
        LOG_INFO(Render, "Recording the first Metal frame at {}", path.string());
        return true;
    }

    MTL4LibraryFunctionDescriptor* Function(std::span<const u8> bytes, const char* name) {
        auto result = compiler.LoadLibrary(bytes, name);
        auto library = result.GetLibrary(instance.GetContext());
        if (!library) {
            throw std::runtime_error(fmt::format("Metal presentation function {}: {}", name,
                                                 result.GetDiagnostic().message));
        }
        auto descriptor = [MTL4LibraryFunctionDescriptor new];
        descriptor.library = library;
        descriptor.name = @(name);
        return descriptor;
    }

    void CompilePresentation() {
        using namespace HostShaders::Metal;
        const auto work = instance.GetContext().AcquireWork();
        const auto native =
            CheckResource(compiler.GetNativeCompiler(work), "Get Metal compositor compiler");
        auto vertex = Function(PRESENT_VERT, "present_vert");
        const std::array<MTL4LibraryFunctionDescriptor*, 5> fragments{
            Function(PRESENT_FRAG, "present_frag"),
            Function(PRESENT_ANAGLYPH_FRAG, "present_anaglyph_frag"),
            Function(PRESENT_INTERLACED_FRAG, "present_interlaced_frag"),
            Function(CURSOR_FRAG, "cursor_frag"),
            Function(PRESENT_FRAG, "present_frag"),
        };
        auto cursor_vertex = Function(CURSOR_VERT, "cursor_vert");
        for (std::size_t i = 0; i < fragments.size(); ++i) {
            const bool cursor = i == static_cast<std::size_t>(CompositeShader::Cursor);
            auto layout = [MTLVertexDescriptor vertexDescriptor];
            layout.attributes[0].format = MTLVertexFormatFloat2;
            layout.attributes[0].bufferIndex = 1;
            layout.layouts[1].stepFunction = MTLVertexStepFunctionPerVertex;
            layout.layouts[1].stride = cursor ? sizeof(f32) * 2 : sizeof(ScreenVertex);
            if (!cursor) {
                layout.attributes[1].format = MTLVertexFormatFloat2;
                layout.attributes[1].offset = offsetof(ScreenVertex, u);
                layout.attributes[1].bufferIndex = 1;
            }
            auto descriptor = [MTL4RenderPipelineDescriptor new];
            descriptor.label = [NSString stringWithFormat:@"Metaliris compositor %zu", i];
            descriptor.vertexFunctionDescriptor = cursor ? cursor_vertex : vertex;
            descriptor.fragmentFunctionDescriptor = fragments[i];
            descriptor.vertexDescriptor = layout;
            descriptor.inputPrimitiveTopology = MTLPrimitiveTopologyClassTriangle;
            descriptor.rasterSampleCount = 1;
            auto attachment = descriptor.colorAttachments[0];
            const bool normalize = i == static_cast<std::size_t>(CompositeShader::Normalize);
            attachment.pixelFormat =
                normalize ? instance.GetFormats().GetTraits(VideoCore::PixelFormat::RGBA8).native
                          : PresentWindow::FrameColorFormat;
            attachment.blendingState = normalize ? MTL4BlendStateDisabled : MTL4BlendStateEnabled;
            attachment.rgbBlendOperation = MTLBlendOperationAdd;
            attachment.sourceRGBBlendFactor =
                cursor ? MTLBlendFactorOneMinusDestinationColor : MTLBlendFactorBlendAlpha;
            attachment.destinationRGBBlendFactor =
                cursor ? MTLBlendFactorOneMinusSourceColor : MTLBlendFactorOneMinusBlendAlpha;
            attachment.alphaBlendOperation = MTLBlendOperationAdd;
            attachment.sourceAlphaBlendFactor =
                cursor ? MTLBlendFactorOne : MTLBlendFactorBlendAlpha;
            attachment.destinationAlphaBlendFactor =
                cursor ? MTLBlendFactorZero : MTLBlendFactorOneMinusBlendAlpha;
            auto guard = instance.GetContext().AcquireSubmission(work);
            if (!guard) {
                throw std::runtime_error("Metal compositor pipeline compilation lost its device");
            }
            NSError* error = nil;
            pipelines[i] = CheckResource([native newRenderPipelineStateWithDescriptor:descriptor
                                                                  compilerTaskOptions:nil
                                                                                error:&error],
                                         "Create Metal compositor pipeline", error);
        }

        for (std::size_t i = 0; i < samplers.size(); ++i) {
            auto descriptor = [MTLSamplerDescriptor new];
            descriptor.sAddressMode = MTLSamplerAddressModeClampToEdge;
            descriptor.tAddressMode = MTLSamplerAddressModeClampToEdge;
            descriptor.minFilter = i ? MTLSamplerMinMagFilterLinear : MTLSamplerMinMagFilterNearest;
            descriptor.magFilter = descriptor.minFilter;
            descriptor.supportArgumentBuffers = YES;
            samplers[i] =
                CheckResource([instance.GetDevice() newSamplerStateWithDescriptor:descriptor],
                              "Create Metal compositor sampler");
        }
    }

    TextureSurface& DisplayTexture(std::optional<TextureSurface>& texture, VideoCore::Extent size) {
        if (!texture || texture->width != size.width || texture->height != size.height) {
            VideoCore::SurfaceParams params;
            params.width = size.width;
            params.height = size.height;
            params.stride = size.width;
            params.pixel_format = VideoCore::PixelFormat::RGBA8;
            params.UpdateParams();
            texture.emplace(display_runtime, params);
        }
        return *texture;
    }

    TextureResource UploadPixels(TextureSurface& surface, VideoCore::Extent extent,
                                 std::span<const u8> pixels) {
        auto staging = display_runtime.FindStaging(static_cast<u32>(pixels.size()), true);
        std::ranges::copy(pixels, staging.mapped.begin());
        surface.Upload(
            {
                .buffer_offset = staging.offset,
                .buffer_size = static_cast<u32>(pixels.size()),
                .texture_rect = {0, extent.height, extent.width, 0},
                .texture_level = 0,
            },
            staging);
        return surface.Resource();
    }

    PresentationImage SolidColor(u32 index, Common::Vec4<u8> rgba) {
        if (!fills[index] || fill_colors[index] != rgba) {
            auto& texture = DisplayTexture(fills[index], {1, 1});
            (void)UploadPixels(texture, {1, 1}, std::span{rgba.AsArray(), 4});
            fill_colors[index] = rgba;
        }
        return {{fills[index]->Resource(), full_uv}, {}};
    }

    PresentationImage LoadScanout(u32 index, const Pica::FramebufferConfig& config) {
        const auto display = DescribeDisplay(config);
        const auto address = DisplayAddress(config, index == 1);
        if (!display || !address ||
            u64{address} + display->bytes > u64{std::numeric_limits<PAddr>::max()} + 1) {
            return SolidColor(index, {0, 0, 0, 255});
        }
        ScreenInfo screen{};
        if (rasterizer.AccelerateDisplay(config, address, display->stride_pixels, screen) &&
            screen.image.Get().sampleCount == 1) {
            return {std::move(screen), {}};
        }

        const auto data = memory.GetPhysicalRef(address);
        if (!data || data.GetSize() < display->bytes) {
            return SolidColor(index, {0, 0, 0, 255});
        }
        rasterizer.FlushRegion(address, display->bytes);
        const auto [width, height] = display->extent;
        const u32 bpp = DisplayBytesPerPixel(config.color_format);
        std::vector<u8> rgba(std::size_t{width} * height * 4);
        for (u32 row = 0; row < height; ++row) {
            const auto* source = data.GetPtr() + std::size_t{row} * config.stride;
            auto* destination = rgba.data() + std::size_t{row} * width * 4;
            for (u32 column = 0; column < width; ++column) {
                const auto color = DecodeDisplayPixel(config.color_format, source + column * bpp);
                std::memcpy(destination + column * 4, color.AsArray(), 4);
            }
        }
        auto& texture = DisplayTexture(fallback[index], display->extent);
        return {{UploadPixels(texture, display->extent, rgba), full_uv}, {}};
    }

    void PrepareScreens() {
        for (u32 i = 0; i < screens.size(); ++i) {
            const bool bottom = i == bottom_screen;
            const auto& fill =
                bottom ? pica.regs_lcd.color_fill_bottom : pica.regs_lcd.color_fill_top;
            screens[i] = fill.is_enabled
                             ? SolidColor(i, {fill.color_r, fill.color_g, fill.color_b, 255})
                             : LoadScanout(i, pica.regs.framebuffer_config[bottom]);
        }
    }

    Access UploadArgument(const DescriptorUpdateQueue::Set& set, u32 slot,
                          std::span<const std::byte> bytes) {
        const auto size = static_cast<u32>(bytes.size());
        auto mapped = uploads.Map(size, 256);
        std::ranges::copy(bytes, mapped.Bytes().begin());
        const auto slice = uploads.Commit(std::move(mapped), size);
        auto access = uploads.AcquireGPUAccess(slice, BufferAccess::Read);
        descriptors.AddBuffer(set, slot, access.GetBuffer(), access.Range().offset, size);
        return access;
    }

    DrawCommand BuildDraw(const ImageArray& images, std::span<const ScreenVertex> vertices,
                          const PresentUniforms& uniforms, CompositeShader shader, f32 opacity,
                          bool linear) {
        const auto vertex = descriptors.CreateSet({.buffers = 2});
        const auto fragment = descriptors.CreateSet({.buffers = 1, .textures = 3, .samplers = 1});
        std::vector<Access> accesses;
        accesses.reserve(3);
        accesses.emplace_back(UploadArgument(vertex, 0, std::as_bytes(std::span{&uniforms, 1})));
        accesses.emplace_back(UploadArgument(vertex, 1, std::as_bytes(vertices)));
        const auto& uniform = accesses.front();
        descriptors.AddBuffer(fragment, 0, uniform.GetBuffer(), uniform.Range().offset,
                              sizeof(uniforms));
        descriptors.AddImageSampler(fragment, 0, 0, images[0].Get(), samplers[linear]);
        for (u32 i = 1; i < images.size(); ++i) {
            descriptors.AddStorageImage(fragment, i, images[i].Get(), BufferAccess::Read);
        }
        return {descriptors.Capture(vertex), descriptors.Capture(fragment), std::move(accesses),
                std::isfinite(opacity) ? std::clamp(opacity, 0.f, 1.f) : 1.f, shader};
    }

    DrawCommand BuildCursor(const CursorVertices& vertices) {
        const auto vertex = descriptors.CreateSet({.buffers = 2});
        std::vector<Access> accesses;
        accesses.emplace_back(UploadArgument(vertex, 1, std::as_bytes(std::span{vertices})));
        return {descriptors.Capture(vertex), {}, std::move(accesses), 1.f, CompositeShader::Cursor};
    }

    void Draw(const RenderPass& pass, ImageArray images, std::vector<DrawCommand> commands,
              std::string_view checkpoint) {
        passes.EndRendering();
        scheduler.Record(
            [checkpoint](EncodingContext& context) { context.Checkpoint(checkpoint); });
        for (const auto& image : images) {
            if (image.scaled) {
                image.scaled.Retain(scheduler);
            }
        }
        passes.BeginRendering(pass);
        passes.RecordDraw([images = std::move(images), commands = std::move(commands),
                           pipelines = pipelines,
                           area = pass.area](EncodingContext& context,
                                             id<MTL4RenderCommandEncoder> encoder) mutable {
            [encoder setCullMode:MTLCullModeNone];
            [encoder setDepthStencilState:nil];
            [encoder setViewport:MTLViewport{0, 0, double(area.width), double(area.height), 0, 1}];
            [encoder setScissorRect:MTLScissorRect{0, 0, area.width, area.height}];
            for (auto& command : commands) {
                const auto pipeline = pipelines[static_cast<std::size_t>(command.shader)];
                [encoder setRenderPipelineState:pipeline];
                command.vertex.BindRender(context, encoder, MTLRenderStageVertex);
                if (command.fragment) {
                    command.fragment->BindRender(context, encoder, MTLRenderStageFragment);
                }
                [encoder setBlendColorRed:0 green:0 blue:0 alpha:command.opacity];
                if (command.shader == CompositeShader::Cursor) {
                    [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:12];
                } else {
                    [encoder drawPrimitives:MTLPrimitiveTypeTriangleStrip
                                vertexStart:0
                                vertexCount:4];
                }
                for (auto& access : command.uploads) {
                    context.Retain(std::move(access));
                }
                context.Retain(pipeline);
            }
            for (const auto& image : images) {
                image.screen.image.Retain(context);
            }
        });
        passes.EndRendering();
    }

    void Normalize(ImageArray& images, u32 index, VideoCore::Extent extent) {
        auto& target = DisplayTexture(normalized[index], extent);
        auto resource = target.Resource();
        const auto image = images[index];
        const auto& uv = image.screen.texcoords;
        const auto w = static_cast<f32>(extent.width);
        const auto h = static_cast<f32>(extent.height);
        const std::array<ScreenVertex, 4> vertices{{{0, 0, uv.top, uv.left},
                                                    {w, 0, uv.bottom, uv.left},
                                                    {0, h, uv.top, uv.right},
                                                    {w, h, uv.bottom, uv.right}}};
        const auto uniforms =
            MakeUniforms(extent, extent, {0, 0, w, h, 0, 0}, false, StereoMode::Off);
        const ImageArray sources{image, image, image};
        std::vector<DrawCommand> commands;
        commands.emplace_back(
            BuildDraw(sources, vertices, uniforms, CompositeShader::Normalize, 1.f, false));
        RenderPass pass{
            .color = {.target = {.texture = resource.Get()},
                      .load_action = MTLLoadActionDontCare,
                      .store_action = MTLStoreActionStore},
            .area = {0, 0, extent.width, extent.height},
            .width = extent.width,
            .height = extent.height,
        };
        scheduler.Record([resource](EncodingContext& context) { resource.Retain(context); });
        Draw(pass, sources, std::move(commands), "Normalize Metal scanout region");
        images[index] = {{std::move(resource), full_uv}, {}};
    }

    u64 PrepareImages(ImageArray& images, const Layout::FramebufferLayout& layout,
                      std::span<const ScreenQuad> quads) {
        std::array<VideoCore::Extent, screen_count> desired{};
        std::array<bool, screen_count> normalize{};
        const bool spatial_enabled = Settings::values.use_metalfx_spatial.GetValue();
        for (const auto& quad : quads) {
            const f32 width = layout.is_rotated ? quad.height : quad.width;
            const f32 height = layout.is_rotated ? quad.width : quad.height;
            const auto extent = VideoCore::Extent{
                static_cast<u32>(std::clamp(std::ceil(width), 1.f, f32{max_dimension})),
                static_cast<u32>(std::clamp(std::ceil(height), 1.f, f32{max_dimension})),
            };
            for (const auto index : {quad.left, quad.right}) {
                desired[index].width = std::max(desired[index].width, extent.width);
                desired[index].height = std::max(desired[index].height, extent.height);
            }
            if (quad.left != quad.right &&
                images[quad.left].screen.texcoords != images[quad.right].screen.texcoords) {
                normalize[quad.left] = normalize[quad.right] = true;
            }
        }

        std::vector<SpatialRequest> requests;
        std::vector<u32> indices;
        for (u32 i = 0; i < images.size(); ++i) {
            const auto extent = TextureExtent(images[i]);
            const auto output = desired[i];
            const bool larger = output.width >= extent.width && output.height >= extent.height &&
                                (output.width > extent.width || output.height > extent.height) &&
                                extent.width && extent.height;
            // Constant LCD fills are exact at every scale.
            const bool upscale =
                spatial_enabled && larger && (extent.width > 1 || extent.height > 1);
            const auto format = images[i].Get().pixelFormat;
            if (normalize[i] || (upscale && (images[i].screen.texcoords != full_uv ||
                                             (format != MTLPixelFormatBGRA8Unorm &&
                                              format != MTLPixelFormatRGBA8Unorm)))) {
                Normalize(images, i, extent);
            }
            if (upscale) {
                requests.push_back({images[i].screen.image, extent, output});
                indices.push_back(i);
            }
        }
        if (requests.empty()) {
            return 0;
        }

        const auto produced = scheduler.Flush();
        auto batch = spatial.Apply(requests, true, produced, instance.GetContext().AcquireWork());
        for (u32 i = 0; i < batch.results.size(); ++i) {
            auto& result = batch.results[i];
            if (result.status == SpatialStatus::Queued) {
                images[indices[i]].scaled = std::move(result.image);
                last_spatial_status[indices[i]] = result.status;
            } else if (last_spatial_status[indices[i]] != result.status) {
                LOG_WARNING(Render, "MetalFX Spatial did not scale screen {} (status {})",
                            indices[i], static_cast<unsigned>(result.status));
                last_spatial_status[indices[i]] = result.status;
            }
        }
        return batch.completion ? batch.completion : produced;
    }

    u64 Compose(const PresentWindow::Frame& frame, const Layout::FramebufferLayout& layout,
                const Frontend::EmuWindow::CursorInfo& cursor) {
        const auto quads =
            ScreenQuads(layout, Settings::values.swap_screen.GetValue(),
                        Settings::values.swap_eyes_3d.GetValue(),
                        static_cast<u32>(Settings::values.mono_render_option.GetValue()));
        auto images = screens;
        const auto dependency = PrepareImages(images, layout, quads);
        const auto orientation = layout.is_rotated ? Layout::DisplayOrientation::Landscape
                                                   : Layout::DisplayOrientation::Portrait;
        const auto shader = ShaderForStereo(layout.render_3d_mode);
        const bool linear = Settings::values.filter_mode.GetValue();
        std::vector<DrawCommand> commands;
        commands.reserve(quads.size() + 1);
        for (const auto& quad : quads) {
            const auto& image = images[quad.left];
            const auto vertices = ScreenVertices(quad, image.screen.texcoords, orientation);
            auto uniforms = MakeUniforms(frame.Extent(), TextureExtent(image), quad,
                                         layout.is_rotated, layout.render_3d_mode);
            commands.emplace_back(
                BuildDraw(images, vertices, uniforms, shader, quad.opacity, linear));
        }
        if (const auto vertices = Crosshair(layout, cursor)) {
            commands.emplace_back(BuildCursor(*vertices));
        }
        const auto clear = MTLClearColorMake(Settings::values.bg_red.GetValue(),
                                             Settings::values.bg_green.GetValue(),
                                             Settings::values.bg_blue.GetValue(), 1.0);
        frame.Retain(scheduler);
        Draw(frame.MakePass(clear), std::move(images), std::move(commands),
             "Compose Metal screens");
        return dependency;
    }

    bool RenderToWindow(PresentWindow& window, const Layout::FramebufferLayout& layout,
                        const Frontend::EmuWindow::CursorInfo& cursor) {
        auto frame = window.GetRenderFrame({layout.width, layout.height});
        if (!frame) {
            return false;
        }
        const auto dependency = Compose(*frame, layout, cursor);
        latest_tick = scheduler.Flush({.wait_for_tick = dependency});
        return window.Present(std::move(*frame), latest_tick);
    }

    void RenderScreenshot(VideoCore::RendererSettings& settings,
                          const Frontend::EmuWindow::CursorInfo& cursor) {
        SCOPE_EXIT({ settings.screenshot_requested.store(false, std::memory_order_release); });
        const auto layout = settings.screenshot_framebuffer_layout;
        if (!settings.screenshot_bits || !settings.screenshot_complete_callback ||
            layout.width > max_dimension || layout.height > max_dimension) {
            LOG_ERROR(Render, "Invalid Metal screenshot destination");
            return;
        }

        try {
            const auto bytes = MakeTextureBufferLayout({layout.width, layout.height}, 1, 1, 4, 256);
            if (!bytes) {
                throw std::invalid_argument("Invalid Metal screenshot dimensions");
            }
            auto frame = primary.GetRenderFrame({layout.width, layout.height});
            if (!frame) {
                throw std::runtime_error("Could not obtain a Metal screenshot frame");
            }
            const auto readback = allocator.AllocateBuffer(bytes->length, MemoryPool::Readback);
            auto access = readback.AcquireGPUAccess({0, bytes->length}, BufferAccess::Write);
            const auto dependency = Compose(*frame, layout, cursor);
            scheduler.Record([texture = frame->Texture(), extent = frame->Extent(),
                              pitch = bytes->row_pitch,
                              access = std::move(access)](EncodingContext& context) mutable {
                auto encoder = CheckResource([context.GetCommandBuffer() computeCommandEncoder],
                                             "Begin Metal screenshot readback");
                const auto buffer = access.GetBuffer();
                const std::array uses{
                    ResourceUse{texture, BufferAccess::Read, MTLStageBlit},
                    ResourceUse{buffer, BufferAccess::Write, MTLStageBlit, access.Range()},
                };
                context.Synchronize(encoder, uses);
                [encoder copyFromTexture:texture
                                 sourceSlice:0
                                 sourceLevel:0
                                sourceOrigin:MTLOriginMake(0, 0, 0)
                                  sourceSize:MTLSizeMake(extent.width, extent.height, 1)
                                    toBuffer:buffer
                           destinationOffset:0
                      destinationBytesPerRow:pitch
                    destinationBytesPerImage:0
                                     options:MTLBlitOptionNone];
                context.Retain(std::move(access));
                context.EndPass(encoder);
            });
            latest_tick = scheduler.Flush({.wait_for_tick = dependency});
            RequireCompletion(scheduler.Wait(latest_tick), "finish screenshot readback");
            const auto map = readback.MapRead({0, bytes->length});
            auto* output = static_cast<std::byte*>(settings.screenshot_bits);
            for (std::size_t row = 0; row < bytes->rows; ++row) {
                std::memcpy(output + row * bytes->row_bytes,
                            map.Bytes().data() + row * bytes->row_pitch, bytes->row_bytes);
            }
            settings.screenshot_complete_callback(false);
        } catch (const std::exception& error) {
            LOG_ERROR(Render, "Metal screenshot failed: {}", error.what());
        }
    }

    Instance instance;
    Memory::MemorySystem& memory;
    Pica::PicaCore& pica;
    MemoryBudget budget;
    MemoryAllocator allocator;
    Scheduler scheduler;
    RenderManager passes;
    DescriptorUpdateQueue descriptors;
    ShaderCompiler compiler;
    StreamBuffer uploads;
    TextureRuntime display_runtime;
    PresentWindow primary;
    std::unique_ptr<PresentWindow> secondary;
    MetalFXSpatial spatial;
    RasterizerMetal rasterizer;
    TextureArray fills;
    TextureArray fallback;
    TextureArray normalized;
    std::array<Common::Vec4<u8>, screen_count> fill_colors{};
    ImageArray screens{};
    std::array<id<MTLRenderPipelineState>, 5> pipelines{};
    std::array<id<MTLSamplerState>, 2> samplers{};
    std::array<SpatialStatus, screen_count> last_spatial_status{};
    std::array<u64, 2> generations{};
    u64 latest_tick{};
    u64 previous_game_frame_tick{};
    bool was_separate{};
    bool was_spatial{};
    bool capture_attempted{};
};

RendererMetal::RendererMetal(Core::System& system, Pica::PicaCore& pica,
                             Frontend::EmuWindow& window, Frontend::EmuWindow* secondary)
    : RendererBase(system, window, secondary),
      impl(std::make_unique<Impl>(*this, system, pica, window, secondary)) {}

RendererMetal::~RendererMetal() = default;

VideoCore::RasterizerInterface* RendererMetal::Rasterizer() {
    return &impl->Rasterizer();
}

void RendererMetal::NotifySurfaceChanged(bool is_secondary) {
    if (auto* window = impl->Presentation(is_secondary)) {
        window->NotifySurfaceChanged();
    }
}

void RendererMetal::SwapBuffers() {
    @autoreleasepool {
        {
            system.perf_stats->StartSwap();
            SCOPE_EXIT({ system.perf_stats->EndSwap(); });
            impl->Swap(settings, render_window, secondary_window);
        }
        impl->Rasterizer().TickFrame();
        EndFrame();
    }
}

void RendererMetal::TryPresent(int timeout_ms, bool is_secondary) {
    if (auto* window = impl->Presentation(is_secondary)) {
        window->NotifySurfaceChanged();
        (void)window->WaitPresent(std::chrono::steady_clock::now() +
                                  std::chrono::milliseconds(std::max(timeout_ms, 0)));
    }
}

void RendererMetal::LimitFramesAheadOfGpu() {
    impl->LimitFramesAhead();
}

} // namespace Metal
