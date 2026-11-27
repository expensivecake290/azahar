#include <cassert>
#include <cstdio>
#include <string_view>
#include "video_core/video_core.h"
#include "video_core/renderer_metal/renderer_metal.mm"

namespace {

class Window final : public Frontend::EmuWindow {
public:
    explicit Window(bool secondary = false) : EmuWindow(secondary) {}
    void PollEvents() override {}
};

void VerifyFramebuffer() {
    using namespace Metal;
    Pica::FramebufferConfig config{};
    config.size = 0;
    config.format = 0;
    config.width.Assign(5);
    config.height.Assign(3);
    config.color_format.Assign(Pica::PixelFormat::RGB8);
    config.stride = 24;
    auto description = DescribeDisplay(config);
    assert(description && description->extent.width == 5 && description->extent.height == 3);
    assert(description->stride_pixels == 8 && description->bytes == 63);
    config.width.Assign(20);
    assert(DescribeDisplay(config)->extent.width == 8);
    config.stride = 23;
    assert(!DescribeDisplay(config));
    config.stride = 24;
    config.color_format.Assign(static_cast<Pica::PixelFormat>(7));
    assert(!DescribeDisplay(config));
    config.color_format.Assign(Pica::PixelFormat::RGB565);
    config.stride = 0;
    assert(!DescribeDisplay(config));

    config.address_left1 = 100;
    config.address_left2 = 200;
    config.address_right1 = 300;
    config.address_right2 = 0;
    config.active_fb = 0;
    assert(DisplayAddress(config, true) == 100);
    config.address_right2 = 400;
    assert(DisplayAddress(config, true) == 300);
    config.active_fb = 1;
    assert(DisplayAddress(config, true) == 400 && DisplayAddress(config, false) == 200);

    const std::array<u8, 4> rgba{0xab, 0x62, 0x39, 0x15};
    assert((DecodeDisplayPixel(Pica::PixelFormat::RGBA8, rgba.data()) ==
            Common::Vec4<u8>{0x15, 0x39, 0x62, 0xab}));
    assert((DecodeDisplayPixel(Pica::PixelFormat::RGB8, rgba.data()) ==
            Common::Vec4<u8>{0x39, 0x62, 0xab, 255}));
    const std::array<u8, 2> red_565{0x00, 0xf8};
    const std::array<u8, 2> green_5551{0xc1, 0x07};
    const std::array<u8, 2> blue_4444{0xff, 0x00};
    assert((DecodeDisplayPixel(Pica::PixelFormat::RGB565, red_565.data()) ==
            Common::Vec4<u8>{255, 0, 0, 255}));
    assert((DecodeDisplayPixel(Pica::PixelFormat::RGB5A1, green_5551.data()) ==
            Common::Vec4<u8>{0, 255, 0, 255}));
    assert((DecodeDisplayPixel(Pica::PixelFormat::RGBA4, blue_4444.data()) ==
            Common::Vec4<u8>{0, 0, 255, 255}));
    const auto screenshot = MakeTextureBufferLayout({337, 11}, 1, 1, 4, 256);
    assert(screenshot && screenshot->row_bytes == 1348 && screenshot->row_pitch == 1536);
    assert(screenshot->length == 16896);
    std::puts("PASS scanout formats, active buffers, missing right eye, checked bounds and odd screenshot rows");
}

void VerifyLayout() {
    using namespace Metal;
    Layout::FramebufferLayout layout{};
    layout.width = 800;
    layout.height = 500;
    layout.top_screen_enabled = true;
    layout.bottom_screen_enabled = true;
    layout.top_screen = {100, 10, 500, 250};
    layout.bottom_screen = {140, 250, 460, 490};
    layout.bottom_opacity = 0.5f;
    layout.top_opacity = 0.7f;
    layout.render_3d_mode = StereoMode::Off;
    auto items = ScreenQuads(layout, false, false, 1);
    assert(items.size() == 2 && items[0].left == 1 && items[1].left == bottom_screen);
    assert(items[0].opacity == 1.f && items[1].opacity == 0.5f);

    layout.render_3d_mode = StereoMode::SideBySide;
    items = ScreenQuads(layout, false, true, 0);
    assert(items.size() == 4 && items[0].x == 50 && items[1].x == 450);
    assert(items[0].width == 200 && items[0].left == 1 && items[1].left == 0);
    assert(items[0].layer == 0 && items[1].layer == 1);
    assert(items[2].left == bottom_screen && items[3].left == bottom_screen);

    layout.render_3d_mode = StereoMode::SideBySideFull;
    items = ScreenQuads(layout, false, false, 0);
    assert(items.size() == 4 && items[1].x == 500 && items[1].width == 400);
    layout.render_3d_mode = StereoMode::CardboardVR;
    layout.cardboard = {123, 231, 0};
    items = ScreenQuads(layout, false, false, 0);
    assert(items[1].x == 523 && items[3].x == 631);

    layout.additional_screen_enabled = true;
    layout.additional_screen = {5, 5, 105, 65};
    const std::array modes{StereoMode::Anaglyph, StereoMode::Interlaced, StereoMode::ReverseInterlaced};
    for (const auto mode : modes) {
        layout.render_3d_mode = mode;
        items = ScreenQuads(layout, true, true, 0);
        assert(items.size() == 3);
        assert(items[0].right == bottom_screen && items[0].opacity == 1.f);
        assert(items[1].left == 1 && items[1].right == 0 && items[1].opacity == 0.7f);
        assert(items[2].left == 1 && items[2].right == 0);
        const auto uniforms = MakeUniforms({800, 500}, {240, 400}, items[1], true, mode);
        assert(uniforms.left_screen == 1 && uniforms.right_screen == 0);
        assert(uniforms.reverse_interlaced == (mode == StereoMode::ReverseInterlaced));
        assert(uniforms.output_resolution[0] == 240 && uniforms.output_resolution[1] == 400);
        assert(ShaderForStereo(mode) != CompositeShader::Screen);
    }
    const auto uv = Common::Rectangle<float>{0.2f, 0.1f, 0.8f, 0.9f};
    using enum Layout::DisplayOrientation;
    const auto landscape = ScreenVertices(items[1], uv, Landscape);
    const auto portrait = ScreenVertices(items[1], uv, Portrait);
    const auto flipped = ScreenVertices(items[1], uv, LandscapeFlipped);
    const auto rotated = ScreenVertices(items[1], uv, PortraitFlipped);
    assert(landscape[0].u == 0.9f && landscape[0].v == 0.2f);
    assert(portrait[0].u == 0.9f && portrait[0].v == 0.8f);
    assert(flipped[0].u == 0.1f && flipped[0].v == 0.8f);
    assert(rotated[0].u == 0.1f && rotated[0].v == 0.2f);
    const auto cursor = Crosshair(layout, {.visible = true, .projected_x = 319, .projected_y = 239});
    assert(cursor);
    const float right = 2.f * layout.bottom_screen.right / layout.width - 1.f;
    for (const auto point : *cursor) {
        assert(point[0] <= right && point[1] <= 2.f * layout.bottom_screen.bottom / layout.height - 1.f);
    }
    assert(!Crosshair(layout, {.visible = false}));
    assert(!Crosshair(layout, {.visible = true, .projected_x = std::numeric_limits<float>::quiet_NaN()}));
    layout.bottom_screen_enabled = false;
    assert(!Crosshair(layout, {.visible = true}));
    assert(ScreenQuads(layout, true, true, 0).size() == 2);
    std::puts("PASS screen layout ordering, all stereo modes, four orientations, opacity and clipped cursor");
}

void VerifyNativeFactory() {
    const auto devices = Metal::EnumerateDevices();
    auto& system = Core::System::GetInstance();
    Memory::MemorySystem memory(system);
    Pica::PicaCore pica(memory, nullptr);
    Window main;
    Window second(true);
    Settings::values.graphics_api.SetValue(Settings::GraphicsAPI::Metal);
    Settings::values.physical_device.SetValue(std::numeric_limits<u32>::max());
    try {
        (void)VideoCore::CreateRenderer(main, &second, pica, system);
        assert(false);
    } catch (const std::runtime_error& error) {
        assert(devices.empty());
        assert(std::string_view(error.what()) == "No Metal devices are available to this process");
    } catch (const std::out_of_range& error) {
        assert(!devices.empty());
        assert(std::string_view(error.what()).find("Metal device index") != std::string_view::npos);
    }
    std::printf("PASS linked native factory rejects unavailable or invalid Metal device (%zu visible)\n",
                devices.size());
    if (devices.empty()) {
        std::puts("PENDING Metal4 device, native pixel readback, MetalFX Spatial, drawable and performance");
    }
}

} // Anonymous namespace

int main() {
    VerifyFramebuffer();
    VerifyLayout();
    VerifyNativeFactory();
}
