#include <array>
#include <barrier>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include "common/file_util.h"
#include "common/logging/backend.h"
#include "core/core.h"
#include "core/frontend/emu_window.h"
#include "video_core/renderer_base.h"

namespace {

class Window final : public Frontend::EmuWindow {
    void PollEvents() override {
        std::abort();
    }
};

class RequestFixture final : public VideoCore::RendererBase {
public:
    explicit RequestFixture(Window& window) : RendererBase(Core::System::GetInstance(), window, nullptr) {}

private:
    VideoCore::RasterizerInterface* Rasterizer() override {
        std::abort();
    }

    void SwapBuffers() override {
        std::abort();
    }

    void TryPresent(int, bool) override {
        std::abort();
    }
};

} // Anonymous namespace

int main(int argc, char** argv) {
    assert(argc == 2);
    FileUtil::SetUserPath(std::string(argv[1]) + "/");
    Common::Log::Initialize();
    Window window;
    RequestFixture renderer(window);
    std::barrier rendezvous{3};
    std::array<int, 2> destinations{};
    std::array<std::jthread, 2> publishers;
    unsigned callback_count{};
    constexpr unsigned iterations = 1024;
    for (unsigned index = 0; index < publishers.size(); ++index) {
        publishers[index] = std::jthread([&, index] {
            for (unsigned i = 0; i < iterations; ++i) {
                rendezvous.arrive_and_wait();
                renderer.RequestScreenshot(&destinations[index],
                                           [&, index](bool invert_y) {
                                               assert(!invert_y);
                                               assert(renderer.IsScreenshotPending());
                                               ++callback_count;
                                               destinations[index] = 1;
                                           },
                                           {.width = 400 + index, .height = 240});
                rendezvous.arrive_and_wait();
            }
        });
    }
    for (unsigned i = 0; i < iterations; ++i) {
        rendezvous.arrive_and_wait();
        rendezvous.arrive_and_wait();
        auto& request = renderer.Settings();
        assert(renderer.IsScreenshotPending());
        const auto index = request.screenshot_framebuffer_layout.width - 400;
        assert(index < destinations.size() && request.screenshot_bits == &destinations[index]);
        request.screenshot_complete_callback(false);
        assert(destinations[index] == 1 && destinations[1 - index] == 0);
        destinations[index] = 0;
        request.screenshot_requested.store(false, std::memory_order_release);
    }
    assert(callback_count == iterations);
    std::puts("PASS 1024 paired concurrent screenshot publications preserve one destination, layout and callback");
}
