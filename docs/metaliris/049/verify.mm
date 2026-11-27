#include <cstdio>
#include "../verify_cursor.h"

int main(int argc, char** argv) {
    @autoreleasepool {
        using namespace VerifyCursor;
        assert(argc == 3);
        constexpr auto verts = Vertices();
        static_assert(sizeof(Position) == 8 && verts.size() == 15);
        static_assert(GuestPosition(0, 0) == Position{-1, -1});
        static_assert(GuestPosition(Width, Height) == Position{1, 1});
        static_assert(verts[3] == GuestPosition(Vertical.x, Vertical.y));
        static_assert(verts[8] == GuestPosition(Vertical.x, Vertical.y + Vertical.height));
        static_assert(verts[9] == GuestPosition(Horizontal.x, Horizontal.y));
        static_assert(Expected(5, 3) == Pixel{17, 200, 51, 255});
        std::puts("PASS: independent cursor position, packed float2 stream and order oracle");
        if (!MTLCreateSystemDefaultDevice()) {
            std::puts("PENDING: native cursor vertex and blend integration require a Metal 4 GPU.");
        } else {
            Run(argv[1], argv[2], true);
            std::puts("PASS: exact native cursor vertex/blend at nonzero vertex start");
        }
    }
}
