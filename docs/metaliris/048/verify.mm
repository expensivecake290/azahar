#include <cstdio>
#include "../verify_cursor.h"

int main(int argc, char** argv) {
    @autoreleasepool {
        using namespace VerifyCursor;
        assert(argc == 3);
        static_assert(Expected(0, 0) == Background);
        static_assert(Expected(5, 3) == Pixel{17, 200, 51, 255});
        static_assert(Expected(5, 1) == Pixel{238, 55, 204, 255});
        static_assert(Expected(3, 3) == Pixel{238, 55, 204, 255});
        std::puts("PASS: independent cursor invert, overlap and untouched color oracle");
        if (!MTLCreateSystemDefaultDevice()) {
            std::puts("PENDING: native cursor blend pixels require a process-visible Metal 4 GPU.");
        } else {
            Run(argv[2], argv[1], false);
            std::puts("PASS: exact native cursor invert, overlap and exterior pixels");
        }
    }
}
