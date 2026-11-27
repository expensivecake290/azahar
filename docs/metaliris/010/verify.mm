#import <Foundation/Foundation.h>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <future>
#include <thread>
#include "video_core/renderer_metal/mtl_common.h"

namespace {
void CheckErrors() {
    auto* object = [NSObject new];
    assert(Metal::CheckResource(object, "retain test object") == object);

    NSError* error = [NSError errorWithDomain:@"MetalirisFixture"
                                        code:-42
                                    userInfo:@{NSLocalizedDescriptionKey : @"Device rejected request"}];
    try {
        (void)Metal::CheckResource(id<MTL4CommandQueue>{}, "Create command queue", error);
        assert(false);
    } catch (const Metal::MetalError& failure) {
        assert(failure.Domain() == "MetalirisFixture");
        assert(failure.Code() == -42);
        assert(std::string_view(failure.what()) ==
               "Create command queue: Device rejected request");
    }

    try {
        Metal::DeviceContext absent(nil);
        assert(false);
    } catch (const Metal::MetalError& failure) {
        assert(failure.Code() == 0);
        assert(failure.Domain().empty());
        assert(std::string_view(failure.what()).find("Create Metal device context") !=
               std::string_view::npos);
    }
    assert(Metal::DeviceContext::WorkLease{}.Generation() == 0);
}

void CheckLeases(id<MTLDevice> device) {
    using Metal::WorkLeaseStatus;
    Metal::DeviceContext first(device);
    Metal::DeviceContext second(device);
    assert(first.GetDevice() == device && first.RegistryID() == device.registryID);
    assert(first.GetLeaseStatus({}) == WorkLeaseStatus::Empty);
    const auto initial = first.AcquireWork();
    assert(initial.Generation() == 1);
    assert(first.GetLeaseStatus(initial) == WorkLeaseStatus::Valid);
    assert(second.GetLeaseStatus(initial) == WorkLeaseStatus::ForeignContext);
    assert(!second.AcquireSubmission(initial));
    assert(first.AdvanceGeneration());
    assert(first.GetLeaseStatus(initial) == WorkLeaseStatus::StaleGeneration);
    assert(!first.AcquireSubmission(initial));
    const auto next = first.AcquireWork();
    assert(next.Generation() == 2);

    auto guard = first.AcquireSubmission(next);
    assert(guard.has_value());
    std::promise<void> started;
    std::promise<void> closed;
    auto started_future = started.get_future();
    auto closed_future = closed.get_future();
    std::thread closing([&] {
        started.set_value();
        first.Close();
        closed.set_value();
    });
    started_future.wait();
    assert(closed_future.wait_for(std::chrono::milliseconds(20)) == std::future_status::timeout);
    guard.reset();
    closing.join();
    assert(first.GetLeaseStatus(next) == WorkLeaseStatus::Closed);
    assert(!first.AcquireSubmission(next));
    assert(first.AcquireWork().Generation() == 0);
    assert(!first.AdvanceGeneration());
    first.Close();
}
} // namespace

int main() {
    @autoreleasepool {
        CheckErrors();
        std::puts("PASS checked ownership, NSError identity, missing device and empty lease");
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            std::puts("PENDING no process-visible Metal device for lease and guarded submission tests");
            return 2;
        }
        CheckLeases(device);
        std::puts("PASS real-device generation, foreign context, submission guard and close");
    }
}
