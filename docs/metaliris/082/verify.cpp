#include "citra_qt/util/metal_util.h"
#include <QString>
#include <cassert>
#include <cstdio>
#include <vector>

#ifdef DETERMINISTIC_DEVICES
std::vector<MetalUtil::DeviceInfo> MetalUtil::EnumerateDevices() {
  return {
      {"Apple M5", true, true},
      {"試作", false, false},
      {std::string{"A\0B", 3}, true, false},
      {"Apple M5", true, false},
      {"", false, false},
  };
}
#endif

int main() {
  const auto devices = MetalUtil::EnumerateDevices();
  const auto labels = GetMetalPhysicalDevices();
  assert(devices.size() == labels.size());
#ifdef DETERMINISTIC_DEVICES
  assert(labels.at(0) == QStringLiteral("Apple M5"));
  assert(labels.at(1) == QStringLiteral("\u8a66\u4f5c"));
  assert(labels.at(2) == QStringLiteral("A\0B"));
  assert(labels.at(3) == labels.at(0));
  assert(labels.at(4).isEmpty());
#endif
  for (size_t i = 0; i < devices.size(); ++i) {
    assert(labels[i].toUtf8().toStdString() == devices[i].name);
  }
  std::printf("PASS Qt native adapter list preserves %zu device indices\n",
              labels.size());
}
