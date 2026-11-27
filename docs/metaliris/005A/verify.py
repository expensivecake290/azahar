"""Verify the real settings generator and its existing platform output conditions."""

import argparse
from pathlib import Path
import re
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[3]
BASELINE = "c621eafbf316699c55f4c9c0c689d4e49498d360"
OWNER = "CMakeModules/GenerateSettingKeys.cmake"
COMMON = "common/setting_keys.h"
QT = "citra_qt/setting_qkeys.h"
JNI = "android/app/src/main/jni/jni_setting_keys.cpp"
KEY = "use_metalfx_spatial"


def run(*command):
    print("+", subprocess.list2cmdline([str(arg) for arg in command]), flush=True)
    subprocess.run(command, check=True)


def generate(source, build, generator, **options):
    top = source / "CMakeLists.txt"
    top.write_text("cmake_minimum_required(VERSION 3.25)\nproject(KeyFixture NONE)\n"
                   f'include("{generator}")\n')
    run("cmake", "-S", source, "-B", build, *[f"-D{key}={value}" for key, value in options.items()])
    return {name: (build / name).read_text() for name in (COMMON, QT, JNI) if (build / name).is_file()}


def parse_array(header):
    return re.findall(r'"([a-zA-Z_0-9]+)"', header.split("keys_array = {", 1)[1].split("};", 1)[0])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work", type=Path)
    work = parser.parse_args().work.resolve()
    work.mkdir(exist_ok=True)
    source = work / "templates"
    source.mkdir(exist_ok=True)
    for name in (COMMON, QT, JNI):
        path = source / (name + ".in")
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / "src" / (name + ".in"), path)

    baseline = work / "baseline.cmake"
    baseline.write_bytes(subprocess.check_output(["git", "-C", ROOT, "show", f"{BASELINE}:{OWNER}"]))
    for name, options in (
        ("headless", {"ENABLE_QT": "OFF"}),
        ("qt", {"ENABLE_QT": "ON"}),
        ("native", {"ENABLE_QT": "OFF", "ENABLE_METAL": "ON"}),
        ("native-qt", {"ENABLE_QT": "ON", "ENABLE_METAL": "ON"}),
        ("android", {"ANDROID": "ON"}),
        ("libretro", {"ENABLE_LIBRETRO": "ON"}),
    ):
        original = generate(source, work / f"before-{name}", baseline, **options)
        build = work / f"after-{name}"
        updated = generate(source, build, ROOT / OWNER, **options)
        assert set(original) == set(updated)
        assert (QT in updated) == (options.get("ENABLE_QT") == "ON")
        assert (JNI in updated) == (options.get("ANDROID") == "ON")
        keys = parse_array(updated[COMMON])
        assert len(keys) == len(set(keys))
        if options.get("ENABLE_METAL") == "ON":
            assert keys.count(KEY) == 1
            assert parse_array(original[COMMON]) == [key for key in keys if key != KEY]
            for path, data in updated.items():
                new_declarations = data.replace(f"\nDEFINE_KEY({KEY})", "")
                if path == COMMON:
                    # A new last key changes the former last key's trailing comma.
                    new_declarations = re.sub(r"keys_array = \{[^}]*\};", "keys_array = {};",
                                              new_declarations, count=1)
                    expected = re.sub(r"keys_array = \{[^}]*\};", "keys_array = {};",
                                      original[path], count=1)
                else:
                    expected = original[path]
                assert new_declarations == expected
        else:
            assert original == updated
        assert generate(source, build, ROOT / OWNER, **options) == updated
        print("PASS generation", name, len(keys))

    generated = work / "after-native-qt" / "citra_qt"
    fixture = work / "qt_key.cpp"
    fixture.write_text('#include "setting_qkeys.h"\n'
                       'const auto& ReadSpatialKey() { return Settings::QKeys::use_metalfx_spatial; }\n')
    qbase = Path("/opt/homebrew/opt/qtbase/lib")
    run("xcrun", "--sdk", "macosx", "clang++", "-std=c++20", "-fsyntax-only",
        "-F", qbase, "-I", qbase / "QtCore.framework/Headers", "-I", generated, fixture)
    print("PASS Qt compile and all settings key generation variants")


if __name__ == "__main__":
    main()
