// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.txt file included.

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <map>
#include <set>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>
#include <fcntl.h>
#include <fmt/format.h>
#include <sys/file.h>
#include <sys/sysctl.h>
#include <unistd.h>
#include "common/file_util.h"
#include "common/hash.h"
#include "common/scm_rev.h"
#include "common/zstd_compression.h"
#include "video_core/renderer_metal/mtl_shader_disk_cache.h"

namespace Metal {

namespace {

namespace FS = std::filesystem;
namespace Compression = Common::Compression;
using ReadStatus = ShaderCacheReadStatus;
using WriteStatus = ShaderCacheWriteStatus;
using Bytes = std::vector<u8>;

constexpr auto MaxRecordSize = ShaderDiskCache::MaxRecordSize;
constexpr std::size_t MaxArchives = 8;
constexpr std::string_view RecordExtension = ".mtlrecord";
constexpr std::string_view ArchiveExtension = ".mtlarchive";

auto Fields(ShaderCacheIdentity& value) {
    return std::tie(value.title_id, value.content_id, value.source_version,
                    value.device_architecture, value.system_build, value.profile);
}

auto Fields(ShaderCompileOptions& value) {
    return std::tie(value.math, value.optimization, value.preserve_invariance, value.definitions);
}

auto Fields(ShaderSource& value) {
    return std::tie(value.text, value.name, value.preamble, value.options);
}

auto Fields(PipelineFunction& value) {
    return std::tie(value.artifact, value.entry);
}

auto Fields(CachedShaderFunction& value) {
    return std::tie(value.stage, value.function);
}

auto Fields(PipelineShaderStages& value) {
    return std::tie(value.vertex, value.fragment, value.object, value.mesh, value.transform,
                    value.expand);
}

auto Fields(PipelineVertexBinding& value) {
    return std::tie(value.buffer_index, value.stride, value.step_rate, value.step);
}

auto Fields(PipelineVertexAttribute& value) {
    return std::tie(value.location, value.buffer_index, value.offset, value.format,
                    value.guest_format, value.components, value.convert_integer_to_float);
}

auto Fields(PipelineVertexLayout& value) {
    return std::tie(value.bindings, value.attributes);
}

auto Fields(PipelineAttachments& value) {
    return std::tie(value.color, value.depth, value.stencil, value.samples);
}

auto Fields(PipelineRasterState& value) {
    return std::tie(value.guest_topology, value.cull_mode, value.primitive, value.index_type,
                    value.flip_viewport, value.rasterization_enabled);
}

auto Fields(PipelineBlendChannel& value) {
    return std::tie(value.source, value.destination, value.equation, value.shader_emulation);
}

auto Fields(PipelineBlendState& value) {
    return std::tie(value.rgb, value.alpha, value.logic_op, value.write_mask, value.enabled);
}

auto Fields(PipelineDepthStencilState& value) {
    return std::tie(value.depth_compare, value.stencil_compare, value.stencil_fail,
                    value.stencil_pass, value.stencil_depth_fail, value.stencil_read_mask,
                    value.stencil_write_mask, value.depth_test, value.depth_write,
                    value.stencil_test);
}

auto Fields(PipelineFragmentState& value) {
    return std::tie(value.feedback, value.shadow, value.barycentrics, value.writes_color,
                    value.writes_depth);
}

auto Fields(PipelineMeshState& value) {
    return std::tie(value.object_threads, value.mesh_threads, value.max_mesh_threadgroups,
                    value.payload_bytes);
}

auto Fields(GraphicsPipelineKey& value) {
    return std::tie(value.route, value.shaders, value.vertex_layout, value.attachments,
                    value.raster, value.blend, value.depth_stencil, value.fragment, value.mesh);
}

/// The persistent format is independent of C++ layout, padding, Objective-C references and
/// machine endianness. Both directions apply the same field order and bounds.
template <bool Reading>
class RecordCodec {
public:
    explicit RecordCodec(Bytes& buffer) : buffer(buffer) {}

    template <typename... T>
    bool operator()(T&... fields) {
        return (Transfer(fields) && ...);
    }

    bool IsFinished() const {
        return offset == buffer.size();
    }

private:
    template <typename T>
    bool Transfer(T& value) {
        using Type = std::remove_cv_t<T>;
        if constexpr (std::is_same_v<Type, bool>) {
            u8 byte = value;
            if (!Transfer(byte) || byte > 1) {
                return false;
            }
            if constexpr (Reading) {
                value = byte != 0;
            }
            return true;
        } else if constexpr (std::is_integral_v<Type>) {
            using Unsigned = std::make_unsigned_t<Type>;
            if constexpr (Reading) {
                if (buffer.size() - offset < sizeof(Type)) {
                    return false;
                }
                u64 number{};
                for (std::size_t i = 0; i < sizeof(Type); ++i) {
                    number |= u64{buffer[offset++]} << (8 * i);
                }
                value = static_cast<Type>(static_cast<Unsigned>(number));
            } else {
                if (MaxRecordSize - buffer.size() < sizeof(Type)) {
                    return false;
                }
                const u64 number = static_cast<Unsigned>(value);
                for (std::size_t i = 0; i < sizeof(Type); ++i) {
                    buffer.push_back(static_cast<u8>(number >> (8 * i)));
                }
            }
            return true;
        } else if constexpr (std::is_enum_v<Type>) {
            auto number = static_cast<std::underlying_type_t<Type>>(value);
            if (!Transfer(number)) {
                return false;
            }
            if constexpr (Reading) {
                value = static_cast<Type>(number);
            }
            return true;
        } else {
            return std::apply([&](auto&... fields) { return (*this)(fields...); }, Fields(value));
        }
    }

    template <typename T>
    bool TransferSequence(T& value, std::size_t limit) {
        if (value.size() > limit) {
            return false;
        }
        auto length = static_cast<u32>(value.size());
        if (!Transfer(length) || length > limit) {
            return false;
        }
        if constexpr (Reading) {
            if (length > buffer.size() - offset) {
                return false;
            }
            value.resize(length);
        }
        return true;
    }

    template <typename T>
    bool TransferBlob(T& value) {
        if (!TransferSequence(value, MaxRecordSize)) {
            return false;
        }
        if (value.empty()) {
            return true;
        }
        if constexpr (Reading) {
            std::memcpy(value.data(), buffer.data() + offset, value.size());
            offset += value.size();
        } else {
            if (value.size() > MaxRecordSize - buffer.size()) {
                return false;
            }
            const auto* first = reinterpret_cast<const u8*>(value.data());
            buffer.insert(buffer.end(), first, first + value.size());
        }
        return true;
    }

    bool Transfer(std::string& value) {
        return TransferBlob(value);
    }

    bool Transfer(Bytes& value) {
        return TransferBlob(value);
    }

    template <typename T>
    bool Transfer(std::vector<T>& value) {
        constexpr auto limit = std::is_same_v<T, PipelineVertexBinding>
                                   ? PipelineVertexLayout::MaxBindings
                                   : PipelineVertexLayout::MaxAttributes;
        return TransferSequence(value, limit) &&
               std::ranges::all_of(value, [&](T& element) { return Transfer(element); });
    }

    bool Transfer(std::map<std::string, std::string, std::less<>>& value) {
        constexpr u32 MaxDefinitions = 1024;
        if (value.size() > MaxDefinitions) {
            return false;
        }
        auto count = static_cast<u32>(value.size());
        if (!Transfer(count) || count > MaxDefinitions) {
            return false;
        }
        if constexpr (Reading) {
            for (u32 i = 0; i < count; ++i) {
                std::string key;
                std::string text;
                if (!(*this)(key, text) || !value.emplace(std::move(key), std::move(text)).second) {
                    return false;
                }
            }
        } else {
            for (const auto& [name, text] : value) {
                auto copy = name;
                auto definition = text;
                if (!(*this)(copy, definition)) {
                    return false;
                }
            }
        }
        return true;
    }

    template <typename T>
    bool Transfer(std::optional<T>& value) {
        bool present = value.has_value();
        if (!Transfer(present)) {
            return false;
        }
        if constexpr (Reading) {
            if (present) {
                value.emplace();
            }
        }
        return !present || Transfer(*value);
    }

    template <typename A, typename B>
    bool Transfer(std::variant<A, B>& value) {
        auto type = static_cast<u8>(value.index());
        if (!Transfer(type) || type > 1) {
            return false;
        }
        if constexpr (Reading) {
            if (type == 0) {
                value.template emplace<0>();
            } else {
                value.template emplace<1>();
            }
        }
        return std::visit([&](auto& member) { return Transfer(member); }, value);
    }

    Bytes& buffer;
    std::size_t offset{};
};

struct FileVersion {
    u32 format = ShaderDiskCache::FormatVersion;
    u32 pipeline = GraphicsPipelineKey::Version;

    bool operator==(const FileVersion&) const = default;
};

bool ValidFunction(const PipelineFunction& function) {
    if (!IsShaderIdentifier(function.entry)) {
        return false;
    }
    if (const auto* binary = std::get_if<Bytes>(&function.artifact)) {
        return HasMetalLibrarySignature(*binary);
    }

    const auto& source = std::get<ShaderSource>(function.artifact);
    const auto has_nul = [](const std::string& text) {
        return text.find('\0') != std::string::npos;
    };
    return source.text.find_first_not_of(" \t\r\n") != std::string::npos && !has_nul(source.text) &&
           !has_nul(source.name) && !has_nul(source.preamble) &&
           static_cast<u32>(source.options.math) <= static_cast<u32>(ShaderMath::Fast) &&
           static_cast<u32>(source.options.optimization) <=
               static_cast<u32>(ShaderOptimization::Size) &&
           std::ranges::all_of(source.options.definitions, [&](const auto& item) {
               return IsShaderIdentifier(item.first) && !has_nul(item.second);
           });
}

bool ValidRecord(const ShaderCacheRecord& record) {
    if (const auto* shader = std::get_if<CachedShaderFunction>(&record)) {
        return shader->stage <= CachedShaderStage::ComputeExpansion &&
               ValidFunction(shader->function);
    }

    const auto& key = std::get<GraphicsPipelineKey>(record);
    const auto& shaders = key.shaders;
    bool valid_stages = false;
    switch (key.route) {
    case PipelineRoute::Vertex:
        valid_stages = shaders.vertex && !shaders.object && !shaders.mesh && !shaders.transform &&
                       !shaders.expand;
        break;
    case PipelineRoute::Mesh:
        valid_stages = !shaders.vertex && shaders.object && shaders.mesh && shaders.transform &&
                       !shaders.expand;
        break;
    case PipelineRoute::ComputeExpansion:
        valid_stages = shaders.vertex && !shaders.object && !shaders.mesh && shaders.transform &&
                       shaders.expand;
        break;
    case PipelineRoute::Compute:
        valid_stages = !shaders.vertex && !shaders.fragment && !shaders.object && !shaders.mesh &&
                       shaders.transform && !shaders.expand;
        break;
    }
    const auto valid = [](const std::optional<PipelineFunction>& stage) {
        return !stage || ValidFunction(*stage);
    };
    return valid_stages && valid(shaders.vertex) && valid(shaders.fragment) &&
           valid(shaders.object) && valid(shaders.mesh) && valid(shaders.transform) &&
           valid(shaders.expand) && key.attachments.samples != 0;
}

u64 Hash(std::span<const u8> bytes) {
    return Common::ComputeHash64(bytes.data(), bytes.size());
}

Bytes EncodeIdentity(ShaderCacheIdentity identity) {
    Bytes bytes(ShaderDiskCache::Magic.begin(), ShaderDiskCache::Magic.end());
    FileVersion versions;
    if (!RecordCodec<false>{bytes}(versions.format, versions.pipeline, identity)) {
        return {};
    }
    return bytes;
}

struct EncodedRecord {
    u64 hash{};
    Bytes bytes;
};

std::optional<EncodedRecord> EncodeRecord(const ShaderCacheIdentity& identity,
                                          ShaderCacheRecord record) {
    if (!ValidRecord(record)) {
        return std::nullopt;
    }
    Bytes payload;
    auto scope = identity;
    if (!RecordCodec<false>{payload}(scope, record)) {
        return std::nullopt;
    }

    auto hash = Hash(payload);
    auto size = static_cast<u32>(payload.size());
    auto compressed = Compression::CompressDataZSTDDefault(payload);
    bool use_compressed = !compressed.empty() && compressed.size() < payload.size();
    Bytes file(ShaderDiskCache::Magic.begin(), ShaderDiskCache::Magic.end());
    FileVersion version;
    if (!RecordCodec<false>{file}(version.format, version.pipeline, size, hash, use_compressed,
                                  use_compressed ? compressed : payload)) {
        return std::nullopt;
    }
    return EncodedRecord{hash, std::move(file)};
}

struct DecodedRecord {
    ReadStatus status = ReadStatus::Recovered;
    std::optional<ShaderCacheRecord> record;
};

DecodedRecord DecodeRecord(Bytes file, const ShaderCacheIdentity& expected) {
    if (file.size() < ShaderDiskCache::Magic.size() ||
        !std::ranges::equal(std::span(file).first(ShaderDiskCache::Magic.size()),
                            ShaderDiskCache::Magic)) {
        return {};
    }
    file.erase(file.begin(), file.begin() + ShaderDiskCache::Magic.size());
    FileVersion version;
    struct {
        u32 size{};
        u64 checksum{};
        bool compressed{};
    } header;
    RecordCodec<true> reader(file);
    if (!reader(version.format, version.pipeline)) {
        return {};
    }
    if (version != FileVersion{}) {
        return {ReadStatus::Incompatible, {}};
    }

    Bytes body;
    if (!reader(header.size, header.checksum, header.compressed, body) || !reader.IsFinished() ||
        header.size == 0 || header.size > MaxRecordSize) {
        return {};
    }
    if (header.compressed) {
        if (Compression::GetDecompressedSize(body) != header.size) {
            return {};
        }
        body = Compression::DecompressDataZSTD(body);
    }
    if (body.size() != header.size || Hash(body) != header.checksum) {
        return {};
    }

    ShaderCacheIdentity identity;
    ShaderCacheRecord record;
    RecordCodec<true> payload(body);
    if (!payload(identity)) {
        return {};
    }
    if (identity != expected) {
        return {ReadStatus::Incompatible, {}};
    }
    if (!payload(record) || !payload.IsFinished() || !ValidRecord(record)) {
        return {};
    }
    return {ReadStatus::Ready, std::move(record)};
}

bool Fsync(int fd) {
    int result;
    do {
        result = fsync(fd);
    } while (result != 0 && errno == EINTR);
    return result == 0;
}

bool SyncPath(const FS::path& path, bool directory) {
    const int fd =
        open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW | (directory ? O_DIRECTORY : 0));
    if (fd < 0) {
        return false;
    }
    const bool result = Fsync(fd);
    close(fd);
    return result;
}

bool WriteFile(const FS::path& path, std::span<const u8> bytes) {
    FileUtil::IOFile file(path.string(), "wb");
    return file && file.WriteBytes(bytes.data(), bytes.size()) == bytes.size() && file.Flush() &&
           Fsync(file.GetFd()) && file.Close();
}

std::optional<Bytes> ReadFile(const FS::path& path) {
    std::error_code error;
    if (!FS::is_regular_file(FS::symlink_status(path, error)) || error) {
        return std::nullopt;
    }
    FileUtil::IOFile file(path.string(), "rb");
    if (!file) {
        return std::nullopt;
    }
    const u64 size = file.GetSize();
    if (size == 0 || size > MaxRecordSize) {
        return std::nullopt;
    }
    Bytes data(size);
    if (file.ReadBytes(data.data(), data.size()) != data.size()) {
        return std::nullopt;
    }
    return data;
}

class CacheLock {
public:
    explicit CacheLock(const FS::path& root) {
        fd = open((root / ".lock").c_str(), O_WRONLY | O_CREAT | O_CLOEXEC | O_NOFOLLOW, 0600);
        if (fd >= 0) {
            int result;
            do {
                result = flock(fd, LOCK_EX);
            } while (result != 0 && errno == EINTR);
            if (result != 0) {
                close(std::exchange(fd, -1));
            }
        }
    }

    ~CacheLock() {
        if (fd >= 0) {
            close(fd);
        }
    }

    CacheLock(const CacheLock&) = delete;
    CacheLock& operator=(const CacheLock&) = delete;

    explicit operator bool() const {
        return fd >= 0;
    }

private:
    int fd{-1};
};

class StagingDirectory {
public:
    explicit StagingDirectory(const FS::path& parent) {
        auto target = (parent / ".staging-XXXXXXXX").string();
        if (mkdtemp(target.data())) {
            path = std::move(target);
        }
    }

    ~StagingDirectory() {
        if (!path.empty()) {
            std::error_code ignored;
            FS::remove_all(path, ignored);
        }
    }

    StagingDirectory(const StagingDirectory&) = delete;
    StagingDirectory& operator=(const StagingDirectory&) = delete;

    FS::path path;
};

bool IsDirectory(const FS::path& path, std::error_code& error) {
    const auto status = FS::symlink_status(path, error);
    if (error == std::errc::no_such_file_or_directory) {
        error.clear();
        return false;
    }
    if (FS::exists(status) && !FS::is_directory(status) && !error) {
        error = std::make_error_code(std::errc::not_a_directory);
    }
    return !error && FS::is_directory(status);
}

bool EnsureDirectory(const FS::path& path) {
    std::error_code error;
    if (IsDirectory(path, error)) {
        return true;
    }
    if (error) {
        return false;
    }
    const auto parent = path.has_parent_path() ? path.parent_path() : FS::path{"."};
    if (!EnsureDirectory(parent)) {
        return false;
    }
    FS::create_directory(path, error);
    return !error && IsDirectory(path, error) && SyncPath(parent, true);
}

std::vector<FS::path> List(const FS::path& directory, std::string_view extension, bool recursive,
                           std::error_code& error) {
    std::vector<FS::path> paths;
    if (!IsDirectory(directory, error)) {
        return paths;
    }
    if (recursive) {
        for (FS::recursive_directory_iterator it(directory, error), end; !error && it != end;
             it.increment(error)) {
            if (it->path().extension() == extension) {
                paths.push_back(it->path());
            }
        }
    } else {
        for (FS::directory_iterator it(directory, error), end; !error && it != end;
             it.increment(error)) {
            if (it->path().extension() == extension) {
                paths.push_back(it->path());
            }
        }
    }
    std::ranges::sort(paths);
    return paths;
}

void NewestFirst(std::vector<FS::path>& paths) {
    std::vector<std::pair<FS::file_time_type, FS::path>> order;
    order.reserve(paths.size());
    for (auto& path : paths) {
        std::error_code error;
        const auto stamp = FS::last_write_time(path, error);
        order.emplace_back(error ? FS::file_time_type::min() : stamp, std::move(path));
    }
    std::ranges::sort(order, std::greater{});
    for (std::size_t i = 0; i < paths.size(); ++i) {
        paths[i] = std::move(order[i].second);
    }
}

std::string SystemBuild() {
    std::array<char, 128> version{};
    std::size_t size = version.size();
    if (sysctlbyname("kern.osversion", version.data(), &size, nullptr, 0) != 0 || size == 0 ||
        size > version.size()) {
        return {};
    }
    return std::string(version.data(), size - 1);
}

std::string DeviceArchitecture(const DeviceContext& context) {
    const char* name = context.GetDevice().architecture.name.UTF8String;
    return name ? name : "";
}

NSURL* PathURL(const FS::path& path) {
    return [NSURL fileURLWithFileSystemRepresentation:path.c_str()
                                          isDirectory:NO
                                        relativeToURL:nil];
}

} // Anonymous namespace

struct ShaderDiskCache::State {
    State(FS::path root, ShaderCacheIdentity key, bool use_cache)
        : identity(std::move(key)), scope(EncodeIdentity(identity)),
          directory(MakeDirectory(std::move(root), identity.title_id, scope)),
          enabled(use_cache && !directory.empty() && identity.IsValid()) {}

    static FS::path MakeDirectory(FS::path root, u64 title, std::span<const u8> scope) {
        if (root.empty() || scope.empty()) {
            return {};
        }
        return root / fmt::format("{:016x}", title) / fmt::format("{:016x}", Hash(scope));
    }

    bool Matches(const DeviceContext& context) const {
        return identity.system_build == SystemBuild() &&
               identity.device_architecture == DeviceArchitecture(context);
    }

    WriteStatus Change(const ShaderCacheRecord& record, bool remove) const {
        if (!enabled) {
            return WriteStatus::Disabled;
        }
        const auto encoded = EncodeRecord(identity, record);
        if (!encoded) {
            return WriteStatus::InvalidRecord;
        }

        const auto root = directory / "records" / fmt::format("{:02x}", encoded->hash >> 56);
        std::error_code error;
        if (remove && !IsDirectory(root, error)) {
            return error ? WriteStatus::IOError : WriteStatus::Missing;
        }
        if (!remove && !EnsureDirectory(root)) {
            return WriteStatus::IOError;
        }
        CacheLock lock(root);
        if (!lock) {
            return WriteStatus::IOError;
        }
        const auto paths = List(root, RecordExtension, false, error);
        if (error) {
            return WriteStatus::IOError;
        }
        const auto prefix = fmt::format("{:016x}-", encoded->hash);
        FS::path replacement;
        for (const auto& path : paths) {
            if (!path.filename().string().starts_with(prefix)) {
                continue;
            }
            auto bytes = ReadFile(path);
            auto decoded = bytes ? DecodeRecord(std::move(*bytes), identity) : DecodedRecord{};
            if (!decoded.record && decoded.status != ReadStatus::Incompatible &&
                FS::is_regular_file(FS::symlink_status(path, error))) {
                replacement = path;
            } else if (decoded.record && *decoded.record == record) {
                if (!remove) {
                    return WriteStatus::Duplicate;
                }
                const bool deleted = FS::remove(path, error);
                return deleted && !error && SyncPath(root, true) ? WriteStatus::Removed
                                                                 : WriteStatus::IOError;
            }
        }
        if (remove) {
            return WriteStatus::Missing;
        }
        for (std::size_t suffix = 0; replacement.empty(); ++suffix) {
            auto candidate = root / fmt::format("{}{}{}", prefix, suffix, RecordExtension);
            if (!std::binary_search(paths.begin(), paths.end(), candidate)) {
                replacement = std::move(candidate);
            }
        }

        StagingDirectory staging(root);
        if (staging.path.empty()) {
            return WriteStatus::IOError;
        }
        const auto source = staging.path / "record";
        if (!WriteFile(source, encoded->bytes)) {
            return WriteStatus::IOError;
        }
        FS::rename(source, replacement, error);
        return !error && SyncPath(root, true) ? WriteStatus::Saved : WriteStatus::IOError;
    }

    const ShaderCacheIdentity identity;
    const Bytes scope;
    const FS::path directory;
    const bool enabled;
};

ShaderCacheIdentity MakeShaderCacheIdentity(const DeviceContext& context, u64 title,
                                            std::string content_id, std::string profile) {
    @autoreleasepool {
        return {title,
                std::move(content_id),
                fmt::format("Metal4/{}+{}", Common::g_scm_rev, Common::g_shader_cache_version),
                DeviceArchitecture(context),
                SystemBuild(),
                std::move(profile)};
    }
}

ShaderDiskCache::ShaderDiskCache(FS::path directory, ShaderCacheIdentity identity, bool enabled)
    : state(std::make_unique<State>(std::move(directory), std::move(identity), enabled)) {}

ShaderDiskCache::~ShaderDiskCache() = default;

const ShaderCacheIdentity& ShaderDiskCache::Identity() const noexcept {
    return state->identity;
}

bool ShaderDiskCache::IsEnabled() const noexcept {
    return state->enabled;
}

ShaderCacheContents ShaderDiskCache::Load(
    const std::atomic_bool& stop, const VideoCore::DiskResourceLoadCallback& callback) const {
    if (!IsEnabled()) {
        return {ReadStatus::Disabled, {}, 0};
    }
    if (stop) {
        return {ReadStatus::Canceled, {}, 0};
    }

    std::error_code error;
    const auto paths = List(state->directory / "records", RecordExtension, true, error);
    if (error) {
        return {ReadStatus::IOError, {}, 0};
    }
    if (paths.empty()) {
        return {};
    }

    ShaderCacheContents contents{ReadStatus::Ready, {}, 0};
    std::set<ShaderCacheRecord> unique;
    std::size_t incompatible{};
    if (callback) {
        callback(VideoCore::LoadCallbackStage::Preload, 0, paths.size(), "Metal shader cache");
    }

    for (std::size_t i = 0; i < paths.size(); ++i) {
        if (stop) {
            return {ReadStatus::Canceled, {}, 0};
        }
        auto bytes = ReadFile(paths[i]);
        auto result = bytes ? DecodeRecord(std::move(*bytes), Identity()) : DecodedRecord{};
        if (result.record) {
            unique.insert(std::move(*result.record));
        } else {
            ++contents.rejected_records;
            incompatible += result.status == ReadStatus::Incompatible;
        }
        if (callback) {
            callback(VideoCore::LoadCallbackStage::Preload, i + 1, paths.size(),
                     "Metal shader cache");
        }
    }
    if (stop) {
        return {ReadStatus::Canceled, {}, 0};
    }
    if (contents.rejected_records) {
        contents.status =
            incompatible == paths.size() ? ReadStatus::Incompatible : ReadStatus::Recovered;
    }
    contents.records.reserve(unique.size());
    while (!unique.empty()) {
        if (stop) {
            return {ReadStatus::Canceled, {}, 0};
        }
        contents.records.emplace_back(std::move(unique.extract(unique.begin()).value()));
    }
    return contents;
}

ShaderCacheWriteStatus ShaderDiskCache::Store(const ShaderCacheRecord& record) {
    return state->Change(record, false);
}

ShaderCacheWriteStatus ShaderDiskCache::Remove(const ShaderCacheRecord& record) {
    return state->Change(record, true);
}

ShaderCacheArchives ShaderDiskCache::LoadArchives(const DeviceContext& context,
                                                  const std::atomic_bool& stop) const {
    @autoreleasepool {
        if (!IsEnabled()) {
            return {ReadStatus::Disabled, nil, 0};
        }
        if (stop) {
            return {ReadStatus::Canceled, nil, 0};
        }
        if (!state->Matches(context)) {
            return {ReadStatus::Incompatible, nil, 0};
        }

        std::error_code error;
        auto paths = List(state->directory / "archives", ArchiveExtension, false, error);
        if (error) {
            return {ReadStatus::IOError, nil, 0};
        }
        if (paths.empty()) {
            return {};
        }
        NewestFirst(paths);
        const auto work = context.AcquireWork();
        auto archives = [NSMutableArray<id<MTL4Archive>> array];
        std::size_t rejected{};
        std::size_t incompatible{};
        for (const auto& path : paths) {
            if (stop) {
                return {ReadStatus::Canceled, nil, 0};
            }
            if (!IsDirectory(path, error)) {
                ++rejected;
                continue;
            }
            const auto bytes = ReadFile(path / "scope");
            const auto library = path / "pipelines.metallib";
            if (!bytes || !FS::is_regular_file(FS::symlink_status(library, error))) {
                ++rejected;
                continue;
            }
            if (*bytes != state->scope) {
                ++incompatible;
                ++rejected;
                continue;
            }

            auto guard = context.AcquireSubmission(work);
            if (!guard) {
                return {ReadStatus::Canceled, nil, 0};
            }
            auto native = [context.GetDevice() newArchiveWithURL:PathURL(library) error:nil];
            if (native) {
                [archives addObject:native];
                if (archives.count == MaxArchives) {
                    break;
                }
            } else {
                ++rejected;
            }
        }
        if (stop || context.GetLeaseStatus(work) != WorkLeaseStatus::Valid) {
            return {ReadStatus::Canceled, nil, 0};
        }

        ReadStatus status = ReadStatus::Ready;
        if (rejected != 0) {
            status =
                incompatible == paths.size() ? ReadStatus::Incompatible : ReadStatus::Recovered;
        }
        return {status, [archives copy], rejected};
    }
}

ShaderCacheWriteStatus ShaderDiskCache::CommitArchive(
    const DeviceContext& context, id<MTL4PipelineDataSetSerializer> serializer) {
    @autoreleasepool {
        if (!IsEnabled()) {
            return WriteStatus::Disabled;
        }
        if (!serializer || !state->Matches(context)) {
            return WriteStatus::InvalidRecord;
        }

        std::error_code error;
        const auto root = state->directory / "archives";
        if (!EnsureDirectory(root)) {
            return WriteStatus::IOError;
        }
        CacheLock lock(root);
        StagingDirectory staging(root);
        if (!lock || staging.path.empty()) {
            return WriteStatus::IOError;
        }

        const auto library = staging.path / "pipelines.metallib";
        const auto work = context.AcquireWork();
        auto guard = context.AcquireSubmission(work);
        if (!guard || ![serializer serializeAsArchiveAndFlushToURL:PathURL(library) error:nil] ||
            ![context.GetDevice() newArchiveWithURL:PathURL(library) error:nil]) {
            return WriteStatus::IOError;
        }
        if (!SyncPath(library, false) || !WriteFile(staging.path / "scope", state->scope) ||
            !SyncPath(staging.path, true)) {
            return WriteStatus::IOError;
        }
        const auto destination =
            root / (staging.path.filename().string().substr(9) + std::string(ArchiveExtension));
        FS::rename(staging.path, destination, error);
        if (error || !SyncPath(root, true)) {
            return WriteStatus::IOError;
        }

        auto paths = List(root, ArchiveExtension, false, error);
        if (!error && paths.size() > MaxArchives) {
            NewestFirst(paths);
            for (std::size_t i = MaxArchives; i < paths.size(); ++i) {
                FS::remove_all(paths[i], error);
            }
            static_cast<void>(SyncPath(root, true));
        }
        return WriteStatus::Saved;
    }
}

} // namespace Metal
