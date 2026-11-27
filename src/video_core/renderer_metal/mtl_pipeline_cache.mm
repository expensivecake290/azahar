// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.txt file included.

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <future>
#include <map>
#include <mutex>
#include <shared_mutex>
#include <utility>
#include <fcntl.h>
#include <fmt/format.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>
#include "common/file_util.h"
#include "common/hash.h"
#include "common/logging/log.h"
#include "common/thread_worker.h"
#include "video_core/pica/regs_internal.h"
#include "video_core/pica/shader_setup.h"
#include "video_core/renderer_metal/mtl_pipeline_cache.h"
#include "video_core/renderer_metal/mtl_scheduler.h"

namespace Metal {

namespace {

namespace MSL = Pica::Shader::Generator::MSL;
namespace FS = std::filesystem;
using Pica::Shader::Generator::AttribLoadFlags;
using Pica::Shader::Generator::PicaFixedGSConfig;
using Pica::Shader::Generator::PicaVSConfig;
using Clock = std::chrono::steady_clock;
using namespace std::chrono_literals;

bool StopRequested(const std::atomic_bool* stop) {
    return stop && stop->load();
}

std::string ProfileName(const PipelineDeviceProfile& profile) {
    return fmt::format("metal4/{}{}{}{}", profile.raster_order_groups,
                       profile.fragment_barycentrics, profile.mesh_shaders,
                       profile.texture_atomics);
}

FS::path WarmupMarkerPath(const FS::path& directory, const ShaderCacheIdentity& identity) {
    const auto scope =
        fmt::format("{}\n{}\n{}\n{}\n{}", identity.content_id, identity.source_version,
                    identity.device_architecture, identity.system_build, identity.profile);
    return directory / "warmup" /
           fmt::format("{:016x}-{:016x}.flag", identity.title_id,
                       Common::ComputeHash64(scope.data(), scope.size()));
}

/// An abandoned warmup leaves a byte on disk. An active process holds the advisory lease, so
/// another process using the same title can take the cold route without mistaking it for a crash.
class WarmupMarker {
public:
    WarmupMarker(const FS::path& file, bool read_only) {
        if (!read_only) {
            std::error_code error;
            FS::create_directories(file.parent_path(), error);
            if (error) {
                return;
            }
        }
        fd = open(file.c_str(), O_NOFOLLOW | O_CLOEXEC | (read_only ? O_RDONLY : O_RDWR | O_CREAT),
                  0600);
        if (fd < 0) {
            return;
        }
        const int mode = read_only ? LOCK_SH : LOCK_EX;
        if (flock(fd, mode | LOCK_NB) != 0) {
            close(std::exchange(fd, -1));
            return;
        }
        struct stat metadata{};
        if (fstat(fd, &metadata) != 0 || !S_ISREG(metadata.st_mode)) {
            close(std::exchange(fd, -1));
            return;
        }
        interrupted = metadata.st_size != 0;
        if (!read_only) {
            const char pending = 1;
            if (pwrite(fd, &pending, 1, 0) != 1 || fsync(fd) != 0) {
                close(std::exchange(fd, -1));
                return;
            }
            clear_on_release = true;
        }
    }

    ~WarmupMarker() {
        if (fd >= 0) {
            if (clear_on_release && (ftruncate(fd, 0) != 0 || fsync(fd) != 0)) {
                LOG_WARNING(Render, "Unable to clear Metal pipeline warmup marker");
            }
            close(fd);
        }
    }

    WarmupMarker(const WarmupMarker&) = delete;
    WarmupMarker& operator=(const WarmupMarker&) = delete;

    bool Acquired() const {
        return fd >= 0;
    }

    bool WasInterrupted() const {
        return interrupted;
    }

private:
    int fd{-1};
    bool interrupted{};
    bool clear_on_release{};
};

PipelineFunction Function(std::string code, std::string_view name, std::string_view entry) {
    return {ShaderSource{std::move(code), std::string(name), {}, {}}, std::string(entry)};
}

std::string DecompileFailure(std::string_view stage, const MSL::DecompiledProgram& program) {
    if (program.diagnostics.empty()) {
        return fmt::format("PICA {} generation produced no MSL", stage);
    }
    const auto& first = program.diagnostics.front();
    return fmt::format("PICA {} decompile issue {} at instruction {:x} ({:08x})", stage,
                       static_cast<unsigned>(first.issue), first.offset, first.instruction);
}

bool CopyVertexAttributes(const PipelineVertexLayout& layout, PicaVSConfig& config,
                          MSL::VertexProfile& profile) {
    profile.load_flags.fill(AttribLoadFlags::Float);
    auto& state = config.state;
    if (layout.attributes.size() > state.input_vertex_attributes.size()) {
        return false;
    }
    state.used_input_vertex_attributes = static_cast<u8>(layout.attributes.size());
    for (std::size_t i = 0; i < layout.attributes.size(); ++i) {
        const auto& attribute = layout.attributes[i];
        if (attribute.location >= profile.load_flags.size() || attribute.components == 0 ||
            attribute.components > 4 || attribute.format == MTLVertexFormatInvalid) {
            return false;
        }
        state.input_vertex_attributes[i] = {static_cast<u8>(attribute.location),
                                            static_cast<u8>(attribute.guest_format),
                                            attribute.components};
        if (!attribute.convert_integer_to_float) {
            continue;
        }
        using enum Pica::PipelineRegs::VertexAttributeFormat;
        switch (attribute.guest_format) {
        case BYTE:
        case SHORT:
            profile.load_flags[attribute.location] = AttribLoadFlags::Sint;
            break;
        case UBYTE:
            profile.load_flags[attribute.location] = AttribLoadFlags::Uint;
            break;
        default:
            return false;
        }
    }
    return true;
}

id<MTL4PipelineDataSetSerializer> MakeSerializer(const DeviceContext& context, bool enabled) {
    if (!enabled) {
        return nil;
    }
    auto descriptor = [MTL4PipelineDataSetSerializerDescriptor new];
    descriptor.configuration = MTL4PipelineDataSetSerializerConfigurationCaptureBinaries;
    return [context.GetDevice() newPipelineDataSetSerializerWithDescriptor:descriptor];
}

class TitleCache {
public:
    struct Compilers {
        Compilers(DeviceContext& context, const PipelineCacheOptions& options,
                  id<MTL4PipelineDataSetSerializer> serializer)
            : shaders(context, options.async_shaders ? options.shader_workers : 0, serializer),
              pipelines(context, shaders, options.async_shaders ? options.pipeline_workers : 0,
                        options.max_pipeline_jobs) {}

        void Close() {
            pipelines.Close();
            shaders.Close();
        }

        ShaderCompiler shaders;
        PipelineCompiler pipelines;
    };

    struct Slot {
        std::promise<PipelineBuild> promise;
        std::shared_future<PipelineBuild> future{promise.get_future().share()};
        std::atomic_bool stored{};
    };

    TitleCache(DeviceContext& context, const PipelineCacheOptions& options,
               ShaderCacheIdentity identity, PipelineTitle title)
        : title(std::move(title)), context(context), options(options),
          disk(options.directory, std::move(identity), options.disk_cache),
          serializer(MakeSerializer(context, disk.IsEnabled())),
          main(context, options, serializer) {}

    PipelineBuild Require(const GraphicsPipelineKey& key, bool cacheable,
                          const std::atomic_bool* stop) {
        @autoreleasepool {
            std::shared_lock active(gate);
            if (Canceled(stop)) {
                return {};
            }
            const auto work = context.AcquireWork();
            const u64 generation = work.Generation();
            std::shared_ptr<Slot> slot;
            bool leader;
            {
                std::lock_guard lock(mutex);
                if (device_generation != generation) {
                    memory.clear();
                    device_generation = generation;
                }
                auto [it, inserted] = memory.try_emplace(key);
                if (inserted) {
                    try {
                        it->second = std::make_shared<Slot>();
                    } catch (...) {
                        memory.erase(it);
                        throw;
                    }
                }
                slot = it->second;
                leader = inserted;
            }

            if (leader) {
                try {
                    PipelineBuild result;
                    auto& compiler = SelectCompilers(cacheable);
                    const auto lookup = cacheable ? CurrentArchives() : nil;
                    if (!Canceled(stop)) {
                        if (options.async_shaders) {
                            const auto job = compiler.pipelines.Queue(key, lookup);
                            while (!Canceled(stop)) {
                                auto resolved = job.WaitUntil(Clock::now() + 20ms);
                                if (!resolved) {
                                    continue;
                                }
                                if (resolved->GetStatus(context) ==
                                    PipelineBuildStatus::QueueFull) {
                                    result = compiler.pipelines.Build(key, lookup);
                                } else {
                                    result = std::move(*resolved);
                                }
                                break;
                            }
                        } else {
                            result = compiler.pipelines.Build(key, lookup);
                        }
                    }
                    if (Canceled(stop) || context.GetLeaseStatus(work) != WorkLeaseStatus::Valid) {
                        result = PipelineBuild{};
                    }
                    if (result.GetStatus(context) != PipelineBuildStatus::Success) {
                        EraseSlot(key, slot);
                    } else if (cacheable && serializer) {
                        archive_dirty = true;
                    }
                    slot->promise.set_value(std::move(result));
                } catch (...) {
                    EraseSlot(key, slot);
                    slot->promise.set_exception(std::current_exception());
                }
            }

            while (!Canceled(stop)) {
                if (slot->future.wait_for(20ms) != std::future_status::ready) {
                    continue;
                }
                const auto result = slot->future.get();
                if (context.GetLeaseStatus(work) != WorkLeaseStatus::Valid) {
                    return {};
                }
                if (cacheable && disk.IsEnabled() &&
                    result.GetStatus(context) == PipelineBuildStatus::Success &&
                    !slot->stored.exchange(true)) {
                    QueueStore(key, slot);
                }
                return result;
            }
            return {};
        }
    }

    PipelineWarmup Warmup(const std::atomic_bool& stop,
                          const VideoCore::DiskResourceLoadCallback& callback) {
        PipelineWarmup result;
        if (!disk.IsEnabled()) {
            result.disk_status = ShaderCacheReadStatus::Disabled;
            return result;
        }
        if (Canceled(&stop)) {
            result.canceled = true;
            result.disk_status = ShaderCacheReadStatus::Canceled;
            return result;
        }

        WarmupMarker marker(WarmupMarkerPath(options.directory, disk.Identity()), false);
        if (!marker.Acquired() || marker.WasInterrupted()) {
            result.previous_load_interrupted = marker.WasInterrupted();
            result.disk_status = marker.Acquired() ? ShaderCacheReadStatus::Recovered
                                                   : ShaderCacheReadStatus::IOError;
            return result;
        }

        auto loaded_archives = disk.LoadArchives(context, stop);
        {
            std::lock_guard lock(mutex);
            archives = loaded_archives.archives;
        }
        const auto contents = disk.Load(stop, callback);
        result.disk_status = contents.status;
        if (loaded_archives.status == ShaderCacheReadStatus::Recovered &&
            (contents.status == ShaderCacheReadStatus::Missing ||
             contents.status == ShaderCacheReadStatus::Ready)) {
            result.disk_status = ShaderCacheReadStatus::Recovered;
        }
        const auto count = contents.records.size();
        if (callback && !Canceled(&stop) && count != 0) {
            callback(VideoCore::LoadCallbackStage::Build, 0, count, "Metal shaders and pipelines");
        }

        for (const auto& entry : contents.records) {
            if (Canceled(&stop)) {
                break;
            }
            const bool success = std::holds_alternative<GraphicsPipelineKey>(entry)
                                     ? Require(std::get<GraphicsPipelineKey>(entry), true, &stop)
                                               .GetStatus(context) == PipelineBuildStatus::Success
                                     : Preload(std::get<CachedShaderFunction>(entry), stop);
            if (Canceled(&stop)) {
                break;
            }
            ++result.completed;
            if (!success) {
                ++result.failed;
                const auto removed = disk.Remove(entry);
                if (removed == ShaderCacheWriteStatus::IOError) {
                    LOG_WARNING(Render, "Unable to discard invalid Metal cache record");
                }
            }
            if (callback) {
                callback(VideoCore::LoadCallbackStage::Build, result.completed, count,
                         "Metal shaders and pipelines");
            }
        }
        result.canceled = Canceled(&stop) || contents.status == ShaderCacheReadStatus::Canceled;
        if (result.canceled) {
            result.disk_status = ShaderCacheReadStatus::Canceled;
        }
        return result;
    }

    ShaderCacheWriteStatus Save(bool retiring) {
        if (retiring) {
            retired = true;
            main.Close();
            std::lock_guard lock(mutex);
            if (transient) {
                transient->Close();
            }
        }

        std::unique_lock active(gate);
        main.pipelines.WaitForIdle();
        if (transient) {
            transient->pipelines.WaitForIdle();
        }
        writer.WaitForRequests();
        if (!disk.IsEnabled()) {
            return ShaderCacheWriteStatus::Disabled;
        }
        if (!serializer || !archive_dirty.load()) {
            return ShaderCacheWriteStatus::Missing;
        }
        const auto status = disk.CommitArchive(context, serializer);
        if (status == ShaderCacheWriteStatus::Saved) {
            archive_dirty = false;
        }
        return status;
    }

    const PipelineTitle title;
    /// Written only while the owning cache holds its title-transition mutex.
    bool warmup_finished{};

private:
    bool Canceled(const std::atomic_bool* stop) const {
        return retired.load() || StopRequested(stop);
    }

    Compilers& SelectCompilers(bool cacheable) {
        if (cacheable || !serializer) {
            return main;
        }
        std::lock_guard lock(mutex);
        if (!transient) {
            transient = std::make_unique<Compilers>(context, options, nil);
            if (retired) {
                transient->Close();
            }
        }
        return *transient;
    }

    NSArray<id<MTL4Archive>>* CurrentArchives() {
        std::lock_guard lock(mutex);
        return archives;
    }

    void EraseSlot(const GraphicsPipelineKey& key, const std::shared_ptr<Slot>& slot) {
        std::lock_guard lock(mutex);
        const auto it = memory.find(key);
        if (it != memory.end() && it->second == slot) {
            memory.erase(it);
        }
    }

    void QueueStore(const GraphicsPipelineKey& key, const std::shared_ptr<Slot>& slot) {
        if (pending_writes.fetch_add(1) >= 32) {
            WriteRecipe(key, *slot);
            return;
        }
        try {
            writer.QueueWork([this, key, slot] { WriteRecipe(key, *slot); });
        } catch (...) {
            pending_writes.fetch_sub(1);
            slot->stored = false;
            LOG_WARNING(Render, "Unable to queue Metal pipeline cache write");
        }
    }

    void WriteRecipe(const GraphicsPipelineKey& key, Slot& slot) noexcept {
        try {
            const auto result = disk.Store(key);
            if (result != ShaderCacheWriteStatus::Saved &&
                result != ShaderCacheWriteStatus::Duplicate) {
                slot.stored = false;
                LOG_WARNING(Render, "Unable to persist Metal pipeline cache record");
            }
        } catch (...) {
            slot.stored = false;
        }
        pending_writes.fetch_sub(1);
    }

    bool Preload(const CachedShaderFunction& cached, const std::atomic_bool& stop) {
        std::shared_lock active(gate);
        if (Canceled(&stop)) {
            return false;
        }
        const auto& function = cached.function;
        ShaderLibrary library;
        if (const auto* source = std::get_if<ShaderSource>(&function.artifact)) {
            library = main.shaders.CompileLibrary(*source);
        } else {
            library = main.shaders.LoadLibrary(std::get<std::vector<u8>>(function.artifact), "");
        }
        auto native = library.GetLibrary(context);
        if (!native) {
            return false;
        }
        const auto name = [[NSString alloc] initWithBytes:function.entry.data()
                                                   length:function.entry.size()
                                                 encoding:NSUTF8StringEncoding];
        return [native.functionNames containsObject:name];
    }

    DeviceContext& context;
    const PipelineCacheOptions& options;
    ShaderDiskCache disk;
    const id<MTL4PipelineDataSetSerializer> serializer;
    Compilers main;
    std::unique_ptr<Compilers> transient;
    std::shared_mutex gate;
    std::mutex mutex;
    std::atomic_bool retired{};
    std::atomic_bool archive_dirty{};
    std::atomic_size_t pending_writes{};
    u64 device_generation{};
    NSArray<id<MTL4Archive>>* archives = nil;
    std::map<GraphicsPipelineKey, std::shared_ptr<Slot>> memory;
    Common::ThreadWorker writer{1, "Metaliris shader cache writer"};
};

} // Anonymous namespace

struct PipelineCache::State {
    State(DeviceContext& context, PipelineDeviceProfile profile, PipelineCacheOptions options)
        : context(context), profile(profile), options(std::move(options)),
          profile_name(ProfileName(profile)) {}

    std::shared_ptr<TitleCache> NewTitle(PipelineTitle title) {
        auto identity =
            MakeShaderCacheIdentity(context, title.program_id, title.content_id, profile_name);
        return std::make_shared<TitleCache>(context, options, std::move(identity),
                                            std::move(title));
    }

    std::shared_ptr<TitleCache> Current(bool create) {
        std::lock_guard lock(mutex);
        if (!closed && !current && create) {
            current = NewTitle({});
        }
        return current;
    }

    DeviceContext& context;
    const PipelineDeviceProfile profile;
    const PipelineCacheOptions options;
    const std::string profile_name;
    std::mutex transition;
    std::mutex mutex;
    std::shared_ptr<TitleCache> current;
    bool closed{};
};

PipelineDeviceProfile PipelineDeviceProfile::Detect(const DeviceContext& context) {
    const auto device = context.GetDevice();
    const bool apple_metal4 =
        [device supportsFamily:MTLGPUFamilyMetal4] && [device supportsFamily:MTLGPUFamilyApple7];
    return {.raster_order_groups = static_cast<bool>(device.areRasterOrderGroupsSupported),
            .fragment_barycentrics = static_cast<bool>(device.supportsShaderBarycentricCoordinates),
            .mesh_shaders = apple_metal4,
            .texture_atomics =
                apple_metal4 && device.readWriteTextureSupport != MTLReadWriteTextureTierNone};
}

PipelineCache::PipelineCache(DeviceContext& context, PipelineDeviceProfile profile,
                             PipelineCacheOptions options)
    : state(std::make_unique<State>(context, profile, std::move(options))) {}

PipelineCache::~PipelineCache() {
    Close();
}

PipelineSelection PipelineCache::Select(const PipelineShaderRequest& request) const {
    PipelineSelection result;
    result.key = request.fixed_state;
    auto& key = result.key;
    key.shaders = {};
    const auto& profile = state->profile;
    namespace M = Pica::Shader::Generator::MSL;
    using enum PipelineSelectionStatus;

    const bool use_geometry = request.geometry_setup != nullptr;
    if (use_geometry != request.geometry_config.has_value() ||
        (use_geometry && !request.vertex_setup)) {
        result.status = InvalidGeometryProgram;
        result.diagnostic = "Missing input program or invocation state for PICA geometry";
        return result;
    }

    M::FragmentShader fragment;
    try {
        fragment = M::GenerateFragmentShader(
            Pica::Shader::FSConfig{request.regs}, request.user,
            {request.guest_color, key.attachments.samples, profile.raster_order_groups,
             profile.fragment_barycentrics, profile.texture_atomics});
    } catch (const std::invalid_argument& error) {
        result.status = InvalidFragmentState;
        result.diagnostic = error.what();
        return result;
    }
    result.fragment = fragment.requirements;
    key.fragment = {result.fragment.feedback, result.fragment.shadow_storage,
                    result.fragment.fragment_barycentrics, result.fragment.writes_color,
                    result.fragment.writes_depth};
    key.blend.rgb.shader_emulation = result.fragment.emulated_rgb_blend;
    key.blend.alpha.shader_emulation = result.fragment.emulated_alpha_blend;
    key.shaders.fragment =
        Function(std::move(fragment.source), "Metaliris PICA fragment", "pica_fragment");
    key.mesh = {};
    const bool lighting = !request.regs.lighting.disable;
    const bool correct_lighting = lighting && !profile.fragment_barycentrics;
    const auto geometry =
        SelectPipelineGeometry(key.raster, profile, request.vertex_count, use_geometry, lighting,
                               fragment.requirements.feedback == M::FragmentFeedback::TextureCopy,
                               request.order_independent, request.force_vertex_expansion);

    using enum M::GeometryPath;
    if (!request.vertex_setup) {
        key.route = PipelineRoute::Vertex;
        key.raster.guest_topology = Pica::PipelineRegs::TriangleTopology::List;
        key.raster.primitive = MTLPrimitiveTypeTriangle;
        key.raster.index_type = PipelineIndexType::None;
        key.shaders.vertex = Function(M::GenerateTrivialVertexShader(),
                                      "Metaliris interpreted vertex", M::VertexEntryPoints::Vertex);
    } else {
        PicaVSConfig config{request.regs, *request.vertex_setup};
        M::VertexProfile vertex_profile;
        if (geometry == Vertex &&
            !CopyVertexAttributes(key.vertex_layout, config, vertex_profile)) {
            result.diagnostic = "Invalid PICA vertex layout";
            return result;
        }
        auto vertex = M::GenerateVertexShader(*request.vertex_setup, config, vertex_profile);
        if (!vertex) {
            result.diagnostic = DecompileFailure("vertex", vertex);
            return result;
        }
        if (geometry == Vertex) {
            key.route = PipelineRoute::Vertex;
            key.raster.primitive =
                key.raster.guest_topology == Pica::PipelineRegs::TriangleTopology::Strip
                    ? MTLPrimitiveTypeTriangleStrip
                    : MTLPrimitiveTypeTriangle;
            key.shaders.vertex = Function(std::move(vertex.source), "Metaliris PICA vertex",
                                          M::VertexEntryPoints::Vertex);
        } else {
            key.vertex_layout = {};
            key.raster.index_type = PipelineIndexType::None;
            key.raster.primitive = MTLPrimitiveTypeTriangle;
            key.shaders.transform = Function(std::move(vertex.source), "Metaliris PICA vertex",
                                             M::VertexEntryPoints::TransformVertices);

            std::string program;
            if (use_geometry) {
                auto configuration = *request.geometry_config;
                configuration.correct_quaternions = correct_lighting;
                auto expanded =
                    M::GenerateProgrammableGeometryShader(*request.geometry_setup, configuration);
                if (!expanded) {
                    result.status = InvalidGeometryProgram;
                    result.diagnostic = DecompileFailure("geometry", expanded);
                    return result;
                }
                program = std::move(expanded.source);
            } else {
                const M::FixedGeometryConfig configuration{PicaFixedGSConfig{request.regs}.state,
                                                           key.raster.guest_topology, geometry,
                                                           correct_lighting};
                program = M::GenerateFixedGeometryShader(configuration);
            }
            if (geometry == Mesh) {
                key.route = PipelineRoute::Mesh;
                key.mesh.payload_bytes = 3 * sizeof(M::ExpandedVertex);
                key.shaders.object =
                    Function(program, "Metaliris PICA mesh", M::VertexEntryPoints::Object);
                key.shaders.mesh =
                    Function(std::move(program), "Metaliris PICA mesh", M::VertexEntryPoints::Mesh);
            } else {
                key.route = PipelineRoute::ComputeExpansion;
                key.shaders.expand = Function(std::move(program), "Metaliris PICA geometry",
                                              M::VertexEntryPoints::Expand);
                key.shaders.vertex =
                    Function(M::GenerateExpandedVertexShader(), "Metaliris expanded vertex",
                             M::VertexEntryPoints::ExpandedVertex);
            }
        }
    }
    result.cacheable = request.user.IsCacheable();
    result.status = Ready;
    return result;
}

PipelineBuild PipelineCache::Require(const GraphicsPipelineKey& key, bool cacheable,
                                     const std::atomic_bool* stop) {
    auto session = state->Current(false);
    if (!session) {
        std::lock_guard transition(state->transition);
        session = state->Current(true);
    }
    return session ? session->Require(key, cacheable, stop) : PipelineBuild{};
}

PipelineWarmup PipelineCache::SwitchTitle(PipelineTitle title, const std::atomic_bool& stop,
                                          const VideoCore::DiskResourceLoadCallback& callback) {
    @autoreleasepool {
        std::lock_guard transition(state->transition);
        auto current = state->Current(false);
        if (state->closed || stop) {
            return {.disk_status = ShaderCacheReadStatus::Canceled, .canceled = true};
        }
        auto next = current;
        if (current && current->title == title) {
            if (current->warmup_finished) {
                return {.disk_status = ShaderCacheReadStatus::Ready};
            }
        } else {
            next = state->NewTitle(std::move(title));
            if (current) {
                const auto saved = current->Save(true);
                if (saved == ShaderCacheWriteStatus::IOError) {
                    LOG_WARNING(Render, "Unable to save previous Metal title archive");
                }
            }
            std::lock_guard lock(state->mutex);
            state->current = next;
        }
        if (callback) {
            callback(VideoCore::LoadCallbackStage::Prepare, 0, 0, "");
        }
        auto result = next->Warmup(stop, callback);
        next->warmup_finished =
            !result.canceled && result.disk_status != ShaderCacheReadStatus::IOError;
        if (callback && !result.canceled) {
            callback(VideoCore::LoadCallbackStage::Complete, 0, 0, "");
        }
        return result;
    }
}

PipelineTitle PipelineCache::CurrentTitle() const {
    auto session = state->Current(false);
    return session ? session->title : PipelineTitle{};
}

bool PipelineCache::PreviousLoadInterrupted(const PipelineTitle& title) const {
    if (!state->options.disk_cache || state->options.directory.empty() || !title.IsIdentified()) {
        return false;
    }
    const auto identity = MakeShaderCacheIdentity(state->context, title.program_id,
                                                  title.content_id, state->profile_name);
    if (!identity.IsValid()) {
        return false;
    }
    const WarmupMarker marker(WarmupMarkerPath(state->options.directory, identity), true);
    return marker.Acquired() && marker.WasInterrupted();
}

ShaderCacheWriteStatus PipelineCache::Save() {
    std::lock_guard transition(state->transition);
    auto session = state->Current(false);
    return session ? session->Save(false) : ShaderCacheWriteStatus::Missing;
}

void PipelineCache::Close() {
    std::lock_guard transition(state->transition);
    std::shared_ptr<TitleCache> session;
    {
        std::lock_guard lock(state->mutex);
        state->closed = true;
        session = std::move(state->current);
    }
    if (session && session->Save(true) == ShaderCacheWriteStatus::IOError) {
        LOG_WARNING(Render, "Unable to save Metal pipeline archive");
    }
}

void PipelineCache::Bind(EncodingContext& encoding, const DeviceContext& context,
                         id<MTL4RenderCommandEncoder> encoder, const PipelineBuild& pipeline,
                         const PipelineDrawState& draw,
                         std::span<const PipelineArguments> arguments) {
    if (!encoding.OwnsWork(context.AcquireWork())) {
        throw std::invalid_argument("Metal pipeline encoding context has a different work lease");
    }
    const auto states = pipeline.GetStates(context);
    if (!states || !states->render) {
        throw std::invalid_argument("Metal draw has no complete pipeline for the active title");
    }
    BindGraphicsPipeline(encoder, *states, draw);
    encoding.Retain(states->render);
    encoding.Retain(states->depth_stencil);
    for (const auto& binding : arguments) {
        binding.snapshot.BindRender(encoding, encoder, binding.stages);
    }
}

} // namespace Metal
