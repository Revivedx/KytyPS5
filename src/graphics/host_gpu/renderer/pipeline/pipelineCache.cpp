#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/host_gpu/pipelineStats.h"

#include "common/assert.h"
#include "common/emulatorConfig.h"
#include "common/file.h"
#include "common/liveSwitches.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "common/threads.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/host_gpu/renderer/colorRenderTarget.h"
#include "graphics/host_gpu/renderer/debug.h"
#include "graphics/host_gpu/regionManager.h"
#include "graphics/host_gpu/renderer/depthRenderTarget.h"
#include "graphics/host_gpu/renderer/image/imageView.h"
#include "graphics/host_gpu/renderer/pipeline/blendMapping.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/sync.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/shader/recompiler/ShaderRecompiler.h"
#include "graphics/shader/recompiler/backend/spirv/SpirvEmitter.h"
#include "graphics/shader/recompiler/frontend/decode/ShaderFunctions.h"
#include "graphics/shader/recompiler/ir/passes/SrtNative.h"
#include "graphics/shader/shaderCompiler.h"
#include "kernel/memory.h"
#include "kytyGitVersion.h"
#include "loader/systemContent.h"

#include <cmath>
#include <algorithm>
#include <array>
#include <memory>
#include <type_traits>
#include <optional>
#include <bit>
#include <unordered_map>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <cctype>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fmt/format.h>
#include <limits>
#include <deque>
#include <mutex>
#include <shared_mutex>
#include <span>
#include <string>
#include <spirv-tools/libspirv.hpp>
#include <string_view>
#include <tuple>
#include <thread>
#include <pthread.h>
#include <unordered_set>
#include <utility>
#include <vector>
#include <xxhash.h>

namespace Libs::Graphics {

bool ShaderFailureNonFatal() {
	return true;
}

namespace {

vk::PolygonMode ResolvePolygonMode(const HW::ModeControl& mode, bool cull_front, bool cull_back) {
	// CxPrimitiveSetup::PolygonMode disables both per-face modes when it is zero.
	if (mode.poly_mode == 0) {
		return vk::PolygonMode::eFill;
	}
	EXIT_NOT_IMPLEMENTED(mode.poly_mode != 1);
	if (cull_front && cull_back) {
		return vk::PolygonMode::eFill;
	}
	if (!cull_front && !cull_back && mode.polymode_front_ptype != mode.polymode_back_ptype) {
		EXIT("Pipeline: different polygon modes for two visible faces are unsupported\n");
	}
	// Vulkan has one polygon mode. A culled face does not constrain that mode.
	const auto polygon_mode = cull_front ? mode.polymode_back_ptype : mode.polymode_front_ptype;
	switch (polygon_mode) {
		case 0: return vk::PolygonMode::ePoint;
		case 1: return vk::PolygonMode::eLine;
		case 2: return vk::PolygonMode::eFill;
		default: EXIT("Pipeline: invalid polygon mode %u\n", polygon_mode);
	}
}

std::string DriverCacheSignature(const vk::PhysicalDeviceProperties& properties) {
	constexpr char hex[] = "0123456789abcdef";
	std::string    uuid(VK_UUID_SIZE * 2, '0');
	for (size_t i = 0; i < VK_UUID_SIZE; i++) {
		uuid[i * 2]     = hex[properties.pipelineCacheUUID[i] >> 4u];
		uuid[i * 2 + 1] = hex[properties.pipelineCacheUUID[i] & 0xfu];
	}
	return fmt::format("KytyPC1:{}:{:08x}:{:08x}:{:08x}:{}\n", KYTY_GIT_REVISION,
	                   properties.vendorID, properties.deviceID, properties.driverVersion, uuid);
}

// KYTY_PIPELINE_CACHE_ANY_REVISION=1 keeps a cache written by another emulator revision, so
// A/B builds of one title don't start cold. The device, driver and UUID must still match, and
// the driver validates every entry itself; entries for pipelines a build no longer creates are
// simply never looked up.
bool DriverCacheSignatureMatches(std::string_view cached, std::string_view current) {
	if (cached == current) {
		return true;
	}
	// On by default in this build (updates and the launcher keep the warm cache);
	// KYTY_PIPELINE_CACHE_ANY_REVISION=0 turns it off.
	static const bool any_revision = [] {
		const char* value = std::getenv("KYTY_PIPELINE_CACHE_ANY_REVISION");
		return value == nullptr || std::strcmp(value, "0") != 0;
	}();
	if (!any_revision) {
		return false;
	}
	// "KytyPC1:<revision>:<vendor>:..." -> ":<vendor>:..."
	const auto without_revision = [](std::string_view signature) -> std::string_view {
		const auto first = signature.find(':');
		const auto second =
		    first == std::string_view::npos ? first : signature.find(':', first + 1);
		return second == std::string_view::npos ? std::string_view {} : signature.substr(second);
	};
	const auto cached_rest = without_revision(cached);
	return !cached_rest.empty() && cached.starts_with("KytyPC1:") &&
	       cached_rest == without_revision(current);
}

std::string PipelineCacheTitleId() {
	std::string title_id;
	if ((!Loader::SystemContentParamSfoGetString("TITLE_ID", &title_id) || title_id.empty()) &&
	    (!Loader::SystemContentParamSfoGetString("CONTENT_ID", &title_id) || title_id.empty())) {
		return {};
	}
	if (!std::ranges::all_of(title_id, [](unsigned char c) {
		    return std::isalnum(c) != 0 || c == '-' || c == '_';
	    })) {
		return {};
	}
	return title_id;
}

template <typename... Args>
void PipelineCacheLog(fmt::format_string<Args...> format, Args&&... args) {
	auto message = fmt::format(format, std::forward<Args>(args)...);
	message += '\n';
	Log::WriteToConsoleAndLog(message);
}

bool UsesShaderClock(const ShaderRecompiler::IR::Program& program) {
	for (auto* block: program.blocks) {
		for (const auto& inst: *block) {
			if (inst.GetOpcode() == ShaderRecompiler::IR::ValueOpcode::ReadClockRealtime64) {
				return true;
			}
		}
	}
	return false;
}

// KYTY_LOCAL_HACK KYTY_PARALLEL_MATERIALIZE: set on the materialize worker thread. Its reads use the
// caches' concurrent GPU-clean queries (the GPU thread's own ones refuse other threads), and a read
// that is not GPU-clean is refused instead of faulting the page in (a fault there would ask the GPU
// thread, which waits for the worker): the stage is then materialized again on the CP.
thread_local bool t_parallel_worker  = false;
thread_local bool t_parallel_refused = false;
thread_local bool t_materialize_evaluated = false; // the last Materialize evaluated (no memo hit)
std::atomic<uint64_t> g_worker_faults {0};
std::atomic<uint64_t> g_la_log_bad {0};     // KYTY_LOOKAHEAD=3 diagnostics
std::atomic<uint64_t> g_la_log_checked {0}; // faults handled on the worker (must stay 0)
uint64_t              g_cp_pre_faults = 0; // faults during the CP's last pixel pre-materialize
// KYTY_LOCAL_HACK research KYTY_SERVE_CENSUS (live, default 0): Materialize time by stage and outcome
// (memo hit, miss whose reads were all GPU-clean = servable by a clean-only reader ahead of the CP,
// miss that needed a non-clean read). Rows: pixel, vertex/mesh, compute, any stage on the worker.
thread_local bool t_walk_unclean = false;
thread_local int  t_census_row   = 1;
struct ServeCensus {
	std::array<std::array<std::atomic<uint64_t>, 3>, 4> ns {};
	std::array<std::array<std::atomic<uint64_t>, 3>, 4> calls {};
};
ServeCensus g_serve_census;
bool ServeCensusOn() {
	static auto& on = Common::LiveSwitches::Get("KYTY_SERVE_CENSUS", 0);
	return on.load(std::memory_order_relaxed) != 0;
}
int CensusRow(Libs::Graphics::ShaderType stage) {
	return stage == Libs::Graphics::ShaderType::Pixel     ? 0
	       : stage == Libs::Graphics::ShaderType::Compute ? 2
	                                                      : 1;
}

bool CleanGuestRead(uint64_t address, void* data, uint64_t size) {
	return t_parallel_worker
	           ? Libs::LibKernel::Memory::TryReadGpuCleanBackingConcurrent(address, data, size)
	           : Libs::LibKernel::Memory::TryReadGpuCleanBacking(address, data, size);
}

// One persistent worker thread running one job at a time for the CP (KYTY_PARALLEL_MATERIALIZE).
// It spins for a while between jobs (draws come every few microseconds), then sleeps.
class MaterializeWorker {
public:
	MaterializeWorker() {
		std::thread([this] { Loop(); }).detach(); // lives as long as the process
	}
	template <typename F>
	void Start(F& job) {
		m_job  = [](void* data) { (*static_cast<F*>(data))(); };
		m_data = &job;
		m_seq.fetch_add(1, std::memory_order_acq_rel);
		m_seq.notify_one();
	}
	void Wait() {
		const auto seq = m_seq.load(std::memory_order_relaxed);
		while (m_done.load(std::memory_order_acquire) != seq) {
			__builtin_ia32_pause();
		}
	}

private:
	void Loop() {
		pthread_setname_np(pthread_self(), "MaterializeWkr");
		t_parallel_worker = true;
		// KYTY_PARALLEL_WORKER_NO_TRACE=1 (env, diagnostics): the worker evaluates without traces.
		if (const char* v = std::getenv("KYTY_PARALLEL_WORKER_NO_TRACE"); v != nullptr && v[0] == '1') {
			ShaderRecompiler::IR::SrtTraceSession::Suppressed() = true;
		}
		uint32_t last = 0;
		for (;;) {
			uint32_t seq = m_seq.load(std::memory_order_acquire);
			for (uint32_t spin = 0; seq == last && spin < 200000u; spin++) {
				__builtin_ia32_pause();
				seq = m_seq.load(std::memory_order_acquire);
			}
			if (seq == last) {
				m_seq.wait(last, std::memory_order_acquire);
				continue;
			}
			m_job(m_data);
			last = seq;
			m_done.store(seq, std::memory_order_release);
		}
	}

	std::atomic<uint32_t> m_seq {0};
	std::atomic<uint32_t> m_done {0};
	void (*m_job)(void*) = nullptr;
	void* m_data         = nullptr;
};

const uint8_t* CleanGuestBacking(uint64_t address, uint64_t size) {
	return t_parallel_worker ? Libs::LibKernel::Memory::FindGpuCleanBackingConcurrent(address, size)
	                         : Libs::LibKernel::Memory::FindGpuCleanBacking(address, size);
}

// The resource walker reads guest memory one dword per scalar load, and every read pays the
// GPU-ownership checks and the address-space lock. KYTY_SHADER_READ_CHUNKS (a live switch):
// 1 reads each aligned 256-byte chunk once per program lookup and serves its dwords from that
// copy; 2 (default) looks up each 4 KiB page's backing bytes once per lookup (FindGpuCleanBacking,
// after Senaxx 820f72e5) and copies every later read of the page straight from them. A chunk or
// page with any GPU-owned byte, or one that is not all mapped, is refused and its reads take the
// per-read path, so the values are the ones read before. Nothing marks a page GPU-written during
// one synchronous lookup on the GPU thread; the verdicts live for one lookup only.
// The same object can log every read and its result (KYTY_RESOURCE_MEMO, ProgramCache).
class ShaderReadChunks {
public:
	struct LoggedRead {
		uint64_t address = 0;
		uint32_t first   = 0; // index of the first word in ReadLog::words
		uint32_t count   = 0;
		bool     ok      = false;
	};
	struct ReadLog {
		std::vector<LoggedRead> reads;
		std::vector<uint32_t>   words;
	};

	explicit ShaderReadChunks(int64_t mode): m_chunks(mode == 1), m_pages(mode == 2) {}

	// Returns false when the chunk cannot serve the read; the caller reads as before.
	bool Read(uint64_t address, std::span<uint32_t> values) {
		if (m_pages) {
			return ReadPage(address, values);
		}
		if (!m_chunks) {
			return false;
		}
		const auto base = address & ~(ChunkSize - 1);
		if (values.size_bytes() > ChunkSize || address + values.size_bytes() > base + ChunkSize) {
			return false;
		}
		auto* slot = Find(base);
		if (slot == nullptr) {
			slot        = &m_slots[m_next++ % m_slots.size()];
			slot->base  = base;
			slot->clean = CleanGuestRead(base, slot->words.data(),
			                                                              ChunkSize);
		}
		if (!slot->clean) {
			return false;
		}
		std::memcpy(values.data(), slot->words.data() + (address - base) / sizeof(uint32_t),
		            values.size_bytes());
		return true;
	}

	void SetLog(ReadLog* log) { m_log = log; }

	[[nodiscard]] bool PageMode() const { return m_pages; }

	// KYTY_LOCAL_HACK (Senaxx fc56ff3d): the page's backing bytes when it is GPU-clean, else null
	// (SrtRuntime::map_clean_page); the same lookup ReadPage makes.
	const uint8_t* MapPage(uint64_t base) {
		const Page* page = nullptr;
		if (m_last_page < m_page_count && m_pages_cache[m_last_page].base == base) {
			page = &m_pages_cache[m_last_page];
		} else {
			for (size_t i = 0; i < m_page_count; i++) {
				if (m_pages_cache[i].base == base) {
					page        = &m_pages_cache[i];
					m_last_page = i;
					break;
				}
			}
		}
		if (page == nullptr) {
			const size_t index = m_page_count < m_pages_cache.size()
			                         ? m_page_count++
			                         : m_next_page++ % m_pages_cache.size();
			auto&        slot  = m_pages_cache[index];
			m_last_page        = index;
			slot = {.base    = base,
			        .backing = CleanGuestBacking(base, TRACKER_PAGE_SIZE)};
			page = &slot;
		}
		return page->backing;
	}

	void Log(uint64_t address, std::span<const uint32_t> values, bool ok) {
		if (m_log == nullptr) {
			return;
		}
		m_log->reads.push_back({.address = address,
		                        .first   = static_cast<uint32_t>(m_log->words.size()),
		                        .count   = static_cast<uint32_t>(values.size()),
		                        .ok      = ok});
		if (ok) {
			// KYTY_LOCAL_HACK: almost every read is one dword; push_back instead of the
			// out-of-line range insert (2% of the process in combat, cp1 10-07).
			if (values.size() == 1) {
				m_log->words.push_back(values[0]);
			} else {
				m_log->words.insert(m_log->words.end(), values.begin(), values.end());
			}
		}
	}

	static int64_t Mode() {
		// Wolverine run 23: see PROGRESS.md for the 1 vs 2 A/B.
		static auto& mode = Common::LiveSwitches::Get("KYTY_SHADER_READ_CHUNKS", 2);
		return mode.load(std::memory_order_relaxed);
	}

private:
	static constexpr uint64_t ChunkSize = 256;

	struct Slot {
		uint64_t                                           base  = UINT64_MAX;
		bool                                               clean = false;
		std::array<uint32_t, ChunkSize / sizeof(uint32_t)> words;
	};

	struct Page {
		uint64_t       base    = UINT64_MAX;
		const uint8_t* backing = nullptr;
	};

	bool ReadPage(uint64_t address, std::span<uint32_t> values) {
		const auto base = Common::AlignDown(address, TRACKER_PAGE_SIZE);
		if (values.empty() ||
		    Common::AlignDown(address + values.size_bytes() - 1, TRACKER_PAGE_SIZE) != base) {
			return false;
		}
		const Page* page = nullptr;
		// Consecutive reads mostly stay on one page: try the last one first (KYTY_READ_OPT, live,
		// default 1, KYTY_LOCAL_HACK).
		static auto& read_opt = Common::LiveSwitches::Get("KYTY_READ_OPT", 1);
		if (read_opt.load(std::memory_order_relaxed) != 0 && m_last_page < m_page_count &&
		    m_pages_cache[m_last_page].base == base) {
			page = &m_pages_cache[m_last_page];
		} else {
			for (size_t i = 0; i < m_page_count; i++) {
				if (m_pages_cache[i].base == base) {
					page        = &m_pages_cache[i];
					m_last_page = i;
					break;
				}
			}
		}
		if (page == nullptr) {
			const size_t index = m_page_count < m_pages_cache.size()
			                         ? m_page_count++
			                         : m_next_page++ % m_pages_cache.size();
			auto&        slot  = m_pages_cache[index];
			m_last_page        = index;
			slot       = {.base = base,
			              .backing =
			                  CleanGuestBacking(base, TRACKER_PAGE_SIZE)};
			page       = &slot;
		}
		if (page->backing == nullptr) {
			return false;
		}
		std::memcpy(values.data(), page->backing + (address - base), values.size_bytes());
		return true;
	}

	Slot* Find(uint64_t base) {
		for (auto& slot: m_slots) {
			if (slot.base == base) {
				return &slot;
			}
		}
		return nullptr;
	}

	bool                 m_chunks = false;
	bool                 m_pages  = false;
	ReadLog*             m_log    = nullptr;
	std::array<Slot, 8>  m_slots;
	size_t               m_next = 0;
	std::array<Page, 16> m_pages_cache;
	size_t               m_page_count = 0;
	size_t               m_next_page  = 0;
	size_t               m_last_page  = SIZE_MAX;
};

// Both SRT readers: only memory the guest has committed. Bytes the GPU has not written are
// current in the backing store; reading them there skips the tracked-page fault, which drains
// the GPU to refresh whatever else on the page the GPU wrote (constants that share a page with
// GPU-written arguments cost a drain per dispatch). Bytes that are mapped but GPU-owned are read
// through the guest mapping, so the fault refreshes them and the value is the one the shader
// would see. Without a reader the walker dereferenced whatever address a descriptor chain
// produced, including 0 on a path the shader never takes.
bool ReadShaderGuestMemoryImpl(ShaderReadChunks* chunks, uint64_t address,
                               std::span<uint32_t> values) {
	if (values.empty()) {
		return false;
	}
	PipelineStats::NoteRead(address, values.size_bytes());
	if (chunks != nullptr && chunks->Read(address, values)) {
		return true;
	}
	if (CleanGuestRead(address, values.data(),
	                                                    values.size_bytes())) {
		return true;
	}
	t_walk_unclean = true;
	if (t_parallel_worker) {
		t_parallel_refused = true;
		return false;
	}
	if (!Libs::LibKernel::Memory::TryReadBacking(address, values.data(), values.size_bytes())) {
		return false;
	}
	std::memcpy(values.data(), reinterpret_cast<const void*>(address), values.size_bytes());
	return true;
}

bool ReadShaderGuestMemoryRaw(void* userdata, uint64_t address, std::span<uint32_t> values) {
	KYTY_PROFILER_BLOCK("ShaderGuestMemoryRead");
	auto*      chunks = static_cast<ShaderReadChunks*>(userdata);
	const bool ok     = ReadShaderGuestMemoryImpl(chunks, address, values);
	if (chunks != nullptr) {
		chunks->Log(address, values, ok);
	}
	return ok;
}

// Research: a subroutine's code for inlining, in 1 KiB steps while the guest has it committed.
std::vector<uint32_t> ReadShaderCode(uint64_t address) {
	constexpr size_t      Chunk = 256;
	constexpr size_t      Limit = 16384;
	std::vector<uint32_t> words;
	while (words.size() < Limit) {
		std::array<uint32_t, Chunk> chunk {};
		if (!ReadShaderGuestMemoryRaw(nullptr, address + words.size() * sizeof(uint32_t), chunk)) {
			break;
		}
		words.insert(words.end(), chunk.begin(), chunk.end());
	}
	return words;
}

bool ReadShaderGuestMemory(void* userdata, uint64_t address, std::span<uint32_t> values) {
	KYTY_PROFILER_BLOCK("ShaderGuestMemoryRead");
	auto*      chunks = static_cast<ShaderReadChunks*>(userdata);
	const bool ok     = ReadShaderGuestMemoryImpl(chunks, address, values);
	if (chunks != nullptr) {
		chunks->Log(address, values, ok);
	}
	return ok;
}

// KYTY_RESOURCE_MEMO=1 (a live switch, off by default): see ProgramCache::Materialize.
bool ResourceMemoEnabled() {
	static auto& enabled = Common::LiveSwitches::Get("KYTY_RESOURCE_MEMO", 0);
	return enabled.load(std::memory_order_relaxed) != 0;
}

// --skip-shaders and KYTY_SKIP_SHADER_HASHES="hash,hash,...": skip the draws and dispatches of
// these guest shaders, the same way a shader that fails to compile is skipped. To see what one
// shader contributes, to step past one that loses the device, or to leave out work nothing can
// use (the ray-tracing BVH updates whose only consumer gives up).
bool SkipShaderRequested(uint64_t shader_hash) {
	static const std::vector<uint64_t> hashes = [] {
		std::vector<uint64_t> result;
		const auto parse = [&result](std::string_view list) {
			size_t start = 0;
			while (start < list.size()) {
				const auto end = std::min(list.find(',', start), list.size());
				if (end > start) {
					result.push_back(
					    std::strtoull(std::string(list.substr(start, end - start)).c_str(), nullptr, 16));
				}
				start = end + 1;
			}
		};
		parse(Config::GetSkipShaderHashes());
		if (const char* value = std::getenv("KYTY_SKIP_SHADER_HASHES"); value != nullptr) {
			parse(value);
		}
		for (const auto hash: result) {
			LOGF("ProgramCache: skipping the draws and dispatches of shader 0x%016" PRIx64 "\n",
			     hash);
		}
		return result;
	}();
	return std::ranges::find(hashes, shader_hash) != hashes.end();
}

// KYTY_LOCAL_HACK (bisection aid): KYTY_LIVE_SKIP_FILE names a file of hex shader hashes, re-read
// whenever the live switch KYTY_LIVE_SKIP_GEN changes; those shaders are skipped from then on.
// Each change of KYTY_SEEN_DUMP writes the hashes used since the previous dump (stage letter +
// hash + uses) to KYTY_LIVE_SKIP_FILE.seen.<value>.
static bool LiveSkipShader(uint64_t shader_hash, ShaderType stage) {
	static const char* path = std::getenv("KYTY_LIVE_SKIP_FILE");
	if (path == nullptr) {
		return false;
	}
	static auto&                                   gen_switch  = Common::LiveSwitches::Get("KYTY_LIVE_SKIP_GEN", 0);
	static auto&                                   dump_switch = Common::LiveSwitches::Get("KYTY_SEEN_DUMP", 0);
	static std::mutex                              mutex;
	static int64_t                                 gen  = 0;
	static int64_t                                 dump = 0;
	static std::unordered_set<uint64_t>            skip;
	static std::unordered_map<uint64_t, std::pair<char, uint64_t>> seen;
	std::lock_guard lock(mutex);
	if (const auto value = gen_switch.load(std::memory_order_relaxed); value != gen) {
		gen = value;
		skip.clear();
		if (FILE* f = std::fopen(path, "r"); f != nullptr) {
			char line[64];
			while (std::fgets(line, sizeof(line), f) != nullptr) {
				if (const auto hash = std::strtoull(line, nullptr, 16); hash != 0) {
					skip.insert(hash);
				}
			}
			std::fclose(f);
		}
		printf("LiveSkip: generation %lld, %zu shaders skipped\n", static_cast<long long>(gen),
		       skip.size());
	}
	if (const auto value = dump_switch.load(std::memory_order_relaxed); value != dump) {
		dump = value;
		const auto out = std::string(path) + ".seen." + std::to_string(value);
		if (FILE* f = std::fopen(out.c_str(), "w"); f != nullptr) {
			for (const auto& [hash, info]: seen) {
				std::fprintf(f, "%c %016" PRIx64 " %" PRIu64 "\n", info.first, hash, info.second);
			}
			std::fclose(f);
		}
		printf("LiveSkip: dumped %zu used shaders to %s\n", seen.size(), out.c_str());
		seen.clear();
	}
	auto& entry = seen[shader_hash];
	entry.first = stage == ShaderType::Pixel ? 'P' : stage == ShaderType::Compute ? 'C' : 'V';
	++entry.second;
	return skip.contains(shader_hash);
}

void DumpShaderSpirv(const char* stage_name, uint64_t shader_hash,
                     const std::vector<uint32_t>& spirv) {
	if (!Config::GraphicsDebugDumpEnabled()) {
		return;
	}
	static std::atomic_int id = 0;
	const auto path = Config::GetShaderLogFolder() / fmt::format("{:04d}_new_shader_{}_{:016x}.spv",
	                                                             id++, stage_name, shader_hash);
	Common::File::CreateDirectories(path.parent_path());
	Common::File file(path);
	if (file.IsInvalid()) {
		const auto path_text = Common::PathToString(path);
		LOGF_COLOR(Log::Color::BrightRed, "Can't create file: %s\n", path_text.c_str());
		return;
	}
	file.Write(spirv.data(), spirv.size() * sizeof(uint32_t));
}

void DumpShaderOriginal(const char* stage_name, uint64_t shader_hash,
                        std::span<const uint32_t> code) {
	if (!Config::GraphicsDebugDumpEnabled()) {
		return;
	}
	EXIT_IF(code.empty());
	static std::atomic_int id = 0;
	const auto path = Config::GetShaderLogFolder() / "original" /
	                  fmt::format("{:04d}_new_shader_{}_{:016x}.bin", id++, stage_name, shader_hash);
	Common::File::CreateDirectories(path.parent_path());
	Common::File file(path);
	if (file.IsInvalid()) {
		const auto path_text = Common::PathToString(path);
		LOGF_COLOR(Log::Color::BrightRed, "Can't create file: %s\n", path_text.c_str());
		return;
	}
	file.Write(code.data(), code.size_bytes());
}

// KYTY_LOCAL_HACK research: recompile phase times (GPU thread only).
uint64_t g_recompile_translate_us = 0;
uint64_t g_recompile_emit_us      = 0;
uint64_t g_recompile_module_us    = 0;

bool ValidateShaderSpirv(const char* label, uint64_t shader_hash,
                         const std::vector<uint32_t>& spirv) {
	if (!Config::ShaderValidationEnabled()) {
		return true;
	}
	spvtools::SpirvTools tools(SPV_ENV_VULKAN_1_3);
	std::string          messages;
	tools.SetMessageConsumer([&messages](spv_message_level_t, const char*,
	                                     const spv_position_t& position, const char* message) {
		messages += fmt::format("{}: {} ({}) {}\n", static_cast<int>(position.line),
		                        static_cast<int>(position.column), static_cast<int>(position.index),
		                        message);
	});
	if (tools.Validate(spirv)) {
		return true;
	}
	spvtools::SpirvTools disassembler(SPV_ENV_VULKAN_1_2);
	std::string          text;
	disassembler.Disassemble(spirv, &text,
	                         static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_NO_HEADER) |
	                             static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_FRIENDLY_NAMES) |
	                             static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_COMMENT) |
	                             static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_INDENT) |
	                             static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_COLOR));
	LOGF_COLOR(Log::Color::BrightRed, "%s SPIR-V validation failed hash=0x%016" PRIx64 ":\n%s",
	           label, shader_hash, messages.c_str());
	LOGF("%s\n", text.c_str());
	return false;
}

} // namespace

// A/B switch for background pipeline compiles (KYTY_ASYNC_PIPELINES=0 compiles every graphics
// pipeline on the GPU thread, as before) and how long a draw waits for a new one
// (KYTY_PIPELINE_WAIT_MS, default 20) before it is skipped.
static const bool g_async_pipelines = [] {
	const char* value = std::getenv("KYTY_ASYNC_PIPELINES");
	return value == nullptr || value[0] != '0';
}();
static const std::chrono::milliseconds g_pipeline_wait = [] {
	const char* value = std::getenv("KYTY_PIPELINE_WAIT_MS");
	return std::chrono::milliseconds(value != nullptr ? std::strtoul(value, nullptr, 10) : 20u);
}();

struct PendingGraphicsPipeline {
	std::unique_ptr<GraphicsPipelineBuild> build;
	vk::PipelineCache                      driver_cache = nullptr;
	vk::Pipeline                           pipeline     = nullptr;
	vk::Result                             result       = vk::Result::eSuccess;
	std::atomic<bool>                      done {false};
};

// Compiles graphics pipelines on worker threads. The driver compiled on the GPU thread, which
// stopped the whole emulated GPU for seconds per new pipeline (103 slow pipelines, 170 s of a
// 600 s first pass through the prologue).
class PipelineCompiler {
public:
	explicit PipelineCompiler(uint32_t threads) {
		for (uint32_t i = 0; i < threads; i++) {
			m_threads.emplace_back([this] { Run(); });
		}
	}
	~PipelineCompiler() { Stop(); }
	KYTY_CLASS_NO_COPY(PipelineCompiler);

	void Submit(std::shared_ptr<PendingGraphicsPipeline> job) {
		{
			std::lock_guard lock(m_mutex);
			m_queue.push_back(std::move(job));
		}
		m_work.notify_one();
	}

	// Waits up to the budget; true when the job is done.
	bool Wait(const PendingGraphicsPipeline& job, std::chrono::milliseconds budget) {
		std::unique_lock lock(m_mutex);
		return m_done.wait_for(lock, budget,
		                       [&job] { return job.done.load(std::memory_order_acquire); });
	}

	// Finishes the compiles in progress; queued ones are left to their owner.
	void Stop() {
		{
			std::lock_guard lock(m_mutex);
			if (m_stopped) {
				return;
			}
			m_stopped = true;
			m_queue.clear();
		}
		m_work.notify_all();
		for (auto& thread: m_threads) {
			thread.join();
		}
		m_threads.clear();
	}

	[[nodiscard]] bool Stopped() {
		std::lock_guard lock(m_mutex);
		return m_stopped;
	}

private:
	void Run() {
		for (;;) {
			std::shared_ptr<PendingGraphicsPipeline> job;
			{
				std::unique_lock lock(m_mutex);
				m_work.wait(lock, [this] { return m_stopped || !m_queue.empty(); });
				if (m_queue.empty()) {
					return;
				}
				job = std::move(m_queue.front());
				m_queue.pop_front();
			}
			job->result = CreateGraphicsPipeline(*job->build, job->driver_cache, &job->pipeline);
			{
				std::lock_guard lock(m_mutex);
				job->done.store(true, std::memory_order_release);
			}
			m_done.notify_all();
		}
	}

	std::mutex                                           m_mutex;
	std::condition_variable                              m_work;
	std::condition_variable                              m_done;
	std::deque<std::shared_ptr<PendingGraphicsPipeline>> m_queue;
	std::vector<std::thread>                             m_threads;
	bool                                                 m_stopped = false;
};

std::size_t PipelineCache::GraphicsPipelineKeyHash::operator()(const GraphicsPipelineKey& key) const {
	std::size_t hash = 0;
	PipelineKeyHash::Mix(hash, key.rendering.color_count);
	for (uint32_t i = 0; i < key.rendering.color_count; i++) {
		PipelineKeyHash::Mix(hash, static_cast<uint32_t>(key.rendering.color_formats[i]));
	}
	PipelineKeyHash::Mix(hash, static_cast<uint32_t>(key.rendering.depth_format));
	PipelineKeyHash::Mix(hash, static_cast<uint32_t>(key.rendering.stencil_format));
	for (const auto id: key.vertex_shader_ids) {
		PipelineKeyHash::Mix(hash, id);
	}
	PipelineKeyHash::Mix(hash, key.ps_shader_id);
	PipelineKeyHash::Mix(hash, key.vertex_input.binding_count);
	for (uint32_t i = 0; i < key.vertex_input.binding_count; i++) {
		PipelineKeyHash::Mix(hash, key.vertex_input.bindings[i].stride);
		PipelineKeyHash::Mix(hash, key.vertex_input.bindings[i].instance);
	}
	PipelineKeyHash::Mix(hash, key.vertex_input.attribute_count);
	for (uint32_t i = 0; i < key.vertex_input.attribute_count; i++) {
		PipelineKeyHash::Mix(hash, key.vertex_input.attributes[i].offset);
		PipelineKeyHash::Mix(hash, key.vertex_input.attributes[i].binding);
	}
	PipelineKeyHash::Mix(hash, XXH3_64bits(&key.static_params, sizeof(key.static_params)));
	return hash;
}

struct PipelineCache::ProgramCache {
	struct ProgramKey {
		ShaderType            stage           = ShaderType::Unknown;
		uint64_t              hash            = 0;
		uint32_t              user_data_count = 0;
		uint32_t              code_size       = 0;
		std::vector<uint32_t> static_state;
		// Exact expanded code includes callees and their original return-PC constants.
		// A different target/body must not reuse a program compiled for an earlier call.
		std::vector<uint32_t> function_code;

		bool operator==(const ProgramKey&) const = default;
	};

	struct Permutation {
		ShaderRecompiler::IR::ResourceSpecialization specialization;
		ShaderRecompiler::IR::CompiledShaderInfo     program;
		ShaderProgram                                handle;
	};

	// KYTY_RESOURCE_MEMO: the inputs and results of a recent MaterializeResources call.
	struct MemoSlot {
		std::vector<uint32_t>                        user_data;
		uint64_t                                     shader_base = 0;
		// Compute dispatch size: bounds the written extent of each written buffer.
		std::array<uint32_t, 3>                      workgroup_count {};
		std::array<uint32_t, 3>                      workgroup_size {};
		ShaderReadChunks::ReadLog                    log;
		ShaderRecompiler::IR::ResourceSnapshot       resources;
		ShaderRecompiler::IR::ResourceSpecialization specialization;
		uint64_t                                     last_use = 0;
		bool                                         ahead    = false; // stored by KYTY_LOOKAHEAD
		uint64_t trusted_epoch = 0; // KYTY_LOOKAHEAD=2: hit without re-reading during this draw
	};
	struct SourceEntry;
	// KYTY_LOOKAHEAD: one stage of the next draw, materialized on the worker into a memo slot.
	struct LookaheadSide {
		bool         used  = false;
		bool         ok    = false;
		SourceEntry* entry = nullptr;
		ShaderParams params;
		MemoSlot     slot;
	};
	// KYTY_MEMO_REBASE: per snapshot word (VisitSnapshotWords order), the rules still consistent
	// with every learned sample: same word, a read user data dword i (copy bit i), or the word
	// plus the change of dword i (add bit i; a 48-bit pointer moved within its 4 GiB window).
	struct RebaseRules {
		std::vector<uint8_t>  same;
		std::vector<uint64_t> copy;
		std::vector<uint64_t> add;
		std::array<size_t, 8> shape {};
		uint32_t              samples   = 0;
		uint32_t              predicted = 0;
		bool                  disabled  = false;
	};
	struct SourceEntry {
		explicit SourceEntry(ShaderRecompiler::IR::ResourcePlan plan)
		    : resource_plan(std::move(plan)) {
			permutations.reserve(8);
		}

		ShaderRecompiler::IR::ResourcePlan           resource_plan;
		ShaderRecompiler::IR::ResourceSnapshot       resources;
		ShaderRecompiler::IR::ResourceSpecialization specialization;
		std::vector<Permutation>                    permutations;
		std::vector<MemoSlot>                        memo;
		// KYTY_RESOURCE_MEMO: user data dwords the plan reads (GetUserData); built on first use.
		std::vector<uint8_t>                         memo_user_data_mask;
		bool                                         memo_mask_built = false;
		RebaseRules                                  rebase;
		// KYTY_MEMO_PLAN_STATS: time in Materialize and how its calls ended, per report period.
		uint64_t                                     stat_ns    = 0;
		uint64_t                                     stat_calls = 0;
		uint64_t                                     stat_exact = 0;
		uint64_t                                     stat_pred  = 0;
		ShaderType                                   stat_stage = ShaderType::Unknown;
		// KYTY_DRAW_PREP: held around every evaluation of resource_plan (its trace and walk state
		// are mutable), which the scanner thread runs too.
		std::mutex                                   walk_mutex;
	};

	static bool MaterializeLocked(SourceEntry& entry, const ShaderRecompiler::IR::SrtRuntime& runtime,
	                              ShaderRecompiler::IR::ResourceSnapshot&       resources,
	                              ShaderRecompiler::IR::ResourceSpecialization& specialization) {
		std::lock_guard lock(entry.walk_mutex);
		return ShaderRecompiler::IR::MaterializeResources(entry.resource_plan, runtime, resources,
		                                                  specialization);
	}

	static constexpr size_t MemoSlots = 8;

	// Materialization reads guest memory only through the runtime's two readers, so the same
	// user data and shader base, with every logged read returning the same result and words,
	// give the same snapshot and specialization. A hit re-reads the logged words and copies the
	// stored result instead of evaluating the resource plan again.
	bool Materialize(SourceEntry& entry, const ShaderRecompiler::IR::SrtRuntime& runtime,
	                 ShaderReadChunks& reads) {
		if (!ServeCensusOn()) {
			return MaterializeImpl(entry, runtime, reads);
		}
		t_walk_unclean   = false;
		const auto start = std::chrono::steady_clock::now();
		const bool ok    = MaterializeImpl(entry, runtime, reads);
		const auto ns    = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                                  std::chrono::steady_clock::now() - start)
                                                  .count());
		const int  row   = t_parallel_worker ? 3 : t_census_row;
		const int  col   = !t_materialize_evaluated ? 0 : (t_walk_unclean ? 2 : 1);
		g_serve_census.ns[row][col].fetch_add(ns, std::memory_order_relaxed);
		g_serve_census.calls[row][col].fetch_add(1, std::memory_order_relaxed);
		return ok;
	}

	bool MaterializeImpl(SourceEntry& entry, const ShaderRecompiler::IR::SrtRuntime& runtime,
	                     ShaderReadChunks& reads) {
		t_materialize_evaluated = true; // cleared again by a memo hit
		if (!ResourceMemoEnabled()) {
			DrawRecordCensus::g_flags |= 1u;
			return MaterializeLocked(entry, runtime, entry.resources, entry.specialization);
		}
		memo_clock++;
		for (auto& slot: entry.memo) {
			if (slot.shader_base == runtime.shader_base &&
			    slot.workgroup_count == runtime.workgroup_count &&
			    slot.workgroup_size == runtime.workgroup_size &&
			    MemoUserDataEqual(entry, slot.user_data, runtime.user_data) &&
			    LookaheadReadsOk(slot, reads)) {
				lookahead_stats[6] += slot.trusted_epoch == lookahead_epoch ? 1u : 0u;
				slot.trusted_epoch = 0;
				entry.resources      = slot.resources;
				entry.specialization = slot.specialization;
				// Dwords the plan does not read may differ; the draw still needs them (push data,
				// vertex/instance offsets).
				entry.resources.user_data.assign(runtime.user_data.begin(), runtime.user_data.end());
				slot.last_use = memo_clock;
				t_materialize_evaluated = false;
				memo_hits++;
				lookahead_stats[4] += slot.ahead ? 1u : 0u;
				if (slot.ahead) {
					DrawPrep::Add(DrawPrep::Counter::AheadHits);
				}
				slot.ahead = false;
				entry.stat_exact++;
				DrawRecordCensus::g_flags |= 4u;
				ReportMemo();
				return true;
			}
		}
		static auto& rebase_on = Common::LiveSwitches::Get("KYTY_MEMO_REBASE", 0);
		MemoSlot*    near      = nullptr;
		bool         near_same = false;
		if (rebase_on.load(std::memory_order_relaxed) != 0 && !entry.rebase.disabled &&
		    runtime.user_data.size() <= 64) {
			near      = NearSlot(entry, runtime);
			near_same = near != nullptr && RebaseInputsSame(entry, *near, runtime.user_data, reads);
			if (near_same && entry.rebase.samples >= RebaseMinSamples &&
			    RebasePredict(entry, *near, runtime.user_data)) {
				near->last_use = memo_clock;
				rebase_hits++;
				entry.stat_pred++;
				if (++entry.rebase.predicted % RebaseVerifyEvery != 0) {
					ReportMemo();
					return true;
				}
				// Sampled proof: evaluate for real and compare; a mismatch ends prediction for
				// this plan and the real result is used.
				const auto predicted      = entry.resources;
				const auto predicted_spec = entry.specialization;
				const bool ok             = MaterializeLocked(entry, runtime, entry.resources, entry.specialization);
				rebase_verified++;
				if (!ok || predicted_spec != entry.specialization ||
				    !SameSnapshotWords(predicted, entry.resources)) {
					entry.rebase.disabled = true;
					rebase_failed++;
					::printf("Resource memo rebase: prediction mismatch, plan 0x%016" PRIx64
					         " disabled (ok %d, spec %d)\n",
					         entry.resource_plan.shader_hash, ok ? 1 : 0,
					         predicted_spec == entry.specialization ? 1 : 0);
				}
				ReportMemo();
				return ok;
			}
		}
		memo_misses++;
		// KYTY_MEMO_CLASSIFY (live, default 0): research counters; they re-read on every miss.
		static auto& classify = Common::LiveSwitches::Get("KYTY_MEMO_CLASSIFY", 0);
		if (classify.load(std::memory_order_relaxed) != 0) {
			ClassifyMemoMiss(entry, runtime, reads);
		}
		// KYTY_LOCAL_HACK KYTY_LOG_REUSE (live, default 1): the miss logs into a scratch log whose
		// buffers are swapped with the stored slot's, so a miss no longer allocates two vectors
		// and frees the evicted slot's.
		static auto& log_reuse = Common::LiveSwitches::Get("KYTY_LOG_REUSE", 1);
		const bool   reuse     = log_reuse.load(std::memory_order_relaxed) != 0;
		ShaderReadChunks::ReadLog fresh_log;
		// Per thread (KYTY_PARALLEL_MATERIALIZE runs two materializations at once).
		thread_local ShaderReadChunks::ReadLog scratch_log;
		auto& log = reuse ? scratch_log : fresh_log;
		log.reads.clear();
		log.words.clear();
		static auto& read_opt = Common::LiveSwitches::Get("KYTY_READ_OPT", 1);
		if (!entry.memo.empty() && read_opt.load(std::memory_order_relaxed) != 0) {
			// The plan reads about as much as last time: no growth reallocations while logging.
			const auto& prev = entry.memo.front().log;
			log.reads.reserve(prev.reads.size() + 8);
			log.words.reserve(prev.words.size() + 16);
		}
		reads.SetLog(&log);
		// Timed only for the miss classifier (KYTY_MEMO_CLASSIFY).
		const auto miss_start = memo_last_class >= 0 ? std::chrono::steady_clock::now()
		                                             : std::chrono::steady_clock::time_point {};
		const bool ok = MaterializeLocked(entry, runtime, entry.resources, entry.specialization);
		reads.SetLog(nullptr);
		ReadStabilityCensus(entry, log);
		DrawRecordCensus::g_flags |= memo_last_class == 6 || memo_last_class == 8 ? 8u : 1u;
		if (memo_last_class >= 0) {
			// KYTY_LOCAL_HACK research (KYTY_MEMO_CLASSIFY): evaluation time per miss class.
			memo_class_ns[memo_last_class] += static_cast<uint64_t>(
			    std::chrono::duration_cast<std::chrono::nanoseconds>(
			        std::chrono::steady_clock::now() - miss_start)
			        .count());
			memo_class_calls[memo_last_class]++;
			memo_last_class = -1;
		}
		if (ok && near_same) {
			// Before the store below, which may move the slots.
			RebaseLearn(entry, *near, runtime.user_data);
		}
		if (ok) {
			MemoSlot* slot = nullptr;
			if (entry.memo.size() < MemoSlots) {
				slot = &entry.memo.emplace_back();
			} else {
				slot = &*std::ranges::min_element(entry.memo, {}, &MemoSlot::last_use);
			}
			slot->user_data.assign(runtime.user_data.begin(), runtime.user_data.end());
			slot->shader_base    = runtime.shader_base;
			slot->workgroup_count = runtime.workgroup_count;
			slot->workgroup_size  = runtime.workgroup_size;
			if (reuse) {
				slot->log.reads.swap(log.reads);
				slot->log.words.swap(log.words);
			} else {
				slot->log = std::move(log);
			}
			slot->resources      = entry.resources;
			slot->specialization = entry.specialization;
			slot->last_use = memo_clock;
			slot->ahead    = false;
			slot->trusted_epoch = 0;
		}
		ReportMemo();
		return ok;
	}

	static constexpr uint32_t RebaseMinSamples  = 4;
	static constexpr uint32_t RebaseVerifyEvery = 16;

	// Every 32-bit word of a snapshot except user_data (a hit re-assigns it), 64-bit fields as
	// lo, hi. f(word) returns the word to store.
	template <typename F>
	static void VisitSnapshotWords(ShaderRecompiler::IR::ResourceSnapshot& s, F&& f) {
		const auto u32 = [&](uint32_t& v) { v = f(v); };
		const auto u64 = [&](uint64_t& v) {
			const uint32_t lo = f(static_cast<uint32_t>(v));
			const uint32_t hi = f(static_cast<uint32_t>(v >> 32u));
			v                 = lo | (static_cast<uint64_t>(hi) << 32u);
		};
		for (auto& e: s.buffer_write_extents) {
			u64(e.begin);
			u64(e.end);
			uint32_t valid = e.valid ? 1u : 0u;
			u32(valid);
			e.valid = valid != 0;
		}
		for (auto& h: s.bindless_heaps) {
			u64(h.base), u64(h.size), u32(h.table_offset), u32(h.image), u32(h.mapping_offset);
		}
		for (auto& h: s.bindless_sampler_heaps) {
			u64(h.base), u64(h.size), u32(h.table_offset), u32(h.sampler), u32(h.mapping_offset);
		}
		for (auto* list: {&s.buffers, &s.images, &s.samplers}) {
			for (auto& d: *list) {
				u32(d.dword_count);
				for (auto& w: d.dwords) {
					u32(w);
				}
			}
		}
		for (auto& w: s.flattened_srt) {
			u32(w);
		}
		for (auto& r: s.specialization_reads) {
			u64(r.first);
			u64(r.second);
		}
		auto kind = static_cast<uint32_t>(s.uniform_fill.kind);
		u32(kind);
		s.uniform_fill.kind = static_cast<ShaderRecompiler::IR::UniformFillKind>(kind);
		u32(s.uniform_fill.resource);
		for (auto& g: s.uniform_fill.group_stride) {
			u32(g);
		}
		u32(s.uniform_fill.words);
		u32(s.uniform_fill.value);
	}

	static std::array<size_t, 8> SnapshotShape(const ShaderRecompiler::IR::ResourceSnapshot& s) {
		return {s.buffer_write_extents.size(), s.bindless_heaps.size(),
		        s.bindless_sampler_heaps.size(), s.buffers.size(), s.images.size(),
		        s.samplers.size(), s.flattened_srt.size(), s.specialization_reads.size()};
	}

	void CollectWords(const ShaderRecompiler::IR::ResourceSnapshot& s, std::vector<uint32_t>& out) {
		out.clear();
		VisitSnapshotWords(const_cast<ShaderRecompiler::IR::ResourceSnapshot&>(s),
		                   [&](uint32_t w) { out.push_back(w); return w; });
	}

	bool SameSnapshotWords(const ShaderRecompiler::IR::ResourceSnapshot& a,
	                       const ShaderRecompiler::IR::ResourceSnapshot& b) {
		if (SnapshotShape(a) != SnapshotShape(b) || a.user_data != b.user_data) {
			return false;
		}
		CollectWords(a, rebase_old);
		CollectWords(b, rebase_new);
		return rebase_old == rebase_new;
	}

	// The slot of the same shader base and dispatch size whose read user data differs in the
	// fewest dwords, 1 to 4 (0 means its reads changed: nothing to rebase).
	MemoSlot* NearSlot(SourceEntry& entry, const ShaderRecompiler::IR::SrtRuntime& runtime) {
		const auto& mask      = MemoUserDataMask(entry);
		MemoSlot*   best      = nullptr;
		size_t      best_diff = SIZE_MAX;
		for (auto& slot: entry.memo) {
			if (slot.shader_base != runtime.shader_base ||
			    slot.workgroup_count != runtime.workgroup_count ||
			    slot.workgroup_size != runtime.workgroup_size ||
			    slot.user_data.size() != runtime.user_data.size()) {
				continue;
			}
			size_t diff = 0;
			for (size_t i = 0; i < slot.user_data.size(); i++) {
				diff += slot.user_data[i] != runtime.user_data[i] && i < mask.size() && mask[i] != 0
				            ? 1
				            : 0;
			}
			if (diff != 0 && diff < best_diff) {
				best      = &slot;
				best_diff = diff;
			}
		}
		return best_diff <= 4 ? best : nullptr;
	}

	// The near slot's inputs are the new ones moved: every differing read dword is part of a
	// 48-bit pointer (lo | (hi & 0xffff) << 32, lo = i or i - 1) that is mapped guest memory both
	// before and after, and every logged read returns the same words at its address moved by the
	// pointer whose 1 MiB window holds it (other reads at their own address).
	bool RebaseInputsSame(SourceEntry& entry, const MemoSlot& slot, std::span<const uint32_t> ud,
	                      ShaderReadChunks& reads) {
		constexpr uint64_t Window = 1u << 20u;
		const auto&        mask   = MemoUserDataMask(entry);
		const auto pointer = [](std::span<const uint32_t> data, size_t lo) {
			return static_cast<uint64_t>(data[lo]) |
			       (static_cast<uint64_t>(data[lo + 1] & 0xffffu) << 32u);
		};
		const auto mapped = [](uint64_t address) {
			uint32_t word = 0;
			return address >= (1ull << 32u) &&
			       Libs::LibKernel::Memory::TryReadBacking(address, &word, sizeof(word));
		};
		std::array<std::pair<uint64_t, uint64_t>, 8> bases {};
		size_t                                       base_count = 0;
		for (size_t i = 0; i < ud.size(); i++) {
			if (slot.user_data[i] == ud[i] || i >= mask.size() || mask[i] == 0) {
				continue;
			}
			bool pointer_found = false;
			for (const size_t lo: {i, i - 1}) {
				if (lo >= ud.size() || lo + 1 >= ud.size()) {
					continue;
				}
				const auto old_base = pointer(slot.user_data, lo);
				const auto new_base = pointer(ud, lo);
				if (!mapped(old_base) || !mapped(new_base)) {
					continue;
				}
				pointer_found = true;
				if (base_count < bases.size()) {
					bases[base_count++] = {old_base, new_base};
				}
			}
			if (!pointer_found) {
				return false;
			}
		}
		// KYTY_LOCAL_HACK KYTY_REBASE_STABLE_SKIP=1 (live, default 0): only the reads under a moved
		// base are read again. Read census 10-05 (rc1): on memo misses, reads at the same address
		// returned the same words in 7,995,442 of 7,996,028 cases; the sampled full evaluation
		// (RebaseVerifyEvery) still checks every word of the prediction.
		static auto& stable_skip = Common::LiveSwitches::Get("KYTY_REBASE_STABLE_SKIP", 0);
		const bool   skip_stable = stable_skip.load(std::memory_order_relaxed) != 0;
		for (const auto& read: slot.log.reads) {
			uint64_t address = read.address;
			for (size_t k = 0; k < base_count; k++) {
				if (read.address >= bases[k].first && read.address - bases[k].first < Window) {
					address = read.address - bases[k].first + bases[k].second;
					break;
				}
			}
			if (skip_stable && address == read.address) {
				continue;
			}
			thread_local std::vector<uint32_t> memo_words; // per thread (parallel materialize)
			memo_words.resize(read.count);
			const bool ok = ReadShaderGuestMemoryImpl(&reads, address, memo_words);
			if (ok != read.ok || (ok && !std::equal(memo_words.begin(), memo_words.end(),
			                                        slot.log.words.begin() + read.first))) {
				return false;
			}
		}
		return true;
	}

	// A fully evaluated near miss whose inputs were the slot's moved: keep, per word, only the
	// rules that explain it. A word no rule explains, another specialization or another shape
	// ends prediction for the plan.
	void RebaseLearn(SourceEntry& entry, const MemoSlot& slot, std::span<const uint32_t> ud) {
		auto& rules = entry.rebase;
		if (rules.disabled) {
			return;
		}
		const auto shape = SnapshotShape(entry.resources);
		if (entry.specialization != slot.specialization || shape != SnapshotShape(slot.resources) ||
		    (rules.samples != 0 && shape != rules.shape)) {
			rules.disabled = true;
			rebase_disabled++;
			return;
		}
		CollectWords(slot.resources, rebase_old);
		CollectWords(entry.resources, rebase_new);
		const auto& mask = MemoUserDataMask(entry);
		if (rules.samples == 0) {
			uint64_t read_dwords = 0;
			for (size_t i = 0; i < ud.size() && i < mask.size(); i++) {
				read_dwords |= mask[i] != 0 ? (1ull << i) : 0;
			}
			rules.shape = shape;
			rules.same.assign(rebase_new.size(), 1);
			rules.copy.assign(rebase_new.size(), read_dwords);
			rules.add.assign(rebase_new.size(), read_dwords);
		}
		for (size_t j = 0; j < rebase_new.size(); j++) {
			const uint32_t o = rebase_old[j];
			const uint32_t w = rebase_new[j];
			if (o != w) {
				rules.same[j] = 0;
			}
			for (uint64_t bits = rules.copy[j]; bits != 0; bits &= bits - 1) {
				const auto i = static_cast<size_t>(std::countr_zero(bits));
				if (w != ud[i] || o != slot.user_data[i]) {
					rules.copy[j] &= ~(1ull << i);
				}
			}
			for (uint64_t bits = rules.add[j]; bits != 0; bits &= bits - 1) {
				const auto i = static_cast<size_t>(std::countr_zero(bits));
				if (static_cast<int64_t>(o) + static_cast<int64_t>(ud[i]) -
				        static_cast<int64_t>(slot.user_data[i]) !=
				    static_cast<int64_t>(w)) {
					rules.add[j] &= ~(1ull << i);
				}
			}
			if (rules.same[j] == 0 && rules.copy[j] == 0 && rules.add[j] == 0) {
				rules.disabled = true;
				rebase_disabled++;
				return;
			}
		}
		rules.samples++;
	}

	// The slot's result with every word predicted; false when a word's remaining rules disagree.
	bool RebasePredict(SourceEntry& entry, const MemoSlot& slot, std::span<const uint32_t> ud) {
		const auto& rules = entry.rebase;
		if (SnapshotShape(slot.resources) != rules.shape) {
			return false;
		}
		entry.resources = slot.resources;
		size_t j        = 0;
		bool   ok       = true;
		VisitSnapshotWords(entry.resources, [&](uint32_t o) -> uint32_t {
			if (!ok || j >= rules.same.size()) {
				ok = false;
				return o;
			}
			bool     have  = false;
			uint32_t value = 0;
			const auto offer = [&](int64_t v) {
				if (v < 0 || v > static_cast<int64_t>(UINT32_MAX)) {
					ok = false;
				} else if (!have) {
					have  = true;
					value = static_cast<uint32_t>(v);
				} else if (value != static_cast<uint32_t>(v)) {
					ok = false;
				}
			};
			if (rules.same[j] != 0) {
				offer(o);
			}
			for (uint64_t bits = rules.copy[j]; bits != 0; bits &= bits - 1) {
				offer(ud[std::countr_zero(bits)]);
			}
			for (uint64_t bits = rules.add[j]; bits != 0; bits &= bits - 1) {
				const auto i = static_cast<size_t>(std::countr_zero(bits));
				offer(static_cast<int64_t>(o) + static_cast<int64_t>(ud[i]) -
				      static_cast<int64_t>(slot.user_data[i]));
			}
			j++;
			return have ? value : o;
		});
		if (!ok || j != rules.same.size()) {
			return false;
		}
		entry.specialization = slot.specialization;
		entry.resources.user_data.assign(ud.begin(), ud.end());
		return true;
	}

	// The walker and the native SRT code read user data only through the plan's GetUserData
	// values, so a dword no such value names cannot change the materialized resources.
	// KYTY_MEMO_UD_MASK (live, default 1): 0 compares every dword as before.
	static const std::vector<uint8_t>& MemoUserDataMask(SourceEntry& entry) {
		if (!entry.memo_mask_built) {
			entry.memo_mask_built = true;
			auto& mask            = entry.memo_user_data_mask;
			const auto& plan      = entry.resource_plan;
			mask.assign(plan.user_data_count, 0);
			for (const auto& inst: plan.value_storage) {
				if (inst.GetOpcode() != ShaderRecompiler::IR::ValueOpcode::GetUserData) {
					continue;
				}
				if (inst.NumArgs() != 1 ||
				    inst.Arg(0).GetType() != ShaderRecompiler::IR::Type::ScalarReg) {
					mask.assign(plan.user_data_count, 1);
					break;
				}
				const auto reg = ShaderRecompiler::IR::RegIndex(inst.Arg(0).ScalarRegister());
				if (reg < plan.user_data_base || reg - plan.user_data_base >= mask.size()) {
					mask.assign(plan.user_data_count, 1);
					break;
				}
				mask[reg - plan.user_data_base] = 1;
			}
		}
		return entry.memo_user_data_mask;
	}

	// KYTY_LOCAL_HACK research (KYTY_DRAW_RECORDS): user data dwords the plan uses as (part of) a
	// memory base: they reach argument 0/1 of GetAddressResource / GetBufferResource within a few
	// operations. A record could relocate them (read the same offsets from the new base).
	static void MarkPointerUserData(const ShaderRecompiler::IR::Value& value,
	                                const ShaderRecompiler::IR::ResourcePlan& plan,
	                                std::vector<uint8_t>& out, int depth) {
		if (depth > 4 || value.IsImmediate()) {
			return;
		}
		auto* inst = value.TryInstruction();
		if (inst == nullptr) {
			return;
		}
		if (inst->GetOpcode() == ShaderRecompiler::IR::ValueOpcode::GetUserData) {
			if (inst->NumArgs() == 1 &&
			    inst->Arg(0).GetType() == ShaderRecompiler::IR::Type::ScalarReg) {
				const auto reg = ShaderRecompiler::IR::RegIndex(inst->Arg(0).ScalarRegister());
				if (reg >= plan.user_data_base && reg - plan.user_data_base < out.size()) {
					out[reg - plan.user_data_base] = 1;
				}
			}
			return;
		}
		for (size_t i = 0; i < inst->NumArgs(); i++) {
			MarkPointerUserData(inst->Arg(i), plan, out, depth + 1);
		}
	}

	// Research: user data dwords that reach what a record would hold fixed: image, sampler and
	// buffer descriptors (beyond the base address), the offsets of SRT and memory reads. The others
	// only reach shader data (copied per draw) or are memory bases (relocated).
	static const std::vector<uint8_t>& StructuralUserDataMask(const SourceEntry& entry) {
		using Op = ShaderRecompiler::IR::ValueOpcode;
		static std::unordered_map<const void*, std::vector<uint8_t>> masks;
		auto [it, inserted] = masks.try_emplace(&entry.resource_plan);
		if (inserted) {
			const auto& plan = entry.resource_plan;
			it->second.assign(plan.user_data_count, 0);
			for (const auto& inst: plan.value_storage) {
				const auto op    = inst.GetOpcode();
				size_t     first = 0;
				switch (op) {
					case Op::GetImageResource:
					case Op::GetSamplerResource: first = 0; break;
					case Op::GetBufferResource: first = 2; break;
					case Op::ReadConst:
					case Op::ReadConstBuffer: first = 1; break;
					case Op::LoadAddressU8:
					case Op::LoadAddressU16:
					case Op::LoadAddressU32: first = 1; break;
					default: continue;
				}
				for (size_t a = first; a < inst.NumArgs(); a++) {
					MarkPointerUserData(inst.Arg(a), plan, it->second, 0);
				}
			}
		}
		return it->second;
	}

	static const std::vector<uint8_t>& PointerUserDataMask(const SourceEntry& entry) {
		static std::unordered_map<const void*, std::vector<uint8_t>> masks;
		auto [it, inserted] = masks.try_emplace(&entry.resource_plan);
		if (inserted) {
			const auto& plan = entry.resource_plan;
			it->second.assign(plan.user_data_count, 0);
			for (const auto& inst: plan.value_storage) {
				const auto op = inst.GetOpcode();
				if ((op == ShaderRecompiler::IR::ValueOpcode::GetAddressResource ||
				     op == ShaderRecompiler::IR::ValueOpcode::GetBufferResource) &&
				    inst.NumArgs() >= 2) {
					MarkPointerUserData(inst.Arg(0), plan, it->second, 0);
					MarkPointerUserData(inst.Arg(1), plan, it->second, 0);
				}
			}
		}
		return it->second;
	}

	static bool MemoUserDataEqual(SourceEntry& entry, std::span<const uint32_t> a,
	                              std::span<const uint32_t> b) {
		static auto& masked = Common::LiveSwitches::Get("KYTY_MEMO_UD_MASK", 1);
		if (a.size() != b.size()) {
			return false;
		}
		if (masked.load(std::memory_order_relaxed) == 0) {
			return std::ranges::equal(a, b);
		}
		const auto& mask = MemoUserDataMask(entry);
		for (size_t i = 0; i < a.size(); i++) {
			if (a[i] != b[i] && (i >= mask.size() || mask[i] != 0)) {
				return false;
			}
		}
		return true;
	}

	// Research: why a lookup missed. Cold: no slot of the same shader base and dispatch size.
	// User data: the nearest slot (fewest different dwords) differs in 1, 2, 3-4 or 5+ dwords.
	// Reads: a slot has the same user data but a logged read returns other words.
	void ClassifyMemoMiss(const SourceEntry& entry, const ShaderRecompiler::IR::SrtRuntime& runtime,
	                      ShaderReadChunks& reads) {
		size_t          best      = SIZE_MAX;
		const MemoSlot* best_slot = nullptr;
		for (const auto& slot: entry.memo) {
			if (slot.shader_base != runtime.shader_base ||
			    slot.workgroup_count != runtime.workgroup_count ||
			    slot.workgroup_size != runtime.workgroup_size ||
			    slot.user_data.size() != runtime.user_data.size()) {
				continue;
			}
			const auto& mask = MemoUserDataMask(const_cast<SourceEntry&>(entry));
			size_t      diff = 0;
			for (size_t i = 0; i < slot.user_data.size(); i++) {
				diff += slot.user_data[i] != runtime.user_data[i] &&
				                (i >= mask.size() || mask[i] != 0)
				            ? 1
				            : 0;
			}
			if (diff < best) {
				best      = diff;
				best_slot = &slot;
			}
		}
		const auto bucket = best == SIZE_MAX ? 0 : best == 0 ? 5 : best == 1 ? 1 : best == 2 ? 2 : best <= 4 ? 3 : 4;
		memo_miss_kinds[bucket]++;
		memo_last_class = static_cast<int>(bucket);
		if (best_slot != nullptr && best >= 1 && best <= 4) {
			ClassifyUserDataDelta(*best_slot, runtime.user_data, reads);
		}
	}

	// Research: what the differing user data dwords of a near miss are. Each differing dword
	// forms 64-bit pointer candidates with its neighbours (lo | (hi & 0xffff) << 32). Pass: no
	// logged read falls in the 1 MiB after an old candidate, so the dwords are not SRT bases;
	// "same" when every logged read still returns its words. Reloc: some reads do; "same" when
	// re-reading them at new base + same offset (others unchanged) returns the logged words.
	void ClassifyUserDataDelta(const MemoSlot& slot, std::span<const uint32_t> user_data,
	                           ShaderReadChunks& reads) {
		constexpr uint64_t Window = 1u << 20u;
		std::array<std::pair<uint64_t, uint64_t>, 16> bases {};
		size_t                                        base_count = 0;
		const auto pointer = [](std::span<const uint32_t> data, size_t lo) {
			return static_cast<uint64_t>(data[lo]) |
			       (static_cast<uint64_t>(data[lo + 1] & 0xffffu) << 32u);
		};
		for (size_t i = 0; i < user_data.size(); i++) {
			if (slot.user_data[i] == user_data[i]) {
				continue;
			}
			for (const size_t lo: {i, i - 1}) {
				if (lo >= user_data.size() || lo + 1 >= user_data.size() || base_count == bases.size()) {
					continue;
				}
				const auto old_base = pointer(slot.user_data, lo);
				if (old_base != 0) {
					bases[base_count++] = {old_base, pointer(user_data, lo)};
				}
			}
		}
		bool relocated = false;
		bool same      = true;
		for (const auto& read: slot.log.reads) {
			uint64_t address = read.address;
			for (size_t k = 0; k < base_count; k++) {
				if (read.address >= bases[k].first && read.address - bases[k].first < Window) {
					address   = read.address - bases[k].first + bases[k].second;
					relocated = true;
					break;
				}
			}
			if (!same) {
				continue;
			}
			thread_local std::vector<uint32_t> memo_words; // per thread (parallel materialize)
			memo_words.resize(read.count);
			const bool ok = ReadShaderGuestMemoryImpl(&reads, address, memo_words);
			same = ok == read.ok && (!ok || std::equal(memo_words.begin(), memo_words.end(),
			                                           slot.log.words.begin() + read.first));
		}
		memo_delta_kinds[(relocated ? 2 : 0) + (same ? 0 : 1)]++;
		memo_last_class = 6 + (relocated ? 2 : 0) + (same ? 0 : 1);
	}

	// KYTY_LOCAL_HACK research, KYTY_READ_CENSUS=1 (live): on a memo miss, each logged read against
	// the same position of the most recent slot's log: same address + same words (a page-epoch
	// proof could skip re-reading it), same address + other words (data changed in place), other
	// address + same words (relocated), other address + other words. Per plan and in total.
	void ReadStabilityCensus(const SourceEntry& entry, const ShaderReadChunks::ReadLog& log) {
		static auto& on = Common::LiveSwitches::Get("KYTY_READ_CENSUS", 0);
		if (on.load(std::memory_order_relaxed) == 0 || entry.memo.empty()) {
			return;
		}
		const auto& prev = std::ranges::max_element(entry.memo, {}, &MemoSlot::last_use)->log;
		struct Counts {
			uint64_t kinds[5] {}; // same/same, same/other, moved/same, moved/other, unmatched
			uint64_t calls = 0;
		};
		static std::unordered_map<uint64_t, Counts> per_plan;
		static Counts                               total;
		static auto                                 last = std::chrono::steady_clock::now();
		auto&                                       plan = per_plan[entry.resource_plan.shader_hash];
		plan.calls++;
		total.calls++;
		const auto n = std::min(prev.reads.size(), log.reads.size());
		for (size_t i = 0; i < log.reads.size(); i++) {
			int kind = 4;
			if (i < n) {
				const auto& a          = prev.reads[i];
				const auto& b          = log.reads[i];
				const bool  same_words = a.ok == b.ok && a.count == b.count &&
				                        (!a.ok || std::equal(prev.words.begin() + a.first,
				                                             prev.words.begin() + a.first + a.count,
				                                             log.words.begin() + b.first));
				kind = (a.address == b.address ? 0 : 2) + (same_words ? 0 : 1);
			}
			plan.kinds[kind]++;
			total.kinds[kind]++;
		}
		if (const auto now = std::chrono::steady_clock::now(); now - last >= std::chrono::seconds(5)) {
			last = now;
			::printf("Read census (5 s): %" PRIu64 " misses; reads same-address same %" PRIu64
			         " / changed %" PRIu64 ", moved same %" PRIu64 " / changed %" PRIu64
			         ", unmatched %" PRIu64 "\n",
			         total.calls, total.kinds[0], total.kinds[1], total.kinds[2], total.kinds[3],
			         total.kinds[4]);
			std::vector<std::pair<uint64_t, const Counts*>> top;
			for (const auto& [hash, counts]: per_plan) top.emplace_back(hash, &counts);
			std::ranges::sort(top, [](const auto& a, const auto& b) {
				return a.second->calls > b.second->calls;
			});
			for (size_t t = 0; t < std::min<size_t>(top.size(), 6); t++) {
				const auto& c = *top[t].second;
				::printf("  plan %016" PRIx64 " misses %" PRIu64 ": same/same %" PRIu64
				         " same/changed %" PRIu64 " moved/same %" PRIu64 " moved/changed %" PRIu64
				         " unmatched %" PRIu64 "\n",
				         top[t].first, c.calls, c.kinds[0], c.kinds[1], c.kinds[2], c.kinds[3],
				         c.kinds[4]);
			}
			per_plan.clear();
			total = {};
		}
	}

	bool ReadsUnchanged(const ShaderReadChunks::ReadLog& log, ShaderReadChunks& reads) {
		// Per thread: the CP (pixel stage) and the MaterializeWkr (vertex stage) check their memo
		// slots at the same time (KYTY_PARALLEL_MATERIALIZE); a shared buffer let one thread
		// compare the words the other had just read.
		thread_local std::vector<uint32_t> memo_words;
		for (const auto& read: log.reads) {
			memo_words.resize(read.count);
			const bool ok = ReadShaderGuestMemoryImpl(&reads, read.address, memo_words);
			if (ok != read.ok || (ok && !std::equal(memo_words.begin(), memo_words.end(),
			                                        log.words.begin() + read.first))) {
				return false;
			}
		}
		return true;
	}

	// Research: the plans that cost the most Materialize time in the last 5 s.
	void ReportPlanStats() {
		const auto now = std::chrono::steady_clock::now();
		if (now - plan_stats_report < std::chrono::seconds(5)) {
			return;
		}
		const double seconds =
		    std::chrono::duration<double>(now - plan_stats_report).count();
		plan_stats_report = now;
		std::vector<SourceEntry*> used;
		uint64_t                  total_ns = 0, total_calls = 0;
		for (auto& [key, entry]: programs) {
			if (entry.stat_calls != 0) {
				used.push_back(&entry);
				total_ns += entry.stat_ns;
				total_calls += entry.stat_calls;
			}
		}
		std::ranges::sort(used, std::greater {}, &SourceEntry::stat_ns);
		::printf("Plan stats (%.1f s): %zu plans, %" PRIu64 " calls, %.1f ms/s in Materialize\n",
		         seconds, used.size(), total_calls, total_ns / 1e6 / seconds);
		for (size_t i = 0; i < used.size() && i < 15; i++) {
			const auto& e     = *used[i];
			const auto  calls = static_cast<double>(e.stat_calls);
			::printf("  %016" PRIx64 " st%u %6.0f/s %6.2f us %5.1f ms/s exact %3.0f%% pred %3.0f%% "
			         "reads %zu buf %zu img %zu smp %zu srt %zu rebase %s/%u\n",
			         e.resource_plan.shader_hash, static_cast<uint32_t>(e.stat_stage),
			         calls / seconds, e.stat_ns / 1e3 / calls, e.stat_ns / 1e6 / seconds,
			         100.0 * e.stat_exact / calls, 100.0 * e.stat_pred / calls,
			         e.memo.empty() ? size_t {0} : e.memo.front().log.reads.size(),
			         e.resources.buffers.size(), e.resources.images.size(),
			         e.resources.samplers.size(), e.resources.flattened_srt.size(),
			         e.rebase.disabled ? "off" : "on", e.rebase.samples);
		}
		std::fflush(stdout);
		for (auto* e: used) {
			e->stat_ns = e->stat_calls = e->stat_exact = e->stat_pred = 0;
		}
	}

	void ReportMemo() {
		// Called for every Materialize: read the clock every 256th call only.
		if ((++memo_report_calls & 255u) != 0u) {
			return;
		}
		const auto now = std::chrono::steady_clock::now();
		if (now - memo_report < std::chrono::seconds(5)) {
			return;
		}
		const auto total = memo_hits + memo_misses;
		::printf("Resource memo (5 s): %" PRIu64 " hits, %" PRIu64 " misses (%.1f%% hits); misses: "
		         "cold %" PRIu64 ", user data 1 %" PRIu64 " / 2 %" PRIu64 " / 3-4 %" PRIu64
		         " / 5+ %" PRIu64 " dwords, reads %" PRIu64 ", programs %zu\n",
		         memo_hits, memo_misses, total != 0 ? 100.0 * memo_hits / total : 0.0,
		         memo_miss_kinds[0], memo_miss_kinds[1], memo_miss_kinds[2], memo_miss_kinds[3],
		         memo_miss_kinds[4], memo_miss_kinds[5], programs.size());
		::printf("Resource memo delta (5 s): pass same %" PRIu64 " / differ %" PRIu64
		         ", reloc same %" PRIu64 " / differ %" PRIu64 "\n",
		         memo_delta_kinds[0], memo_delta_kinds[1], memo_delta_kinds[2], memo_delta_kinds[3]);
		if (ServeCensusOn()) {
			static const char* rows[4] = {"px", "vs", "cs", "wkr"};
			::printf("Serve census (5 s):");
			for (int r = 0; r < 4; r++) {
				::printf(" %s", rows[r]);
				for (int c = 0; c < 3; c++) {
					::printf(" %.1fms/%" PRIu64,
					         static_cast<double>(g_serve_census.ns[r][c].exchange(0)) / 1e6,
					         g_serve_census.calls[r][c].exchange(0));
				}
				::printf(";");
			}
			::printf(" (hit / miss clean / miss unclean)\n");
		}
		::printf("Resource memo rebase (5 s): %" PRIu64 " predicted, %" PRIu64 " verified, %" PRIu64
		         " failed, %" PRIu64 " plans disabled\n",
		         rebase_hits, rebase_verified, rebase_failed, rebase_disabled);
		{
			static const char* names[10] = {"cold", "ud1", "ud2", "ud3-4", "ud5+", "reads",
			                                 "pass-same", "pass-differ", "reloc-same", "reloc-differ"};
			::printf("Resource memo miss time (5 s):");
			for (size_t c = 0; c < memo_class_ns.size(); c++) {
				if (memo_class_calls[c] != 0) {
					::printf(" %s %.1f ms/%" PRIu64, names[c],
					         static_cast<double>(memo_class_ns[c]) / 1e6, memo_class_calls[c]);
				}
			}
			::printf("\n");
			memo_class_ns    = {};
			memo_class_calls = {};
		}
		rebase_hits = rebase_verified = rebase_failed = rebase_disabled = 0;
		memo_miss_kinds = {};
		memo_delta_kinds = {};
		std::fflush(stdout);
		memo_hits   = 0;
		memo_misses = 0;
		memo_report = now;
	}

	// KYTY_PREDICT_STATS=1 (live, default 0): how often a lookup finds the same program as the
	// previous lookup of its stage (a candidate for evaluating the resources ahead from the
	// upcoming user data), and how often the user data is the same too.
	struct PredictStage {
		const SourceEntry*    last = nullptr;
		std::vector<uint32_t> user_data;
		uint64_t              same_program = 0, same_inputs = 0, other = 0;
	};
	std::array<PredictStage, 3>           predict_stages;
	std::chrono::steady_clock::time_point predict_report = std::chrono::steady_clock::now();

	void NotePredictability(ShaderType stage, const SourceEntry* entry,
	                        std::span<const uint32_t> user_data) {
		static auto& enabled = Common::LiveSwitches::Get("KYTY_PREDICT_STATS", 0);
		if (enabled.load(std::memory_order_relaxed) == 0 || entry == nullptr) {
			return;
		}
		const size_t index = stage == ShaderType::Pixel     ? 1u
		                     : stage == ShaderType::Compute ? 2u
		                                                    : 0u;
		auto&        s     = predict_stages[index];
		if (s.last == entry) {
			const bool same = std::ranges::equal(s.user_data, user_data);
			(same ? s.same_inputs : s.same_program)++;
		} else {
			s.other++;
		}
		s.last = entry;
		s.user_data.assign(user_data.begin(), user_data.end());
		const auto now = std::chrono::steady_clock::now();
		if (now - predict_report < std::chrono::seconds(5)) {
			return;
		}
		predict_report                       = now;
		static constexpr const char* Names[] = {"vertex", "pixel", "compute"};
		::printf("Predictability (5 s):");
		for (size_t i = 0; i < predict_stages.size(); i++) {
			auto&      p     = predict_stages[i];
			const auto total = p.same_program + p.same_inputs + p.other;
			::printf(" %s %" PRIu64 ": same program %.1f%% (same inputs %.1f%%)", Names[i], total,
			         total ? 100.0 * static_cast<double>(p.same_program + p.same_inputs) /
			                     static_cast<double>(total)
			               : 0.0,
			         total ? 100.0 * static_cast<double>(p.same_inputs) / static_cast<double>(total)
			               : 0.0);
			p.same_program = p.same_inputs = p.other = 0;
		}
		::printf("\n");
	}

	struct ProgramKeyHash {
		std::size_t operator()(const ProgramKey& key) const {
			std::size_t hash = static_cast<std::size_t>(key.stage);
			PipelineKeyHash::Mix(hash, static_cast<std::size_t>(key.hash));
			if constexpr (sizeof(std::size_t) < sizeof(uint64_t)) {
				PipelineKeyHash::Mix(hash, static_cast<std::size_t>(key.hash >> 32u));
			}
			PipelineKeyHash::Mix(hash, key.user_data_count);
			PipelineKeyHash::Mix(hash, key.code_size);
			PipelineKeyHash::Mix(hash, key.static_state.size());
			PipelineKeyHash::Mix(hash, key.function_code.size());
			// Bucket same-shape static variants by source. ProgramKey equality performs the one
			// exact state comparison needed on a stable hit without hashing the full state first.
			return hash;
		}
	};

	static constexpr std::size_t MaxStaticKeyWords = 32 + ShaderVertexInputInfo::RES_MAX * 6;

	Permutation CompilePermutation(const char*                                  stage_name,
	                               const ShaderRecompiler::CompileOptions&      options,
	                               ShaderRecompiler::TranslateResult            translated,
	                               ShaderRecompiler::IR::ResourceSpecialization specialization,
	                               uint32_t push_data_start_dword) {
		const auto emit_start = std::chrono::steady_clock::now();
		auto result = ShaderRecompiler::CompileProgram(std::move(translated), options,
		                                               specialization, push_data_start_dword);
		const auto module_start = std::chrono::steady_clock::now();
		g_recompile_emit_us += static_cast<uint64_t>(
		    std::chrono::duration_cast<std::chrono::microseconds>(module_start - emit_start).count());
		if (!ValidateShaderSpirv(options.dump_label, options.shader_hash, result.spirv)) {
			DumpShaderSpirv(stage_name, options.shader_hash, result.spirv);
			EXIT("%s failed hash=0x%016" PRIx64 ": SPIR-V validation failed\n", options.dump_label,
			     options.shader_hash);
		}
		DumpShaderSpirv(stage_name, options.shader_hash, result.spirv);

		const auto module = CompileSPV(result.spirv, device);
		EXIT_IF(module == nullptr);
		g_recompile_module_us += static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
		    std::chrono::steady_clock::now() - module_start).count());
		if (options.dump_ir) {
			LOGF("%s SPIR-V words=%" PRIu64 " wave_size=%u\n", options.dump_label,
			     static_cast<uint64_t>(result.spirv.size()), options.wave_size);
		}
		return {
		    .specialization = std::move(specialization),
		    .program        = std::move(result.program).TakeCompiledInfo(),
		    .handle         = {.id = ++next_shader_id, .module = module},
		};
	}

	// KYTY_LOCAL_HACK KYTY_PARALLEL_MATERIALIZE: a stage looked up and materialized ahead (possibly on
	// the worker thread). Get uses it only when the entry it finds is still that one.
	struct PreMaterialized {
		bool               ok    = false;
		const SourceEntry* entry = nullptr;
		bool               evaluated = false; // diagnostics (verify)
		std::string        diag;
		std::vector<std::pair<uint32_t, uint8_t>> walk;
	};

	template <typename InputInfo>
	void BuildLookupKey(const ShaderParams& params, const InputInfo& input_info, ShaderType stage,
	                    std::span<const uint32_t> user_data, ProgramKey& key) {
		key.stage           = stage;
		key.hash            = params.hash;
		key.user_data_count = params.user_data_count;
		key.code_size       = static_cast<uint32_t>(params.code.size());
		BuildStageStaticKey(input_info, key.static_state);
		if (const auto found = call_targets.find(params.hash); found != call_targets.end()) {
			for (const auto index: found->second) {
				key.static_state.push_back(index < user_data.size() ? user_data[index] : 0u);
			}
		}
		key.function_code.clear();
	}

	// The lookup and materialization of a graphics stage, without inserting anything (a missing
	// program is left to Get). On the worker thread, reads that are not GPU-clean refuse it.
	template <typename InputInfo>
	void PreMaterialize(const ShaderParams& params, InputInfo& input_info, ShaderType stage,
	                    ProgramKey& key, PreMaterialized& out) {
		out = {};
		if (SkipShaderRequested(params.hash) || LiveSkipShader(params.hash, stage)) {
			return;
		}
		const auto user_data = std::span(params.user_data).first(params.user_data_count);
		BuildLookupKey(params, input_info, stage, user_data, key);
		if (unsupported.contains(key)) {
			return;
		}
		const auto entry = programs.find(key);
		if (entry == programs.end()) {
			return;
		}
		ShaderReadChunks                 read_chunks(ShaderReadChunks::Mode());
		ShaderRecompiler::IR::SrtRuntime runtime {
		    .user_data                  = user_data,
		    .shader_base                = params.Base(),
		    .read_memory                = ReadShaderGuestMemoryRaw,
		    .userdata                   = &read_chunks,
		    .read_specialization_memory = ReadShaderGuestMemory,
		    .float_image_atomics        = Config::FloatImageAtomicsEnabled(),
		};
		if (read_chunks.PageMode()) {
			runtime.map_clean_page = +[](void* userdata, uint64_t page) {
				return static_cast<ShaderReadChunks*>(userdata)->MapPage(page);
			};
			runtime.log_read = +[](void* userdata, uint64_t address, uint32_t word) {
				static_cast<ShaderReadChunks*>(userdata)->Log(address, {&word, 1}, true);
			};
			runtime.page_userdata = &read_chunks;
		}
		t_parallel_refused = false;
		t_census_row            = CensusRow(stage);
		const bool materialized = Materialize(entry->second, runtime, read_chunks);
		out.ok    = materialized && !t_parallel_refused;
		out.entry = &entry->second;
		out.evaluated = t_materialize_evaluated;
		static auto& mode = Common::LiveSwitches::Get("KYTY_PARALLEL_MATERIALIZE", 1);
		if (out.evaluated && !t_parallel_worker && mode.load(std::memory_order_relaxed) >= 2) {
			out.walk = entry->second.resource_plan.active_walk; // verify diagnostics only
			out.diag = ShaderRecompiler::IR::TakeSrtDiagnostics();
		}
	}

	void CheckFresh(const ShaderParams& params, SourceEntry& entry,
	                const std::vector<std::pair<uint32_t, uint8_t>>& pre_walk, const std::string& diag) {
		const auto ahead = entry.resources;
		const auto user_data = std::span(params.user_data).first(params.user_data_count);
		ShaderReadChunks                 read_chunks(ShaderReadChunks::Mode());
		ShaderRecompiler::IR::SrtRuntime runtime {
		    .user_data                  = user_data,
		    .shader_base                = params.Base(),
		    .read_memory                = ReadShaderGuestMemoryRaw,
		    .userdata                   = &read_chunks,
		    .read_specialization_memory = ReadShaderGuestMemory,
		    .float_image_atomics        = Config::FloatImageAtomicsEnabled(),
		};
		ShaderRecompiler::IR::ResourceSnapshot       fresh;
		ShaderRecompiler::IR::ResourceSpecialization fresh_spec;
		const bool ok = MaterializeLocked(entry, runtime, fresh,
		                                                           fresh_spec);
		static uint64_t checks = 0, differ = 0;
		checks++;
		const bool same = ok && ahead.buffers == fresh.buffers && ahead.images == fresh.images &&
		                  ahead.samplers == fresh.samplers && ahead.flattened_srt == fresh.flattened_srt;
		if (!same && differ++ < 12) {
			const auto& plan = entry.resource_plan;
			const auto& walk = plan.active_walk;
			std::string wd;
			for (size_t i = 0; i < std::max(walk.size(), pre_walk.size()); i++) {
				const auto a = i < pre_walk.size() ? pre_walk[i] : std::pair<uint32_t, uint8_t> {~0u, 9};
				const auto b = i < walk.size() ? walk[i] : std::pair<uint32_t, uint8_t> {~0u, 9};
				if (a != b) {
					wd = fmt::format("walk[{}] block {} outcome {} -> block {} outcome {} (len {} -> {})", i,
					                 a.first, a.second, b.first, b.second, pre_walk.size(), walk.size());
					break;
				}
			}
			std::printf("Parallel fresh MISMATCH: hash 0x%016" PRIx64 " native %d trace %d alias %d: %s || pre: %s\n",
			            params.hash, plan.native_code != nullptr ? 1 : 0, plan.srt_trace != nullptr ? 1 : 0,
			            plan.srt_trace != nullptr ? plan.srt_trace->alias : -1, wd.c_str(), diag.c_str());
			std::fflush(stdout);
		}
		if ((checks & 0xffffu) == 0u) {
			std::printf("Parallel fresh verify: %" PRIu64 " checked, %" PRIu64 " differ\n", checks, differ);
		}
	}

	static MaterializeWorker& Worker() {
		static MaterializeWorker worker;
		return worker;
	}

	// KYTY_LOOKAHEAD: while the CP binds and records this draw, the worker evaluates the next
	// draw's stages (looked up here, on the CP) into side memo slots. Nothing of the entry is
	// written: the CP's draw may use the same entry's resources.
	template <typename InputInfo>
	bool LookaheadFind(const ShaderParams& params, const InputInfo& input_info, ShaderType stage,
	                   ProgramKey& key, LookaheadSide& side) {
		side.used = false;
		side.ok   = false;
		if (SkipShaderRequested(params.hash) || LiveSkipShader(params.hash, stage)) {
			return false;
		}
		BuildLookupKey(params, input_info, stage, std::span(params.user_data).first(params.user_data_count),
		               key);
		if (unsupported.contains(key)) {
			return false;
		}
		const auto entry = programs.find(key);
		if (entry == programs.end()) {
			return false;
		}
		side.entry  = &entry->second;
		side.params = params;
		side.used   = true;
		return true;
	}

	void LookaheadEvaluate(LookaheadSide& side) {
		const auto faults0   = RenderContext::ThreadFaultCount();
		const auto user_data = std::span(side.params.user_data).first(side.params.user_data_count);
		ShaderReadChunks                 read_chunks(ShaderReadChunks::Mode());
		ShaderRecompiler::IR::SrtRuntime runtime {
		    .user_data                  = user_data,
		    .shader_base                = side.params.Base(),
		    .read_memory                = ReadShaderGuestMemoryRaw,
		    .userdata                   = &read_chunks,
		    .read_specialization_memory = ReadShaderGuestMemory,
		    .float_image_atomics        = Config::FloatImageAtomicsEnabled(),
		};
		if (read_chunks.PageMode()) {
			runtime.map_clean_page = +[](void* userdata, uint64_t page) {
				return static_cast<ShaderReadChunks*>(userdata)->MapPage(page);
			};
			runtime.log_read = +[](void* userdata, uint64_t address, uint32_t word) {
				static_cast<ShaderReadChunks*>(userdata)->Log(address, {&word, 1}, true);
			};
			runtime.page_userdata = &read_chunks;
		}
		auto& slot = side.slot;
		slot.log.reads.clear();
		slot.log.words.clear();
		read_chunks.SetLog(&slot.log);
		t_parallel_refused = false;
		const bool ok      = MaterializeLocked(*side.entry, runtime, slot.resources, slot.specialization);
		read_chunks.SetLog(nullptr);
		const auto faults = RenderContext::ThreadFaultCount() - faults0;
		g_worker_faults += faults;
		side.ok = ok && !t_parallel_refused && faults == 0;
		static auto& lookahead_mode = Common::LiveSwitches::Get("KYTY_LOOKAHEAD", 0);
		if (side.ok && lookahead_mode.load(std::memory_order_relaxed) == 3) {
			// Diagnostics: the logged words against guest memory right after the evaluation.
			for (const auto& read: slot.log.reads) {
				for (uint32_t i = 0; read.ok && i < read.count; i++) {
					uint32_t word = 0;
					if (Libs::LibKernel::Memory::TryReadGpuCleanBackingConcurrent(read.address + i * 4u, &word, 4) &&
					    word != slot.log.words[read.first + i]) {
						if (g_la_log_bad.fetch_add(1, std::memory_order_relaxed) < 20) {
							std::printf("Lookahead worker LOG != MEMORY: 0x%016" PRIx64 " logged 0x%08x memory 0x%08x (read %u words)\n",
							            read.address + i * 4u, slot.log.words[read.first + i], word, read.count);
						}
					}
				}
			}
			g_la_log_checked.fetch_add(1, std::memory_order_relaxed);
		}
		if (side.ok) {
			slot.user_data.assign(user_data.begin(), user_data.end());
			slot.shader_base     = runtime.shader_base;
			slot.workgroup_count = runtime.workgroup_count;
			slot.workgroup_size  = runtime.workgroup_size;
			slot.resources.user_data.assign(user_data.begin(), user_data.end());
		}
	}

	void LookaheadNextDraw() { lookahead_epoch++; }

	// KYTY_DRAW_PREP (drawPrep.h). A stage the scanner evaluated: a memo slot for its entry.
	struct ScanStage {
		SourceEntry* entry   = nullptr;
		ShaderType   stage   = ShaderType::Unknown;
		bool         adopted = false;
		MemoSlot     slot;
	};
	struct ScanResult {
		DrawPrep::Key            key;
		uint32_t                 count = 0;
		std::array<ScanStage, 2> stages;
	};
	struct ScanStore {
		std::mutex                                               mutex;
		std::array<std::deque<ScanResult>, DrawPrep::MaxQueues> pending;
		std::vector<ScanResult>                                  recycled; // buffers to reuse
	};
	static constexpr size_t ScanMaxPending  = 8192;
	static constexpr size_t ScanMaxRecycled = 4096;

	// Scanner thread: evaluates one stage into `out` (worker reads: GPU-clean only).
	template <typename InputInfo>
	bool ScanEvaluate(const ShaderParams& params, const InputInfo& input_info, ShaderType stage,
	                  ScanStage& out) {
		if (SkipShaderRequested(params.hash) || LiveSkipShader(params.hash, stage)) {
			DrawPrep::Add(DrawPrep::Counter::SkipUnsupported);
			return false;
		}
		thread_local ProgramKey key;
		const auto              user_data = std::span(params.user_data).first(params.user_data_count);
		SourceEntry*            entry     = nullptr;
		{
			std::shared_lock programs_lock(programs_mutex);
			BuildLookupKey(params, input_info, stage, user_data, key);
			if (!unsupported.contains(key)) {
				if (const auto found = programs.find(key); found != programs.end()) {
					entry = &found->second;
				}
			}
		}
		if (entry == nullptr) {
			DrawPrep::Add(DrawPrep::Counter::SkipNoEntry);
			return false;
		}
		ShaderReadChunks                 read_chunks(ShaderReadChunks::Mode());
		ShaderRecompiler::IR::SrtRuntime runtime {
		    .user_data                  = user_data,
		    .shader_base                = params.Base(),
		    .read_memory                = ReadShaderGuestMemoryRaw,
		    .userdata                   = &read_chunks,
		    .read_specialization_memory = ReadShaderGuestMemory,
		    .float_image_atomics        = Config::FloatImageAtomicsEnabled(),
		};
		if (read_chunks.PageMode()) {
			runtime.map_clean_page = +[](void* userdata, uint64_t page) {
				return static_cast<ShaderReadChunks*>(userdata)->MapPage(page);
			};
			runtime.log_read = +[](void* userdata, uint64_t address, uint32_t word) {
				static_cast<ShaderReadChunks*>(userdata)->Log(address, {&word, 1}, true);
			};
			runtime.page_userdata = &read_chunks;
		}
		if constexpr (std::is_same_v<InputInfo, ShaderComputeInputInfo>) {
			for (uint32_t axis = 0; axis < 3u; axis++) {
				runtime.workgroup_count[axis] = input_info.dispatch_groups[axis];
				runtime.workgroup_size[axis]  = input_info.threads_num[axis];
			}
		}
		auto& slot = out.slot;
		slot.log.reads.clear();
		slot.log.words.clear();
		read_chunks.SetLog(&slot.log);
		t_parallel_refused = false;
		const auto faults0 = RenderContext::ThreadFaultCount();
		const bool ok = MaterializeLocked(*entry, runtime, slot.resources, slot.specialization);
		read_chunks.SetLog(nullptr);
		if (!ok || t_parallel_refused || RenderContext::ThreadFaultCount() != faults0) {
			DrawPrep::Add(DrawPrep::Counter::FailWalk);
			return false;
		}
		slot.user_data.assign(user_data.begin(), user_data.end());
		slot.shader_base     = runtime.shader_base;
		slot.workgroup_count = runtime.workgroup_count;
		slot.workgroup_size  = runtime.workgroup_size;
		slot.resources.user_data.assign(user_data.begin(), user_data.end());
		slot.ahead         = true;
		slot.trusted_epoch = 0;
		out.entry          = entry;
		out.stage          = stage;
		out.adopted        = false;
		DrawPrep::Add(DrawPrep::Counter::Prepared);
		return true;
	}

	// Scanner thread: a result object, with buffers of one the GPU thread is done with.
	ScanResult ScanBegin(const DrawPrep::Key& key) {
		ScanResult result;
		{
			std::lock_guard lock(scan_store.mutex);
			if (!scan_store.recycled.empty()) {
				result = std::move(scan_store.recycled.back());
				scan_store.recycled.pop_back();
			}
		}
		result.key   = key;
		result.count = 0;
		return result;
	}

	void ScanPublish(ScanResult&& result) {
		if (result.count == 0) {
			std::lock_guard lock(scan_store.mutex);
			if (scan_store.recycled.size() < ScanMaxRecycled) {
				scan_store.recycled.push_back(std::move(result));
			}
			return;
		}
		std::lock_guard lock(scan_store.mutex);
		auto&           pending = scan_store.pending[result.key.queue % DrawPrep::MaxQueues];
		if (pending.size() >= ScanMaxPending) {
			DrawPrep::Add(DrawPrep::Counter::Dropped);
			return;
		}
		pending.push_back(std::move(result));
	}

	// GPU thread: the scanner's result for the packet being executed, or null. Results of
	// packets passed without asking are recycled.
	ScanResult* ScanTake() {
		const auto* key = DrawPrep::Current();
		if (key == nullptr || DrawPrep::Mode() == 0) {
			return nullptr;
		}
		if (scan_taken_valid && scan_taken.key == *key) {
			return &scan_taken; // another draw of the same packet
		}
		DrawPrep::Add(DrawPrep::Counter::Takes);
		std::lock_guard lock(scan_store.mutex);
		if (scan_taken_valid && scan_store.recycled.size() < ScanMaxRecycled) {
			scan_store.recycled.push_back(std::move(scan_taken));
		}
		scan_taken_valid = false;
		auto& pending    = scan_store.pending[key->queue % DrawPrep::MaxQueues];
		while (!pending.empty() && pending.front().key.Before(*key)) {
			DrawPrep::Add(DrawPrep::Counter::Stale);
			if (scan_store.recycled.size() < ScanMaxRecycled) {
				scan_store.recycled.push_back(std::move(pending.front()));
			}
			pending.pop_front();
		}
		if (pending.empty() || !(pending.front().key == *key)) {
			DrawPrep::Add(DrawPrep::Counter::MissNone);
			return nullptr;
		}
		scan_taken = std::move(pending.front());
		pending.pop_front();
		scan_taken_valid = true;
		return &scan_taken;
	}

	// GPU thread: stores the prepared slot of `stage` in its entry's memo when it was made from
	// these inputs; Materialize then uses it only as a memo hit (logged reads unchanged).
	void ScanAdopt(ShaderType stage, const ShaderParams& params) {
		auto* result = ScanTake();
		if (result == nullptr) {
			return;
		}
		const auto user_data = std::span(params.user_data).first(params.user_data_count);
		for (uint32_t i = 0; i < result->count; i++) {
			auto& prepared = result->stages[i];
			if (prepared.stage != stage || prepared.adopted) {
				continue;
			}
			if (prepared.slot.shader_base != params.Base() ||
			    !std::ranges::equal(prepared.slot.user_data, user_data)) {
				DrawPrep::Add(DrawPrep::Counter::MissInputs);
				continue;
			}
			auto&     entry = *prepared.entry;
			MemoSlot* slot  = nullptr;
			for (auto& s: entry.memo) {
				if (s.shader_base == prepared.slot.shader_base && s.user_data == prepared.slot.user_data &&
				    s.workgroup_count == prepared.slot.workgroup_count &&
				    s.workgroup_size == prepared.slot.workgroup_size) {
					slot = &s;
					break;
				}
			}
			if (slot == nullptr) {
				if (entry.memo.size() < MemoSlots) {
					slot = &entry.memo.emplace_back();
				} else {
					slot = &*std::ranges::min_element(entry.memo, {}, &MemoSlot::last_use);
				}
			}
			// Swapped: the replaced slot's buffers go back to the scanner with the result.
			std::swap(*slot, prepared.slot);
			slot->last_use   = ++memo_clock;
			prepared.adopted = true;
			DrawPrep::Add(DrawPrep::Counter::Adopted);
		}
	}

	// Scanner thread: both stages of a draw.
	void ScanDraw(const DrawPrep::Key& key, const ShaderParams* pixel_params,
	              const ShaderPixelInputInfo& pixel_info, const ShaderParams& vertex_params,
	              const ShaderVertexInputInfo& vertex_info) {
		auto result = ScanBegin(key);
		if (pixel_params != nullptr &&
		    ScanEvaluate(*pixel_params, pixel_info, ShaderType::Pixel, result.stages[result.count])) {
			result.count++;
		}
		if (ScanEvaluate(vertex_params, vertex_info, vertex_info.logical_stage, result.stages[result.count])) {
			result.count++;
		}
		ScanPublish(std::move(result));
	}

	void ScanDispatch(const DrawPrep::Key& key, const ShaderParams& params,
	                  const ShaderComputeInputInfo& input_info) {
		auto result = ScanBegin(key);
		if (ScanEvaluate(params, input_info, ShaderType::Compute, result.stages[0])) {
			result.count = 1;
		}
		ScanPublish(std::move(result));
	}

	// The memo's read check; a slot trusted for this draw skips it (KYTY_LOOKAHEAD=3 still
	// re-reads and counts the trusted slots whose reads changed).
	bool LookaheadReadsOk(MemoSlot& slot, ShaderReadChunks& reads) {
		if (slot.trusted_epoch != lookahead_epoch) {
			return ReadsUnchanged(slot.log, reads);
		}
		static auto& lookahead = Common::LiveSwitches::Get("KYTY_LOOKAHEAD", 0);
		if (lookahead.load(std::memory_order_relaxed) == 3 && !ReadsUnchanged(slot.log, reads)) {
			lookahead_stats[7]++;
			static uint32_t reported = 0;
			if (reported < 40) {
				reported++;
				thread_local std::vector<uint32_t> memo_words; // per thread
				for (const auto& read: slot.log.reads) {
					memo_words.resize(read.count);
					const bool ok = ReadShaderGuestMemoryImpl(&reads, read.address, memo_words);
					if (ok != read.ok || (ok && !std::equal(memo_words.begin(), memo_words.end(),
					                                        slot.log.words.begin() + read.first))) {
						uint32_t i = 0;
						while (ok && read.ok && i < read.count && memo_words[i] == slot.log.words[read.first + i]) {
							i++;
						}
						uint32_t raw = 0;
						const uint64_t at = read.address + i * 4u;
						const bool raw_ok = Libs::LibKernel::Memory::TryReadGpuCleanBackingConcurrent(at, &raw, 4);
						// The CP's views of the same word: a fresh page reader, the page-backing verdicts,
						// the clean read and the backing store.
						std::array<uint32_t, 1> fresh_word {};
						ShaderReadChunks        fresh(ShaderReadChunks::Mode());
						const bool fresh_ok = ReadShaderGuestMemoryImpl(&fresh, at, fresh_word);
						const uint64_t page = at & ~uint64_t {TRACKER_PAGE_SIZE - 1};
						const auto* cp_page = Libs::LibKernel::Memory::FindGpuCleanBacking(page, TRACKER_PAGE_SIZE);
						const auto* wk_page = Libs::LibKernel::Memory::FindGpuCleanBackingConcurrent(page, TRACKER_PAGE_SIZE);
						uint32_t clean_word = 0, backing_word = 0;
						const bool clean_ok   = Libs::LibKernel::Memory::TryReadGpuCleanBacking(at, &clean_word, 4);
						const bool backing_ok = Libs::LibKernel::Memory::TryReadBacking(at, &backing_word, 4);
						std::printf("Lookahead views 0x%016" PRIx64 ": fresh %d 0x%08x, cp page %s%s 0x%08x, worker page %s, "
						            "clean %d 0x%08x, backing %d 0x%08x\n",
						            at, fresh_ok ? 1 : 0, fresh_word[0], cp_page != nullptr ? "clean" : "NULL",
						            cp_page != nullptr && cp_page != wk_page && wk_page != nullptr ? " (DIFFERENT ptr)" : "",
						            cp_page != nullptr ? *reinterpret_cast<const uint32_t*>(cp_page + (at - page)) : 0u,
						            wk_page != nullptr ? "clean" : "NULL", clean_ok ? 1 : 0, clean_word,
						            backing_ok ? 1 : 0, backing_word);
						std::printf("Lookahead trusted CHANGED (raw now %d 0x%08x; worker log!=memory %llu of %llu slots): ",
						            raw_ok ? 1 : 0, raw, (unsigned long long)g_la_log_bad.load(),
						            (unsigned long long)g_la_log_checked.load());
						std::printf("%.1f us after start, cp writes %llu, commands %llu; "
						            "read 0x%016" PRIx64 " +%u words (of %zu reads): ok %d -> %d, "
						            "word %u 0x%08x -> 0x%08x\n",
						            ElapsedNs(lookahead_start_time) / 1e3,
						            (unsigned long long)(Lookahead::g_cp_writes - lookahead_start_writes),
						            (unsigned long long)(Lookahead::g_commands - lookahead_start_commands),
						            read.address, read.count, slot.log.reads.size(), read.ok ? 1 : 0, ok ? 1 : 0, i,
						            read.ok && i < read.count ? slot.log.words[read.first + i] : 0u,
						            ok && i < read.count ? memo_words[i] : 0u);
						break;
					}
				}
			}
			slot.trusted_epoch = 0;
			return false;
		}
		return true;
	}

	// A logged read inside a GPU write made while the worker ran.
	static bool LookaheadReadsWritten(const ShaderReadChunks::ReadLog& log,
	                                  std::span<const std::pair<uint64_t, uint64_t>> writes) {
		for (const auto& read: log.reads) {
			const uint64_t end = read.address + uint64_t {read.count} * 4u;
			for (const auto& [begin, write_end]: writes) {
				if (read.address < write_end && begin < end) {
					return true;
				}
			}
		}
		return false;
	}

	void LookaheadStart(const ShaderParams& pixel_params, const ShaderPixelInputInfo& pixel_info,
	                    const ShaderParams& vertex_params, const ShaderVertexInputInfo& vertex_info,
	                    bool trusted) {
		LookaheadJoin();
		if (!ResourceMemoEnabled()) {
			return;
		}
		const bool pixel = LookaheadFind(pixel_params, pixel_info, ShaderType::Pixel, lookahead_key[0],
		                                 lookahead_side[0]);
		const bool vertex = LookaheadFind(vertex_params, vertex_info, vertex_info.logical_stage,
		                                  lookahead_key[1], lookahead_side[1]);
		if (!pixel && !vertex) {
			return;
		}
		if (!lookahead_job) {
			lookahead_job = [this] {
				for (auto& side: lookahead_side) {
					if (side.used) {
						LookaheadEvaluate(side);
					}
				}
			};
		}
		lookahead_stats[0]++;
		static auto& lookahead = Common::LiveSwitches::Get("KYTY_LOOKAHEAD", 0);
		if (lookahead.load(std::memory_order_relaxed) == 4) {
			// Overhead control: the prediction, preparation and lookups without the evaluation.
			lookahead_side[0].used = false;
			lookahead_side[1].used = false;
			return;
		}
		lookahead_pending = true;
		lookahead_trusted = trusted;
		lookahead_start_time     = std::chrono::steady_clock::now();
		lookahead_start_writes   = Lookahead::g_cp_writes;
		lookahead_start_commands = Lookahead::g_commands;
		lookahead_start_gpu      = Lookahead::g_write_count;
		if (trusted) {
			while (Lookahead::g_watch_lock.test_and_set(std::memory_order_acquire)) {
				__builtin_ia32_pause();
			}
			Lookahead::g_write_count = 0;
			Lookahead::g_watch_lock.clear(std::memory_order_release);
			Lookahead::g_watch.store(true, std::memory_order_release);
		}
		Worker().Start(lookahead_job);
	}

	// Waits for the worker's lookahead and stores its results as memo slots of their entries: the
	// next Materialize of that entry hits one only if its user data and every logged read match.
	void LookaheadJoin() {
		if (!lookahead_pending) {
			return;
		}
		const auto wait_start = std::chrono::steady_clock::now();
		Worker().Wait();
		lookahead_wait_ns += static_cast<uint64_t>(
		    std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - wait_start).count());
		lookahead_pending = false;
		std::array<std::pair<uint64_t, uint64_t>, Lookahead::WatchSize> writes;
		size_t                                                          write_count = 0;
		if (lookahead_trusted) {
			Lookahead::g_watch.store(false, std::memory_order_release);
			while (Lookahead::g_watch_lock.test_and_set(std::memory_order_acquire)) {
				__builtin_ia32_pause();
			}
			write_count = Lookahead::g_write_count;
			std::copy_n(Lookahead::g_writes.begin(), std::min(write_count, Lookahead::WatchSize), writes.begin());
			Lookahead::g_watch_lock.clear(std::memory_order_release);
			lookahead_trusted = write_count <= Lookahead::WatchSize;
		}
		for (auto& side: lookahead_side) {
			if (!side.used) {
				continue;
			}
			side.used = false;
			lookahead_stats[1]++;
			if (!side.ok) {
				lookahead_stats[2]++;
				continue;
			}
			auto&     entry = *side.entry;
			MemoSlot* slot  = nullptr;
			for (auto& s: entry.memo) {
				if (s.shader_base == side.slot.shader_base && s.user_data == side.slot.user_data &&
				    s.workgroup_count == side.slot.workgroup_count &&
				    s.workgroup_size == side.slot.workgroup_size) {
					slot = &s;
					break;
				}
			}
			if (slot == nullptr) {
				if (entry.memo.size() < MemoSlots) {
					slot = &entry.memo.emplace_back();
				} else {
					slot = &*std::ranges::min_element(entry.memo, {}, &MemoSlot::last_use);
				}
			}
			std::swap(*slot, side.slot);
			slot->last_use = ++memo_clock;
			slot->ahead    = true;
			slot->trusted_epoch =
			    lookahead_trusted && !LookaheadReadsWritten(slot->log, std::span(writes).first(write_count))
			        ? lookahead_epoch
			        : 0;
			lookahead_stats[5] += slot->trusted_epoch != 0 ? 1u : 0u;
			lookahead_stats[3]++;
		}
		if (const auto now = std::chrono::steady_clock::now(); now - lookahead_report > std::chrono::seconds(5)) {
			std::printf("Lookahead materialize (5 s): draws %llu, stages %llu, refused %llu, adopted %llu, hits %llu, "
			            "trusted %llu, trusted hits %llu, trusted but changed %llu; CP wait %.1f ms, prepare %.1f ms\n",
			            (unsigned long long)lookahead_stats[0], (unsigned long long)lookahead_stats[1],
			            (unsigned long long)lookahead_stats[2], (unsigned long long)lookahead_stats[3],
			            (unsigned long long)lookahead_stats[4], (unsigned long long)lookahead_stats[5],
			            (unsigned long long)lookahead_stats[6], (unsigned long long)lookahead_stats[7],
			            lookahead_wait_ns / 1e6, lookahead_prepare_ns / 1e6);
			lookahead_wait_ns    = 0;
			lookahead_prepare_ns = 0;
			std::fflush(stdout);
			lookahead_stats  = {};
			lookahead_report = now;
		}
	}

	static uint64_t ElapsedNs(std::chrono::steady_clock::time_point start) {
		return static_cast<uint64_t>(
		    std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count());
	}

	// KYTY_PARALLEL_SWAP=2: per shader, a running average of its stage's materialization time;
	// the costlier stage of a draw runs on the CP (the worker's reads are slower), the other on
	// the worker. Unknown shaders keep the default (pixel on the CP).
	bool SwapStages(uint64_t pixel_hash, uint64_t vertex_hash) {
		const auto pixel  = stage_cost.find(pixel_hash);
		const auto vertex = stage_cost.find(vertex_hash ^ 0x9e3779b97f4a7c15ull);
		return pixel != stage_cost.end() && vertex != stage_cost.end() && vertex->second > pixel->second;
	}
	void NoteStageCost(uint64_t pixel_hash, uint64_t vertex_hash, uint64_t pixel_ns, uint64_t vertex_ns) {
		const auto note = [&](uint64_t key, uint64_t ns) {
			auto [it, inserted] = stage_cost.try_emplace(key, static_cast<uint32_t>(std::min<uint64_t>(ns, 1u << 30)));
			if (!inserted) {
				it->second = static_cast<uint32_t>((uint64_t {it->second} * 7u + std::min<uint64_t>(ns, 1u << 30)) / 8u);
			}
		};
		note(pixel_hash, pixel_ns);
		note(vertex_hash ^ 0x9e3779b97f4a7c15ull, vertex_ns);
	}

	// The pixel and vertex stages of a draw: their lookups and materializations run at the same
	// time (the vertex stage on the worker), then Get runs for each in the usual order (pixel
	// first: the push data cursor and any compile stay sequential).
	void GetPixelVertex(const ShaderParams& pixel_params, ShaderPixelInputInfo& pixel_info,
	                    const ShaderParams& vertex_params, ShaderVertexInputInfo& vertex_info,
	                    uint32_t& push_data_cursor, ShaderProgram& pixel, ShaderProgram& vertex) {
		auto&                    worker = Worker();
		LookaheadJoin();
		// KYTY_DRAW_PREP: the scanner's slots of this draw go to the memo before either stage runs.
		ScanAdopt(ShaderType::Pixel, pixel_params);
		ScanAdopt(vertex_info.logical_stage, vertex_params);
		PreMaterialized          pre_vertex;
		PreMaterialized          pre_pixel;
		auto job = [&] {
			const auto faults0 = RenderContext::ThreadFaultCount();
			PreMaterialize(vertex_params, vertex_info, vertex_info.logical_stage, worker_key, pre_vertex);
			g_worker_faults += RenderContext::ThreadFaultCount() - faults0;
		};
		const auto cp_faults0 = RenderContext::ThreadFaultCount();
		// KYTY_PARALLEL_MATERIALIZE=3 (control for the verify mode): the same ahead-of-time
		// materializations, both on the CP, one after the other.
		static auto& parallel = Common::LiveSwitches::Get("KYTY_PARALLEL_MATERIALIZE", 1);
		if (parallel.load(std::memory_order_relaxed) == 3) {
			PreMaterialize(pixel_params, pixel_info, ShaderType::Pixel, cp_pre_key, pre_pixel);
			job();
			if (pre_pixel.ok && pre_pixel.evaluated) {
				CheckFresh(pixel_params, *const_cast<SourceEntry*>(pre_pixel.entry), pre_pixel.walk,
				           pre_pixel.diag);
			}
		} else if (parallel.load(std::memory_order_relaxed) == 4) {
			// Control: the vertex stage on the worker, but not at the same time as the pixel stage.
			worker.Start(job);
			worker.Wait();
			PreMaterialize(pixel_params, pixel_info, ShaderType::Pixel, cp_pre_key, pre_pixel);
		} else if (static auto& swap = Common::LiveSwitches::Get("KYTY_PARALLEL_SWAP", 0);
		           swap.load(std::memory_order_relaxed) == 1 ||
		           (swap.load(std::memory_order_relaxed) == 2 && SwapStages(pixel_params.hash, vertex_params.hash))) {
			// KYTY_LOCAL_HACK KYTY_PARALLEL_SWAP (live, default 0: within noise, pn23/po23 off +1%): the pixel stage on the worker, the vertex
			// stage on the CP (the CP waited ~4% for the vertex stage; the worker's reads are slower).
			uint64_t pixel_ns  = 0;
			auto     pixel_job = [&] {
                const auto faults0 = RenderContext::ThreadFaultCount();
                const auto start   = std::chrono::steady_clock::now();
                PreMaterialize(pixel_params, pixel_info, ShaderType::Pixel, worker_key, pre_pixel);
                pixel_ns = ElapsedNs(start);
                g_worker_faults += RenderContext::ThreadFaultCount() - faults0;
			};
			worker.Start(pixel_job);
			const auto start = std::chrono::steady_clock::now();
			PreMaterialize(vertex_params, vertex_info, vertex_info.logical_stage, cp_pre_key, pre_vertex);
			const auto vertex_ns = ElapsedNs(start);
			worker.Wait();
			NoteStageCost(pixel_params.hash, vertex_params.hash, pixel_ns, vertex_ns);
		} else if (swap.load(std::memory_order_relaxed) == 2) {
			uint64_t vertex_ns  = 0;
			auto     vertex_job = [&] {
                const auto start = std::chrono::steady_clock::now();
                job();
                vertex_ns = ElapsedNs(start);
			};
			worker.Start(vertex_job);
			const auto start = std::chrono::steady_clock::now();
			PreMaterialize(pixel_params, pixel_info, ShaderType::Pixel, cp_pre_key, pre_pixel);
			const auto pixel_ns = ElapsedNs(start);
			worker.Wait();
			NoteStageCost(pixel_params.hash, vertex_params.hash, pixel_ns, vertex_ns);
		} else {
			worker.Start(job);
			PreMaterialize(pixel_params, pixel_info, ShaderType::Pixel, cp_pre_key, pre_pixel);
			worker.Wait();
			// KYTY_PARALLEL_MATERIALIZE=2: an evaluation of the pixel stage made while the worker ran,
			// evaluated again now from scratch (no memo) and compared.
			if (parallel.load(std::memory_order_relaxed) == 2 && pre_pixel.entry != nullptr &&
			    pre_vertex.entry != nullptr) {
				const auto& a = pre_pixel.entry->resource_plan;
				const auto& b = pre_vertex.entry->resource_plan;
				static uint32_t shared_reports = 0;
				if ((&a == &b || (a.srt_trace != nullptr && a.srt_trace == b.srt_trace) ||
				     (a.compiled_srt != nullptr && a.compiled_srt == b.compiled_srt)) &&
				    shared_reports++ < 8) {
					std::printf("Parallel SHARED: plan %d trace %d compiled %d (ps 0x%016" PRIx64 " vs 0x%016" PRIx64 ")\n",
					            &a == &b ? 1 : 0, a.srt_trace != nullptr && a.srt_trace == b.srt_trace ? 1 : 0,
					            a.compiled_srt != nullptr && a.compiled_srt == b.compiled_srt ? 1 : 0,
					            pixel_params.hash, vertex_params.hash);
				}
			}
			if (parallel.load(std::memory_order_relaxed) == 2 && pre_pixel.ok && pre_pixel.evaluated) {
				CheckFresh(pixel_params, *const_cast<SourceEntry*>(pre_pixel.entry), pre_pixel.walk,
				           pre_pixel.diag);
			}
		}
		g_cp_pre_faults = RenderContext::ThreadFaultCount() - cp_faults0;
		pixel  = Get(pixel_params, pixel_info, push_data_cursor, &pre_pixel);
		vertex = Get(vertex_params, vertex_info, push_data_cursor, &pre_vertex);
	}

	template <typename InputInfo>
	ShaderProgram Get(const ShaderParams& params, InputInfo& input_info,
	                  uint32_t& push_data_cursor, const PreMaterialized* pre = nullptr) {
		LookaheadJoin();
		ShaderType stage;
		if constexpr (std::is_same_v<InputInfo, ShaderVertexInputInfo>) {
			stage = input_info.logical_stage;
		} else if constexpr (std::is_same_v<InputInfo, ShaderPixelInputInfo>) {
			stage = ShaderType::Pixel;
		} else {
			static_assert(std::is_same_v<InputInfo, ShaderComputeInputInfo>);
			stage = ShaderType::Compute;
		}

		if (SkipShaderRequested(params.hash) || LiveSkipShader(params.hash, stage)) {
			// KYTY_LOCAL_HACK research (KYTY_MEMO_CLASSIFY=1): which skipped shaders are asked for.
			static auto& classify = Common::LiveSwitches::Get("KYTY_MEMO_CLASSIFY", 0);
			if (classify.load(std::memory_order_relaxed) != 0) {
				static std::unordered_map<uint64_t, uint64_t> skipped;
				static auto last = std::chrono::steady_clock::now();
				skipped[params.hash]++;
				if (const auto now = std::chrono::steady_clock::now();
				    now - last >= std::chrono::seconds(5)) {
					::printf("Skipped shader lookups (5 s):");
					for (const auto& [hash, count]: skipped) {
						::printf(" 0x%016" PRIx64 " x%" PRIu64, hash, count);
					}
					::printf("\n");
					skipped.clear();
					last = now;
				}
			}
			return ShaderProgram {};
		}
		BufferCache::SetReadbackStatsShader(params.hash);
		const auto user_data = std::span(params.user_data).first(params.user_data_count);
		{
			KYTY_PROFILER_BLOCK("ProgramCache::BuildKey");
			lookup_key.stage           = stage;
			lookup_key.hash            = params.hash;
			lookup_key.user_data_count = params.user_data_count;
			lookup_key.code_size       = static_cast<uint32_t>(params.code.size());
			BuildStageStaticKey(input_info, lookup_key.static_state);
			// Research: a program with inlined calls is valid only for the same call targets.
			if (const auto found = call_targets.find(params.hash); found != call_targets.end()) {
				for (const auto index: found->second) {
					lookup_key.static_state.push_back(index < user_data.size() ? user_data[index]
					                                                           : 0u);
				}
			}
			// Research: ShaderFunctions expands calls first; the calls it refuses reach the
			// recompiler's own inliner.
			lookup_key.function_code.clear();
			if constexpr (std::is_same_v<InputInfo, ShaderComputeInputInfo>) {
				ExpandShaderFunctions(params, user_data, input_info.wave_size,
				                      lookup_key.function_code);
			}
		}
		auto entry = programs.end();
		{
			KYTY_PROFILER_BLOCK("ProgramCache::Lookup");
			if (unsupported.contains(lookup_key)) {
				return ShaderProgram {};
			}
			entry = programs.find(lookup_key);
		}
		NotePredictability(stage, entry != programs.end() ? &entry->second : nullptr, user_data);
		ShaderReadChunks                 read_chunks(ShaderReadChunks::Mode());
		ShaderRecompiler::IR::SrtRuntime runtime {
		    .user_data                  = user_data,
		    .shader_base                = params.Base(),
		    .read_memory                = ReadShaderGuestMemoryRaw,
		    .userdata                   = &read_chunks,
		    .read_specialization_memory = ReadShaderGuestMemory,
		    .float_image_atomics        = Config::FloatImageAtomicsEnabled(),
		};
		if (read_chunks.PageMode()) {
			static_assert(TRACKER_PAGE_SIZE == 4096u);
			runtime.map_clean_page = +[](void* userdata, uint64_t page) {
				return static_cast<ShaderReadChunks*>(userdata)->MapPage(page);
			};
			runtime.log_read = +[](void* userdata, uint64_t address, uint32_t word) {
				static_cast<ShaderReadChunks*>(userdata)->Log(address, {&word, 1}, true);
			};
			runtime.page_userdata = &read_chunks;
		}
		if constexpr (std::is_same_v<InputInfo, ShaderComputeInputInfo>) {
			for (uint32_t axis = 0; axis < 3u; axis++) {
				runtime.workgroup_count[axis] = input_info.dispatch_groups[axis];
				runtime.workgroup_size[axis]  = input_info.threads_num[axis];
			}
		}
		if (entry != programs.end()) {
			bool materialized = false;
			t_census_row      = CensusRow(stage);
			if (pre == nullptr) {
				ScanAdopt(stage, params); // KYTY_DRAW_PREP
			}
			if (pre != nullptr && pre->ok && pre->entry == &entry->second) {
				materialized = true; // done ahead (KYTY_PARALLEL_MATERIALIZE)
				// KYTY_PARALLEL_MATERIALIZE=2: materialize again here and compare (verify).
				static auto& parallel = Common::LiveSwitches::Get("KYTY_PARALLEL_MATERIALIZE", 1);
				if (parallel.load(std::memory_order_relaxed) >= 2) {
					const auto ahead = entry->second.resources;
					materialized     = Materialize(entry->second, runtime, read_chunks);
					const auto& now  = entry->second.resources;
					const bool  same = materialized && ahead.buffers == now.buffers &&
					                  ahead.images == now.images && ahead.samplers == now.samplers &&
					                  ahead.flattened_srt == now.flattened_srt;
					static uint64_t checked = 0, differ = 0;
					checked++;
					if (!same && differ++ < 16) {
						std::string detail;
						const auto diff_desc = [&](const char* name, const auto& a, const auto& b) {
							if (a.size() != b.size()) {
								detail += fmt::format(" {} size {}->{}", name, a.size(), b.size());
								return;
							}
							for (size_t i = 0; i < a.size(); i++) {
								if (!(a[i] == b[i])) {
									detail += fmt::format(" {}[{}]", name, i);
									if constexpr (std::is_same_v<std::decay_t<decltype(a[i])>, uint32_t>) {
										detail += fmt::format("={:08x}->{:08x}", a[i], b[i]);
									} else {
										detail += fmt::format("={:08x}.{:08x}->{:08x}.{:08x}", a[i].dwords[0],
										                      a[i].dwords[1], b[i].dwords[0], b[i].dwords[1]);
									}
									return;
								}
							}
						};
						diff_desc("buf", ahead.buffers, now.buffers);
						diff_desc("img", ahead.images, now.images);
						diff_desc("smp", ahead.samplers, now.samplers);
						diff_desc("srt", ahead.flattened_srt, now.flattened_srt);
						// Did a word the ahead evaluation read change since (memory), or not (evaluation)?
						for (const auto& slot: entry->second.memo) {
							if (slot.resources.flattened_srt != ahead.flattened_srt ||
							    !(slot.resources.buffers == ahead.buffers)) {
								continue;
							}
							size_t changed = 0;
							for (const auto& read: slot.log.reads) {
								std::vector<uint32_t> words(read.count);
								const bool ok = ReadShaderGuestMemoryImpl(nullptr, read.address, words);
								const bool same_words =
								    ok == read.ok &&
								    (!ok || std::equal(words.begin(), words.end(),
								                       slot.log.words.begin() + read.first));
								if (!same_words && changed++ == 0) {
									detail += fmt::format(" | read 0x{:x} ok {}->{} word {:08x}->{:08x}",
									                      read.address, read.ok, ok,
									                      read.ok ? slot.log.words[read.first] : 0u,
									                      ok ? words[0] : 0u);
								}
							}
							detail += fmt::format(" | {} of {} logged reads changed", changed,
							                      slot.log.reads.size());
							break;
						}
						std::printf("Parallel materialize MISMATCH: stage %u hash 0x%016" PRIx64
						            " ok %d (cp pre faults %llu, worker faults %llu):%s\n",
						            static_cast<uint32_t>(stage), params.hash, materialized ? 1 : 0,
						            (unsigned long long)g_cp_pre_faults,
						            (unsigned long long)g_worker_faults.load(), detail.c_str());
					}
					if ((checked & 0xffffu) == 0u) {
						std::printf("Parallel materialize verify: %" PRIu64 " checked, %" PRIu64
						            " differ, worker faults %llu\n",
						            checked, differ, (unsigned long long)g_worker_faults.load());
						std::fflush(stdout);
					}
				}
			} else {
				KYTY_PROFILER_BLOCK("ProgramCache::MaterializeResources");
				static auto& plan_stats = Common::LiveSwitches::Get("KYTY_MEMO_PLAN_STATS", 0);
				if (plan_stats.load(std::memory_order_relaxed) != 0) {
					const auto start = std::chrono::steady_clock::now();
					materialized     = Materialize(entry->second, runtime, read_chunks);
					entry->second.stat_ns += static_cast<uint64_t>(
					    std::chrono::duration_cast<std::chrono::nanoseconds>(
					        std::chrono::steady_clock::now() - start)
					        .count());
					entry->second.stat_calls++;
					entry->second.stat_stage = stage;
					ReportPlanStats();
				} else {
					materialized = Materialize(entry->second, runtime, read_chunks);
				}
				if (materialized) {
					// KYTY_LOCAL_HACK research (KYTY_DRAW_RECORDS): the record key of this stage.
					static auto& records = Common::LiveSwitches::Get("KYTY_DRAW_RECORDS", 0);
					if (records.load(std::memory_order_relaxed) != 0) {
						const auto& mask = MemoUserDataMask(entry->second);
						uint64_t    h    = 0xcbf29ce484222325ull ^ runtime.shader_base;
						for (size_t i = 0; i < runtime.user_data.size(); i++) {
							const uint32_t word =
							    i < mask.size() && mask[i] == 0 ? 0u : runtime.user_data[i];
							h = (h ^ word ^ (static_cast<uint64_t>(i) << 40u)) * 0x100000001b3ull;
						}
						entry->second.resources.record_key = h;
						// Relocation: user data the plan uses as a memory base leaves the key.
						const auto& pointers   = PointerUserDataMask(entry->second);
						const auto& structural = StructuralUserDataMask(entry->second);
						uint64_t r = 0xcbf29ce484222325ull ^ runtime.shader_base;
						const auto ud = runtime.user_data;
						for (size_t i = 0; i < ud.size(); i++) {
							uint32_t word = i < mask.size() && mask[i] == 0 ? 0u : ud[i];
							if ((i < pointers.size() && pointers[i] != 0) ||
							    i >= structural.size() || structural[i] == 0) {
								word = 0xffffffffu;
							}
							r = (r ^ word ^ (static_cast<uint64_t>(i) << 40u)) * 0x100000001b3ull;
						}
						entry->second.resources.record_key_reloc = r;
						// Research: which read user data dwords change between uses of a program.
						struct UdDiag {
							std::vector<uint32_t> prev;
							std::vector<uint64_t> changes;
							std::vector<std::pair<uint32_t, uint32_t>> sample;
							uint64_t              calls = 0;
						};
						static std::unordered_map<uint64_t, UdDiag> ud_diag;
						static auto ud_diag_last = std::chrono::steady_clock::now();
						auto& d = ud_diag[params.hash];
						d.calls++;
						if (d.prev.size() == ud.size()) {
							d.changes.resize(ud.size());
							d.sample.resize(ud.size());
							for (size_t i = 0; i < ud.size(); i++) {
								if (i < mask.size() && mask[i] != 0 && d.prev[i] != ud[i]) {
									d.changes[i]++;
									d.sample[i] = {d.prev[i], ud[i]};
								}
							}
						}
						d.prev.assign(ud.begin(), ud.end());
						if (const auto now = std::chrono::steady_clock::now();
						    now - ud_diag_last >= std::chrono::seconds(5)) {
							ud_diag_last = now;
							std::vector<std::pair<uint64_t, const UdDiag*>> top;
							for (const auto& [hash, diag]: ud_diag) top.emplace_back(hash, &diag);
							std::ranges::sort(top, [](const auto& a, const auto& b) {
								return a.second->calls > b.second->calls;
							});
							for (size_t t = 0; t < std::min<size_t>(top.size(), 8); t++) {
								const auto& diag = *top[t].second;
								::printf("UD diag 0x%016" PRIx64 " calls %" PRIu64 ":", top[t].first,
								         diag.calls);
								for (size_t i = 0; i < diag.changes.size(); i++) {
									if (diag.changes[i] != 0) {
										::printf(" ud%zu x%" PRIu64 " (%08x->%08x)", i, diag.changes[i],
										         diag.sample[i].first, diag.sample[i].second);
									}
								}
								::printf("\n");
							}
							ud_diag.clear();
						}
					}
				}
				if (ShaderRecompiler::IR::SrtNativeStats stats;
				    ShaderRecompiler::IR::TakeSrtNativeReport(stats)) {
					::printf("SRT native: %u plans compiled, %u failed, %" PRIu64 " KB of code, %" PRIu64
					     " instructions, %" PRIu64 " through the interpreter\n",
					     stats.plans, stats.failed, stats.bytes / 1024u, stats.instructions,
					     stats.interpreted);
				}
			}
			if (!materialized) {
				// A descriptor source that cannot be read right now (memory the guest has not
				// mapped or filled yet) skips this draw rather than the session; the next one
				// re-evaluates from scratch.
				if (!ShaderFailureNonFatal()) {
					EXIT("shader resource materialization failed\n");
				}
				static std::atomic<uint32_t> reported = 0;
				if (reported.fetch_add(1) < 16) {
					LOGF("ProgramCache: skipping stage %u hash=0x%016" PRIx64
					     ": resource materialization failed (materialization line %d)\n",
					     static_cast<uint32_t>(stage), params.hash,
					     ShaderRecompiler::IR::LastIndirectImageFailureLine());
				}
				return ShaderProgram {};
			}
			auto permutation = entry->second.permutations.end();
			{
				KYTY_PROFILER_BLOCK("ProgramCache::FindPermutation");
				permutation = std::ranges::find_if(
				    entry->second.permutations, [&](const Permutation& candidate) {
					    const auto& layout = candidate.program.bindings;
					    return layout.push_data_start_dword ==
					               ShaderRecompiler::IR::PushData::StartFor(
					                   push_data_cursor, layout.ShaderDataDwords()) &&
					           candidate.specialization == entry->second.specialization;
				    });
			}
			if (permutation != entry->second.permutations.end()) {
				input_info.stage = {.program   = &permutation->program,
				                    .resources = &entry->second.resources};
				permutation->program.bindings.AdvancePushData(push_data_cursor);
				return permutation->handle;
			}
		}

		WarmReport();
		DrawRecordCensus::g_flags |= 1u; // no materialized program or permutation to replay
		const auto verify_start = push_data_cursor;
		if (auto handle = TryPrepared(input_info, push_data_cursor, entry, runtime, stage, params.hash);
		    handle.has_value()) {
			if (!warm_verify_pending.has_value()) {
				return *handle;
			}
			// KYTY_WARM_VERIFY: compile the same permutation here and compare (result unused).
			auto expected = std::move(*warm_verify_pending);
			warm_verify_pending.reset();
			ShaderRecompiler::CompileOptions verify_options;
			verify_options.stage          = stage;
			verify_options.shader_hash    = params.hash;
			verify_options.user_data      = user_data;
			verify_options.back_code      = params.back_code;
			verify_options.dump_ir        = false;
			verify_options.dump_label     = "WarmVerify";
			ShaderStageInputInfo verify_input {};
			if constexpr (std::is_same_v<InputInfo, ShaderVertexInputInfo>) {
				verify_input.vertex = &input_info;
				verify_options.user_data_base = 8;
				verify_options.wave_size      = input_info.wave_size;
				if (stage == ShaderType::Mesh || stage == ShaderType::TessellationControl) {
					verify_options.user_data_base = 0;
					verify_options.wave_size = stage == ShaderType::Mesh ? input_info.mesh.wave_size : 64u;
				}
			} else if constexpr (std::is_same_v<InputInfo, ShaderPixelInputInfo>) {
				verify_input.pixel       = &input_info;
				verify_options.wave_size = input_info.wave_size;
			} else {
				verify_input.compute     = &input_info;
				verify_options.wave_size = input_info.wave_size;
			}
			verify_options.input_info      = verify_input;
			verify_options.non_fatal       = true;
			verify_options.bindless_images = bindless_images;
			verify_options.read_code       = ReadShaderCode;
			const auto verify_code = lookup_key.function_code.empty()
			                             ? params.code
			                             : std::span<const uint32_t>(lookup_key.function_code);
			auto       translated  = ShaderRecompiler::TranslateProgram(verify_code, verify_options);
			if (!translated.unsupported) {
				auto result = ShaderRecompiler::CompileProgram(std::move(translated), verify_options,
				                                               entry->second.specialization, verify_start);
				if (result.spirv == expected) {
					warm_verify_ok++;
				} else {
					warm_verify_bad++;
					::printf("Warm cache: VERIFY MISMATCH stage %u hash=0x%016" PRIx64 " (%zu vs %zu words)\n",
					         static_cast<uint32_t>(stage), params.hash, expected.size(), result.spirv.size());
				}
			}
			return *handle;
		}

		if (warm_async && stage != ShaderType::Compute) {
			if (AsyncCompile(params, input_info, AsyncOptions(params, input_info, stage, user_data),
			                 entry, push_data_cursor, stage)) {
				return ShaderProgram {};
			}
		}

		// A program (or permutation) this session has not built yet: recompiled on the GPU
		// thread. Wolverine run 54: the 0.3-0.7 s frames while turning were spent here.
		KYTY_PROFILER_BLOCK("ProgramCache::Recompile", profiler::colors::RedA100);
		// KYTY_LOCAL_HACK research: synchronous recompile time on the GPU thread, per 5 s.
		struct RecompileTimer {
			std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
			~RecompileTimer() {
				static std::chrono::steady_clock::time_point report = std::chrono::steady_clock::now();
				static uint64_t total_us = 0, count = 0, max_us = 0;
				const auto      now = std::chrono::steady_clock::now();
				const auto us = static_cast<uint64_t>(
				    std::chrono::duration_cast<std::chrono::microseconds>(now - start).count());
				total_us += us;
				count++;
				max_us = std::max(max_us, us);
				if (now - report >= std::chrono::seconds(5)) {
					::printf("Recompile (5 s): %" PRIu64 " programs, %.1f ms total, max %.1f ms; translate "
					         "%.1f ms, emit %.1f ms, module %.1f ms\n",
					         count, total_us / 1e3, max_us / 1e3, g_recompile_translate_us / 1e3,
					         g_recompile_emit_us / 1e3, g_recompile_module_us / 1e3);
					g_recompile_translate_us = g_recompile_emit_us = g_recompile_module_us = 0;
					std::fflush(stdout);
					total_us = count = max_us = 0;
					report   = now;
				}
			}
		} recompile_timer;
		ShaderStageInputInfo stage_input {};
		if constexpr (std::is_same_v<InputInfo, ShaderVertexInputInfo>) {
			stage_input.vertex = &input_info;
		} else if constexpr (std::is_same_v<InputInfo, ShaderPixelInputInfo>) {
			stage_input.pixel = &input_info;
		} else {
			stage_input.compute = &input_info;
		}
		const char* label = nullptr;
		const char* stage_name = nullptr;
		switch (stage) {
			case ShaderType::Vertex: label = "ShaderRecompiler VS"; stage_name = "vs"; break;
			case ShaderType::Mesh: label = "ShaderRecompiler MS"; stage_name = "ms"; break;
			case ShaderType::Local: label = "ShaderRecompiler LS"; stage_name = "ls"; break;
			case ShaderType::TessellationControl: label = "ShaderRecompiler HS"; stage_name = "hs"; break;
			case ShaderType::TessellationEvaluation: label = "ShaderRecompiler DS"; stage_name = "ds"; break;
			case ShaderType::Pixel: label = "ShaderRecompiler PS"; stage_name = "ps"; break;
			case ShaderType::Compute: label = "ShaderRecompiler CS"; stage_name = "cs"; break;
			default: EXIT("invalid pipeline shader stage\n");
		}
		ShaderRecompiler::CompileOptions options;
		options.stage       = stage;
		options.shader_hash = params.hash;
		options.user_data   = user_data;
		options.back_code      = params.back_code;
		options.dump_ir     = Config::GetShaderLogDirection() != Config::LogDirection::Silent;
		options.early_dump  = options.dump_ir;
		options.dump_label  = label;
		options.input_info  = stage_input;

		if constexpr (std::is_same_v<InputInfo, ShaderVertexInputInfo>) {
			options.user_data_base = 8;
			options.wave_size = input_info.wave_size;
			if (stage == ShaderType::Mesh || stage == ShaderType::TessellationControl) {
				options.user_data_base = 0;
				options.wave_size = stage == ShaderType::Mesh ? input_info.mesh.wave_size : 64u;
			}
		} else {
			options.wave_size = input_info.wave_size;
		}
		DumpShaderOriginal(stage_name, options.shader_hash, params.code);
		options.non_fatal = ShaderFailureNonFatal();
		options.bindless_images = bindless_images;
		options.read_code       = ReadShaderCode;
		const auto compile_code = lookup_key.function_code.empty()
		                              ? params.code
		                              : std::span<const uint32_t>(lookup_key.function_code);
		const auto translate_start = std::chrono::steady_clock::now();
		auto translated = ShaderRecompiler::TranslateProgram(compile_code, options);
		g_recompile_translate_us += static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
		    std::chrono::steady_clock::now() - translate_start).count());
		if (!translated.unsupported && !translated.call_target_user_data.empty() &&
		    [&] {
		        std::unique_lock programs_lock(programs_mutex);
		        return call_targets.try_emplace(params.hash, translated.call_target_user_data).second;
		    }()) {
			for (const auto index: translated.call_target_user_data) {
				lookup_key.static_state.push_back(index < user_data.size() ? user_data[index] : 0u);
			}
		}
		if (translated.unsupported) {
			// Remember the refusal: a skipped shader is dispatched again every frame, and
			// re-deriving the same answer costs as much as a compile each time.
			{
				std::unique_lock programs_lock(programs_mutex);
				unsupported.insert(lookup_key);
			}
			return ShaderProgram {};
		}
		if (!shader_clock && UsesShaderClock(translated.program)) {
			if (!ShaderFailureNonFatal()) {
				EXIT("S_MEMREALTIME needs shaderDeviceClock\n");
			}
			static std::atomic<uint32_t> reported = 0;
			if (reported.fetch_add(1) < 16) {
				LOGF("ProgramCache: skipping stage %u hash=0x%016" PRIx64
				     ": S_MEMREALTIME needs shaderDeviceClock\n",
				     static_cast<uint32_t>(stage), params.hash);
			}
			{
				std::unique_lock programs_lock(programs_mutex);
				unsupported.insert(lookup_key);
			}
			return ShaderProgram {};
		}
		if (entry == programs.end()) {
			{
				std::unique_lock programs_lock(programs_mutex);
				entry = programs.try_emplace(lookup_key,
				    ShaderRecompiler::IR::ExtractResourcePlan(translated.program)).first;
			}
			if (!MaterializeLocked(entry->second, runtime, entry->second.resources,
			        entry->second.specialization)) {
				if (!ShaderFailureNonFatal()) {
					EXIT("shader resource materialization failed\n");
				}
				static std::atomic<uint32_t> reported = 0;
				if (reported.fetch_add(1) < 16) {
					LOGF("ProgramCache: skipping stage %u hash=0x%016" PRIx64
					     ": resource materialization failed on first use (materialization line %d)\n",
					     static_cast<uint32_t>(stage), params.hash,
					     ShaderRecompiler::IR::LastIndirectImageFailureLine());
				}
				return ShaderProgram {};
			}
		}
		const auto push_start = push_data_cursor;
		entry->second.permutations.push_back(CompilePermutation(
		    stage_name, options, std::move(translated), entry->second.specialization, push_data_cursor));
		WarmRecordCompile(params, input_info, options, entry->second.specialization, push_start);
		const auto& permutation = entry->second.permutations.back();
		input_info.stage = {.program = &permutation.program, .resources = &entry->second.resources};
		permutation.program.bindings.AdvancePushData(push_data_cursor);

		std::array<size_t, static_cast<size_t>(ShaderType::TessellationEvaluation) + 1> counts {};
		for (const auto& [key, source]: programs) {
			counts[static_cast<size_t>(key.stage)] += source.permutations.size();
		}
		// Guest geometry shaders are compiled through the host mesh stage.
		std::printf("Shaders: VS %zu | PS %zu | CS %zu | GS %zu | LS %zu | HS %zu | TES %zu\n",
		            counts[static_cast<size_t>(ShaderType::Vertex)],
		            counts[static_cast<size_t>(ShaderType::Pixel)],
		            counts[static_cast<size_t>(ShaderType::Compute)],
		            counts[static_cast<size_t>(ShaderType::Mesh)],
		            counts[static_cast<size_t>(ShaderType::Local)],
		            counts[static_cast<size_t>(ShaderType::TessellationControl)],
		            counts[static_cast<size_t>(ShaderType::TessellationEvaluation)]);
		return permutation.handle;
	}

	// ---- KYTY_LOCAL_HACK: warm shader cache (KYTY_WARM_CACHE=<file>) ----------------------------
	// Every program compiled on the GPU thread appends its inputs (guest code, program key, compile
	// context, specialization, push data start) to the file. The next session translates and emits
	// those records on worker threads from startup, in the order they were first needed, so a draw
	// that meets one only creates the shader module. The file holds inputs only: the current
	// translator always produces the result, so a rebuilt emulator never uses stale code.
	// KYTY_WARM_THREADS (default 6) workers; KYTY_WARM_VERIFY=1 also compiles every prepared
	// permutation on the GPU thread and compares the SPIR-V.
	struct WarmRecord {
		ProgramKey                                   key;
		std::vector<uint32_t>                        code;
		std::vector<uint32_t>                        back_code;
		std::vector<uint32_t>                        user_data;
		uint32_t                                     wave_size      = 64;
		uint32_t                                     user_data_base = 0;
		uint32_t                                     push_start     = 0;
		uint8_t                                      input_kind     = 0; // 0 vertex, 1 pixel, 2 compute
		std::vector<uint8_t>                         input_bytes;
		ShaderRecompiler::IR::ResourceSpecialization specialization;
		// Async (KYTY_ASYNC_SHADERS): only the resource plan is needed (a new source); not stored.
		bool                                         plan_only = false;
		uint64_t                                     pending   = 0; // digest in warm_pending
	};
	struct PreparedPermutation {
		ShaderRecompiler::IR::ResourceSpecialization specialization;
		uint32_t                                     push_start = 0;
		ShaderRecompiler::IR::CompiledShaderInfo     info;
		std::vector<uint32_t>                        spirv;
		bool                                         used = false;
	};
	struct PreparedSource {
		std::unique_ptr<ShaderRecompiler::IR::ResourcePlan> plan;
		bool                                                plan_taken = false;
		std::vector<PreparedPermutation>                    permutations;
	};

	static constexpr uint64_t WarmMagic = 0x3130304d5241574bull; // "KWARM001"

	static std::array<uint32_t, 6> WarmLayout() {
		return {sizeof(ShaderVertexInputInfo), sizeof(ShaderPixelInputInfo),
		        sizeof(ShaderComputeInputInfo),
		        sizeof(ShaderRecompiler::IR::ResourceSpecialization::Buffer),
		        sizeof(ShaderRecompiler::IR::ResourceSpecialization::Image),
		        sizeof(ShaderRecompiler::IR::ResourceSpecialization::Sampler)};
	}

	struct WarmWriter {
		std::vector<uint8_t> bytes;
		void Raw(const void* data, size_t size) {
			const auto* begin = static_cast<const uint8_t*>(data);
			bytes.insert(bytes.end(), begin, begin + size);
		}
		template <typename T>
		void Pod(const T& value) {
			static_assert(std::is_trivially_copyable_v<T>);
			Raw(&value, sizeof(T));
		}
		template <typename T>
		void Vec(const std::vector<T>& values) {
			static_assert(std::is_trivially_copyable_v<T>);
			Pod(static_cast<uint32_t>(values.size()));
			Raw(values.data(), values.size() * sizeof(T));
		}
	};
	struct WarmReader {
		const uint8_t* data = nullptr;
		size_t         size = 0;
		size_t         pos  = 0;
		bool           ok   = true;
		void Raw(void* out, size_t count) {
			if (!ok || pos + count > size) {
				ok = false;
				return;
			}
			std::memcpy(out, data + pos, count);
			pos += count;
		}
		template <typename T>
		void Pod(T& value) {
			static_assert(std::is_trivially_copyable_v<T>);
			Raw(&value, sizeof(T));
		}
		template <typename T>
		void Vec(std::vector<T>& values) {
			static_assert(std::is_trivially_copyable_v<T>);
			uint32_t count = 0;
			Pod(count);
			if (!ok || static_cast<size_t>(count) * sizeof(T) > size - pos) {
				ok = false;
				return;
			}
			values.resize(count);
			Raw(values.data(), count * sizeof(T));
		}
	};

	static std::vector<uint8_t> SerializeWarm(const WarmRecord& r) {
		WarmWriter w;
		w.Pod(static_cast<uint32_t>(r.key.stage));
		w.Pod(r.key.hash);
		w.Pod(r.key.user_data_count);
		w.Pod(r.key.code_size);
		w.Vec(r.key.static_state);
		w.Vec(r.key.function_code);
		w.Vec(r.code);
		w.Vec(r.back_code);
		w.Vec(r.user_data);
		w.Pod(r.wave_size);
		w.Pod(r.user_data_base);
		w.Pod(r.push_start);
		w.Pod(r.input_kind);
		w.Vec(r.input_bytes);
		w.Vec(r.specialization.buffers);
		w.Vec(r.specialization.images);
		w.Vec(r.specialization.samplers);
		return std::move(w.bytes);
	}

	static bool DeserializeWarm(const uint8_t* data, size_t size, WarmRecord& r) {
		WarmReader rd {.data = data, .size = size};
		uint32_t   stage = 0;
		rd.Pod(stage);
		r.key.stage = static_cast<ShaderType>(stage);
		rd.Pod(r.key.hash);
		rd.Pod(r.key.user_data_count);
		rd.Pod(r.key.code_size);
		rd.Vec(r.key.static_state);
		rd.Vec(r.key.function_code);
		rd.Vec(r.code);
		rd.Vec(r.back_code);
		rd.Vec(r.user_data);
		rd.Pod(r.wave_size);
		rd.Pod(r.user_data_base);
		rd.Pod(r.push_start);
		rd.Pod(r.input_kind);
		rd.Vec(r.input_bytes);
		rd.Vec(r.specialization.buffers);
		rd.Vec(r.specialization.images);
		rd.Vec(r.specialization.samplers);
		static constexpr std::array<size_t, 3> InputSizes = {
		    sizeof(ShaderVertexInputInfo), sizeof(ShaderPixelInputInfo), sizeof(ShaderComputeInputInfo)};
		return rd.ok && rd.pos == size && r.input_kind < 3 &&
		       r.input_bytes.size() == InputSizes[r.input_kind] && !r.code.empty();
	}

	void WarmStart() {
		warm_async = [] {
			const char* v = std::getenv("KYTY_ASYNC_SHADERS");
			return v != nullptr && std::strcmp(v, "0") != 0;
		}();
		const char* path = std::getenv("KYTY_WARM_CACHE");
		if (path == nullptr || *path == 0) {
			if (warm_async) {
				path = "";
			} else {
				return;
			}
		}
		warm_active = true;
		warm_path = path;
		warm_verify = [] {
			const char* v = std::getenv("KYTY_WARM_VERIFY");
			return v != nullptr && std::strcmp(v, "0") != 0;
		}();
		// Load the records written by earlier sessions.
		std::vector<uint8_t> file;
		if (FILE* f = *path != 0 ? std::fopen(path, "rb") : nullptr; f != nullptr) {
			std::fseek(f, 0, SEEK_END);
			const long size = std::ftell(f);
			std::fseek(f, 0, SEEK_SET);
			if (size > 0) {
				file.resize(static_cast<size_t>(size));
				if (std::fread(file.data(), 1, file.size(), f) != file.size()) {
					file.clear();
				}
			}
			std::fclose(f);
		}
		const auto layout      = WarmLayout();
		const size_t header    = sizeof(uint64_t) + sizeof(layout);
		bool         valid     = file.size() >= header;
		if (valid) {
			uint64_t magic = 0;
			std::memcpy(&magic, file.data(), sizeof(magic));
			valid = magic == WarmMagic &&
			        std::memcmp(file.data() + sizeof(magic), layout.data(), sizeof(layout)) == 0;
		}
		size_t loaded = 0, bad = 0;
		if (valid) {
			size_t pos = header;
			while (pos + sizeof(uint32_t) <= file.size()) {
				uint32_t length = 0;
				std::memcpy(&length, file.data() + pos, sizeof(length));
				pos += sizeof(length);
				if (length == 0 || pos + length > file.size()) {
					bad++;
					break;
				}
				const uint64_t digest = XXH3_64bits(file.data() + pos, length);
				WarmRecord     record;
				if (warm_seen.insert(digest).second &&
				    DeserializeWarm(file.data() + pos, length, record)) {
					warm_jobs.push_back(std::move(record));
					loaded++;
				} else if (!warm_seen.contains(digest)) {
					bad++;
				}
				pos += length;
			}
		}
		// Rewrite the file compacted (no duplicates, current layout), then append to it.
		warm_file = *path != 0 ? std::fopen(path, "wb") : nullptr;
		if (warm_file != nullptr) {
			std::fwrite(&WarmMagic, sizeof(WarmMagic), 1, warm_file);
			std::fwrite(layout.data(), sizeof(layout), 1, warm_file);
			for (const auto& record: warm_jobs) {
				const auto bytes  = SerializeWarm(record);
				const auto length = static_cast<uint32_t>(bytes.size());
				std::fwrite(&length, sizeof(length), 1, warm_file);
				std::fwrite(bytes.data(), 1, bytes.size(), warm_file);
			}
			std::fflush(warm_file);
		}
		::printf("Warm cache: %s, %zu records loaded%s (%zu bad), file %s\n", path, loaded,
		         valid || file.empty() ? "" : " (layout changed: discarded)", bad,
		         warm_file != nullptr ? "open" : "NOT writable");
		std::fflush(stdout);
		warm_initial = warm_jobs.size();
		for (auto& job: warm_jobs) {
			warm_queue.push_back(std::move(job));
		}
		warm_jobs.clear();
		if (warm_initial == 0 && !warm_async) {
			return;
		}
		uint32_t threads = 6;
		if (const char* v = std::getenv("KYTY_WARM_THREADS"); v != nullptr) {
			threads = std::clamp<uint32_t>(static_cast<uint32_t>(std::strtoul(v, nullptr, 10)), 1u, 16u);
		}
		warm_start_time = std::chrono::steady_clock::now();
		for (uint32_t i = 0; i < threads; i++) {
			warm_threads.emplace_back([this] { WarmWorker(); });
		}
	}

	void WarmStop() {
		{
			std::lock_guard lock(warm_mutex);
			warm_stop.store(true, std::memory_order_relaxed);
		}
		warm_cv.notify_all();
		for (auto& thread: warm_threads) {
			if (thread.joinable()) {
				thread.join();
			}
		}
		warm_threads.clear();
		if (warm_file != nullptr) {
			std::fclose(warm_file);
			warm_file = nullptr;
		}
	}

	void WarmWorker() {
		for (;;) {
			WarmRecord record;
			{
				std::unique_lock lock(warm_mutex);
				warm_cv.wait(lock, [this] {
					return warm_stop.load(std::memory_order_relaxed) || !warm_queue.empty();
				});
				if (warm_stop.load(std::memory_order_relaxed)) {
					return;
				}
				record = std::move(warm_queue.front());
				warm_queue.pop_front();
			}
			const bool ok = WarmPrepare(record);
			if (record.pending != 0) {
				std::lock_guard lock(warm_mutex);
				warm_pending.erase(record.pending);
				if (!ok) {
					warm_async_rejected.insert(record.pending);
				}
				(ok ? warm_async_done : warm_async_failed)++;
				continue;
			}
			ok ? warm_done.fetch_add(1) : warm_failed.fetch_add(1);
			if (warm_done.load() + warm_failed.load() == warm_initial) {
				{
					const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
					                    std::chrono::steady_clock::now() - warm_start_time)
					                    .count();
					::printf("Warm cache: prepared %zu of %zu records in %lld ms (%zu failed); native SRT "
					         "compile %.1f ms of thread time\n",
					         warm_done.load(), warm_initial, static_cast<long long>(ms),
					         warm_failed.load(), warm_native_us.load() / 1e3);
					std::fflush(stdout);
				}
			}
		}
	}

	bool WarmPrepare(const WarmRecord& record) {
		ShaderVertexInputInfo  vertex {};
		ShaderPixelInputInfo   pixel {};
		ShaderComputeInputInfo compute {};
		ShaderStageInputInfo   stage_input {};
		switch (record.input_kind) {
			case 0:
				std::memcpy(&vertex, record.input_bytes.data(), sizeof(vertex));
				vertex.stage       = {};
				stage_input.vertex = &vertex;
				break;
			case 1:
				std::memcpy(&pixel, record.input_bytes.data(), sizeof(pixel));
				pixel.stage       = {};
				stage_input.pixel = &pixel;
				break;
			default:
				std::memcpy(&compute, record.input_bytes.data(), sizeof(compute));
				compute.stage       = {};
				stage_input.compute = &compute;
				break;
		}
		ShaderRecompiler::CompileOptions options;
		options.stage           = record.key.stage;
		options.shader_hash     = record.key.hash;
		options.user_data       = record.user_data;
		options.back_code       = record.back_code;
		options.dump_ir         = false;
		options.early_dump      = false;
		options.dump_label      = "Warm";
		options.input_info      = stage_input;
		options.user_data_base  = record.user_data_base;
		options.wave_size       = record.wave_size;
		options.non_fatal       = true;
		options.bindless_images = bindless_images;
		// Records never hold inlined calls through user data (see WarmRecord writer).
		options.read_code = [](uint64_t) { return std::vector<uint32_t> {}; };
		const auto code   = record.key.function_code.empty()
		                        ? std::span<const uint32_t>(record.code)
		                        : std::span<const uint32_t>(record.key.function_code);
		auto translated   = ShaderRecompiler::TranslateProgram(code, options);
		// Shader-clock programs are left to the GPU thread: KYTY_WARM_VERIFY found the frame pacer
		// CS 0x7c5c0785877c0999 compiling differently there (1 of 987 permutations).
		if (translated.unsupported || !translated.call_target_user_data.empty() ||
		    UsesShaderClock(translated.program)) {
			return false;
		}
		bool need_plan = false;
		{
			std::lock_guard lock(warm_mutex);
			auto&           source = prepared[record.key];
			need_plan              = source.plan == nullptr && !source.plan_taken;
		}
		std::unique_ptr<ShaderRecompiler::IR::ResourcePlan> plan;
		if (need_plan) {
			plan = std::make_unique<ShaderRecompiler::IR::ResourcePlan>(
			    ShaderRecompiler::IR::ExtractResourcePlan(translated.program));
			// The GPU thread compiles a plan to native code on its 8th use (SrtWalker::BindNative),
			// up to tens of ms in gameplay; the plan is not shared yet, so compile it here.
			static auto& native = Common::LiveSwitches::Get("KYTY_SRT_NATIVE", 1);
			static auto& warm_native = Common::LiveSwitches::Get("KYTY_WARM_NATIVE", 1);
			if (native.load(std::memory_order_relaxed) != 0 &&
			    warm_native.load(std::memory_order_relaxed) != 0 &&
			    ShaderRecompiler::IR::SrtNativeCode::Supported()) {
				const auto start         = std::chrono::steady_clock::now();
				plan->native_attempted   = true;
				plan->native_code        = ShaderRecompiler::IR::SrtNativeCode::Compile(*plan);
				warm_native_us.fetch_add(static_cast<uint64_t>(
				    std::chrono::duration_cast<std::chrono::microseconds>(
				        std::chrono::steady_clock::now() - start)
				        .count()));
			}
		}
		if (record.plan_only) {
			std::lock_guard lock(warm_mutex);
			auto&           source = prepared[record.key];
			if (plan != nullptr && source.plan == nullptr && !source.plan_taken) {
				source.plan = std::move(plan);
			}
			return true;
		}
		auto result = ShaderRecompiler::CompileProgram(std::move(translated), options,
		                                               record.specialization, record.push_start);
		std::lock_guard lock(warm_mutex);
		auto&           source = prepared[record.key];
		if (plan != nullptr && source.plan == nullptr && !source.plan_taken) {
			source.plan = std::move(plan);
		}
		source.permutations.push_back({.specialization = record.specialization,
		                               .push_start     = record.push_start,
		                               .info           = std::move(result.program).TakeCompiledInfo(),
		                               .spirv          = std::move(result.spirv)});
		return true;
	}

	// A prepared program for this draw: creates the source entry from the prepared plan if needed,
	// then takes a prepared permutation of the entry's specialization. nullopt: compile as usual.
	template <typename InputInfo, typename EntryIterator>
	std::optional<ShaderProgram> TryPrepared(InputInfo& input_info, uint32_t& push_data_cursor,
	                                         EntryIterator& entry,
	                                         const ShaderRecompiler::IR::SrtRuntime& runtime,
	                                         ShaderType stage, uint64_t hash) {
		if (!warm_active) {
			return std::nullopt;
		}
		std::unique_lock lock(warm_mutex);
		const auto found = prepared.find(lookup_key);
		if (found == prepared.end()) {
			return std::nullopt;
		}
		auto& source = found->second;
		if (entry == programs.end()) {
			if (source.plan == nullptr) {
				return std::nullopt;
			}
			auto plan         = std::move(source.plan);
			source.plan_taken = true;
			{
				std::unique_lock programs_lock(programs_mutex);
				entry = programs.try_emplace(lookup_key, std::move(*plan)).first;
			}
			if (!MaterializeLocked(entry->second, runtime, entry->second.resources,
			        entry->second.specialization)) {
				if (!ShaderFailureNonFatal()) {
					EXIT("shader resource materialization failed\n");
				}
				LOGF("ProgramCache: skipping stage %u hash=0x%016" PRIx64
				     ": resource materialization failed on first use (warm)\n",
				     static_cast<uint32_t>(stage), hash);
				return ShaderProgram {};
			}
		}
		const auto start = push_data_cursor;
		auto       match = std::ranges::find_if(source.permutations, [&](const PreparedPermutation& p) {
            return !p.used && p.push_start == start &&
                   p.specialization == entry->second.specialization;
        });
		if (match == source.permutations.end()) {
			warm_misses++;
			return std::nullopt;
		}
		match->used       = true;
		auto info         = std::move(match->info);
		auto spirv        = std::move(match->spirv);
		lock.unlock();
		if (warm_verify) {
			warm_verify_pending = spirv;
		}
		const auto module = CompileSPV(spirv, device);
		EXIT_IF(module == nullptr);
		entry->second.permutations.push_back({.specialization = entry->second.specialization,
		                                      .program        = std::move(info),
		                                      .handle         = {.id = ++next_shader_id, .module = module}});
		const auto& permutation = entry->second.permutations.back();
		input_info.stage = {.program = &permutation.program, .resources = &entry->second.resources};
		permutation.program.bindings.AdvancePushData(push_data_cursor);
		warm_hits++;
		return permutation.handle;
	}

	template <typename InputInfo>
	static WarmRecord MakeWarmRecord(const ProgramKey& key, const ShaderParams& params,
	                                 const InputInfo& input_info,
	                                 const ShaderRecompiler::CompileOptions& options,
	                                 const ShaderRecompiler::IR::ResourceSpecialization& specialization,
	                                 uint32_t push_start) {
		WarmRecord record;
		record.key = key;
		record.code.assign(params.code.begin(), params.code.end());
		record.back_code.assign(params.back_code.begin(), params.back_code.end());
		record.user_data.assign(options.user_data.begin(), options.user_data.end());
		record.wave_size      = options.wave_size;
		record.user_data_base = options.user_data_base;
		record.push_start     = push_start;
		auto copy             = input_info;
		copy.stage            = {};
		static_assert(std::is_trivially_copyable_v<InputInfo>);
		record.input_kind = std::is_same_v<InputInfo, ShaderVertexInputInfo>  ? 0
		                    : std::is_same_v<InputInfo, ShaderPixelInputInfo> ? 1
		                                                                       : 2;
		record.input_bytes.resize(sizeof(copy));
		std::memcpy(record.input_bytes.data(), &copy, sizeof(copy));
		record.specialization = specialization;
		return record;
	}

	void WarmWrite(const std::vector<uint8_t>& bytes) {
		if (warm_file == nullptr || !warm_seen.insert(XXH3_64bits(bytes.data(), bytes.size())).second) {
			return;
		}
		const auto length = static_cast<uint32_t>(bytes.size());
		std::fwrite(&length, sizeof(length), 1, warm_file);
		std::fwrite(bytes.data(), 1, bytes.size(), warm_file);
		std::fflush(warm_file);
		warm_recorded++;
	}

	// KYTY_ASYNC_SHADERS=1: a graphics program this session has not built is compiled on a worker
	// while its draws are skipped (true). Never for compute, a program with inlined calls, or a
	// source whose plan says it writes memory (those compile here as before: false).
	template <typename InputInfo, typename EntryIterator>
	bool AsyncCompile(const ShaderParams& params, const InputInfo& input_info,
	                  const ShaderRecompiler::CompileOptions& options, EntryIterator entry,
	                  uint32_t push_start, ShaderType stage) {
		if (!warm_async || stage == ShaderType::Compute || call_targets.contains(params.hash)) {
			return false;
		}
		const bool new_source = entry == programs.end();
		if (!new_source) {
			const auto& info = entry->second.resource_plan.info;
			if (std::ranges::any_of(info.buffers, [](const auto& b) { return b.written; }) ||
			    std::ranges::any_of(info.images, [](const auto& i) { return i.written; })) {
				return false;
			}
		}
		auto record = MakeWarmRecord(lookup_key, params, input_info, options,
		                             new_source ? ShaderRecompiler::IR::ResourceSpecialization {}
		                                        : entry->second.specialization,
		                             push_start);
		record.plan_only  = new_source;
		// One job per program key (and specialization + push start for a permutation), whatever
		// the user data of the draws that wait for it.
		uint64_t pending = ProgramKeyHash {}(lookup_key) * 0x9e3779b97f4a7c15ull + (new_source ? 1u : 2u);
		if (!new_source) {
			const auto& spec = record.specialization;
			pending ^= XXH3_64bits(spec.buffers.data(), spec.buffers.size() * sizeof(spec.buffers[0]));
			pending ^= XXH3_64bits(spec.images.data(), spec.images.size() * sizeof(spec.images[0])) << 1u;
			pending ^= XXH3_64bits(spec.samplers.data(), spec.samplers.size() * sizeof(spec.samplers[0])) << 2u;
			pending ^= static_cast<uint64_t>(push_start) << 48u;
			WarmWrite(SerializeWarm(record));
		}
		record.pending = pending == 0 ? 1 : pending;
		{
			std::lock_guard lock(warm_mutex);
			if (warm_async_rejected.contains(record.pending)) {
				return false;
			}
			if (warm_pending.insert(record.pending).second) {
				warm_queue.push_back(std::move(record));
				warm_async_queued++;
			}
		}
		warm_cv.notify_one();
		warm_async_skips++;
		return true;
	}

	template <typename InputInfo>
	void WarmRecordCompile(const ShaderParams& params, const InputInfo& input_info,
	                       const ShaderRecompiler::CompileOptions& options,
	                       const ShaderRecompiler::IR::ResourceSpecialization& specialization,
	                       uint32_t push_start) {
		if (warm_file == nullptr || call_targets.contains(params.hash)) {
			return;
		}
		WarmWrite(SerializeWarm(
		    MakeWarmRecord(lookup_key, params, input_info, options, specialization, push_start)));
	}


	// The compile options the GPU-thread path builds for this stage (see Get).
	template <typename InputInfo>
	static ShaderRecompiler::CompileOptions AsyncOptions(const ShaderParams& params,
	                                                     const InputInfo& input_info, ShaderType stage,
	                                                     std::span<const uint32_t> user_data) {
		ShaderRecompiler::CompileOptions options;
		options.stage       = stage;
		options.shader_hash = params.hash;
		options.user_data   = user_data;
		options.back_code   = params.back_code;
		if constexpr (std::is_same_v<InputInfo, ShaderVertexInputInfo>) {
			options.user_data_base = 8;
			options.wave_size      = input_info.wave_size;
			if (stage == ShaderType::Mesh || stage == ShaderType::TessellationControl) {
				options.user_data_base = 0;
				options.wave_size = stage == ShaderType::Mesh ? input_info.mesh.wave_size : 64u;
			}
		} else {
			options.wave_size = input_info.wave_size;
		}
		return options;
	}

	void WarmReport() {
		if (!warm_active) {
			return;
		}
		const auto now = std::chrono::steady_clock::now();
		if (now - warm_report < std::chrono::seconds(5)) {
			return;
		}
		warm_report = now;
		size_t async_done = 0, async_failed = 0;
		{
			std::lock_guard lock(warm_mutex);
			async_done   = warm_async_done;
			async_failed = warm_async_failed;
			warm_async_done = warm_async_failed = 0;
		}
		::printf("Warm cache (5 s): %" PRIu64 " used, %" PRIu64 " no matching permutation, %" PRIu64
		         " recorded, verify %" PRIu64 " ok / %" PRIu64 " MISMATCH; prepared %zu/%zu; async: %" PRIu64
		         " skipped draws, %" PRIu64 " queued, %zu done, %zu failed\n",
		         warm_hits, warm_misses, warm_recorded, warm_verify_ok, warm_verify_bad,
		         warm_done.load(), warm_initial, warm_async_skips, warm_async_queued, async_done,
		         async_failed);
		std::fflush(stdout);
		warm_hits = warm_misses = warm_recorded = warm_async_skips = warm_async_queued = 0;
	}

	std::string                                                 warm_path;
	FILE*                                                       warm_file = nullptr;
	bool                                                        warm_verify = false;
	std::vector<WarmRecord>                                     warm_jobs;
	std::deque<WarmRecord>                                      warm_queue;
	std::condition_variable                                     warm_cv;
	std::unordered_set<uint64_t>                                warm_pending;
	std::unordered_set<uint64_t>                                warm_async_rejected;
	size_t                                                      warm_initial = 0;
	bool                                                        warm_active  = false;
	bool                                                        warm_async   = false;
	size_t                                                      warm_async_done = 0, warm_async_failed = 0;
	uint64_t                                                    warm_async_skips = 0, warm_async_queued = 0;
	std::unordered_set<uint64_t>                                warm_seen;
	std::vector<std::thread>                                    warm_threads;
	std::atomic<bool>                                           warm_stop {false};
	std::atomic<size_t>                                         warm_next {0};
	std::atomic<size_t>                                         warm_done {0};
	std::atomic<size_t>                                         warm_failed {0};
	std::atomic<size_t>                                         warm_finished {0};
	std::atomic<uint64_t>                                       warm_native_us {0};
	std::chrono::steady_clock::time_point                       warm_start_time;
	std::chrono::steady_clock::time_point                       warm_report = std::chrono::steady_clock::now();
	std::mutex                                                  warm_mutex;
	std::unordered_map<ProgramKey, PreparedSource, ProgramKeyHash> prepared;
	std::optional<std::vector<uint32_t>>                        warm_verify_pending;
	uint64_t warm_hits = 0, warm_misses = 0, warm_recorded = 0, warm_verify_ok = 0, warm_verify_bad = 0;

	ProgramCache(vk::Device device, bool shader_clock, bool bindless_images)
	    : device(device), shader_clock(shader_clock), bindless_images(bindless_images) {
		lookup_key.static_state.reserve(MaxStaticKeyWords);
		WarmStart();
	}
	~ProgramCache() {
		WarmStop();
		for (const auto& [key, entry]: programs) {
			(void)key;
			for (const auto& permutation: entry.permutations) {
				device.destroyShaderModule(permutation.handle.module, nullptr);
			}
		}
	}

	// Shader function expansion decodes the shader and every callee, and a shader with table
	// calls can be dispatched hundreds of times a frame; the expander keeps what it can reuse.
	void ExpandShaderFunctions(const ShaderParams& params, std::span<const uint32_t> user_data,
	                           uint32_t wave_size, std::vector<uint32_t>& expanded) {
		// KYTY_LOCAL_HACK KYTY_EXPAND_MEMO (live, default 0: em1 no gain): within one guest submission the same
		// shader with the same base, wave size and user data expands the same way (what the game
		// wrote before submitting is what its commands see, as PrepareBda assumes), so the call
		// resolution and the callee re-read/compare run once per submission, not per dispatch
		// (~2.5% of the CP, pf5 perf 2026-10-06).
		static auto& memo_on    = Common::LiveSwitches::Get("KYTY_EXPAND_MEMO", 0);
		const auto   submission = g_guest_submission_seq.load(std::memory_order_relaxed);
		const bool   use_memo   = memo_on.load(std::memory_order_relaxed) != 0;
		ExpandMemo*  memo       = nullptr;
		if (use_memo) {
			memo = &expand_memo[params.hash];
			if (memo->submission == submission && memo->base == params.Base() &&
			    memo->wave_size == wave_size && memo->code_size == params.code.size() &&
			    std::ranges::equal(memo->user_data, user_data)) {
				expanded = memo->expanded;
				return;
			}
		}
		std::string reason;
		const bool  ok = function_expander.Expand(
		    params.code, params.Base(), user_data,
		    [](uint64_t address, std::span<uint32_t> words) {
			    return ReadShaderGuestMemoryRaw(nullptr, address, words);
		    },
		    expanded, reason, wave_size, params.hash);
		if (memo != nullptr) {
			memo->submission = submission;
			memo->base       = params.Base();
			memo->wave_size  = wave_size;
			memo->code_size  = params.code.size();
			memo->user_data.assign(user_data.begin(), user_data.end());
			memo->expanded   = expanded;
		}
		if (!ok) {
			static std::atomic<uint32_t> reports {0};
			if (reports.fetch_add(1) < 16) {
				::printf("Shader function expansion hash=%016" PRIx64 ": %s\n", params.hash,
				         reason.c_str());
			}
		}
	}

	struct ExpandMemo {
		uint64_t              submission = 0;
		uint64_t              base       = 0;
		uint32_t              wave_size  = 0;
		size_t                code_size  = 0;
		std::vector<uint32_t> user_data;
		std::vector<uint32_t> expanded;
	};
	std::unordered_map<uint64_t, ExpandMemo> expand_memo;

	std::unordered_map<ProgramKey, SourceEntry, ProgramKeyHash> programs;
	// KYTY_DRAW_PREP: taken exclusively by the GPU thread when it inserts into programs,
	// unsupported or call_targets (entries are never erased), shared by the scanner's lookups.
	std::shared_mutex                                            programs_mutex;
	ScanStore                                                    scan_store;
	std::unordered_set<ProgramKey, ProgramKeyHash>              unsupported;
	ShaderRecompiler::Decoder::ShaderFunctionExpander          function_expander;
	ProgramKey                                                  lookup_key;
	ProgramKey                                                  worker_key; // KYTY_PARALLEL_MATERIALIZE
	// KYTY_LOOKAHEAD: the next draw's stages, materialized on the worker into memo slots.
	std::array<LookaheadSide, 2>                                lookahead_side; // 0 pixel, 1 vertex
	std::array<ProgramKey, 2>                                   lookahead_key;
	bool                                                        lookahead_pending = false;
	std::function<void()>                                       lookahead_job;
	// started draws, materialized stages, refused stages, adopted slots, memo hits on them,
	// trusted slots, trusted hits
	std::array<uint64_t, 8> lookahead_stats {}; // [7]: trusted slots whose reads changed (=3)
	std::chrono::steady_clock::time_point lookahead_start_time;
	uint64_t                lookahead_start_writes = 0, lookahead_start_commands = 0, lookahead_start_gpu = 0;
	uint64_t                lookahead_wait_ns    = 0;
	uint64_t                lookahead_prepare_ns = 0; // GetGraphicsPrograms: next draw's PrepareProgram + LookaheadStart
	uint64_t                lookahead_epoch   = 1; // the draw being prepared (GetGraphicsPrograms)
	bool                    lookahead_trusted = false;
	std::chrono::steady_clock::time_point lookahead_report = std::chrono::steady_clock::now();
	ProgramKey                                                  cp_pre_key;
	std::unordered_map<uint64_t, uint32_t>                      stage_cost; // KYTY_PARALLEL_SWAP=2
	// Research: per shader hash, the user-data dwords holding its inlined call targets.
	std::unordered_map<uint64_t, std::vector<uint32_t>>         call_targets;
	// KYTY_RESOURCE_MEMO bookkeeping (GPU thread).
	uint64_t                              memo_clock  = 0;
	ScanResult                            scan_taken;       // KYTY_DRAW_PREP: GPU thread's current result
	bool                                  scan_taken_valid = false;
	uint64_t                              memo_hits   = 0;
	uint64_t                              memo_misses = 0;
	std::array<uint64_t, 6>               memo_miss_kinds {};
	// Research: 0 cold, 1-3 user data 1/2/3-4 dwords (unclassified), 4 5+ dwords, 5 reads;
	// 6 pass same, 7 pass differ, 8 reloc same, 9 reloc differ.
	int                                   memo_last_class = -1;
	uint32_t                              memo_report_calls = 0;
	std::array<uint64_t, 10>              memo_class_ns {};
	std::array<uint64_t, 10>              memo_class_calls {};
	std::array<uint64_t, 4>               memo_delta_kinds {};
	std::chrono::steady_clock::time_point plan_stats_report = std::chrono::steady_clock::now();
	uint64_t                              rebase_hits     = 0;
	uint64_t                              rebase_verified = 0;
	uint64_t                              rebase_failed   = 0;
	uint64_t                              rebase_disabled = 0;
	std::vector<uint32_t>                 rebase_old;
	std::vector<uint32_t>                 rebase_new;
	std::chrono::steady_clock::time_point memo_report = std::chrono::steady_clock::now();
	vk::Device                                                  device;
	bool                                                        shader_clock = false;
	bool                                                        bindless_images = false;
	uint64_t                                                    next_shader_id = 0;
};

PipelineCache::PipelineCache(GraphicContext& graphics)
    : m_graphics(graphics),
      m_program_cache(std::make_unique<ProgramCache>(
          graphics.device, graphics.shader_device_clock_enabled,
          graphics.bindless_enabled && Config::BindlessImagesEnabled())) {
	EXIT_NOT_IMPLEMENTED(!Common::Thread::IsMainThread());
	// The shader device clock shares the timestamp domain: 10 ns on RADV (100 MHz, the console's
	// rate), 1 ns on NVIDIA.
	const auto period_ns = graphics.GetPhysicalDeviceProperties().limits.timestampPeriod;
	const auto divisor = period_ns > 0.0f ? static_cast<uint32_t>(std::lround(10.0f / period_ns)) : 8u;
	ShaderRecompiler::Spirv::SetShaderClockDivisor(divisor);
	// S_MEMREALTIME and the CPU-side 100 MHz reference clock (EOP timestamps, labels) must share
	// an epoch: games compare GPU clock reads against deadlines the CPU wrote.
	if (graphics.calibrated_timestamps_enabled &&
	    VULKAN_HPP_DEFAULT_DISPATCHER.vkGetCalibratedTimestampsKHR != nullptr) {
		VkCalibratedTimestampInfoKHR info {VK_STRUCTURE_TYPE_CALIBRATED_TIMESTAMP_INFO_KHR};
		info.timeDomain       = VK_TIME_DOMAIN_DEVICE_KHR;
		uint64_t device_ticks = 0;
		uint64_t deviation    = 0;
		const auto reference  = Sync::ReadReferenceClock();
		if (VULKAN_HPP_DEFAULT_DISPATCHER.vkGetCalibratedTimestampsKHR(
		        graphics.device, 1, &info, &device_ticks, &deviation) == VK_SUCCESS) {
			const auto offset = reference - device_ticks / std::max(divisor, 1u);
			ShaderRecompiler::Spirv::SetShaderClockOffset(offset);
			PipelineCacheLog("Shader clock: divisor {}, epoch offset {} ticks", divisor,
			                 static_cast<int64_t>(offset));
		}
	}
	InitializeDriverCache();
	if (g_async_pipelines) {
		const auto threads = std::clamp(std::thread::hardware_concurrency() / 4u, 1u, 4u);
		m_compiler         = std::make_unique<PipelineCompiler>(threads);
		PipelineCacheLog("Vulkan pipelines: compiled on {} worker threads (wait {} ms)", threads,
		                 g_pipeline_wait.count());
	}
}

PipelineCache::~PipelineCache() {
	Save();
	auto destroy = [this](const auto& pipelines) {
		for (const auto& [key, pipeline]: pipelines) {
			(void)key;
			if (pipeline->pending && pipeline->pending->done.load(std::memory_order_acquire)) {
				m_graphics.device.destroyPipeline(pipeline->pending->pipeline, nullptr);
			}
			m_graphics.device.destroyPipeline(pipeline->pipeline, nullptr);
			m_graphics.device.destroyPipelineLayout(pipeline->pipeline_layout, nullptr);
			m_graphics.device.destroyDescriptorSetLayout(pipeline->descriptor_set_layout, nullptr);
		}
	};
	destroy(m_graphics_pipelines);
	destroy(m_compute_pipelines);
	if (m_driver_cache != nullptr) {
		m_graphics.device.destroyPipelineCache(m_driver_cache, nullptr);
	}
}

void PipelineCache::InitializeDriverCache() {
	const auto title_id = PipelineCacheTitleId();
	if (title_id.empty()) {
		return;
	}
	if (KYTY_BUILD != KYTY_BUILD_RELEASE) {
		PipelineCacheLog("Vulkan pipeline cache: disabled (non-Release build)");
		return;
	}
	const std::string_view git_hash     = KYTY_GIT_HASH;
	const std::string_view git_revision = KYTY_GIT_REVISION;
	if (git_hash == "unknown" || git_revision == "unknown") {
		PipelineCacheLog("Vulkan pipeline cache: disabled (unknown git revision)");
		return;
	}
	if (git_hash.ends_with("-dirty")) {
		PipelineCacheLog("Vulkan pipeline cache: disabled (dirty build)");
		return;
	}

	m_driver_cache_path     = std::filesystem::path("_PipelineCache") / (title_id + ".bin");
	const auto path         = Common::PathToString(m_driver_cache_path);
	const bool cache_exists = Common::File::IsFileExisting(m_driver_cache_path);
	if (cache_exists) {
		PipelineCacheLog("Vulkan pipeline cache: loading {}", path);
	} else {
		PipelineCacheLog("Vulkan pipeline cache: initializing {}", path);
	}
	std::vector<uint8_t> initial_data;
	if (cache_exists) {
		Common::File file(m_driver_cache_path, Common::File::Mode::Read);
		const auto   file_size = file.IsInvalid() ? 0 : file.Size();
		const auto   signature = DriverCacheSignature(m_graphics.GetPhysicalDeviceProperties());
		if (file_size >= signature.size() + sizeof(uint64_t) &&
		    file_size <= std::numeric_limits<uint32_t>::max()) {
			std::string cached_signature(signature.size(), '\0');
			uint64_t    payload_hash = 0;
			initial_data.resize(file_size - signature.size() - sizeof(payload_hash));
			uint32_t signature_read = 0;
			uint32_t hash_read      = 0;
			uint32_t payload_read   = 0;
			file.Read(cached_signature.data(), static_cast<uint32_t>(cached_signature.size()),
			          &signature_read);
			file.Read(&payload_hash, sizeof(payload_hash), &hash_read);
			file.Read(initial_data.data(), static_cast<uint32_t>(initial_data.size()),
			          &payload_read);
			file.Close();
			if (signature_read != cached_signature.size() || hash_read != sizeof(payload_hash) ||
			    payload_read != initial_data.size() ||
			    !DriverCacheSignatureMatches(cached_signature, signature) ||
			    XXH3_64bits(initial_data.data(), initial_data.size()) != payload_hash) {
				initial_data.clear();
				PipelineCacheLog(
				    "Vulkan pipeline cache: invalidating {} (driver, emulator, or data mismatch)",
				    path);
			} else if (cached_signature != signature) {
				PipelineCacheLog("Vulkan pipeline cache: keeping {} from another emulator revision",
				                 path);
			}
		} else {
			file.Close();
			PipelineCacheLog("Vulkan pipeline cache: invalidating {} (invalid file size)", path);
		}
	}

	vk::PipelineCacheCreateInfo create {};
	create.initialDataSize = initial_data.size();
	create.pInitialData    = initial_data.empty() ? nullptr : initial_data.data();
	auto result = m_graphics.device.createPipelineCache(&create, nullptr, &m_driver_cache);
	if (result != vk::Result::eSuccess && !initial_data.empty()) {
		PipelineCacheLog("Vulkan pipeline cache: driver rejected {} ({}); starting empty", path,
		                 vk::to_string(result));
		initial_data.clear();
		create.initialDataSize = 0;
		create.pInitialData    = nullptr;
		result = m_graphics.device.createPipelineCache(&create, nullptr, &m_driver_cache);
	}
	if (result != vk::Result::eSuccess) {
		PipelineCacheLog("Vulkan pipeline cache: disabled ({})", vk::to_string(result));
		m_driver_cache = nullptr;
		return;
	}
	if (!initial_data.empty()) {
		PipelineCacheLog("Vulkan pipeline cache: loaded {} bytes from {}", initial_data.size(),
		                 path);
	} else {
		PipelineCacheLog("Vulkan pipeline cache: initialized empty");
	}
}

void PipelineCache::Save() {
	// No worker may use the driver cache once it is destroyed below; later pipelines are
	// compiled on the GPU thread (FinishPending).
	if (m_compiler != nullptr) {
		m_compiler->Stop();
	}
	if (m_driver_cache == nullptr) {
		return;
	}

	size_t               size = 0;
	vk::Result           result;
	std::vector<uint8_t> payload;
	for (uint32_t attempt = 0; attempt < 3; attempt++) {
		size   = 0;
		result = m_graphics.device.getPipelineCacheData(m_driver_cache, &size, nullptr);
		if (result != vk::Result::eSuccess || size == 0 ||
		    size > std::numeric_limits<uint32_t>::max()) {
			break;
		}
		payload.resize(size);
		result = m_graphics.device.getPipelineCacheData(m_driver_cache, &size, payload.data());
		if (result != vk::Result::eIncomplete) {
			break;
		}
	}
	if (result != vk::Result::eSuccess || size == 0 ||
	    size > std::numeric_limits<uint32_t>::max()) {
		PipelineCacheLog("Vulkan pipeline cache: save failed ({}, {} bytes)",
		                 vk::to_string(result), size);
		return;
	}
	payload.resize(size);
	auto       prefix       = DriverCacheSignature(m_graphics.GetPhysicalDeviceProperties());
	const auto payload_hash = XXH3_64bits(payload.data(), payload.size());
	prefix.append(reinterpret_cast<const char*>(&payload_hash), sizeof(payload_hash));
	if (!Common::File::CreateDirectories(m_driver_cache_path.parent_path())) {
		PipelineCacheLog("Vulkan pipeline cache: failed to create cache directory");
		return;
	}
	auto temp_path = m_driver_cache_path;
	temp_path += ".tmp";
	Common::File file;
	uint32_t     prefix_written  = 0;
	uint32_t     payload_written = 0;
	if (file.Create(temp_path)) {
		file.Write(prefix.data(), static_cast<uint32_t>(prefix.size()), &prefix_written);
		file.Write(payload.data(), static_cast<uint32_t>(payload.size()), &payload_written);
	}
	const bool flushed = !file.IsInvalid() && file.Flush();
	file.Close();
	if (prefix_written != prefix.size() || payload_written != payload.size() || !flushed ||
	    !Common::File::RenameFile(temp_path, m_driver_cache_path)) {
		PipelineCacheLog("Vulkan pipeline cache: failed to write {}",
		                 Common::PathToString(m_driver_cache_path));
		return;
	}
	PipelineCacheLog("Vulkan pipeline cache: saved {} bytes to {}", payload.size(),
	                 Common::PathToString(m_driver_cache_path));
	m_graphics.device.destroyPipelineCache(m_driver_cache, nullptr);
	m_driver_cache = nullptr;
}

// The host-side mesh fields; false when the shader exceeds the host limits.
bool PipelineCache::MeshHost(ShaderVertexInputInfo& info, bool mesh_draw_indirect) const {
	EXIT_NOT_IMPLEMENTED(!m_graphics.mesh_shader_enabled);
	auto& mesh              = info.mesh;
	mesh.host_subgroup_size = m_graphics.subgroup_size;
	const auto& limits      = m_graphics.mesh_shader_properties;
	// A subgroup with more threads than a mesh workgroup may have (NVIDIA: 128) runs its
	// waves in passes; see EmitMeshEntryPoint.
	mesh.passes = mesh.PassesFor(
	    std::min(limits.maxMeshWorkGroupInvocations, limits.maxMeshWorkGroupSize[0]));
	// Guest wave32 waves sharing one wider host subgroup (RADV runs mesh shaders as wave64
	// and cannot be asked for a subgroup size there) would read each other's ballots:
	// wave 1 then used wave 0's EXEC/VCC masks and dropped its primitives (Wolverine
	// skin "cracks"). One guest wave per pass keeps every host subgroup to one wave.
	// KYTY_MESH_WAVE_PASSES=0 restores the shared-subgroup layout for comparison.
	static const bool wave_passes = [] {
		const char* value = std::getenv("KYTY_MESH_WAVE_PASSES");
		return value == nullptr || value[0] != '0';
	}();
	if (wave_passes && mesh.passes != 0 && mesh.wave_size < mesh.host_subgroup_size &&
	    mesh.Waves() > 1) {
		mesh.passes = mesh.Waves();
	}
	// Indirect mesh draws read their arguments on the GPU (MeshIndirectDraw).
	// KYTY_MESH_INDIRECT_GPU=0 (live) restores the CPU read for every draw.
	static auto& indirect_gpu = Common::LiveSwitches::Get("KYTY_MESH_INDIRECT_GPU", 1);
	mesh.draw_data_indirect =
	    mesh_draw_indirect && indirect_gpu.load(std::memory_order_relaxed) != 0;
	return !(mesh.passes == 0 || mesh.max_vertices > limits.maxMeshOutputVertices ||
	         mesh.max_primitives > limits.maxMeshOutputPrimitives ||
	         mesh.lds_size_dwords * sizeof(uint32_t) > limits.maxMeshSharedMemorySize);
}

// The pixel stage's target and blending fields (context state).
void PipelineCache::PixelFinish(const HW::Context& context, ShaderPixelInputInfo& info) {
	// SPI_SHADER_COL_FORMAT describes export packing, not the attachment numeric type.
	// In particular, 32-bit exports can carry raw integer material data.
	for (uint32_t slot = 0; slot < RENDER_COLOR_ATTACHMENTS_MAX; slot++) {
		const auto& rt = context.GetRenderTarget(slot);
		if (rt.base.addr == 0 ||
		    render_target_mask_slot(context.GetRenderTargetMask(), slot) == 0) {
			continue;
		}
		if (rt.info.channel_type == Prospero::ChannelType::kUInt) {
			info.target_uint_mask |= 1u << slot;
		} else if (rt.info.channel_type == Prospero::ChannelType::kSInt) {
			info.target_sint_mask |= 1u << slot;
		}
	}
	const auto& blend = context.GetBlendControl(0);
	info.dual_source_blending =
	    blend.enable && !context.GetRenderTarget(0).info.blend_bypass &&
	    (BlendFactorIsDualSource(blend.color_srcblend) ||
	     BlendFactorIsDualSource(blend.color_destblend) ||
	     (blend.separate_alpha_blend && (BlendFactorIsDualSource(blend.alpha_srcblend) ||
	                                     BlendFactorIsDualSource(blend.alpha_destblend))));
	if (info.dual_source_blending) {
		// MRT1 supplies the second blend source for target 0.
		info.target_output_mode[1]    = info.target_output_mode[0];
		info.target_export_mapping[1] = info.target_export_mapping[0];
		info.target_uint_mask =
		    (info.target_uint_mask & ~2u) | ((info.target_uint_mask & 1u) << 1u);
		info.target_sint_mask =
		    (info.target_sint_mask & ~2u) | ((info.target_sint_mask & 1u) << 1u);
	} else if (blend.enable && !context.GetRenderTarget(0).info.blend_bypass &&
	           info.target_output_mode[0] != 0 && info.target_output_mode[0] != 7 &&
	           std::all_of(std::begin(info.target_output_mode) + 1,
	                       std::end(info.target_output_mode),
	                       [](uint8_t mode) { return mode == 0; }) &&
	           ClassifyBlendMapping(blend, info.target_export_mapping[0]) ==
	               BlendMappingSupport::SourceAlpha) {
		// Preserve logical alpha when the export mapping moves it.
		info.alpha_blend_source_remap = true;
		info.dual_source_blending     = true;
		info.target_output_mode[1]    = info.target_output_mode[0];
		info.target_export_mapping[1] = {};
	}
}

// The clip-space fields of the last geometry stage (context state).
void PipelineCache::ClipFinish(const HW::Context& context, ShaderVertexInputInfo& info) const {
	if (context.GetClipControl().clip_disable) {
		const auto& viewport = context.GetScreenViewport().viewports[0];
		const auto& limits   = m_graphics.GetPhysicalDeviceProperties().limits;
		auto&       clip     = info.clip_space;
		clip.scale[0]        = viewport.xscale;
		clip.scale[1]        = viewport.yscale;
		clip.offset[0]       = viewport.xoffset;
		clip.offset[1]       = viewport.yoffset;
		clip.half_extent[0] =
		    static_cast<float>(std::min(limits.maxViewportDimensions[0], 16384u)) * 0.5f;
		clip.half_extent[1] =
		    static_cast<float>(std::min(limits.maxViewportDimensions[1], 16384u)) * 0.5f;
		clip.enabled = true;
	}
}

// KYTY_DRAW_PREP: the vertex-side identity of a draw for DrawPrep::NoteGoodPair / IsGoodPair.
static uint64_t DrawPrepVertexIdentity(const HW::VertexShaderInfo& vertex_regs) {
	return vertex_regs.es_regs.data_addr ^ (vertex_regs.gs_regs.data_addr << 1u);
}

PipelineCache::GraphicsPrograms PipelineCache::GetGraphicsPrograms(
    const HW::VertexShaderInfo& vertex_regs, const HW::PixelShaderInfo& pixel_regs,
    const HW::ShaderRegisters& sh, const HW::Context& context, const HW::UserConfig& user_config,
    std::span<const Prospero::ColorComponentMapping, 8> target_export_mapping, bool pixel_active,
    std::array<ShaderVertexInputInfo, 3>& vertex_info, ShaderPixelInputInfo& pixel_info,
    bool mesh_draw_indirect) {
	const bool tess_active = user_config.GetPrimType() == Prospero::PrimitiveType::kPatch;
	m_program_cache->LookaheadNextDraw();
	std::array<ShaderParams, 3> vertex_params;
	if (tess_active) {
		if (!PrepareTessellationPrograms(vertex_regs, context, vertex_info, vertex_params)) {
			if (!ShaderFailureNonFatal()) {
				EXIT("unsupported tessellation programs\n");
			}
			return {};
		}
	} else {
		KYTY_PROFILER_BLOCK("PipelineCache::PrepareProgram(VS)");
		vertex_params[0] = PrepareProgram(vertex_regs, context, user_config, vertex_info[0]);
	}
	// The host-side mesh fields; false when the shader exceeds the host limits.
	const auto mesh_host = [&](ShaderVertexInputInfo& info) { return MeshHost(info, mesh_draw_indirect); };
	const bool mesh_active = vertex_info[0].logical_stage == ShaderType::Mesh;
	if (mesh_active) {
		if (!mesh_host(vertex_info[0])) {
			const auto& mesh = vertex_info[0].mesh;
			// Skipped like the draws of a shader that gives up, and reported once per shader.
			static std::mutex                   logged_mutex;
			static std::unordered_set<uint64_t> logged;
			std::lock_guard                     lock(logged_mutex);
			if (logged.insert(vertex_params[0].hash).second) {
				LOGF("mesh shader 0x%016" PRIx64 " exceeds host limits, draws skipped: wave%u "
				     "threads=%u vertices=%u primitives=%u LDS=%u\n",
				     vertex_params[0].hash, mesh.wave_size, mesh.HostThreads(), mesh.max_vertices,
				     mesh.max_primitives, mesh.lds_size_dwords);
			}
			return {};
		}
	}
	// The pixel stage's target and blending fields (context state).
	const auto pixel_finish = [&](ShaderPixelInputInfo& info) { PixelFinish(context, info); };
	ShaderParams pixel_params;
	if (pixel_active) {
		KYTY_PROFILER_BLOCK("PipelineCache::PrepareProgram(PS)");
		pixel_params      = PrepareProgram(pixel_regs, sh, target_export_mapping, pixel_info);
		pixel_finish(pixel_info);
	}
	// The clip-space fields of the last geometry stage (context state).
	const auto clip_finish = [&](ShaderVertexInputInfo& info) { ClipFinish(context, info); };
	clip_finish(vertex_info[tess_active ? 2u : 0u]);
	uint32_t          push_data_cursor =
	    mesh_active ? ShaderRecompiler::IR::PushData::MeshDrawDwordCount : 0;
	GraphicsPrograms  result;
	// KYTY_LOCAL_HACK KYTY_PARALLEL_MATERIALIZE (live, default 1; pm4 +4%): the pixel and vertex stages' resource
	// materialization at the same time (ProgramCache::GetPixelVertex).
	static auto& parallel = Common::LiveSwitches::Get("KYTY_PARALLEL_MATERIALIZE", 1);
	// With KYTY_DRAW_PREP the stages are mostly memo hits: mode 1 runs no worker (dp4: same fps,
	// one core less spinning); the verify/control modes 2-4 still do.
	if (pixel_active && !tess_active && parallel.load(std::memory_order_relaxed) != 0 &&
	    (DrawPrep::Mode() == 0 || parallel.load(std::memory_order_relaxed) >= 2)) {
		m_program_cache->GetPixelVertex(pixel_params, pixel_info, vertex_params[0], vertex_info[0],
		                                push_data_cursor, result.pixel, result.vertex[0]);
		if (!result.vertex[0]) {
			return {};
		}
		if (result.pixel) {
			DrawPrep::NoteGoodPair(DrawPrepVertexIdentity(vertex_regs), pixel_regs.ps_regs.data_addr);
		}
		// KYTY_LOOKAHEAD: the next draw's programs, prepared from its predicted shader registers
		// and this draw's context (the CP publishes a prediction only when they stay valid).
		if (const auto* next = Lookahead::g_next; next != nullptr && next->GetPs().ps_regs.data_addr != 0) {
			const auto prepare_start = std::chrono::steady_clock::now();
			thread_local ShaderVertexInputInfo next_vertex_info;
			thread_local ShaderPixelInputInfo  next_pixel_info;
			next_vertex_info = {};
			next_pixel_info  = {};
			const auto next_vertex = PrepareProgram(next->GetVs(), context, user_config, next_vertex_info);
			if (next_vertex_info.logical_stage != ShaderType::Mesh || mesh_host(next_vertex_info)) {
				const auto next_pixel = PrepareProgram(next->GetPs(), sh, target_export_mapping, next_pixel_info);
				pixel_finish(next_pixel_info);
				clip_finish(next_vertex_info);
				static auto& lookahead = Common::LiveSwitches::Get("KYTY_LOOKAHEAD", 0);
				m_program_cache->LookaheadStart(next_pixel, next_pixel_info, next_vertex, next_vertex_info,
				                                Lookahead::g_next_trusted &&
				                                    lookahead.load(std::memory_order_relaxed) == 2 ||
				                                    lookahead.load(std::memory_order_relaxed) == 3);
			}
			m_program_cache->lookahead_prepare_ns += static_cast<uint64_t>(
			    std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - prepare_start)
			        .count());
		}
		return result;
	}
	if (pixel_active) {
		result.pixel = m_program_cache->Get(pixel_params, pixel_info, push_data_cursor);
	}
	for (uint32_t i = 0; i < (tess_active ? 3u : 1u); i++) {
		result.vertex[i] = m_program_cache->Get(vertex_params[i], vertex_info[i], push_data_cursor);
		if (!result.vertex[i]) {
			return {};
		}
	}
	if (!tess_active && (!pixel_active || result.pixel)) {
		DrawPrep::NoteGoodPair(DrawPrepVertexIdentity(vertex_regs),
		                       pixel_active ? pixel_regs.ps_regs.data_addr : 0u);
	}
	return result;
}

void PipelineCache::MarkDrawPrepThread() {
	t_parallel_worker = true;
}

void PipelineCache::DrawPrepGraphics(const HW::VertexShaderInfo& vertex_regs,
                                     const HW::PixelShaderInfo& pixel_regs, const HW::ShaderRegisters& sh,
                                     const HW::Context& context, const HW::UserConfig& user_config,
                                     std::span<const Prospero::ColorComponentMapping, 8> target_export_mapping,
                                     bool pixel_active, bool mesh_draw_indirect, const DrawPrep::Key& key) {
	if (user_config.GetPrimType() == Prospero::PrimitiveType::kPatch) {
		DrawPrep::Add(DrawPrep::Counter::SkipUnsupported);
		return;
	}
	if (!DrawPrep::IsGoodPair(DrawPrepVertexIdentity(vertex_regs),
	                          pixel_active ? pixel_regs.ps_regs.data_addr : 0u)) {
		DrawPrep::Add(DrawPrep::Counter::SkipUnknownPair);
		return;
	}
	ShaderVertexInputInfo vertex_info {};
	ShaderPixelInputInfo  pixel_info {};
	const auto            vertex_params = PrepareProgram(vertex_regs, context, user_config, vertex_info);
	if (vertex_info.logical_stage == ShaderType::Mesh && !MeshHost(vertex_info, mesh_draw_indirect)) {
		DrawPrep::Add(DrawPrep::Counter::SkipUnsupported);
		return;
	}
	ShaderParams pixel_params;
	if (pixel_active) {
		pixel_params = PrepareProgram(pixel_regs, sh, target_export_mapping, pixel_info);
		PixelFinish(context, pixel_info);
	}
	ClipFinish(context, vertex_info);
	m_program_cache->ScanDraw(key, pixel_active ? &pixel_params : nullptr, pixel_info, vertex_params,
	                          vertex_info);
}

void PipelineCache::DrawPrepCompute(const HW::ComputeShaderInfo& regs, const HW::ShaderRegisters& sh,
                                    ShaderComputeInputInfo& input_info, const DrawPrep::Key& key) {
	if (!DrawPrep::IsGoodPair(regs.cs_regs.data_addr, UINT64_MAX)) {
		DrawPrep::Add(DrawPrep::Counter::SkipUnknownPair);
		return;
	}
	// As GetComputeProgram.
	input_info.host_subgroup_size = m_graphics.SupportsComputeWave64() ? 64u : 32u;
	const auto params             = PrepareProgram(regs, sh, input_info);
	const auto max_lds_dwords =
	    m_graphics.GetPhysicalDeviceProperties().limits.maxComputeSharedMemorySize / 4u;
	input_info.lds_size_dwords = std::min(input_info.lds_size_dwords, max_lds_dwords);
	m_program_cache->ScanDispatch(key, params, input_info);
}

ShaderProgram PipelineCache::GetComputeProgram(const HW::ComputeShaderInfo& regs,
                                               const HW::ShaderRegisters&   sh,
                                               ShaderComputeInputInfo&      input_info) {
	input_info.host_subgroup_size = m_graphics.SupportsComputeWave64() ? 64u : 32u;
	const auto        params      = PrepareProgram(regs, sh, input_info);
	// Use one effective size for the cache key, LDS declaration, and access bounds.
	const auto max_lds_dwords =
	    m_graphics.GetPhysicalDeviceProperties().limits.maxComputeSharedMemorySize / 4u;
	if (input_info.lds_size_dwords > max_lds_dwords) {
		static std::atomic_bool warned = false;
		if (!warned.exchange(true, std::memory_order_relaxed)) {
			PipelineCacheLog("GPU warning: game compute shader requests {} bytes of LDS, but "
			                 "the Vulkan device limit is {} bytes. Clamping LDS; rendering may "
			                 "be incorrect.",
			                 input_info.lds_size_dwords * 4u, max_lds_dwords * 4u);
		}
	}
	input_info.lds_size_dwords = std::min(input_info.lds_size_dwords, max_lds_dwords);
	uint32_t          push_data_cursor = 0;
	auto              program          = m_program_cache->Get(params, input_info, push_data_cursor);
	if (program) {
		// KYTY_DRAW_PREP: compute shaders join the known-good list with a vertex identity no draw has.
		DrawPrep::NoteGoodPair(regs.cs_regs.data_addr, UINT64_MAX);
	}
	return program;
}

bool PipelineStaticParameters::operator==(const PipelineStaticParameters& other) const noexcept {
	return std::memcmp(this, &other, sizeof(*this)) == 0;
}

PipelineCache::Pipeline& PipelineCache::GetGraphicsPipeline(
    std::span<const RenderColorInfo> colors, const RenderDepthInfo& depth,
    std::span<const ShaderVertexInputInfo> vertex_info, CommandBuffer& command,
    const ShaderPixelInputInfo* ps_input_info, vk::PrimitiveTopology topology,
    bool primitive_restart_enable, const GraphicsPrograms& programs) {
	auto* pipeline = TryGetGraphicsPipeline(colors, depth, vertex_info, command, ps_input_info,
	                                        topology, primitive_restart_enable, programs, false);
	EXIT_IF(pipeline == nullptr);
	return *pipeline;
}

bool PipelineCache::FinishPending(Pipeline& pipeline) {
	auto& job = *pipeline.pending;
	if (!job.done.load(std::memory_order_acquire)) {
		if (m_compiler != nullptr && !m_compiler->Stopped()) {
			return false;
		}
		// The workers stopped before this job started: compile it here.
		job.result = CreateGraphicsPipeline(*job.build, m_driver_cache, &job.pipeline);
		job.done.store(true, std::memory_order_release);
	}
	EXIT_NOT_IMPLEMENTED(job.result != vk::Result::eSuccess || job.pipeline == nullptr);
	pipeline.pipeline = job.pipeline;
	pipeline.pending.reset();
	return true;
}

PipelineCache::Pipeline* PipelineCache::TryGetGraphicsPipeline(
    std::span<const RenderColorInfo> colors, const RenderDepthInfo& depth,
    std::span<const ShaderVertexInputInfo> vertex_info, CommandBuffer& command,
    const ShaderPixelInputInfo* ps_input_info, vk::PrimitiveTopology topology,
    bool primitive_restart_enable, const GraphicsPrograms& programs, bool may_defer) {
	const auto& vs_input_info  = vertex_info.front();
	const auto& vertex_program = programs.vertex[0];
	const auto& pixel_program  = programs.pixel;
	KYTY_PROFILER_BLOCK("PipelineCache::CreatePipeline(Gfx)", profiler::colors::DeepOrangeA200);

	EXIT_IF(colors.size() > RENDER_COLOR_ATTACHMENTS_MAX);
	EXIT_IF(!vertex_program);
	const bool ps_active = ps_input_info != nullptr;
	EXIT_IF(ps_active && !pixel_program);
	const auto color_count = static_cast<uint32_t>(colors.size());

	auto&             ctx = command.GetRegisters();

	const HW::ModeControl& mc = ctx.GetModeControl();

	const auto vs_id = vertex_program.id;
	const auto ps_id = ps_active ? pixel_program.id : 0;

	GraphicsPipelineKey key {};
	for (uint32_t i = 0; i < programs.vertex.size(); i++) {
		key.vertex_shader_ids[i] = programs.vertex[i].id;
	}
	key.ps_shader_id            = ps_id;
	auto& static_params         = key.static_params;
	auto& rendering             = key.rendering;
	rendering.color_count       = 0;
	uint32_t attachment_samples = 0;
	for (uint32_t i = 0; i < color_count; i++) {
		const auto slot = colors[i].target_slot;
		EXIT_IF(slot >= RENDER_COLOR_ATTACHMENTS_MAX);
		rendering.color_count = std::max(rendering.color_count, slot + 1);
		EXIT_IF(!colors[i].image_id || colors[i].desc.view_info.format == vk::Format::eUndefined);
		static_params.color_mask[slot] = colors[i].export_mapping.ApplyMask(
		    render_target_mask_slot(ctx.GetRenderTargetMask(), colors[i].target_slot));
		rendering.color_formats[slot] = colors[i].desc.view_info.format;
		if (attachment_samples == 0) {
			attachment_samples = colors[i].desc.info.samples;
		} else if (attachment_samples != colors[i].desc.info.samples) {
			EXIT("mixed color attachment sample counts are unsupported: %u and %u\n",
			     attachment_samples, colors[i].desc.info.samples);
		}
		const auto& rt                        = ctx.GetRenderTarget(colors[i].target_slot);
		const auto& bc                        = ctx.GetBlendControl(colors[i].target_slot);
		const bool alpha_remap =
		    slot == 0 && ps_input_info != nullptr && ps_input_info->alpha_blend_source_remap;
		static_params.blend_enable[slot] = bc.enable && !rt.info.blend_bypass;
		if (static_params.blend_enable[slot] && !alpha_remap &&
		    ClassifyBlendMapping(bc, colors[i].export_mapping) != BlendMappingSupport::Direct) {
			static_params.blend_enable[slot] = false;
			static std::atomic_bool warned = false;
			if (!warned.exchange(true, std::memory_order_relaxed)) {
				Log::WriteToConsoleAndLog(fmt::format(
				    "Warning: blending disabled for unsupported color mapping "
				    "(slot={} mapping=0x{:02x} color={}/{} alpha={}/{} separate={}).\n",
				    slot, colors[i].export_mapping.packed, bc.color_srcblend, bc.color_destblend,
				    bc.alpha_srcblend, bc.alpha_destblend, bc.separate_alpha_blend ? 1 : 0));
			}
		}
		if (alpha_remap) {
			static_params.blend_alpha_source_remap = true;
		}
		if (static_params.blend_enable[slot]) {
			static_params.color_srcblend[slot]       = bc.color_srcblend;
			static_params.color_comb_fcn[slot]       = bc.color_comb_fcn;
			static_params.color_destblend[slot]      = bc.color_destblend;
			static_params.separate_alpha_blend[slot] = bc.separate_alpha_blend;
			if (bc.separate_alpha_blend) {
				static_params.alpha_srcblend[slot]  = bc.alpha_srcblend;
				static_params.alpha_comb_fcn[slot]  = bc.alpha_comb_fcn;
				static_params.alpha_destblend[slot] = bc.alpha_destblend;
			}
		}
	}
	const bool with_depth =
	    depth.desc.view_info.format != vk::Format::eUndefined && static_cast<bool>(depth.image_id);
	if (with_depth) {
		const auto aspects       = ImageViewOps::DepthAspectMask(depth.desc.view_info.format);
		rendering.depth_format   = aspects & vk::ImageAspectFlagBits::eDepth
		                               ? depth.desc.view_info.format
		                               : vk::Format::eUndefined;
		rendering.stencil_format = aspects & vk::ImageAspectFlagBits::eStencil
		                               ? depth.desc.view_info.format
		                               : vk::Format::eUndefined;
		if (attachment_samples == 0) {
			attachment_samples = depth.desc.info.samples;
		} else if (attachment_samples != depth.desc.info.samples) {
			EXIT("mixed color/depth sample counts are unsupported: %u and %u\n", attachment_samples,
			     depth.desc.info.samples);
		}
	}
	if (color_count == 0 && !with_depth) {
		attachment_samples = render_sample_count(ctx.GetAaConfig().msaa_num_samples);
		EXIT_IF(!static_cast<bool>(
		    m_graphics.GetPhysicalDeviceProperties().limits.framebufferNoAttachmentsSampleCounts &
		    vulkan_sample_count(attachment_samples)));
	}
	EXIT_IF(attachment_samples == 0 ||
	        vulkan_sample_count(attachment_samples) == vk::SampleCountFlagBits {});

	if (ps_active && depth.depth_test_enable && ps_input_info->ps_execute_on_noop) {
		static std::atomic<uint32_t> log_count {0};
		if (log_count.fetch_add(1, std::memory_order_relaxed) < 16) {
			LOGF("Pipeline: temporary: accepting EXEC_ON_NOOP with depth test enabled\n");
		}
	}

	const auto& clip_control               = ctx.GetClipControl();
	static_params.negative_one_to_one      = !clip_control.dx_clip_space;
	static_params.depth_clip_enable        = clip_control.IsZClipEnabled();
	{
		// KYTY_LOCAL_HACK (research): which Z clip planes the guest enables, logged once per
		// combination; KYTY_ZCLIP_PARTIAL=1 keeps Vulkan depth clipping when only one is on.
		static std::atomic<uint32_t> seen {0};
		const uint32_t combo = (clip_control.min_z_clip_disable ? 1u : 0u) |
		                       (clip_control.max_z_clip_disable ? 2u : 0u);
		static const uint64_t probe_hash = [] {
			const char* value = std::getenv("KYTY_PROBE_HASH");
			return value != nullptr ? std::strtoull(value, nullptr, 16) : 0ull;
		}();
		if (probe_hash != 0 && vs_input_info.stage.program != nullptr &&
		    vs_input_info.stage.program->shader_hash == probe_hash) {
			static std::atomic<uint32_t> probe_seen {0};
			const uint32_t key = combo | (clip_control.dx_clip_space ? 4u : 0u) |
			                     (clip_control.clip_disable ? 8u : 0u);
			if (ps_input_info != nullptr && ps_input_info->stage.program != nullptr) {
				static std::mutex                   ps_mutex;
				static std::unordered_set<uint64_t> ps_seen;
				std::lock_guard lock(ps_mutex);
				if (ps_seen.insert(ps_input_info->stage.program->shader_hash).second) {
					std::printf("ZClip probe draw: pixel shader %016" PRIx64 "\n",
					            ps_input_info->stage.program->shader_hash);
				}
			}
			if ((probe_seen.fetch_or(1u << key) & (1u << key)) == 0) {
				std::printf("ZClip probe draw: near_disable=%u far_disable=%u dx=%u clip_disable=%u "
				            "topology=%s restart=%u\n",
				            combo & 1u, combo >> 1u, clip_control.dx_clip_space ? 1u : 0u,
				            clip_control.clip_disable ? 1u : 0u, vk::to_string(topology).c_str(),
				            primitive_restart_enable ? 1u : 0u);
			}
		}
		if ((seen.fetch_or(1u << combo) & (1u << combo)) == 0) {
			std::printf("ZClip: guest near_disable=%u far_disable=%u dx_clip_space=%u\n",
			            combo & 1u, combo >> 1u, clip_control.dx_clip_space ? 1u : 0u);
		}
		static const bool partial = [] {
			const char* value = std::getenv("KYTY_ZCLIP_PARTIAL");
			return value != nullptr && std::strcmp(value, "1") == 0;
		}();
		if (partial && combo != 3u) {
			static_params.depth_clip_enable = true;
		}
		// KYTY_FORCE_ZCLIP_HASH=<hex>: draws of that vertex shader always clip depth.
		static const uint64_t force_hash = [] {
			const char* value = std::getenv("KYTY_FORCE_ZCLIP_HASH");
			return value != nullptr ? std::strtoull(value, nullptr, 16) : 0ull;
		}();
		if (force_hash != 0 && vs_input_info.stage.program != nullptr &&
		    vs_input_info.stage.program->shader_hash == force_hash) {
			static_params.depth_clip_enable = true;
		}
	}
	static_params.topology                 = topology;
	static_params.primitive_restart_enable = primitive_restart_enable;
	static_params.samples                  = attachment_samples;
	static_params.sample_shading_enable =
	    ps_active && attachment_samples > 1 && ps_input_info->ps_sample_shading;
	if (static_params.sample_shading_enable && !m_graphics.sample_rate_shading_enabled) {
		EXIT("Pipeline: sample-rate shading is required but unsupported by the host\n");
	}
	static_params.depth_bounds_test_enable = depth.depth_bounds_test_enable;
	static_params.depth_min_bounds         = depth.depth_min_bounds;
	static_params.depth_max_bounds         = depth.depth_max_bounds;
	const bool rect_list = Prospero::IsRectList(command.GetUserConfig().GetPrimType());
	static_params.cull_back  = !rect_list && mc.cull_back;
	static_params.cull_front = !rect_list && mc.cull_front;
	static_params.face       = mc.face;
	static_params.provoking_vtx_last = mc.provoking_vtx_last;
	static_params.polygon_mode =
	    ResolvePolygonMode(mc, static_params.cull_front, static_params.cull_back);

	if (vs_input_info.stage.program->stage != ShaderType::Mesh) {
		EXIT_IF(vs_input_info.buffers_num < 0 ||
		        vs_input_info.buffers_num > ShaderVertexInputInfo::RES_MAX ||
		        vs_input_info.resources_num < 0 ||
		        vs_input_info.resources_num > ShaderVertexInputInfo::RES_MAX);
		key.vertex_input.binding_count   = static_cast<uint8_t>(vs_input_info.buffers_num);
		key.vertex_input.attribute_count = static_cast<uint8_t>(vs_input_info.resources_num);
		for (int binding = 0; binding < vs_input_info.buffers_num; binding++) {
			const auto& buffer = vs_input_info.buffers[binding];
			key.vertex_input.bindings[binding] = {.stride   = buffer.stride,
			                                      .instance = buffer.fetch_index != 0};
		}
		for (int attribute = 0; attribute < vs_input_info.resources_num; attribute++) {
			const auto binding = vs_input_info.resources_dst[attribute].buffer_index;
			EXIT_IF(binding < 0 || binding >= vs_input_info.buffers_num);
			key.vertex_input.attributes[attribute] = {
			    .offset = static_cast<uint32_t>(vs_input_info.resources[attribute].Base48() -
			                                    vs_input_info.buffers[binding].addr),
			    .binding = static_cast<uint8_t>(binding),
			};
		}
	}

	const auto defer = [this] {
		if (++m_deferred_draws % 256 == 1) {
			LOGF("PipelineCache: %" PRIu64 " draws skipped while their pipeline compiled in the "
			     "background\n",
			     m_deferred_draws);
		}
		return nullptr;
	};
	if (auto iter = m_graphics_pipelines.find(key); iter != m_graphics_pipelines.end()) {
		auto& found = *iter->second;
		if (found.pending && !FinishPending(found)) {
			return defer();
		}
		return &found;
	}

	if (graphics_debug_dump_enabled()) {
		ShaderDbgDumpInputInfo(vs_input_info);
		if (ps_active) {
			ShaderDbgDumpInputInfo(*ps_input_info);
		}
		LOGF("PipelineTrace: shader modules VS=%" PRIu64 " module=%p PS=%" PRIu64 " module=%p\n",
		     vs_id, static_cast<void*>(vertex_program.module), ps_id,
		     static_cast<void*>(pixel_program.module));
	}

	auto cached = std::make_unique<Pipeline>();
	LogPipelineTrace("CreatePipelineInternal begin", vs_id, ps_id);
	auto build = PrepareGraphicsPipeline(m_graphics, *cached, rendering, key.vertex_input,
	                                     vertex_info, ps_input_info, programs, static_params);
	EXIT_NOT_IMPLEMENTED(cached->pipeline_layout == nullptr);
	if (may_defer && m_compiler != nullptr && !m_compiler->Stopped()) {
		// A pipeline the driver has cached finishes within the wait; a new one is compiled in
		// the background, and its draws are skipped until it is ready.
		auto job          = std::make_shared<PendingGraphicsPipeline>();
		job->build        = std::move(build);
		job->driver_cache = m_driver_cache;
		cached->pending   = job;
		m_compiler->Submit(job);
		(void)m_compiler->Wait(*job, g_pipeline_wait);
		auto [iter, inserted] = m_graphics_pipelines.emplace(std::move(key), std::move(cached));
		EXIT_IF(!inserted);
		auto& pipeline = *iter->second;
		if (!FinishPending(pipeline)) {
			return defer();
		}
		LogPipelineTrace("CreatePipelineInternal done", vs_id, ps_id);
		return &pipeline;
	}
	const auto result = CreateGraphicsPipeline(*build, m_driver_cache, &cached->pipeline);
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);
	LogPipelineTrace("CreatePipelineInternal done", vs_id, ps_id);

	EXIT_NOT_IMPLEMENTED(cached->pipeline == nullptr);

	auto [iter, inserted] = m_graphics_pipelines.emplace(std::move(key), std::move(cached));
	EXIT_IF(!inserted);

	return iter->second.get();
}

PipelineCache::Pipeline&
PipelineCache::GetComputePipeline(const ShaderComputeInputInfo& input_info,
                                  const ShaderProgram&          compute_program) {
	KYTY_PROFILER_BLOCK("PipelineCache::CreatePipeline(Compute)", profiler::colors::RedA100);

	EXIT_IF(!compute_program);

	if (auto iter = m_compute_pipelines.find(compute_program.id);
	    iter != m_compute_pipelines.end()) {
		return *iter->second;
	}

	if (graphics_debug_dump_enabled()) {
		ShaderDbgDumpInputInfo(input_info);
	}

	auto cached = std::make_unique<Pipeline>();
	CreatePipelineInternal(m_graphics, *cached, input_info, compute_program.module, m_driver_cache);

	EXIT_NOT_IMPLEMENTED(cached->pipeline == nullptr);
	EXIT_NOT_IMPLEMENTED(cached->pipeline_layout == nullptr);

	auto [iter, inserted] = m_compute_pipelines.emplace(compute_program.id, std::move(cached));
	EXIT_IF(!inserted);

	return *iter->second;
}

// KYTY_LOCAL_HACK: the --skip-shaders list, for the dispatch fast path (KYTY_FAST_SKIP).
bool IsSkipListedShader(uint64_t shader_hash) {
	return SkipShaderRequested(shader_hash);
}

namespace DrawRecordCensus {
uint64_t NowNs() {
	return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
	                                 std::chrono::steady_clock::now().time_since_epoch())
	                                 .count());
}

bool PhasesOn() {
	static auto& on = Common::LiveSwitches::Get("KYTY_DRAW_PHASES", 0);
	return on.load(std::memory_order_relaxed) != 0;
}

void Record(bool compute, uint64_t start_ns) {
	if (PhasesOn()) {
		static uint64_t phase_report = NowNs();
		if (const auto now = NowNs(); now - phase_report >= 5'000'000'000ull) {
			static const char* names[10] = {"refresh-shaders", "targets",  "prepare-bindings",
			                                "graphics-bindings", "pipeline", "acquire-targets",
			                                "commit",            "resolved-total", "resolve-texture",
			                                "samplers+data"};
			::printf("Draw phases (5 s):");
			for (int i = 0; i < 10; i++) {
				::printf(" %s %.1f ms/%" PRIu64, names[i], static_cast<double>(g_phase_ns[i]) / 1e6,
				         g_phase_calls[i]);
			}
			::printf("\n");
			std::memset(g_phase_ns, 0, sizeof(g_phase_ns));
			std::memset(g_phase_calls, 0, sizeof(g_phase_calls));
			phase_report = now;
		}
	}
	static auto& classify = Common::LiveSwitches::Get("KYTY_MEMO_CLASSIFY", 0);
	if (classify.load(std::memory_order_relaxed) == 0) {
		return;
	}
	// [kind][class]: class 0 none (no lookup), 1 exact, 2 pass/reloc same, 3 not replayable.
	static uint64_t ns[2][4] {};
	static uint64_t calls[2][4] {};
	static uint64_t last_report = NowNs();
	const auto      now         = NowNs();
	const auto      cls = g_flags == 0 ? 0 : (g_flags & 1u) != 0 ? 3 : (g_flags & 8u) != 0 ? 2 : 1;
	ns[compute ? 1 : 0][cls] += now - start_ns;
	calls[compute ? 1 : 0][cls]++;
	if (now - last_report >= 5'000'000'000ull) {
		for (int kind = 0; kind < 2; kind++) {
			::printf("Draw record census (5 s) %s: none %.1f ms/%" PRIu64 ", exact %.1f ms/%" PRIu64
			         ", pass/reloc-same %.1f ms/%" PRIu64 ", not replayable %.1f ms/%" PRIu64 "\n",
			         kind == 0 ? "draws" : "dispatches", static_cast<double>(ns[kind][0]) / 1e6,
			         calls[kind][0], static_cast<double>(ns[kind][1]) / 1e6, calls[kind][1],
			         static_cast<double>(ns[kind][2]) / 1e6, calls[kind][2],
			         static_cast<double>(ns[kind][3]) / 1e6, calls[kind][3]);
		}
		std::memset(ns, 0, sizeof(ns));
		std::memset(calls, 0, sizeof(calls));
		last_report = now;
	}
}
} // namespace DrawRecordCensus

} // namespace Libs::Graphics
