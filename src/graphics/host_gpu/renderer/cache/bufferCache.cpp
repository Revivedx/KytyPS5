#include "graphics/host_gpu/renderer/cache/bufferCache.h"
#include "common/waitStats.h"
#include "graphics/host_gpu/pipelineStats.h"

#include "common/alignment.h"
#include "common/assert.h"
#include "common/liveSwitches.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/host_gpu/addressBindingReport.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/cache/textureCache.h"
#include "graphics/host_gpu/renderer/commandRecorder.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "kernel/memory.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <map>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include <thread>
#include <pthread.h>
#include <fcntl.h>
#include <linux/udmabuf.h>
#include <sys/ioctl.h>
#include <unistd.h>

namespace Libs::Graphics {

namespace {

constexpr uint64_t MiB           = 1024 * 1024;
constexpr uint64_t GdsBufferSize = 64 * 1024;

// KYTY_READBACK_STATS=1: every 5 s, print the buffers whose guest page faults were handed to the
// GPU thread (BufferCache::ReadMemory) and how each one ended. For downloads it also reports the
// written binding that last marked the faulting bytes (shader, size, whether that GPU work had
// already finished, how long ago it was bound) and how many downloaded bytes differed from guest
// memory. Only the GPU thread records, except the byte comparison (download completions).
class ReadbackStats {
public:
	enum Outcome : uint32_t {
		Unregistered,
		Unmarked,
		Downloaded,
		NothingToDownload,
		DirectRead,
		OutcomeCount
	};
	static constexpr uint32_t AgeBuckets = 5; // <1 ms, <10 ms, <50 ms, <250 ms, older

	struct Writer {
		bool     found    = false;
		bool     finished = false;
		uint32_t age      = 0;
		uint64_t shader   = 0;
		uint64_t size     = 0;
	};

	class Scope {
	public:
		Scope(uint64_t vaddr, bool is_write)
		    : m_stats(Get()), m_vaddr(vaddr), m_is_write(is_write),
		      m_start(m_stats != nullptr ? std::chrono::steady_clock::now()
		                                 : std::chrono::steady_clock::time_point {}) {}
		~Scope() {
			if (m_stats != nullptr) {
				const auto elapsed = std::chrono::steady_clock::now() - m_start;
				m_stats->Record(m_buffer_begin, m_buffer_size, m_vaddr, m_is_write, m_outcome,
				                m_download_bytes, m_writer,
				                std::chrono::duration<double, std::micro>(elapsed).count());
			}
		}
		KYTY_CLASS_NO_COPY(Scope);

		[[nodiscard]] bool Enabled() const { return m_stats != nullptr; }
		void               SetBuffer(uint64_t begin, uint64_t size) {
            m_buffer_begin = begin;
            m_buffer_size  = size;
		}
		void SetOutcome(Outcome outcome) { m_outcome = outcome; }
		void SetDownloadBytes(uint64_t bytes) { m_download_bytes = bytes; }
		// The latest written binding covering the faulting address; is_finished(tick) tells
		// whether that GPU work has completed.
		template <typename IsFinished>
		void FindWriter(IsFinished&& is_finished) {
			if (m_stats != nullptr) {
				m_writer = m_stats->FindWriter(m_vaddr, is_finished);
			}
		}

	private:
		ReadbackStats*                        m_stats;
		uint64_t                              m_vaddr;
		bool                                  m_is_write;
		std::chrono::steady_clock::time_point m_start;
		uint64_t                              m_buffer_begin   = 0;
		uint64_t                              m_buffer_size    = 0;
		Outcome                               m_outcome        = Unregistered;
		uint64_t                              m_download_bytes = 0;
		Writer                                m_writer;
	};

	// GPU thread: the shader whose resources are being bound next.
	static void SetShader(uint64_t hash) { s_shader = hash; }

	// GPU thread: a binding the GPU may write.
	static void NoteWrite(uint64_t begin, uint64_t size, uint64_t tick) {
		if (auto* stats = Get(); stats != nullptr) {
			auto& mark = stats->m_marks[stats->m_next_mark++ % stats->m_marks.size()];
			mark       = {begin, begin + size, tick, s_shader, std::chrono::steady_clock::now()};
		}
	}

	// Download completion: how many bytes the GPU copy changed in guest memory.
	static void NoteCompare(uint64_t changed, uint64_t total) {
		if (Get() != nullptr) {
			s_changed_bytes.fetch_add(changed, std::memory_order_relaxed);
			s_compared_bytes.fetch_add(total, std::memory_order_relaxed);
		}
	}

	[[nodiscard]] static bool On() { return Get() != nullptr; }

private:
	struct Mark {
		uint64_t                              begin  = 0;
		uint64_t                              end    = 0;
		uint64_t                              tick   = 0;
		uint64_t                              shader = 0;
		std::chrono::steady_clock::time_point time;
	};

	static ReadbackStats* Get() {
		static auto&         enabled = Common::LiveSwitches::Get("KYTY_READBACK_STATS", 0);
		static ReadbackStats stats;
		return enabled.load(std::memory_order_relaxed) != 0 ? &stats : nullptr;
	}

	template <typename IsFinished>
	Writer FindWriter(uint64_t vaddr, IsFinished&& is_finished) const {
		// Newest first.
		for (size_t i = 0; i < m_marks.size(); i++) {
			const auto& mark = m_marks[(m_next_mark + m_marks.size() - 1 - i) % m_marks.size()];
			if (mark.end == 0 || vaddr < mark.begin || vaddr >= mark.end) {
				continue;
			}
			const auto age_ms = std::chrono::duration<double, std::milli>(
			                        std::chrono::steady_clock::now() - mark.time)
			                        .count();
			const uint32_t age = age_ms < 1     ? 0
			                     : age_ms < 10  ? 1
			                     : age_ms < 50  ? 2
			                     : age_ms < 250 ? 3
			                                    : 4;
			return {true, is_finished(mark.tick), age, mark.shader, mark.end - mark.begin};
		}
		return {};
	}

	void Record(uint64_t buffer_begin, uint64_t buffer_size, uint64_t vaddr, bool is_write,
	            Outcome outcome, uint64_t download_bytes, const Writer& writer, double micros) {
		auto& entry = m_entries[buffer_begin];
		entry.size  = buffer_size;
		(is_write ? entry.writes : entry.reads)++;
		if (s_forwarded) {
			entry.forwarded++;
			entry.forwarded_micros += micros;
		}
		entry.outcomes[outcome]++;
		entry.download_bytes += download_bytes;
		entry.micros += micros;
		entry.pages.insert(vaddr >> 12u);
		if (writer.found) {
			(writer.finished ? entry.writer_finished : entry.writer_running)++;
			entry.writer_age[writer.age]++;
			entry.writer_shaders[writer.shader]++;
			entry.writer_size = std::max(entry.writer_size, writer.size);
		} else if (outcome == Downloaded) {
			entry.writer_unknown++;
		}
		const auto now = std::chrono::steady_clock::now();
		if (now - m_last >= std::chrono::seconds(5)) {
			Print();
			m_entries.clear();
			m_last = now;
		}
	}

	void Print() const {
		std::vector<std::pair<uint64_t, const Entry*>> sorted;
		double                                         total = 0;
		for (const auto& [begin, entry]: m_entries) {
			sorted.emplace_back(begin, &entry);
			total += entry.micros;
		}
		std::ranges::sort(sorted, [](const auto& a, const auto& b) {
			return a.second->micros > b.second->micros;
		});
		const auto changed  = s_changed_bytes.exchange(0, std::memory_order_relaxed);
		const auto compared = s_compared_bytes.exchange(0, std::memory_order_relaxed);
		::printf("Readback stats (5 s): %zu buffers, %.1f ms on the GPU thread; downloads changed "
		         "%" PRIu64 " of %" PRIu64 " KiB of guest memory\n",
		         sorted.size(), total / 1000.0, changed / 1024, compared / 1024);
		for (size_t i = 0; i < std::min<size_t>(sorted.size(), 8); i++) {
			const auto& [begin, e] = sorted[i];
			::printf("  buffer 0x%016" PRIx64 " size %" PRIu64 " KiB: writes %" PRIu64
			         " reads %" PRIu64 ", unregistered %" PRIu64 " unmarked %" PRIu64
			         " downloaded %" PRIu64 " (%" PRIu64 " KiB) nothing-to-download %" PRIu64
			         " direct-read %" PRIu64 ", %zu pages, %.1f ms (guest threads %" PRIu64
			         " / %.1f ms)\n",
			         begin, e->size / 1024, e->writes, e->reads, e->outcomes[Unregistered],
			         e->outcomes[Unmarked], e->outcomes[Downloaded], e->download_bytes / 1024,
			         e->outcomes[NothingToDownload], e->outcomes[DirectRead], e->pages.size(),
			         e->micros / 1000.0, e->forwarded, e->forwarded_micros / 1000.0);
			if (e->pages.size() <= 16) {
				std::vector<uint64_t> pages(e->pages.begin(), e->pages.end());
				std::ranges::sort(pages);
				::printf("    pages:");
				for (const auto page: pages) {
					::printf(" 0x%" PRIx64, page << 12u);
				}
				::printf("\n");
			}
			if (e->writer_finished + e->writer_running + e->writer_unknown == 0) {
				continue;
			}
			::printf("    writer: finished %" PRIu64 " still-running %" PRIu64 " unknown %" PRIu64
			         "; bound <1ms %" PRIu64 " <10ms %" PRIu64 " <50ms %" PRIu64 " <250ms %" PRIu64
			         " older %" PRIu64 "; largest binding %" PRIu64 " KiB; shaders",
			         e->writer_finished, e->writer_running, e->writer_unknown, e->writer_age[0],
			         e->writer_age[1], e->writer_age[2], e->writer_age[3], e->writer_age[4],
			         e->writer_size / 1024);
			std::vector<std::pair<uint64_t, uint64_t>> shaders(e->writer_shaders.begin(),
			                                                   e->writer_shaders.end());
			std::ranges::sort(shaders,
			                  [](const auto& a, const auto& b) { return a.second > b.second; });
			for (size_t s = 0; s < std::min<size_t>(shaders.size(), 4); s++) {
				::printf(" 0x%016" PRIx64 " x%" PRIu64, shaders[s].first, shaders[s].second);
			}
			::printf("\n");
		}
		std::fflush(stdout);
	}

	struct Entry {
		uint64_t                               size   = 0;
		uint64_t                               writes = 0;
		uint64_t                               reads  = 0;
		uint64_t                               outcomes[OutcomeCount] {};
		uint64_t                               download_bytes = 0;
		double                                 micros         = 0;
		std::unordered_set<uint64_t>           pages;
		uint64_t                               writer_finished = 0;
		uint64_t                               writer_running  = 0;
		uint64_t                               writer_unknown  = 0;
		uint64_t                               writer_age[AgeBuckets] {};
		uint64_t                               writer_size = 0;
		std::unordered_map<uint64_t, uint64_t> writer_shaders;
		// Faults of guest threads handed to the GPU thread (the rest are its own).
		uint64_t forwarded        = 0;
		double   forwarded_micros = 0;
	};

public:
	// GPU thread: set while a guest thread's fault is handled (BufferCache::ReadMemory).
	inline static thread_local bool s_forwarded = false;
	[[nodiscard]] static uint64_t CurrentShader() { return s_shader; }

private:
	std::unordered_map<uint64_t, Entry>   m_entries;
	std::chrono::steady_clock::time_point m_last = std::chrono::steady_clock::now();
	std::array<Mark, 1024>                m_marks {};
	size_t                                m_next_mark = 0;

	inline static thread_local uint64_t s_shader = 0;
	inline static std::atomic<uint64_t> s_changed_bytes {0};
	inline static std::atomic<uint64_t> s_compared_bytes {0};
};

// KYTY_LOCAL_HACK research (printed with KYTY_READBACK_STATS): where the GPU thread's time in the
// asynchronous readback goes: 0 lookup/unmark, 1 recording the download, 2 the submission (Flush),
// 3 FinishWriteReadback, 4 the synchronous read of an unregistered region. Only the GPU thread writes.
struct ReadbackPhases {
	uint64_t ns[9] {};
	uint64_t calls[9] {};
	std::chrono::steady_clock::time_point last = std::chrono::steady_clock::now();
};
ReadbackPhases g_rb_phases;
// 5 sync branch of ReadMemory, 6 BeginWriteReadback, 7 FinishWriteReadback, 8 other callers.
int g_rb_caller = 8;
// GPU thread: the readback being served is the GPU thread's own fault, not a guest thread's.
bool g_rb_own_fault = false;
// Research: GPU-thread own faults per (program being evaluated, page): ns and count.
std::map<std::pair<uint64_t, uint64_t>, std::pair<uint64_t, uint64_t>> g_rb_readers;
struct ReadbackCaller {
	int previous;
	explicit ReadbackCaller(int tag): previous(g_rb_caller) { g_rb_caller = tag; }
	~ReadbackCaller() { g_rb_caller = previous; }
};
void NoteReadbackPhase(int phase, std::chrono::steady_clock::time_point start) {
	const auto now = std::chrono::steady_clock::now();
	g_rb_phases.ns[phase] += static_cast<uint64_t>(
	    std::chrono::duration_cast<std::chrono::nanoseconds>(now - start).count());
	g_rb_phases.calls[phase]++;
	if (now - g_rb_phases.last >= std::chrono::seconds(5)) {
		::printf("Readback phases (5 s): lookup %.1f ms/%" PRIu64 ", record %.1f ms/%" PRIu64
		         ", flush %.1f ms/%" PRIu64 ", finish %.1f ms/%" PRIu64 ", unregistered sync %.1f ms/%" PRIu64
		         "; ReadMemoryOnGpu by caller: sync-branch %.1f ms/%" PRIu64 ", begin %.1f ms/%" PRIu64
		         ", finish %.1f ms/%" PRIu64 ", other %.1f ms/%" PRIu64 "\n",
		         static_cast<double>(g_rb_phases.ns[0]) / 1e6, g_rb_phases.calls[0],
		         static_cast<double>(g_rb_phases.ns[1]) / 1e6, g_rb_phases.calls[1],
		         static_cast<double>(g_rb_phases.ns[2]) / 1e6, g_rb_phases.calls[2],
		         static_cast<double>(g_rb_phases.ns[3]) / 1e6, g_rb_phases.calls[3],
		         static_cast<double>(g_rb_phases.ns[4]) / 1e6, g_rb_phases.calls[4],
		         static_cast<double>(g_rb_phases.ns[5]) / 1e6, g_rb_phases.calls[5],
		         static_cast<double>(g_rb_phases.ns[6]) / 1e6, g_rb_phases.calls[6],
		         static_cast<double>(g_rb_phases.ns[7]) / 1e6, g_rb_phases.calls[7],
		         static_cast<double>(g_rb_phases.ns[8]) / 1e6, g_rb_phases.calls[8]);
		std::vector<std::pair<std::pair<uint64_t, uint64_t>, std::pair<uint64_t, uint64_t>>> top(
		    g_rb_readers.begin(), g_rb_readers.end());
		std::ranges::sort(top, [](const auto& a, const auto& b) { return a.second.first > b.second.first; });
		for (size_t i = 0; i < std::min<size_t>(top.size(), 6); i++) {
			::printf("  own fault: program 0x%016" PRIx64 " page 0x%" PRIx64 ": %.1f ms / %" PRIu64 "\n",
			         top[i].first.first, top[i].first.second,
			         static_cast<double>(top[i].second.first) / 1e6, top[i].second.second);
		}
		g_rb_readers.clear();
		g_rb_phases = {};
		g_rb_phases.last = now;
	}
}


bool AsyncWriteReadbackEnabled() {
	static auto& enabled = Common::LiveSwitches::Get("KYTY_ASYNC_WRITE_READBACK", 0);
	return enabled.load(std::memory_order_relaxed) != 0;
}

// KYTY_ASYNC_READ_SNAPSHOT=1 (live, default 0): see Memory::RequestServeFromBacking. Wolverine's
// busiest readback buffer is rewritten by the GPU before the asynchronous download lands (run 41:
// the writer was still running for 194 of 402 readbacks per 5 s); the guest never waits for it,
// so it reads whatever the GPU had written by then, as the download captured.
bool AsyncReadSnapshotEnabled() {
	static auto& enabled = Common::LiveSwitches::Get("KYTY_ASYNC_READ_SNAPSHOT", 0);
	return enabled.load(std::memory_order_relaxed) != 0;
}

// KYTY_RACE_READS=1 (live, default 0; 2026-10-10): in heavy combat the game threads spend ~18 s per
// 5 s (summed) in read faults on GPU-written pages (hi5 WAIT_STATS): a few pages of three big
// buffers (22/105/7 MiB) rewritten every frame by compute shaders 85a58319, 7b779927, 7e7bc09f,
// read by the game while that GPU work is still running (cm2 READBACK_STATS: writer bound <1 ms
// ago, not finished) -- the game did not wait for its label, it races the GPU. Each read waited
// for the GPU to drain to that work and downloaded a 4 MiB window (~4-5 ms). On the console such
// a read returns whatever memory holds at that moment. With the switch, a guest read of a page
// whose buffer's last GPU write has not finished is served from guest memory at once (the bytes
// of the last completed download or CPU write: at most about a frame old) and a download is
// started without waiting, so the next read sees fresher bytes. Reads after the writer finished
// keep the exact path. The command processor's own reads never take this path.
bool RaceReadsEnabled() {
	static auto& enabled = Common::LiveSwitches::Get("KYTY_RACE_READS", 0);
	return enabled.load(std::memory_order_relaxed) != 0;
}

bool AsyncReadReadbackEnabled() {
	// Run 37 A/B at the Wolverine spot: readback time on the GPU thread 147 -> 113 ms/s, mean
	// frame 73.5 -> 72.5 ms. The busiest buffer gains nothing: the GPU rewrites its pages before
	// the download lands, so FinishWriteReadback falls back to the synchronous path.
	static auto& enabled = Common::LiveSwitches::Get("KYTY_ASYNC_READBACK", 1);
	return enabled.load(std::memory_order_relaxed) != 0;
}

// A guest fault on GPU-written memory downloads an aligned window around the faulting bytes, so
// nearby accesses share one GPU drain. KYTY_READBACK_WINDOW_KB (a live switch; a power of two,
// at least the tracker page) changes its width for A/B measurements. The default, 2 MiB, measured
// best in Marvel's Wolverine (512 KiB +3.5% frame time, 8 MiB +6%, 64 KiB +17%).
uint64_t ReadbackWindowSize() {
	constexpr int64_t DefaultKib = 2048;
	static auto&      kib        = Common::LiveSwitches::Get("KYTY_READBACK_WINDOW_KB", DefaultKib);
	const auto        value      = kib.load(std::memory_order_relaxed);
	const auto        bytes      = static_cast<uint64_t>(value) * 1024;
	if (value <= 0 || value > 1024 * 1024 || bytes < TRACKER_PAGE_SIZE ||
	    (bytes & (bytes - 1)) != 0) {
		return static_cast<uint64_t>(DefaultKib) * 1024;
	}
	return bytes;
}

} // namespace

void BufferCache::SetReadbackStatsShader(uint64_t hash) {
	ReadbackStats::SetShader(hash);
}

void BufferCache::WriteDataBuffer(Buffer& buffer, uint64_t address, const void* source,
                                  uint64_t size) {
	auto* bytes = static_cast<const uint8_t*>(source);
	while (size != 0) {
		const auto chunk  = std::min(size, m_staging_buffer.Size());
		const auto offset = m_staging_buffer.Copy(bytes, chunk, 4);
		buffer.CopyFrom(m_scheduler.Current(), m_staging_buffer, offset, buffer.Offset(address),
		                chunk, vk::AccessFlagBits::eHostWrite);
		bytes += chunk;
		address += chunk;
		size -= chunk;
	}
}

void BufferCache::Register(BufferId id) {
	ChangeRegister<true>(id);
}

void BufferCache::Unregister(BufferId id) {
	ChangeRegister<false>(id);
}

template <bool insert>
void BufferCache::ChangeRegister(BufferId id) {
	auto& buffer = m_slot_buffers[id];
	PageTable::PageRange pages {};
	EXIT_IF(!(GuestRange {buffer.CpuAddress(), buffer.Size()}.Valid()) ||
	        !PageTable::TryGetPageRange(buffer.CpuAddress(), buffer.Size(), pages));
	for (size_t page = pages.first; page < pages.last_exclusive; ++page) {
		if constexpr (insert) {
			m_page_table[page] = id;
		} else {
			m_page_table[page] = {};
		}
	}
	const auto size_pages = pages.last_exclusive - pages.first;
	if constexpr (insert) {
		const auto [it, inserted] = m_buffers.emplace(buffer.CpuAddress(), id);
		(void)it;
		EXIT_IF(!inserted);
		m_bda_ever_registered.Add(buffer.CpuAddress(), size_pages << CACHING_PAGEBITS);
		m_total_used_memory += buffer.Size();
		g_cpu_dirty_epoch.fetch_add(1, std::memory_order_release);
		buffer.lru_id = m_lru_cache.Insert(id, LruClock());
		WriteBdaEntries(PageIndex(buffer.CpuAddress()), size_pages, buffer.BufferDeviceAddress());
	} else {
		const auto found = m_buffers.find(buffer.CpuAddress());
		EXIT_IF(found == m_buffers.end() || found->second != id);
		m_buffers.erase(found);
		EXIT_IF(buffer.Size() > m_total_used_memory);
		m_total_used_memory -= buffer.Size();
		m_lru_cache.Free(buffer.lru_id);
		WriteBdaEntries(PageIndex(buffer.CpuAddress()), size_pages, 0);
		buffer.is_deleted = true;
	}
}

void BufferCache::WriteBdaEntries(uint64_t first_page, uint64_t pages,
                                  vk::DeviceAddress first_address) {
	constexpr auto Entry = sizeof(vk::DeviceAddress);
	EXIT_IF(first_page + pages > CACHING_NUMPAGES);
	if (m_bda_directory.empty()) {
		m_bda_directory.assign(BDA_DIRECTORY_ENTRIES, 0);
		m_bda_chunk_live.assign(BDA_CHUNK_COUNT, 0);
		m_bda_free_chunks.resize(BDA_CHUNK_COUNT);
		for (uint32_t slot = 0; slot < BDA_CHUNK_COUNT; slot++) {
			m_bda_free_chunks[slot] = static_cast<uint32_t>(BDA_CHUNK_COUNT - 1u - slot);
		}
	}
	std::vector<vk::DeviceAddress> addresses;
	for (uint64_t page = first_page; page < first_page + pages;) {
		const auto directory = page >> BDA_CHUNK_BITS;
		const auto run_end   = std::min(first_page + pages, (directory + 1u) << BDA_CHUNK_BITS);
		const auto count     = run_end - page;
		auto&      slot_plus = m_bda_directory[directory];
		if (first_address != 0 && slot_plus == 0) {
			if (m_bda_free_chunks.empty()) {
				EXIT("BufferCache: BDA chunk pool exhausted (%" PRIu64 " chunks of %" PRIu64
				     " MiB guest space); raise BDA_CHUNK_COUNT\n",
				     BDA_CHUNK_COUNT, (BDA_CHUNK_PAGES << CACHING_PAGEBITS) >> 20u);
			}
			slot_plus = m_bda_free_chunks.back() + 1u;
			m_bda_free_chunks.pop_back();
			m_bda_chunks_peak = std::max(
			    m_bda_chunks_peak, static_cast<uint32_t>(BDA_CHUNK_COUNT - m_bda_free_chunks.size()));
			const vk::DeviceAddress chunk_element =
			    BDA_DIRECTORY_ENTRIES + uint64_t {slot_plus - 1u} * BDA_CHUNK_PAGES;
			WriteDataBuffer(m_bda_pagetable_buffer, directory * Entry, &chunk_element, Entry);
		}
		if (slot_plus != 0) {
			const auto slot    = slot_plus - 1u;
			const auto element = BDA_DIRECTORY_ENTRIES + uint64_t {slot} * BDA_CHUNK_PAGES +
			                     (page & (BDA_CHUNK_PAGES - 1u));
			if (first_address != 0) {
				addresses.resize(count);
				for (uint64_t i = 0; i < count; i++) {
					addresses[i] = first_address + ((page - first_page + i) << CACHING_PAGEBITS);
				}
				WriteDataBuffer(m_bda_pagetable_buffer, element * Entry, addresses.data(),
				                count * Entry);
				m_bda_chunk_live[slot] += static_cast<uint32_t>(count);
			} else {
				m_bda_pagetable_buffer.Fill(element * Entry, count * Entry, 0);
				EXIT_IF(m_bda_chunk_live[slot] < count);
				m_bda_chunk_live[slot] -= static_cast<uint32_t>(count);
				if (m_bda_chunk_live[slot] == 0) {
					// Its pages are all zero again: the chunk can serve another region.
					m_bda_pagetable_buffer.Fill(directory * Entry, Entry, 0);
					m_bda_free_chunks.push_back(slot);
					slot_plus = 0;
				}
			}
		}
		page = run_end;
	}
}

void BufferCache::DescribeGuestPages(uint64_t address, uint64_t size) {
	PageTable::PageRange pages {};
	if (!PageTable::TryGetPageRange(address, size, pages)) {
		return;
	}
	for (size_t page = pages.first; page < pages.last_exclusive; ++page) {
		const auto  page_address = static_cast<uint64_t>(page) << CACHING_PAGEBITS;
		const auto* id           = m_page_table.Find(page);
		const auto* buffer = id != nullptr && *id ? m_slot_buffers.try_get(*id) : nullptr;
		if (buffer == nullptr) {
			std::printf("      guest page 0x%" PRIx64 ": no buffer (table entry should be 0)\n",
			            page_address);
			continue;
		}
		const auto first_page = PageIndex(buffer->CpuAddress()) << CACHING_PAGEBITS;
		std::printf("      guest page 0x%" PRIx64 ": buffer guest=0x%" PRIx64 " size=0x%" PRIx64
		            " deleted=%d table va=0x%" PRIx64 "\n",
		            page_address, buffer->CpuAddress(), buffer->Size(), buffer->is_deleted ? 1 : 0,
		            buffer->HasDeviceAddress()
		                ? buffer->BufferDeviceAddress() + (page_address - first_page)
		                : 0);
	}
}

void BufferCache::TouchBuffer(const Buffer& buffer) {
	if (!buffer.is_deleted) {
		m_lru_cache.Touch(buffer.lru_id, LruClock());
	}
}

void BufferCache::DeleteBuffer(BufferId id) {
	if (IsBufferInvalid(id)) {
		return;
	}
	for (size_t i = 0; i < GuestCopySlots; i++) {
		if (m_guest_copy_slots[i].busy &&
		    m_guest_copy_slots[i].buffer_handle == m_slot_buffers[id].Handle()) {
			CompleteGuestCopy(i);
		}
	}
	Unregister(id);
	if (m_scheduler.Active()) {
		m_scheduler.DeferOperation([this, id] { m_slot_buffers.erase(id); });
	} else {
		m_slot_buffers.erase(id);
	}
}

bool BufferCache::DownloadBufferMemory(Buffer& buffer, uint64_t vaddr, uint64_t size) {
	// One reservation cannot exceed the download ring, so a larger range goes in ring-sized
	// windows; a window that does not fit drains the ring before it is mapped.
	const auto capacity = m_download_buffer.Size();
	bool       any      = false;
	for (uint64_t offset = 0; offset < size; offset += capacity) {
		any |= DownloadBufferWindow(buffer, vaddr + offset, std::min(capacity, size - offset));
	}
	return any;
}

bool BufferCache::DownloadBufferWindow(Buffer& buffer, uint64_t vaddr, uint64_t size) {
	std::vector<vk::BufferCopy> copies;
	uint64_t                    total_size     = 0;
	const auto                  buffer_address = buffer.CpuAddress();
	m_memory_tracker.ForEachDownloadRange<false>(
	    vaddr, size, [&](uint64_t address, uint64_t bytes) noexcept {
		    m_memory_tracker.ValidateGpuDirtyPages(m_gpu_modified_ranges, address, bytes,
		                                           "buffer download");
		    std::unique_lock lock(m_dirty_ranges_mutex);
		    m_gpu_modified_ranges.ForEachInRange(address, bytes, [&](uint64_t start, uint64_t end) {
			    copies.emplace_back(start - buffer_address, total_size, end - start);
			    // Keep packed ranges on separate cache lines, as in shadPS4.
			    total_size += Common::AlignUp(end - start, 64);
			    m_downloading_ranges.Add(start, end - start);
		    });
		    m_gpu_modified_ranges.Subtract(address, bytes);
	    });
	if (copies.empty()) {
		return false;
	}
	const auto capacity = m_download_buffer.Size();
	for (size_t first = 0; first < copies.size();) {
		const auto base       = copies[first].dstOffset;
		auto       last       = first;
		uint64_t   batch_size = 0;
		while (last < copies.size()) {
			const auto end = copies[last].dstOffset - base + Common::AlignUp(copies[last].size, 64);
			if (end > capacity) {
				break;
			}
			batch_size = end;
			last++;
		}
		EXIT_IF(last == first);
		std::vector<vk::BufferCopy> batch(copies.begin() + static_cast<std::ptrdiff_t>(first),
		                                  copies.begin() + static_cast<std::ptrdiff_t>(last));
		for (auto& copy: batch) {
			copy.dstOffset -= base;
		}
		DownloadBufferCopies(buffer, std::move(batch), batch_size);
		first = last;
	}
	return true;
}

void BufferCache::DownloadBufferCopies(Buffer& buffer, std::vector<vk::BufferCopy> copies,
                                       uint64_t total_size) {
	const auto buffer_address = buffer.CpuAddress();
	auto [mapped, offset]     = m_download_buffer.Map(total_size, 64);
	std::unique_ptr<Buffer> temporary;
	if (mapped == nullptr) {
		temporary = std::make_unique<Buffer>(m_graphics, m_scheduler, MemoryUsage::Download, 0,
		                                     vk::BufferUsageFlagBits::eTransferDst, total_size);
		mapped = temporary->Mapped().data();
	} else {
		m_download_buffer.Commit();
	}
	const auto& download = temporary ? *temporary : m_download_buffer;
	for (auto& copy: copies) {
		copy.dstOffset += offset;
	}

	auto& command = m_scheduler.Current();
	command.EndRendering();
	const auto              native = command.Recorder();
	vk::BufferMemoryBarrier before {};
	before.srcAccessMask       = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
	before.dstAccessMask       = vk::AccessFlagBits::eTransferRead;
	before.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	before.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	before.buffer              = buffer.Handle();
	before.offset              = 0;
	before.size                = buffer.Size();
	native.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
	                       vk::PipelineStageFlagBits::eTransfer, {}, 0, nullptr, 1, &before, 0,
	                       nullptr);
	native.copyBuffer(buffer.Handle(), download.Handle(),
	                  static_cast<uint32_t>(copies.size()), copies.data());

	auto after          = before;
	after.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
	after.dstAccessMask = vk::AccessFlagBits::eHostRead;
	after.buffer        = download.Handle();
	after.offset        = offset;
	after.size          = total_size;
	native.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
	                       vk::PipelineStageFlagBits::eAllCommands |
	                           vk::PipelineStageFlagBits::eHost,
	                       {}, 0, nullptr, 1, &after, 0, nullptr);
	m_scheduler.DeferPriorityOperation([this, mapped, offset, total_size, buffer_address,
	                                    copies = std::move(copies), owner = std::move(temporary)] {
		(owner ? *owner : m_download_buffer).Invalidate(offset, total_size);
		// KYTY_READBACK_COMPARE (live, default 1): the stats' byte comparison of every download.
		static auto& compare = Common::LiveSwitches::Get("KYTY_READBACK_COMPARE", 1);
		if (ReadbackStats::On() && compare.load(std::memory_order_relaxed) != 0) {
			uint64_t             changed = 0;
			uint64_t             total   = 0;
			std::vector<uint8_t> current;
			for (const auto& copy: copies) {
				current.resize(copy.size);
				const auto* downloaded = mapped + (copy.dstOffset - offset);
				if (Libs::LibKernel::Memory::TryReadBacking(buffer_address + copy.srcOffset,
				                                            current.data(), copy.size)) {
					for (uint64_t b = 0; b < copy.size; b++) {
						changed += current[b] != downloaded[b] ? 1 : 0;
					}
					total += copy.size;
				}
			}
			ReadbackStats::NoteCompare(changed, total);
		}
		for (const auto& copy: copies) {
			Libs::LibKernel::Memory::WriteBacking(buffer_address + copy.srcOffset,
			                                      mapped + (copy.dstOffset - offset), copy.size);
		}
		std::unique_lock lock(m_dirty_ranges_mutex);
		for (const auto& copy: copies) {
			m_downloading_ranges.Subtract(buffer_address + copy.srcOffset, copy.size);
		}
	});
}

BufferCache::BufferCache(GraphicContext& graphics, CommandScheduler& scheduler,
                         PageManager& page_manager, TextureCache& texture_cache)
    : m_graphics(graphics), m_scheduler(scheduler), m_fault_manager(graphics, scheduler, *this),
      m_gds_buffer(graphics, scheduler, MemoryUsage::Stream, 0, AllFlags, GdsBufferSize),
      m_bda_pagetable_buffer(graphics, scheduler, MemoryUsage::DeviceLocal, 0, AllFlags,
                             BDA_PAGETABLE_SIZE),
      m_memory_tracker(page_manager),
      m_staging_buffer(graphics, scheduler, MemoryUsage::Upload, 512 * MiB),
      m_stream_buffer(graphics, scheduler, MemoryUsage::Stream, 64 * MiB),
      m_download_buffer(graphics, scheduler, MemoryUsage::Download, 64 * MiB),
      m_device_buffer(graphics, scheduler, MemoryUsage::DeviceLocal, 128 * MiB),
      m_texture_cache(texture_cache) {
	std::memset(m_gds_buffer.Mapped().data(), 0, static_cast<size_t>(m_gds_buffer.Size()));
	m_gds_buffer.Flush(0, m_gds_buffer.Size());
	SetVulkanObjectNameF(m_graphics.device, m_bda_pagetable_buffer.Handle(),
	                     "BDA Page Table Buffer");
	SetGuestPageDescriber(
	    [this](uint64_t address, uint64_t size) { DescribeGuestPages(address, size); });
	const auto null_id =
	    m_slot_buffers.insert(m_graphics, m_scheduler, MemoryUsage::DeviceLocal, 0, AllFlags, 16);
	EXIT_IF(null_id != NULL_BUFFER_ID);
	SetVulkanObjectNameF(m_graphics.device, GetBuffer(null_id).Handle(), "Kyty.NullBuffer");
	if (!m_graphics.CanReportMemoryUsage()) {
		return;
	}
	constexpr int64_t GiB              = 1024ll * 1024 * 1024;
	constexpr int64_t target_threshold = 8 * GiB;
	const auto        budget =
	    static_cast<int64_t>(std::min<uint64_t>(m_graphics.GetTotalMemoryBudget(), INT64_MAX));
	const auto threshold = std::min(budget, target_threshold);
	const auto expected  = std::min(budget - 6 * threshold / 10, budget - GiB);
	const auto critical  = std::min(budget - 2 * threshold / 10, budget - GiB / 2);
	m_trigger_gc_memory  = static_cast<uint64_t>(std::max<int64_t>(expected, GiB));
	m_critical_gc_memory = static_cast<uint64_t>(std::max<int64_t>(critical, 2 * GiB));
}

BufferCache::~BufferCache() {
	if (auto& state = m_copy_queue_readback; state.pool != nullptr) {
		state.staging.reset();
		m_graphics.device.destroyFence(state.fence, nullptr);
		m_graphics.device.destroyCommandPool(state.pool, nullptr);
		state = {};
	}
	if (!m_gpu_modified_ranges.Empty()) {
		EXIT("BufferCache: destroyed with pending GPU-modified ranges\n");
	}
	for (const auto& [vaddr, id]: m_buffers) {
		(void)vaddr;
		const auto& buffer = m_slot_buffers[id];
		if (m_memory_tracker.IsRegionGpuModified(buffer.CpuAddress(), buffer.Size())) {
			EXIT("BufferCache: destroyed with GPU-modified buffer\n");
		}
	}
	m_buffers.clear();
}

void BufferCache::InvalidateMemory(uint64_t vaddr, uint64_t size) {
	KYTY_PROFILER_FUNCTION();
	if (!GuestRange {vaddr, size}.Valid()) {
		EXIT("BufferCache: invalid memory-invalidation range\n");
	}
	m_memory_tracker.InvalidateRegion(vaddr, size,
	                                  [this, vaddr, size] { ReadMemory(vaddr, size, true); });
}

void BufferCache::InvalidateMemoryAhead(uint64_t vaddr, uint64_t size) {
	KYTY_PROFILER_FUNCTION();
	if (GuestRange {vaddr, size}.Valid()) {
		m_memory_tracker.InvalidateRegionAhead(vaddr, size);
	}
}

bool BufferCache::WriteClean(uint64_t vaddr, const void* data, uint64_t size) {
	if (!GuestGpu::IsGpuThread() || size == 0 || !GuestRange {vaddr, size}.Valid() ||
	    !m_memory_tracker.IsRegionGpuModified(vaddr, size)) {
		return false;
	}
	{
		// A download publishes the GPU's older value of these bytes when it lands.
		std::shared_lock lock(m_dirty_ranges_mutex);
		if (m_downloading_ranges.Intersects(vaddr, size)) {
			return false;
		}
	}

	const auto* owner = m_page_table.Find(vaddr >> PageTable::kPageBits);
	if (owner == nullptr || !*owner || !m_slot_buffers[*owner].IsInBounds(vaddr, size) ||
	    !Libs::LibKernel::Memory::TryWriteBacking(vaddr, data, size)) {
		return false;
	}
	WriteDataBuffer(m_slot_buffers[*owner], vaddr, data, size);
	if (HasGpuDirtyBytes(vaddr, size)) {
		// Overwritten in full: guest memory holds the value the GPU copy will have.
		std::unique_lock lock(m_dirty_ranges_mutex);
		m_gpu_modified_ranges.Subtract(vaddr, size);
	}
	m_texture_cache.InvalidateMemory(vaddr, size);
	return true;
}

// KYTY_GUEST_COPY_QUEUE: where BeginWriteReadback reports a guest copy slot (GPU thread, set by
// the game thread's ReadMemory around the command).
struct GuestCopyOut {
	int      slot       = -1;
	uint64_t generation = 0;
};
static GuestCopyOut* s_guest_copy_out = nullptr;

void BufferCache::ReadMemory(uint64_t vaddr, uint64_t size, bool is_write) {
	KYTY_PROFILER_FUNCTION();
	if (!GuestGpu::IsGpuThread() && CommandScheduler::InDeferredOperation()) {
		EXIT("unsupported buffer readback from an asynchronous GPU completion, "
		     "addr=0x%016" PRIx64 " size=0x%016" PRIx64 "\n",
		     vaddr, size);
	}
	auto& gpu = m_scheduler.Context().GetGpu();
	// KYTY_ASYNC_WRITE_READBACK=1 (a live switch, off by default): a guest write to GPU-written
	// memory no longer drains the GPU on the GPU thread. The GPU thread records the window's
	// download and submits it; the faulting guest thread waits for that submission, and a second
	// command lifts the window's GPU ownership, or resolves the fault synchronously if the GPU
	// wrote there again in the meantime.
	// KYTY_ASYNC_READBACK=1 (live, default 0) does the same for guest reads. Wolverine run 36
	// (KYTY_READBACK_STATS): every readback at the spot came from a guest thread, and serving
	// them synchronously cost the GPU thread ~1.3 s per 5 s, mostly waiting for the GPU.
	if (!GuestGpu::IsGpuThread() && !Libs::LibKernel::Memory::ForceSyncReadback() &&
	    (is_write ? AsyncWriteReadbackEnabled() : AsyncReadReadbackEnabled())) {
		uint64_t tick         = 0;
		uint64_t window_begin = 0;
		uint64_t window_end   = 0;
		bool     raced        = false;
		GuestCopyOut copy_out;
		std::optional<Common::WaitStats::Inner> hop(std::in_place, Common::WaitStats::ReadHop);
		gpu.SendCommandSync([&] {
			ReadbackStats::s_forwarded = true;
			if (!is_write && RaceReadsEnabled()) {
				raced = TryRaceRead(vaddr, size);
			}
			if (!raced) {
				s_guest_copy_out = &copy_out;
				Common::WaitStats::CpPart begin_part(Common::WaitStats::CpBegin);
				tick = BeginWriteReadback(vaddr, size, is_write, window_begin, window_end);
				s_guest_copy_out = nullptr;
			}
			ReadbackStats::s_forwarded = false;
		});
		hop.reset();
		if (raced) {
			Libs::LibKernel::Memory::RequestServeFromBacking();
			return;
		}
		if (copy_out.slot >= 0) {
			{
				std::optional<Common::WaitStats::Inner> part;
				if (!is_write) {
					part.emplace(Common::WaitStats::ReadGpu);
				}
				WaitGuestCopy(copy_out.slot);
			}
			Common::WaitStats::Inner finish(Common::WaitStats::ReadFinish);
			gpu.SendCommandSync([&] {
				FinishGuestCopy(copy_out.slot, copy_out.generation, vaddr, size, is_write);
			});
			return;
		}
		if (tick != 0) {
			{
				KYTY_PROFILER_BLOCK("BufferCache::WaitWriteReadback");
				std::optional<Common::WaitStats::Inner> part;
				if (!is_write) {
					part.emplace(Common::WaitStats::ReadGpu);
				}
				// Submitted by BeginWriteReadback, so this waits without submitting.
				EXIT_IF(tick >= m_scheduler.CurrentTick());
				m_scheduler.Wait(tick);
				m_scheduler.WaitPriorityOperations(tick);
			}
			bool                     redirtied = false;
			Common::WaitStats::Inner finish(Common::WaitStats::ReadFinish);
			gpu.SendCommandSync([&] {
				redirtied = !FinishWriteReadback(vaddr, size, is_write, window_begin, window_end,
				                                 tick, !is_write && AsyncReadSnapshotEnabled());
			});
			if (redirtied) {
				// The download holds the bytes as of this read; the fault handler completes the
				// load from them (KYTY_ASYNC_READ_SNAPSHOT).
				Libs::LibKernel::Memory::RequestServeFromBacking();
			}
		}
		return;
	}
	const bool own_fault = GuestGpu::IsGpuThread();
	Common::WaitStats::Inner sync_part(Common::WaitStats::ReadSync);
	gpu.SendCommandSync([this, vaddr, size, is_write, own_fault] {
		ReadbackStats::s_forwarded = true;
		ReadbackCaller caller(5);
		const bool previous_own = std::exchange(g_rb_own_fault, own_fault);
		const auto own_start    = std::chrono::steady_clock::now();
		ReadMemoryOnGpu(vaddr, size, is_write);
		g_rb_own_fault = previous_own;
		if (own_fault && ReadbackStats::On()) {
			// Research: which program the GPU thread was evaluating, and the faulting page.
			const auto ns = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
			                                          std::chrono::steady_clock::now() - own_start)
			                                          .count());
			auto& e = g_rb_readers[{ReadbackStats::CurrentShader(), vaddr & ~uint64_t {0xfff}}];
			e.first += ns;
			e.second++;
		}
		ReadbackStats::s_forwarded = false;
	});
}

void BufferCache::ReadMemoryOnGpu(uint64_t vaddr, uint64_t size, bool is_write) {
	Common::WaitStats::Scope wait(Common::WaitStats::Readback);
	KYTY_PROFILER_BLOCK("BufferCache::ReadMemory(GPU thread)");
	struct CallerPhase {
		int                                   tag   = g_rb_caller;
		std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
		~CallerPhase() {
			if (ReadbackStats::On()) {
				NoteReadbackPhase(tag, start);
			}
		}
	} caller_phase;
	ReadbackStats::Scope stats(vaddr, is_write);
	if (is_write && !IsRegionRegistered(vaddr, size)) {
		return;
	}
	auto& buffer = m_slot_buffers[FindBuffer(vaddr, size)];
	stats.SetBuffer(buffer.CpuAddress(), buffer.Size());

	// The page is protected as GPU-written, but none of its bytes are waiting for a download:
	// the bytes the GPU wrote are elsewhere in the window, or were downloaded with another
	// page. The page is current, so lift its protection. Downloading the window instead drained
	// the GPU (~30 ms a fault) for guest reads of structs next to the command processor's
	// labels.
	const auto page_begin = Common::AlignDown(vaddr, TRACKER_PAGE_SIZE);
	const auto page_end   = Common::AlignUp(vaddr + size, TRACKER_PAGE_SIZE);
	// Pages of an asynchronous write readback stay GPU-owned until its download is
	// published.
	CompletePendingWriteReadbacks(page_begin, page_end);
	if (m_memory_tracker.IsRegionGpuModified(page_begin, page_end - page_begin) &&
	    !HasGpuDirtyBytes(page_begin, page_end - page_begin)) {
		m_memory_tracker.UnmarkRegionAsGpuModified(page_begin, page_end - page_begin);
		if (is_write) {
			m_memory_tracker.MarkRegionAsCpuModified(vaddr, size);
		}
		stats.SetOutcome(ReadbackStats::Unmarked);
		return;
	}

	if (TryDirectReadback(buffer, vaddr, size, is_write)) {
		stats.SetOutcome(ReadbackStats::DirectRead);
		return;
	}

	// Widen nearby CPU reads so they share one GPU drain.
	uint64_t WindowSize = ReadbackWindowSize();
	// KYTY_LOCAL_HACK KYTY_READBACK_GPU_WINDOW_KB (live, default 0 = the same window): the GPU
	// thread's own faults (resource materialization reading GPU-written dwords) download a
	// smaller window. Wolverine 10-05 rp2: ~290 such faults per 5 s, 2 MiB each, ~1.1 ms apiece.
	static auto& gpu_window_kb = Common::LiveSwitches::Get("KYTY_READBACK_GPU_WINDOW_KB", 0);
	if (const auto kb = gpu_window_kb.load(std::memory_order_relaxed);
	    kb > 0 && g_rb_own_fault) {
		const auto bytes = std::max<uint64_t>(static_cast<uint64_t>(kb) * 1024, TRACKER_PAGE_SIZE);
		if ((bytes & (bytes - 1)) == 0) {
			WindowSize = std::min(WindowSize, bytes);
		}
	}
	const auto     buffer_begin = buffer.CpuAddress();
	const auto     buffer_end   = buffer_begin + buffer.Size();
	const auto     window_begin = std::max(Common::AlignDown(vaddr, WindowSize), buffer_begin);
	const auto window_end = std::min(std::max(window_begin + WindowSize, vaddr + size), buffer_end);

	if (stats.Enabled()) {
		stats.FindWriter([this](uint64_t tick) { return m_scheduler.IsFree(tick); });
		uint64_t         dirty_bytes = 0;
		std::shared_lock lock(m_dirty_ranges_mutex);
		m_gpu_modified_ranges.ForEachInRange(
		    window_begin, window_end - window_begin,
		    [&](uint64_t start, uint64_t end) { dirty_bytes += end - start; });
		stats.SetDownloadBytes(dirty_bytes);
	}
	if (TryCopyQueueReadback(buffer, window_begin, window_end)) {
		stats.SetOutcome(ReadbackStats::Downloaded);
		m_memory_tracker.UnmarkRegionAsGpuModified(window_begin, window_end - window_begin);
	} else if (DownloadBufferMemory(buffer, window_begin, window_end - window_begin)) {
		stats.SetOutcome(ReadbackStats::Downloaded);
		const auto tick = m_scheduler.CurrentTick();
		m_scheduler.Wait(tick);
		m_scheduler.WaitPriorityOperations(tick);
		m_memory_tracker.UnmarkRegionAsGpuModified(window_begin, window_end - window_begin);
	} else {
		stats.SetOutcome(ReadbackStats::NothingToDownload);
	}
	if (is_write) {
		m_memory_tracker.MarkRegionAsCpuModified(vaddr, size);
	}
}

// KYTY_DIRECT_READBACK=1 (a live switch, off by default): when the buffer lives in host-visible
// device memory and every GPU write recorded into it has completed, the faulting pages' GPU-written
// bytes are copied straight from the mapping into guest memory: no GPU copy, submission or drain.
// Submissions end with a device-to-host barrier while such buffers exist (CommandScheduler).
// KYTY_LOCAL_HACK KYTY_DIRECT_READBACK=2: also when the writer has not finished, the GPU thread
// waits for the writer's tick only (submitting the open command buffer if the writer is in it)
// instead of downloading the window and draining everything recorded. RADV has no second queue in
// the graphics family, so the copy-queue readback never runs there (Wolverine 10-05 rg1: ~300 GPU
// thread faults per 5 s, ~0.75 ms each, all full drains).
bool BufferCache::TryDirectReadback(Buffer& buffer, uint64_t vaddr, uint64_t size, bool is_write,
                                    int forced_mode) {
	static auto& mode = Common::LiveSwitches::Get("KYTY_DIRECT_READBACK", 0);
	const auto   direct = forced_mode != 0 ? forced_mode : mode.load(std::memory_order_relaxed);
	if (direct == 0 || buffer.Mapped().empty()) {
		return false;
	}
	WaitUploadWorker(); // mapped bytes must hold every queued upload
	if (!m_scheduler.IsFree(buffer.last_gpu_write_tick)) {
		if (direct < 2) {
			return false;
		}
		KYTY_PROFILER_BLOCK("BufferCache::DirectReadbackWait");
		// Wait() submits the open command buffer when the writer is in it.
		m_scheduler.Wait(buffer.last_gpu_write_tick);
		m_scheduler.WaitPriorityOperations(buffer.last_gpu_write_tick);
		m_direct_waits++;
	}
	const auto page_begin = Common::AlignDown(vaddr, TRACKER_PAGE_SIZE);
	const auto page_end   = Common::AlignUp(vaddr + size, TRACKER_PAGE_SIZE);
	if (!buffer.IsInBounds(page_begin, page_end - page_begin) ||
	    OverlapsPendingWriteReadback(page_begin, page_end)) {
		return false;
	}
	{
		KYTY_PROFILER_BLOCK("BufferCache::DirectReadback");
		std::unique_lock lock(m_dirty_ranges_mutex);
		// A queued download of these bytes would publish them again when it lands.
		if (m_downloading_ranges.Intersects(page_begin, page_end - page_begin)) {
			return false;
		}
		const auto* mapped = buffer.Mapped().data();
		m_gpu_modified_ranges.ForEachInRange(
		    page_begin, page_end - page_begin, [&](uint64_t start, uint64_t end) {
			    Libs::LibKernel::Memory::WriteBacking(start, mapped + buffer.Offset(start),
			                                          end - start);
		    });
		m_gpu_modified_ranges.Subtract(page_begin, page_end - page_begin);
	}
	m_memory_tracker.UnmarkRegionAsGpuModified(page_begin, page_end - page_begin);
	if (is_write) {
		m_memory_tracker.MarkRegionAsCpuModified(vaddr, size);
	}
	m_direct_readbacks++;
	return true;
}

// Research: KYTY_COPY_QUEUE_READBACK=1 (live). A readback recorded the window's download into the
// current command buffer and waited for its tick: a drain of everything recorded and queued,
// ~4 ms a fault. In Wolverine the writer of the faulting bytes has almost always finished long
// before (readback stats: 1016 of 1017), so the bytes are final in device memory. When every GPU
// write to the buffer has executed, the GPU-written bytes of the window are copied on the second
// queue of the graphics family instead, behind nothing: its submission waits for the writer's
// tick on the timeline semaphore (the memory dependency), and only this copy is waited for.
static bool CopyQueueReadbackEnabled() {
	// Wolverine run 18, same process: 0 -> 110 frames / 15 s, 1 -> 126-136; readback time on
	// Thread_Gpu 1.22 -> 0.62 s per 5 s; no visual change.
	static auto& enabled = Common::LiveSwitches::Get("KYTY_COPY_QUEUE_READBACK", 1);
	return enabled.load(std::memory_order_relaxed) != 0;
}

// KYTY_COPY_QUEUE_INFLIGHT=1 (live, default 0): the copy-queue readback also serves buffers whose
// last GPU write has not finished yet. Its submission already waits for the writer's tick on the
// timeline semaphore, so only that work is waited for; before, such a readback recorded the copy
// on the main queue and waited for everything recorded so far (run 31 readback stats: 180 of 406
// readbacks of Wolverine's busiest buffer per 5 s). A writer still in the open command buffer is
// submitted first.
static bool CopyQueueInflightEnabled() {
	static auto& enabled = Common::LiveSwitches::Get("KYTY_COPY_QUEUE_INFLIGHT", 0);
	return enabled.load(std::memory_order_relaxed) != 0;
}

bool BufferCache::TryCopyQueueReadback(Buffer& buffer, uint64_t window_begin, uint64_t window_end) {
	constexpr uint64_t StagingSize = 4ull * 1024 * 1024;
	if (!CopyQueueReadbackEnabled() || m_graphics.readback_queue == nullptr ||
	    OverlapsPendingWriteReadback(window_begin, window_end)) {
		return false;
	}
	const bool writer_in_flight = !m_scheduler.IsFree(buffer.last_gpu_write_tick);
	if (writer_in_flight && !CopyQueueInflightEnabled()) {
		return false;
	}
	std::vector<vk::BufferCopy> copies;
	uint64_t                    total_size     = 0;
	const auto                  buffer_address = buffer.CpuAddress();
	{
		std::shared_lock lock(m_dirty_ranges_mutex);
		// A queued main-queue download of these bytes would publish them again when it lands.
		if (m_downloading_ranges.Intersects(window_begin, window_end - window_begin)) {
			return false;
		}
		m_memory_tracker.ForEachDownloadRange<false>(
		    window_begin, window_end - window_begin,
		    [&](uint64_t address, uint64_t bytes) noexcept {
			    m_gpu_modified_ranges.ForEachInRange(
			        address, bytes, [&](uint64_t start, uint64_t end) {
				        copies.emplace_back(start - buffer_address, total_size, end - start);
				        total_size += Common::AlignUp(end - start, 64);
			        });
		    });
	}
	if (copies.empty() || total_size > StagingSize) {
		return false;
	}
	if (writer_in_flight) {
		KYTY_PROFILER_BLOCK("BufferCache::CopyQueueReadbackFlush");
		if (buffer.last_gpu_write_tick >= m_scheduler.CurrentTick()) {
			// The writer is in the open command buffer: submit it so the copy can wait for its
			// tick.
			m_scheduler.Flush();
		}
		// With KYTY_RECORD_THREAD a submitted tick can still be queued on the recording thread.
		// The copy must not wait for a value whose signal the driver has not seen: run 34 hung
		// with the readback waiting on it, the present (holding queue_mutex) stalled behind that
		// wait, and the recording thread blocked on queue_mutex with the writer's submission.
		(void)m_scheduler.DrainRecording();
	}

	KYTY_PROFILER_BLOCK("BufferCache::CopyQueueReadback");
	auto& state  = m_copy_queue_readback;
	auto  device = m_graphics.device;
	if (state.pool == nullptr) {
		vk::CommandPoolCreateInfo pool_info {};
		pool_info.flags            = vk::CommandPoolCreateFlagBits::eResetCommandBuffer;
		pool_info.queueFamilyIndex = m_graphics.readback_queue_family;
		RequireVulkanSuccess(device.createCommandPool(&pool_info, nullptr, &state.pool),
		                     "create readback command pool");
		vk::CommandBufferAllocateInfo allocate {};
		allocate.commandPool        = state.pool;
		allocate.level              = vk::CommandBufferLevel::ePrimary;
		allocate.commandBufferCount = 1;
		RequireVulkanSuccess(device.allocateCommandBuffers(&allocate, &state.command),
		                     "allocate readback command buffer");
		vk::FenceCreateInfo fence_info {};
		RequireVulkanSuccess(device.createFence(&fence_info, nullptr, &state.fence),
		                     "create readback fence");
		state.staging =
		    std::make_unique<Buffer>(m_graphics, m_scheduler, MemoryUsage::Download, 0,
		                             vk::BufferUsageFlagBits::eTransferDst, StagingSize);
	}

	vk::CommandBufferBeginInfo begin {};
	begin.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;
	RequireVulkanSuccess(state.command.begin(&begin), "begin readback command buffer");
	state.command.copyBuffer(buffer.Handle(), state.staging->Handle(),
	                         static_cast<uint32_t>(copies.size()), copies.data());
	vk::BufferMemoryBarrier to_host {};
	to_host.srcAccessMask       = vk::AccessFlagBits::eTransferWrite;
	to_host.dstAccessMask       = vk::AccessFlagBits::eHostRead;
	to_host.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	to_host.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	to_host.buffer              = state.staging->Handle();
	to_host.offset              = 0;
	to_host.size                = total_size;
	state.command.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
	                              vk::PipelineStageFlagBits::eHost, {}, 0, nullptr, 1, &to_host, 0,
	                              nullptr);
	RequireVulkanSuccess(state.command.end(), "end readback command buffer");

	const uint64_t                  writer_tick = buffer.last_gpu_write_tick;
	const vk::Semaphore             master      = m_scheduler.GetMasterSemaphore().Handle();
	const vk::PipelineStageFlags    wait_stage  = vk::PipelineStageFlagBits::eTransfer;
	vk::TimelineSemaphoreSubmitInfo timeline {};
	timeline.waitSemaphoreValueCount = 1;
	timeline.pWaitSemaphoreValues    = &writer_tick;
	vk::SubmitInfo submit {};
	submit.pNext              = &timeline;
	submit.waitSemaphoreCount = 1;
	submit.pWaitSemaphores    = &master;
	submit.pWaitDstStageMask  = &wait_stage;
	submit.commandBufferCount = 1;
	submit.pCommandBuffers    = &state.command;
	RequireVulkanSuccess(m_graphics.readback_queue.submit(1, &submit, state.fence),
	                     "submit readback");
	RequireVulkanSuccess(device.waitForFences(1, &state.fence, VK_TRUE, UINT64_MAX),
	                     "wait for readback");
	RequireVulkanSuccess(device.resetFences(1, &state.fence), "reset readback fence");

	state.staging->Invalidate(0, total_size);
	const auto* mapped = state.staging->Mapped().data();
	for (const auto& copy: copies) {
		Libs::LibKernel::Memory::WriteBacking(buffer_address + copy.srcOffset,
		                                      mapped + copy.dstOffset, copy.size);
	}
	// Guest memory holds the bytes before they stop counting as GPU-written, so a concurrent
	// clean read never sees them stale.
	std::unique_lock lock(m_dirty_ranges_mutex);
	for (const auto& copy: copies) {
		m_gpu_modified_ranges.Subtract(buffer_address + copy.srcOffset, copy.size);
	}
	return true;
}

int BufferCache::StartGuestCopy(Buffer& buffer, uint64_t window_begin, uint64_t window_end) {
	Common::WaitStats::CpPart part(Common::WaitStats::CpCopyStart);
	constexpr uint64_t StagingSize = 4ull * 1024 * 1024;
	if (m_graphics.readback_queue == nullptr || !m_scheduler.IsFree(buffer.last_gpu_write_tick)) {
		return -1; // no second queue, or the writer is still running: the main queue orders it
	}
	// gc4: the CP spent ~1.2 s per 5 s completing (waiting for) copies of the same window that
	// another game thread had started: join that copy instead; this thread waits for its fence.
	for (size_t i = 0; i < GuestCopySlots; i++) {
		auto& slot = m_guest_copy_slots[i];
		if (slot.busy && slot.window_begin == window_begin && slot.window_end == window_end &&
		    slot.buffer_handle == buffer.Handle() &&
		    slot.writer_tick == buffer.last_gpu_write_tick) {
			slot.waiters.fetch_add(1, std::memory_order_acq_rel);
			return static_cast<int>(i);
		}
	}
	CompleteGuestCopies(window_begin, window_end);
	std::vector<vk::BufferCopy> copies;
	uint64_t                    total_size     = 0;
	const auto                  buffer_address = buffer.CpuAddress();
	{
		std::shared_lock lock(m_dirty_ranges_mutex);
		if (m_downloading_ranges.Intersects(window_begin, window_end - window_begin)) {
			return -1;
		}
		m_memory_tracker.ForEachDownloadRange<false>(
		    window_begin, window_end - window_begin,
		    [&](uint64_t address, uint64_t bytes) noexcept {
			    m_gpu_modified_ranges.ForEachInRange(
			        address, bytes, [&](uint64_t start, uint64_t end) {
				        copies.emplace_back(start - buffer_address, total_size, end - start);
				        total_size += Common::AlignUp(end - start, 64);
			        });
		    });
	}
	if (copies.empty() || total_size > StagingSize) {
		return -1;
	}
	auto device = m_graphics.device;
	if (m_guest_copy_pool == nullptr) {
		vk::CommandPoolCreateInfo pool_info {};
		pool_info.flags            = vk::CommandPoolCreateFlagBits::eResetCommandBuffer;
		pool_info.queueFamilyIndex = m_graphics.readback_queue_family;
		RequireVulkanSuccess(device.createCommandPool(&pool_info, nullptr, &m_guest_copy_pool),
		                     "create guest copy pool");
	}
	size_t index = GuestCopySlots;
	for (size_t i = 0; i < GuestCopySlots; i++) {
		if (!m_guest_copy_slots[i].busy &&
		    m_guest_copy_slots[i].waiters.load(std::memory_order_acquire) == 0) {
			index = i;
			break;
		}
	}
	if (index == GuestCopySlots) {
		return -1;
	}
	auto& slot = m_guest_copy_slots[index];
	if (slot.command == nullptr) {
		vk::CommandBufferAllocateInfo allocate {};
		allocate.commandPool        = m_guest_copy_pool;
		allocate.level              = vk::CommandBufferLevel::ePrimary;
		allocate.commandBufferCount = 1;
		RequireVulkanSuccess(device.allocateCommandBuffers(&allocate, &slot.command),
		                     "allocate guest copy command buffer");
		vk::FenceCreateInfo fence_info {};
		RequireVulkanSuccess(device.createFence(&fence_info, nullptr, &slot.fence),
		                     "create guest copy fence");
		slot.staging = std::make_unique<Buffer>(m_graphics, m_scheduler, MemoryUsage::Download, 0,
		                                        vk::BufferUsageFlagBits::eTransferDst, StagingSize);
	} else {
		RequireVulkanSuccess(device.resetFences(1, &slot.fence), "reset guest copy fence");
	}
	vk::CommandBufferBeginInfo begin {};
	begin.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;
	RequireVulkanSuccess(slot.command.begin(&begin), "begin guest copy");
	slot.command.copyBuffer(buffer.Handle(), slot.staging->Handle(),
	                        static_cast<uint32_t>(copies.size()), copies.data());
	vk::BufferMemoryBarrier to_host {};
	to_host.srcAccessMask       = vk::AccessFlagBits::eTransferWrite;
	to_host.dstAccessMask       = vk::AccessFlagBits::eHostRead;
	to_host.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	to_host.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	to_host.buffer              = slot.staging->Handle();
	to_host.offset              = 0;
	to_host.size                = total_size;
	slot.command.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
	                             vk::PipelineStageFlagBits::eHost, {}, 0, nullptr, 1, &to_host, 0,
	                             nullptr);
	RequireVulkanSuccess(slot.command.end(), "end guest copy");
	vk::SubmitInfo submit {};
	submit.commandBufferCount = 1;
	submit.pCommandBuffers    = &slot.command;
	{
		Common::WaitStats::CpPart submit_part(Common::WaitStats::CpCopySubmit);
		RequireVulkanSuccess(m_graphics.readback_queue.submit(1, &submit, slot.fence),
		                     "submit guest copy");
	}
	slot.copies         = std::move(copies);
	slot.buffer_address = buffer_address;
	slot.buffer_handle  = buffer.Handle();
	slot.writer_tick    = buffer.last_gpu_write_tick;
	slot.window_begin   = window_begin;
	slot.window_end     = window_end;
	slot.generation++;
	slot.publish_state.store(0, std::memory_order_release);
	slot.busy = true;
	slot.waiters.store(1, std::memory_order_release); // the faulting game thread
	return static_cast<int>(index);
}

void BufferCache::CompleteGuestCopy(size_t index) {
	auto& slot = m_guest_copy_slots[index];
	if (!slot.busy) {
		return;
	}
	Common::WaitStats::CpPart part(Common::WaitStats::CpCopyDone);
	auto device = m_graphics.device;
	RequireVulkanSuccess(device.waitForFences(1, &slot.fence, VK_TRUE, UINT64_MAX),
	                     "wait for guest copy");
	slot.busy = false;
	// Published whatever happens next: bytes the GPU re-marked meanwhile stay GPU-owned (their
	// pages protected), so an older value in guest memory is never read for them.
	PublishGuestCopy(slot);
	// The buffer may have been deleted or re-marked GPU-written by a later binding meanwhile: then
	// its bytes stay GPU-owned (a later fault downloads them again).
	const auto* owner = m_page_table.Find(slot.window_begin >> PageTable::kPageBits);
	if (owner == nullptr || !*owner || IsBufferInvalid(*owner) ||
	    m_slot_buffers[*owner].Handle() != slot.buffer_handle ||
	    m_slot_buffers[*owner].last_gpu_write_tick != slot.writer_tick) {
		return;
	}
	{
		std::unique_lock lock(m_dirty_ranges_mutex);
		for (const auto& copy: slot.copies) {
			m_gpu_modified_ranges.Subtract(slot.buffer_address + copy.srcOffset, copy.size);
		}
	}
	if (!HasGpuDirtyBytes(slot.window_begin, slot.window_end - slot.window_begin)) {
		m_memory_tracker.UnmarkRegionAsGpuModified(slot.window_begin,
		                                           slot.window_end - slot.window_begin);
	}
}

void BufferCache::CompleteGuestCopies(uint64_t begin, uint64_t end) {
	for (size_t i = 0; i < GuestCopySlots; i++) {
		const auto& slot = m_guest_copy_slots[i];
		if (slot.busy && slot.window_begin < end && begin < slot.window_end) {
			CompleteGuestCopy(i);
		}
	}
}

void BufferCache::PublishGuestCopy(GuestCopySlot& slot) {
	int expected = 0;
	if (!slot.publish_state.compare_exchange_strong(expected, 1, std::memory_order_acq_rel)) {
		while (slot.publish_state.load(std::memory_order_acquire) != 2) {
			__builtin_ia32_pause();
		}
		return;
	}
	slot.staging->Invalidate(0, slot.staging->Size());
	const auto* mapped = slot.staging->Mapped().data();
	for (const auto& copy: slot.copies) {
		Libs::LibKernel::Memory::WriteBacking(slot.buffer_address + copy.srcOffset,
		                                      mapped + copy.dstOffset, copy.size);
	}
	slot.publish_state.store(2, std::memory_order_release);
}

void BufferCache::WaitGuestCopy(int index) {
	auto& slot   = m_guest_copy_slots[static_cast<size_t>(index)];
	auto  device = m_graphics.device;
	// The slot cannot be reused (its fence reset) while this thread is counted as a waiter.
	RequireVulkanSuccess(device.waitForFences(1, &slot.fence, VK_TRUE, UINT64_MAX),
	                     "wait for guest copy (game thread)");
	// The game thread copies the bytes into guest memory itself (gc2: on the CP it cost ~1.3 s
	// per 5 s and starved the CP); the CP's FinishGuestCopy only updates the dirty state.
	PublishGuestCopy(slot);
}

void BufferCache::FinishGuestCopy(int index, uint64_t generation, uint64_t vaddr, uint64_t size,
                                  bool is_write) {
	Common::WaitStats::CpPart part(Common::WaitStats::CpCopyFinish);
	auto& slot = m_guest_copy_slots[static_cast<size_t>(index)];
	if (slot.generation == generation) {
		CompleteGuestCopy(static_cast<size_t>(index));
	}
	slot.waiters.fetch_sub(1, std::memory_order_acq_rel);
	if (!IsRegionRegistered(vaddr, size)) {
		return;
	}
	const auto page_begin = Common::AlignDown(vaddr, TRACKER_PAGE_SIZE);
	const auto page_end   = Common::AlignUp(vaddr + size, TRACKER_PAGE_SIZE);
	if (m_memory_tracker.IsRegionGpuModified(page_begin, page_end - page_begin)) {
		// Re-marked meanwhile (or not published): resolve this fault the synchronous way.
		ReadbackCaller caller(7);
		ReadMemoryOnGpu(vaddr, size, is_write);
		return;
	}
	if (is_write) {
		m_memory_tracker.MarkRegionAsCpuModified(vaddr, size);
	}
}

// GPU thread. Returns the tick the guest must wait for, or 0 when the fault is already resolved.
uint64_t BufferCache::BeginWriteReadback(uint64_t vaddr, uint64_t size, bool is_write,
                                         uint64_t& window_begin, uint64_t& window_end) {
	KYTY_PROFILER_FUNCTION();
	if (!is_write && !IsRegionRegistered(vaddr, size)) {
		// A read of an unregistered region: the synchronous path finds its buffer.
		const auto     sync_start = std::chrono::steady_clock::now();
		ReadbackCaller caller(6);
		ReadMemoryOnGpu(vaddr, size, false);
		if (ReadbackStats::On()) {
			NoteReadbackPhase(4, sync_start);
		}
		return 0;
	}
	ReadbackStats::Scope stats(vaddr, is_write);
	const auto           phase_begin = std::chrono::steady_clock::now();
	if (!IsRegionRegistered(vaddr, size)) {
		return 0;
	}
	const auto page_begin = Common::AlignDown(vaddr, TRACKER_PAGE_SIZE);
	const auto page_end   = Common::AlignUp(vaddr + size, TRACKER_PAGE_SIZE);
	// Another guest's readback of this page is still landing: wait for the same download.
	for (const auto& pending: m_pending_write_readbacks) {
		if (pending.begin < page_end && page_begin < pending.end) {
			window_begin = pending.begin;
			window_end   = pending.end;
			return pending.tick;
		}
	}
	auto& buffer = m_slot_buffers[FindBuffer(vaddr, size)];
	stats.SetBuffer(buffer.CpuAddress(), buffer.Size());

	// As in ReadMemoryOnGpu: a GPU-owned page without bytes to download only needs unprotecting.
	if (m_memory_tracker.IsRegionGpuModified(page_begin, page_end - page_begin) &&
	    !HasGpuDirtyBytes(page_begin, page_end - page_begin)) {
		m_memory_tracker.UnmarkRegionAsGpuModified(page_begin, page_end - page_begin);
		if (is_write) {
			m_memory_tracker.MarkRegionAsCpuModified(vaddr, size);
		}
		stats.SetOutcome(ReadbackStats::Unmarked);
		return 0;
	}

	// KYTY_DIRECT_GUEST_READBACK=1 (live, default 0; 2026-10-10): rd1 (heavy combat, WAIT_STATS
	// inner timers): game threads spent ~22 s per 5 s (summed) in read faults, 93% of it waiting
	// for the readback's GPU tick (~3.6 ms each, ~5.8k reads), although for ~97% of them the
	// writer had already finished: the download was queued behind all pending GPU work. When the
	// buffer lives in host-visible device memory (KYTY_MAPPED_DEVICE_BUFFERS) and every GPU write
	// to it has executed, the bytes are final there: copy the faulting pages' GPU-written bytes to
	// guest memory directly (TryDirectReadback mode 1), no GPU work and no wait.
	static auto& direct_guest = Common::LiveSwitches::Get("KYTY_DIRECT_GUEST_READBACK", 0);
	if (direct_guest.load(std::memory_order_relaxed) != 0) {
		static uint64_t direct_ok = 0, direct_fallback = 0;
		static auto     direct_at = std::chrono::steady_clock::now();
		const bool      done      = TryDirectReadback(buffer, vaddr, size, is_write, 1);
		(done ? direct_ok : direct_fallback)++;
		if (const auto now = std::chrono::steady_clock::now();
		    now - direct_at >= std::chrono::seconds(5)) {
			::printf("Direct guest readback (5 s): %" PRIu64 " direct, %" PRIu64 " downloads\n",
			         direct_ok, direct_fallback);
			std::fflush(stdout);
			direct_ok = direct_fallback = 0;
			direct_at                   = now;
		}
		if (done) {
			stats.SetOutcome(ReadbackStats::DirectRead);
			return 0;
		}
	}

	const uint64_t WindowSize   = ReadbackWindowSize();
	const auto     buffer_begin = buffer.CpuAddress();
	const auto     buffer_end   = buffer_begin + buffer.Size();
	window_begin                = std::max(Common::AlignDown(vaddr, WindowSize), buffer_begin);
	window_end = std::min(std::max(window_begin + WindowSize, vaddr + size), buffer_end);
	if (OverlapsPendingWriteReadback(window_begin, window_end)) {
		// The page itself is not pending (checked above), but the window shares bytes with a
		// landing readback (neighbouring pages read by other guest threads): download only the
		// faulting pages. Before, reads went synchronous here (run 42: ~110 ms/s left on the GPU
		// thread for Wolverine's busiest readback buffer).
		window_begin = std::max(page_begin, buffer_begin);
		window_end   = std::min(page_end, buffer_end);
	}
	const bool phases     = ReadbackStats::On();
	auto       phase_time = std::chrono::steady_clock::now();
	if (phases) {
		NoteReadbackPhase(0, phase_begin);
	}
	// KYTY_GUEST_COPY_QUEUE=1 (live, default 1 since gc5/gc6; needs the compute readback queue,
	// env KYTY_READBACK_COMPUTE_QUEUE not 0; 2026-10-10): rd1 showed the game threads' read faults waiting ~3.6 ms each for the window
	// download queued behind all pending GPU work, although the writer had finished for ~97% of
	// them. When it has, copy the window's GPU-written bytes on the compute-family readback queue
	// (TryCopyQueueReadback: behind nothing, the CP waits for that copy only) and resolve the fault
	// now; the guest no longer waits for the graphics queue to drain.
	// gc1 (the CP waiting for each copy itself): r-gpu -96% but the hop to the CP grew 0.9 -> 5.7 s
	// per 5 s (~2 ms a copy on the CP). So the CP only records and submits the copy into one of a
	// few slots, and the game thread waits for the slot's fence (ReadMemory -> WaitGuestCopy ->
	// FinishGuestCopy). A window with a copy in flight is completed first by anyone touching it.
	// Heavy combat (combatab, live A/B): gc5 fps 8.34 -> 10.70 (steady scene ~7.2 -> ~11.1), gc6
	// 8.70 -> 13.60; idle neutral (gi1 24.2 / 24.2 / 24.8).
	static auto& guest_copy = Common::LiveSwitches::Get("KYTY_GUEST_COPY_QUEUE", 1);
	if (guest_copy.load(std::memory_order_relaxed) != 0 && s_guest_copy_out != nullptr) {
		static uint64_t copied = 0, queued = 0;
		static auto     copy_at = std::chrono::steady_clock::now();
		const int       slot    = StartGuestCopy(buffer, window_begin, window_end);
		(slot >= 0 ? copied : queued)++;
		if (const auto now = std::chrono::steady_clock::now();
		    now - copy_at >= std::chrono::seconds(5)) {
			::printf("Guest copy-queue readback (5 s): %" PRIu64 " copied, %" PRIu64
			         " main-queue downloads\n",
			         copied, queued);
			std::fflush(stdout);
			copied = queued = 0;
			copy_at         = now;
		}
		if (slot >= 0) {
			stats.SetOutcome(ReadbackStats::Downloaded);
			s_guest_copy_out->slot       = slot;
			s_guest_copy_out->generation = m_guest_copy_slots[slot].generation;
			return 0;
		}
	}
	if (!DownloadBufferMemory(buffer, window_begin, window_end - window_begin)) {
		stats.SetOutcome(ReadbackStats::NothingToDownload);
		if (is_write) {
			m_memory_tracker.MarkRegionAsCpuModified(vaddr, size);
		}
		return 0;
	}
	stats.SetOutcome(ReadbackStats::Downloaded);
	if (phases) {
		NoteReadbackPhase(1, phase_time);
		phase_time = std::chrono::steady_clock::now();
	}
	const auto tick = m_scheduler.CurrentTick();
	m_scheduler.Flush();
	if (phases) {
		NoteReadbackPhase(2, phase_time);
	}
	m_pending_write_readbacks.push_back({window_begin, window_end, tick});
	return tick;
}

// GPU thread: KYTY_RACE_READS (see RaceReadsEnabled). True when the guest read is served from
// guest memory without waiting; a download of the page's window is then on its way (started now
// or by an earlier race read) and published when a later fault finds it done.
bool BufferCache::TryRaceRead(uint64_t vaddr, uint64_t size) {
	struct Counters {
		uint64_t served = 0, started = 0, completed = 0, exact = 0;
		std::chrono::steady_clock::time_point at = std::chrono::steady_clock::now();
	};
	static Counters counters;
	const auto      report = [&] {
        const auto now = std::chrono::steady_clock::now();
        if (now - counters.at >= std::chrono::seconds(5)) {
            ::printf("Race reads (5 s): served %" PRIu64 " (downloads started %" PRIu64
			              "), finished downloads published %" PRIu64 ", exact path %" PRIu64 "\n",
			              counters.served, counters.started, counters.completed, counters.exact);
            std::fflush(stdout);
            counters    = {};
            counters.at = now;
        }
	};
	const auto page_begin = Common::AlignDown(vaddr, TRACKER_PAGE_SIZE);
	const auto page_end   = Common::AlignUp(vaddr + size, TRACKER_PAGE_SIZE);
	// Publish the race downloads of this page that have landed.
	for (size_t i = 0; i < m_pending_write_readbacks.size();) {
		const auto& pending = m_pending_write_readbacks[i];
		if (pending.begin < page_end && page_begin < pending.end &&
		    m_scheduler.IsFree(pending.tick)) {
			CompletePendingWriteReadback(i);
			counters.completed++;
		} else {
			i++;
		}
	}
	if (!IsRegionRegistered(vaddr, size) ||
	    !m_memory_tracker.IsRegionGpuModified(page_begin, page_end - page_begin) ||
	    !HasGpuDirtyBytes(page_begin, page_end - page_begin)) {
		counters.exact++;
		report();
		return false;
	}
	const auto* owner = m_page_table.Find(vaddr >> PageTable::kPageBits);
	if (owner == nullptr || !*owner || IsBufferInvalid(*owner) ||
	    m_scheduler.IsFree(m_slot_buffers[*owner].last_gpu_write_tick)) {
		counters.exact++;
		report();
		return false; // the writer has finished: the exact readback costs no GPU wait
	}
	if (!OverlapsPendingWriteReadback(page_begin, page_end)) {
		uint64_t window_begin = 0;
		uint64_t window_end   = 0;
		if (BeginWriteReadback(vaddr, size, false, window_begin, window_end) != 0) {
			counters.started++;
		}
	}
	counters.served++;
	report();
	return true;
}

// GPU thread, after the guest waited for the readback's tick and its publication.
bool BufferCache::FinishWriteReadback(uint64_t vaddr, uint64_t size, bool is_write,
                                      uint64_t window_begin, uint64_t window_end, uint64_t tick,
                                      bool snapshot) {
	KYTY_PROFILER_FUNCTION();
	struct FinishPhase {
		std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
		~FinishPhase() {
			if (ReadbackStats::On()) {
				NoteReadbackPhase(3, start);
			}
		}
	} finish_phase;
	for (size_t i = 0; i < m_pending_write_readbacks.size(); i++) {
		const auto& pending = m_pending_write_readbacks[i];
		if (pending.begin == window_begin && pending.end == window_end && pending.tick == tick) {
			CompletePendingWriteReadback(i);
			break;
		}
	}
	if (!IsRegionRegistered(vaddr, size)) {
		return true;
	}
	const auto page_begin = Common::AlignDown(vaddr, TRACKER_PAGE_SIZE);
	const auto page_end   = Common::AlignUp(vaddr + size, TRACKER_PAGE_SIZE);
	if (m_memory_tracker.IsRegionGpuModified(page_begin, page_end - page_begin)) {
		if (snapshot) {
			// The GPU wrote the page again after the download was recorded: the backing store
			// holds the bytes as of the guest's read, which the caller serves; the page stays
			// GPU-owned and protected.
			m_snapshot_reads++;
			return false;
		}
		// The GPU wrote the window again while the download was landing, or another readback
		// still covers the page: resolve this fault the synchronous way.
		ReadbackCaller caller(7);
		ReadMemoryOnGpu(vaddr, size, is_write);
		return true;
	}
	if (is_write) {
		m_memory_tracker.MarkRegionAsCpuModified(vaddr, size);
	}
	return true;
}

// GPU thread: waits for a pending readback's publication and, unless the GPU wrote its window
// again, lifts the window's GPU ownership as the synchronous path does after its download.
void BufferCache::CompletePendingWriteReadback(size_t index) {
	const auto pending = m_pending_write_readbacks[index];
	m_pending_write_readbacks.erase(m_pending_write_readbacks.begin() +
	                                static_cast<std::ptrdiff_t>(index));
	m_scheduler.Wait(pending.tick);
	m_scheduler.WaitPriorityOperations(pending.tick);
	if (!HasGpuDirtyBytes(pending.begin, pending.end - pending.begin)) {
		m_memory_tracker.UnmarkRegionAsGpuModified(pending.begin, pending.end - pending.begin);
	}
}

void BufferCache::CompletePendingWriteReadbacks(uint64_t begin, uint64_t end) {
	for (size_t i = 0; i < m_pending_write_readbacks.size();) {
		const auto& pending = m_pending_write_readbacks[i];
		if (pending.begin < end && begin < pending.end) {
			CompletePendingWriteReadback(i);
		} else {
			i++;
		}
	}
}

bool BufferCache::OverlapsPendingWriteReadback(uint64_t begin, uint64_t end) const {
	return std::ranges::any_of(m_pending_write_readbacks, [&](const PendingWriteReadback& pending) {
		return pending.begin < end && begin < pending.end;
	});
}

BufferId BufferCache::FindBuffer(uint64_t vaddr, uint64_t size) {
	if (vaddr == 0) {
		return NULL_BUFFER_ID;
	}
	if (!GuestRange {vaddr, size}.Valid()) {
		EXIT("BufferCache: invalid buffer discovery request\n");
	}
	const auto* owner = m_page_table.Find(vaddr >> PageTable::kPageBits);
	if (owner != nullptr && *owner) {
		auto& buffer = m_slot_buffers[*owner];
		if (buffer.IsInBounds(vaddr, size)) {
			return *owner;
		}
	}
	return CreateBuffer(vaddr, size);
}

BufferCache::OverlapResult BufferCache::ResolveOverlaps(uint64_t vaddr, uint64_t size) {
	static constexpr int      StreamLeapThreshold = 16;
	static constexpr uint64_t StreamLeapSize      = CACHING_PAGESIZE * 128;

	auto       begin      = vaddr;
	auto       end        = vaddr + size;
	const auto find_first = [&](uint64_t address) {
		auto first = m_buffers.lower_bound(address);
		if (first != m_buffers.begin()) {
			const auto  previous = std::prev(first);
			const auto& buffer   = m_slot_buffers[previous->second];
			if (buffer.CpuAddress() + buffer.Size() > address) {
				first = previous;
			}
		}
		return first;
	};
	auto first           = find_first(begin);
	auto last            = first;
	int  stream_score    = 0;
	bool has_stream_leap = false;
	for (; last != m_buffers.end() && last->first < end; ++last) {
		const auto& buffer        = m_slot_buffers[last->second];
		const auto  buffer_begin  = buffer.CpuAddress();
		const auto  buffer_end    = buffer_begin + buffer.Size();
		const bool  expands_left  = buffer_begin < begin;
		const bool  expands_right = buffer_end > end;
		begin                     = std::min(begin, buffer_begin);
		end                       = std::max(end, buffer_end);
		if (!has_stream_leap && (stream_score += buffer.StreamScore()) > StreamLeapThreshold) {
			has_stream_leap = true;
			// Reserve space in the incoming stream's direction of growth.
			// The old buffer extending left of the request predicts growth to the right, and vice versa.
			if (expands_left) {
				end += std::min(StreamLeapSize, (vaddr < LOWER_ADDRESS_SIZE ? LOWER_ADDRESS_SIZE
				                                       : LibKernel::Memory::kExtendedMemoryBase +
				                                             LibKernel::Memory::kExtendedMemorySize) - end);
			}
			if (expands_right) {
				const auto minimum = vaddr < LOWER_ADDRESS_SIZE
				                         ? CACHING_PAGESIZE * 2
				                         : LibKernel::Memory::kExtendedMemoryBase;
				if (begin > minimum) {
					begin -= std::min(StreamLeapSize, begin - minimum);
				}
				first = find_first(begin);
				begin = std::min(begin, first->first);
			}
		}
	}
	return {first, last, begin, end, has_stream_leap};
}

void BufferCache::JoinOverlap(BufferId new_id, BufferId overlap_id, bool accumulate_stream_score) {
	auto& new_buffer = m_slot_buffers[new_id];
	auto& overlap    = m_slot_buffers[overlap_id];
	if (accumulate_stream_score) {
		new_buffer.IncreaseStreamScore(overlap.StreamScore() + 1);
	}
	new_buffer.CopyFrom(m_scheduler.Current(), overlap, 0,
	                    overlap.CpuAddress() - new_buffer.CpuAddress(), overlap.Size());
	new_buffer.last_gpu_write_tick = m_scheduler.CurrentTick();
	new_buffer.last_gpu_copy_tick  = m_scheduler.CurrentTick();
	DeleteBuffer(overlap_id);
}

BufferId BufferCache::CreateBuffer(uint64_t vaddr, uint64_t size) {
	EXIT_IF(m_scheduler.Current().IsInvalid());
	const auto end = Common::AlignUp(vaddr + size, CACHING_PAGESIZE);
	if (vaddr < CACHING_PAGESIZE) {
		// Guest page 0 is never mapped; a request here is a garbage or null descriptor that
		// slipped past the null checks. The memory tracker cannot hold address 0, so name the
		// caller now rather than in the garbage collector later.
		static std::atomic<uint32_t> reported = 0;
		if (reported.fetch_add(1) < 8) {
			LOGF("BufferCache: buffer requested in guest page 0: vaddr=0x%016" PRIx64
			     " size=0x%016" PRIx64 "\n%s",
			     vaddr, size, Common::HostBacktrace().c_str());
		}
	}
	vaddr = Common::AlignDown(vaddr, CACHING_PAGESIZE);
	size               = end - vaddr;
	if (m_import_on) {
		if (auto* zone = FindImportZone(vaddr, size); zone != nullptr) {
			if (vaddr >= zone->begin && end <= zone->begin + zone->size) {
				if (CreateZoneBuffer(*zone)) {
					return zone->id;
				}
			} else if (!zone->failed) {
				DropZone(*zone, true, "a binding crosses the zone");
			}
		}
	}
	const auto overlap = ResolveOverlaps(vaddr, size);

	m_buffers_created++;
	const auto id = m_slot_buffers.insert(
	    m_graphics, m_scheduler, MemoryUsage::DeviceLocal, overlap.begin,
	    AllFlags | vk::BufferUsageFlagBits::eShaderDeviceAddress, overlap.end - overlap.begin);
	const auto& buffer = m_slot_buffers[id];
	SetVulkanObjectNameF(m_graphics.device, buffer.Handle(),
	                     "Kyty.GameBuffer[guest=0x{:016x} size=0x{:x}]", overlap.begin,
	                     overlap.end - overlap.begin);
	for (auto it = overlap.first; it != overlap.last;) {
		const auto old_id = (it++)->second;
		JoinOverlap(id, old_id, !overlap.has_stream_leap);
	}
	Register(id);
	return id;
}

static void ReadGuestForUpload(uint8_t* destination, uint64_t address, uint64_t size);

// KYTY_HOST_IMPORT=1 (live, default 1; 2026-10-10): heavy combat is bound by the game's job
// threads writing CPU-only data (per-object constants in 0x10a0000000 and 0x11e0000000..): every
// write after a sync faults (write protection), the command processor re-uploads the page at the
// next sync (~1 GB per 5 s) and protects it again (hb1: ~150k write faults per 5 s, ~75% there).
// The GPU never writes those ranges and no binding crosses them, so each zone
// (KYTY_HOST_IMPORT_RANGES=<hex addr>+<hex size>[,...]) gets ONE buffer whose memory is the guest
// pages themselves: a udmabuf of the direct-memory memfd slice, imported as a dma-buf. Its pages
// stay writable (no faults), nothing is uploaded, and the GPU reads the game's bytes when it runs,
// as on the console (labels are written after the host GPU finishes). A zone falls back to normal
// buffers when a binding crosses it, a buffer in it holds GPU-written bytes, or the import fails.
// GPU writes into a zone go straight to guest memory (not marked GPU-written: no readback).
// VK_EXT_external_memory_host cannot do this: the kernel accepts only anonymous memory there.
BufferCache::ImportZone* BufferCache::FindImportZone(uint64_t vaddr, uint64_t size) {
	for (auto& zone: m_import_zones) {
		if (vaddr < zone.begin + zone.size && zone.begin < vaddr + size) {
			return &zone;
		}
	}
	return nullptr;
}

bool BufferCache::InActiveImportZone(uint64_t vaddr, uint64_t size) {
	if (!m_import_on) {
		return false;
	}
	const auto* zone = FindImportZone(vaddr, size);
	return zone != nullptr && zone->id && !IsBufferInvalid(zone->id);
}

void BufferCache::DropZone(ImportZone& zone, bool failed, const char* reason) {
	if (zone.id && !IsBufferInvalid(zone.id)) {
		DeleteBuffer(zone.id);
	}
	zone.id     = {};
	zone.failed = zone.failed || failed;
	::printf("Host import: zone 0x%016" PRIx64 "+0x%" PRIx64 " dropped (%s)%s\n", zone.begin,
	         zone.size, reason, failed ? ", normal buffers from now on" : "");
	std::fflush(stdout);
}

bool BufferCache::CreateZoneBuffer(ImportZone& zone) {
	if (zone.failed) {
		return false;
	}
	if (zone.id && !IsBufferInvalid(zone.id)) {
		return true;
	}
	zone.id   = {};
	int                                        memfd = -1;
	std::vector<std::pair<uint64_t, uint64_t>> slices;
	if (!Libs::LibKernel::Memory::FindBackingSlices(zone.begin, zone.size, &memfd, &slices)) {
		static uint32_t reported = 0;
		if (reported++ < 16) {
			::printf("Host import: zone 0x%016" PRIx64 "+0x%" PRIx64 " not wholly mapped yet\n",
			         zone.begin, zone.size);
			std::fflush(stdout);
		}
		return false; // try again on the next request
	}
	const auto fail = [&](const char* reason) {
		zone.failed = true;
		::printf("Host import: zone 0x%016" PRIx64 "+0x%" PRIx64 " not imported (%s)\n",
		         zone.begin, zone.size, reason);
		std::fflush(stdout);
		return false;
	};
	if (m_memory_tracker.IsRegionGpuModified(zone.begin, zone.size)) {
		return fail("GPU-written bytes");
	}
	// The buffers already covering the zone must lie inside it; they are replaced.
	std::vector<BufferId> inside;
	auto                  it = m_buffers.lower_bound(zone.begin);
	if (it != m_buffers.begin()) {
		--it;
	}
	for (; it != m_buffers.end() && it->first < zone.begin + zone.size; ++it) {
		const auto& buffer = m_slot_buffers[it->second];
		if (buffer.CpuAddress() + buffer.Size() <= zone.begin) {
			continue;
		}
		if (buffer.CpuAddress() < zone.begin ||
		    buffer.CpuAddress() + buffer.Size() > zone.begin + zone.size) {
			return fail("a cached buffer crosses the zone");
		}
		inside.push_back(it->second);
	}
	if (m_udmabuf_fd < 0) {
		m_udmabuf_fd = open("/dev/udmabuf", O_RDWR | O_CLOEXEC);
		if (m_udmabuf_fd < 0) {
			return fail("cannot open /dev/udmabuf");
		}
	}
	// One list entry per backing slice (the guest may map a zone in several pieces).
	if (slices.size() > 1024) {
		return fail("more than 1024 backing slices");
	}
	std::vector<uint8_t> request(sizeof(udmabuf_create_list) +
	                             slices.size() * sizeof(udmabuf_create_item));
	auto* list  = reinterpret_cast<udmabuf_create_list*>(request.data());
	list->flags = UDMABUF_FLAGS_CLOEXEC;
	list->count = static_cast<uint32_t>(slices.size());
	for (size_t index = 0; index < slices.size(); index++) {
		if (slices[index].first % 4096 != 0 || slices[index].second % 4096 != 0) {
			return fail("unaligned backing slice");
		}
		list->list[index].memfd  = static_cast<uint32_t>(memfd);
		list->list[index].offset = slices[index].first;
		list->list[index].size   = slices[index].second;
	}
	const int dma_buf = ioctl(m_udmabuf_fd, UDMABUF_CREATE_LIST, list);
	if (dma_buf < 0) {
		::printf("Host import: UDMABUF_CREATE_LIST errno %d\n", errno);
		return fail("UDMABUF_CREATE_LIST failed");
	}
	const auto id = m_slot_buffers.insert(m_graphics, m_scheduler, zone.begin,
	                                      AllFlags | vk::BufferUsageFlagBits::eShaderDeviceAddress,
	                                      zone.size, ImportedGuestMemory {dma_buf});
	if (!m_slot_buffers[id].Handle()) {
		m_slot_buffers.erase(id);
		return fail("Vulkan import failed");
	}
	for (const auto old: inside) {
		DeleteBuffer(old);
	}
	SetVulkanObjectNameF(m_graphics.device, m_slot_buffers[id].Handle(),
	                     "Kyty.HostImport[guest=0x{:016x} size=0x{:x}]", zone.begin, zone.size);
	m_buffers_created++;
	Register(id);
	zone.id = id;
	// Lift the write protection the replaced buffers left: nothing is uploaded from here on.
	m_memory_tracker.MarkRegionAsCpuModified(zone.begin, zone.size);
	::printf("Host import: zone 0x%016" PRIx64 "+0x%" PRIx64 " imported (%zu backing slices, first "
	         "0x%" PRIx64 ", replaced %zu buffers)\n",
	         zone.begin, zone.size, slices.size(), slices[0].first, inside.size());
	std::fflush(stdout);
	return true;
}

void BufferCache::PollHostImport() {
	// Default 1 since 2026-10-10 on top of KYTY_GUEST_COPY_QUEUE: combat ac1 write faults -72%, steady
	// scene 24.5 -> 26.2 fps; idle ai1 25.1 -> 27.4 -> 25.5, ai2 (on from boot) 27.4 / 25.6 off / 27.4.
	static auto& host_import = Common::LiveSwitches::Get("KYTY_HOST_IMPORT", 1);
	const bool   on = host_import.load(std::memory_order_relaxed) != 0 && m_graphics.dma_buf_import;
	if (on == m_import_on) {
		return;
	}
	m_import_on = on;
	if (m_import_zones.empty()) {
		// hi3: the 0x10a0000000 heap's buffer is 5.5 MiB (crosses a 4 MiB zone) -> 8 MiB there.
		const char* value = std::getenv("KYTY_HOST_IMPORT_RANGES");
		if (value == nullptr) {
			value = "10a0000000+800000,11e0000000+400000,11e0400000+400000,11e0800000+400000,"
			        "11e0c00000+400000";
		}
		while (value != nullptr && *value != '\0') {
			char*      end   = nullptr;
			const auto begin = std::strtoull(value, &end, 16);
			uint64_t   bytes = CACHING_PAGESIZE;
			if (end != nullptr && *end == '+') {
				bytes = std::strtoull(end + 1, &end, 16);
			}
			if (begin % CACHING_PAGESIZE == 0 && bytes % CACHING_PAGESIZE == 0 && bytes != 0 &&
			    GuestRange {begin, bytes}.Valid()) {
				m_import_zones.push_back({begin, bytes});
			}
			value = end != nullptr && *end == ',' ? end + 1 : nullptr;
		}
	}
	::printf("Host import: %s (%zu zones)\n", on ? "on" : "off", m_import_zones.size());
	std::fflush(stdout);
	for (auto& zone: m_import_zones) {
		if (on) {
			zone.failed = false;
			(void)CreateZoneBuffer(zone);
		} else if (zone.id) {
			DropZone(zone, false, "switch off");
		}
	}
}

void BufferCache::UnmapHostImport(uint64_t vaddr, uint64_t size) {
	for (auto& zone: m_import_zones) {
		if (zone.id && vaddr < zone.begin + zone.size && zone.begin < vaddr + size) {
			DropZone(zone, false, "unmapped");
		}
	}
}

// KYTY_LOCAL_HACK KYTY_UPLOAD_WORKER (live, default 1; uw1 +2.4% fps): the guest-to-device copies of direct uploads run on a
// worker thread (FIFO, so copies into the same bytes keep their order); the GPU thread waits for
// them before every submit (CommandScheduler::Submit) and before reading mapped device memory.
// As on the console, the GPU reads the memory when it runs, so a copy taken a little later is no
// staler than the hardware's view.
namespace {
class UploadWorker {
public:
	static UploadWorker& Get() {
		static UploadWorker worker;
		return worker;
	}
	// GPU thread. False when the queue is full (the caller copies itself).
	bool Push(uint8_t* destination, uint64_t address, uint64_t size) {
		const auto head = m_head.load(std::memory_order_relaxed);
		if (head - m_tail.load(std::memory_order_acquire) >= Capacity) {
			return false;
		}
		m_jobs[head % Capacity] = {destination, address, size};
		m_head.store(head + 1, std::memory_order_release);
		m_head.notify_one();
		return true;
	}
	// GPU thread: every copy pushed so far is done.
	void Wait() {
		const auto head = m_head.load(std::memory_order_relaxed);
		while (m_tail.load(std::memory_order_acquire) != head) {
			__builtin_ia32_pause();
		}
	}

private:
	struct Job {
		uint8_t* destination = nullptr;
		uint64_t address     = 0;
		uint64_t size        = 0;
	};
	static constexpr uint64_t Capacity = 8192;

	UploadWorker() {
		std::thread([this] {
			pthread_setname_np(pthread_self(), "UploadWorker");
			uint64_t tail = 0;
			for (;;) {
				uint64_t head = m_head.load(std::memory_order_acquire);
				for (uint32_t spin = 0; head == tail && spin < 100000u; spin++) {
					__builtin_ia32_pause();
					head = m_head.load(std::memory_order_acquire);
				}
				if (head == tail) {
					m_head.wait(tail, std::memory_order_acquire);
					continue;
				}
				while (tail != head) {
					const auto& job = m_jobs[tail % Capacity];
					ReadGuestForUpload(job.destination, job.address, job.size);
					tail++;
					m_tail.store(tail, std::memory_order_release);
				}
			}
		}).detach();
	}

	std::array<Job, Capacity> m_jobs {};
	std::atomic<uint64_t>     m_head {0};
	std::atomic<uint64_t>     m_tail {0};
};
} // namespace

void WaitUploadWorker() {
	UploadWorker::Get().Wait();
}
// Bytes uploaded through staging copies [0] and direct writes [1] (KYTY_SYNC_STATS).
static std::array<std::atomic<uint64_t>, 2> g_upload_bytes {};

// KYTY_HOT_BIND_STATS=1 (live, research, 2026-10-10): how the guest ranges where the game's write
// faults concentrate (KYTY_HOT_RANGES=<hex addr>+<hex size>[,...], default the combat ones) are
// bound, to design importing them as host memory (VK_EXT_external_memory_host / udmabuf): every
// 5 s, synchronizations touching them by caller, inside or crossing the range, written, the
// cached buffers that cover them, the bytes uploaded from them, GPU-written marks and shaders.
namespace HotBindStats {
enum Caller : uint32_t { Obtain, ObtainWrittenRead, ObtainWritten, Image, RangeSync, Other, Count };
inline thread_local uint32_t t_caller = Other;
struct Scope {
	uint32_t previous;
	explicit Scope(uint32_t caller): previous(std::exchange(t_caller, caller)) {}
	~Scope() { t_caller = previous; }
};
static bool On() {
	static auto& on = Common::LiveSwitches::Get("KYTY_HOT_BIND_STATS", 0);
	return on.load(std::memory_order_relaxed) != 0;
}
static const std::vector<std::pair<uint64_t, uint64_t>>& Ranges() {
	static const std::vector<std::pair<uint64_t, uint64_t>> ranges = [] {
		std::vector<std::pair<uint64_t, uint64_t>> result;
		const char* value = std::getenv("KYTY_HOT_RANGES");
		if (value == nullptr) {
			value = "10a0000000+400000,11e0000000+1000000";
		}
		while (value != nullptr && *value != '\0') {
			char*      end   = nullptr;
			const auto begin = std::strtoull(value, &end, 16);
			uint64_t   bytes = 0x1000;
			if (end != nullptr && *end == '+') {
				bytes = std::strtoull(end + 1, &end, 16);
			}
			result.emplace_back(begin, bytes);
			value = end != nullptr && *end == ',' ? end + 1 : nullptr;
		}
		return result;
	}();
	return ranges;
}
// Bytes of [vaddr, vaddr + size) inside the hot ranges; `inside` = all of them.
static uint64_t Overlap(uint64_t vaddr, uint64_t size, bool* inside = nullptr) {
	uint64_t bytes = 0;
	for (const auto& [begin, length]: Ranges()) {
		const auto lo = std::max(begin, vaddr);
		const auto hi = std::min(begin + length, vaddr + size);
		if (lo < hi) {
			bytes += hi - lo;
		}
	}
	if (inside != nullptr) {
		*inside = bytes == size;
	}
	return bytes;
}
struct Data {
	uint64_t calls[Count][2] {};    // [caller][crossing]
	uint64_t written_calls     = 0;
	uint64_t bytes_requested   = 0; // sum of the hot bytes of the synchronized ranges
	uint64_t largest           = 0;
	uint64_t uploaded          = 0; // hot bytes uploaded
	uint64_t upload_runs       = 0;
	uint64_t stream_hits       = 0; // small CPU-dirty reads served from the stream buffer
	uint64_t bda_requests      = 0;
	uint64_t gpu_marks         = 0;
	uint64_t gpu_mark_bytes    = 0;
	uint64_t image_tracks      = 0;
	std::map<std::pair<uint64_t, uint64_t>, uint64_t> buffers; // (begin, size) -> syncs
	std::map<uint64_t, uint64_t>                      writers; // shader -> GPU-written marks
	std::map<uint64_t, uint64_t>                      readers; // shader -> read syncs
	std::chrono::steady_clock::time_point             at = std::chrono::steady_clock::now();
};
static std::mutex s_mutex;
static Data       s_data;
static void       Report(Data& data) {
    const auto now = std::chrono::steady_clock::now();
    if (now - data.at < std::chrono::seconds(5)) {
        return;
    }
    static const char* names[Count] = {"obtain", "obtain-wr-read", "obtain-written", "image",
	                                         "range-sync", "other"};
    ::printf("Hot binds (5 s):");
    for (uint32_t caller = 0; caller < Count; caller++) {
        if (data.calls[caller][0] + data.calls[caller][1] != 0) {
            ::printf(" %s %" PRIu64 "/%" PRIu64 "x", names[caller], data.calls[caller][0],
			                  data.calls[caller][1]);
        }
    }
    ::printf("; written %" PRIu64 ", hot bytes synced %.1f MiB (largest %.2f MiB), uploaded %.2f MiB in %" PRIu64
	               " runs, stream hits %" PRIu64 ", bda requests %" PRIu64 ", GPU marks %" PRIu64 " (%.2f MiB), image tracks %" PRIu64 "\n",
	               data.written_calls, static_cast<double>(data.bytes_requested) / 1048576.0,
	               static_cast<double>(data.largest) / 1048576.0,
	               static_cast<double>(data.uploaded) / 1048576.0, data.upload_runs, data.stream_hits,
	               data.bda_requests, data.gpu_marks, static_cast<double>(data.gpu_mark_bytes) / 1048576.0,
	               data.image_tracks);
    std::vector<std::pair<uint64_t, std::pair<uint64_t, uint64_t>>> buffers;
    for (const auto& [range, count]: data.buffers) {
        buffers.emplace_back(count, range);
    }
    std::sort(buffers.rbegin(), buffers.rend());
    ::printf("  hot buffers %zu:", buffers.size());
    for (size_t i = 0; i < buffers.size() && i < 8; i++) {
        ::printf(" 0x%" PRIx64 "+0x%" PRIx64 " x%" PRIu64, buffers[i].second.first,
		                  buffers[i].second.second, buffers[i].first);
    }
    const auto top = [](const char* label, const std::map<uint64_t, uint64_t>& map) {
        std::vector<std::pair<uint64_t, uint64_t>> list;
        for (const auto& [shader, count]: map) {
            list.emplace_back(count, shader);
        }
        std::sort(list.rbegin(), list.rend());
        ::printf("; %s %zu:", label, list.size());
        for (size_t i = 0; i < list.size() && i < 6; i++) {
            ::printf(" %016" PRIx64 " x%" PRIu64, list[i].second, list[i].first);
        }
    };
    top("writers", data.writers);
    top("readers", data.readers);
    ::printf("\n");
    std::fflush(stdout);
    data    = {};
    data.at = now;
}
static void NoteSync(uint64_t vaddr, uint64_t size, bool is_written, uint64_t buffer_begin,
                     uint64_t buffer_size, uint64_t shader) {
	bool       inside = false;
	const auto hot    = Overlap(vaddr, size, &inside);
	if (hot == 0) {
		return;
	}
	std::lock_guard lock(s_mutex);
	s_data.calls[t_caller][inside ? 0 : 1]++;
	s_data.written_calls += is_written ? 1 : 0;
	s_data.bytes_requested += hot;
	s_data.largest = std::max(s_data.largest, size);
	s_data.buffers[{buffer_begin, buffer_size}]++;
	if (shader != 0 && !is_written) {
		s_data.readers[shader]++;
	}
	Report(s_data);
}
static void NoteUpload(uint64_t vaddr, uint64_t size) {
	const auto hot = Overlap(vaddr, size);
	if (hot == 0) {
		return;
	}
	std::lock_guard lock(s_mutex);
	s_data.uploaded += hot;
	s_data.upload_runs++;
}
static void NoteCounter(uint64_t Data::* counter, uint64_t vaddr, uint64_t size) {
	if (Overlap(vaddr, size) == 0) {
		return;
	}
	std::lock_guard lock(s_mutex);
	s_data.*counter += 1;
}
static void NoteGpuWrite(uint64_t vaddr, uint64_t size, uint64_t shader) {
	const auto hot = Overlap(vaddr, size);
	if (hot == 0) {
		return;
	}
	std::lock_guard lock(s_mutex);
	s_data.gpu_marks++;
	s_data.gpu_mark_bytes += hot;
	s_data.writers[shader]++;
}
} // namespace HotBindStats

void NoteHotImageTrack(uint64_t vaddr, uint64_t size) {
	if (HotBindStats::On()) {
		HotBindStats::NoteCounter(&HotBindStats::Data::image_tracks, vaddr, size);
	}
}

bool BufferCache::SynchronizeBuffer(Buffer& buffer, uint64_t vaddr, uint64_t size, bool is_written,
                                    bool is_texel_buffer) {
	if (buffer.IsImported()) {
		return false; // KYTY_HOST_IMPORT: the buffer is the guest memory
	}
	// KYTY_SYNC_STATS=1 (live): every 5 s, synchronizations and the bytes they walked, split by
	// read and written ranges (a written range is locked and marked GPU-modified page by page).
	static auto& stats = Common::LiveSwitches::Get("KYTY_SYNC_STATS", 0);
	if (stats.load(std::memory_order_relaxed) != 0) {
		static uint64_t calls[2]   = {};
		static uint64_t bytes[2]   = {};
		static uint64_t largest[2] = {};
		static auto     report     = std::chrono::steady_clock::now();

		const size_t kind = is_written ? 1 : 0;
		calls[kind]++;
		bytes[kind] += size;
		largest[kind] = std::max(largest[kind], size);

		const auto now = std::chrono::steady_clock::now();
		if (now - report >= std::chrono::seconds(5)) {
			report               = now;
			constexpr double GiB = 1024.0 * 1024.0 * 1024.0;
			::printf("Buffer upload (5 s): copied %.1f MiB, direct %.1f MiB\n",
			         static_cast<double>(g_upload_bytes[0].exchange(0)) / 1048576.0,
			         static_cast<double>(g_upload_bytes[1].exchange(0)) / 1048576.0);
			::printf("Buffer sync (5 s): read %" PRIu64 " calls %.1f GiB (largest %.1f MiB), "
			         "written %" PRIu64 " calls %.1f GiB (largest %.1f MiB)\n",
			         calls[0], static_cast<double>(bytes[0]) / GiB,
			         static_cast<double>(largest[0]) / 1048576.0, calls[1],
			         static_cast<double>(bytes[1]) / GiB,
			         static_cast<double>(largest[1]) / 1048576.0);
			std::fflush(stdout);
			calls[0] = calls[1] = bytes[0] = bytes[1] = largest[0] = largest[1] = 0;
		}
	}
	const bool hot_stats = HotBindStats::On();
	if (hot_stats) {
		HotBindStats::NoteSync(vaddr, size, is_written, buffer.CpuAddress(), buffer.Size(),
		                       s_diag_shader_hash);
	}
	std::vector<vk::BufferCopy> copies;
	uint64_t                    total_size = 0;
	vk::Buffer                  source;
	// KYTY_DIRECT_UPLOAD=1 (live, default 0; needs KYTY_MAPPED_DEVICE_BUFFERS=1): a buffer in
	// host-visible device memory with no GPU copy into it in flight gets the guest's bytes written
	// through its mapping, instead of a staging copy between two full barriers that also ends the
	// render pass (Wolverine: ~37,000 a second, ~7 a draw). As on the console, the guest only
	// changes bytes no pending GPU command reads (labels are published after the host GPU); the
	// other bytes of the uploaded pages are rewritten with their current values.
	static auto& direct_upload = Common::LiveSwitches::Get("KYTY_DIRECT_UPLOAD", 0);
	// KYTY_LOCAL_HACK KYTY_LAZY_DIRECT (live, default 1): the GPU-copy check is only needed when
	// there is something to upload; most of the ~280k syncs per second upload nothing, and the
	// check asked the driver for the timeline value (DRM_IOCTL_SYNCOBJ_QUERY, ~2k per frame,
	// st1 strace 2026-10-06) whenever the buffer's last copy was recent.
	static auto& lazy_direct  = Common::LiveSwitches::Get("KYTY_LAZY_DIRECT", 1);
	const bool   lazy         = lazy_direct.load(std::memory_order_relaxed) != 0;
	const bool   direct_wanted = direct_upload.load(std::memory_order_relaxed) != 0 &&
	                           !buffer.Mapped().empty();
	bool direct = direct_wanted && !lazy && m_scheduler.IsFree(buffer.last_gpu_copy_tick);
	m_memory_tracker.ForEachUploadRange(
	    vaddr, size, is_written,
	    [&](uint64_t address, uint64_t bytes) noexcept {
		    copies.emplace_back(total_size, buffer.Offset(address), bytes);
		    total_size += bytes;
		    if (hot_stats) {
			    HotBindStats::NoteUpload(address, bytes);
		    }
	    },
	    [&]() noexcept {
		    if (lazy && direct_wanted && !copies.empty()) {
			    direct = m_scheduler.IsFree(buffer.last_gpu_copy_tick);
		    }
		    // Locked adds only for KYTY_SYNC_STATS (they drained the store buffer per upload).
		    const bool count = stats.load(std::memory_order_relaxed) != 0;
		    if (!direct) {
			    if (count) g_upload_bytes[0].fetch_add(total_size, std::memory_order_relaxed);
			    source = UploadCopies(buffer, copies, total_size);
			    return;
		    }
		    static auto& upload_worker = Common::LiveSwitches::Get("KYTY_UPLOAD_WORKER", 1);
		    const bool   deferred = upload_worker.load(std::memory_order_relaxed) != 0 &&
		                          buffer.IsCoherent();
		    for (const auto& copy: copies) {
			    if (count) g_upload_bytes[1].fetch_add(copy.size, std::memory_order_relaxed);
			    if (deferred && UploadWorker::Get().Push(buffer.Mapped().data() + copy.dstOffset,
			                                            buffer.CpuAddress() + copy.dstOffset,
			                                            copy.size)) {
				    continue;
			    }
			    ReadGuestForUpload(buffer.Mapped().data() + copy.dstOffset,
			                       buffer.CpuAddress() + copy.dstOffset, copy.size);
			    if (!buffer.IsCoherent()) {
				    buffer.Flush(copy.dstOffset, copy.size);
			    }
		    }
	    });
	if (source) {
		buffer.last_gpu_copy_tick = m_scheduler.CurrentTick();
		auto& command = m_scheduler.Current();
		command.EndRendering();
		const auto              native = command.Recorder();
		vk::BufferMemoryBarrier before {};
		before.srcAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite |
		                       vk::AccessFlagBits::eTransferRead |
		                       vk::AccessFlagBits::eTransferWrite;
		before.dstAccessMask       = vk::AccessFlagBits::eTransferWrite;
		before.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		before.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		before.buffer              = buffer.Handle();
		before.offset              = 0;
		before.size                = buffer.Size();
		native.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
		                       vk::PipelineStageFlagBits::eTransfer,
		                       vk::DependencyFlagBits::eByRegion, 0, nullptr, 1, &before, 0, nullptr);
		native.copyBuffer(source, buffer.Handle(), static_cast<uint32_t>(copies.size()),
		                  copies.data());
		auto after          = before;
		after.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
		after.dstAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
		native.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
		                       vk::PipelineStageFlagBits::eAllCommands,
		                       vk::DependencyFlagBits::eByRegion, 0, nullptr, 1, &after, 0, nullptr);
	}
	if (is_texel_buffer && !is_written) {
		return SynchronizeBufferFromImage(buffer, vaddr, size);
	}
	return false;
}

// Reads guest memory for an upload. The guest can unmap memory a cached buffer still covers: AMM
// maps streaming textures in blocks, and a texture upload once read one byte past the end of the
// block it had mapped. Such bytes read as zero, like ObtainBufferForImage's direct path.
static void ReadGuestForUpload(uint8_t* destination, uint64_t address, uint64_t size) {
	if (!Libs::LibKernel::Memory::TryReadBacking(address, destination, size) &&
	    !Libs::LibKernel::Memory::TryReadSparseBacking(address, destination, size)) {
		static std::atomic<uint32_t> reported {0};
		if (reported.fetch_add(1) < 16) {
			LOGF("BufferCache: upload of unmapped guest memory 0x%016" PRIx64 " size=0x%" PRIx64
			     " reads as zero\n",
			     address, size);
		}
		std::memset(destination, 0, static_cast<size_t>(size));
	}
}

vk::Buffer BufferCache::UploadCopies(Buffer& buffer, std::span<vk::BufferCopy> copies,
                                     uint64_t total_size) {
	if (copies.empty()) {
		return nullptr;
	}

	auto [mapped, base_offset] = m_staging_buffer.Map(total_size, 4);
	if (mapped != nullptr) {
		// KYTY_UPLOAD_WORKER: the staging copies too (the GPU copy reads them after the submit).
		static auto& upload_worker = Common::LiveSwitches::Get("KYTY_UPLOAD_WORKER", 1);
		const bool   deferred      = upload_worker.load(std::memory_order_relaxed) != 0 &&
		                      m_staging_buffer.IsCoherent();
		for (auto& copy: copies) {
			const auto address = buffer.CpuAddress() + copy.dstOffset;
			if (!deferred || !UploadWorker::Get().Push(mapped + copy.srcOffset, address, copy.size)) {
				ReadGuestForUpload(mapped + copy.srcOffset, address, copy.size);
			}
			copy.srcOffset += base_offset;
		}
		m_staging_buffer.Commit();
		return m_staging_buffer.Handle();
	}

	auto temporary = std::make_unique<Buffer>(m_graphics, m_scheduler, MemoryUsage::Upload, 0,
	                                         vk::BufferUsageFlagBits::eTransferSrc, total_size);
	for (const auto& copy: copies) {
		const auto address = buffer.CpuAddress() + copy.dstOffset;
		ReadGuestForUpload(temporary->Mapped().data() + copy.srcOffset, address, copy.size);
	}
	temporary->Flush(0, total_size);
	const auto handle = temporary->Handle();
	m_scheduler.DeferOperation([owner = std::move(temporary)]() mutable { owner.reset(); });
	return handle;
}

std::pair<Buffer*, uint64_t> BufferCache::ObtainBuffer(uint64_t vaddr, uint64_t size,
                                                       bool is_written, bool is_texel_buffer,
                                                       BufferId id, bool needs_device_address) {
	auto& command = m_scheduler.Current();
	if (command.IsInvalid() || !GuestRange {vaddr, size}.Valid()) {
		EXIT("BufferCache: buffer request requires a recording command buffer\n");
	}
	PollHostImport();

	if (!is_written && size <= CACHING_PAGESIZE && !InActiveImportZone(vaddr, size) &&
	    (!needs_device_address || m_stream_buffer.HasDeviceAddress()) &&
	    !m_memory_tracker.IsRegionGpuModified(vaddr, size) &&
	    m_memory_tracker.IsRegionCpuModified(vaddr, size)) {
		const auto alignment = std::max<uint64_t>(
		    m_graphics.physical_device_properties.limits.minUniformBufferOffsetAlignment, 1);
		auto [mapped, offset] = m_stream_buffer.Map(size, alignment, false);
		if (mapped != nullptr) {
			if (HotBindStats::On()) {
				HotBindStats::NoteCounter(&HotBindStats::Data::stream_hits, vaddr, size);
			}
			std::memcpy(mapped, reinterpret_cast<const void*>(vaddr), size);
			m_stream_buffer.Commit();
			return {&m_stream_buffer, offset};
		}
	}

	if (IsBufferInvalid(id) || !m_slot_buffers[id].IsInBounds(vaddr, size)) {
		id = FindBuffer(vaddr, size);
	}
	auto& buffer = m_slot_buffers[id];
	TouchBuffer(buffer);
	if (HotBindStats::On() && needs_device_address) {
		HotBindStats::NoteCounter(&HotBindStats::Data::bda_requests, vaddr, size);
	}
	{
		HotBindStats::Scope caller(HotBindStats::Obtain);
		(void)SynchronizeBuffer(buffer, vaddr, size, is_written, is_texel_buffer);
	}
	if (is_written && buffer.IsImported()) {
		m_import_skipped_writes++;
	} else if (is_written) {
		MarkGpuWritten(vaddr, size);
		buffer.last_gpu_write_tick = m_scheduler.CurrentTick();
	}
	return {&buffer, buffer.Offset(vaddr)};
}

std::pair<Buffer*, uint64_t> BufferCache::ObtainBufferWritten(uint64_t vaddr, uint64_t size,
                                                              uint64_t written_vaddr,
                                                              uint64_t written_size, BufferId id) {
	auto& command = m_scheduler.Current();
	if (command.IsInvalid() || !GuestRange {vaddr, size}.Valid() ||
	    written_vaddr < vaddr || written_size > size || written_vaddr - vaddr > size - written_size) {
		EXIT("BufferCache: invalid written-range buffer request\n");
	}
	PollHostImport();
	if (IsBufferInvalid(id) || !m_slot_buffers[id].IsInBounds(vaddr, size)) {
		id = FindBuffer(vaddr, size);
	}
	auto& buffer = m_slot_buffers[id];
	TouchBuffer(buffer);
	// The whole binding is current on the GPU, as for a written buffer; only the bytes the
	// stores can reach become GPU-written.
	{
		HotBindStats::Scope caller(HotBindStats::ObtainWrittenRead);
		(void)SynchronizeBuffer(buffer, vaddr, size, false, false);
	}
	if (written_size != 0 && buffer.IsImported()) {
		m_import_skipped_writes++;
	} else if (written_size != 0) {
		HotBindStats::Scope caller(HotBindStats::ObtainWritten);
		(void)SynchronizeBuffer(buffer, written_vaddr, written_size, true, false);
		MarkGpuWritten(written_vaddr, written_size);
		buffer.last_gpu_write_tick = m_scheduler.CurrentTick();
	}
	return {&buffer, buffer.Offset(vaddr)};
}

void BufferCache::MarkGpuWritten(uint64_t vaddr, uint64_t size) {
	KYTY_PROFILER_BLOCK("Obtain::MarkWritten");
	{
		std::unique_lock lock(m_dirty_ranges_mutex);
		m_gpu_modified_ranges.Add(vaddr, size);
	}
	PipelineStats::NoteWrite(vaddr, size);
	ReadbackStats::NoteWrite(vaddr, size, m_scheduler.CurrentTick());
	if (HotBindStats::On()) {
		HotBindStats::NoteGpuWrite(vaddr, size, s_diag_shader_hash);
	}
	// Diagnostics: KYTY_WATCH_GPU_WRITE=<address>[+<size>][,<address>[+<size>]...] (hex) names
	// the bindings that mark bytes of those ranges GPU-written: the first write of each
	// (range, shader), with a host stack when no shader is being bound (copies, fills).
	static const std::vector<std::pair<uint64_t, uint64_t>> watches = [] {
		std::vector<std::pair<uint64_t, uint64_t>> result;
		const char*                                value = std::getenv("KYTY_WATCH_GPU_WRITE");
		while (value != nullptr && *value != '\0') {
			char*      end   = nullptr;
			const auto begin = std::strtoull(value, &end, 16);
			uint64_t   bytes = 1;
			if (end != nullptr && *end == '+') {
				bytes = std::strtoull(end + 1, &end, 16);
			}
			result.emplace_back(begin, bytes);
			value = end != nullptr && *end == ',' ? end + 1 : nullptr;
		}
		return result;
	}();
	static std::atomic<uint32_t> watched {0};
	for (size_t index = 0; index < watches.size(); index++) {
		const auto& [watch_begin, watch_size] = watches[index];
		if (watch_size == 0 || vaddr >= watch_begin + watch_size ||
		    watch_begin >= vaddr + size) {
			continue;
		}
		static std::mutex                             seen_mutex;
		static std::set<std::pair<size_t, uint64_t>> seen;
		bool                                          first = false;
		{
			std::lock_guard lock(seen_mutex);
			first = seen.emplace(index, s_diag_shader_hash).second;
		}
		if (first && watched.fetch_add(1) < 128) {
			::printf("GPU write covers watched #%zu 0x%016" PRIx64 "+0x%" PRIx64
			         ": range=0x%016" PRIx64 " size=0x%" PRIx64 " shader=0x%016" PRIx64 "\n%s",
			         index, watch_begin, watch_size, vaddr, size, s_diag_shader_hash,
			         s_diag_shader_hash == 0 ? Common::HostBacktrace().c_str() : "");
		}
	}
}

std::pair<Buffer*, uint64_t> BufferCache::ObtainBufferForImage(uint64_t vaddr, uint64_t size) {
	if (!GuestRange {vaddr, size}.Valid()) {
		EXIT("BufferCache: invalid image source\n");
	}
	const auto* owner = m_page_table.Find(vaddr >> PageTable::kPageBits);
	if (owner != nullptr && *owner) {
		auto& buffer = m_slot_buffers[*owner];
		if (buffer.IsInBounds(vaddr, size)) {
			TouchBuffer(buffer);
			HotBindStats::Scope caller(HotBindStats::Image);
			(void)SynchronizeBuffer(buffer, vaddr, size, false, false);
			return {&buffer, buffer.Offset(vaddr)};
		}
	}
	if (IsRegionGpuModified(vaddr, size)) {
		return ObtainBuffer(vaddr, size, false, false);
	}

	auto [staging, stage_offset] = m_staging_buffer.Map(size, 16);
	if (staging == nullptr) {
		EXIT("BufferCache: staging reservation failed for guest image\n");
	}
	if (!Libs::LibKernel::Memory::TryReadSparseBacking(vaddr, staging, size)) {
		std::memset(staging, 0, static_cast<size_t>(size));
	}
	m_staging_buffer.Commit();
	return {&m_staging_buffer, stage_offset};
}

void BufferCache::FillBuffer(uint64_t vaddr, uint64_t size, uint32_t value, bool is_gds) {
	if (!is_gds) PipelineStats::NoteWrite(vaddr, size);
	if ((vaddr & 3u) != 0 || size == 0 || (size & 3u) != 0 || size > UINT64_MAX - vaddr) {
		EXIT("BufferCache: fill range must be dword aligned\n");
	}
	if (is_gds) {
		if (vaddr > m_gds_buffer.Size() || size > m_gds_buffer.Size() - vaddr) {
			EXIT("BufferCache: GDS fill range is out of bounds\n");
		}
		m_gds_buffer.Fill(vaddr, size, value);
		return;
	}
	if (vaddr == 0) {
		EXIT("BufferCache: invalid fill memory address\n");
	}
	(void)m_texture_cache.ClearMeta(vaddr);
	if (!IsRegionGpuModified(vaddr, size)) {
		// Access the guest mapping so write faults invalidate cached buffers and images.
		auto* destination = reinterpret_cast<uint32_t*>(vaddr);
		std::fill(destination, destination + size / sizeof(uint32_t), value);
		return;
	}

	m_texture_cache.InvalidateMemoryFromGPU(vaddr, size);
	auto [dst, dst_offset] = ObtainBuffer(vaddr, size, true, true);
	dst->Fill(dst_offset, size, value);
}

void BufferCache::CopyBuffer(uint64_t dst_vaddr, uint64_t src_vaddr, uint64_t size, bool dst_gds,
                             bool src_gds) {
	if (!dst_gds) PipelineStats::NoteWrite(dst_vaddr, size);
	const bool dst_memory = !dst_gds;
	const bool src_memory = !src_gds;
	if ((dst_memory && dst_vaddr == 0) || (src_memory && src_vaddr == 0) || size == 0 ||
	    ((dst_gds || src_gds) && ((dst_vaddr | src_vaddr | size) & 3u) != 0) ||
	    size > UINT64_MAX - dst_vaddr || size > UINT64_MAX - src_vaddr || (dst_gds && src_gds) ||
	    (dst_gds && (dst_vaddr > m_gds_buffer.Size() || size > m_gds_buffer.Size() - dst_vaddr)) ||
	    (src_gds && (src_vaddr > m_gds_buffer.Size() || size > m_gds_buffer.Size() - src_vaddr))) {
		EXIT("BufferCache: invalid copy range, src=0x%016" PRIx64 " dst=0x%016" PRIx64
		     " size=0x%016" PRIx64 " src_gds=%d dst_gds=%d\n",
		     src_vaddr, dst_vaddr, size, static_cast<int>(src_gds), static_cast<int>(dst_gds));
	}
	if (src_memory && dst_memory && !IsRegionGpuModified(dst_vaddr, size) &&
	    !IsRegionGpuModified(src_vaddr, size) && !m_texture_cache.FindImageFromRange(src_vaddr, size)) {
		std::memcpy(reinterpret_cast<void*>(dst_vaddr), reinterpret_cast<const void*>(src_vaddr),
		            size);
		return;
	}

	auto& command = m_scheduler.Current();
	if (dst_memory) {
		m_texture_cache.InvalidateMemoryFromGPU(dst_vaddr, size);
	}
	const auto src_id      = src_memory ? FindBuffer(src_vaddr, size) : BufferId {};
	const auto dst_id      = dst_memory ? FindBuffer(dst_vaddr, size) : BufferId {};
	auto [src, src_offset] = src_memory ? ObtainBuffer(src_vaddr, size, false, true, src_id)
	                                    : std::pair {&m_gds_buffer, src_vaddr};
	auto [dst, dst_offset] = dst_memory ? ObtainBuffer(dst_vaddr, size, true, true, dst_id)
	                                    : std::pair {&m_gds_buffer, dst_vaddr};
	dst->CopyFrom(command, *src, src_offset, dst_offset, size);
}

bool BufferCache::IsRegionRegistered(uint64_t vaddr, uint64_t size) {
	if (!GuestRange {vaddr, size}.Valid()) {
		EXIT("BufferCache: invalid registered-region query\n");
	}
	// Cached buffers are ordered and non-overlapping. The last buffer beginning before the query
	// end is therefore the only possible intersection.
	const auto candidate = m_buffers.lower_bound(vaddr + size);
	if (candidate == m_buffers.begin()) {
		return false;
	}
	const auto& [address, id] = *std::prev(candidate);
	return address + m_slot_buffers[id].Size() > vaddr;
}

bool BufferCache::IsRegionGpuModified(uint64_t vaddr, uint64_t size) {
	return m_memory_tracker.IsRegionGpuModified(vaddr, size);
}

bool BufferCache::HasGpuDirtyBytes(uint64_t vaddr, uint64_t size) {
	// Bytes of a pending asynchronous write readback are GPU-owned until published.
	return m_gpu_modified_ranges.Intersects(vaddr, size) ||
	       OverlapsPendingWriteReadback(vaddr, vaddr + size);
}

bool BufferCache::IsCleanForConcurrentRead(uint64_t vaddr, uint64_t size) const {
	std::shared_lock lock(m_dirty_ranges_mutex);
	return !m_gpu_modified_ranges.Intersects(vaddr, size) &&
	       !m_downloading_ranges.Intersects(vaddr, size);
}

bool BufferCache::IsRegionCpuModified(uint64_t vaddr, uint64_t size) {
	return m_memory_tracker.IsRegionCpuModified(vaddr, size);
}

uint64_t BufferCache::LruClock() const noexcept {
	return m_graphics.presented_frames.load(std::memory_order_relaxed) + m_gc_tick / 512;
}

// KYTY_BDA_VERIFY=N (live, seconds, 0 = off, research): every N seconds the GPU's BDA page table
// entries of all live buffers are copied to host memory and compared with the addresses
// ChangeRegister wrote. Wolverine run 88 lost the device reading a buffer page through the
// address of a buffer destroyed 316 s earlier, while the host's view of the table named the
// live buffer that replaced it: this tells a GPU table out of step from a stale reference
// elsewhere. Mismatches print the guest page, both addresses and what owned the stale one.
void BufferCache::VerifyBdaPageTable() {
	static auto& interval = Common::LiveSwitches::Get("KYTY_BDA_VERIFY", 0);
	if (m_bda_verify_tick != 0) {
		if (!m_scheduler.IsFree(m_bda_verify_tick)) {
			return;
		}
		auto& download = *m_bda_verify_download;
		if (!download.IsCoherent()) {
			download.Invalidate(0, download.Size());
		}
		const auto* data       = download.Mapped().data();
		uint64_t    pages      = 0;
		uint64_t    mismatches = 0;
		uint64_t    stale      = 0;
		const auto table_entry = [&](uint64_t element) {
			vk::DeviceAddress value = 0;
			std::memcpy(&value, data + element * sizeof(value), sizeof(value));
			return value;
		};
		for (const auto& run: m_bda_verify_runs) {
			for (uint64_t i = 0; i < run.pages; i++) {
				// Through the GPU's own directory, as a shader reads it.
				const auto page  = PageIndex(run.guest_address + (i << CACHING_PAGEBITS));
				const auto chunk = table_entry(page >> BDA_CHUNK_BITS);
				const auto actual =
				    chunk != 0 ? table_entry(chunk + (page & (BDA_CHUNK_PAGES - 1u))) : 0;
				const auto expected = m_bda_verify_expected[run.first_address + i];
				pages++;
				if (actual == expected) {
					continue;
				}
				stale += expected == 0 ? 1u : 0u;
				if (mismatches++ < 16) {
					std::printf("BDA verify: guest page 0x%" PRIx64 " GPU table 0x%" PRIx64
					            " expected 0x%" PRIx64 "%s\n",
					            run.guest_address + (i << CACHING_PAGEBITS), actual, expected,
					            expected == 0 ? " (no buffer: stale entry)" : "");
					if (actual != 0) {
						AddressBindingDescribe(actual);
					}
				}
			}
		}
		std::printf("BDA verify: %" PRIu64 " pages in %zu registered ranges, %" PRIu64
		            " mismatches (%" PRIu64 " stale on pages without a buffer); chunks %zu in use,"
		            " peak %u of %" PRIu64 "\n",
		            pages, m_bda_verify_runs.size(), mismatches, stale,
		            static_cast<size_t>(BDA_CHUNK_COUNT - m_bda_free_chunks.size()),
		            m_bda_chunks_peak, BDA_CHUNK_COUNT);
		std::fflush(stdout);
		m_bda_verify_tick = 0;
		m_bda_verify_last = std::chrono::steady_clock::now();
		return;
	}
	const auto seconds = interval.load(std::memory_order_relaxed);
	if (seconds <= 0 ||
	    std::chrono::steady_clock::now() - m_bda_verify_last < std::chrono::seconds(seconds)) {
		return;
	}
	// Every guest range ever registered, not only live buffers: a page whose buffer is gone must
	// read 0. Expected values are taken now, in recording order with the copy below.
	m_bda_verify_runs.clear();
	m_bda_verify_expected.clear();
	uint64_t bytes = 0;
	m_bda_ever_registered.ForEach([&](uint64_t start, uint64_t end) {
		PageTable::PageRange range {};
		if (!PageTable::TryGetPageRange(start, end - start, range)) {
			return;
		}
		const auto pages = static_cast<uint64_t>(range.last_exclusive - range.first);
		m_bda_verify_runs.push_back({
		    .guest_address   = start,
		    .table_offset    = PageIndex(start) * sizeof(vk::DeviceAddress),
		    .download_offset = bytes,
		    .first_address   = m_bda_verify_expected.size(),
		    .pages           = pages,
		});
		for (size_t page = range.first; page < range.last_exclusive; ++page) {
			const auto* id     = m_page_table.Find(page);
			const auto* buffer = id != nullptr && *id ? m_slot_buffers.try_get(*id) : nullptr;
			vk::DeviceAddress expected = 0;
			if (buffer != nullptr && buffer->HasDeviceAddress()) {
				const auto page_address = static_cast<uint64_t>(page) << CACHING_PAGEBITS;
				const auto first_page   = PageIndex(buffer->CpuAddress()) << CACHING_PAGEBITS;
				const auto packed_page  = PageIndex(page_address) << CACHING_PAGEBITS;
				expected                = buffer->BufferDeviceAddress() + (packed_page - first_page);
			}
			m_bda_verify_expected.push_back(expected);
		}
		bytes += pages * sizeof(vk::DeviceAddress);
	});
	if (bytes == 0) {
		return;
	}
	// The whole two-level table (directory + chunk pool) in one copy.
	if (m_bda_verify_download == nullptr) {
		m_bda_verify_download = std::make_unique<Buffer>(
		    m_graphics, m_scheduler, MemoryUsage::Download, 0,
		    vk::BufferUsageFlagBits::eTransferDst, BDA_PAGETABLE_SIZE);
	}
	m_bda_verify_download->CopyFrom(m_scheduler.Current(), m_bda_pagetable_buffer, 0, 0,
	                                BDA_PAGETABLE_SIZE);
	m_bda_verify_tick = m_scheduler.CurrentTick();
}

// KYTY_BUFFER_AUDIT=1 (live, research, PLAN.md P1.3): every 10 s, what the buffer cache holds —
// count and size histogram, and the largest buffers with their guest range, frames since use
// and whether textures live in the same memory — to find buffer memory worth collecting.
void BufferCache::AuditBuffers() {
	static auto& audit = Common::LiveSwitches::Get("KYTY_BUFFER_AUDIT", 0);
	static auto  last  = std::chrono::steady_clock::now();
	const auto   now   = std::chrono::steady_clock::now();
	if (audit.load(std::memory_order_relaxed) == 0 || now - last < std::chrono::seconds(10)) {
		return;
	}
	last = now;
	constexpr std::array<uint64_t, 6> Limits {1u << 20u, 4u << 20u, 16u << 20u, 64u << 20u,
	                                          256u << 20u, UINT64_MAX};
	std::array<uint64_t, Limits.size()> counts {};
	std::array<uint64_t, Limits.size()> bytes {};
	std::vector<BufferId>               ids;
	uint64_t                            total = 0;
	for (const auto& [address, id]: m_buffers) {
		const auto& buffer = m_slot_buffers[id];
		const auto  size   = buffer.Size();
		const auto  bucket = static_cast<size_t>(
		    std::ranges::find_if(Limits, [&](uint64_t limit) { return size < limit; }) -
		    Limits.begin());
		counts[bucket]++;
		bytes[bucket] += size;
		total += size;
		ids.push_back(id);
	}
	static constexpr const char* Names[] = {"<1M", "<4M", "<16M", "<64M", "<256M", ">=256M"};
	std::printf("Buffer audit: %zu buffers, %.2f GiB:", ids.size(),
	            static_cast<double>(total) / (1u << 30u));
	for (size_t i = 0; i < Limits.size(); i++) {
		std::printf(" %s %" PRIu64 " (%.0f MiB)", Names[i], counts[i],
		            static_cast<double>(bytes[i]) / (1u << 20u));
	}
	std::printf("\n");
	std::ranges::sort(ids, [&](BufferId a, BufferId b) {
		return m_slot_buffers[a].Size() > m_slot_buffers[b].Size();
	});
	const auto clock = LruClock();
	for (size_t i = 0; i < std::min<size_t>(ids.size(), 12); i++) {
		const auto& buffer = m_slot_buffers[ids[i]];
		const auto  idle   = clock - std::min(clock, m_lru_cache.TickOf(buffer.lru_id));
		std::printf("  0x%011" PRIx64 " %7.1f MiB idle %4" PRIu64 " frames%s%s\n", buffer.CpuAddress(),
		            static_cast<double>(buffer.Size()) / (1u << 20u), idle,
		            m_texture_cache.HasImagesInRegion(buffer.CpuAddress(), buffer.Size())
		                ? " shares memory with textures"
		                : "",
		            m_memory_tracker.IsRegionGpuModified(buffer.CpuAddress(), buffer.Size())
		                ? " GPU-written"
		                : "");
	}
	std::fflush(stdout);
}

void BufferCache::RunGarbageCollector() {
	VerifyBdaPageTable();
	AuditBuffers();
	m_gc_tick++;
	const auto clock = LruClock();
	// Pressure is judged by this cache's own bytes. Device-wide usage also counts the images
	// spilled to host memory, which on a 6 GB card keeps it above the critical mark forever
	// and has the collector destroy and recreate every buffer twice a second (a third of
	// the GPU thread's time in the world). KYTY_GC_DEVICE_BYTES=1 restores that policy.
	static const bool device_bytes = std::getenv("KYTY_GC_DEVICE_BYTES") != nullptr;
	if (device_bytes && m_graphics.CanReportMemoryUsage()) {
		m_total_used_memory = m_graphics.GetDeviceMemoryUsage();
	}
	// KYTY_GC_COMBINED=1: the images count too. Each cache derives its thresholds from the whole
	// device budget, so judged apart the two can fill well past it together; on a 12 GB card
	// Windows then pages cache memory to system RAM and every frame waits on PCIe.
	// Wolverine run 16 (paging, device usage 11.0 > budget 9.9 GiB): on trimmed to 9.8 GiB, GPU
	// busy 86% -> 27%, 167 -> 117 ms/frame; a one-time stutter while it trims.
	static auto&   combined = Common::LiveSwitches::Get("KYTY_GC_COMBINED", 1);
	// KYTY_GC_HEADROOM_MB: see TextureCache::RunGarbageCollector.
	static auto&   headroom = Common::LiveSwitches::Get("KYTY_GC_HEADROOM_MB", 0);
	const uint64_t used =
	    m_total_used_memory +
	    (combined.load(std::memory_order_relaxed) != 0 ? m_texture_cache.UsedMemory() : 0) +
	    (static_cast<uint64_t>(std::max<int64_t>(0, headroom.load(std::memory_order_relaxed)))
	     << 20u);
	if (used < m_trigger_gc_memory) {
		return;
	}
	const bool aggressive = used >= m_critical_gc_memory;
	// KYTY_BUFFER_STATS=1 (live): every 5 s, the collections and what they freed, and the
	// buffers created meanwhile, to see whether the collector churns buffers still in use.
	static auto& churn_stats = Common::LiveSwitches::Get("KYTY_BUFFER_STATS", 0);
	struct GcReport {
		uint64_t runs = 0, deleted = 0, downloaded = 0, deleted_bytes = 0;
		uint64_t created_seen                    = 0;
		std::chrono::steady_clock::time_point at = std::chrono::steady_clock::now();
	};
	static GcReport gc_report;
	const auto      report_gc = [&](size_t deleted, size_t downloaded, uint64_t bytes) {
        if (churn_stats.load(std::memory_order_relaxed) == 0) {
            return;
        }
        gc_report.runs++;
        gc_report.deleted += deleted;
        gc_report.downloaded += downloaded;
        gc_report.deleted_bytes += bytes;
        const auto now = std::chrono::steady_clock::now();
        if (now - gc_report.at >= std::chrono::seconds(5)) {
            ::printf("Buffer GC (5 s): %" PRIu64
			              " runs (used %.2f GiB, trigger %.2f, critical %.2f%s), "
			                   "%" PRIu64 " deleted (%.1f MiB), %" PRIu64 " downloaded first; %" PRIu64
			              " buffers created\n",
			              gc_report.runs, static_cast<double>(used) / (1u << 30u),
			              static_cast<double>(m_trigger_gc_memory) / (1u << 30u),
			              static_cast<double>(m_critical_gc_memory) / (1u << 30u),
                     aggressive ? ", aggressive" : "", gc_report.deleted,
			              static_cast<double>(gc_report.deleted_bytes) / (1u << 20u),
			              gc_report.downloaded, m_buffers_created - gc_report.created_seen);
            gc_report              = {};
            gc_report.created_seen = m_buffers_created;
        }
	};
	// Ages in frames, as in the texture cache: a buffer used this frame or the last is
	// never a candidate, whatever the submission count.
	// KYTY_BUFFER_GC_AGE (live, frames): the age below the critical mark for buffers that share no
	// memory with a cached image (default set below).
	// Wolverine sits between the marks (7.7 of 5.2/8.4 GiB with the images): at 4 the collector ran
	// on every submission and retired ~170 small buffers/s that came back at once (~340 created/s,
	// run 28). Run 29 A/B 4/120: buffers created 1500 -> 15 per 5 s, same memory use, median 83.1
	// -> 66.8 ms, mean 76.7 -> 72.5 ms. Above the critical mark the age stays 2 frames.
	// But 120 for every buffer loses UI glyphs (run 30: the title menu draws "C N INU G M"): a
	// buffer kept alive over an image's memory feeds the image stale bytes (see the GC comment
	// below and SynchronizeBufferFromImage). So the longer age applies only to buffers that share
	// no bytes with a cached image; those keep the base age of 4 frames. Run 31 A/B 4/120 with this
	// rule (age 120 from startup): menu and HUD text intact, buffers created ~1500 -> ~15 per 5 s,
	// median 83.2 -> 66.9 ms, mean 76.8 -> 73.2 ms.
	static auto&   gc_age = Common::LiveSwitches::Get("KYTY_BUFFER_GC_AGE", 120);
	const uint64_t relaxed_age =
	    static_cast<uint64_t>(std::max<int64_t>(2, gc_age.load(std::memory_order_relaxed)));
	// KYTY_BUFFER_GC_CRITICAL_AGE (live, frames, default 60): the age above the critical mark.
	// Wolverine combat (run 49): the working set sat at 8.4-8.5 GiB, just over the critical mark,
	// so at age 2 the collector stayed aggressive and every 5 s deleted ~1000 buffers (~200
	// downloaded first, each waiting for the GPU) that were all created again (~2500). Age 60
	// cut that churn 10x with no frame-time loss (run 50); each deletion also risked the stale
	// buffer reads behind the device losts of runs 55/57.
	static auto&   critical_age = Common::LiveSwitches::Get("KYTY_BUFFER_GC_CRITICAL_AGE", 60);
	const uint64_t aggressive_age =
	    static_cast<uint64_t>(std::max<int64_t>(1, critical_age.load(std::memory_order_relaxed)));
	const uint64_t age        = std::min<uint64_t>(aggressive ? aggressive_age : 4, clock);
	const size_t   limit      = aggressive ? 64 : 32;
	// Kept buffers stay at the front of the LRU list; bound the walk past them.
	constexpr size_t MaxKeptVisits = 256;
	size_t           kept_visits   = 0;

	std::vector<BufferId> dirty_buffers;
	size_t                retire_count  = 0;
	uint64_t              retired_bytes = 0;
	m_lru_cache.ForEachItemBelow(clock - age, [&](BufferId id) {
		auto& buffer = m_slot_buffers[id];
		EXIT_IF(buffer.is_deleted);
		if (buffer.CpuAddress() == 0 || buffer.IsImported()) {
			// See CreateBuffer: the tracker rejects address 0, so this one is never collected.
			// A host-imported zone costs no device memory.
			return false;
		}
		if (!aggressive && relaxed_age > age &&
		    clock - m_lru_cache.TickOf(buffer.lru_id) < relaxed_age &&
		    !m_texture_cache.HasImagesInRegion(buffer.CpuAddress(), buffer.Size())) {
			return ++kept_visits == MaxKeptVisits;
		}
		if (OverlapsPendingWriteReadback(buffer.CpuAddress(),
		                                 buffer.CpuAddress() + buffer.Size())) {
			// A guest write's readback is still landing: its pages are GPU-owned without dirty
			// bytes, which the download below would reject. Collect the buffer later.
			return false;
		}
		m_memory_tracker.ValidateGpuDirtyOwnership(m_gpu_modified_ranges, buffer.CpuAddress(),
		                                           buffer.Size(), "garbage collection");
		// An aged GPU-dirty buffer is downloaded and dropped under either policy. Retaining it
		// (the non-aggressive rule before) leaves its GPU-written bytes owning the pages
		// forever, and the images that share those pages are then re-sourced from the
		// buffer's stale contents: the world renders black. Measured 2026-09-21.
		const bool dirty = m_memory_tracker.IsRegionGpuModified(buffer.CpuAddress(), buffer.Size());
		if (dirty) {
			EXIT_IF(!DownloadBufferMemory(buffer, buffer.CpuAddress(), buffer.Size()));
			dirty_buffers.push_back(id);
		} else {
			m_memory_tracker.UntrackMemory(buffer.CpuAddress(), buffer.Size());
			retired_bytes += buffer.Size();
			DeleteBuffer(id);
		}
		return ++retire_count == limit;
	});
	report_gc(retire_count, dirty_buffers.size(), retired_bytes);
	if (dirty_buffers.empty()) {
		return;
	}

	// Publish all queued downloads before releasing their tracked pages and owners.
	const auto completion_tick = m_scheduler.CurrentTick();
	m_scheduler.Wait(completion_tick);
	m_scheduler.WaitPriorityOperations(completion_tick);
	for (const auto id: dirty_buffers) {
		auto& buffer = m_slot_buffers[id];
		m_memory_tracker.UnmarkRegionAsGpuModified(buffer.CpuAddress(), buffer.Size());
		if (m_memory_tracker.IsRegionGpuModified(buffer.CpuAddress(), buffer.Size()) ||
		    m_gpu_modified_ranges.Intersects(buffer.CpuAddress(), buffer.Size())) {
			EXIT("BufferCache: garbage collection retained GPU ownership\n");
		}
		m_memory_tracker.UntrackMemory(buffer.CpuAddress(), buffer.Size());
		Unregister(id);
		m_slot_buffers.erase(id);
	}
}

void BufferCache::ProcessFaultBuffer() {
	m_fault_manager.ProcessFaultBuffer();
}

void BufferCache::SynchronizeCpuDirtyBuffersInRange(uint64_t vaddr, uint64_t size) {
	m_memory_tracker.ForEachMaybeCpuDirtyRegion(
	    vaddr, size, [this](uint64_t address, uint64_t bytes) {
		    SynchronizeBuffersInRange(address, bytes);
	    });
}

void BufferCache::SynchronizeBuffersInRange(uint64_t vaddr, uint64_t size) {
	const auto end = vaddr + size;
	auto       it  = m_buffers.upper_bound(vaddr);
	if (it != m_buffers.begin()) {
		--it;
	}
	for (; it != m_buffers.end() && it->first < end; ++it) {
		auto&      buffer = m_slot_buffers[it->second];
		const auto start  = std::max(buffer.CpuAddress(), vaddr);
		const auto finish = std::min(buffer.CpuAddress() + buffer.Size(), end);
		if (start < finish) {
			HotBindStats::Scope caller(HotBindStats::RangeSync);
			(void)SynchronizeBuffer(buffer, start, finish - start, false, false);
		}
	}
}

} // namespace Libs::Graphics
