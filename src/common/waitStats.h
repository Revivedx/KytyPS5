#ifndef EMULATOR_INCLUDE_EMULATOR_COMMON_WAITSTATS_H_
#define EMULATOR_INCLUDE_EMULATOR_COMMON_WAITSTATS_H_

#include "common/liveSwitches.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <ctime>
#include <unistd.h>

// KYTY_LOCAL_HACK research: KYTY_WAIT_STATS=1 (live). Where the command processor (the GPU thread)
// spends its wall time: each blocking point it can reach is a Scope; only the outermost scope on
// the thread counts (a readback inside a fault counts as the fault). Every 5 s one line: ms per
// kind, the thread's CPU time and the wall time, so on-CPU + listed waits + unlisted = wall.
namespace Common::WaitStats {

enum Kind : uint32_t {
	NoWork,        // ThreadRun: no submission queued
	BlockedQueues, // ThreadRun: every queue blocked (label waits), 100 us polls
	GpuWait,       // MasterSemaphore::Wait (a GPU tick)
	PriorityOps,   // WaitPriorityOperations / DrainPriorityOperations
	RendererMutex, // RenderContext::LockMutexProfiled (shared with the present thread)
	Fault,         // RenderContext::HandleFault on the CP (guest page faults, readbacks)
	Submit,        // queue mutex + vkQueueSubmit
	RecordDrain,   // CommandScheduler::DrainRecording (waiting for the record thread)
	Readback,      // BufferCache readbacks the CP asks for itself (not inside a fault)
	FaultRead,     // RenderContext::HandleFault, read access (Fault = write access)
	CommandSync,   // GuestGpu::SendCommandSync (a game thread waiting for the CP to run a command)
	FaultBuffers,  // write fault: BufferCache::InvalidateMemory (inner timer, any depth)
	FaultTextures, // write fault: TextureCache::InvalidateMemory (inner timer, any depth)
	FaultAheadK,   // write fault: FaultAhead (inner timer, any depth)
	FaultLock,     // write fault: the tracker region lock wait (inner)
	FaultState,    // write fault: ChangeState under the lock (unprotect, inner)
	FaultFlush,    // write fault on GPU-written bytes: the readback before the write (inner)
	CpLockHold,    // CP: region locks held across a written upload (CpPart, CP only)
	ReadHop,       // read fault, async path: waiting for the CP to start the readback (inner)
	ReadGpu,       // read fault, async path: waiting for the readback's GPU tick + publication (inner)
	ReadFinish,    // read fault, async path: waiting for the CP to finish the readback (inner)
	ReadSync,      // read fault, sync path: the CP doing the whole readback (inner)
	CpBegin,       // CP: BeginWriteReadback for a game thread (CpPart)
	CpCopyStart,   // CP: StartGuestCopy (CpPart)
	CpCopySubmit,  // CP: its vkQueueSubmit (CpPart)
	CpCopyDone,    // CP: CompleteGuestCopy (fence wait + dirty state, CpPart)
	CpCopyFinish,  // CP: FinishGuestCopy (CpPart)
	ReadCopyFence, // read fault: the game thread waiting for its guest copy's fence (inner)
	ReadCopyPub,   // read fault: the game thread publishing the copy into guest memory (inner)
	Count,
};

inline thread_local bool     t_cp_thread = false; // set by GuestGpu::ThreadRun
inline thread_local uint32_t t_depth     = 0;
// Other threads (the game's): wall time in the same scopes, summed over threads.
inline std::array<std::atomic<uint64_t>, Count> g_other_ns {};
inline std::array<std::atomic<uint64_t>, Count> g_other_calls {};

// Research 2026-10-13: the same scopes per game thread (outer scopes and inner parts), plus the
// guest submissions each thread enqueued: which thread's waits are on the frame's path.
struct ThreadSlot {
	std::atomic<uint32_t>                       tid {0};
	std::array<std::atomic<uint64_t>, Count>    ns {};
	std::atomic<uint64_t>                       submits {0};
};
inline std::array<ThreadSlot, 128> g_thread_slots {};
inline thread_local ThreadSlot*     t_slot = nullptr;

inline ThreadSlot* Slot() {
	if (t_slot != nullptr) return t_slot;
	const auto tid = static_cast<uint32_t>(gettid());
	for (auto& slot: g_thread_slots) {
		uint32_t expected = 0;
		if (slot.tid.load(std::memory_order_relaxed) == tid ||
		    slot.tid.compare_exchange_strong(expected, tid, std::memory_order_relaxed)) {
			t_slot = &slot;
			return t_slot;
		}
	}
	return nullptr;
}

// Research 2026-10-13: KYTY_TIMELINE_N also records the game threads' outer waits ('X': arg = kind |
// microseconds << 8 | tid << 32, stamped at the end of the wait); set by graphicsRun.cpp.
inline void (*g_timeline_hook)(char, uint64_t) = nullptr;

inline void NoteThread(uint32_t kind, uint64_t ns) {
	if (auto* slot = Slot(); slot != nullptr) {
		slot->ns[kind].fetch_add(ns, std::memory_order_relaxed);
	}
}

struct State {
	std::array<uint64_t, Count> ns {};
	std::array<uint64_t, Count> calls {};
	std::chrono::steady_clock::time_point report = std::chrono::steady_clock::now();
	uint64_t                              cpu_ns = 0;
};

inline State& GetState() {
	static State state; // CP thread only
	return state;
}

inline bool On() {
	static auto& on = Common::LiveSwitches::Get("KYTY_WAIT_STATS", 0);
	return on.load(std::memory_order_relaxed) != 0;
}

inline uint64_t ThreadCpuNs() {
	timespec ts {};
	clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts);
	return static_cast<uint64_t>(ts.tv_sec) * 1000000000ull + static_cast<uint64_t>(ts.tv_nsec);
}

inline void Report() {
	auto&      s   = GetState();
	const auto now = std::chrono::steady_clock::now();
	if (now - s.report < std::chrono::seconds(5)) {
		return;
	}
	const auto wall = static_cast<double>(
	    std::chrono::duration_cast<std::chrono::nanoseconds>(now - s.report).count());
	const auto cpu = ThreadCpuNs();
	static const char* names[Count] = {"no-work", "blocked-queues", "gpu-wait", "priority-ops",
	                                   "renderer-mutex", "fault-write", "submit", "record-drain", "readback",
	                                   "fault-read", "command-sync", "  w-buffers", "  w-textures",
	                                   "  w-ahead", "    lock-wait", "    change-state",
	                                   "    flush-readback", "cp-written-upload-hold", "  r-hop", "  r-gpu",
	                                   "  r-finish", "  r-sync", "cp-begin", "cp-copy-start",
	                                   "cp-copy-submit", "cp-copy-done", "cp-copy-finish",
	                                   "  r-copy-fence", "  r-copy-pub"};
	std::printf("CP waits (%.1f s): cpu %.0f ms;", wall / 1e9,
	            s.cpu_ns != 0 ? static_cast<double>(cpu - s.cpu_ns) / 1e6 : 0.0);
	uint64_t listed = 0;
	for (uint32_t k = 0; k < Count; k++) {
		listed += s.ns[k];
		std::printf(" %s %.0f ms/%" PRIu64, names[k], static_cast<double>(s.ns[k]) / 1e6, s.calls[k]);
	}
	std::printf("; listed %.0f ms of %.0f ms wall; other threads:", static_cast<double>(listed) / 1e6,
	            wall / 1e6);
	for (uint32_t k = 0; k < Count; k++) {
		const auto ns = g_other_ns[k].exchange(0, std::memory_order_relaxed);
		const auto n  = g_other_calls[k].exchange(0, std::memory_order_relaxed);
		if (n != 0) {
			std::printf(" %s %.0f ms/%" PRIu64, names[k], static_cast<double>(ns) / 1e6, n);
		}
	}
	std::printf("\n");
	// Per game thread: the 6 with the most outer-scope wait, their submissions and top kinds.
	struct Row {
		uint32_t                    tid;
		uint64_t                    total;
		uint64_t                    submits;
		std::array<uint64_t, Count> ns;
	};
	std::array<Row, 128> rows {};
	size_t               count = 0;
	for (auto& slot: g_thread_slots) {
		const auto tid = slot.tid.load(std::memory_order_relaxed);
		if (tid == 0) continue;
		Row row {tid, 0, slot.submits.exchange(0, std::memory_order_relaxed), {}};
		for (uint32_t k = 0; k < Count; k++) {
			row.ns[k] = slot.ns[k].exchange(0, std::memory_order_relaxed);
			// Outer kinds only in the total (inner parts are a breakdown of them).
			if (k <= CommandSync) row.total += row.ns[k];
		}
		if (row.total != 0 || row.submits != 0) rows[count++] = row;
	}
	std::sort(rows.begin(), rows.begin() + count, [](const Row& a, const Row& b) {
		return a.total + a.submits * 1000000ull > b.total + b.submits * 1000000ull;
	});
	for (size_t i = 0; i < count && i < 6; i++) {
		char  comm[32] = "?";
		char  path[64];
		std::snprintf(path, sizeof(path), "/proc/self/task/%u/comm", rows[i].tid);
		if (FILE* f = std::fopen(path, "r"); f != nullptr) {
			if (std::fgets(comm, sizeof(comm), f) != nullptr) {
				for (char* c = comm; *c != 0; c++) {
					if (*c == '\n') *c = 0;
				}
			}
			std::fclose(f);
		}
		std::printf("  thread %u (%s): submits %" PRIu64 ", waits %.0f ms:", rows[i].tid, comm,
		            rows[i].submits, static_cast<double>(rows[i].total) / 1e6);
		for (uint32_t k = 0; k < Count; k++) {
			if (rows[i].ns[k] >= 20000000ull) {
				std::printf(" %s %.0f", names[k], static_cast<double>(rows[i].ns[k]) / 1e6);
			}
		}
		std::printf("\n");
	}
	std::fflush(stdout);
	s.ns     = {};
	s.calls  = {};
	s.report = now;
	s.cpu_ns = cpu;
}

class Scope {
public:
	explicit Scope(Kind kind): m_kind(kind) {
		if (t_depth++ == 0 && On()) {
			m_start = std::chrono::steady_clock::now();
			m_on    = true;
		}
	}
	~Scope() {
		t_depth--;
		if (!m_on) {
			return;
		}
		if (!t_cp_thread) {
			const auto ns = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
			                                          std::chrono::steady_clock::now() - m_start)
			                                          .count());
			g_other_ns[m_kind].fetch_add(ns, std::memory_order_relaxed);
			g_other_calls[m_kind].fetch_add(1, std::memory_order_relaxed);
			NoteThread(m_kind, ns);
			if (g_timeline_hook != nullptr) {
				g_timeline_hook('X', m_kind | (std::min<uint64_t>(ns / 1000u, 0xffffffu) << 8u) |
				                         (static_cast<uint64_t>(gettid()) << 32u));
			}
			return;
		}
		auto& s = GetState();
		s.ns[m_kind] += static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
		                                          std::chrono::steady_clock::now() - m_start)
		                                          .count());
		s.calls[m_kind]++;
		Report();
	}
	Scope(const Scope&)            = delete;
	Scope& operator=(const Scope&) = delete;

private:
	Kind                                  m_kind;
	bool                                  m_on = false;
	std::chrono::steady_clock::time_point m_start {};
};

// Times a part of an outer scope (any depth), other threads only: a breakdown, not in "listed".
class Inner {
public:
	explicit Inner(Kind kind): m_kind(kind) {
		if (!t_cp_thread && On()) {
			m_start = std::chrono::steady_clock::now();
			m_on    = true;
		}
	}
	~Inner() {
		if (m_on) {
			const auto ns = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
			                                          std::chrono::steady_clock::now() - m_start)
			                                          .count());
			g_other_ns[m_kind].fetch_add(ns, std::memory_order_relaxed);
			g_other_calls[m_kind].fetch_add(1, std::memory_order_relaxed);
			NoteThread(m_kind, ns);
		}
	}
	Inner(const Inner&)            = delete;
	Inner& operator=(const Inner&) = delete;

private:
	Kind                                  m_kind;
	bool                                  m_on = false;
	std::chrono::steady_clock::time_point m_start {};
};

// A part of the CP's own work (any depth), CP only, reported with the other threads' parts.
class CpPart {
public:
	explicit CpPart(Kind kind): m_kind(kind) {
		if (t_cp_thread && On()) {
			m_start = std::chrono::steady_clock::now();
			m_on    = true;
		}
	}
	~CpPart() {
		if (m_on) {
			g_other_ns[m_kind].fetch_add(static_cast<uint64_t>(
			                                 std::chrono::duration_cast<std::chrono::nanoseconds>(
			                                     std::chrono::steady_clock::now() - m_start)
			                                     .count()),
			                             std::memory_order_relaxed);
			g_other_calls[m_kind].fetch_add(1, std::memory_order_relaxed);
		}
	}
	CpPart(const CpPart&)            = delete;
	CpPart& operator=(const CpPart&) = delete;

private:
	Kind                                  m_kind;
	bool                                  m_on = false;
	std::chrono::steady_clock::time_point m_start {};
};

} // namespace Common::WaitStats

#endif // EMULATOR_INCLUDE_EMULATOR_COMMON_WAITSTATS_H_
