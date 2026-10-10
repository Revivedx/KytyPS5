#ifndef EMULATOR_INCLUDE_EMULATOR_COMMON_WAITSTATS_H_
#define EMULATOR_INCLUDE_EMULATOR_COMMON_WAITSTATS_H_

#include "common/liveSwitches.h"

#include <array>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <ctime>

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
	Count,
};

inline thread_local bool     t_cp_thread = false; // set by GuestGpu::ThreadRun
inline thread_local uint32_t t_depth     = 0;
// Other threads (the game's): wall time in the same scopes, summed over threads.
inline std::array<std::atomic<uint64_t>, Count> g_other_ns {};
inline std::array<std::atomic<uint64_t>, Count> g_other_calls {};

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
	                                   "    flush-readback", "cp-written-upload-hold"};
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
			g_other_ns[m_kind].fetch_add(static_cast<uint64_t>(
			                                 std::chrono::duration_cast<std::chrono::nanoseconds>(
			                                     std::chrono::steady_clock::now() - m_start)
			                                     .count()),
			                             std::memory_order_relaxed);
			g_other_calls[m_kind].fetch_add(1, std::memory_order_relaxed);
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
			g_other_ns[m_kind].fetch_add(static_cast<uint64_t>(
			                                 std::chrono::duration_cast<std::chrono::nanoseconds>(
			                                     std::chrono::steady_clock::now() - m_start)
			                                     .count()),
			                             std::memory_order_relaxed);
			g_other_calls[m_kind].fetch_add(1, std::memory_order_relaxed);
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
