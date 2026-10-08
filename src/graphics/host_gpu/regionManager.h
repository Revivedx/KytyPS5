#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_REGIONMANAGER_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_REGIONMANAGER_H_

#include "common/assert.h"
#include "common/liveSwitches.h"
#include "graphics/host_gpu/pageManager.h"
#include "graphics/host_gpu/regionDefinitions.h"

#include <cstdlib>
#include <array>
#include <immintrin.h>
#include <thread>
#include <atomic>
#include <mutex>
#include <utility>

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#undef min
#undef max
#undef MemoryBarrier
#elif defined(__APPLE__)
#include <pthread.h>
#elif defined(__linux__)
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace Libs::Graphics {

class TrackingSpinLock final {
public:
	void lock() noexcept {
		const auto thread = CurrentThread();
		// Test-and-test-and-set: spin on a plain load with PAUSE, and yield the CPU once the
		// holder is slow (it may be inside mprotect, which contends on the process mmap lock on
		// Linux), instead of hammering the cache line and starving the holder's core.
		for (uint32_t spins = 0; m_lock.test_and_set(std::memory_order_acquire);) {
			EXIT_NOT_IMPLEMENTED(m_owner.load(std::memory_order_relaxed) == thread);
			while (m_lock.test(std::memory_order_relaxed)) {
				if (++spins < 64) {
					_mm_pause();
				} else {
					std::this_thread::yield();
				}
			}
		}
		m_owner.store(thread, std::memory_order_relaxed);
	}
	void unlock() noexcept {
		EXIT_NOT_IMPLEMENTED(m_owner.load(std::memory_order_relaxed) != CurrentThread());
		m_owner.store(0, std::memory_order_relaxed);
		m_lock.clear(std::memory_order_release);
	}

private:
	static uint32_t CurrentThread() noexcept {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		return GetCurrentThreadId();
#elif defined(__APPLE__)
		// mach thread port is a nonzero per-thread id (0 is the "no owner" sentinel).
		return static_cast<uint32_t>(pthread_mach_thread_np(pthread_self()));
#elif defined(__linux__)
		static thread_local const uint32_t tid = static_cast<uint32_t>(::syscall(SYS_gettid));
		return tid;
#else
		EXIT("region tracking thread identity is unsupported on this platform\n");
#endif
	}

	std::atomic_flag     m_lock = ATOMIC_FLAG_INIT;
	std::atomic_uint32_t m_owner {0};
};

static_assert(std::atomic_uint32_t::is_always_lock_free);

inline std::atomic<uint64_t> g_cpu_dirty_epoch {1};

// Bumped by the GPU thread when it starts a guest submission (its first slice).
inline std::atomic<uint64_t> g_guest_submission_seq {1};

class RegionManager final {
public:
	RegionManager(PageManager& page_manager, uint64_t cpu_addr)
	    : m_page_manager(page_manager), m_cpu_addr(cpu_addr) {
		if (m_cpu_addr % TRACKER_REGION_SIZE != 0) {
			EXIT("invalid region tracking manager construction\n");
		}
		m_cpu_dirty.Fill();
		m_writable.Fill();
		m_readable.Fill();
	}

	KYTY_CLASS_NO_COPY(RegionManager);

	[[nodiscard]] uint64_t GetCpuAddr() const { return m_cpu_addr; }

	template <DirtySource source>
	[[nodiscard]] bool MaybeModified() const noexcept {
		return GetSummary<source>().load(std::memory_order_acquire);
	}
	template <DirtySource source>
	[[nodiscard]] bool IsModified(uint64_t offset, uint64_t size) const {
		const auto [start, end] = GetPageRange(m_cpu_addr + offset, size);
		return GetBits<source>().FirstRangeFrom(start).first < end;
	}

	template <DirtySource source, bool enable>
	void ChangeState(uint64_t vaddr, uint64_t size) {
		const auto [start, end] = GetPageRange(vaddr, size);
		// KYTY_LOCAL_HACK KYTY_REGION_FAST (live, default 1): word-wise range checks instead of a
		// masked copy of the region's bit array; a GPU write over pages already GPU-dirty (the
		// same written binding draw after draw) changes nothing; the heat walk only while hot
		// pages are on.
		static auto& region_fast = Common::LiveSwitches::Get("KYTY_REGION_FAST", 1);
		const bool   fast        = region_fast.load(std::memory_order_relaxed) != 0;
		if constexpr (source == DirtySource::Cpu && enable) {
			if (fast ? m_gpu_dirty.AnyInRange(start, end) : RegionBits(m_gpu_dirty, start, end).Any()) {
				EXIT("CPU dirty state conflicts with GPU dirty state\n");
			}
		}
		if constexpr (source == DirtySource::Gpu && enable) {
			if (fast && HotPageThreshold() == 0 && m_gpu_dirty.AllInRange(start, end)) {
				// Already GPU-dirty: the CPU-dirty conflict check below held when they were set,
				// and protection was updated then. Only the summary is restated.
				SetSummary<source>(true);
				return;
			}
			// A hot page was uploaded before this GPU write; it stops being kept CPU-dirty.
			bool cooled = false;
			for (size_t page = start; page < end && (!fast || HotPageThreshold() != 0); ++page) {
				if (m_heat[page] != 0 && IsHot(page)) {
					m_cpu_dirty.UnsetRange(page, page + 1);
					cooled = true;
				}
				m_heat[page] = 0;
			}
			if (cooled) {
				// Write-protect the cooled pages now: a later UpdateProtection of the opposite
				// direction would otherwise see them in its mask and drop a watcher never added.
				if (m_cpu_dirty.None()) {
					SetSummary<DirtySource::Cpu>(false);
				}
				UpdateProtection<true, false>();
			}
			if (fast ? m_cpu_dirty.AnyInRange(start, end) : RegionBits(m_cpu_dirty, start, end).Any()) {
				EXIT("GPU dirty state conflicts with CPU dirty state\n");
			}
		}
		auto& bits = GetBits<source>();
		if constexpr (enable) {
			bits.SetRange(start, end);
			SetSummary<source>(true);
			if constexpr (source == DirtySource::Cpu) {
				g_cpu_dirty_epoch.fetch_add(1, std::memory_order_release);
				for (size_t page = start; page < end; ++page) {
					if (m_heat[page] != 0xff) ++m_heat[page];
				}
			}
		} else {
			bits.UnsetRange(start, end);
			if (bits.None()) {
				SetSummary<source>(false);
			}
		}
		if constexpr (source == DirtySource::Cpu) {
			UpdateProtection<!enable, false>();
		} else {
			UpdateProtection<enable, true>();
		}
	}

	template <DirtySource source, bool clear, typename Func>
	void ForEachModifiedRange(uint64_t vaddr, uint64_t size, Func&& func) {
		const auto [start, end] = GetPageRange(vaddr, size);
		auto&      bits         = GetBits<source>();
		if (bits.FirstRangeFrom(start).first >= end) {
			return;
		}
		RegionBits mask(bits, start, end);
		if constexpr (clear) {
			bits.UnsetRange(start, end);
			if constexpr (source == DirtySource::Cpu) {
				for (const auto [first, last]: mask) {
					for (size_t page = first; page < last; ++page) {
						if (IsHot(page)) bits.Set(page);
					}
				}
			}
			if (bits.None()) {
				SetSummary<source>(false);
			}
			if constexpr (source == DirtySource::Cpu) {
				UpdateProtection<true, false>();
			} else {
				UpdateProtection<false, true>();
			}
		}
		for (const auto [first, last]: mask) {
			func(m_cpu_addr + first * TRACKER_PAGE_SIZE, (last - first) * TRACKER_PAGE_SIZE);
		}
	}

	// Mirrors the CPU summary into one bit of the tracker's region bitmap, so range walks can skip
	// clean regions without touching their managers (MemoryTracker, KYTY_REGION_BITMAP). Set
	// before the manager is published.
	void SetCpuSummaryBit(std::atomic<uint64_t>* word, uint64_t bit) noexcept {
		m_cpu_summary_word = word;
		m_cpu_summary_bit  = bit;
		SetSummary<DirtySource::Cpu>(m_cpu_maybe_dirty.load(std::memory_order_relaxed));
	}

	TrackingSpinLock lock;

private:
	// Pages the CPU dirties again and again (per-frame constants) cost one write fault and two
	// protection changes per frame. Past this many CPU dirtyings a page stays CPU-dirty: it keeps
	// write access and is uploaded whenever the GPU uses it. KYTY_HOT_PAGES=0 turns this off.
	// KYTY_HOT_PAGES is live: lowering it (to 0: off) is safe at any time, the pages it releases
	// are write-protected again at their next upload.
	static uint32_t HotPageThreshold() noexcept {
		static auto& threshold = Common::LiveSwitches::Get("KYTY_HOT_PAGES", 8);
		return static_cast<uint32_t>(threshold.load(std::memory_order_relaxed));
	}
	bool IsHot(size_t page) const noexcept {
		const auto threshold = HotPageThreshold();
		return threshold != 0 && m_heat[page] >= threshold;
	}

	template <DirtySource source>
	void SetSummary(bool value) noexcept {
		GetSummary<source>().store(value, std::memory_order_release);
		if constexpr (source == DirtySource::Cpu) {
			if (m_cpu_summary_word != nullptr) {
				if (value) {
					m_cpu_summary_word->fetch_or(m_cpu_summary_bit, std::memory_order_release);
				} else {
					m_cpu_summary_word->fetch_and(~m_cpu_summary_bit, std::memory_order_release);
				}
			}
		}
	}

	template <bool track, bool is_read>
	void UpdateProtection() {
		const auto protection = is_read ? ~m_gpu_dirty : m_cpu_dirty;
		auto&      previous   = is_read ? m_readable : m_writable;
		auto       mask       = protection ^ previous;
		if (mask.None()) {
			return;
		}
		previous = protection;
		m_page_manager.UpdatePageWatchersForRegion<track, is_read>(m_cpu_addr, mask);
	}

	template <DirtySource source>
	RegionBits& GetBits() {
		if constexpr (source == DirtySource::Cpu) {
			return m_cpu_dirty;
		} else {
			return m_gpu_dirty;
		}
	}

	template <DirtySource source>
	const RegionBits& GetBits() const {
		if constexpr (source == DirtySource::Cpu) {
			return m_cpu_dirty;
		} else {
			return m_gpu_dirty;
		}
	}

	template <DirtySource source>
	std::atomic<bool>& GetSummary() {
		if constexpr (source == DirtySource::Cpu) {
			return m_cpu_maybe_dirty;
		} else {
			return m_gpu_maybe_dirty;
		}
	}

	template <DirtySource source>
	const std::atomic<bool>& GetSummary() const {
		if constexpr (source == DirtySource::Cpu) {
			return m_cpu_maybe_dirty;
		} else {
			return m_gpu_maybe_dirty;
		}
	}

	[[nodiscard]] std::pair<size_t, size_t> GetPageRange(uint64_t vaddr, uint64_t size) const {
		if (size == 0 || vaddr < m_cpu_addr || vaddr >= m_cpu_addr + TRACKER_REGION_SIZE ||
		    size > m_cpu_addr + TRACKER_REGION_SIZE - vaddr) {
			EXIT("range lies outside its tracking region\n");
		}
		const auto offset = vaddr - m_cpu_addr;
		return {static_cast<size_t>(offset / TRACKER_PAGE_SIZE),
		        static_cast<size_t>((offset + size + TRACKER_PAGE_SIZE - 1) / TRACKER_PAGE_SIZE)};
	}

	PageManager& m_page_manager;
	uint64_t     m_cpu_addr = 0;
	RegionBits   m_cpu_dirty;
	RegionBits   m_gpu_dirty;
	std::array<uint8_t, TRACKER_REGION_PAGES> m_heat {};
	std::atomic<bool> m_cpu_maybe_dirty {true};
	std::atomic<bool> m_gpu_maybe_dirty {false};
	std::atomic<uint64_t>* m_cpu_summary_word = nullptr;
	uint64_t               m_cpu_summary_bit  = 0;
	RegionBits   m_writable;
	RegionBits   m_readable;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_REGIONMANAGER_H_
