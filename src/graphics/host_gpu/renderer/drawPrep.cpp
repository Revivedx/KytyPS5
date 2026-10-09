#include "graphics/host_gpu/renderer/drawPrep.h"

#include "common/liveSwitches.h"
#include "graphics/guest_gpu/pm4.h"

#include <array>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <mutex>
#include <shared_mutex>
#include <unordered_set>

namespace Libs::Graphics::DrawPrep {

namespace {

thread_local Key  t_current {};
thread_local bool t_has_current = false;
// GPU thread progress per queue, packed: submission (40 bits), ordinal (24 bits).
std::array<std::atomic<uint64_t>, MaxQueues> g_progress {};

std::array<std::atomic<uint64_t>, static_cast<size_t>(Counter::Count)> g_counters {};

struct GoodPairs {
	std::shared_mutex            mutex;
	std::unordered_set<uint64_t> pairs;
};

GoodPairs& GetGoodPairs() {
	static GoodPairs pairs;
	return pairs;
}

uint64_t PairKey(uint64_t vs, uint64_t ps) {
	return (vs >> 8u) * 0x9e3779b97f4a7c15ull ^ (ps >> 8u);
}

} // namespace

int Mode() {
	// Default 1 since 2026-10-09: dp3 (0 -> 1 -> 0) fps 22.98 / 25.03 / 22.76, CP 37.8 / 33.8 / 38.2 ms.
	static auto& mode = Common::LiveSwitches::Get("KYTY_DRAW_PREP", 1);
	return static_cast<int>(mode.load(std::memory_order_relaxed));
}

bool CountsAsDrawOrDispatch(uint32_t opcode) {
	switch (opcode) {
		case Pm4::IT_DRAW_INDEX_2:
		case Pm4::IT_DRAW_INDEX_OFFSET_2:
		case Pm4::IT_DRAW_INDEX_AUTO:
		case Pm4::IT_DISPATCH_DRAW_PREAMBLE:
		case Pm4::IT_DRAW_INDIRECT:
		case Pm4::IT_DRAW_INDEX_INDIRECT:
		case Pm4::IT_DRAW_INDIRECT_MULTI:
		case Pm4::IT_DRAW_INDEX_INDIRECT_MULTI:
		case Pm4::IT_DISPATCH_DRAW:
		case Pm4::IT_DISPATCH_DIRECT:
		case Pm4::IT_DISPATCH_INDIRECT: return true;
		default: return false;
	}
}

void SetCurrent(const Key& key) {
	t_current     = key;
	t_has_current = true;
	g_progress[key.queue % MaxQueues].store((key.seq << 24u) | (key.ordinal & 0xffffffu),
	                                        std::memory_order_relaxed);
}

void ClearCurrent() {
	t_has_current = false;
}

const Key* Current() {
	return t_has_current ? &t_current : nullptr;
}

Key GpuProgress(uint32_t queue) {
	const auto packed = g_progress[queue % MaxQueues].load(std::memory_order_relaxed);
	return {.queue = queue, .seq = packed >> 24u, .ordinal = static_cast<uint32_t>(packed & 0xffffffu)};
}

void NoteGoodPair(uint64_t vs, uint64_t ps) {
	auto&      pairs = GetGoodPairs();
	const auto key   = PairKey(vs, ps);
	{
		std::shared_lock lock(pairs.mutex);
		if (pairs.pairs.contains(key)) {
			return;
		}
	}
	std::unique_lock lock(pairs.mutex);
	pairs.pairs.insert(key);
}

bool IsGoodPair(uint64_t vs, uint64_t ps) {
	auto&            pairs = GetGoodPairs();
	std::shared_lock lock(pairs.mutex);
	return pairs.pairs.contains(PairKey(vs, ps));
}

void Add(Counter counter, uint64_t value) {
	g_counters[static_cast<size_t>(counter)].fetch_add(value, std::memory_order_relaxed);
}

void LogCountersIfDue() {
	static std::atomic<int64_t> last {0};
	const auto now = std::chrono::duration_cast<std::chrono::seconds>(
	                     std::chrono::steady_clock::now().time_since_epoch())
	                     .count();
	auto previous = last.load(std::memory_order_relaxed);
	if (now - previous < 5 || !last.compare_exchange_strong(previous, now)) {
		return;
	}
	std::array<uint64_t, static_cast<size_t>(Counter::Count)> c {};
	for (size_t i = 0; i < c.size(); i++) {
		c[i] = g_counters[i].exchange(0, std::memory_order_relaxed);
	}
	using enum Counter;
	const auto at = [&](Counter k) { return static_cast<unsigned long long>(c[static_cast<size_t>(k)]); };
	std::printf("DrawPrep (5 s): scanned %llu, prepared %llu stages, skip behind %llu / unknown pair %llu / "
	            "unsupported %llu / no entry %llu, walk failed %llu, tainted %llu (indirect %llu, context %llu), "
	            "dropped %llu; GPU thread: takes %llu, adopted %llu stages, no result %llu, inputs differ %llu, "
	            "stale %llu, ahead hits %llu\n",
	            at(Scanned), at(Prepared), at(SkipBehind), at(SkipUnknownPair), at(SkipUnsupported),
	            at(SkipNoEntry), at(FailWalk), at(Tainted), at(TaintIndirect), at(TaintContext), at(Dropped),
	            at(Takes), at(Adopted), at(MissNone), at(MissInputs), at(Stale), at(AheadHits));
	std::fflush(stdout);
}

} // namespace Libs::Graphics::DrawPrep
