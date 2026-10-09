#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DRAWPREP_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DRAWPREP_H_

#include <cstdint>

// KYTY_LOCAL_HACK KYTY_DRAW_PREP (live, default 1; port of Senaxx 6632a241 onto this fork's resource
// memo): a scanner thread (guest_gpu/drawPrepScanner.cpp) reads every graphics and compute
// submission as soon as it is queued, follows its register writes on command processors of its
// own, and for each draw or dispatch evaluates the resource plans of its stages with the worker's
// clean-only reader, logging every read (a memo slot). The GPU thread executes exactly as before:
// at the draw or dispatch the scanner prepared, it stores the prepared slots in their programs'
// memo, and Materialize uses one only as a memo hit (same user data and shader base, every logged
// read returns the same words), else evaluates. 1 = draws and dispatches, 2 = draws only.
// With it on, KYTY_PARALLEL_MATERIALIZE=1 runs no worker (dp4: same fps, the worker spun a core).

namespace Libs::Graphics::DrawPrep {

[[nodiscard]] int Mode(); // 0 off, 1 draws + dispatches, 2 draws only

constexpr uint32_t MaxQueues = 64;

// A draw or dispatch packet: its queue, its submission (numbered at enqueue), its position among
// the draw and dispatch packets of that submission's command stream, and its address.
struct Key {
	uint32_t        queue   = 0;
	uint64_t        seq     = 0;
	uint32_t        ordinal = 0;
	const uint32_t* packet  = nullptr;

	[[nodiscard]] bool operator==(const Key&) const = default;
	[[nodiscard]] bool Before(const Key& other) const {
		return seq < other.seq || (seq == other.seq && ordinal < other.ordinal);
	}
};

[[nodiscard]] bool CountsAsDrawOrDispatch(uint32_t opcode);

// GPU thread: the draw or dispatch packet being executed (null outside numbered submissions).
void                     SetCurrent(const Key& key);
void                     ClearCurrent();
[[nodiscard]] const Key* Current();
[[nodiscard]] Key        GpuProgress(uint32_t queue);

// Programs the GPU thread prepared successfully (vertex identity, pixel address; compute: shader
// address, UINT64_MAX): the scanner only prepares these.
void               NoteGoodPair(uint64_t vs, uint64_t ps);
[[nodiscard]] bool IsGoodPair(uint64_t vs, uint64_t ps);

enum class Counter : uint32_t {
	Scanned,
	Prepared,
	SkipBehind,
	SkipUnknownPair,
	SkipUnsupported,
	SkipNoEntry,
	FailWalk,
	Tainted,
	TaintIndirect,
	TaintContext,
	Dropped,       // results the store had no room for
	Takes,         // GPU thread asked for the current packet
	Adopted,       // prepared stages stored as memo slots
	MissNone,      // no result for the current packet
	MissInputs,    // user data or shader base differ
	Stale,         // results of packets the GPU thread passed without asking
	AheadHits,     // memo hits on adopted slots
	Count,
};
void Add(Counter counter, uint64_t value = 1);
void LogCountersIfDue();

} // namespace Libs::Graphics::DrawPrep

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DRAWPREP_H_
