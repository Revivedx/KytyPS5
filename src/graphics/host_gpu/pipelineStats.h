#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_PIPELINESTATS_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_PIPELINESTATS_H_

#include <array>
#include <atomic>
#include <cstdint>
#include <utility>

namespace Libs::Graphics::HW {
class Shader;
} // namespace Libs::Graphics::HW

// KYTY_LOCAL_HACK KYTY_LOOKAHEAD (live): while the CP runs a draw, the shader registers the next
// draw will see (predicted from the PM4 ahead, graphicsRun.cpp), or null. The pipeline cache
// materializes that draw's stages on its worker into memo slots (ProgramCache::LookaheadStart).
namespace Libs::Graphics::Lookahead {
inline const HW::Shader* g_next         = nullptr;
inline bool              g_next_trusted = false; // no memory-writing packet was skipped

// GPU writes made while the worker evaluates the next draw (any thread): a logged read inside
// one is not trusted (KYTY_LOOKAHEAD=2 adopts the other results without re-reading them).
constexpr size_t                                    WatchSize = 64;
inline std::atomic<bool>                            g_watch {false};
inline std::atomic_flag                             g_watch_lock;
inline std::array<std::pair<uint64_t, uint64_t>, WatchSize> g_writes {};
inline size_t                                       g_write_count = 0; // > WatchSize: overflow
// Diagnostics (KYTY_LOOKAHEAD=3): guest memory writes made by the CP itself (WRITE_DATA, DMA_DATA,
// DUMP_CONST_RAM) and commands run from other threads (GuestGpu::ProcessCommands), GPU thread only.
inline uint64_t g_cp_writes = 0;
inline uint64_t g_commands  = 0;

inline void NoteWrite(uint64_t vaddr, uint64_t size) noexcept {
	if (!g_watch.load(std::memory_order_acquire) || size == 0) return;
	while (g_watch_lock.test_and_set(std::memory_order_acquire)) {
		__builtin_ia32_pause();
	}
	if (g_write_count < WatchSize) {
		g_writes[g_write_count] = {vaddr, vaddr + size};
	}
	g_write_count++;
	g_watch_lock.clear(std::memory_order_release);
}
} // namespace Libs::Graphics::Lookahead

// KYTY_LOCAL_HACK KYTY_PIPELINE_STATS (research): what a pipelined command processor (a thread that
// prepares the next draws' resources while the current one is bound and recorded) would face.
// - op: a draw or dispatch; ops are numbered on the GPU thread.
// - a write: a range an op (or a DMA/fill) makes GPU-written, tagged with the op number.
// - a hazard: a resource read by op N inside a range written by op M with N - M <= Window, since
//   the last sync packet: the read would run before the write was even recorded.
namespace Libs::Graphics::PipelineStats {

constexpr uint64_t Window    = 8;
constexpr size_t   RingSize  = 256;

struct Write {
	uint64_t begin = 0;
	uint64_t end   = 0;
	uint64_t op    = 0;
};

inline bool                         g_on        = false; // set by the CP each packet batch
inline uint64_t                     g_op        = 0;     // current op number (GPU thread)
inline uint64_t                     g_sync_op   = 0;     // op number at the last sync packet
inline std::array<Write, RingSize>  g_writes {};
inline size_t                       g_next      = 0;
inline uint64_t                     g_hazard_reads = 0;
inline uint64_t                     g_hazard_ops   = 0;
inline uint64_t                     g_last_hazard_op = 0;

inline void NoteWrite(uint64_t vaddr, uint64_t size) noexcept {
	Lookahead::NoteWrite(vaddr, size);
	if (!g_on || size == 0) return;
	g_writes[g_next++ % RingSize] = {vaddr, vaddr + size, g_op};
}

inline void NoteRead(uint64_t vaddr, uint64_t size) noexcept {
	if (!g_on) return;
	for (const auto& w: g_writes) {
		if (w.end == 0 || w.op >= g_op || w.op < g_sync_op || g_op - w.op > Window) continue;
		if (vaddr < w.end && w.begin < vaddr + size) {
			g_hazard_reads++;
			if (g_last_hazard_op != g_op) {
				g_last_hazard_op = g_op;
				g_hazard_ops++;
			}
			return;
		}
	}
}

} // namespace Libs::Graphics::PipelineStats

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_PIPELINESTATS_H_
