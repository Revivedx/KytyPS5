#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_BINDLESSTABLE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_BINDLESSTABLE_H_

#include "common/common.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/cache/streamBuffer.h"
#include "graphics/host_gpu/renderer/image/image.h"
#include "graphics/shader/recompiler/ir/BindlessBindings.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"

#include <array>
#include <atomic>
#include <deque>
#include <memory>
#include <mutex>
#include <span>
#include <unordered_map>
#include <vector>
#include <vulkan/vulkan.hpp>

namespace Libs::Graphics {

struct GraphicContext;
class CommandScheduler;
class PageManager;
class SamplerCache;

// KYTY_LOCAL_HACK KYTY_BINDLESS_ENGINE (env, default 1, 2026-10-11; port of Senaxx c5a6961b, the
// write-watch part "E2"): the guest pages holding a heap's entries are write-watched; at the start
// of each guest submission the entries in pages the guest wrote are compared with the T# their
// keys were settled from, and changed keys are settled again before the submission's draws (on
// the console the guest's writes before a submit are visible to all of it). Replaces the rolling
// RevalidateBindlessKeys scan (2048 keys a frame: up to 32 frames to notice a rewrite, sampling
// the old texture or the placeholder meanwhile) for watched heaps; heaps in host-imported zones
// (no write faults there) keep the scan. 0 = scan only, as before.
[[nodiscard]] bool BindlessEngineEnabled();

// Descriptor set 1 of every pipeline that samples bindless images: typed arrays of sampled
// images a shader indexes with a slot it looks up per pixel, the key -> slot translation the
// host fills per guest descriptor heap, and a bitset where shaders flag keys whose texture is
// not resident yet. Created only when the device supports descriptor indexing.
class BindlessTable {
public:
	enum Binding : uint32_t {
		Images2D      = 0,
		Images2DArray = 1,
		ImagesCube    = 2,
		Images3D      = 3,
		Translation   = 4,
		Feedback      = 5,
		Samplers      = 6,
	};
	static_assert(Images2D == ShaderRecompiler::IR::BindlessImages2D &&
	              Images2DArray == ShaderRecompiler::IR::BindlessImages2DArray &&
	              ImagesCube == ShaderRecompiler::IR::BindlessImagesCube &&
	              Images3D == ShaderRecompiler::IR::BindlessImages3D &&
	              Translation == ShaderRecompiler::IR::BindlessTranslation &&
	              Feedback == ShaderRecompiler::IR::BindlessFeedback &&
	              Samplers == ShaderRecompiler::IR::BindlessSamplers);
	static constexpr uint32_t ImageArrays        = 4;
	// Slots 0-2: not resident (grey), outside the heap (red), pending (blue); slot 3 (green): what a
	// slot of an image the texture cache dropped is repointed to (KYTY_BINDLESS_DEBUG_COLORS).
	static constexpr uint32_t PlaceholderColors  = 4;
	static constexpr uint32_t DroppedPlaceholder = 3;
	static constexpr uint32_t Placeholders       = ImageArrays * PlaceholderColors;
	static constexpr uint32_t MaxImagesPerArray  = 16384;
	static constexpr uint32_t TranslationEntries = 1u << 20u;
	static constexpr uint32_t MaxSamplers        = 4096;

	BindlessTable(GraphicContext& graphics, CommandScheduler& scheduler, PageManager& page_manager);
	~BindlessTable();
	KYTY_CLASS_NO_COPY(BindlessTable);

	[[nodiscard]] bool                    Enabled() const noexcept { return m_set != nullptr; }
	[[nodiscard]] vk::DescriptorSetLayout Layout() const noexcept { return m_layout; }
	[[nodiscard]] vk::DescriptorSet       Set() const noexcept { return m_set; }
	[[nodiscard]] uint32_t                ImagesPerArray() const noexcept { return m_images_per_array; }

	// A guest descriptor heap as one view type samples it: a region of the translation (and
	// feedback) buffer, one entry per key, and the textures resolved so far.
	struct Heap {
		uint64_t                            base         = 0;
		uint32_t                            table_offset = 0;
		uint32_t                            binding      = 0;
		uint32_t                            region       = 0;
		// Keys the region holds. The guest appends textures to a heap as it streams them in and
		// its descriptor grows; each draw patches its own descriptor's entry count.
		uint32_t                            entries      = 0;
		ShaderRecompiler::IR::ImageResource resource;
		std::vector<uint32_t>               slots;    // per key; 0 = not resolved
		std::vector<uint8_t>                settled;  // per key; resolved, or known placeholder
		std::vector<ImageId>                resolved; // images to keep readable for draws
		std::vector<ImageId>                images;   // per key; the image a resolved key samples
		// Per settled key, the T# it was settled from. The guest rewrites entries as it streams
		// textures out and others in; RenderExecutor::RevalidateBindlessKeys settles a key whose
		// T# changed again (Senaxx 0563e10e).
		std::vector<std::array<uint32_t, 8>> descriptors;
		uint32_t                             revalidate_cursor = 0; // next key to compare
		// What RenderExecutor::CommitBindings last checked: g_bindless_state_generation then, and
		// how many resolved images. Unchanged since, only the images resolved after need a look.
		uint64_t checked_generation = 0;
		size_t   checked_count      = 0;
		// Engine method: the most keys a draw's descriptor declared (the guest's heap; entries
		// includes headroom), and what the write watch covers.
		uint32_t guest_entries    = 0;
		uint32_t watched_entries  = 0;
		uint64_t watch_generation = 0;
		bool     watch_excluded   = false; // in a host-imported zone: the scan keeps it
	};

	// The heap for (base, table offset, view binding), created with every entry pending and
	// room to grow, or moved to a larger region when a descriptor covers more keys; null when
	// the translation buffer is full.
	[[nodiscard]] Heap* FindOrCreateHeap(uint64_t base, uint32_t table_offset, uint32_t binding,
	                                     uint32_t entries,
	                                     const ShaderRecompiler::IR::ImageResource& resource);
	[[nodiscard]] std::deque<Heap>& Heaps() noexcept { return m_heaps; }
	// 0 when the array is full. Slots of dropped images are reused once the GPU is past them.
	[[nodiscard]] uint32_t AllocateSlot(uint32_t binding);
	// Slots handed out so far and slots waiting in the free list, per image array.
	[[nodiscard]] uint32_t SlotsUsed(uint32_t binding) const noexcept { return m_next_slot[binding]; }
	[[nodiscard]] size_t   SlotsFree(uint32_t binding) const noexcept { return m_free_slots[binding].size(); }
	void WriteSlot(uint32_t binding, uint32_t slot, vk::ImageView view, vk::ImageLayout layout);
	// Resolve a key to a slot (0 = placeholder), or back to pending.
	void SetTranslation(const Heap& heap, uint32_t key, uint32_t slot);
	// KYTY_BINDLESS_ORDERED (live, default 1): SetTranslation only queues the change, and this
	// records the queued changes into the command stream (before the next bindless draw), so
	// work recorded earlier keeps the mapping it was recorded with. Called by RenderExecutor.
	void FlushTranslations();
	// KYTY_BINDLESS_EAGER (live, default 1): keys whose image the texture cache dropped (merged
	// into a bigger image as the guest streams mips, re-created, unmapped) are resolved again at
	// the next bindless draw instead of waiting for shader feedback a few frames later.
	void TakeEagerKeys(std::vector<std::pair<Heap*, uint32_t>>& keys) { keys.swap(m_eager_keys); }
	// The host reads the feedback flags from a copy in host-cached memory. The feedback buffer
	// itself is device memory, and reading every key of three heaps from it uncached took a
	// fifth of the GPU thread. Records that copy at the current point of the command stream.
	void RecordFeedbackSnapshot(CommandScheduler& scheduler);
	[[nodiscard]] bool SnapshotRecorded() const noexcept { return m_snapshot_tick != 0; }
	// Whether the GPU has executed the recorded copy; never waits.
	[[nodiscard]] bool SnapshotReady(CommandScheduler& scheduler);
	void               ConsumeSnapshot() noexcept { m_snapshot_tick = 0; }
	// Pending keys flagged in the snapshot, and the images of resolved keys sampled since the last
	// call (shaders mark both); their flags are cleared.
	void TakeRequests(const Heap& heap, std::vector<uint32_t>& keys, std::vector<ImageId>& used);
	// Logs the loop-watchdog trip reports shaders appended since the last call
	// (BindlessBindings.h), then clears the count.
	void DrainWatchdogReports(uint64_t frame);
	void AddImageReference(ImageId id, Heap& heap, uint32_t key);
	// The key no longer samples what it was settled to: it is pending again, and its image loses
	// the key's reference. The slot keeps its view for the frames in flight and is reused once
	// the GPU is past them. True when the image has no reference left (it need not stay pinned).
	[[nodiscard]] bool ReleaseKey(Heap& heap, uint32_t key);

	// A guest sampler heap as one kind of use samples it: a region of the sampler array that
	// mirrors its S# records, key for key. Uses differ in whether the depth-compare function is
	// kept, whether point filtering is forced and whether border colours are integer, so each
	// kind gets its own region.
	static constexpr uint32_t SamplerDepthCompare   = 1u;
	static constexpr uint32_t SamplerPointFiltering = 2u;
	static constexpr uint32_t SamplerIntegerBorder  = 4u;
	struct SamplerHeap {
		uint64_t                             base          = 0;
		uint32_t                             table_offset  = 0;
		uint32_t                             flags         = 0;
		uint32_t                             region        = 0; // first slot; 0 = none yet
		uint32_t                             capacity      = 0;
		std::vector<std::array<uint32_t, 4>> records;           // mirrored S# records
		uint64_t                             checked_frame = UINT64_MAX;
	};
	[[nodiscard]] SamplerHeap* FindOrCreateSamplerHeap(uint64_t base, uint32_t table_offset,
	                                                   uint32_t flags);
	// Mirrors the heap's records into its region. New records are appended in place (their slots
	// were never used); a changed record, or more records than the region holds, moves the heap
	// to a new region, because slots the GPU may be reading are never rewritten. False when the
	// sampler array is full; the heap then keeps its previous records.
	bool MirrorSamplerHeap(SamplerHeap& heap, std::span<const std::array<uint32_t, 4>> records,
	                       SamplerCache& cache);
	// Slot 0 of the sampler array, used for keys outside their heap; written once.
	void WriteDefaultSampler(vk::Sampler sampler);
	[[nodiscard]] bool SamplersEnabled() const noexcept { return m_samplers_per_array != 0; }
	// The texture cache dropped a resolved image: point its slots back at the placeholder and
	// make its keys pending again, so a draw that still needs it asks for it anew.
	void OnImageUnregistered(ImageId id);
	// Device-loss triage (addressBindingReport.h): the slots holding a view of the image.
	static void ReportImageUsers(uint64_t image);

	// Engine method (write watch). NeedsWatch: the heap grew, or its watch was dropped by an
	// unmap, since WatchHeap last covered it.
	[[nodiscard]] bool NeedsWatch(const Heap& heap) const noexcept {
		return heap.watched_entries != heap.guest_entries ||
		       heap.watch_generation != m_watch_generation.load(std::memory_order_acquire);
	}
	void WatchHeap(Heap& heap);
	// A CPU write fault or an emulator write: stop watching the written heap pages (the write
	// must go through) and record them. Cheap outside the watched span.
	void OnCpuWrite(uint64_t vaddr, uint64_t size);
	void UnwatchRange(uint64_t vaddr, uint64_t size);
	[[nodiscard]] bool HasWrittenRanges() const noexcept {
		return m_heap_writes.load(std::memory_order_acquire);
	}
	// The written pages since the last call, watched again before this returns.
	void TakeWrittenRanges(std::vector<std::pair<uint64_t, uint64_t>>& ranges);

private:
	[[nodiscard]] bool AllocateRegion(Heap& heap, uint32_t entries);
public:

private:
	// Slot 0 of every image array: a 1x1 grey texture of that view type, sampled for keys
	// outside a heap and while a texture is pending. Created, cleared and made read-only once.
	void CreatePlaceholders(CommandScheduler& scheduler);

	[[nodiscard]] static uint64_t ImageKey(ImageId id) {
		return static_cast<uint64_t>(id.index) | (static_cast<uint64_t>(id.generation) << 32u);
	}

	GraphicContext&         m_graphics;
	CommandScheduler&       m_scheduler;
	PageManager&            m_page_manager;
	std::deque<Heap>        m_heaps;
	struct Watch {
		uint64_t             begin = 0;
		uint64_t             end   = 0;
		std::vector<uint8_t> watched; // per page
		std::vector<uint8_t> written; // per page
	};
	std::mutex            m_watch_mutex;
	std::vector<Watch>    m_watches;
	std::atomic<uint64_t> m_watch_low {UINT64_MAX};
	std::atomic<uint64_t> m_watch_high {0};
	std::atomic<uint64_t> m_watch_generation {1};
	std::atomic<bool>     m_heap_writes {false};
	std::unordered_map<uint64_t, std::vector<std::pair<Heap*, uint32_t>>> m_image_refs;
	uint32_t                m_next_region = 1; // translation[0] is the out-of-range entry
	std::array<uint32_t, ImageArrays> m_next_slot {PlaceholderColors, PlaceholderColors,
	                                               PlaceholderColors, PlaceholderColors};
	// Slots of unregistered images per array, with the tick after which no command buffer reads
	// them.
	std::array<std::deque<std::pair<uint64_t, uint32_t>>, ImageArrays> m_free_slots;
	// What each slot's descriptor holds now (WriteSlot), for device-loss triage.
	std::array<std::vector<vk::ImageView>, ImageArrays> m_slot_views;
	// The image each slot's view belonged to when it was written: a view handle the driver
	// reuses for another image after the first is destroyed would otherwise hide the slot.
	std::array<std::vector<uint64_t>, ImageArrays> m_slot_view_images;
	std::array<VulkanImage, Placeholders>   m_placeholders;
	std::array<vk::ImageView, Placeholders> m_placeholder_views {};
	vk::DescriptorPool      m_pool   = nullptr;
	vk::DescriptorSetLayout m_layout = nullptr;
	vk::DescriptorSet       m_set    = nullptr;
	uint32_t                m_images_per_array = 0;
	uint32_t                m_samplers_per_array = 0;
	uint32_t                m_next_sampler_slot  = 1; // slot 0 is the default sampler
	bool                    m_default_sampler_written = false;
	std::deque<SamplerHeap> m_sampler_heaps;
	std::unique_ptr<Buffer> m_translation;
	std::vector<std::pair<uint32_t, uint32_t>> m_pending_translations; // (entry, value)
	struct QueuedFree {
		uint32_t binding     = 0;
		uint32_t slot        = 0;
		bool     placeholder = false; // a dropped image's slot: repoint it before reuse
	};
	std::vector<QueuedFree> m_queued_frees; // released with the next FlushTranslations
	void FreeSlot(uint32_t binding, uint32_t slot, bool placeholder);
	void ReleaseQueuedSlots();
	std::vector<std::pair<Heap*, uint32_t>>    m_eager_keys;
	std::unique_ptr<Buffer> m_feedback;
	std::unique_ptr<Buffer> m_feedback_snapshot;
	uint64_t                m_snapshot_tick = 0; // 0 = none recorded
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_BINDLESSTABLE_H_
