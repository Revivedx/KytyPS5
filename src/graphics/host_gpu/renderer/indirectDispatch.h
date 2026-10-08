#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_INDIRECTDISPATCH_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_INDIRECTDISPATCH_H_

#include "common/common.h"
#include "graphics/host_gpu/renderer/cache/streamBuffer.h"

#include <array>
#include <span>
#include <vulkan/vulkan.hpp>

namespace Libs::Graphics {

struct GraphicContext;
class CommandScheduler;
class CommandRecorder;

// DISPATCH_INDIRECT with the thread-dimension initiator carries thread counts, not workgroup
// counts. The command processor used to read them on the CPU, but the previous dispatch writes
// them on the GPU, so every read drained the GPU; the title issues about 200 of these a frame.
// This records a one-invocation pass that converts the counts on the GPU instead.
class IndirectDispatchGroups {
public:
	IndirectDispatchGroups(GraphicContext& graphics, CommandScheduler& scheduler);
	~IndirectDispatchGroups();
	KYTY_CLASS_NO_COPY(IndirectDispatchGroups);

	struct Result {
		vk::Buffer        groups_buffer;
		vk::DeviceSize    groups_offset = 0;
		// A copy of the thread counts that stays valid until the dispatch has run, unlike the
		// cache buffer the guest's arguments live in, which a later binding may merge away.
		vk::DeviceAddress threads = 0;
	};
	// Records the conversion of the three thread counts at `threads` and returns the workgroup
	// counts' location, ready for dispatchIndirect. Binds a compute pipeline and push state, so
	// call it before committing the guest dispatch's bindings.
	[[nodiscard]] Result Convert(const CommandRecorder& command, vk::DeviceAddress threads,
	                             const std::array<uint32_t, 3>& local_size);

private:
	// Ring of converted counts. An entry is rewritten only after 4096 later conversions, each
	// ordered behind the indirect reads before it.
	static constexpr uint32_t Entries      = 4096;
	static constexpr uint32_t EntryDwords  = 8; // groups at 0, thread counts at 4

	GraphicContext&         m_graphics;
	Buffer                  m_groups;
	vk::DescriptorSetLayout m_set_layout      = nullptr;
	vk::PipelineLayout      m_pipeline_layout = nullptr;
	vk::Pipeline            m_pipeline        = nullptr;
	uint32_t                m_next            = 0;
};

// Indirect draws through a mesh-shader pipeline: converts the guest's draw arguments into mesh
// workgroup counts and the program's six draw parameters on the GPU (mesh_indirect_draw.comp),
// instead of reading arguments an earlier GPU pass wrote on the CPU, which drained the queue.
class MeshIndirectDraw {
public:
	MeshIndirectDraw(GraphicContext& graphics, CommandScheduler& scheduler);
	~MeshIndirectDraw();
	KYTY_CLASS_NO_COPY(MeshIndirectDraw);

	struct Params {
		vk::DeviceAddress args          = 0;
		uint64_t          index_address = 0;
		bool              indexed       = false;
		uint32_t          max_index_count      = 0;
		uint32_t          element_size         = 0;
		uint32_t          primitive_size       = 0;
		uint32_t          primitive_step       = 0;
		uint32_t          primitives_per_group = 0;
		bool              fast_launch          = false;
	};
	struct Result {
		vk::Buffer        groups_buffer;
		vk::DeviceSize    groups_offset = 0;
		vk::DeviceAddress draw_data     = 0; // the six MeshDrawParameter dwords
	};
	// Records the conversion; binds a compute pipeline and push state and must run outside a
	// render pass, before the draw's own bindings are committed.
	[[nodiscard]] Result Convert(const CommandRecorder& command, const Params& params);
	// The same for several draws behind one pair of barriers (KYTY_MESH_PRECONVERT): the
	// draws of a render pass converted before it begins. At most MaxBatch draws.
	void ConvertBatch(const CommandRecorder& command, std::span<const Params> params,
	                  std::span<Result> results);
	static constexpr uint32_t MaxBatch = 1024;
	// Conversions recorded so far; an entry stays valid for Entries - 1 later ones.
	[[nodiscard]] uint64_t Count() const noexcept { return m_count; }
	static constexpr uint32_t RingEntries = 4096;

private:
	Result RecordOne(const CommandRecorder& command, const Params& params);

	// Ring of entries (groups at 0, draw parameters at 4). An entry is rewritten only after 4096
	// later conversions, each ordered behind the reads before it.
	static constexpr uint32_t Entries     = RingEntries;
	static constexpr uint32_t EntryDwords = 16;
	uint64_t                  m_count     = 0;

	GraphicContext&         m_graphics;
	Buffer                  m_entries;
	vk::DescriptorSetLayout m_set_layout      = nullptr;
	vk::PipelineLayout      m_pipeline_layout = nullptr;
	vk::Pipeline            m_pipeline        = nullptr;
	uint32_t                m_next            = 0;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_INDIRECTDISPATCH_H_
