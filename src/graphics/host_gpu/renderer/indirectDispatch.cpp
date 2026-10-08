#include "graphics/host_gpu/renderer/indirectDispatch.h"

#include "common/assert.h"
#include "common/liveSwitches.h"
#include "gpu_tiler_shaders/dispatch_indirect_groups_spv.h"
#include "gpu_tiler_shaders/mesh_indirect_draw_spv.h"
#include "graphics/shader/shader.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/commandRecorder.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/vulkanCommon.h"

namespace Libs::Graphics {

namespace {

struct PushConstants {
	uint32_t threads_low;
	uint32_t threads_high;
	uint32_t local_size[3];
	uint32_t output_dword;
};

} // namespace

IndirectDispatchGroups::IndirectDispatchGroups(GraphicContext& graphics, CommandScheduler& scheduler)
    : m_graphics(graphics),
      m_groups(graphics, scheduler, MemoryUsage::DeviceLocal, 0,
               vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eIndirectBuffer |
                   vk::BufferUsageFlagBits::eShaderDeviceAddress,
               Entries * EntryDwords * sizeof(uint32_t)) {
	SetVulkanObjectNameF(graphics.device, m_groups.Handle(), "Indirect Dispatch Groups");

	const vk::DescriptorSetLayoutBinding binding {0, vk::DescriptorType::eStorageBuffer, 1,
	                                              vk::ShaderStageFlagBits::eCompute, nullptr};
	vk::DescriptorSetLayoutCreateInfo    layout_info {};
	layout_info.flags        = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR;
	layout_info.bindingCount = 1;
	layout_info.pBindings    = &binding;
	RequireVulkanSuccess(graphics.device.createDescriptorSetLayout(&layout_info, nullptr,
	                                                               &m_set_layout),
	                     "create indirect-dispatch descriptor layout");

	const vk::PushConstantRange  push_range {vk::ShaderStageFlagBits::eCompute, 0,
                                            sizeof(PushConstants)};
	vk::PipelineLayoutCreateInfo pipeline_layout_info {};
	pipeline_layout_info.setLayoutCount         = 1;
	pipeline_layout_info.pSetLayouts            = &m_set_layout;
	pipeline_layout_info.pushConstantRangeCount = 1;
	pipeline_layout_info.pPushConstantRanges    = &push_range;
	RequireVulkanSuccess(graphics.device.createPipelineLayout(&pipeline_layout_info, nullptr,
	                                                          &m_pipeline_layout),
	                     "create indirect-dispatch pipeline layout");

	const auto                        module = CompileSPV(DISPATCH_INDIRECT_GROUPS_SPV, graphics.device);
	vk::PipelineShaderStageCreateInfo stage {};
	stage.stage  = vk::ShaderStageFlagBits::eCompute;
	stage.module = module;
	stage.pName  = "main";
	vk::ComputePipelineCreateInfo pipeline_info {};
	pipeline_info.stage  = stage;
	pipeline_info.layout = m_pipeline_layout;
	const auto result =
	    graphics.device.createComputePipelines(nullptr, 1, &pipeline_info, nullptr, &m_pipeline);
	graphics.device.destroyShaderModule(module, nullptr);
	RequireVulkanSuccess(result, "create indirect-dispatch pipeline");
}

IndirectDispatchGroups::~IndirectDispatchGroups() {
	m_graphics.device.destroyPipeline(m_pipeline, nullptr);
	m_graphics.device.destroyPipelineLayout(m_pipeline_layout, nullptr);
	m_graphics.device.destroyDescriptorSetLayout(m_set_layout, nullptr);
}

IndirectDispatchGroups::Result
IndirectDispatchGroups::Convert(const CommandRecorder& command, vk::DeviceAddress threads,
                                const std::array<uint32_t, 3>& local_size) {
	const auto entry  = m_next;
	m_next            = (m_next + 1u) % Entries;
	const auto offset = vk::DeviceSize {entry} * EntryDwords * sizeof(uint32_t);

	// The counts come from earlier shader or transfer writes; the entry's previous indirect read
	// must finish before it is rewritten.
	vk::MemoryBarrier counts_ready {};
	counts_ready.srcAccessMask =
	    vk::AccessFlagBits::eShaderWrite | vk::AccessFlagBits::eTransferWrite;
	counts_ready.dstAccessMask = vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite;
	command.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
	                        vk::PipelineStageFlagBits::eComputeShader, {}, 1, &counts_ready, 0,
	                        nullptr, 0, nullptr);

	const vk::DescriptorBufferInfo info {m_groups.Handle(), 0, VK_WHOLE_SIZE};
	vk::WriteDescriptorSet         write {};
	write.dstBinding      = 0;
	write.descriptorCount = 1;
	write.descriptorType  = vk::DescriptorType::eStorageBuffer;
	write.pBufferInfo     = &info;
	const PushConstants push {
	    static_cast<uint32_t>(threads), static_cast<uint32_t>(threads >> 32u),
	    {local_size[0], local_size[1], local_size[2]}, entry * EntryDwords};
	command.bindPipeline(vk::PipelineBindPoint::eCompute, m_pipeline);
	command.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, m_pipeline_layout, 0, 1, &write);
	command.pushConstants(m_pipeline_layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(push),
	                      &push);
	command.dispatch(1, 1, 1);

	vk::MemoryBarrier groups_ready {};
	groups_ready.srcAccessMask = vk::AccessFlagBits::eShaderWrite;
	groups_ready.dstAccessMask =
	    vk::AccessFlagBits::eIndirectCommandRead | vk::AccessFlagBits::eShaderRead;
	command.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
	                        vk::PipelineStageFlagBits::eDrawIndirect |
	                            vk::PipelineStageFlagBits::eComputeShader,
	                        {}, 1, &groups_ready, 0, nullptr, 0, nullptr);
	return {m_groups.Handle(), offset,
	        m_groups.BufferDeviceAddress() + offset + 4u * sizeof(uint32_t)};
}

namespace {

struct MeshPushConstants {
	uint32_t args_low;
	uint32_t args_high;
	uint32_t index_low;
	uint32_t index_high;
	uint32_t indexed;
	uint32_t max_index_count;
	uint32_t element_size;
	uint32_t primitive_size;
	uint32_t primitive_step;
	uint32_t primitives_per_group;
	uint32_t max_groups_x;
	uint32_t max_groups_y;
	uint32_t max_groups_z;
	uint32_t max_groups_total;
	uint32_t split_stride;
	uint32_t output_dword;
	uint32_t fast_launch;
};

} // namespace

MeshIndirectDraw::MeshIndirectDraw(GraphicContext& graphics, CommandScheduler& scheduler)
    : m_graphics(graphics),
      m_entries(graphics, scheduler, MemoryUsage::DeviceLocal, 0,
                vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eIndirectBuffer |
                    vk::BufferUsageFlagBits::eShaderDeviceAddress,
                Entries * EntryDwords * sizeof(uint32_t)) {
	SetVulkanObjectNameF(graphics.device, m_entries.Handle(), "Mesh Indirect Draws");

	const vk::DescriptorSetLayoutBinding binding {0, vk::DescriptorType::eStorageBuffer, 1,
	                                              vk::ShaderStageFlagBits::eCompute, nullptr};
	vk::DescriptorSetLayoutCreateInfo    layout_info {};
	layout_info.flags        = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR;
	layout_info.bindingCount = 1;
	layout_info.pBindings    = &binding;
	RequireVulkanSuccess(graphics.device.createDescriptorSetLayout(&layout_info, nullptr,
	                                                               &m_set_layout),
	                     "create mesh indirect-draw descriptor layout");

	const vk::PushConstantRange  push_range {vk::ShaderStageFlagBits::eCompute, 0,
                                            sizeof(MeshPushConstants)};
	vk::PipelineLayoutCreateInfo pipeline_layout_info {};
	pipeline_layout_info.setLayoutCount         = 1;
	pipeline_layout_info.pSetLayouts            = &m_set_layout;
	pipeline_layout_info.pushConstantRangeCount = 1;
	pipeline_layout_info.pPushConstantRanges    = &push_range;
	RequireVulkanSuccess(graphics.device.createPipelineLayout(&pipeline_layout_info, nullptr,
	                                                          &m_pipeline_layout),
	                     "create mesh indirect-draw pipeline layout");

	const auto                        module = CompileSPV(MESH_INDIRECT_DRAW_SPV, graphics.device);
	vk::PipelineShaderStageCreateInfo stage {};
	stage.stage  = vk::ShaderStageFlagBits::eCompute;
	stage.module = module;
	stage.pName  = "main";
	vk::ComputePipelineCreateInfo pipeline_info {};
	pipeline_info.stage  = stage;
	pipeline_info.layout = m_pipeline_layout;
	const auto result =
	    graphics.device.createComputePipelines(nullptr, 1, &pipeline_info, nullptr, &m_pipeline);
	graphics.device.destroyShaderModule(module, nullptr);
	RequireVulkanSuccess(result, "create mesh indirect-draw pipeline");
}

MeshIndirectDraw::~MeshIndirectDraw() {
	m_graphics.device.destroyPipeline(m_pipeline, nullptr);
	m_graphics.device.destroyPipelineLayout(m_pipeline_layout, nullptr);
	m_graphics.device.destroyDescriptorSetLayout(m_set_layout, nullptr);
}

MeshIndirectDraw::Result MeshIndirectDraw::Convert(const CommandRecorder& command,
                                                   const Params&          params) {
	Result result;
	ConvertBatch(command, std::span {&params, 1}, std::span {&result, 1});
	return result;
}

void MeshIndirectDraw::ConvertBatch(const CommandRecorder& command, std::span<const Params> params,
                                    std::span<Result> results) {
	EXIT_IF(params.size() != results.size() || params.size() > MaxBatch);
	if (params.empty()) {
		return;
	}
	// The arguments come from earlier shader or transfer writes; the entry's previous reads
	// (indirect counts, mesh-shader parameter loads) must finish before it is rewritten.
	vk::MemoryBarrier args_ready {};
	args_ready.srcAccessMask = vk::AccessFlagBits::eShaderWrite | vk::AccessFlagBits::eTransferWrite;
	args_ready.dstAccessMask = vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite;
	// KYTY_MESH_CONVERT_NARROW=1 (live, research): wait only for compute and transfer writers of
	// the arguments (culling passes, DMA), not for the draws of the render pass just ended.
	static auto& narrow = Common::LiveSwitches::Get("KYTY_MESH_CONVERT_NARROW", 0);
	command.pipelineBarrier(narrow.load(std::memory_order_relaxed) != 0
	                            ? vk::PipelineStageFlagBits::eComputeShader |
	                                  vk::PipelineStageFlagBits::eTransfer
	                            : vk::PipelineStageFlagBits::eAllCommands,
	                        vk::PipelineStageFlagBits::eComputeShader, {}, 1, &args_ready, 0,
	                        nullptr, 0, nullptr);

	const vk::DescriptorBufferInfo info {m_entries.Handle(), 0, VK_WHOLE_SIZE};
	vk::WriteDescriptorSet         write {};
	write.dstBinding      = 0;
	write.descriptorCount = 1;
	write.descriptorType  = vk::DescriptorType::eStorageBuffer;
	write.pBufferInfo     = &info;
	command.bindPipeline(vk::PipelineBindPoint::eCompute, m_pipeline);
	command.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, m_pipeline_layout, 0, 1, &write);
	for (size_t i = 0; i < params.size(); i++) {
		results[i] = RecordOne(command, params[i]);
	}

	vk::MemoryBarrier entry_ready {};
	entry_ready.srcAccessMask = vk::AccessFlagBits::eShaderWrite;
	entry_ready.dstAccessMask =
	    vk::AccessFlagBits::eIndirectCommandRead | vk::AccessFlagBits::eShaderRead;
	command.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
	                        vk::PipelineStageFlagBits::eDrawIndirect |
	                            vk::PipelineStageFlagBits::eAllGraphics,
	                        {}, 1, &entry_ready, 0, nullptr, 0, nullptr);
}

MeshIndirectDraw::Result MeshIndirectDraw::RecordOne(const CommandRecorder& command,
                                                     const Params&          params) {
	const auto entry  = m_next;
	m_next            = (m_next + 1u) % Entries;
	m_count++;
	const auto offset = vk::DeviceSize {entry} * EntryDwords * sizeof(uint32_t);

	const auto& limits = m_graphics.mesh_shader_properties;
	const MeshPushConstants push {
	    static_cast<uint32_t>(params.args),
	    static_cast<uint32_t>(params.args >> 32u),
	    static_cast<uint32_t>(params.index_address),
	    static_cast<uint32_t>(params.index_address >> 32u),
	    params.indexed ? 1u : 0u,
	    params.max_index_count,
	    params.element_size,
	    params.primitive_size,
	    std::max(params.primitive_step, 1u),
	    std::max(params.primitives_per_group, 1u),
	    limits.maxMeshWorkGroupCount[0],
	    limits.maxMeshWorkGroupCount[1],
	    limits.maxMeshWorkGroupCount[2],
	    limits.maxMeshWorkGroupTotalCount,
	    MeshGroupSplitStride,
	    entry * EntryDwords,
	    params.fast_launch ? 1u : 0u};
	command.pushConstants(m_pipeline_layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(push),
	                      &push);
	command.dispatch(1, 1, 1);
	return {m_entries.Handle(), offset,
	        m_entries.BufferDeviceAddress() + offset + 4u * sizeof(uint32_t)};
}

} // namespace Libs::Graphics
