#include "graphics/host_gpu/renderer/pipeline/bindlessTable.h"

#include "common/alignment.h"
#include "common/assert.h"
#include "common/liveSwitches.h"
#include "common/logging/log.h"
#include "common/threads.h"
#include "graphics/host_gpu/addressBindingReport.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/pageManager.h"
#include "graphics/host_gpu/regionDefinitions.h"
#include "graphics/host_gpu/renderer/cache/samplerCache.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/shader/shaderBindings.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <atomic>
#include <cinttypes>
#include <array>
#include <atomic>
#include <fmt/format.h>
#include <string>
#include <cstdlib>
#include <cstring>
#include <map>
#include <tuple>

namespace Libs::Graphics {

bool BindlessEngineEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_BINDLESS_ENGINE");
		return value == nullptr || value[0] != '0';
	}();
	return enabled;
}

BindlessTable::BindlessTable(GraphicContext& graphics, CommandScheduler& scheduler,
                             PageManager& page_manager)
    : m_graphics(graphics), m_scheduler(scheduler), m_page_manager(page_manager) {
	if (!graphics.bindless_enabled) {
		return;
	}
	m_images_per_array =
	    std::min(MaxImagesPerArray, graphics.bindless_max_sampled_images / ImageArrays);
	if (m_images_per_array == 0) {
		LOGF("Bindless table: the device allows no update-after-bind sampled images\n");
		return;
	}

	constexpr auto image_flags = vk::DescriptorBindingFlagBits::ePartiallyBound |
	                             vk::DescriptorBindingFlagBits::eUpdateAfterBind |
	                             vk::DescriptorBindingFlagBits::eUpdateUnusedWhilePending;
	// The buffers are written once, here, so they need no update-after-bind.
	constexpr auto buffer_flags = vk::DescriptorBindingFlags {};
	// A device without update-after-bind samplers still gets a one-entry array, holding the
	// default sampler, so every bindless pipeline sees the same layout.
	m_samplers_per_array = std::min(MaxSamplers, graphics.bindless_max_samplers);
	m_storage_images     = MaxStorageImages;
	const std::array<vk::DescriptorSetLayoutBinding, 8> bindings {{
	    {Images2D, vk::DescriptorType::eSampledImage, m_images_per_array, vk::ShaderStageFlagBits::eAll,
	     nullptr},
	    {Images2DArray, vk::DescriptorType::eSampledImage, m_images_per_array,
	     vk::ShaderStageFlagBits::eAll, nullptr},
	    {ImagesCube, vk::DescriptorType::eSampledImage, m_images_per_array,
	     vk::ShaderStageFlagBits::eAll, nullptr},
	    {Images3D, vk::DescriptorType::eSampledImage, m_images_per_array, vk::ShaderStageFlagBits::eAll,
	     nullptr},
	    {Translation, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eAll, nullptr},
	    {Feedback, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eAll, nullptr},
	    {Samplers, vk::DescriptorType::eSampler, std::max(m_samplers_per_array, 1u),
	     vk::ShaderStageFlagBits::eAll, nullptr},
	    {StorageImages2D, vk::DescriptorType::eStorageImage, m_storage_images,
	     vk::ShaderStageFlagBits::eAll, nullptr},
	}};
	const std::array<vk::DescriptorBindingFlags, 8> binding_flags {
	    image_flags, image_flags, image_flags, image_flags, buffer_flags, buffer_flags, image_flags,
	    image_flags};
	vk::DescriptorSetLayoutBindingFlagsCreateInfo flags_info {};
	flags_info.bindingCount  = static_cast<uint32_t>(binding_flags.size());
	flags_info.pBindingFlags = binding_flags.data();
	vk::DescriptorSetLayoutCreateInfo layout_info {};
	layout_info.pNext        = &flags_info;
	layout_info.flags        = vk::DescriptorSetLayoutCreateFlagBits::eUpdateAfterBindPool;
	layout_info.bindingCount = static_cast<uint32_t>(bindings.size());
	layout_info.pBindings    = bindings.data();
	RequireVulkanSuccess(graphics.device.createDescriptorSetLayout(&layout_info, nullptr, &m_layout),
	                     "create bindless descriptor layout");

	const std::array<vk::DescriptorPoolSize, 4> sizes {{
	    {vk::DescriptorType::eSampledImage, m_images_per_array * ImageArrays},
	    {vk::DescriptorType::eStorageBuffer, 2},
	    {vk::DescriptorType::eSampler, std::max(m_samplers_per_array, 1u)},
	    {vk::DescriptorType::eStorageImage, m_storage_images},
	}};
	vk::DescriptorPoolCreateInfo pool_info {};
	pool_info.flags         = vk::DescriptorPoolCreateFlagBits::eUpdateAfterBind;
	pool_info.maxSets       = 1;
	pool_info.poolSizeCount = static_cast<uint32_t>(sizes.size());
	pool_info.pPoolSizes    = sizes.data();
	RequireVulkanSuccess(graphics.device.createDescriptorPool(&pool_info, nullptr, &m_pool),
	                     "create bindless descriptor pool");

	vk::DescriptorSetAllocateInfo allocate_info {};
	allocate_info.descriptorPool     = m_pool;
	allocate_info.descriptorSetCount = 1;
	allocate_info.pSetLayouts        = &m_layout;
	RequireVulkanSuccess(graphics.device.allocateDescriptorSets(&allocate_info, &m_set),
	                     "allocate bindless descriptor set");

	// Both live in host-visible device memory: the host writes translations and clears feedback
	// flags directly, while shaders access them at device speed. The host reads the flags from
	// a host-cached snapshot (RecordFeedbackSnapshot).
	m_translation = std::make_unique<Buffer>(graphics, scheduler, MemoryUsage::Stream, 0,
	                                         AllFlags, TranslationEntries * sizeof(uint32_t));
	m_feedback    = std::make_unique<Buffer>(graphics, scheduler, MemoryUsage::Stream, 0,
	                                         AllFlags, TranslationEntries * sizeof(uint32_t));
	m_feedback_snapshot = std::make_unique<Buffer>(graphics, scheduler, MemoryUsage::Download, 0,
	                                               AllFlags, TranslationEntries * sizeof(uint32_t));
	EXIT_IF(m_translation->Mapped().empty() || m_feedback->Mapped().empty() ||
	        m_feedback_snapshot->Mapped().empty());
	// A region allocated after a snapshot was recorded reads as "no requests" until the next.
	std::fill(m_feedback_snapshot->Mapped().begin(), m_feedback_snapshot->Mapped().end(),
	          uint8_t {0});
	m_feedback_snapshot->Flush(0, m_feedback_snapshot->Size());
	SetVulkanObjectNameF(graphics.device, m_translation->Handle(), "Bindless Translation");
	SetVulkanObjectNameF(graphics.device, m_feedback->Handle(), "Bindless Feedback");

	const std::array<vk::DescriptorBufferInfo, 2> buffer_infos {{
	    {m_translation->Handle(), 0, VK_WHOLE_SIZE},
	    {m_feedback->Handle(), 0, VK_WHOLE_SIZE},
	}};
	std::array<vk::WriteDescriptorSet, 2> writes {};
	for (uint32_t i = 0; i < writes.size(); i++) {
		writes[i].dstSet          = m_set;
		writes[i].dstBinding      = Translation + i;
		writes[i].descriptorCount = 1;
		writes[i].descriptorType  = vk::DescriptorType::eStorageBuffer;
		writes[i].pBufferInfo     = &buffer_infos[i];
	}
	graphics.device.updateDescriptorSets(static_cast<uint32_t>(writes.size()), writes.data(), 0,
	                                     nullptr);
	CreatePlaceholders(scheduler);
	graphics.bindless_layout = m_layout;
	graphics.bindless_set    = m_set;
	LOGF("Bindless table: %u images per array, %u translation entries, %u samplers\n",
	     m_images_per_array, TranslationEntries, m_samplers_per_array);
}

BindlessTable::SamplerHeap* BindlessTable::FindOrCreateSamplerHeap(uint64_t base,
                                                                   uint32_t table_offset,
                                                                   uint32_t flags) {
	for (auto& heap: m_sampler_heaps) {
		if (heap.base == base && heap.table_offset == table_offset && heap.flags == flags) {
			return &heap;
		}
	}
	auto& heap        = m_sampler_heaps.emplace_back();
	heap.base         = base;
	heap.table_offset = table_offset;
	heap.flags        = flags;
	return &heap;
}

void BindlessTable::WriteDefaultSampler(vk::Sampler sampler) {
	if (m_default_sampler_written || m_set == nullptr) {
		return;
	}
	vk::DescriptorImageInfo info {sampler, nullptr, vk::ImageLayout::eUndefined};
	vk::WriteDescriptorSet  write {};
	write.dstSet          = m_set;
	write.dstBinding      = Samplers;
	write.dstArrayElement = 0;
	write.descriptorCount = 1;
	write.descriptorType  = vk::DescriptorType::eSampler;
	write.pImageInfo      = &info;
	m_graphics.device.updateDescriptorSets(1, &write, 0, nullptr);
	m_default_sampler_written = true;
}

bool BindlessTable::MirrorSamplerHeap(SamplerHeap&                              heap,
                                      std::span<const std::array<uint32_t, 4>> records,
                                      SamplerCache&                             cache) {
	const auto same_prefix =
	    records.size() >= heap.records.size() &&
	    std::equal(heap.records.begin(), heap.records.end(), records.begin());
	if (same_prefix && records.size() == heap.records.size()) {
		return true;
	}
	uint32_t first = static_cast<uint32_t>(heap.records.size());
	if (!same_prefix || records.size() > heap.capacity || heap.region == 0) {
		const auto capacity = std::max<uint32_t>(64u, static_cast<uint32_t>(records.size()) * 2u);
		if (m_samplers_per_array <= 1u || capacity > m_samplers_per_array - m_next_sampler_slot) {
			static std::atomic<uint32_t> reported = 0;
			if (reported.fetch_add(1) < 8) {
				LOGF("Bindless samplers: array full (%u of %u slots), heap 0x%016" PRIx64
				     " keeps %zu records\n",
				     m_next_sampler_slot, m_samplers_per_array, heap.base, heap.records.size());
			}
			return false;
		}
		heap.region   = m_next_sampler_slot;
		heap.capacity = capacity;
		m_next_sampler_slot += capacity;
		first = 0;
	}
	std::vector<vk::DescriptorImageInfo> infos;
	infos.reserve(records.size() - first);
	// KYTY_BINDLESS_SAMPLERS_CLAMP_ONLY=1 (environment): take only the address modes (CLAMP_X/Y/Z)
	// from the game's records and keep the fallback sampler's filtering and LOD range. Run 65: the
	// full records turned Wolverine's terrain into flat colours while the moon needs its record's
	// clamp; this tells the fields apart from the key -> record mapping.
	static const bool clamp_only = [] {
		const char* value = std::getenv("KYTY_BINDLESS_SAMPLERS_CLAMP_ONLY");
		return value != nullptr && value[0] == '1';
	}();
	for (size_t key = first; key < records.size(); key++) {
		ShaderSamplerResource descriptor;
		std::copy(records[key].begin(), records[key].end(), descriptor.fields);
		if (clamp_only) {
			const auto& fallback = ShaderRecompiler::IR::BindlessDefaultSampler;
			descriptor.fields[0] = (fallback[0] & ~0x1ffu) | (records[key][0] & 0x1ffu);
			descriptor.fields[1] = fallback[1];
			descriptor.fields[2] = fallback[2];
			descriptor.fields[3] = fallback[3];
		}
		if ((heap.flags & SamplerDepthCompare) == 0u) {
			descriptor.fields[0] &= ~(0x7u << 12u);
		}
		if ((heap.flags & SamplerPointFiltering) != 0u) {
			descriptor.SetPointFiltering();
		}
		infos.push_back({cache.GetSampler(descriptor, (heap.flags & SamplerIntegerBorder) != 0u),
		                 nullptr, vk::ImageLayout::eUndefined});
	}
	if (!infos.empty()) {
		vk::WriteDescriptorSet write {};
		write.dstSet          = m_set;
		write.dstBinding      = Samplers;
		write.dstArrayElement = heap.region + first;
		write.descriptorCount = static_cast<uint32_t>(infos.size());
		write.descriptorType  = vk::DescriptorType::eSampler;
		write.pImageInfo      = infos.data();
		m_graphics.device.updateDescriptorSets(1, &write, 0, nullptr);
	}
	static std::atomic<uint32_t> logged = 0;
	if (logged.fetch_add(1) < 32) {
		LOGF("Bindless samplers: heap 0x%016" PRIx64 "+0x%x flags=%u region=%u records=%zu"
		     " (%u new)\n",
		     heap.base, heap.table_offset, heap.flags, heap.region, records.size(),
		     static_cast<uint32_t>(infos.size()));
	}
	// KYTY_SAMPLER_DUMP=1 (environment): the first mirrors of each heap print what its S# records
	// ask for, grouped: address modes, LOD range, mip filter, unnormalized coordinates, LOD bias.
	static const bool dump = [] {
		const char* value = std::getenv("KYTY_SAMPLER_DUMP");
		return value != nullptr && value[0] == '1';
	}();
	static std::atomic<uint32_t> dumped = 0;
	if (dump && !infos.empty() && dumped.fetch_add(1) < 24) {
		std::map<std::tuple<uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t>,
		         uint32_t>
		    groups;
		for (const auto& record: records) {
			ShaderSamplerResource r;
			std::copy(record.begin(), record.end(), r.fields);
			groups[{static_cast<uint32_t>(r.ClampX() | (r.ClampY() << 3u) | (r.ClampZ() << 6u)),
			        r.MinLod(), r.MaxLod(), static_cast<uint32_t>(r.MipFilter()),
			        r.ForceUnormCoords() ? 1u : 0u, r.LodBias(),
			        static_cast<uint32_t>(r.XyMinFilter() | (r.XyMagFilter() << 2u))}]++;
		}
		::printf("Sampler heap 0x%016" PRIx64 "+0x%x flags=%u: %zu records, %zu distinct\n", heap.base,
		         heap.table_offset, heap.flags, records.size(), groups.size());
		size_t shown = 0;
		for (const auto& [key, count]: groups) {
			if (shown++ == 16) {
				break;
			}
			const auto [clamp, min_lod, max_lod, mip, unnorm, bias, filters] = key;
			::printf("  %4u x clamp=%u/%u/%u min_lod=%.2f max_lod=%.2f mip=%u unnorm=%u "
			         "bias=0x%04x xyfilt=0x%x\n",
			         count, clamp & 7u, (clamp >> 3u) & 7u, (clamp >> 6u) & 7u,
			         static_cast<double>(min_lod) / 256.0, static_cast<double>(max_lod) / 256.0, mip,
			         unnorm, bias, filters);
		}
		for (size_t i = 0; i < std::min<size_t>(records.size(), 4); i++) {
			::printf("  raw[%zu] = %08x %08x %08x %08x\n", i, records[i][0], records[i][1],
			         records[i][2], records[i][3]);
		}
		std::fflush(stdout);
	}
	heap.records.assign(records.begin(), records.end());
	return true;
}

void BindlessTable::CreatePlaceholders(CommandScheduler& /*scheduler*/) {
	struct Kind {
		vk::ImageType        type;
		vk::ImageViewType    view;
		uint32_t             layers;
		vk::ImageCreateFlags flags;
	};
	const std::array<Kind, ImageArrays> kinds {{
	    {vk::ImageType::e2D, vk::ImageViewType::e2D, 1, {}},
	    {vk::ImageType::e2D, vk::ImageViewType::e2DArray, 1, {}},
	    {vk::ImageType::e2D, vk::ImageViewType::eCube, 6, vk::ImageCreateFlagBits::eCubeCompatible},
	    {vk::ImageType::e3D, vk::ImageViewType::e3D, 1, {}},
	}};
	for (uint32_t i = 0; i < Placeholders; i++) {
		vk::ImageCreateInfo info {};
		info.flags         = kinds[i % ImageArrays].flags;
		info.imageType     = kinds[i % ImageArrays].type;
		info.format        = vk::Format::eR8G8B8A8Unorm;
		info.extent        = vk::Extent3D {1, 1, 1};
		info.mipLevels     = 1;
		info.arrayLayers   = kinds[i % ImageArrays].layers;
		info.samples       = vk::SampleCountFlagBits::e1;
		info.tiling        = vk::ImageTiling::eOptimal;
		info.usage         = vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst;
		info.initialLayout = vk::ImageLayout::eUndefined;
		EXIT_IF(!m_graphics.CreateImage(info, m_placeholders[i]));
		vk::ImageViewCreateInfo view {};
		view.image            = m_placeholders[i].image;
		view.viewType         = kinds[i % ImageArrays].view;
		view.format           = info.format;
		view.subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, kinds[i % ImageArrays].layers};
		RequireVulkanSuccess(
		    m_graphics.device.createImageView(&view, nullptr, &m_placeholder_views[i]),
		    "create bindless placeholder view");
	}

	// One fenced submission, before any rendering: clears and a buffer fill are not allowed
	// inside the render passes draws record into.
	vk::CommandPoolCreateInfo pool_info {};
	pool_info.flags            = vk::CommandPoolCreateFlagBits::eTransient;
	pool_info.queueFamilyIndex = m_graphics.queue_family;
	vk::CommandPool pool       = nullptr;
	RequireVulkanSuccess(m_graphics.device.createCommandPool(&pool_info, nullptr, &pool),
	                     "create bindless init pool");
	vk::CommandBufferAllocateInfo allocate {};
	allocate.commandPool        = pool;
	allocate.level              = vk::CommandBufferLevel::ePrimary;
	allocate.commandBufferCount = 1;
	vk::CommandBuffer command   = nullptr;
	RequireVulkanSuccess(m_graphics.device.allocateCommandBuffers(&allocate, &command),
	                     "allocate bindless init command buffer");
	vk::CommandBufferBeginInfo begin {};
	begin.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;
	RequireVulkanSuccess(command.begin(&begin), "begin bindless init command buffer");

	std::array<vk::ImageMemoryBarrier, Placeholders> to_transfer {};
	std::array<vk::ImageMemoryBarrier, Placeholders> to_read {};
	for (uint32_t i = 0; i < Placeholders; i++) {
		const vk::ImageSubresourceRange range {vk::ImageAspectFlagBits::eColor, 0, 1, 0,
		                                       kinds[i % ImageArrays].layers};
		to_transfer[i].srcAccessMask       = {};
		to_transfer[i].dstAccessMask       = vk::AccessFlagBits::eTransferWrite;
		to_transfer[i].oldLayout           = vk::ImageLayout::eUndefined;
		to_transfer[i].newLayout           = vk::ImageLayout::eTransferDstOptimal;
		to_transfer[i].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		to_transfer[i].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		to_transfer[i].image               = m_placeholders[i].image;
		to_transfer[i].subresourceRange    = range;
		to_read[i]                         = to_transfer[i];
		to_read[i].srcAccessMask           = vk::AccessFlagBits::eTransferWrite;
		to_read[i].dstAccessMask           = vk::AccessFlagBits::eShaderRead;
		to_read[i].oldLayout               = vk::ImageLayout::eTransferDstOptimal;
		to_read[i].newLayout               = vk::ImageLayout::eShaderReadOnlyOptimal;
	}
	command.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe,
	                        vk::PipelineStageFlagBits::eTransfer, {}, 0, nullptr, 0, nullptr,
	                        Placeholders, to_transfer.data());
	// Transparent black by default. KYTY_BINDLESS_DEBUG_COLORS=1: slot 0 magenta (not resident), slot
	// 1 red (key outside its heap), slot 2 blue (pending), slot 3 green (a dropped image's slot), to
	// see which case a surface is in.
	const bool debug_colors = std::getenv("KYTY_BINDLESS_DEBUG_COLORS") != nullptr;
	const auto color        = [debug_colors](float r, float g, float b) {
        return debug_colors ? vk::ClearColorValue {std::array<float, 4> {r, g, b, 1.0f}}
		                           : vk::ClearColorValue {std::array<float, 4> {0.0f, 0.0f, 0.0f, 0.0f}};
	};
	const std::array<vk::ClearColorValue, PlaceholderColors> colors {
	    color(1.0f, 0.0f, 1.0f), color(0.8f, 0.1f, 0.1f), color(0.1f, 0.2f, 0.8f),
	    color(0.1f, 0.8f, 0.1f)};
	// KYTY_BINDLESS_DEBUG_COLORS=2 (research 2026-10-14): one colour per array instead, whatever the
	// slot: 2D magenta, 2D array yellow, cube cyan, 3D white (the storage placeholder is orange), to
	// tell which of a shader's bindless textures is missing.
	const char* debug_mode = std::getenv("KYTY_BINDLESS_DEBUG_COLORS");
	const bool  per_array  = debug_mode != nullptr && std::strcmp(debug_mode, "2") == 0;
	const std::array<vk::ClearColorValue, ImageArrays> array_colors {
	    color(1.0f, 0.0f, 1.0f), color(1.0f, 1.0f, 0.0f), color(0.0f, 1.0f, 1.0f),
	    color(1.0f, 1.0f, 1.0f)};
	for (uint32_t i = 0; i < Placeholders; i++) {
		command.clearColorImage(m_placeholders[i].image, vk::ImageLayout::eTransferDstOptimal,
		                        per_array ? &array_colors[i % ImageArrays] : &colors[i / ImageArrays], 1,
		                        &to_transfer[i].subresourceRange);
	}
	// translation[0], the entry of keys outside their heap, is the red slot.
	command.fillBuffer(m_translation->Handle(), 4, VK_WHOLE_SIZE, 0u);
	command.fillBuffer(m_translation->Handle(), 0, 4, 1u);
	command.fillBuffer(m_feedback->Handle(), 0, VK_WHOLE_SIZE, 0u);
	vk::BufferMemoryBarrier translation_ready {};
	translation_ready.srcAccessMask       = vk::AccessFlagBits::eTransferWrite;
	translation_ready.dstAccessMask       = vk::AccessFlagBits::eShaderRead;
	translation_ready.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	translation_ready.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	translation_ready.buffer              = m_translation->Handle();
	translation_ready.offset              = 0;
	translation_ready.size                = VK_WHOLE_SIZE;
	auto feedback_ready                   = translation_ready;
	feedback_ready.dstAccessMask          = vk::AccessFlagBits::eShaderRead |
	                                        vk::AccessFlagBits::eShaderWrite |
	                                        vk::AccessFlagBits::eHostRead;
	feedback_ready.buffer                 = m_feedback->Handle();
	const std::array<vk::BufferMemoryBarrier, 2> buffers_ready {translation_ready, feedback_ready};
	command.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
	                        vk::PipelineStageFlagBits::eAllCommands |
	                            vk::PipelineStageFlagBits::eHost,
	                        {}, 0, nullptr, static_cast<uint32_t>(buffers_ready.size()),
	                        buffers_ready.data(), Placeholders, to_read.data());
	RequireVulkanSuccess(command.end(), "end bindless init command buffer");

	vk::FenceCreateInfo fence_info {};
	vk::Fence           fence = nullptr;
	RequireVulkanSuccess(m_graphics.device.createFence(&fence_info, nullptr, &fence),
	                     "create bindless init fence");
	vk::SubmitInfo submit {};
	submit.commandBufferCount = 1;
	submit.pCommandBuffers    = &command;
	{
		Common::LockGuard lock(m_graphics.queue_mutex);
		RequireVulkanSuccess(m_graphics.queue.submit(1, &submit, fence), "submit bindless init");
	}
	RequireVulkanSuccess(m_graphics.device.waitForFences(1, &fence, VK_TRUE, UINT64_MAX),
	                     "wait for bindless init");
	m_graphics.device.destroyFence(fence, nullptr);
	m_graphics.device.destroyCommandPool(pool, nullptr);

	std::array<vk::DescriptorImageInfo, Placeholders> infos {};
	std::array<vk::WriteDescriptorSet, Placeholders> writes {};
	for (uint32_t i = 0; i < Placeholders; i++) {
		infos[i]                  = {nullptr, m_placeholder_views[i],
		                             vk::ImageLayout::eShaderReadOnlyOptimal};
		writes[i].dstSet          = m_set;
		writes[i].dstBinding      = Images2D + i % ImageArrays;
		writes[i].dstArrayElement = i / ImageArrays;
		writes[i].descriptorCount = 1;
		writes[i].descriptorType  = vk::DescriptorType::eSampledImage;
		writes[i].pImageInfo      = &infos[i];
	}
	m_graphics.device.updateDescriptorSets(Placeholders, writes.data(), 0, nullptr);
	CreateStoragePlaceholder();
}

// Slots 0-3 of StorageImages2D: one 1x1 RGBA16F storage image in the general layout, transparent
// black (magenta with KYTY_BINDLESS_DEBUG_COLORS). Loads of a key without a resident image read it,
// stores to it are lost, as on the sampled arrays' placeholders.
void BindlessTable::CreateStoragePlaceholder() {
	vk::ImageCreateInfo info {};
	info.imageType     = vk::ImageType::e2D;
	info.format        = vk::Format::eR16G16B16A16Sfloat;
	info.extent        = vk::Extent3D {1, 1, 1};
	info.mipLevels     = 1;
	info.arrayLayers   = 1;
	info.samples       = vk::SampleCountFlagBits::e1;
	info.tiling        = vk::ImageTiling::eOptimal;
	info.usage         = vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eTransferDst;
	info.initialLayout = vk::ImageLayout::eUndefined;
	EXIT_IF(!m_graphics.CreateImage(info, m_storage_placeholder));
	vk::ImageViewCreateInfo view {};
	view.image            = m_storage_placeholder.image;
	view.viewType         = vk::ImageViewType::e2D;
	view.format           = info.format;
	view.subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
	RequireVulkanSuccess(m_graphics.device.createImageView(&view, nullptr, &m_storage_placeholder_view),
	                     "create bindless storage placeholder view");

	vk::CommandPoolCreateInfo pool_info {};
	pool_info.flags            = vk::CommandPoolCreateFlagBits::eTransient;
	pool_info.queueFamilyIndex = m_graphics.queue_family;
	vk::CommandPool pool       = nullptr;
	RequireVulkanSuccess(m_graphics.device.createCommandPool(&pool_info, nullptr, &pool),
	                     "create bindless storage init pool");
	vk::CommandBufferAllocateInfo allocate {};
	allocate.commandPool        = pool;
	allocate.level              = vk::CommandBufferLevel::ePrimary;
	allocate.commandBufferCount = 1;
	vk::CommandBuffer command   = nullptr;
	RequireVulkanSuccess(m_graphics.device.allocateCommandBuffers(&allocate, &command),
	                     "allocate bindless storage init command buffer");
	vk::CommandBufferBeginInfo begin {};
	begin.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;
	RequireVulkanSuccess(command.begin(&begin), "begin bindless storage init command buffer");
	const vk::ImageSubresourceRange range {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
	vk::ImageMemoryBarrier to_general {};
	to_general.srcAccessMask       = {};
	to_general.dstAccessMask       = vk::AccessFlagBits::eTransferWrite;
	to_general.oldLayout           = vk::ImageLayout::eUndefined;
	to_general.newLayout           = vk::ImageLayout::eGeneral;
	to_general.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	to_general.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	to_general.image               = m_storage_placeholder.image;
	to_general.subresourceRange    = range;
	command.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eTransfer,
	                        {}, 0, nullptr, 0, nullptr, 1, &to_general);
	const char* debug_mode   = std::getenv("KYTY_BINDLESS_DEBUG_COLORS");
	const bool  debug_colors = debug_mode != nullptr;
	const bool  per_array    = debug_colors && std::strcmp(debug_mode, "2") == 0; // orange
	const vk::ClearColorValue color {
	    std::array<float, 4> {debug_colors ? 1.0f : 0.0f, per_array ? 0.5f : 0.0f,
	                          debug_colors && !per_array ? 1.0f : 0.0f, debug_colors ? 1.0f : 0.0f}};
	command.clearColorImage(m_storage_placeholder.image, vk::ImageLayout::eGeneral, &color, 1, &range);
	auto ready          = to_general;
	ready.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
	ready.dstAccessMask = vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite;
	ready.oldLayout     = vk::ImageLayout::eGeneral;
	command.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eAllCommands,
	                        {}, 0, nullptr, 0, nullptr, 1, &ready);
	RequireVulkanSuccess(command.end(), "end bindless storage init command buffer");
	vk::FenceCreateInfo fence_info {};
	vk::Fence           fence = nullptr;
	RequireVulkanSuccess(m_graphics.device.createFence(&fence_info, nullptr, &fence),
	                     "create bindless storage init fence");
	vk::SubmitInfo submit {};
	submit.commandBufferCount = 1;
	submit.pCommandBuffers    = &command;
	{
		Common::LockGuard lock(m_graphics.queue_mutex);
		RequireVulkanSuccess(m_graphics.queue.submit(1, &submit, fence), "submit bindless storage init");
	}
	RequireVulkanSuccess(m_graphics.device.waitForFences(1, &fence, VK_TRUE, UINT64_MAX),
	                     "wait for bindless storage init");
	m_graphics.device.destroyFence(fence, nullptr);
	m_graphics.device.destroyCommandPool(pool, nullptr);

	std::array<vk::DescriptorImageInfo, PlaceholderColors> infos {};
	std::array<vk::WriteDescriptorSet, PlaceholderColors> writes {};
	for (uint32_t i = 0; i < PlaceholderColors; i++) {
		infos[i]                  = {nullptr, m_storage_placeholder_view, vk::ImageLayout::eGeneral};
		writes[i].dstSet          = m_set;
		writes[i].dstBinding      = StorageImages2D;
		writes[i].dstArrayElement = i;
		writes[i].descriptorCount = 1;
		writes[i].descriptorType  = vk::DescriptorType::eStorageImage;
		writes[i].pImageInfo      = &infos[i];
	}
	m_graphics.device.updateDescriptorSets(PlaceholderColors, writes.data(), 0, nullptr);
}

BindlessTable::~BindlessTable() {
	for (uint32_t i = 0; i < Placeholders; i++) {
		if (m_placeholder_views[i] != nullptr) {
			m_graphics.device.destroyImageView(m_placeholder_views[i], nullptr);
		}
		if (m_placeholders[i].image != nullptr) {
			m_graphics.DeleteImage(m_placeholders[i]);
		}
	}
	if (m_storage_placeholder_view != nullptr) {
		m_graphics.device.destroyImageView(m_storage_placeholder_view, nullptr);
	}
	if (m_storage_placeholder.image != nullptr) {
		m_graphics.DeleteImage(m_storage_placeholder);
	}
	m_translation.reset();
	m_feedback.reset();
	m_feedback_snapshot.reset();
	if (m_pool != nullptr) {
		m_graphics.device.destroyDescriptorPool(m_pool, nullptr);
	}
	if (m_layout != nullptr) {
		m_graphics.device.destroyDescriptorSetLayout(m_layout, nullptr);
	}
}

bool BindlessTable::AllocateRegion(Heap& heap, uint32_t entries) {
	// Headroom so a growing heap rarely moves; a move re-writes the keys resolved so far.
	const auto capacity = std::max(entries, std::min<uint32_t>(entries * 2u, 1u << 16u));
	if (capacity > ShaderRecompiler::IR::WatchdogReportBase - m_next_region) {
		return false;
	}
	const auto old_region  = heap.region;
	const auto old_entries = heap.entries;
	heap.region            = m_next_region;
	heap.entries           = capacity;
	m_next_region += capacity;
	heap.slots.resize(capacity, 0u);
	heap.settled.resize(capacity, 0u);
	heap.images.resize(capacity);
	heap.descriptors.resize(capacity);
	auto* translation = reinterpret_cast<uint32_t*>(m_translation->Mapped().data());
	auto* feedback    = reinterpret_cast<uint32_t*>(m_feedback->Mapped().data());
	for (uint32_t key = 0; key < capacity; key++) {
		uint32_t value = ShaderRecompiler::IR::BindlessPending;
		if (key < old_entries && heap.settled[key] != 0) {
			value = heap.slots[key];
		}
		translation[heap.region + key] = value;
		feedback[heap.region + key]    = 0;
	}
	if (!m_translation->IsCoherent()) {
		m_translation->Flush(heap.region * sizeof(uint32_t), capacity * sizeof(uint32_t));
	}
	LOGF("Bindless heap region: base=0x%016" PRIx64 " binding=%u region=%u entries=%u (was %u at "
	     "%u)\n",
	     heap.base, heap.binding, heap.region, capacity, old_entries, old_region);
	return true;
}

BindlessTable::Heap* BindlessTable::FindOrCreateHeap(
    uint64_t base, uint32_t table_offset, uint32_t binding, uint32_t entries,
    const ShaderRecompiler::IR::ImageResource& resource) {
	if (entries == 0) {
		return nullptr;
	}
	for (auto& heap: m_heaps) {
		if (heap.base == base && heap.table_offset == table_offset && heap.binding == binding) {
			if (entries > heap.entries && !AllocateRegion(heap, entries)) {
				return nullptr;
			}
			heap.guest_entries = std::max(heap.guest_entries, std::min(entries, heap.entries));
			return &heap;
		}
	}
	auto& heap        = m_heaps.emplace_back();
	heap.base         = base;
	heap.table_offset = table_offset;
	heap.binding      = binding;
	heap.resource     = resource;
	if (!AllocateRegion(heap, std::max(entries, 1u << 14u))) {
		m_heaps.pop_back();
		return nullptr;
	}
	heap.guest_entries = std::min(entries, heap.entries);
	return &heap;
}

// Engine method (Senaxx c5a6961b): the write watch over the heaps' guest pages.
void BindlessTable::WatchHeap(Heap& heap) {
	heap.watched_entries  = heap.guest_entries;
	heap.watch_generation = m_watch_generation.load(std::memory_order_acquire);
	const auto table = heap.base + heap.table_offset;
	const auto begin = Common::AlignDown(table, TRACKER_PAGE_SIZE);
	const auto end = Common::AlignUp(table + uint64_t {heap.guest_entries} * 32u, TRACKER_PAGE_SIZE);
	if (end <= begin) {
		return;
	}
	std::scoped_lock lock {m_watch_mutex};
	auto watch = std::find_if(m_watches.begin(), m_watches.end(),
	                          [&](const Watch& w) { return w.begin == begin; });
	if (watch == m_watches.end()) {
		watch        = m_watches.insert(m_watches.end(), Watch {});
		watch->begin = begin;
		watch->end   = begin;
	}
	if (end <= watch->end) {
		return;
	}
	// The heap grew (or is new): watch the pages it now covers.
	const auto first = static_cast<size_t>((watch->end - begin) / TRACKER_PAGE_SIZE);
	const auto pages = static_cast<size_t>((end - begin) / TRACKER_PAGE_SIZE);
	watch->watched.resize(pages, 0u);
	watch->written.resize(pages, 0u);
	watch->end = end;
	for (size_t page = first; page < pages; page++) {
		watch->watched[page] = 1;
	}
	m_page_manager.UpdatePageWatchers<true>(begin + first * TRACKER_PAGE_SIZE,
	                                        (pages - first) * TRACKER_PAGE_SIZE);
	if (begin < m_watch_low.load(std::memory_order_relaxed)) {
		m_watch_low.store(begin, std::memory_order_release);
	}
	if (end > m_watch_high.load(std::memory_order_relaxed)) {
		m_watch_high.store(end, std::memory_order_release);
	}
	::printf("Bindless engine: watching heap 0x%016" PRIx64 "+0x%x binding=%u keys=%u (%zu pages)\n",
	         heap.base, heap.table_offset, heap.binding, heap.guest_entries, pages);
}

void BindlessTable::OnCpuWrite(uint64_t vaddr, uint64_t size) {
	// Called for every CPU write fault: most are nowhere near a heap and must not take the lock.
	if (vaddr + size <= m_watch_low.load(std::memory_order_acquire) ||
	    vaddr >= m_watch_high.load(std::memory_order_acquire)) {
		return;
	}
	std::scoped_lock lock {m_watch_mutex};
	for (auto& watch: m_watches) {
		const auto begin = std::max(watch.begin, Common::AlignDown(vaddr, TRACKER_PAGE_SIZE));
		const auto end   = std::min(watch.end, vaddr + size);
		for (auto address = begin; address < end; address += TRACKER_PAGE_SIZE) {
			const auto page = static_cast<size_t>((address - watch.begin) / TRACKER_PAGE_SIZE);
			if (watch.watched[page] == 0) {
				continue;
			}
			// The faulting write must go through: stop watching the page until the next sync.
			watch.watched[page] = 0;
			watch.written[page] = 1;
			m_page_manager.UpdatePageWatchers<false>(address, TRACKER_PAGE_SIZE);
			m_heap_writes.store(true, std::memory_order_release);
		}
	}
}

void BindlessTable::UnwatchRange(uint64_t vaddr, uint64_t size) {
	std::scoped_lock lock {m_watch_mutex};
	for (auto& watch: m_watches) {
		if (vaddr + size <= watch.begin || vaddr >= watch.end) {
			continue;
		}
		// Part of the table is unmapped: drop the whole watch. WatchHeap builds it again the next
		// time a draw uses the heap, over what is mapped then.
		for (size_t page = 0; page < watch.watched.size(); page++) {
			if (watch.watched[page] != 0) {
				m_page_manager.UpdatePageWatchers<false>(watch.begin + page * TRACKER_PAGE_SIZE,
				                                         TRACKER_PAGE_SIZE);
			}
		}
		watch.watched.clear();
		watch.written.clear();
		watch.end = watch.begin;
		m_watch_generation.fetch_add(1, std::memory_order_acq_rel);
	}
}

void BindlessTable::TakeWrittenRanges(std::vector<std::pair<uint64_t, uint64_t>>& ranges) {
	ranges.clear();
	std::scoped_lock lock {m_watch_mutex};
	m_heap_writes.store(false, std::memory_order_release);
	for (auto& watch: m_watches) {
		for (size_t page = 0; page < watch.written.size(); page++) {
			if (watch.written[page] == 0) {
				continue;
			}
			const auto address  = watch.begin + page * TRACKER_PAGE_SIZE;
			watch.written[page] = 0;
			// Watched again before the caller reads the entries: a later write is not lost.
			if (watch.watched[page] == 0) {
				watch.watched[page] = 1;
				m_page_manager.UpdatePageWatchers<true>(address, TRACKER_PAGE_SIZE);
			}
			if (!ranges.empty() && ranges.back().second == address) {
				ranges.back().second += TRACKER_PAGE_SIZE;
			} else {
				ranges.emplace_back(address, address + TRACKER_PAGE_SIZE);
			}
		}
	}
}

uint32_t BindlessTable::AllocateSlot(uint32_t binding) {
	if (binding == StorageImages2D) {
		auto& free = m_storage_free_slots;
		if (!free.empty() && m_scheduler.IsFree(free.front().first)) {
			const auto slot = free.front().second;
			free.pop_front();
			return slot;
		}
		return m_storage_next_slot < m_storage_images ? m_storage_next_slot++ : 0u;
	}
	if (binding >= ImageArrays) {
		return 0;
	}
	// KYTY_BINDLESS_SLOT_REUSE=0 (live, research): never reuse slots, to tell a stale slot
	// reference from other bugs (Wolverine runs 54-58: wrong textures, then device losts).
	static auto& reuse = Common::LiveSwitches::Get("KYTY_BINDLESS_SLOT_REUSE", 1);
	auto&        free  = m_free_slots[binding];
	if (reuse.load(std::memory_order_relaxed) != 0 && !free.empty() &&
	    m_scheduler.IsFree(free.front().first)) {
		const auto slot = free.front().second;
		free.pop_front();
		return slot;
	}
	if (m_next_slot[binding] >= m_images_per_array) {
		return 0;
	}
	return m_next_slot[binding]++;
}

namespace {
// The table whose slots the device-loss report searches (one bindless table per context).
BindlessTable* g_reported_table = nullptr;
} // namespace

void BindlessTable::ReportImageUsers(uint64_t image) {
	if (g_reported_table == nullptr) {
		return;
	}
	auto& table = *g_reported_table;
	int   found = 0;
	for (uint32_t binding = 0; binding < ImageArrays; binding++) {
		const auto& views  = table.m_slot_views[binding];
		const auto& images = table.m_slot_view_images[binding];
		for (uint32_t slot = 0; slot < views.size(); slot++) {
			const auto view = reinterpret_cast<uint64_t>(static_cast<VkImageView>(views[slot]));
			const bool now  = view != 0 && AddressBindingImageOfView(view) == image;
			const bool when_written = slot < images.size() && images[slot] == image;
			if (!now && !when_written) {
				continue;
			}
			found++;
			std::printf("      bindless slot %u/%u still holds a view of it%s", binding, slot,
			            now ? "" : " (its view handle now names another image)");
			for (auto& heap: table.m_heaps) {
				if (heap.binding != binding) {
					continue;
				}
				for (uint32_t key = 0; key < heap.slots.size(); key++) {
					if (heap.slots[key] == slot) {
						std::printf(" (heap base=0x%" PRIx64 " key=%u settled=%u)", heap.base, key,
						            heap.settled[key]);
					}
				}
			}
			std::printf("\n");
		}
	}
	if (found == 0) {
		std::printf("      no bindless slot holds a view of it (stale reference is elsewhere)\n");
	}
}

void BindlessTable::WriteSlot(uint32_t binding, uint32_t slot, vk::ImageView view,
                              vk::ImageLayout layout) {
	if (g_reported_table == nullptr && m_graphics.address_binding_report_enabled) {
		g_reported_table = this;
		AddressBindingSetImageUserReporter(&BindlessTable::ReportImageUsers);
	}
	if (binding < ImageArrays) {
		auto& views = m_slot_views[binding];
		if (views.size() <= slot) {
			views.resize(slot + 1);
		}
		views[slot] = view;
		if (m_graphics.address_binding_report_enabled) {
			auto& images = m_slot_view_images[binding];
			if (images.size() <= slot) {
				images.resize(slot + 1);
			}
			if (images[slot] != 0) {
				AddressBindingAnnotate(images[slot], "bindless slot overwritten");
			}
			images[slot] = AddressBindingImageOfView(
			    reinterpret_cast<uint64_t>(static_cast<VkImageView>(view)));
			if (images[slot] != 0) {
				AddressBindingAnnotate(images[slot], "bindless slot written");
			}
		}
	}
	const vk::DescriptorImageInfo info {nullptr, view, layout};
	vk::WriteDescriptorSet        write {};
	write.dstSet          = m_set;
	write.dstBinding      = binding;
	write.dstArrayElement = slot;
	write.descriptorCount = 1;
	write.descriptorType  = binding == StorageImages2D ? vk::DescriptorType::eStorageImage
	                                                   : vk::DescriptorType::eSampledImage;
	write.pImageInfo      = &info;
	m_graphics.device.updateDescriptorSets(1, &write, 0, nullptr);
}

// A host write of the mapped translation table was seen at once by command buffers recorded (or
// submitted) earlier: frames still in flight then sampled the placeholder of a key released for a
// later frame, or a new slot whose image upload was recorded after them (one or two frames of
// missing or garbage textures and decals while the camera moves). The ordered path records each
// change at the current point of the command stream instead.
static bool OrderedTranslations() {
	static auto& ordered = Common::LiveSwitches::Get("KYTY_BINDLESS_ORDERED", 1);
	return ordered.load(std::memory_order_relaxed) != 0;
}

void BindlessTable::FlushTranslations() {
	if (m_pending_translations.empty() || !m_scheduler.Active()) {
		return;
	}
	// The last change of an entry wins; runs of consecutive entries become one update.
	std::stable_sort(m_pending_translations.begin(), m_pending_translations.end(),
	                 [](const auto& a, const auto& b) { return a.first < b.first; });
	std::vector<std::pair<uint32_t, uint32_t>> changes;
	changes.reserve(m_pending_translations.size());
	for (const auto& change: m_pending_translations) {
		if (!changes.empty() && changes.back().first == change.first) {
			changes.back().second = change.second;
		} else {
			changes.push_back(change);
		}
	}
	m_pending_translations.clear();
	m_scheduler.EndRendering();
	auto command = m_scheduler.Current().Handle();
	vk::BufferMemoryBarrier before {};
	before.srcAccessMask       = vk::AccessFlagBits::eShaderRead;
	before.dstAccessMask       = vk::AccessFlagBits::eTransferWrite;
	before.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	before.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	before.buffer              = m_translation->Handle();
	before.offset              = 0;
	before.size                = VK_WHOLE_SIZE;
	command.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
	                        vk::PipelineStageFlagBits::eTransfer, {}, 0, nullptr, 1, &before, 0,
	                        nullptr);
	std::vector<uint32_t> run;
	for (size_t i = 0; i < changes.size();) {
		size_t end = i + 1;
		while (end < changes.size() && changes[end].first == changes[end - 1].first + 1 &&
		       end - i < 16384) {
			end++;
		}
		run.clear();
		for (size_t j = i; j < end; j++) {
			run.push_back(changes[j].second);
		}
		command.updateBuffer(m_translation->Handle(),
		                     vk::DeviceSize {changes[i].first} * sizeof(uint32_t),
		                     run.size() * sizeof(uint32_t), run.data());
		i = end;
	}
	vk::BufferMemoryBarrier after = before;
	after.srcAccessMask           = vk::AccessFlagBits::eTransferWrite;
	after.dstAccessMask           = vk::AccessFlagBits::eShaderRead;
	command.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
	                        vk::PipelineStageFlagBits::eAllCommands, {}, 0, nullptr, 1, &after, 0,
	                        nullptr);
	ReleaseQueuedSlots();
}

// KYTY_BINDLESS_FLUSH_TICK_FREE (live, default 1; with KYTY_BINDLESS_ORDERED): a released slot is
// reused only after the GPU is past the tick in which the key's new translation entry was
// recorded. Freed at the release tick, the slot could be handed to another texture while work
// recorded between the release and the flush (often the next tick) still mapped the key to it:
// a frame of the wrong texture. A dropped image's slot is still repointed to the placeholder at
// the release tick, before the texture cache's deferred destruction of the image.
void BindlessTable::FreeSlot(uint32_t binding, uint32_t slot, bool placeholder) {
	static auto& at_flush = Common::LiveSwitches::Get("KYTY_BINDLESS_FLUSH_TICK_FREE", 1);
	const bool   queue    = OrderedTranslations() && m_scheduler.Active() &&
	                   at_flush.load(std::memory_order_relaxed) != 0;
	if (placeholder) {
		const auto tick = m_scheduler.CurrentTick();
		m_scheduler.DeferOperation([this, binding, slot, tick, queue] {
			if (binding == StorageImages2D) {
				WriteSlot(binding, slot, m_storage_placeholder_view, vk::ImageLayout::eGeneral);
			} else {
				WriteSlot(binding, slot, m_placeholder_views[DroppedPlaceholder * ImageArrays + binding],
				          vk::ImageLayout::eShaderReadOnlyOptimal);
			}
			if (!queue) {
				FreeList(binding).emplace_back(tick, slot);
			}
		});
	}
	if (queue) {
		m_queued_frees.push_back({binding, slot, placeholder});
		return;
	}
	if (!placeholder) {
		FreeList(binding).emplace_back(m_scheduler.CurrentTick(), slot);
	}
}

void BindlessTable::ReleaseQueuedSlots() {
	const auto tick = m_scheduler.CurrentTick();
	for (const auto& free: m_queued_frees) {
		if (!free.placeholder) {
			FreeList(free.binding).emplace_back(tick, free.slot);
			continue;
		}
		// After the placeholder write queued at the release (deferred operations run in order).
		const auto binding = free.binding;
		const auto slot    = free.slot;
		m_scheduler.DeferOperation(
		    [this, binding, slot, tick] { FreeList(binding).emplace_back(tick, slot); });
	}
	m_queued_frees.clear();
}

void BindlessTable::SetTranslation(const Heap& heap, uint32_t key, uint32_t slot) {
	if (OrderedTranslations()) {
		m_pending_translations.emplace_back(heap.region + key, slot);
		return;
	}
	auto* translation = reinterpret_cast<uint32_t*>(m_translation->Mapped().data());
	translation[heap.region + key] = slot;
	if (!m_translation->IsCoherent()) {
		m_translation->Flush((heap.region + key) * sizeof(uint32_t), sizeof(uint32_t));
	}
}

void BindlessTable::RecordFeedbackSnapshot(CommandScheduler& scheduler) {
	// Regions are allocated contiguously after word 0, so one copy covers every heap.
	const vk::DeviceSize size = vk::DeviceSize {m_next_region} * sizeof(uint32_t);
	scheduler.EndRendering();
	auto                    command = scheduler.Current().Handle();
	vk::BufferMemoryBarrier flags_written {};
	flags_written.srcAccessMask       = vk::AccessFlagBits::eShaderWrite;
	flags_written.dstAccessMask       = vk::AccessFlagBits::eTransferRead;
	flags_written.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	flags_written.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	flags_written.buffer              = m_feedback->Handle();
	flags_written.offset              = 0;
	flags_written.size                = size;
	command.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
	                        vk::PipelineStageFlagBits::eTransfer, {}, 0, nullptr, 1,
	                        &flags_written, 0, nullptr);
	const vk::BufferCopy copy {0, 0, size};
	command.copyBuffer(m_feedback->Handle(), m_feedback_snapshot->Handle(), 1, &copy);
	auto snapshot_written          = flags_written;
	snapshot_written.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
	snapshot_written.dstAccessMask = vk::AccessFlagBits::eHostRead;
	snapshot_written.buffer        = m_feedback_snapshot->Handle();
	command.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
	                        vk::PipelineStageFlagBits::eHost, {}, 0, nullptr, 1,
	                        &snapshot_written, 0, nullptr);
	m_snapshot_tick = scheduler.CurrentTick();
}

bool BindlessTable::SnapshotReady(CommandScheduler& scheduler) {
	return m_snapshot_tick != 0 && scheduler.IsFree(m_snapshot_tick);
}

void BindlessTable::TakeRequests(const Heap& heap, std::vector<uint32_t>& keys,
                                 std::vector<ImageId>& used) {
	const auto* snapshot = reinterpret_cast<const uint32_t*>(m_feedback_snapshot->Mapped().data());
	auto*       feedback = reinterpret_cast<uint32_t*>(m_feedback->Mapped().data());
	m_feedback_snapshot->Invalidate(heap.region * sizeof(uint32_t),
	                                heap.entries * sizeof(uint32_t));
	bool cleared = false;
	// Research (2026-10-14, black smoke): KYTY_BINDLESS_KEY_LOG keys (hex) also log the feedback
	// flags shaders left for them and the translation entry the shaders read.
	// "all": every pending flag (requests), not the used ones.
	static const char*                 log_text = std::getenv("KYTY_BINDLESS_KEY_LOG");
	static const bool                  log_all  = log_text != nullptr && std::strcmp(log_text, "all") == 0;
	static const std::vector<uint32_t> logged   = [] {
		std::vector<uint32_t> list;
		for (const char* p = log_all ? nullptr : log_text; p != nullptr && *p != 0;) {
			char* end = nullptr;
			list.push_back(static_cast<uint32_t>(std::strtoul(p, &end, 16)));
			if (end == p) break;
			p = *end == ',' ? end + 1 : end;
		}
		return list;
	}();
	if (log_all) {
		for (uint32_t key = 0; key < heap.entries; key++) {
			if (snapshot[heap.region + key] == ShaderRecompiler::IR::BindlessFlagPending) {
				::printf("Bindless request: heap=0x%" PRIx64 "+0x%x binding=%u key=0x%x settled=%u\n",
				         heap.base, heap.table_offset, heap.binding, key, heap.settled[key]);
			}
		}
	}
	for (const auto key: logged) {
		if (key < heap.entries && snapshot[heap.region + key] != 0) {
			static std::atomic<uint32_t> lines {0};
			if (lines.fetch_add(1) < 400) {
				const auto* translation = reinterpret_cast<const uint32_t*>(m_translation->Mapped().data());
				::printf("Bindless feedback: heap=0x%" PRIx64 "+0x%x binding=%u region=%u key=0x%x flag=%u "
				         "settled=%u slot=%u translation=0x%x\n",
				         heap.base, heap.table_offset, heap.binding, heap.region, key,
				         snapshot[heap.region + key], heap.settled[key], heap.slots[key],
				         translation[heap.region + key]);
			}
		}
	}
	for (uint32_t key = 0; key < heap.entries; key++) {
		const auto flag = snapshot[heap.region + key];
		if (flag != 0) {
			// A flag the GPU sets again after the snapshot may be lost here; a key that is
			// still pending (or in use) is flagged again by the next draw that samples it.
			feedback[heap.region + key] = 0;
			if (flag == ShaderRecompiler::IR::BindlessFlagPending) {
				keys.push_back(key);
			} else if (flag == ShaderRecompiler::IR::BindlessFlagUsed && heap.images[key]) {
				used.push_back(heap.images[key]);
			}
			cleared = true;
		}
	}
	if (cleared) {
		m_feedback->Flush(heap.region * sizeof(uint32_t), heap.entries * sizeof(uint32_t));
	}
}

void BindlessTable::DrainWatchdogReports(uint64_t frame) {
	using ShaderRecompiler::IR::WatchdogReportBase;
	using ShaderRecompiler::IR::WatchdogReportPhis;
	using ShaderRecompiler::IR::WatchdogReportSlots;
	using ShaderRecompiler::IR::WatchdogReportWords;
	static_assert(WatchdogReportBase + WatchdogReportWords * (WatchdogReportSlots + 1) <=
	              TranslationEntries);
	auto* feedback = reinterpret_cast<uint32_t*>(m_feedback->Mapped().data());
	if (!m_feedback->IsCoherent()) {
		m_feedback->Invalidate(WatchdogReportBase * sizeof(uint32_t),
		                       WatchdogReportWords * (WatchdogReportSlots + 1) * sizeof(uint32_t));
	}
	const auto count = feedback[WatchdogReportBase];
	if (count == 0) {
		return;
	}
	const auto stored = std::min(count, WatchdogReportSlots);
	// Per (shader, loop): how many tripped, and the first report in full.
	std::unordered_map<uint64_t, std::pair<uint32_t, const uint32_t*>> loops;
	for (uint32_t n = 0; n < stored; n++) {
		const auto* report = feedback + WatchdogReportBase + WatchdogReportWords * (n + 1);
		if ((report[0] >> 16u) != 0x5744u) {
			continue;
		}
		const auto key = (static_cast<uint64_t>(report[2]) << 32u | report[1]) ^
		                 (static_cast<uint64_t>(report[0] & 0xffffu) << 48u);
		auto& entry = loops[key];
		if (entry.first++ == 0) {
			entry.second = report;
		}
	}
	static std::atomic<uint32_t> logged = 0;
	LOGF("WatchdogTrips: frame=%" PRIu64 " reports=%u loops=%zu\n", frame, count, loops.size());
	for (const auto& [key, entry]: loops) {
		if (logged.fetch_add(1) >= 256) {
			break;
		}
		const auto* r = entry.second;
		std::string phis;
		for (uint32_t half = 0; half < 2; half++) {
			phis += half == 0 ? " lo:" : " hi:";
			for (uint32_t i = 0; i < WatchdogReportPhis; i++) {
				phis += fmt::format(" {:x}", r[16 + half * WatchdogReportPhis + i]);
			}
		}
		LOGF("WatchdogTrip: hash=0x%08x%08x loop_block=%u trips=%u counter=%u group=%u lane=%u "
		     "ud=[%08x %08x %08x %08x %08x %08x %08x %08x]%s\n",
		     r[2], r[1], r[0] & 0xffffu, entry.first, r[3], r[4], r[5], r[6], r[7], r[8], r[9],
		     r[10], r[11], r[12], r[13], phis.c_str());
	}
	feedback[WatchdogReportBase] = 0;
	if (!m_feedback->IsCoherent()) {
		m_feedback->Flush(WatchdogReportBase * sizeof(uint32_t), sizeof(uint32_t));
	}
}

void BindlessTable::AddImageReference(ImageId id, Heap& heap, uint32_t key) {
	m_image_refs[ImageKey(id)].emplace_back(&heap, key);
}

bool BindlessTable::ReleaseKey(Heap& heap, uint32_t key) {
	const auto id      = heap.images[key];
	bool       no_refs = false;
	if (id) {
		const auto found = m_image_refs.find(ImageKey(id));
		if (found != m_image_refs.end()) {
			std::erase(found->second, std::pair<Heap*, uint32_t> {&heap, key});
			const bool in_heap = std::any_of(found->second.begin(), found->second.end(),
			                                 [&](const auto& ref) { return ref.first == &heap; });
			if (!in_heap) {
				std::erase(heap.resolved, id);
				// Erasing shifts the images RenderExecutor::CommitBindings counted as checked.
				heap.checked_generation = 0;
			}
			if (found->second.empty()) {
				m_image_refs.erase(found);
				no_refs = true;
			}
		}
	}
	if (heap.slots[key] != 0) {
		// Command buffers recorded so far may still sample the slot; with ordered translations,
		// so may the ones recorded until the key's new entry reaches the command stream.
		FreeSlot(heap.binding, heap.slots[key], false);
	}
	heap.slots[key]   = 0;
	heap.settled[key] = 0;
	heap.images[key]  = {};
	SetTranslation(heap, key, ShaderRecompiler::IR::BindlessPending);
	return no_refs;
}

void BindlessTable::OnImageUnregistered(ImageId id) {
	const auto found = m_image_refs.find(ImageKey(id));
	if (found == m_image_refs.end()) {
		return;
	}
	// KYTY_BINDLESS_RESOLVE_STATS=1: every 5 s, images dropped while keys used them, and the keys
	// sent back to pending (research 2026-10-11, textures flickering while the camera moves).
	{
		static auto&                 stats_on = Common::LiveSwitches::Get("KYTY_BINDLESS_RESOLVE_STATS", 0);
		static uint64_t              images = 0, keys = 0;
		static auto                  report = std::chrono::steady_clock::now();
		if (stats_on.load(std::memory_order_relaxed) != 0) {
			images++;
			keys += found->second.size();
			const auto now = std::chrono::steady_clock::now();
			if (now - report >= std::chrono::seconds(5)) {
				std::printf("Bindless unregister (5 s): %" PRIu64 " images, %" PRIu64 " keys back to pending\n",
				            images, keys);
				std::fflush(stdout);
				images = keys = 0;
				report = now;
			}
		}
	}
	// KYTY_BINDLESS_LATE_PLACEHOLDER (live, default 1; with KYTY_BINDLESS_ORDERED): the slot keeps
	// the dropped image's view until the GPU is past the work recorded so far, which was recorded
	// with the key still mapped to it (the image itself is destroyed by a deferred operation queued
	// after this one). Writing the placeholder at once made those frames sample it: one or two
	// frames of flat or white textures each time the texture cache merged or re-created an image.
	static auto& late = Common::LiveSwitches::Get("KYTY_BINDLESS_LATE_PLACEHOLDER", 1);
	const bool   late_placeholder = late.load(std::memory_order_relaxed) != 0 &&
	                              OrderedTranslations() && m_scheduler.Active();
	for (const auto& [heap, key]: found->second) {
		if (heap->slots[key] != 0 && late_placeholder) {
			FreeSlot(heap->binding, heap->slots[key], true);
		} else if (heap->slots[key] != 0) {
			if (heap->binding == StorageImages2D) {
				WriteSlot(heap->binding, heap->slots[key], m_storage_placeholder_view,
				          vk::ImageLayout::eGeneral);
			} else {
				WriteSlot(heap->binding, heap->slots[key],
				          m_placeholder_views[DroppedPlaceholder * ImageArrays + heap->binding],
				          vk::ImageLayout::eShaderReadOnlyOptimal);
			}
			// Command buffers recorded so far may still sample the slot.
			FreeList(heap->binding).emplace_back(m_scheduler.CurrentTick(), heap->slots[key]);
		}
		heap->images[key]  = {};
		heap->slots[key]   = 0;
		heap->settled[key] = 0;
		SetTranslation(*heap, key, ShaderRecompiler::IR::BindlessPending);
		std::erase(heap->resolved, id);
		static auto& eager = Common::LiveSwitches::Get("KYTY_BINDLESS_EAGER", 1);
		if (eager.load(std::memory_order_relaxed) != 0 && m_eager_keys.size() < 65536) {
			m_eager_keys.emplace_back(heap, key);
		}
	}
	m_image_refs.erase(found);
}

} // namespace Libs::Graphics
