#include "graphics/host_gpu/renderer/renderDraw.h"

#include "common/assert.h"
#include "common/common.h"
#include "common/emulatorConfig.h"
#include "common/file.h"
#include "common/liveSwitches.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "common/stringUtils.h"
#include "common/threads.h"
#include "graphics/guest_gpu/gpu_defs.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/guest_gpu/tile.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/colorRenderTarget.h"
#include "graphics/host_gpu/renderer/commandRecorder.h"
#include "graphics/host_gpu/renderer/debug.h"
#include "graphics/host_gpu/renderer/depthRenderTarget.h"
#include "graphics/host_gpu/renderer/image/textureCommon.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/host_gpu/renderer/pipeline/shaderResourceBarrier.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/shader/recompiler/BufferFormat.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/shader/shader.h"
#include "kernel/eventQueue.h"
#include "kernel/memory.h"
#include "kernel/pthread.h"
#include "libs/errno.h"

#include <string>
#include <set>
#include <unordered_set>
#include <algorithm>
#include <array>
#include <chrono>
#include <cinttypes>
#include <type_traits>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <unordered_map>
#include <vector>

namespace Libs::Graphics {

std::pair<int32_t, uint32_t> ResolveDrawOffsets(uint32_t index_offset,
	                                           const ShaderVertexInputInfo& vs_input_info) {
	auto     vertex_offset   = static_cast<int32_t>(index_offset);
	uint32_t instance_offset = 0;
	if (!vs_input_info.fetch_embedded) {
		return {vertex_offset, instance_offset};
	}

	EXIT_IF(!vs_input_info.stage);
	const auto& program   = *vs_input_info.stage.program;
	const auto& resources = *vs_input_info.stage.resources;
	if (index_offset == 0 &&
	    program.info.vertex_offset_sgpr >= static_cast<int32_t>(program.user_data_base)) {
		const auto index =
		    static_cast<uint32_t>(program.info.vertex_offset_sgpr) - program.user_data_base;
		if (index < resources.user_data.size()) {
			vertex_offset = static_cast<int32_t>(resources.user_data[index]);
		}
	}
	if (program.info.instance_offset_sgpr >= static_cast<int32_t>(program.user_data_base)) {
		const auto index =
		    static_cast<uint32_t>(program.info.instance_offset_sgpr) - program.user_data_base;
		if (index < resources.user_data.size()) {
			instance_offset = resources.user_data[index];
		}
	}

	return {vertex_offset, instance_offset};
}

static std::atomic<uint32_t> g_draw_state_log_count   = 0;
static std::atomic<uint32_t> g_draw_input_log_count   = 0;
static std::atomic<uint32_t> g_mrt_state_log_count    = 0;

static std::atomic<uint32_t> g_framebuffer_skip_log_count = 0;

static float ConvertPolygonOffsetConstantFactor(float guest_factor, const HW::PolyOffset& offset,
                                                vk::Format host_depth_format) {
	if (offset.db_is_float_fmt) {
		return guest_factor;
	}

	int host_depth_bits = 0;
	switch (host_depth_format) {
		case vk::Format::eD16Unorm:
		case vk::Format::eD16UnormS8Uint: host_depth_bits = 16; break;
		case vk::Format::eD24UnormS8Uint: host_depth_bits = 24; break;
		default:
			// A fixed-point guest bias cannot be represented exactly by a floating-point host
			// attachment without VK_EXT_depth_bias_control.
			return guest_factor;
	}
	return std::ldexp(guest_factor, host_depth_bits + offset.neg_num_db_bits);
}

static const char* RenderColorTypeName(const RenderColorInfo& color) {
	return color.image_id ? "RenderTexture" : "NoColorOutput";
}

static void LogFramebufferSkip(const char* draw_name, const RenderColorInfo& color,
                               const RenderDepthInfo& depth, const CommandBuffer& buffer,
                               uint32_t index_count, uint32_t flags) {
	const auto& ctx  = buffer.GetRegisters();
	const auto& ucfg = buffer.GetUserConfig();
	if (!graphics_debug_dump_enabled()) {
		return;
	}

	auto log_id = g_framebuffer_skip_log_count.fetch_add(1, std::memory_order_relaxed);
	if (log_id >= 128) {
		return;
	}

	LOGF(
	    "DrawFramebufferSkip[%u]: %s color=%s color_addr=0x%010" PRIx64 " color_size=0x%016" PRIx64
	    " color_image=%s depth_format=%s depth_image=%s depth_vaddr_num=%d target_mask=0x%08" PRIx32
	    " prim=%u index_count=%u flags=0x%08" PRIx32 "\n",
	    log_id, draw_name, RenderColorTypeName(color), color.desc.info.data.address,
	    color.desc.info.data.size, color.image_id ? "yes" : "no",
	    vk::to_string(depth.desc.view_info.format).c_str(), depth.image_id ? "yes" : "no",
	    static_cast<int>(!depth.desc.info.data.Empty()) +
	        static_cast<int>(depth.desc.info.HasStencil()),
	    ctx.GetRenderTargetMask(), static_cast<uint32_t>(ucfg.GetPrimType()), index_count, flags);
}

static void LogMrtState(const char* draw_name, const CommandBuffer& buffer,
                        const ShaderPixelInputInfo& ps_input_info) {
	const auto& ctx            = buffer.GetRegisters();
	const auto& sh_regs        = ctx.GetShaderRegisters();
	const auto  rt_mask        = ctx.GetRenderTargetMask();
	const auto  cb_shader_mask = sh_regs.m_cbShaderMask;
	const auto& bc0            = ctx.GetBlendControl(0);

	auto log_id = g_mrt_state_log_count.fetch_add(1);
	if (log_id >= 32) {
		return;
	}

	LOGF("MrtState[%u]: %s rt_mask=0x%08" PRIx32 " cb_shader_mask=0x%08" PRIx32
	     " blend0=%s src=%u dst=%u alpha_src=%u alpha_dst=%u sep_alpha=%s\n",
	     log_id, draw_name, rt_mask, cb_shader_mask, bc0.enable ? "true" : "false",
	     bc0.color_srcblend, bc0.color_destblend, bc0.alpha_srcblend, bc0.alpha_destblend,
	     bc0.separate_alpha_blend ? "true" : "false");

	for (uint32_t i = 0; i < 8; i++) {
		const auto& rt  = ctx.GetRenderTarget(i);
		const auto& bc  = ctx.GetBlendControl(i);
		const auto  ctm = (rt_mask >> (i * 4u)) & 0x0fu;
		const auto  csm = (cb_shader_mask >> (i * 4u)) & 0x0fu;

		if (rt.base.addr == 0 && ps_input_info.target_output_mode[i] == 0 && ctm == 0 && csm == 0 &&
		    !bc.enable) {
			continue;
		}

		LOGF("MrtState[%u]: slot=%u addr=0x%010" PRIx64
		     " target_mask=0x%x shader_mask=0x%x out_mode=%u"
		     " fmt=0x%08" PRIx32 " nfmt=0x%08" PRIx32 " order=0x%08" PRIx32
		     " width=%u height=%u tile=%u"
		     " blend=%s src=%u dst=%u alpha_src=%u alpha_dst=%u\n",
		     log_id, i, rt.base.addr, ctm, csm, ps_input_info.target_output_mode[i],
		     static_cast<uint32_t>(rt.info.format), static_cast<uint32_t>(rt.info.channel_type),
		     static_cast<uint32_t>(rt.info.channel_order), rt.attrib2.width + 1,
		     rt.attrib2.height + 1, static_cast<uint32_t>(rt.attrib3.tile_mode),
		     bc.enable ? "true" : "false", bc.color_srcblend, bc.color_destblend, bc.alpha_srcblend,
		     bc.alpha_destblend);
	}
}

static void LogDrawTargetState(const char* draw_name, const RenderColorInfo& color,
                               const RenderDepthInfo& depth, const CommandBuffer& buffer,
                               const ShaderPixelInputInfo& ps_input_info, uint32_t index_count,
                               uint32_t flags) {
	const auto& ctx  = buffer.GetRegisters();
	const auto& ucfg = buffer.GetUserConfig();
	if (!color.image_id) {
		return;
	}

	auto log_id = g_draw_state_log_count.fetch_add(1);
	if (log_id >= 192) {
		return;
	}

	const auto& cc             = ctx.GetColorControl();
	const auto& bc             = ctx.GetBlendControl(color.target_slot);
	const auto& dc             = ctx.GetDepthControl();
	const auto& vp             = ctx.GetScreenViewport();
	const auto& vp0            = vp.viewports[0];
	const auto& ps_resources   = ps_input_info.stage.program->info;
	const auto  sampled_images = std::count_if(
	    ps_resources.images.begin(), ps_resources.images.end(), [](const auto& image) {
		    return image.resource_class == ShaderRecompiler::IR::ImageResourceClass::Sampled;
	    });

	const auto extent = color.Extent();
	const auto sc     = calc_final_scissor(vp, ctx.GetScanModeControl(), extent, 0);

	LOGF(
	    "DrawTargetState[%u]: frame=%d %s target=%s addr=0x%010" PRIx64
	    " extent=%ux%u prim=%u index_count=%u flags=0x%08" PRIx32 " color_mask=0x%08" PRIx32
	    " cc_mode=%u cc_op=0x%02x"
	    " blend=%s src=%u dst=%u comb=%u ps_tex=%d sampled=%d storage=%d ps_kill=%s target_mode0=%u"
	    " depth_test=%s depth_write=%s depth_func=%u depth_clear=%s viewport=(%.1f,%.1f %.1fx%.1f) "
	    "scissor=(%d,%d)-(%d,%d)\n",
	    log_id, buffer.GetContext().GetGpu().GetFrameNum(), draw_name, RenderColorTypeName(color),
	    color.desc.info.data.address, extent.width, extent.height,
	    static_cast<uint32_t>(ucfg.GetPrimType()), index_count, flags, ctx.GetRenderTargetMask(),
	    cc.mode, cc.op,
	    bc.enable ? "true" : "false", bc.color_srcblend, bc.color_destblend, bc.color_comb_fcn,
	    static_cast<int>(ps_resources.images.size()), static_cast<int>(sampled_images),
	    static_cast<int>(ps_resources.images.size() - sampled_images),
	    ps_input_info.ps_pixel_kill_enable ? "true" : "false", ps_input_info.target_output_mode[0],
	    dc.z_enable ? "true" : "false", dc.z_write_enable ? "true" : "false", dc.zfunc,
	    depth.depth_clear_enable ? "true" : "false", vp0.xoffset - vp0.xscale,
	    vp0.yoffset - vp0.yscale, vp0.xscale * 2.0f, vp0.yscale * 2.0f, sc.left, sc.top, sc.right,
	    sc.bottom);

	LogMrtState(draw_name, buffer, ps_input_info);
}

static void LogDrawInputState(const CommandBuffer& buffer, const RenderColorInfo& color,
                              const ShaderVertexInputInfo& vs_input_info,
                              uint32_t index_type_and_size, uint32_t index_count,
                              const void* index_addr) {
	auto log_id = g_draw_input_log_count.fetch_add(1);
	if (log_id >= 64) {
		return;
	}

	LOGF("DrawInputState[%u]: frame=%d target=%s addr=0x%010" PRIx64
	     " index_type=%u index_count=%u index_addr=0x%016" PRIx64
	     " vs_resources=%d vs_buffers=%d\n",
	     log_id, buffer.GetContext().GetGpu().GetFrameNum(), RenderColorTypeName(color),
	     color.desc.info.data.address, index_type_and_size, index_count,
	     reinterpret_cast<uint64_t>(index_addr), vs_input_info.resources_num,
	     vs_input_info.buffers_num);

	for (int bi = 0; bi < vs_input_info.buffers_num; bi++) {
		const auto& b = vs_input_info.buffers[bi];
		LOGF("DrawInputState[%u]: vb[%d] addr=0x%010" PRIx64
		     " stride=%u records=%u fetch_index=%u\n",
		     log_id, bi, b.addr, b.stride, b.num_records, b.fetch_index);

		const auto* bytes = reinterpret_cast<const uint8_t*>(b.addr);
		if (bytes != nullptr && b.stride != 0) {
			const uint32_t records = std::min<uint32_t>(b.num_records, 4u);
			for (uint32_t rec = 0; rec < records; rec++) {
				const auto* rec_bytes = bytes + static_cast<uint64_t>(rec) * b.stride;
				const auto  dword_num = std::min<uint32_t>(b.stride / 4u, 12u);
				uint32_t    raw[12]   = {};
				float       flt[12]   = {};
				for (uint32_t i = 0; i < dword_num; i++) {
					std::memcpy(&raw[i], rec_bytes + i * 4u, sizeof(raw[i]));
					std::memcpy(&flt[i], rec_bytes + i * 4u, sizeof(flt[i]));
				}
				LOGF("DrawInputState[%u]: vb[%d].rec[%u] stride=%u dwords=%u raw=%08" PRIx32
				     " %08" PRIx32 " %08" PRIx32 " %08" PRIx32 " %08" PRIx32 " %08" PRIx32
				     " %08" PRIx32 " %08" PRIx32 " %08" PRIx32
				     " f=(%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f)\n",
				     log_id, bi, rec, b.stride, dword_num, raw[0], raw[1], raw[2], raw[3], raw[4],
				     raw[5], raw[6], raw[7], raw[8], flt[0], flt[1], flt[2], flt[3], flt[4], flt[5],
				     flt[6], flt[7], flt[8]);

				for (int ai = 0; ai < vs_input_info.resources_num; ai++) {
					const auto& r  = vs_input_info.resources[ai];
					const auto& rd = vs_input_info.resources_dst[ai];
					if (rd.buffer_index != bi) {
						continue;
					}
					const auto offset = static_cast<uint32_t>(r.Base48() - b.addr);
					if (offset + 4u <= b.stride &&
					    r.Format() == Prospero::BufferFormat::k8_8_8_8UNorm) {
						uint32_t packed = 0;
						std::memcpy(&packed, rec_bytes + offset, sizeof(packed));
						const auto r8 = (packed >> 0u) & 0xffu;
						const auto g8 = (packed >> 8u) & 0xffu;
						const auto b8 = (packed >> 16u) & 0xffu;
						const auto a8 = (packed >> 24u) & 0xffu;
						LOGF("DrawInputState[%u]: vb[%d].rec[%u].attr[%d] dst=v%d fmt=56 "
						     "rgba8=%02" PRIx32 "%02" PRIx32 "%02" PRIx32 "%02" PRIx32
						     " rgba=(%.3f,%.3f,%.3f,%.3f)\n",
						     log_id, bi, rec, ai, rd.register_start, r8, g8, b8, a8,
						     static_cast<double>(r8) / 255.0, static_cast<double>(g8) / 255.0,
						     static_cast<double>(b8) / 255.0, static_cast<double>(a8) / 255.0);
					}
				}
			}
		}

		for (int ai = 0; ai < vs_input_info.resources_num; ai++) {
			const auto& r  = vs_input_info.resources[ai];
			const auto& rd = vs_input_info.resources_dst[ai];
			if (rd.buffer_index != bi) {
				continue;
			}
			LOGF("DrawInputState[%u]: attr[%d] offset=%u dst=v%d regs=%d fetch_index=%u "
			     "sharp=%08" PRIx32 " %08" PRIx32 " %08" PRIx32 " %08" PRIx32 "\n",
			     log_id, ai, static_cast<uint32_t>(r.Base48() - b.addr), rd.register_start,
			     rd.registers_num, rd.fetch_index, r.fields[0], r.fields[1], r.fields[2], r.fields[3]);
		}
	}
}

static void SetGraphicsDynamicParams(const CommandBuffer& buffer, const CommandRecorder& vk_buffer,
                                     const ShaderVertexInputInfo& vs_input_info,
                                     const RenderDepthInfo& depth, const RenderState& rendering) {
	KYTY_PROFILER_FUNCTION();

	const auto& ctx = buffer.GetRegisters();
	const auto&        vp  = ctx.GetScreenViewport();
	const vk::Extent2D framebuffer_extent {rendering.width, rendering.height};
	const auto& outputs = vs_input_info.stage.program->info.outputs;
	const bool  indexed_viewports =
	    std::any_of(outputs.begin(), outputs.end(), [](const auto& output) {
		    return output.kind == ShaderRecompiler::IR::StageOutputKind::ViewportIndex;
	    });
	constexpr uint32_t viewport_slots = std::size(HW::ScreenViewport {}.viewports);
	std::array<vk::Viewport, viewport_slots> viewports {};
	std::array<vk::Rect2D, viewport_slots>   scissors {};
	const uint32_t viewport_count = indexed_viewports ? viewport_slots : 1;
	for (uint32_t i = 0; i < viewport_count; i++) {
		const auto& guest    = vp.viewports[i];
		auto&       viewport = viewports[i];
		if (ctx.GetClipControl().clip_disable) {
			const auto& limits = buffer.GetGraphics().GetPhysicalDeviceProperties().limits;
			viewport.width  = static_cast<float>(std::min(limits.maxViewportDimensions[0], 16384u));
			viewport.height = static_cast<float>(std::min(limits.maxViewportDimensions[1], 16384u));
		} else {
			viewport.x      = guest.xoffset - guest.xscale;
			viewport.y      = guest.yoffset - guest.yscale;
			viewport.width  = guest.xscale * 2.0f;
			viewport.height = guest.yscale * 2.0f;
		}
		viewport.minDepth =
		    guest.zoffset - (ctx.GetClipControl().dx_clip_space ? 0.0f : guest.zscale);
		viewport.maxDepth = guest.zscale + guest.zoffset;

		const auto final_scissor =
		    calc_final_scissor(vp, ctx.GetScanModeControl(), framebuffer_extent, i);
		auto& scissor  = scissors[i];
		scissor.offset = {final_scissor.left, final_scissor.top};
		scissor.extent = {static_cast<uint32_t>(final_scissor.right - final_scissor.left),
		                  static_cast<uint32_t>(final_scissor.bottom - final_scissor.top)};
		if (viewport.width == 0.0f) {
			// Keep empty slots at their guest index; Vulkan requires a positive viewport width.
			viewport.width = 1.0f;
			scissor.extent = {0, 0};
		}
	}
	// KYTY_STATE_CACHE: record only the dynamic state that differs from what this command buffer
	// already holds (see CommandBuffer::DynamicState).
	static auto& state_cache = Common::LiveSwitches::Get("KYTY_STATE_CACHE", 0);
	auto&        cached      = buffer.GetDynamicState();
	const bool   use_cache   = state_cache.load(std::memory_order_relaxed) != 0;
	const bool   known       = use_cache && cached.valid;
	if (!use_cache) {
		cached = {};
	}
	const auto viewport_bytes = sizeof(vk::Viewport) * viewport_count;
	const auto scissor_bytes  = sizeof(vk::Rect2D) * viewport_count;
	if (!known || cached.viewport_count != viewport_count ||
	    std::memcmp(cached.viewports.data(), viewports.data(), viewport_bytes) != 0) {
		vk_buffer.setViewportWithCount(viewport_count, viewports.data());
	}
	if (!known || cached.viewport_count != viewport_count ||
	    std::memcmp(cached.scissors.data(), scissors.data(), scissor_bytes) != 0) {
		vk_buffer.setScissorWithCount(viewport_count, scissors.data());
	}
	if (use_cache) {
		cached.viewport_count = viewport_count;
		std::memcpy(cached.viewports.data(), viewports.data(), viewport_bytes);
		std::memcpy(cached.scissors.data(), scissors.data(), scissor_bytes);
	}

	float line_width = ctx.GetLineWidth();
	if (line_width != 1.0f) {
		static bool logged = false;
		if (!logged) {
			LOGF("Render: temporary: clamping Vulkan line width %f to 1.0 because wideLines is "
			     "not enabled\n",
			     line_width);
			logged = true;
		}
		line_width = 1.0f;
	}
	if (!known || cached.line_width != line_width) {
		vk_buffer.setLineWidth(line_width);
	}
	const auto&      blend = ctx.GetBlendColor();
	const std::array blend_constants {blend.red, blend.green, blend.blue, blend.alpha};
	if (!known || cached.blend_constants != blend_constants) {
		vk_buffer.setBlendConstants(blend_constants.data());
	}
	if (!known || cached.depth_test_enable != depth.depth_test_enable) {
		vk_buffer.setDepthTestEnable(depth.depth_test_enable ? VK_TRUE : VK_FALSE);
	}
	if (!known || cached.depth_write_enable != depth.depth_write_enable) {
		vk_buffer.setDepthWriteEnable(depth.depth_write_enable ? VK_TRUE : VK_FALSE);
	}
	// KYTY_LOCAL_HACK (research): KYTY_EQUAL_RELAX=1 turns depth EQUAL into GREATER_OR_EQUAL
	// (reverse Z), to test whether material passes lose pixels to inexact prepass depth.
	static const bool equal_relax = [] {
		const char* value = std::getenv("KYTY_EQUAL_RELAX");
		return value != nullptr && value[0] == '1';
	}();
	static const bool equal_always = [] {
		const char* value = std::getenv("KYTY_EQUAL_RELAX");
		return value != nullptr && value[0] == '2';
	}();
	const auto compare_op = (equal_relax || equal_always) &&
	                                depth.depth_compare_op == vk::CompareOp::eEqual
	                            ? (equal_always ? vk::CompareOp::eAlways
	                                            : vk::CompareOp::eGreaterOrEqual)
	                            : depth.depth_compare_op;
	if (!known || cached.depth_compare_op != compare_op) {
		vk_buffer.setDepthCompareOp(compare_op);
	}

	const auto& mode              = ctx.GetModeControl();
	const auto& poly_offset       = ctx.GetPolyOffset();
	const bool  use_front         = mode.poly_offset_front_enable && !mode.cull_front;
	const bool  use_back          = mode.poly_offset_back_enable && !mode.cull_back;
	const bool  depth_bias_enable = use_front || use_back;
	if (!known || cached.depth_bias_enable != depth_bias_enable) {
		vk_buffer.setDepthBiasEnable(depth_bias_enable ? VK_TRUE : VK_FALSE);
	}
	std::array<float, 3> depth_bias {};
	if (depth_bias_enable) {
		// Vulkan has one bias for both faces. Prefer a visible front face when both are enabled.
		const float guest_constant_factor =
		    use_front ? poly_offset.front_offset : poly_offset.back_offset;
		const float constant_factor = ConvertPolygonOffsetConstantFactor(
		    guest_constant_factor, poly_offset, depth.desc.view_info.format);
		const float slope_factor =
		    (use_front ? poly_offset.front_scale : poly_offset.back_scale) / 16.0f;
		depth_bias = {constant_factor, poly_offset.clamp, slope_factor};
		{
			// KYTY_LOCAL_HACK: report each distinct depth-bias setup once.
			static std::mutex                    bias_log_mutex;
			static std::vector<std::array<float, 6>> bias_seen;
			const std::array<float, 6> key {guest_constant_factor, slope_factor, poly_offset.clamp,
			                                static_cast<float>(poly_offset.neg_num_db_bits),
			                                poly_offset.db_is_float_fmt ? 1.0f : 0.0f,
			                                static_cast<float>(depth.desc.view_info.format)};
			std::lock_guard bias_lock(bias_log_mutex);
			if (bias_seen.size() < 64 &&
			    std::find(bias_seen.begin(), bias_seen.end(), key) == bias_seen.end()) {
				bias_seen.push_back(key);
				printf("DEPTHBIAS: host=%s float_fmt=%d neg_bits=%d offset=%g -> %g slope=%g clamp=%g "
				       "front=%d back=%d\n",
				       vk::to_string(depth.desc.view_info.format).c_str(),
				       poly_offset.db_is_float_fmt ? 1 : 0, poly_offset.neg_num_db_bits,
				       guest_constant_factor, constant_factor, slope_factor, poly_offset.clamp,
				       use_front ? 1 : 0, use_back ? 1 : 0);
			}
		}
		// The bias values only matter while biasing is enabled, so they are compared then only.
		if (!known || !cached.depth_bias_enable || cached.depth_bias != depth_bias) {
			vk_buffer.setDepthBias(constant_factor, poly_offset.clamp, slope_factor);
		}
	}

	if (!known || cached.stencil_test_enable != depth.stencil_test_enable) {
		vk_buffer.setStencilTestEnable(depth.stencil_test_enable ? VK_TRUE : VK_FALSE);
	}
	if (depth.stencil_test_enable) {
		const auto set_stencil = [&](vk::StencilFaceFlagBits face, const vk::StencilOpState& state,
		                             const vk::StencilOpState& previous) {
			const bool all = !known || !cached.stencil_test_enable;
			if (all || state.failOp != previous.failOp || state.passOp != previous.passOp ||
			    state.depthFailOp != previous.depthFailOp ||
			    state.compareOp != previous.compareOp) {
				vk_buffer.setStencilOp(face, state.failOp, state.passOp, state.depthFailOp,
				                       state.compareOp);
			}
			if (all || state.compareMask != previous.compareMask) {
				vk_buffer.setStencilCompareMask(face, state.compareMask);
			}
			if (all || state.writeMask != previous.writeMask) {
				vk_buffer.setStencilWriteMask(face, state.writeMask);
			}
			if (all || state.reference != previous.reference) {
				vk_buffer.setStencilReference(face, state.reference);
			}
		};
		set_stencil(vk::StencilFaceFlagBits::eFront, depth.stencil_front, cached.stencil_front);
		set_stencil(vk::StencilFaceFlagBits::eBack, depth.stencil_back, cached.stencil_back);
	}
	if (use_cache) {
		cached.valid              = true;
		cached.line_width         = line_width;
		cached.blend_constants    = blend_constants;
		cached.depth_test_enable  = depth.depth_test_enable;
		cached.depth_write_enable = depth.depth_write_enable;
		cached.depth_compare_op   = compare_op;
		// Keep the last bias values recorded: they still hold while biasing is off.
		if (depth_bias_enable) {
			cached.depth_bias = depth_bias;
		}
		cached.depth_bias_enable = depth_bias_enable;
		// Likewise the stencil state recorded last stays in the command buffer while it is off.
		if (depth.stencil_test_enable) {
			cached.stencil_front = depth.stencil_front;
			cached.stencil_back  = depth.stencil_back;
		}
		cached.stencil_test_enable = depth.stencil_test_enable;
	}

#if defined(__APPLE__)
	// MoltenVK has no VK_EXT_color_write_enable; the pipeline is created without the
	// eColorWriteEnableEXT dynamic state and relies on the static colorWriteMask instead.
#else
	vk::Bool32 enable[RENDER_COLOR_ATTACHMENTS_MAX] = {};
	for (uint32_t slot = 0; slot < rendering.num_color_attachments; slot++) {
		enable[slot] = rendering.color_attachments[slot].image_view != nullptr;
	}
	if (rendering.num_color_attachments != 0) {
		vk_buffer.setColorWriteEnableEXT(rendering.num_color_attachments, enable);
	}
#endif
}

static bool DrawHasValidVertexShader(const HW::Shader& sh_ctx) {

	const auto& vs = sh_ctx.GetVs();
	return vs.es_regs.data_addr != 0;
}

static bool PixelShaderHasDepthOrCoverageSideEffects(const HW::ShaderRegisters& sh_regs) {
	const auto& db = sh_regs.db_shader_control;
	return db.shader_kill_enable || db.shader_z_export_enable || db.shader_mask_export_enable ||
	       db.shader_dual_export_enable || db.shader_execute_on_noop;
}

struct DrawRenderState {
	RenderDepthInfo       depth_info;
	RenderColorInfo       color_info[RENDER_COLOR_ATTACHMENTS_MAX] = {};
	uint32_t              color_count                              = 0;
	bool                  ps_active                                = true;
	std::array<ShaderVertexInputInfo, 3> vertex_info;
	ShaderPixelInputInfo  ps_input_info;
	PipelineCache::GraphicsPrograms programs;
	// Set before PrepareDrawRenderState: an indirect draw, whose mesh program (if any) reads its
	// parameters on the GPU (ShaderMeshInputInfo::draw_data_indirect).
	bool                  mesh_draw_indirect = false;
};

struct DrawCallInfo {
	CommandBufferDebugOp debug_op       = CommandBufferDebugOp::DrawIndex;
	uint32_t             index_count    = 0;
	uint32_t             instance_count = 0;
	uint32_t             first_instance = 0;

	[[nodiscard]] bool IsIndexed() const { return debug_op == CommandBufferDebugOp::DrawIndex; }
	[[nodiscard]] const char* Name() const { return IsIndexed() ? "DrawIndex" : "DrawIndexAuto"; }
};

RenderState RenderExecutor::AcquireRenderTargets(CommandBuffer& buffer, RenderColorInfo* colors,
                                                 uint32_t color_count, RenderDepthInfo& depth,
                                                 vk::ImageAspectFlags& feedback_aspects,
                                                 std::span<PreparedBindings* const> stages) {
	EXIT_IF(colors == nullptr || color_count > RENDER_COLOR_ATTACHMENTS_MAX);
	feedback_aspects = {};
	auto&       cache = m_context.GetTextureCache();
	RenderState state {};
	state.width                 = std::numeric_limits<uint32_t>::max();
	state.height                = std::numeric_limits<uint32_t>::max();
	state.num_layers            = std::numeric_limits<uint32_t>::max();
	state.num_color_attachments = 0;
	for (uint32_t i = 0; i < color_count; i++) {
		auto& target = colors[i];
		EXIT_IF(!target.image_id);
		const auto owner = cache.m_slot_images.try_get(target.image_id);
		if (owner == nullptr || (!owner->registered && !owner->info.data.Empty()) ||
		    owner->binding.needs_rebind) {
			EXIT("color target changed after render-state discovery\n");
		}
		const auto image_view = cache.FindRenderTarget(target.image_id, target.desc);
		auto&      image      = cache.GetImage(target.image_id);
		EXIT_IF(image.backing.samples != target.desc.info.samples || image_view == nullptr);
		const auto& view   = target.desc.view_info;
		const auto  layout = image.binding.is_bound ? vk::ImageLayout::eGeneral
		                                            : vk::ImageLayout::eColorAttachmentOptimal;
		image.binding.attachment_layout = layout;
		image.binding.attachment_access =
		    vk::AccessFlagBits2::eColorAttachmentRead | vk::AccessFlagBits2::eColorAttachmentWrite;
		image.Transit(layout, image.binding.attachment_access,
		              ImageSubresourceRange {view.base_level, view.level_count, view.base_layer,
		                                     view.layer_count},
		              buffer.Recorder());
		const auto extent       = target.Extent();
		state.width             = std::min(state.width, extent.width);
		state.height            = std::min(state.height, extent.height);
		state.num_layers        = std::min(state.num_layers, view.layer_count);
		state.num_color_attachments = std::max(state.num_color_attachments, target.target_slot + 1);
		auto& attachment            = state.color_attachments[target.target_slot];
		attachment.image_view   = image_view;
		attachment.image_layout = layout;
	}
	if (depth.image_id) {
		const auto owner = cache.m_slot_images.try_get(depth.image_id);
		if (owner == nullptr || !owner->registered || owner->binding.needs_rebind) {
			EXIT("depth target changed after render-state discovery\n");
		}
		const auto  image_view = cache.FindDepthTarget(depth.image_id, depth.desc);
		const auto& metadata   = depth.desc.info.metadata;
		if (metadata.kind == ImageMetadataKind::Htile && depth.depth_clear_enable &&
		    !cache.ClearMeta(metadata.range.address)) {
			EXIT("failed to acquire HTile metadata for a depth clear\n");
		}
		const bool meta_clear =
		    metadata.kind == ImageMetadataKind::Htile &&
		    cache.IsMetaCleared(metadata.range.address, depth.desc.view_info.base_layer);
		depth.depth_load_clear_enable = depth.depth_clear_enable || meta_clear;
		if (meta_clear &&
		    !cache.TouchMeta(metadata.range.address, depth.desc.view_info.base_layer, false)) {
			EXIT("failed to consume HTile clear state\n");
		}
		auto& image = cache.GetImage(depth.image_id);
		EXIT_IF(image_view == nullptr || image.backing.samples != depth.desc.info.samples);
		const auto draw_writes = depth.AttachmentWriteAspects();
		vk::ImageAspectFlags sampled_aspects;
		for (const auto* stage: stages) {
			for (const auto& binding: stage->images) {
				if (binding.image_id != depth.image_id ||
				    binding.desc.type != TextureCache::BindingType::Texture) continue;
				const auto native =
				    std::ranges::find(image.views, binding.image_view, &CachedImageView::view);
				EXIT_IF(native == image.views.end());
				sampled_aspects |= native->info.aspect;
				feedback_aspects |= DepthFeedbackAspects(draw_writes, depth.desc.view_info,
				                                         native->info);
			}
		}
		if (feedback_aspects && !m_context.GetGraphics().attachment_feedback_loop_enabled) {
			EXIT("depth attachment feedback loop is not supported by the host\n");
		}
		auto layout = depth_attachment_layout(depth);
		if (sampled_aspects & ~DepthReadableAspects(layout)) {
			layout = m_context.GetGraphics().attachment_feedback_loop_enabled
			             ? vk::ImageLayout::eAttachmentFeedbackLoopOptimalEXT
			             : vk::ImageLayout::eGeneral;
		}
		// The attachment store writes even when guest depth/stencil tests do not.
		const auto access = vk::AccessFlagBits2::eDepthStencilAttachmentRead |
		                    vk::AccessFlagBits2::eDepthStencilAttachmentWrite;
		image.binding.attachment_layout = layout;
		image.binding.attachment_access = access;
		const auto& view                = depth.desc.view_info;
		image.Transit(layout, access,
		              ImageSubresourceRange {view.base_level, view.level_count, view.base_layer,
		                                     view.layer_count},
		              buffer.Recorder());
		state.width               = std::min(state.width, depth.desc.info.extent.width);
		state.height              = std::min(state.height, depth.desc.info.extent.height);
		state.num_layers          = std::min(state.num_layers, view.layer_count);
		const auto aspects        = ImageViewOps::DepthAspectMask(depth.desc.view_info.format);
		auto&      attachment     = state.depth_stencil_attachment;
		attachment.image_view     = image_view;
		attachment.image_layout   = layout;
		attachment.clear_value[0] = std::bit_cast<uint32_t>(depth.depth_clear_value);
		attachment.clear_value[1] = depth.stencil_clear_value;
		attachment.has_depth      = static_cast<bool>(aspects & vk::ImageAspectFlagBits::eDepth);
		attachment.depth_clear    = depth.depth_load_clear_enable;
		attachment.has_stencil    = static_cast<bool>(aspects & vk::ImageAspectFlagBits::eStencil);
		attachment.stencil_clear  = depth.stencil_clear_enable;
	}
	if (color_count == 0 && !depth.image_id) {
		const auto& limits = buffer.GetGraphics().GetPhysicalDeviceProperties().limits;
		state.width        = limits.maxFramebufferWidth;
		state.height       = limits.maxFramebufferHeight;
	}
	if (state.num_layers == std::numeric_limits<uint32_t>::max()) {
		state.num_layers = 1;
	}
	EXIT_IF(state.width == 0 || state.height == 0 || state.num_layers == 0 ||
	        state.width == std::numeric_limits<uint32_t>::max() ||
	        state.height == std::numeric_limits<uint32_t>::max());
	return state;
}

static uint32_t DrawColorOutputMask(const HW::Context& ctx) {
	const auto& sh_regs     = ctx.GetShaderRegisters();
	const auto  write_mask  = ctx.GetRenderTargetMask() & sh_regs.m_cbShaderMask;
	uint32_t    output_mask = 0;
	for (uint32_t slot = 0; slot < RENDER_COLOR_ATTACHMENTS_MAX; slot++) {
		if (sh_regs.target_output_mode[slot] != 0 &&
		    render_target_mask_slot(write_mask, slot) != 0) {
			output_mask |= 1u << slot;
		}
	}
	return output_mask;
}

enum class CbColorMode : uint8_t {
	Disable            = 0,
	Normal             = 1,
	EliminateFastClear = 2,
	Resolve            = 3,
	FmaskDecompress    = 5,
	DccDecompress      = 6,
};

static bool ConsumeMetadataColorOperation(const CommandBuffer& buffer) {
	const auto& ctx  = buffer.GetRegisters();
	const auto  mode = ctx.GetColorControl().mode;
	// These special modes run color-buffer metadata or decompression operations. The shader is a
	// vehicle for that operation, and its exported color must not be applied as a normal draw.
	// Kyty stores expanded Vulkan images rather than compressed guest surfaces, so no equivalent
	// hardware pass is emitted. Tracked DCC clear state is materialized on attachment bind;
	// future CMask/FMask support can consume its state through the same TextureCache path.
	return mode == static_cast<uint8_t>(CbColorMode::EliminateFastClear) ||
	       mode == static_cast<uint8_t>(CbColorMode::FmaskDecompress) ||
	       mode == static_cast<uint8_t>(CbColorMode::DccDecompress);
}

struct DrawEmitInfo {
	int32_t  vertex_offset = 0;
	uint32_t first_vertex  = 0;
	uint32_t first_instance = 0;
	// Research: guest address of indirect arguments the GPU reads itself (0: CPU counts).
	uint64_t indirect_args = 0;
	// An indirect mesh draw converted on the GPU (MeshIndirectDraw): workgroup counts for
	// drawMeshTasksIndirectEXT and the device address of the six draw parameters.
	vk::Buffer        mesh_groups_buffer = nullptr;
	vk::DeviceSize    mesh_groups_offset = 0;
	vk::DeviceAddress mesh_draw_data     = 0;
	uint32_t          mesh_draw_slices   = 1; // KYTY_MESH_FAST_SLICES: one indirect draw per entry
};

struct DrawIndexBufferSource {
	uint64_t      address   = 0;
	const void*   host_data = nullptr;
	uint64_t      size      = 0;
	vk::IndexType type      = vk::IndexType::eUint16;
	uint32_t      guest_element_size = 0;
};

struct PreparedIndexBuffer {
	vk::Buffer     buffer = nullptr;
	vk::DeviceSize offset = 0;
	vk::IndexType  type   = vk::IndexType::eUint16;
};

static uint64_t VertexBufferDescriptorSize(int binding, const ShaderVertexInputInfo& info) {
	const auto& buffer = info.buffers[binding];
	if (buffer.stride != 0 || buffer.num_records == 0) {
		return static_cast<uint64_t>(buffer.stride) * buffer.num_records;
	}

	uint64_t size = 0;
	for (int i = 0; i < info.resources_num; i++) {
		if (info.resources_dst[i].buffer_index != binding) {
			continue;
		}
		const auto& resource = info.resources[i];
		// RDNA2 OOB_SELECT=2 only checks NumRecords != 0. A constant attribute still
		// fetches its entire format; NumRecords is not a byte count in this mode.
		const uint64_t extent = resource.OutOfBounds() == 2
		                            ? resource.Base48() - buffer.addr +
		                                  ShaderRecompiler::Format::GetFormatInfo(resource.Format()).byte_size
		                            : buffer.num_records;
		size = std::max(size, extent);
	}
	return size;
}

struct VertexBufferRange {
	uint64_t                     base_address  = 0;
	uint64_t                     requested_end = 0;
	uint64_t                     acquired_end  = 0;
	std::pair<Buffer*, uint64_t> binding;

	[[nodiscard]] uint64_t RequestedSize() const { return requested_end - base_address; }
};

struct PreparedVertexBuffers {
	static constexpr uint32_t MaxBuffers = ShaderVertexInputInfo::RES_MAX;

	std::array<vk::Buffer, MaxBuffers>     buffers {};
	std::array<vk::DeviceSize, MaxBuffers> offsets {};
	std::array<vk::DeviceSize, MaxBuffers> sizes {};
	uint32_t                               count = 0;
};

static PreparedVertexBuffers AcquireVertexBuffers(CommandBuffer&               buffer,
                                                  const ShaderVertexInputInfo& vs_input_info) {
	EXIT_IF(vs_input_info.buffers_num < 0 ||
	        vs_input_info.buffers_num > ShaderVertexInputInfo::RES_MAX);

	// Collect the non-empty guest vertex ranges.
	std::array<uint64_t, ShaderVertexInputInfo::RES_MAX>          sizes {};
	std::array<VertexBufferRange, ShaderVertexInputInfo::RES_MAX> ranges {};
	uint32_t                                                      range_count = 0;
	for (int i = 0; i < vs_input_info.buffers_num; i++) {
		const auto& vertex = vs_input_info.buffers[i];
		const auto  size   = VertexBufferDescriptorSize(i, vs_input_info);
		sizes[i]           = size;
		if (size == 0) {
			continue;
		}
		if (vertex.addr == 0 || size > UINT64_MAX - vertex.addr) {
			EXIT("invalid vertex buffer range: addr=0x%016" PRIx64 " size=0x%016" PRIx64 "\n",
			     vertex.addr, size);
		}
		ranges[range_count++] = {vertex.addr, vertex.addr + size};
	}

	std::sort(ranges.begin(), ranges.begin() + range_count,
	          [](const VertexBufferRange& left, const VertexBufferRange& right) {
		          return left.base_address < right.base_address;
	          });

	// Merge overlapping or touching ranges before acquiring host buffers.
	std::array<VertexBufferRange, ShaderVertexInputInfo::RES_MAX> merged_ranges {};
	uint32_t                                                      merged_count = 0;
	for (uint32_t i = 0; i < range_count; i++) {
		const auto& range = ranges[i];
		if (merged_count != 0 &&
		    merged_ranges[merged_count - 1].requested_end >= range.base_address) {
			merged_ranges[merged_count - 1].requested_end =
			    std::max(merged_ranges[merged_count - 1].requested_end, range.requested_end);
			continue;
		}
		merged_ranges[merged_count++] = {range.base_address, range.requested_end};
	}

	auto& cache = buffer.GetContext().GetBufferCache();
	for (uint32_t i = 0; i < merged_count; i++) {
		auto& range = merged_ranges[i];
		// PPSA20298
		const auto size =
		    Libs::LibKernel::Memory::ClampRangeSize(range.base_address, range.RequestedSize());
		range.acquired_end = range.base_address + size;
		range.binding      = cache.ObtainBuffer(range.base_address, size, false);
	}

	// Rebuild slot bindings, offsetting non-empty slots into their acquired merged range.
	PreparedVertexBuffers prepared;
	prepared.count         = static_cast<uint32_t>(vs_input_info.buffers_num);
	vk::Buffer null_buffer = nullptr;
	for (int i = 0; i < vs_input_info.buffers_num; i++) {
		const auto& vertex = vs_input_info.buffers[i];
		const auto  size   = sizes[i];
		if (size == 0) {
			if (null_buffer == nullptr) {
				null_buffer = cache.GetBuffer(NULL_BUFFER_ID).Handle();
			}
			prepared.buffers[i] = null_buffer;
			prepared.offsets[i] = 0;
			continue;
		}

		const auto range = std::find_if(merged_ranges.begin(), merged_ranges.begin() + merged_count,
		                                [&](const VertexBufferRange& value) {
			                                return vertex.addr >= value.base_address &&
			                                       vertex.addr < value.acquired_end;
		                                });
		if (range == merged_ranges.begin() + merged_count) {
			EXIT("vertex buffer address is outside the acquired range: addr=0x%016" PRIx64 "\n",
			     vertex.addr);
		}

		prepared.buffers[i] = range->binding.first->Handle();
		prepared.offsets[i] = range->binding.second + vertex.addr - range->base_address;
		prepared.sizes[i]   = std::min(size, range->acquired_end - vertex.addr);
	}

	return prepared;
}

// arg0 = phase (0x100 indexed / 0x200 auto ... 0x500 pipeline bound, 0x700 done), arg1 = index
// count, arg3 = instance count, arg4 = the pixel shader's hash (0 without one), which device-loss
// triage needs to tell which draw read a freed resource.
static void SetDrawDebugPhase(CommandBuffer& buffer, uint64_t submit_id, const DrawCallInfo& draw,
                              uint32_t phase, uint64_t ps_hash) {
	buffer.SetDebugInfo(static_cast<uint32_t>(draw.debug_op), submit_id, phase, draw.index_count,
	                    draw.first_instance, draw.instance_count, ps_hash);
}

static bool GetDrawTopology(const HW::UserConfig& ucfg, vk::PrimitiveTopology& topology) {

	topology = vk::PrimitiveTopology::ePointList;

	switch (ucfg.GetPrimType()) {
		case Prospero::PrimitiveType::kNone: return false;
		case Prospero::PrimitiveType::kPointList:
			topology = vk::PrimitiveTopology::ePointList;
			break;
		case Prospero::PrimitiveType::kLineList: topology = vk::PrimitiveTopology::eLineList; break;
		case Prospero::PrimitiveType::kLineStrip:
			topology = vk::PrimitiveTopology::eLineStrip;
			break;
		case Prospero::PrimitiveType::kTriList:
			topology = vk::PrimitiveTopology::eTriangleList;
			break;
		case Prospero::PrimitiveType::kTriFan:
			topology = vk::PrimitiveTopology::eTriangleFan;
			break;
		case Prospero::PrimitiveType::kTriStrip:
			topology = vk::PrimitiveTopology::eTriangleStrip;
			break;
		case Prospero::PrimitiveType::kPatch:
			if (!Config::TessellationEnabled()) return false;
			[[fallthrough]];
		case Prospero::PrimitiveType::kRectList:
		case Prospero::PrimitiveType::kRectListLegacy:
			topology = vk::PrimitiveTopology::ePatchList;
			break;
		case Prospero::PrimitiveType::kQuadListLegacy:
			topology = vk::PrimitiveTopology::eTriangleFan;
			break;
		default: {
			static std::atomic_bool logged = false;
			if (!logged.exchange(true, std::memory_order_relaxed)) {
				std::printf("Skipping draw with unknown primitive type: %u\n",
				            static_cast<uint32_t>(ucfg.GetPrimType()));
			}
			return false;
		}
	}

	return true;
}

static bool ResolvePrimitiveRestart(const CommandBuffer& buffer,
                                    const DrawIndexBufferSource& source) {
	const auto control = buffer.GetUserConfig().GetPrimitiveResetControl();
	EXIT_NOT_IMPLEMENTED((control & ~0x3u) != 0);
	if ((control & 0x1u) == 0) {
		return false;
	}
	switch (buffer.GetUserConfig().GetPrimType()) {
		case Prospero::PrimitiveType::kLineStrip:
		case Prospero::PrimitiveType::kTriFan:
		case Prospero::PrimitiveType::kTriStrip: break;
		default: return false;
	}

	const auto element_size = source.guest_element_size;
	const auto index_mask   = UINT32_MAX >> ((4 - element_size) * 8);
	const auto reset_index  = buffer.GetRegisters().GetPrimitiveResetIndex();
	if ((control & 0x2u) != 0 && (reset_index & ~index_mask) != 0) {
		return false;
	}
	const auto restart_index = reset_index & index_mask;
	if (restart_index == index_mask) {
		// Use native restart; the 8-bit path widens its marker to 0xffff.
		return true;
	}

	// A game can set a custom reset value without using it in the index buffer.
	// Keep restart off in that case; fail if we actually find the value.
	// Scan before preparing draw resources: readback can restart the command buffer.
	EXIT_NOT_IMPLEMENTED(source.address == 0);
	const auto* indices = reinterpret_cast<const uint8_t*>(source.address);
	for (uint64_t offset = 0; offset < source.size; offset += element_size) {
		uint32_t index = 0;
		std::memcpy(&index, indices + offset, element_size);
		EXIT_NOT_IMPLEMENTED(index == restart_index);
	}
	return false;
}

// KYTY_DRAW_PREP scanner: the draw's vertex-shader check and pixel state, as
// PrepareDrawRenderState and RefreshShaders compute them.
bool DrawPrepHasVertexShader(const HW::Shader& sh_ctx) {
	return DrawHasValidVertexShader(sh_ctx);
}

bool DrawPrepPixelState(const HW::Context& ctx, const HW::Shader& sh_ctx,
                        std::array<Prospero::ColorComponentMapping, 8>& target_export_mapping) {
	static_assert(RENDER_COLOR_ATTACHMENTS_MAX == 8);
	const auto& shader_regs       = ctx.GetShaderRegisters();
	const auto  color_output_mask = DrawColorOutputMask(ctx);
	const bool  ps_active         = sh_ctx.GetPs().ps_regs.data_addr != 0 &&
	                       (color_output_mask != 0 ||
	                        PixelShaderHasDepthOrCoverageSideEffects(shader_regs));
	target_export_mapping = {};
	for (uint32_t slot = 0; slot < RENDER_COLOR_ATTACHMENTS_MAX; slot++) {
		const auto& rt = ctx.GetRenderTarget(slot);
		if ((color_output_mask & (1u << slot)) != 0 && rt.base.addr != 0) {
			target_export_mapping[slot] =
			    TextureGetRenderTargetFormat(rt.info.format, rt.info.channel_type,
			                                 rt.info.channel_order)
			        .export_mapping;
		}
	}
	return ps_active;
}

static void RefreshShaders(CommandBuffer& buffer, const DrawCallInfo& draw,
                           uint32_t color_output_mask, DrawRenderState& state) {
	auto& ctx    = buffer.GetRegisters();
	auto& sh_ctx = buffer.GetShaders();

	const auto& vertex_shader_info = sh_ctx.GetVs();
	const auto& pixel_shader_info  = sh_ctx.GetPs();
	const auto& shader_regs        = ctx.GetShaderRegisters();

	state.programs      = {};
	state.ps_input_info = {};
	std::array<Prospero::ColorComponentMapping, RENDER_COLOR_ATTACHMENTS_MAX>
	    target_export_mapping {};
	for (uint32_t slot = 0; slot < RENDER_COLOR_ATTACHMENTS_MAX; slot++) {
		const auto& rt = ctx.GetRenderTarget(slot);
		if ((color_output_mask & (1u << slot)) != 0 && rt.base.addr != 0) {
			target_export_mapping[slot] =
			    TextureGetRenderTargetFormat(rt.info.format, rt.info.channel_type,
			                                 rt.info.channel_order)
			        .export_mapping;
		}
	}
	auto& pipeline_cache = buffer.GetContext().GetPipelineCache();
	if (draw.IsIndexed()) {
		LogDrawPhase(draw.Name(), "GetGraphicsPrograms");
	}
	state.programs = pipeline_cache.GetGraphicsPrograms(
	    vertex_shader_info, pixel_shader_info, shader_regs, ctx, buffer.GetUserConfig(),
	    target_export_mapping, state.ps_active, state.vertex_info, state.ps_input_info,
	    state.mesh_draw_indirect);
}

bool RenderExecutor::PrepareDrawRenderState(CommandBuffer& buffer, const DrawCallInfo& draw,
                                            uint32_t            render_target_slice_offset,
	                                        DrawRenderState& state) {
	KYTY_PROFILER_FUNCTION();
	const auto& shader_regs       = buffer.GetRegisters().GetShaderRegisters();
	const auto  color_output_mask = DrawColorOutputMask(buffer.GetRegisters());
	state.ps_active = buffer.GetShaders().GetPs().ps_regs.data_addr != 0 &&
	                  (color_output_mask != 0 ||
	                   PixelShaderHasDepthOrCoverageSideEffects(shader_regs));
	const bool phases      = DrawRecordCensus::PhasesOn();
	uint64_t   phase_start = phases ? DrawRecordCensus::NowNs() : 0;
	RefreshShaders(buffer, draw, color_output_mask, state);
	if (phases) DrawRecordCensus::Phase(0, phase_start);
	if (!state.programs.vertex[0] || (state.ps_active && !state.programs.pixel)) {
		return false;
	}
	uint32_t mrt_mask = 0;
	if (state.ps_active) {
		for (const auto& output: state.ps_input_info.stage.program->info.outputs) {
			if (output.kind == ShaderRecompiler::IR::StageOutputKind::Mrt) {
				mrt_mask |= 1u << output.index;
			}
		}
	}
	mrt_mask &= color_output_mask;
	if (draw.IsIndexed()) {
		LogDrawPhase(draw.Name(), "ResolveRenderColorTarget");
	}
	for (uint32_t slot = 0; slot < RENDER_COLOR_ATTACHMENTS_MAX; slot++) {
		if ((mrt_mask & (1u << slot)) != 0) {
			ResolveRenderColorTarget(buffer, state.color_info[state.color_count],
			                         render_target_slice_offset, slot);
			if (state.color_info[state.color_count].image_id) {
				state.color_count++;
			}
		}
	}
	if (draw.IsIndexed()) {
		LogDrawPhase(draw.Name(), "ResolveRenderDepthTarget");
	}
	ResolveRenderDepthTarget(buffer, state.depth_info);
	if (phases) DrawRecordCensus::Phase(1, phase_start);

	if (state.color_count == 0 && !state.depth_info.image_id && !state.ps_active) {
		LogFramebufferSkip(draw.Name(), state.color_info[0], state.depth_info, buffer,
		                   draw.index_count, 0);
		return false;
	}

	return true;
}

static PreparedIndexBuffer PrepareIndexBuffer(CommandBuffer&               buffer,
                                              const DrawIndexBufferSource& source) {
	PreparedIndexBuffer prepared;
	if (source.size == 0) {
		return prepared;
	}
	prepared.type = source.type;
	if (source.host_data != nullptr) {
		auto& stream = buffer.GetContext().GetBufferCache().GetUtilityBuffer(MemoryUsage::Stream);
		prepared.offset = stream.Copy(source.host_data, source.size, 16);
		prepared.buffer = stream.Handle();
	} else {
		auto [buffer_ptr, offset] =
		    buffer.GetContext().GetBufferCache().ObtainBuffer(source.address, source.size, false);
		prepared.buffer = buffer_ptr->Handle();
		prepared.offset = offset;
	}
	return prepared;
}

static void CommitVertexBuffers(const CommandRecorder&       vk_buffer,
                                const PreparedVertexBuffers& prepared) {
	for (uint32_t i = 0; i < prepared.count; i++) {
		EXIT_IF(prepared.buffers[i] == nullptr);
	}
	if (prepared.count != 0) {
		// Guest descriptor bounds must survive allocation merging in the cache.
		vk_buffer.bindVertexBuffers2(0, prepared.count, prepared.buffers.data(),
		                             prepared.offsets.data(), prepared.sizes.data(), nullptr);
	}
}

static void CommitIndexBuffer(const CommandRecorder&     vk_buffer,
                              const PreparedIndexBuffer& prepared) {
	if (prepared.buffer == nullptr) {
		return;
	}
	vk_buffer.bindIndexBuffer(prepared.buffer, prepared.offset, prepared.type);
}

static void LogDrawStateIfNeeded(const CommandBuffer& buffer, const DrawCallInfo& draw,
	                             const DrawRenderState& state, uint32_t index_type_and_size,
                                 const void* index_addr) {
	if (!graphics_debug_dump_enabled()) {
		return;
	}

	if (!draw.IsIndexed() && !Prospero::IsRectList(buffer.GetUserConfig().GetPrimType())) {
		return;
	}

	if (state.ps_active) {
		LogDrawTargetState(draw.Name(), state.color_info[0], state.depth_info, buffer,
		                   state.ps_input_info, draw.index_count, 0);
	}
	LogDrawInputState(buffer, state.color_info[0], state.vertex_info[0], index_type_and_size,
	                  draw.index_count, index_addr);
}

static void EmitDrawPrimitives(const HW::UserConfig& ucfg, const CommandRecorder& vk_buffer,
                               const DrawCallInfo& draw, const DrawEmitInfo& emit,
                               vk::Buffer     indirect_buffer = nullptr,
                               vk::DeviceSize indirect_offset = 0) {
	if (indirect_buffer) {
		// The guest records share Vulkan's layouts: {count, instances, first, (vertex offset,)
		// first instance}.
		if (draw.IsIndexed()) {
			vk_buffer.drawIndexedIndirect(indirect_buffer, indirect_offset, 1, 20);
		} else {
			vk_buffer.drawIndirect(indirect_buffer, indirect_offset, 1, 16);
		}
		return;
	}
	switch (ucfg.GetPrimType()) {
		case Prospero::PrimitiveType::kPointList:
		case Prospero::PrimitiveType::kLineList:
		case Prospero::PrimitiveType::kLineStrip:
		case Prospero::PrimitiveType::kTriList:
		case Prospero::PrimitiveType::kTriFan:
		case Prospero::PrimitiveType::kTriStrip:
		case Prospero::PrimitiveType::kRectList:
		case Prospero::PrimitiveType::kRectListLegacy:
		case Prospero::PrimitiveType::kPatch:
			if (draw.IsIndexed()) {
				vk_buffer.drawIndexed(draw.index_count, draw.instance_count, 0, emit.vertex_offset,
				                      emit.first_instance);
			} else {
				vk_buffer.draw(draw.index_count, draw.instance_count, emit.first_vertex,
				               emit.first_instance);
			}
			break;
		case Prospero::PrimitiveType::kQuadListLegacy:
			EXIT_NOT_IMPLEMENTED((draw.index_count & 0x3u) != 0);
			for (uint32_t i = 0; i < draw.index_count; i += 4) {
				if (draw.IsIndexed()) {
					vk_buffer.drawIndexed(4, draw.instance_count, i, emit.vertex_offset,
					                      emit.first_instance);
				} else {
					vk_buffer.draw(4, draw.instance_count, i + emit.first_vertex,
					               emit.first_instance);
				}
			}
			break;
		default: EXIT("unknown primitive type: %u\n", static_cast<uint32_t>(ucfg.GetPrimType()));
	}
}


// KYTY_LOCAL_HACK research, KYTY_DRAW_RECORDS=2 (live, default 0): verify-only draw records. A
// draw's key is its compiled programs, their user data, its render targets and pipeline; on the
// key's second sighting the prepared result (per part) is stored, and every later sighting
// compares what the normal path prepared with it. No draw changes: this measures how often a
// replayed record would be right, and whether its differences are data (shader data, flattened
// SRT: patchable) or structure (buffers, images, samplers, bindless: the record would be stale).
namespace {
struct DrawRecordVerify {
	// buffers (vk handles + ranges), images, samplers, shader data, flattened SRT, bindless,
	// buffer offsets + guest addresses.
	static constexpr int Parts = 7;
	struct Record {
		std::array<uint64_t, Parts> sig {};
		uint32_t                    sightings = 0;
	};
	struct Stats {
		uint64_t first = 0, stored = 0, same = 0, data_only = 0, structural = 0;
		std::array<uint64_t, Parts> differ {};
	};
	// [0] K1: programs, masked user data (record_key), pipeline, targets.
	// [1] K3: programs, user data minus pointer-looking pairs (record_key_reloc), pipeline, targets.
	std::array<std::unordered_map<uint64_t, Record>, 2> records;
	std::array<Stats, 2>                                stats;
	std::unordered_map<uint64_t, uint32_t>              occurrences; // K2 base -> uses this frame
	uint64_t                                            frame = 0;
	uint64_t                                            draws = 0;
	std::chrono::steady_clock::time_point last = std::chrono::steady_clock::now();
};
DrawRecordVerify g_draw_record_verify;

struct Fnv {
	uint64_t h = 0xcbf29ce484222325ull;
	void     Add(const void* data, size_t bytes) {
        const auto* p = static_cast<const uint8_t*>(data);
        for (size_t i = 0; i < bytes; i++) {
            h = (h ^ p[i]) * 0x100000001b3ull;
        }
	}
	template <typename T>
	void Value(const T& value) {
		static_assert(std::is_trivially_copyable_v<T>);
		Add(&value, sizeof(value));
	}
	template <typename T>
	void Range(const std::vector<T>& values) {
		Value(values.size());
		if (!values.empty()) {
			Add(values.data(), values.size() * sizeof(T));
		}
	}
};

int DrawRecordsMode() {
	static auto& mode = Common::LiveSwitches::Get("KYTY_DRAW_RECORDS", 0);
	return static_cast<int>(mode.load(std::memory_order_relaxed));
}

void VerifyDrawRecord(std::span<PreparedBindings* const> stages, const void* pipeline,
                      const RenderColorInfo* colors, uint32_t color_count,
                      const RenderDepthInfo& depth, uint64_t frame) {
	auto& v = g_draw_record_verify;
	v.draws++;
	if (frame != v.frame) {
		v.frame = frame;
		v.occurrences.clear();
	}
	Fnv base;
	base.Value(pipeline);
	for (uint32_t i = 0; i < color_count; i++) {
		base.Value(colors[i].image_id);
	}
	base.Value(depth.image_id);
	std::array<Fnv, DrawRecordVerify::Parts> sig;
	Fnv k1 = base;
	Fnv k3 = base;
	for (const auto* stage: stages) {
		const auto& runtime = *stage->runtime;
		base.Value(runtime.program);
		k1.Value(runtime.program);
		k1.Value(runtime.resources->record_key);
		k3.Value(runtime.program);
		k3.Value(runtime.resources->record_key_reloc);
		for (const auto& b: stage->buffers) {
			sig[0].Value(b.buffer);
			sig[0].Value(b.range);
			sig[6].Value(b.offset);
		}
		for (const auto& src: stage->buffer_sources) {
			sig[6].Value(src.address);
			sig[0].Value(src.size);
		}
		for (const auto& image: stage->images) {
			sig[1].Value(image.image_id);
			sig[1].Value(image.image_view);
			sig[1].Value(image.layout);
			sig[1].Range(image.mip_views);
		}
		sig[2].Range(stage->samplers);
		sig[3].Range(stage->shader_data);
		sig[4].Range(runtime.resources->flattened_srt);
		sig[5].Range(stage->bindless_patches);
		sig[5].Range(stage->bindless_heaps);
	}
	const std::array<uint64_t, 2> keys {k1.h, k3.h};
	for (int k = 0; k < 2; k++) {
		auto& st     = v.stats[k];
		auto& record = v.records[k][keys[k]];
		record.sightings++;
		if (record.sightings == 1) {
			st.first++;
		} else if (record.sightings == 2) {
			st.stored++;
		} else {
			bool data = false, structure = false;
			for (int part = 0; part < DrawRecordVerify::Parts; part++) {
				if (record.sig[part] != sig[part].h) {
					st.differ[part]++;
					(part == 3 || part == 4 || part == 6 ? data : structure) = true;
				}
			}
			(structure ? st.structural : data ? st.data_only : st.same)++;
		}
		for (int part = 0; part < DrawRecordVerify::Parts; part++) {
			record.sig[part] = sig[part].h; // a re-store keeps the latest result
		}
		if (v.records[k].size() > 400000) {
			v.records[k].clear();
		}
	}
	if (const auto now = std::chrono::steady_clock::now(); now - v.last >= std::chrono::seconds(5)) {
		for (int k = 0; k < 2; k++) {
			const auto& st = v.stats[k];
			std::printf("Draw records verify %s (5 s): %" PRIu64 " draws, %" PRIu64 " first, %" PRIu64
			            " stored, replays: %" PRIu64 " identical, %" PRIu64 " data-only, %" PRIu64
			            " structural; differ: buffers %" PRIu64 " images %" PRIu64 " samplers %" PRIu64
			            " shader-data %" PRIu64 " flattened %" PRIu64 " bindless %" PRIu64
			            " buffer-offsets %" PRIu64 "; %zu keys\n",
			            k == 0 ? "K1 masked-ud" : "K5 structural", v.draws, st.first, st.stored, st.same,
			            st.data_only, st.structural, st.differ[0], st.differ[1], st.differ[2],
			            st.differ[3], st.differ[4], st.differ[5], st.differ[6], v.records[k].size());
		}
		v.stats = {};
		v.draws = 0;
		v.last  = now;
	}
}
} // namespace

void RenderExecutor::ExecutePreparedDraw(uint64_t submit_id, CommandBuffer& buffer,
                                         const DrawCallInfo& draw, DrawRenderState& state,
                                         vk::PrimitiveTopology topology, const DrawEmitInfo& emit,
                                         const DrawIndexBufferSource& index_source,
                                         bool primitive_restart_enable) {
	KYTY_PROFILER_FUNCTION();
	const bool mesh = state.vertex_info[0].stage.program->stage == ShaderType::Mesh;
	const bool quad =
	    buffer.GetUserConfig().GetPrimType() == Prospero::PrimitiveType::kQuadListLegacy;
	if (emit.indirect_args == 0 || (!mesh && !quad)) {
		ExecutePreparedDrawResolved(submit_id, buffer, draw, state, topology, emit, index_source,
		                            primitive_restart_enable);
		return;
	}
	const auto& mesh_info = state.vertex_info[0].mesh;
	{
		// KYTY_LOCAL_HACK (research): KYTY_DRAW_STATS=1 (live) counts the indirect draws whose
		// arguments are read here, by kind, every 5 s.
		static auto& draw_stats = Common::LiveSwitches::Get("KYTY_DRAW_STATS", 0);
		if (draw_stats.load(std::memory_order_relaxed) != 0) {
			static std::array<uint64_t, 5> counts {};
			static auto                    report = std::chrono::steady_clock::now();
			counts[!mesh ? 0 : quad ? 1 : mesh_info.draw_data_indirect ? 3 : mesh_info.fast_launch ? 2 : 4]++;
			if (const auto now = std::chrono::steady_clock::now(); now - report > std::chrono::seconds(5)) {
				std::printf("Indirect draws (5 s): quad %llu, mesh quad %llu, mesh fast-launch CPU %llu, "
				            "mesh GPU %llu, mesh CPU %llu (indexed last %d)\n",
				            (unsigned long long)counts[0], (unsigned long long)counts[1],
				            (unsigned long long)counts[2], (unsigned long long)counts[3],
				            (unsigned long long)counts[4], draw.IsIndexed() ? 1 : 0);
				counts = {};
				report = now;
			}
		}
	}
	if (mesh && !quad && mesh_info.draw_data_indirect) {
		// The arguments stay on the GPU: a pass converts them into the workgroup counts and
		// the mesh program's draw parameters (MeshIndirectDraw). It is a dispatch, so it runs
		// outside the render pass, before the draw's bindings.
		if (m_mesh_indirect == nullptr) {
			m_mesh_indirect = std::make_unique<MeshIndirectDraw>(
			    m_context.GetGraphics(), m_context.GetCommandScheduler());
		}
		{
			// KYTY_LOCAL_HACK (research): KYTY_MESH_PRED_STATS=1 (live) reports every 5 s how many
			// GPU-converted mesh draws end an active render pass and how many repeat the
			// arguments and parameters of a draw of the previous frame(s).
			static auto& pred_stats = Common::LiveSwitches::Get("KYTY_MESH_PRED_STATS", 0);
			if (pred_stats.load(std::memory_order_relaxed) != 0) {
				static std::unordered_set<uint64_t> cur, prev, prev2, cur_addr, prev_addr;
				static uint64_t frame = 0, frames = 0, total = 0, active = 0, hit1 = 0, hit2 = 0,
				                addr_hit = 0, distinct = 0;
				static auto     report = std::chrono::steady_clock::now();
				const uint64_t  now_frame =
				    m_context.GetGraphics().presented_frames.load(std::memory_order_relaxed);
				if (now_frame != frame) {
					distinct += cur.size();
					prev2 = std::move(prev);
					prev  = std::move(cur);
					cur.clear();
					prev_addr = std::move(cur_addr);
					cur_addr.clear();
					frame = now_frame;
					frames++;
				}
				uint64_t key = emit.indirect_args;
				for (const uint64_t v: {index_source.address, uint64_t {draw.IsIndexed()},
				                        uint64_t {draw.index_count},
				                        uint64_t {index_source.guest_element_size},
				                        uint64_t {mesh_info.InputPrimitiveSize()},
				                        uint64_t {mesh_info.InputPrimitiveStep()},
				                        uint64_t {mesh_info.primitives_per_group},
				                        uint64_t {mesh_info.fast_launch}}) {
					key = (key ^ v) * 0x100000001b3ull + (key >> 29u);
				}
				total++;
				active += buffer.IsRendering() ? 1u : 0u;
				const bool in1 = prev.contains(key);
				hit1 += in1 ? 1u : 0u;
				hit2 += (in1 || prev2.contains(key)) ? 1u : 0u;
				addr_hit += prev_addr.contains(emit.indirect_args) ? 1u : 0u;
				cur.insert(key);
				cur_addr.insert(emit.indirect_args);
				if (const auto now = std::chrono::steady_clock::now();
				    now - report > std::chrono::seconds(5)) {
					std::printf("Mesh pred (5 s): %llu conversions, %llu frames, %llu in an active render "
					            "pass, same tuple prev frame %llu, prev 2 frames %llu, same args addr %llu, "
					            "distinct/frame %.1f\n",
					            (unsigned long long)total, (unsigned long long)frames,
					            (unsigned long long)active, (unsigned long long)hit1,
					            (unsigned long long)hit2, (unsigned long long)addr_hit,
					            frames != 0 ? double(distinct) / double(frames) : 0.0);
					auto& pre = m_mesh_pre;
					std::printf("Mesh preconvert (5 s): segments %llu, batched %llu, hits %llu, misses %llu, "
					            "unused %llu, write invalidations %llu, learned keys %zu\n",
					            (unsigned long long)pre.starts, (unsigned long long)pre.batched,
					            (unsigned long long)pre.hits, (unsigned long long)pre.misses,
					            (unsigned long long)pre.unused, (unsigned long long)pre.writes,
					            pre.learned.size());
					pre.starts = pre.batched = pre.hits = pre.misses = pre.unused = pre.writes = 0;
					total = active = hit1 = hit2 = addr_hit = distinct = frames = 0;
					report = now;
				}
			}
		}
		const auto args_size = draw.IsIndexed() ? sizeof(vk::DrawIndexedIndirectCommand)
		                                        : sizeof(vk::DrawIndirectCommand);
		MeshIndirectDraw::Params params {.args                 = 0,
		                                 .index_address        = index_source.address,
		                                 .indexed              = draw.IsIndexed(),
		                                 .max_index_count      = draw.index_count,
		                                 .element_size         = index_source.guest_element_size,
		                                 .primitive_size       = mesh_info.InputPrimitiveSize(),
		                                 .primitive_step       = mesh_info.InputPrimitiveStep(),
		                                 .primitives_per_group = mesh_info.primitives_per_group,
		                                 .fast_launch          = mesh_info.fast_launch};
		auto& cache = m_context.GetBufferCache();
		// The arguments' device address, or 0 when they are not in a device-address buffer.
		const auto args_address = [&](uint64_t args, uint64_t size) -> vk::DeviceAddress {
			const auto [args_buffer, args_offset] =
			    cache.ObtainBuffer(args, size, false, false, {}, true);
			return args_buffer != nullptr && args_buffer->HasDeviceAddress()
			           ? args_buffer->BufferDeviceAddress() + args_offset
			           : 0;
		};
		const auto run = [&](const MeshIndirectDraw::Result& converted) {
			auto gpu_emit               = emit;
			gpu_emit.indirect_args      = 0; // consumed by the conversion
			gpu_emit.mesh_groups_buffer = converted.groups_buffer;
			gpu_emit.mesh_groups_offset = converted.groups_offset;
			gpu_emit.mesh_draw_data     = converted.draw_data;
			gpu_emit.mesh_draw_slices   = converted.slices;
			ExecutePreparedDrawResolved(submit_id, buffer, draw, state, topology, gpu_emit,
			                            index_source, primitive_restart_enable);
		};
		auto&       pre = m_mesh_pre;
		const auto end_segment = [&] {
			if (pre.segment_key != 0) {
				if (pre.segment.empty()) {
					pre.learned.erase(pre.segment_key);
				} else {
					if (pre.learned.size() > 65536) {
						pre.learned.clear();
					}
					pre.learned[pre.segment_key] = std::move(pre.segment);
				}
			}
			pre.unused += pre.converted.size();
			pre.segment.clear();
			pre.converted.clear();
			pre.segment_key    = 0;
			pre.segment_serial = 0;
		};
		// The render pass the draw ran in continues the segment; one ended for another reason
		// ends it (whatever ended it may have written arguments).
		const auto after_draw = [&](bool own_break) {
			if (!buffer.IsRendering()) {
				end_segment();
			} else if (buffer.RenderSerial() != pre.segment_serial) {
				if (own_break) {
					pre.segment_serial = buffer.RenderSerial();
				} else {
					end_segment();
				}
			}
		};
		static auto& preconvert = Common::LiveSwitches::Get("KYTY_MESH_PRECONVERT", 1);
		if (preconvert.load(std::memory_order_relaxed) == 0) {
			if (pre.segment_key != 0 || !pre.learned.empty()) {
				end_segment();
				pre.learned.clear();
			}
			buffer.EndRendering();
			params.args = args_address(emit.indirect_args, args_size);
			EXIT_IF(params.args == 0);
			run(m_mesh_indirect->Convert(buffer.Recorder(), params));
			return;
		}
		uint64_t key = emit.indirect_args;
		for (const uint64_t value:
		     {params.index_address, uint64_t {params.indexed}, uint64_t {params.max_index_count},
		      uint64_t {params.element_size}, uint64_t {params.primitive_size},
		      uint64_t {params.primitive_step}, uint64_t {params.primitives_per_group},
		      uint64_t {params.fast_launch}}) {
			key = (key ^ value) * 0x100000001b3ull + (key >> 29u);
		}
		key |= 1u; // never 0 (no segment)
		if (pre.segment_serial != 0 && buffer.IsRendering() &&
		    buffer.RenderSerial() == pre.segment_serial) {
			// A later mesh draw of the segment's render pass: learned for the next time, and
			// drawn from its batch conversion when there is one. The batch read the arguments
			// before the render pass began; only GPU-written arguments are batched, and
			// nothing in the render pass since then wrote buffers (a draw that does clears the
			// batch), so they are unchanged.
			if (key != pre.segment_key && pre.segment.size() < MeshIndirectDraw::MaxBatch - 1 &&
			    std::ranges::none_of(pre.segment, [&](const auto& l) { return l.key == key; })) {
				pre.segment.push_back({key, emit.indirect_args, args_size, params});
			}
			const auto it = std::ranges::find_if(pre.converted,
			                                     [&](const auto& c) { return c.key == key; });
			if (it != pre.converted.end() &&
			    m_mesh_indirect->Count() - pre.batch_start < MeshIndirectDraw::RingEntries &&
			    cache.IsRegionGpuModified(emit.indirect_args, args_size)) {
				pre.hits++;
				const auto result = it->result;
				pre.converted.erase(it);
				run(result);
				after_draw(false);
				return;
			}
			pre.misses++;
			buffer.EndRendering();
			params.args = args_address(emit.indirect_args, args_size);
			EXIT_IF(params.args == 0);
			run(m_mesh_indirect->Convert(buffer.Recorder(), params));
			after_draw(true);
			return;
		}
		// The first mesh draw of a render pass: converted together with the draws that
		// followed it last time.
		end_segment();
		pre.starts++;
		pre.segment_key = key;
		buffer.EndRendering();
		params.args = args_address(emit.indirect_args, args_size);
		EXIT_IF(params.args == 0);
		pre.batch_params.assign(1, params);
		pre.batch_keys.assign(1, key);
		if (const auto learned = pre.learned.find(key); learned != pre.learned.end()) {
			for (const auto& item: learned->second) {
				if (pre.batch_params.size() >= MeshIndirectDraw::MaxBatch) {
					break;
				}
				if (!cache.IsRegionGpuModified(item.args, item.args_size)) {
					continue;
				}
				auto item_params = item.params;
				item_params.args = args_address(item.args, item.args_size);
				if (item_params.args == 0) {
					continue;
				}
				pre.batch_params.push_back(item_params);
				pre.batch_keys.push_back(item.key);
			}
		}
		pre.batch_results.resize(pre.batch_params.size());
		pre.batch_start = m_mesh_indirect->Count();
		m_mesh_indirect->ConvertBatch(buffer.Recorder(), pre.batch_params, pre.batch_results);
		pre.batched += pre.batch_params.size() - 1;
		for (size_t i = 1; i < pre.batch_params.size(); i++) {
			pre.converted.push_back({pre.batch_keys[i], pre.batch_results[i]});
		}
		run(pre.batch_results[0]);
		if (buffer.IsRendering()) {
			pre.segment_serial = buffer.RenderSerial();
		} else {
			end_segment();
		}
		return;
	}
	// Research: mesh and legacy quad draws expand their counts on the host, so their
	// GPU-written arguments are read here (a GPU-dirty page drains the queue first).
	auto resolved_draw  = draw;
	auto resolved_emit  = emit;
	auto resolved_index = index_source;
	resolved_emit.indirect_args = 0;
	if (draw.IsIndexed()) {
		vk::DrawIndexedIndirectCommand args {};
		std::memcpy(&args, reinterpret_cast<const void*>(emit.indirect_args), sizeof(args));
		if (args.indexCount == 0 || args.instanceCount == 0) {
			return;
		}
		resolved_draw.index_count    = std::min(args.indexCount, draw.index_count);
		resolved_draw.instance_count = args.instanceCount;
		resolved_draw.first_instance = args.firstInstance;
		resolved_index.address += static_cast<uint64_t>(args.firstIndex) *
		                          index_source.guest_element_size;
		resolved_index.size = static_cast<uint64_t>(resolved_draw.index_count) *
		                      index_source.guest_element_size;
		resolved_emit.vertex_offset  = args.vertexOffset;
		resolved_emit.first_instance = args.firstInstance;
	} else {
		vk::DrawIndirectCommand args {};
		std::memcpy(&args, reinterpret_cast<const void*>(emit.indirect_args), sizeof(args));
		if (args.vertexCount == 0 || args.instanceCount == 0) {
			return;
		}
		resolved_draw.index_count    = args.vertexCount;
		resolved_draw.instance_count = args.instanceCount;
		resolved_draw.first_instance = args.firstInstance;
		resolved_emit.first_vertex   = args.firstVertex;
		resolved_emit.first_instance = args.firstInstance;
	}
	ExecutePreparedDrawResolved(submit_id, buffer, resolved_draw, state, topology, resolved_emit,
	                            resolved_index, primitive_restart_enable);
}

void RenderExecutor::ExecutePreparedDrawResolved(uint64_t submit_id, CommandBuffer& buffer,
                                                 const DrawCallInfo& draw, DrawRenderState& state,
                                                 vk::PrimitiveTopology       topology,
                                                 const DrawEmitInfo&         emit,
                                                 const DrawIndexBufferSource& index_source,
                                                 bool primitive_restart_enable) {
	auto& ucfg = buffer.GetUserConfig();
	const auto vertex_stages =
	    std::span {state.vertex_info.data(), state.programs.VertexStageCount()};
	const bool mesh_active = state.vertex_info[0].stage.program->stage == ShaderType::Mesh;
	uint32_t   mesh_groups = 0;
	uint32_t   mesh_slices = 1;
	uint32_t   mesh_fast_total = 0;
	std::array<uint32_t, ShaderRecompiler::IR::PushData::MeshDrawDwordCount> mesh_draw_data {};
	if (mesh_active) {
		const auto& mesh = state.vertex_info[0].mesh;
		EXIT_NOT_IMPLEMENTED(mesh.fast_launch && (draw.IsIndexed() || primitive_restart_enable));
		static std::atomic_bool restart_warned = false;
		if (primitive_restart_enable && !restart_warned.exchange(true, std::memory_order_relaxed)) {
			std::printf("Warning: primitive restart is not implemented for mesh shaders; "
			            "continuing draw (primitive=%u indexed=%u)\n",
			            static_cast<uint32_t>(ucfg.GetPrimType()), draw.IsIndexed());
		}
		if (mesh.primitives_per_group == 0) {
			EXIT("unsupported mesh draw: primitive=%u indexed=%u restart=%u\n",
			     static_cast<uint32_t>(ucfg.GetPrimType()), draw.IsIndexed(), primitive_restart_enable);
		}
		const auto primitives = mesh.InputPrimitiveCount(draw.index_count);
		if (emit.mesh_draw_data == 0 && (primitives == 0 || draw.instance_count == 0)) {
			return;
		}
		// GPU-converted indirect draws: the counts are computed and bounded on the GPU.
		mesh_groups        = emit.mesh_draw_data != 0 ? 1u
		                                              : (primitives - 1u) / mesh.primitives_per_group + 1u;
		const auto& limits = m_context.GetGraphics().mesh_shader_properties;
		// Non-fast-launch shaders read WorkgroupId.x + WorkgroupId.z * MeshGroupSplitStride, and
		// groups past the draw's end emit no vertices.
		if (mesh_groups > limits.maxMeshWorkGroupCount[0] && !mesh.fast_launch) {
			mesh_slices = (mesh_groups - 1u) / MeshGroupSplitStride + 1u;
			mesh_groups = MeshGroupSplitStride;
		} else if (mesh_groups > limits.maxMeshWorkGroupCount[0]) {
			// Fast launch derives the base vertex from draw(1) + WorkgroupId.x, so issue several
			// draws and advance draw(1) by each one's first group.
			mesh_fast_total = mesh_groups;
			mesh_groups     = MeshGroupSplitStride;
		}
		// KYTY_LOCAL_HACK research: KYTY_MESH_DUMP=<mesh shader hash, hex> (env) prints each new
		// shape of that shader's draws (group counts, passes, render area) once.
		static const uint64_t mesh_dump = [] {
			const char* value = std::getenv("KYTY_MESH_DUMP");
			return value != nullptr ? std::strtoull(value, nullptr, 16) : 0ull;
		}();
		if (mesh_dump != 0 && state.vertex_info[0].stage.program->shader_hash == mesh_dump) {
			static std::mutex            dump_mutex;
			static std::set<std::string> dumped;
			const auto line = fmt::format(
			    "MESH DUMP {:016x}: indirect_gpu={} index_count={} primitives={} per_group={} groups={} "
			    "instances={} slices={} fast_total={} fast_launch={} wave={} passes={} host_threads={} "
			    "max_vertices={} max_primitives={} lds_dwords={} input_prim={}",
			    mesh_dump, emit.mesh_draw_data != 0 ? 1 : 0, draw.index_count, primitives,
			    mesh.primitives_per_group, mesh_groups, draw.instance_count, mesh_slices, mesh_fast_total,
			    mesh.fast_launch ? 1 : 0, mesh.wave_size, mesh.passes, mesh.HostThreads(), mesh.max_vertices,
			    mesh.max_primitives, mesh.lds_size_dwords, mesh.input_primitive);
			std::lock_guard lock(dump_mutex);
			if (dumped.size() < 64 && dumped.insert(line).second) {
				std::printf("%s\n", line.c_str());
				std::fflush(stdout);
			}
		}
		// GPU-converted: the entry an earlier draw of this shader got (~8 draws ago, long finished
		// in practice), as the GPU computed it: x y z groups, count, first, instance, groups.
		if (mesh_dump != 0 && emit.mesh_draw_data != 0 && m_mesh_indirect != nullptr &&
		    state.vertex_info[0].stage.program->shader_hash == mesh_dump) {
			static std::array<vk::DeviceSize, 16> recent {};
			static uint32_t                        seen = 0;
			static uint32_t                        printed = 0;
			const auto old_offset = recent[seen % recent.size()];
			recent[seen % recent.size()] = emit.mesh_groups_offset;
			if (++seen > recent.size() && printed < 80) {
				if (const auto* e = m_mesh_indirect->EntryHost(old_offset); e != nullptr) {
					printed++;
					std::printf("MESH GPU ENTRY: groups %u x %u x %u = %llu, count %u, first %u, instance %u, "
					            "group count %u\n",
					            e[0], e[1], e[2],
					            static_cast<unsigned long long>(e[0]) * e[1] * e[2], e[4], e[5], e[6], e[7]);
					std::fflush(stdout);
				}
			}
		}
		if (mesh_groups > limits.maxMeshWorkGroupCount[0] ||
		    draw.instance_count > limits.maxMeshWorkGroupCount[1] ||
		    mesh_slices > limits.maxMeshWorkGroupCount[2] ||
		    static_cast<uint64_t>(mesh_groups) * draw.instance_count * mesh_slices >
		        limits.maxMeshWorkGroupTotalCount) {
			EXIT("mesh draw exceeds host workgroup limits: %ux%ux%u\n", mesh_groups,
			     draw.instance_count, mesh_slices);
		}
	}

	if (mesh_active && draw.IsIndexed()) {
		// Register the original guest indices for shader reads; PrepareGraphicsBindings
		// synchronizes registered BDA ranges before any draw commands are committed.
		(void)m_context.GetBufferCache().FindBuffer(
		    index_source.address, static_cast<uint64_t>(draw.index_count) *
		                              index_source.guest_element_size);
	}
	LogDrawPhase(draw.Name(), "PrepareBindings");
	const bool phases       = DrawRecordCensus::PhasesOn();
	uint64_t   phase_start  = phases ? DrawRecordCensus::NowNs() : 0;
	auto&                            bindings = m_graphics_bindings;
	std::array<PreparedBindings*, 4> descriptor_stages {};
	uint32_t                         stage_count = 0;
	for (uint32_t i = 0; i < vertex_stages.size(); i++) {
		PrepareBindings(state.vertex_info[i].stage, bindings.vertex[i]);
		descriptor_stages[stage_count++] = &bindings.vertex[i];
	}
	if (state.ps_active) {
		if (!bindings.pixel) bindings.pixel.emplace();
		PrepareBindings(state.ps_input_info.stage, *bindings.pixel);
		descriptor_stages[stage_count++] = &*bindings.pixel;
	}
	const auto stages = std::span {descriptor_stages.data(), stage_count};
	if (phases) DrawRecordCensus::Phase(2, phase_start);
	PrepareGraphicsBindings(stages, std::span {state.color_info, state.color_count});
	if (phases) DrawRecordCensus::Phase(3, phase_start);
	PreparedVertexBuffers vertex_bindings;
	PreparedIndexBuffer   index_binding;
	if (!mesh_active) {
		LogDrawPhase(draw.Name(), "PrepareVertexBuffers");
		vertex_bindings = AcquireVertexBuffers(buffer, state.vertex_info[0]);
		index_binding   = PrepareIndexBuffer(buffer, index_source);
	}
	Buffer*  indirect_buffer = nullptr;
	uint64_t indirect_offset = 0;
	if (emit.indirect_args != 0) {
		EXIT_IF(mesh_active || (emit.indirect_args & 3u) != 0);
		const auto [args_buffer, args_offset] = m_context.GetBufferCache().ObtainBuffer(
		    emit.indirect_args, draw.IsIndexed() ? sizeof(vk::DrawIndexedIndirectCommand)
		                                         : sizeof(vk::DrawIndirectCommand),
		    false);
		EXIT_IF(args_buffer == nullptr || (args_offset & 3u) != 0);
		indirect_buffer = args_buffer;
		indirect_offset = args_offset;
	}
	if (draw.IsIndexed()) {
		LogDrawPhase(draw.Name(), "CreatePipeline");
	}
	// A draw that writes no memory may be skipped while its new pipeline compiles in the
	// background; its render targets are not touched. One that writes buffers or storage images
	// waits for the pipeline, as others may read what it writes.
	const auto writes_memory = [](const ShaderStageRuntime& runtime) {
		return HasShaderBufferWrites(runtime) ||
		       std::ranges::any_of(runtime.program->info.images, [](const auto& image) {
			       return image.written &&
			              image.resource_class == ShaderRecompiler::IR::ImageResourceClass::Storage;
		       });
	};
	bool may_defer = !(state.ps_active && writes_memory(state.ps_input_info.stage));
	for (const auto& stage: vertex_stages) {
		may_defer = may_defer && !writes_memory(stage.stage);
	}
	auto* deferred_pipeline = m_context.GetPipelineCache().TryGetGraphicsPipeline(
	    std::span {state.color_info, state.color_count}, state.depth_info, vertex_stages, buffer,
	    state.ps_active ? &state.ps_input_info : nullptr, topology, primitive_restart_enable,
	    state.programs, may_defer);
	if (phases) DrawRecordCensus::Phase(4, phase_start);
	if (deferred_pipeline == nullptr) {
		return;
	}
	auto& pipeline = *deferred_pipeline;
	vk::ImageAspectFlags feedback_aspects;
	const auto rendering =
	    AcquireRenderTargets(buffer, state.color_info, state.color_count, state.depth_info,
	                         feedback_aspects, stages);
	if (phases) DrawRecordCensus::Phase(5, phase_start);

	// Resource preparation above may synchronously finish and restart the scheduler. From this
	// point onward, every operation targets the current command buffer and cannot touch guest
	// memory.
	const auto vk_buffer = buffer.Recorder();
	if (indirect_buffer != nullptr) {
		// Earlier passes wrote the arguments; indirect reads happen outside the render pass.
		buffer.EndRendering();
		vk::MemoryBarrier barrier {};
		barrier.srcAccessMask =
		    vk::AccessFlagBits::eShaderWrite | vk::AccessFlagBits::eTransferWrite;
		barrier.dstAccessMask = vk::AccessFlagBits::eIndirectCommandRead;
		vk_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eAllGraphics |
		                              vk::PipelineStageFlagBits::eComputeShader |
		                              vk::PipelineStageFlagBits::eTransfer,
		                          vk::PipelineStageFlagBits::eDrawIndirect, {}, 1, &barrier, 0,
		                          nullptr, 0, nullptr);
	}
	uint64_t ps_hash = 0;
	for (const auto* stage: stages) {
		if (stage->runtime != nullptr && *stage->runtime &&
		    stage->runtime->program->stage == ShaderType::Pixel) {
			ps_hash = stage->runtime->program->shader_hash;
		}
	}
	SetDrawDebugPhase(buffer, submit_id, draw, draw.IsIndexed() ? 0x100u : 0x200u, ps_hash);
	if (!mesh_active) {
		CommitVertexBuffers(vk_buffer, vertex_bindings);
	}
	if (state.ps_active && !draw.IsIndexed()) {
		SetDrawDebugPhase(buffer, submit_id, draw, 0x300u, ps_hash);
	}
	CommitBindings(buffer, vk::PipelineBindPoint::eGraphics, pipeline, stages);
	if (phases) DrawRecordCensus::Phase(6, phase_start);
	if (DrawRecordsMode() == 2) {
		VerifyDrawRecord(stages, &pipeline, state.color_info, state.color_count, state.depth_info,
		                 m_context.GetGraphics().presented_frames.load(std::memory_order_relaxed));
	}
	if (mesh_active && emit.mesh_draw_data != 0) {
		// The mesh program reads its parameters through this address (draw_data_indirect).
		mesh_draw_data = {static_cast<uint32_t>(emit.mesh_draw_data),
		                  static_cast<uint32_t>(emit.mesh_draw_data >> 32u), 0u, 0u, 0u, 0u};
		vk_buffer.pushConstants(pipeline.pipeline_layout,
		                        vk::ShaderStageFlagBits::eMeshEXT |
		                            vk::ShaderStageFlagBits::eFragment,
		                        0, sizeof(mesh_draw_data), mesh_draw_data.data());
	} else if (mesh_active) {
		mesh_draw_data = {
		    draw.index_count,
		    draw.IsIndexed() ? static_cast<uint32_t>(emit.vertex_offset) : emit.first_vertex,
		    emit.first_instance, index_source.guest_element_size,
		    static_cast<uint32_t>(index_source.address),
		    static_cast<uint32_t>(index_source.address >> 32u)};
		static_assert(std::tuple_size_v<decltype(mesh_draw_data)> == 6);
		vk_buffer.pushConstants(pipeline.pipeline_layout,
		                        vk::ShaderStageFlagBits::eMeshEXT |
		                            vk::ShaderStageFlagBits::eFragment,
		                        0, sizeof(mesh_draw_data), mesh_draw_data.data());
	} else {
		CommitIndexBuffer(vk_buffer, index_binding);
	}

	// KYTY_LOCAL_HACK (research): KYTY_ZPS=<hex[,hex]> logs the depth state of draws with those
	// pixel shaders once; KYTY_ZPS_MODE=always|nowrite|off changes it for them.
	auto depth_info = state.depth_info;
	{
		static const auto listed = [] {
			std::vector<uint64_t> result;
			const char* value = std::getenv("KYTY_ZPS");
			while (value != nullptr && *value != 0) {
				char* end = nullptr;
				result.push_back(std::strtoull(value, &end, 16));
				value = *end == ',' ? end + 1 : nullptr;
			}
			return result;
		}();
		static const std::string mode = [] {
			const char* value = std::getenv("KYTY_ZPS_MODE");
			return std::string(value != nullptr ? value : "");
		}();
		const uint64_t ps_hash = state.ps_active && state.ps_input_info.stage.program != nullptr
		                             ? state.ps_input_info.stage.program->shader_hash
		                             : 0;
		if (ps_hash != 0 && std::ranges::find(listed, ps_hash) != listed.end()) {
			static std::mutex                   mutex;
			static std::set<std::string>        seen;
			const auto line = fmt::format(
			    "ZPS {:016x}: test={} op={} write={} vs={:016x} stencil={} front(op={} ref={} "
			    "cmp={:x} wr={:x}) bounds={} [{} {}] early_z={} kill={} zexport={}",
			    ps_hash, depth_info.depth_test_enable, vk::to_string(depth_info.depth_compare_op),
			    depth_info.depth_write_enable, vertex_stages.back().stage.program->shader_hash,
			    depth_info.stencil_test_enable, vk::to_string(depth_info.stencil_front.compareOp),
			    depth_info.stencil_front.reference, depth_info.stencil_front.compareMask,
			    depth_info.stencil_front.writeMask, depth_info.depth_bounds_test_enable,
			    depth_info.depth_min_bounds, depth_info.depth_max_bounds,
			    state.ps_input_info.ps_early_z, state.ps_input_info.ps_pixel_kill_enable,
			    state.ps_input_info.ps_depth_export_enable);
			{
				std::lock_guard lock(mutex);
				if (seen.insert(line).second) {
					std::printf("%s\n", line.c_str());
				}
			}
			if (mode == "always") {
				depth_info.depth_compare_op = vk::CompareOp::eAlways;
			} else if (mode == "off") {
				depth_info.depth_test_enable = false;
			} else if (mode == "nostencil") {
				depth_info.stencil_test_enable = false;
			}
		}
	}
	SetGraphicsDynamicParams(buffer, vk_buffer, vertex_stages.back(), depth_info, rendering);
	if (m_context.GetGraphics().attachment_feedback_loop_enabled) {
		vk_buffer.setAttachmentFeedbackLoopEnableEXT(feedback_aspects);
	}

	if (feedback_aspects && buffer.HasPendingGlobalBarrier()) {
		// A feedback loop samples what the render pass's earlier draws wrote: the deferred
		// guest barrier must come first.
		buffer.EndRendering();
	}
	LogDrawPhase(draw.Name(), "BeginRendering");
	if (!draw.IsIndexed()) {
		SetDrawDebugPhase(buffer, submit_id, draw, 0x400u, ps_hash);
	}
	m_context.GetCommandScheduler().BeginRendering(rendering);
	// KYTY_STATE_CACHE: the same pipeline bound again is a no-op. GetDynamicState().valid is set
	// only while the cache is on (SetGraphicsDynamicParams above).
	if (auto& cached = buffer.GetDynamicState();
	    !cached.valid || cached.pipeline != pipeline.pipeline) {
		vk_buffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline.pipeline);
		cached.pipeline = pipeline.pipeline;
	}
	if (!draw.IsIndexed()) {
		SetDrawDebugPhase(buffer, submit_id, draw, 0x500u, ps_hash);
	}
	if (mesh_active && emit.mesh_draw_data != 0) {
		vk_buffer.drawMeshTasksIndirectEXT(emit.mesh_groups_buffer, emit.mesh_groups_offset, 1,
		                                   sizeof(vk::DrawMeshTasksIndirectCommandEXT));
		// Further slices: their own entry and draw parameters (empty ones draw nothing).
		for (uint32_t k = 1; k < emit.mesh_draw_slices; k++) {
			const auto address = emit.mesh_draw_data + k * MeshIndirectDraw::EntryBytes;
			const std::array<uint32_t, 6> slice_data {static_cast<uint32_t>(address),
			                                          static_cast<uint32_t>(address >> 32u), 0u, 0u, 0u, 0u};
			vk_buffer.pushConstants(pipeline.pipeline_layout,
			                        vk::ShaderStageFlagBits::eMeshEXT | vk::ShaderStageFlagBits::eFragment,
			                        0, sizeof(slice_data), slice_data.data());
			vk_buffer.drawMeshTasksIndirectEXT(emit.mesh_groups_buffer,
			                                   emit.mesh_groups_offset + k * MeshIndirectDraw::EntryBytes, 1,
			                                   sizeof(vk::DrawMeshTasksIndirectCommandEXT));
		}
	} else if (mesh_active) {
		if (mesh_fast_total == 0) {
			vk_buffer.drawMeshTasksEXT(mesh_groups, draw.instance_count, mesh_slices);
		} else {
			const auto first_vertex = mesh_draw_data[1];
			for (uint32_t first = 0; first < mesh_fast_total; first += MeshGroupSplitStride) {
				mesh_draw_data[1] = first_vertex + first;
				vk_buffer.pushConstants(pipeline.pipeline_layout,
				                        vk::ShaderStageFlagBits::eMeshEXT |
				                            vk::ShaderStageFlagBits::eFragment,
				                        0, sizeof(mesh_draw_data), mesh_draw_data.data());
				vk_buffer.drawMeshTasksEXT(std::min(MeshGroupSplitStride, mesh_fast_total - first),
				                           draw.instance_count, 1);
			}
		}
	} else {
		EmitDrawPrimitives(ucfg, vk_buffer, draw, emit,
		                   indirect_buffer != nullptr ? indirect_buffer->Handle() : nullptr,
		                   indirect_offset);
	}

	m_context.GetCommandScheduler().ProfileMark(
	    mesh_active ? (emit.mesh_draw_data != 0 ? 4u : 3u) : 2u,
	    vertex_stages[0].stage.program->shader_hash, ps_hash);
	if (!draw.IsIndexed()) {
		SetDrawDebugPhase(buffer, submit_id, draw, 0x600u, ps_hash);
	}
	vk::PipelineStageFlags shader_write_stages = {};
	for (const auto& stage: vertex_stages) {
		if (HasShaderBufferWrites(stage.stage)) {
			shader_write_stages |= ShaderPipelineStages(NativeShaderStage(stage.logical_stage));
		}
	}
	if (state.ps_active && HasShaderBufferWrites(state.ps_input_info.stage)) {
		shader_write_stages |= vk::PipelineStageFlagBits::eFragmentShader;
	}
	if (shader_write_stages) {
		m_context.GetCommandScheduler().EndRendering();
		ShaderWriteBarrier(vk_buffer, shader_write_stages);
	}
	LogDrawPhase(draw.Name(), "DrawComplete");
	if (!draw.IsIndexed()) {
		SetDrawDebugPhase(buffer, submit_id, draw, 0x700u, ps_hash);
	}
}

void RenderExecutor::DrawIndex(uint64_t submit_id, CommandBuffer& buffer,
                               const DrawIndexArgs& args) {
	KYTY_PROFILER_FUNCTION();
	// KYTY_LOCAL_HACK research: draw-record census (KYTY_MEMO_CLASSIFY=1).
	DrawRecordCensus::g_flags = 0;
	struct CensusEnd {
		bool     compute;
		uint64_t start;
		~CensusEnd() { DrawRecordCensus::Record(compute, start); }
	} census_end {false, DrawRecordCensus::NowNs()};

	EXIT_IF(buffer.IsInvalid());
	EXIT_IF(args.offset_source == DrawOffsetSource::DrawState && args.first_instance != 0);
	m_context.GetCommandScheduler().PopPendingOperations();
	auto& ucfg   = buffer.GetUserConfig();
	auto& sh_ctx = buffer.GetShaders();

	buffer.SetDebugInfo(static_cast<uint32_t>(CommandBufferDebugOp::DrawIndex), submit_id,
	                    args.index_count, 0, 1, args.instance_count,
	                    reinterpret_cast<uint64_t>(args.index_addr));

	Common::LockGuard lock = m_context.LockMutexProfiled();
	if (args.gpu_args == 0 && (args.index_count == 0 || args.instance_count == 0)) {
		return;
	}

	if (ConsumeMetadataColorOperation(buffer) || DepthStencilCopy(buffer) ||
	    ResolveColorTargets(buffer, args.render_target_slice_offset)) {
		ResetBindings();
		return;
	}

	if (!DrawHasValidVertexShader(sh_ctx)) {
		return;
	}

	if (graphics_debug_dump_enabled()) {
		LOGF("GraphicsRenderDrawIndex():Shader:\n");
		uc_print("GraphicsRenderDrawIndex():UserConfig:", ucfg);
		hw_print(buffer);

		LOGF("GraphicsRenderDrawIndex():Parameters:\n"
		     "\t index_type_and_size = 0x%08" PRIx32 "\n"
		     "\t index_count         = 0x%08" PRIx32 "\n"
		     "\t index_addr          = 0x%016" PRIx64 "\n"
		     "\t instance_count      = 0x%08" PRIx32 "\n"
		     "\t base_vertex         = 0x%08" PRIx32 "\n"
		     "\t first_instance      = 0x%08" PRIx32 "\n",
		     args.index_type_and_size, args.index_count,
		     reinterpret_cast<uint64_t>(args.index_addr), args.instance_count,
		     static_cast<uint32_t>(args.base_vertex), args.first_instance);
	}

	uc_check(ucfg);

	hw_check(buffer);

	vk::PrimitiveTopology topology = vk::PrimitiveTopology::ePointList;
	if (!GetDrawTopology(ucfg, topology)) {
		return;
	}

	DrawIndexBufferSource index_source {};
	index_source.address = reinterpret_cast<uint64_t>(args.index_addr);
	switch (static_cast<Prospero::IndexType>(args.index_type_and_size)) {
		case Prospero::IndexType::kIndex16:
			index_source.type               = vk::IndexType::eUint16;
			index_source.guest_element_size = 2;
			break;
		case Prospero::IndexType::kIndex32:
			index_source.type               = vk::IndexType::eUint32;
			index_source.guest_element_size = 4;
			break;
		case Prospero::IndexType::kIndex8:
			index_source.guest_element_size = 1;
			break;
		default: EXIT("unknown index_type_and_size: %u\n", args.index_type_and_size);
	}
	index_source.size = static_cast<uint64_t>(args.index_count) * index_source.guest_element_size;
	const bool primitive_restart = ResolvePrimitiveRestart(buffer, index_source);

	std::vector<uint16_t> expanded_indices;
	if (index_source.guest_element_size == 1) {
		EXIT_NOT_IMPLEMENTED(args.index_addr == nullptr);
		const auto* src = static_cast<const uint8_t*>(args.index_addr);
		expanded_indices.resize(args.index_count);
		for (uint32_t i = 0; i < args.index_count; i++) {
			expanded_indices[i] = primitive_restart && src[i] == 0xffu ? 0xffffu : src[i];
		}
		index_source.host_data = expanded_indices.data();
		index_source.size      = expanded_indices.size() * sizeof(uint16_t);
	}

	const DrawCallInfo draw {CommandBufferDebugOp::DrawIndex, args.index_count,
	                        args.instance_count, args.first_instance};
	DrawRenderState state {};
	state.mesh_draw_indirect = args.gpu_args != 0;
	if (!PrepareDrawRenderState(buffer, draw, args.render_target_slice_offset, state)) {
		ResetBindings();
		return;
	}

	LogDrawStateIfNeeded(buffer, draw, state, args.index_type_and_size,
	                     args.index_addr);

	const bool indirect = args.offset_source == DrawOffsetSource::IndirectArgs;
	const auto [vertex_offset, instance_offset] =
	    indirect ? std::pair<int32_t, uint32_t> {0, args.first_instance}
	             : ResolveDrawOffsets(ucfg.GetIndexOffset(), state.vertex_info[0]);

	DrawEmitInfo emit {};
	emit.vertex_offset  = vertex_offset + args.base_vertex;
	emit.first_instance = instance_offset;
	emit.indirect_args  = args.gpu_args;

	ExecutePreparedDraw(submit_id, buffer, draw, state, topology, emit, index_source,
	                    primitive_restart);
	ResetBindings();
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
void RenderExecutor::DrawAuto(uint64_t submit_id, CommandBuffer& buffer, const DrawAutoArgs& args) {
	KYTY_PROFILER_FUNCTION();
	// KYTY_LOCAL_HACK research: draw-record census (KYTY_MEMO_CLASSIFY=1).
	DrawRecordCensus::g_flags = 0;
	struct CensusEnd {
		uint64_t start;
		~CensusEnd() { DrawRecordCensus::Record(false, start); }
	} census_end {DrawRecordCensus::NowNs()};

	EXIT_IF(buffer.IsInvalid());
	EXIT_IF(args.offset_source == DrawOffsetSource::DrawState && args.first_instance != 0);
	m_context.GetCommandScheduler().PopPendingOperations();
	auto& ucfg   = buffer.GetUserConfig();
	auto& sh_ctx = buffer.GetShaders();

	buffer.SetDebugInfo(static_cast<uint32_t>(CommandBufferDebugOp::DrawIndexAuto), submit_id,
	                    args.vertex_count, 0, args.first_vertex, args.instance_count,
	                    args.first_instance);

	Common::LockGuard lock = m_context.LockMutexProfiled();
	if (args.vertex_count == 0 || args.instance_count == 0) {
		return;
	}

	if (ConsumeMetadataColorOperation(buffer) || DepthStencilCopy(buffer) ||
	    ResolveColorTargets(buffer, args.render_target_slice_offset)) {
		ResetBindings();
		return;
	}

	if (!DrawHasValidVertexShader(sh_ctx)) {
		return;
	}

	if (graphics_debug_dump_enabled()) {
		LOGF("GraphicsRenderDrawIndexAuto():Shader:\n");
		uc_print("GraphicsRenderDrawIndexAuto():UserConfig:", ucfg);
		hw_print(buffer);

		LOGF("GraphicsRenderDrawIndexAuto():Parameters:\n"
		     "\t vertex_count        = 0x%08" PRIx32 "\n"
		     "\t instance_count      = 0x%08" PRIx32 "\n"
		     "\t first_vertex        = 0x%08" PRIx32 "\n"
		     "\t first_instance      = 0x%08" PRIx32 "\n",
		     args.vertex_count, args.instance_count, args.first_vertex, args.first_instance);
	}

	uc_check(ucfg);

	hw_check(buffer);

	const DrawCallInfo draw {CommandBufferDebugOp::DrawIndexAuto,
	                         args.vertex_count, args.instance_count, args.first_instance};

	vk::PrimitiveTopology topology = vk::PrimitiveTopology::ePointList;
	if (!GetDrawTopology(ucfg, topology)) {
		ResetBindings();
		return;
	}
	DrawRenderState state {};
	state.mesh_draw_indirect = args.gpu_args != 0;
	if (!PrepareDrawRenderState(buffer, draw, args.render_target_slice_offset, state)) {
		ResetBindings();
		return;
	}

	const bool rect_list = Prospero::IsRectList(ucfg.GetPrimType());
	if (rect_list && state.vertex_info[0].buffers_num == 0 &&
	    state.vertex_info[0].stage.program->param_export_mask == 0 &&
	    state.ps_input_info.input_num != 0) {
		if (graphics_debug_dump_enabled()) {
			LOGF("DrawIndexAuto: skipping rect-list draw with no VS param exports and PS inputs: "
			     "ps_inputs=%u ps=0x%016" PRIx64 " es=0x%016" PRIx64 " gs=0x%016" PRIx64 "\n",
			     state.ps_input_info.input_num, sh_ctx.GetPs().ps_regs.data_addr,
			     sh_ctx.GetVs().es_regs.data_addr, sh_ctx.GetVs().gs_regs.data_addr);
		}
		ResetBindings();
		return;
	}

	LogDrawStateIfNeeded(buffer, draw, state, 0, nullptr);

	const bool indirect = args.offset_source == DrawOffsetSource::IndirectArgs;
	const auto [vertex_offset, instance_offset] =
	    indirect ? std::pair<int32_t, uint32_t> {0, args.first_instance}
	             : ResolveDrawOffsets(ucfg.GetIndexOffset(), state.vertex_info[0]);
	DrawEmitInfo emit {};
	emit.first_vertex = static_cast<uint32_t>(vertex_offset + static_cast<int32_t>(args.first_vertex));
	emit.first_instance = instance_offset;
	emit.indirect_args  = args.gpu_args;

	DrawIndexBufferSource index_source {};
	ExecutePreparedDraw(submit_id, buffer, draw, state, topology, emit, index_source, false);
	ResetBindings();
}

bool RenderExecutor::ResolveColorTargets(CommandBuffer& buffer, uint32_t render_target_slice_offset) {
	const auto& hw = buffer.GetRegisters();
	if (hw.GetColorControl().mode != 3) {
		return false;
	}

	const auto& src_rt = hw.GetRenderTarget(0);
	const auto& dst_rt = hw.GetRenderTarget(1);
	if (src_rt.base.addr == 0 || dst_rt.base.addr == 0) {
		return false;
	}

	RenderColorInfo src {};
	RenderColorInfo dst {};
	ResolveRenderColorTarget(buffer, src, render_target_slice_offset, 0, true, true);
	ResolveRenderColorTarget(buffer, dst, render_target_slice_offset, 1, true, true);
	if (!src.image_id || !dst.image_id) {
		return false;
	}
	if (src.desc.info.data.address == dst.desc.info.data.address &&
	    src.guest_mip_level == dst.guest_mip_level &&
	    src.guest_array_layer == dst.guest_array_layer) {
		return true;
	}

	auto& cache = m_context.GetTextureCache();
	cache.MarkGpuWritten(dst.image_id);
	auto& source      = cache.GetImage(src.image_id);
	auto& destination = cache.GetImage(dst.image_id);
	destination.Resolve(source, {src.guest_mip_level, 1, src.guest_array_layer, 1},
	                    {dst.guest_mip_level, 1, dst.guest_array_layer, 1});
	return true;
}

} // namespace Libs::Graphics
