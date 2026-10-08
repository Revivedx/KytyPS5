#include "graphics/shader/recompiler/backend/spirv/spirvEmitterInstructions.h"

#include "common/logging/log.h"
#include "graphics/shader/recompiler/ir/passes/ReadLaneElimination.h"

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>

namespace Libs::Graphics::ShaderRecompiler::Spirv::Emitter {
namespace {

bool UserDataDwordIndex(const EmitterState& state, IR::ScalarReg reg, uint32_t& dword_index) {
	const auto register_index = IR::RegIndex(reg);
	const auto& registers = state.program.bindings.user_data_registers;
	const auto  found     = std::lower_bound(registers.begin(), registers.end(), register_index);
	if (found == registers.end() || *found != register_index) {
		return false;
	}
	dword_index = static_cast<uint32_t>(found - registers.begin());
	return true;
}

uint32_t EmitBuiltinU32(EmitterState& state, IR::StageInputKind kind, uint32_t component) {
	if (kind == IR::StageInputKind::DispatchThreadCount) {
		const auto dword = state.program.bindings.DispatchDimensionsDword();
		if (!DispatchDimensionsIndirect(state)) {
			return EmitShaderDataDwordLoad(state, dword + component);
		}
		// The counts stay where the guest's indirect arguments are; the shader data holds
		// their device address.
		const auto u64     = TypeScalarU64(state);
		const auto low     = Unary(state, spv::OpUConvert, u64, EmitShaderDataDwordLoad(state, dword));
		const auto high    = Unary(state, spv::OpUConvert, u64,
		                           EmitShaderDataDwordLoad(state, dword + 1u));
		const auto base    = Binary(state, spv::OpBitwiseOr, u64, low,
		                            Binary(state, spv::OpShiftLeftLogical, u64, high,
		                                   state.builder.Constant(spv::OpConstant, u64, 32u, 0u)));
		const auto address = Binary(state, spv::OpIAdd, u64, base,
		                            state.builder.Constant(spv::OpConstant, u64,
		                                                   component * 4u, 0u));
		const auto pointer = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpConvertUToPtr, TypePhysicalU32Pointer(state), pointer,
		                          address);
		const auto value = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpLoad, TypeU32(state), value, pointer,
		                          spv::MemoryAccessAlignedMask, 4u);
		return value;
	}
	if (kind == IR::StageInputKind::LocalInvocationIndex) {
		return EmitLocalInvocationIndex(state);
	}
	if (state.lane_count == 2 && (kind == IR::StageInputKind::LocalInvocationId ||
	                              kind == IR::StageInputKind::GlobalInvocationId)) {
		const auto* cs      = ShaderWorkgroupInput(state.program.stage, state.input_info);
		uint32_t    divisor = 1;
		for (uint32_t axis = 0; axis < component; axis++) {
			divisor *= std::max(cs->threads_num[axis], 1u);
		}
		const auto size    = std::max(cs->threads_num[component], 1u);
		const auto divided = EmitBinaryU32(state, spv::OpUDiv, EmitLocalInvocationIndex(state),
		                                   ConstantU32(state, divisor));
		const auto local   = EmitBinaryU32(state, spv::OpUMod, divided, ConstantU32(state, size));
		if (kind == IR::StageInputKind::LocalInvocationId) {
			return local;
		}
		const auto group = EmitInputComponentU32(state, IR::StageInputKind::WorkgroupId, component);
		return EmitAddU32(state, local,
		                  EmitBinaryU32(state, spv::OpIMul, group, ConstantU32(state, size)));
	}
	const bool centroid = kind == IR::StageInputKind::BaryCoordSmoothCentroid;
	const auto variable = InputVariableForKind(
	    state, centroid ? IR::StageInputKind::BaryCoordSmooth : kind);
	if (variable == 0) {
		return ConstantU32(state, 0);
	}
	if (kind == IR::StageInputKind::FrontFacing) {
		const auto value = state.builder.AllocateId();
		const auto bits  = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpLoad, TypeBool(state), value, variable);
		// PS5 initializes v_front_face with float +1.0/-1.0 bits.
		state.builder.AddFunction(spv::OpSelect, TypeU32(state), bits, value,
		                          ConstantU32(state, 0x3f800000u), ConstantU32(state, 0xbf800000u));
		return bits;
	}
	if (kind == IR::StageInputKind::VertexIndex || kind == IR::StageInputKind::InstanceIndex ||
	    kind == IR::StageInputKind::InvocationId || kind == IR::StageInputKind::PrimitiveId ||
	    kind == IR::StageInputKind::Layer || kind == IR::StageInputKind::SampleId) {
		const auto value = state.builder.AllocateId();
		const auto bits  = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpLoad, TypeI32(state), value, variable);
		state.builder.AddFunction(spv::OpBitcast, TypeU32(state), bits, value);
		return bits;
	}
	if (kind == IR::StageInputKind::FragCoord || kind == IR::StageInputKind::TessCoord) {
		const auto pointer = state.builder.AllocateId();
		const auto value   = state.builder.AllocateId();
		const auto bits    = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpAccessChain,
		                          TypePointer(state, spv::StorageClassInput, TypeF32(state)),
		                          pointer, variable, ConstantU32(state, component));
		state.builder.AddFunction(spv::OpLoad, TypeF32(state), value, pointer);
		state.builder.AddFunction(spv::OpBitcast, TypeU32(state), bits, value);
		return bits;
	}
	if (centroid || kind == IR::StageInputKind::BaryCoordSmooth ||
	    kind == IR::StageInputKind::BaryCoordNoPerspective) {
		const auto value   = state.builder.AllocateId();
		const auto bits    = state.builder.AllocateId();
		if (centroid) {
			const auto coordinates = state.builder.AllocateId();
			state.builder.RequireCapability(spv::CapabilityInterpolationFunction);
			state.builder.AddFunction(spv::OpExtInst, TypeF32Vector(state, 3), coordinates,
			                          GlslStd450(state), GLSLstd450InterpolateAtCentroid, variable);
			state.builder.AddFunction(spv::OpCompositeExtract, TypeF32(state), value,
			                          coordinates, component + 1u);
		} else {
			const auto pointer = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpAccessChain,
			                          TypePointer(state, spv::StorageClassInput, TypeF32(state)),
			                          pointer, variable, ConstantU32(state, component + 1u));
			state.builder.AddFunction(spv::OpLoad, TypeF32(state), value, pointer);
		}
		state.builder.AddFunction(spv::OpBitcast, TypeU32(state), bits, value);
		return bits;
	}
	return EmitInputComponentU32(state, kind, component);
}

uint32_t EmitDppWriteCondition(ValueEmitContext& ctx, const IR::DppMoveFlags& flags,
                               uint32_t exec) {
	auto&      state      = ctx.state;
	const auto lane       = EmitSubgroupLocalInvocationId(state);
	const auto bank_shift = state.builder.AllocateId();
	const auto row_shift  = state.builder.AllocateId();
	const auto bank       = state.builder.AllocateId();
	const auto row        = state.builder.AllocateId();
	const auto bank_bit   = state.builder.AllocateId();
	const auto row_bit    = state.builder.AllocateId();
	const auto bank_hit   = state.builder.AllocateId();
	const auto row_hit    = state.builder.AllocateId();
	const auto bank_ok    = state.builder.AllocateId();
	const auto row_ok     = state.builder.AllocateId();
	const auto masks_ok   = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpShiftRightLogical, TypeU32(state), bank_shift, lane,
	                          ConstantU32(state, 2));
	state.builder.AddFunction(spv::OpShiftRightLogical, TypeU32(state), row_shift, lane,
	                          ConstantU32(state, 4));
	state.builder.AddFunction(spv::OpBitwiseAnd, TypeU32(state), bank, bank_shift,
	                          ConstantU32(state, 3));
	state.builder.AddFunction(spv::OpBitwiseAnd, TypeU32(state), row, row_shift,
	                          ConstantU32(state, 3));
	state.builder.AddFunction(spv::OpShiftLeftLogical, TypeU32(state), bank_bit,
	                          ConstantU32(state, 1), bank);
	state.builder.AddFunction(spv::OpShiftLeftLogical, TypeU32(state), row_bit,
	                          ConstantU32(state, 1), row);
	state.builder.AddFunction(spv::OpBitwiseAnd, TypeU32(state), bank_hit,
	                          ConstantU32(state, flags.bank_mask), bank_bit);
	state.builder.AddFunction(spv::OpBitwiseAnd, TypeU32(state), row_hit,
	                          ConstantU32(state, flags.row_mask), row_bit);
	state.builder.AddFunction(spv::OpINotEqual, TypeBool(state), bank_ok, bank_hit,
	                          ConstantU32(state, 0));
	state.builder.AddFunction(spv::OpINotEqual, TypeBool(state), row_ok, row_hit,
	                          ConstantU32(state, 0));
	state.builder.AddFunction(spv::OpLogicalAnd, TypeBool(state), masks_ok, bank_ok, row_ok);
	uint32_t writable = masks_ok;
	if (!flags.bound_control) {
		const auto target  = EmitDppTargetLane(state, flags);
		const auto bounded = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpLogicalAnd, TypeBool(state), bounded, writable,
		                          target.valid);
		writable = bounded;
	}
	const auto result = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpLogicalAnd, TypeBool(state), result, exec, writable);
	return result;
}

uint32_t EmitAttribute(EmitterState& state, uint32_t attr, uint32_t chan) {
	const auto* input = InputBindingForParameter(state, attr);
	if (input == nullptr || input->variable_id == 0) {
		return ConstantU32(state, 0);
	}
	if (state.program.stage == ShaderType::Vertex || state.program.stage == ShaderType::Local) {
		return EmitVertexParameterComponentU32(state, *input, chan & 3u);
	}
	const auto load_per_vertex = [&](uint32_t vertex) {
		const auto pointer = state.builder.AllocateId();
		const auto value   = state.builder.AllocateId();
		state.builder.AddFunction(
		    spv::OpAccessChain, TypePointer(state, spv::StorageClassInput, TypeF32(state)), pointer,
		    input->variable_id, ConstantU32(state, vertex), ConstantU32(state, chan & 3u));
		state.builder.AddFunction(spv::OpLoad, TypeF32(state), value, pointer);
		return value;
	};
	if (input->per_vertex) {
		const auto barycentric_kind = state.input_info.pixel->ps_no_perspective
		                                  ? IR::StageInputKind::BaryCoordNoPerspective
		                                  : IR::StageInputKind::BaryCoordSmooth;
		const auto barycentric      = InputVariableForKind(state, barycentric_kind);
		uint32_t   sum              = 0;
		for (uint32_t vertex = 0; vertex < 3u; vertex++) {
			const auto pointer = state.builder.AllocateId();
			const auto weight  = state.builder.AllocateId();
			const auto product = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpAccessChain,
			                          TypePointer(state, spv::StorageClassInput, TypeF32(state)),
			                          pointer, barycentric, ConstantU32(state, vertex));
			state.builder.AddFunction(spv::OpLoad, TypeF32(state), weight, pointer);
			state.builder.AddFunction(spv::OpFMul, TypeF32(state), product, load_per_vertex(vertex),
			                          weight);
			if (vertex == 0u) {
				sum = product;
			} else {
				const auto next = state.builder.AllocateId();
				state.builder.AddFunction(spv::OpFAdd, TypeF32(state), next, sum, product);
				sum = next;
			}
		}
		const auto bits = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpBitcast, TypeU32(state), bits, sum);
		return bits;
	}
	const auto vector    = state.builder.AllocateId();
	const auto component = state.builder.AllocateId();
	const auto bits      = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpLoad, TypeF32Vector(state, 4), vector, input->variable_id);
	state.builder.AddFunction(spv::OpCompositeExtract, TypeF32(state), component, vector,
	                          chan & 3u);
	state.builder.AddFunction(spv::OpBitcast, TypeU32(state), bits, component);
	return bits;
}

uint32_t EmitInterpolationParameter(ValueEmitContext& ctx, uint32_t attr, uint32_t chan,
                                    uint32_t mode) {
	auto&       state = ctx.state;
	const auto* input = InputBindingForParameter(state, attr);
	if (!input->per_vertex) {
		return EmitAttribute(ctx.state, attr, chan);
	}
	const auto load_vertex = [&](uint32_t vertex) {
		const auto pointer = state.builder.AllocateId();
		const auto value   = state.builder.AllocateId();
		state.builder.AddFunction(
		    spv::OpAccessChain, TypePointer(state, spv::StorageClassInput, TypeF32(state)), pointer,
		    input->variable_id, ConstantU32(state, vertex), ConstantU32(state, chan & 3u));
		state.builder.AddFunction(spv::OpLoad, TypeF32(state), value, pointer);
		return value;
	};

	const auto selected_vertex = (mode + 1u) % 3u;
	uint32_t   value           = load_vertex(selected_vertex);
	if (!PixelParameterIsCustom(state, attr) && mode < 2u) {
		const auto delta = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpFSub, TypeF32(state), delta, value, load_vertex(0));
		value = delta;
	}
	const auto bits = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpBitcast, TypeU32(state), bits, value);
	return bits;
}

uint32_t MrtOutputMode(const EmitterState& state, const IR::ExportInfo& exp) {
	if (state.program.stage != ShaderType::Pixel || exp.kind != IR::ExportTargetKind::Mrt ||
	    exp.index >= std::size(state.input_info.pixel->target_output_mode)) {
		return 0;
	}
	return state.input_info.pixel->target_output_mode[exp.index];
}

uint32_t ExportRawComponent(ValueEmitContext& ctx, uint32_t vector, uint32_t component) {
	const auto value = ctx.state.builder.AllocateId();
	ctx.state.builder.AddFunction(spv::OpCompositeExtract, TypeU32(ctx.state), value, vector,
	                              component);
	return value;
}

uint32_t ExportVector(ValueEmitContext& ctx, uint32_t data, const IR::ExportInfo& exp,
                      bool uint_output) {
	auto& state = ctx.state;
	if (exp.compr && !uint_output) {
		const auto unpack =
		    MrtOutputMode(state, exp) == 5u ? GLSLstd450UnpackUnorm2x16 : GLSLstd450UnpackHalf2x16;
		uint32_t f32[4] = {ConstantF32(state, 0), ConstantF32(state, 0), ConstantF32(state, 0),
		                   ConstantF32(state, 0x3f800000u)};
		for (uint32_t pair = 0; pair < 2u; pair++) {
			if ((exp.en & (3u << (pair * 2u))) == 0u) {
				continue;
			}
			const auto packed   = state.builder.AllocateId();
			const auto unpacked = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpCompositeExtract, TypeU32(state), packed, data, pair);
			state.builder.AddFunction(spv::OpExtInst, TypeF32Vector(state, 2), unpacked,
			                          GlslStd450(state), unpack, packed);
			for (uint32_t lane = 0; lane < 2u; lane++) {
				const auto component = pair * 2u + lane;
				if (((exp.en >> component) & 1u) != 0u) {
					f32[component] = state.builder.AllocateId();
					state.builder.AddFunction(spv::OpCompositeExtract, TypeF32(state),
					                          f32[component], unpacked, lane);
				}
			}
		}
		const auto vector = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpCompositeConstruct, TypeF32Vector(state, 4), vector,
		                          f32[0], f32[1], f32[2], f32[3]);
		return vector;
	}
	uint32_t raw[4] = {
	    ConstantU32(state, 0),
	    ConstantU32(state, 0),
	    ConstantU32(state, 0),
	    ConstantU32(state, uint_output ? 1u : 0x3f800000u),
	};
	if (exp.compr) {
		for (uint32_t pair = 0; pair < 2u; pair++) {
			if ((exp.en & (3u << (pair * 2u))) == 0u) {
				continue;
			}
			const auto packed = ExportRawComponent(ctx, data, pair);
			for (uint32_t lane = 0; lane < 2u; lane++) {
				const auto component = pair * 2u + lane;
				if (((exp.en >> component) & 1u) == 0u) {
					continue;
				}
				raw[component] = state.builder.AllocateId();
				state.builder.AddFunction(spv::OpBitFieldUExtract, TypeU32(state), raw[component],
				                          packed, ConstantU32(state, lane * 16u),
				                          ConstantU32(state, 16));
			}
		}
	} else {
		for (uint32_t component = 0; component < 4u; component++) {
			if (((exp.en >> component) & 1u) != 0u) {
				raw[component] = ExportRawComponent(ctx, data, component);
			}
		}
	}
	if (uint_output) {
		const auto vector = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpCompositeConstruct, TypeU32Vector(state, 4), vector,
		                          raw[0], raw[1], raw[2], raw[3]);
		return vector;
	}
	uint32_t f32[4] {};
	for (uint32_t component = 0; component < 4u; component++) {
		f32[component] = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpBitcast, TypeF32(state), f32[component], raw[component]);
	}
	const auto vector = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpCompositeConstruct, TypeF32Vector(state, 4), vector, f32[0],
	                          f32[1], f32[2], f32[3]);
	return vector;
}

void EmitAuxPositionExport(ValueEmitContext& ctx, uint32_t data, const IR::ExportInfo& exp) {
	auto& state = ctx.state;
	for (uint32_t component = 0; component < 4; component++) {
		if ((exp.en & (1u << component)) == 0) {
			continue;
		}
		const auto output = IR::DecodePositionExportComponent(
		    state.input_info.vertex->pa_cl_vs_out_cntl, exp.index, component);
		if (output.layer || output.viewport) {
			const auto raw = ExportRawComponent(ctx, data, component);
			if (output.layer) {
				const auto layer = state.builder.AllocateId();
				state.builder.AddFunction(spv::OpBitwiseAnd, TypeU32(state), layer, raw,
				                          ConstantU32(state, 0x7ffu));
				const auto pointer = state.program.stage == ShaderType::Mesh
				                         ? MeshOutputPointer(state, IR::StageOutputKind::Layer)
				                         : state.layer_variable;
				state.builder.AddFunction(spv::OpStore, pointer, layer);
			}
			if (output.viewport) {
				// GFX10 MISC.z packs the viewport index in bits 16..19 alongside the layer.
				const auto viewport = state.builder.AllocateId();
				state.builder.AddFunction(spv::OpBitFieldUExtract, TypeU32(state), viewport, raw,
				                          ConstantU32(state, 16), ConstantU32(state, 4));
				state.builder.AddFunction(spv::OpStore, state.viewport_index_variable, viewport);
			}
			continue;
		}
		if (!output.point_size && output.clip_distance == UINT32_MAX &&
		    output.cull_distance == UINT32_MAX) {
			continue;
		}

		const auto raw = ExportRawComponent(ctx, data, component);
		const auto f32 = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpBitcast, TypeF32(state), f32, raw);
		if (output.point_size) {
			state.builder.AddFunction(spv::OpStore, state.point_size_variable, f32);
			continue;
		}
		auto StoreDistance = [&](IR::StageOutputKind kind, uint32_t variable, uint32_t index) {
			if (index == UINT32_MAX) {
				return;
			}
			if (state.program.stage == ShaderType::Mesh) {
				// Kept per lane; the mesh entry point copies it to the vertex's array.
				state.builder.AddFunction(spv::OpStore, MeshOutputPointer(state, kind, index), f32);
				return;
			}
			const auto pointer = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpAccessChain,
			                          TypePointer(state, spv::StorageClassOutput, TypeF32(state)),
			                          pointer, variable, ConstantU32(state, index));
			state.builder.AddFunction(spv::OpStore, pointer, f32);
		};
		StoreDistance(IR::StageOutputKind::ClipDistance, state.clip_distance_variable,
		              output.clip_distance);
		StoreDistance(IR::StageOutputKind::CullDistance, state.cull_distance_variable,
		              output.cull_distance);
	}
}

uint32_t ConvertClipCoordinate(EmitterState& state, uint32_t coordinate, float scale,
                               float offset, float half_extent) {
	const auto window  = state.builder.AllocateId();
	const auto biased  = state.builder.AllocateId();
	const auto divided = state.builder.AllocateId();
	const auto ndc     = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpFMul, TypeF32(state), window, coordinate,
	                          ConstantF32Value(state, scale));
	state.builder.AddFunction(spv::OpFAdd, TypeF32(state), biased, window,
	                          ConstantF32Value(state, offset));
	state.builder.AddFunction(spv::OpFDiv, TypeF32(state), divided, biased,
	                          ConstantF32Value(state, half_extent));
	state.builder.AddFunction(spv::OpFSub, TypeF32(state), ndc, divided,
	                          ConstantF32Value(state, 1.0f));
	return ndc;
}

uint32_t ConvertPositionToClipSpace(EmitterState& state, uint32_t position) {
	const auto& transform = state.input_info.vertex->clip_space;
	uint32_t    components[4] {};
	for (uint32_t i = 0; i < 4; i++) {
		components[i] = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpCompositeExtract, TypeF32(state), components[i], position,
		                          i);
	}
	components[0] = ConvertClipCoordinate(state, components[0], transform.scale[0],
	                                      transform.offset[0], transform.half_extent[0]);
	components[1] = ConvertClipCoordinate(state, components[1], transform.scale[1],
	                                      transform.offset[1], transform.half_extent[1]);
	const auto converted = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpCompositeConstruct, TypeF32Vector(state, 4), converted,
	                          components[0], components[1], components[2], components[3]);
	return converted;
}

} // namespace
uint32_t EmitWqmU64(EmitterState& state, uint32_t value) {
	const auto shifted_one = state.builder.AllocateId();
	const auto merged_one  = state.builder.AllocateId();
	const auto shifted_two = state.builder.AllocateId();
	const auto merged_two  = state.builder.AllocateId();
	const auto quad_bits   = state.builder.AllocateId();
	const auto result      = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpShiftRightLogical, TypeU64(state), shifted_one, value,
	                          ConstantU64(state, 0x0000000100000001ull));
	state.builder.AddFunction(spv::OpBitwiseOr, TypeU64(state), merged_one, value, shifted_one);
	state.builder.AddFunction(spv::OpShiftRightLogical, TypeU64(state), shifted_two, merged_one,
	                          ConstantU64(state, 0x0000000200000002ull));
	state.builder.AddFunction(spv::OpBitwiseOr, TypeU64(state), merged_two, merged_one,
	                          shifted_two);
	state.builder.AddFunction(spv::OpBitwiseAnd, TypeU64(state), quad_bits, merged_two,
	                          ConstantU64(state, 0x1111111111111111ull));
	state.builder.AddFunction(spv::OpIMul, TypeU64(state), result, quad_bits,
	                          ConstantU64(state, 0x0000000f0000000full));
	return result;
}

void EmitPixelHistory(ValueEmitContext& ctx, uint32_t data, uint32_t mrt);

void EmitSetAttribute(ValueEmitContext& ctx, const IR::Inst& inst) {
	auto&       state = ctx.state;
	const auto& exp   = ctx.Export(inst);
	const auto  exec  = ctx.Arg(inst, 1);
	if (state.program.stage == ShaderType::Pixel && exp.vm && state.requirements.pixel_valid_mask &&
	    state.pixel_valid_mask_variable != 0) {
		const auto value = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpSelect, TypeU32(state), value, exec, ConstantU32(state, 1),
		                          ConstantU32(state, 0));
		state.builder.AddFunction(spv::OpStore, state.pixel_valid_mask_variable, value);
	}
	if (exp.kind == IR::ExportTargetKind::Null || exp.en == 0u) {
		return;
	}
	// Skip dormant color exports after their valid mask; MRT1 is reserved for logical alpha.
	if (state.program.stage == ShaderType::Pixel && exp.kind == IR::ExportTargetKind::Mrt &&
	    exp.index != 0 && state.input_info.pixel->alpha_blend_source_remap) {
		return;
	}
	EmitIfCondition(state, exec, [&]() {
		const auto data = ctx.Arg(inst, 0);
		if (exp.kind == IR::ExportTargetKind::Primitive) {
			if (state.program.stage == ShaderType::Mesh) {
				state.builder.AddFunction(spv::OpStore, MeshPrimitivePointer(state),
				                          ExportRawComponent(ctx, data, 0));
			}
			return;
		}
		if (exp.kind == IR::ExportTargetKind::Position && exp.index != 0) {
			EmitAuxPositionExport(ctx, data, exp);
			return;
		}
		if (state.program.stage == ShaderType::Pixel && exp.kind == IR::ExportTargetKind::Mrt) {
			EmitPixelHistory(ctx, data, exp.index);
		}
		if (exp.kind == IR::ExportTargetKind::MrtZ) {
			if ((exp.en & 1u) != 0u && state.depth_variable != 0) {
				const auto raw = ExportRawComponent(ctx, data, 0);
				const auto f32 = state.builder.AllocateId();
				state.builder.AddFunction(spv::OpBitcast, TypeF32(state), f32, raw);
				state.builder.AddFunction(spv::OpStore, state.depth_variable, f32);
			}
			if ((exp.en & 4u) != 0u && state.sample_mask_variable != 0) {
				const auto raw     = ExportRawComponent(ctx, data, 2);
				const auto value   = state.builder.AllocateId();
				const auto pointer = state.builder.AllocateId();
				state.builder.AddFunction(spv::OpBitcast, TypeI32(state), value, raw);
				state.builder.AddFunction(
				    spv::OpAccessChain, TypePointer(state, spv::StorageClassOutput, TypeI32(state)),
				    pointer, state.sample_mask_variable, ConstantU32(state, 0));
				state.builder.AddFunction(spv::OpStore, pointer, value);
			}
			return;
		}
		const auto variable =
		    state.program.stage == ShaderType::Mesh ? 0u : OutputVariableForExport(state, exp);
		if (state.program.stage != ShaderType::Mesh && variable == 0) {
			return;
		}
		const bool mrt =
		    state.program.stage == ShaderType::Pixel && exp.kind == IR::ExportTargetKind::Mrt;
		// An integer render target, or a UINT16_ABGR export (format 7) whatever the target.
		const bool uint_output =
		    MrtOutputMode(state, exp) == 7u ||
		    (mrt && (state.input_info.pixel->target_uint_mask & (1u << exp.index)) != 0);
		const bool sint_output =
		    mrt && (state.input_info.pixel->target_sint_mask & (1u << exp.index)) != 0;
		const auto vector_type = uint_output   ? TypeU32Vector(state, 4)
		                         : sint_output ? TypeI32Vector(state, 4)
		                                       : TypeF32Vector(state, 4);
		auto       value       = ExportVector(ctx, data, exp, uint_output || sint_output);
		if (mrt && !uint_output && !sint_output && NanScrubShader(state.program.shader_hash)) {
			value = EmitScrubNanF32x4(state, value);
		}
		if (sint_output) {
			const auto signed_value = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpBitcast, vector_type, signed_value, value);
			value = signed_value;
			if (exp.compr) {
				const auto shift = state.builder.AllocateId();
				const auto bits  = ConstantU32(state, 16);
				state.builder.AddFunction(spv::OpCompositeConstruct, TypeU32Vector(state, 4), shift,
				                          bits, bits, bits, bits);
				const auto high = state.builder.AllocateId();
				state.builder.AddFunction(spv::OpShiftLeftLogical, vector_type, high, value, shift);
				value = state.builder.AllocateId();
				state.builder.AddFunction(spv::OpShiftRightArithmetic, vector_type, value, high,
				                          shift);
			}
		}
		if (state.program.stage == ShaderType::Pixel && exp.kind == IR::ExportTargetKind::Mrt &&
		    exp.index == 0 && !uint_output && state.input_info.pixel->alpha_blend_source_remap) {
			// Broadcast logical alpha before swizzling the primary output.
			const auto blend_output =
			    OutputVariableForExport(state, {.kind = IR::ExportTargetKind::Mrt, .index = 1});
			if (blend_output != 0) {
				const auto alpha = state.builder.AllocateId();
				state.builder.AddFunction(spv::OpVectorShuffle, vector_type, alpha, value, value,
				                          3u, 3u, 3u, 3u);
				state.builder.AddFunction(spv::OpStore, blend_output, alpha);
			}
		}
		if (state.program.stage == ShaderType::Pixel && exp.kind == IR::ExportTargetKind::Mrt &&
		    exp.index < state.input_info.pixel->target_export_mapping.size()) {
			const auto mapping = state.input_info.pixel->target_export_mapping[exp.index];
			if (!mapping.IsIdentity()) {
				const auto mapped = state.builder.AllocateId();
				state.builder.AddFunction(spv::OpVectorShuffle, vector_type, mapped, value, value,
				                          mapping.Map(0), mapping.Map(1), mapping.Map(2),
				                          mapping.Map(3));
				value = mapped;
			}
		}
		if (exp.kind == IR::ExportTargetKind::Position &&
		    state.input_info.vertex->clip_space.enabled) {
			value = ConvertPositionToClipSpace(state, value);
		}
		if (state.program.stage == ShaderType::Mesh) {
			const auto kind = exp.kind == IR::ExportTargetKind::Position
			                      ? IR::StageOutputKind::Position
			                      : IR::StageOutputKind::Parameter;
			state.builder.AddFunction(spv::OpStore, MeshOutputPointer(state, kind, exp.index),
			                          value);
		} else if (exp.kind == IR::ExportTargetKind::Position) {
			if (state.invalid_position_clip_distance != UINT32_MAX) {
				const auto zero = state.builder.Constant(spv::OpConstantNull, TypeF32Vector(state, 4));
				const auto equal = state.builder.AllocateId();
				const auto invalid = state.builder.AllocateId();
				const auto distance = state.builder.AllocateId();
				const auto distance_pointer = state.builder.AllocateId();
				state.builder.AddFunction(spv::OpFOrdEqual, TypeBoolVector(state, 4), equal,
				                          value, zero);
				const auto all_zero = state.builder.AllocateId();
				state.builder.AddFunction(spv::OpAll, TypeBool(state), all_zero, equal);
				// Guest vertex shaders also kill vertices with a non-finite position (Wolverine
				// hair strands export x = +Inf, w = 0); the PS5 clipper discards such primitives.
				const auto inf = state.builder.AllocateId();
				const auto nan = state.builder.AllocateId();
				const auto any_inf = state.builder.AllocateId();
				const auto any_nan = state.builder.AllocateId();
				state.builder.AddFunction(spv::OpIsInf, TypeBoolVector(state, 4), inf, value);
				state.builder.AddFunction(spv::OpIsNan, TypeBoolVector(state, 4), nan, value);
				state.builder.AddFunction(spv::OpAny, TypeBool(state), any_inf, inf);
				state.builder.AddFunction(spv::OpAny, TypeBool(state), any_nan, nan);
				state.builder.AddFunction(
				    spv::OpLogicalOr, TypeBool(state), invalid, all_zero,
				    Binary(state, spv::OpLogicalOr, TypeBool(state), any_inf, any_nan));
				// A finite stand-in keeps the clipper's interpolation defined (0 * Inf is NaN);
				// the clip distance below still removes every primitive that uses the vertex.
				const auto invalid_vector = state.builder.AllocateId();
				state.builder.AddFunction(spv::OpCompositeConstruct, TypeBoolVector(state, 4),
				                          invalid_vector, invalid, invalid, invalid, invalid);
				const auto stand_in = state.builder.AllocateId();
				state.builder.AddFunction(spv::OpCompositeConstruct, TypeF32Vector(state, 4),
				                          stand_in, ConstantF32Value(state, 0.0f),
				                          ConstantF32Value(state, 0.0f),
				                          ConstantF32Value(state, 0.0f),
				                          ConstantF32Value(state, 1.0f));
				const auto guarded = state.builder.AllocateId();
				state.builder.AddFunction(spv::OpSelect, TypeF32Vector(state, 4), guarded,
				                          invalid_vector, stand_in, value);
				value = guarded;
				// Zero at valid vertices makes a primitive containing an invalid position
				// collapse to its remaining edge, before the undefined 0/0 perspective divide.
				// KYTY_LOCAL_HACK (research): KYTY_CLIP_ALL_HASH=<hex> clips every vertex of that shader.
				static const uint64_t clip_all_hash = [] {
					const char* value = std::getenv("KYTY_CLIP_ALL_HASH");
					return value != nullptr ? std::strtoull(value, nullptr, 16) : 0ull;
				}();
				const bool clip_all = clip_all_hash != 0 && clip_all_hash == state.program.shader_hash;
				state.builder.AddFunction(spv::OpSelect, TypeF32(state), distance, invalid,
				                          ConstantF32Value(state, -1.0f),
				                          ConstantF32Value(state, clip_all ? -1.0f : 0.0f));
				state.builder.AddFunction(
				    spv::OpAccessChain, TypePointer(state, spv::StorageClassOutput, TypeF32(state)),
				    distance_pointer, state.clip_distance_variable,
				    ConstantU32(state, state.invalid_position_clip_distance));
				state.builder.AddFunction(spv::OpStore, distance_pointer, distance);
				static std::atomic_bool logged = false;
				if (!logged.exchange(true, std::memory_order_relaxed)) {
					Log::WriteToConsoleAndLog(
					    "Shader: emitted zero-position clip guard\n");
				}
			}
			const auto pointer = state.builder.AllocateId();
			state.builder.AddFunction(
			    spv::OpAccessChain,
			    TypePointer(state, spv::StorageClassOutput, TypeF32Vector(state, 4)), pointer,
			    variable, ConstantU32(state, 0));
			state.builder.AddFunction(spv::OpStore, pointer, value);
		} else {
			state.builder.AddFunction(spv::OpStore, variable, value);
		}
	});
}

// A guest vertex shader lane that never exports its position has no vertex on hardware, and
// the primitive using it is dropped. Vulkan leaves such an output undefined (RADV: zero, which
// draws wedges to the screen center), so every invocation starts as a killed vertex that the
// position export overwrites. Wolverine hair strands (VS b5c409dab00a0247) depend on this.
bool PixelHistoryEnabled(int* x, int* y, int* w, int* h) {
	static const auto target = [] {
		std::array<int, 5> result {0, -1, -1, 1, 1};
		if (const char* value = std::getenv("KYTY_PIXEL_HISTORY"); value != nullptr) {
			char* end = nullptr;
			result[1] = static_cast<int>(std::strtol(value, &end, 10));
			result[2] = static_cast<int>(std::strtol(end + 1, &end, 10));
			if (*end == ',') {
				result[3] = static_cast<int>(std::strtol(end + 1, &end, 10));
				result[4] = static_cast<int>(std::strtol(end + 1, &end, 10));
			}
			result[0] = 1;
		}
		return result;
	}();
	if (x != nullptr) *x = target[1];
	if (y != nullptr) *y = target[2];
	if (w != nullptr) *w = target[3];
	if (h != nullptr) *h = target[4];
	return target[0] != 0;
}

// KYTY_PIXEL_HISTORY=x,y[,w,h]: every pixel shader color export inside that rectangle records
// hash (low dword), (mrt << 24 | dx << 12 | dy), the four raw exported dwords and FragCoord.z.
void EmitPixelHistory(ValueEmitContext& ctx, uint32_t data, uint32_t mrt) {
	auto& state = ctx.state;
	int   x = 0, y = 0, w = 1, h = 1;
	if (state.pixel_history_frag_coord == 0 || !PixelHistoryEnabled(&x, &y, &w, &h)) {
		return;
	}
	const auto coord = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpLoad, TypeF32Vector(state, 4), coord,
	                          state.pixel_history_frag_coord);
	const auto component = [&](uint32_t index) {
		const auto value = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpCompositeExtract, TypeF32(state), value, coord, index);
		return value;
	};
	const auto offset = [&](uint32_t index, int origin) {
		return Binary(state, spv::OpISub, TypeI32(state),
		              Unary(state, spv::OpConvertFToS, TypeI32(state), component(index)),
		              state.builder.Constant(spv::OpConstant, TypeI32(state),
		                                     static_cast<uint32_t>(origin)));
	};
	const auto dx = Unary(state, spv::OpBitcast, TypeU32(state), offset(0, x));
	const auto dy = Unary(state, spv::OpBitcast, TypeU32(state), offset(1, y));
	// Unsigned compares also reject negative offsets.
	const auto inside = Binary(
	    state, spv::OpLogicalAnd, TypeBool(state),
	    Binary(state, spv::OpULessThan, TypeBool(state), dx, ConstantU32(state, w)),
	    Binary(state, spv::OpULessThan, TypeBool(state), dy, ConstantU32(state, h)));
	EmitIfCondition(state, inside, [&]() {
		const auto packed = Binary(
		    state, spv::OpBitwiseOr, TypeU32(state), ConstantU32(state, mrt << 24u),
		    Binary(state, spv::OpBitwiseOr, TypeU32(state),
		           Binary(state, spv::OpShiftLeftLogical, TypeU32(state), dx, ConstantU32(state, 12)),
		           dy));
		EmitDebugProbe(state, 0, ConstantU32(state, static_cast<uint32_t>(state.program.shader_hash)),
		               packed, ExportRawComponent(ctx, data, 0), ExportRawComponent(ctx, data, 1),
		               ExportRawComponent(ctx, data, 2), ExportRawComponent(ctx, data, 3),
		               Unary(state, spv::OpBitcast, TypeU32(state), component(2)));
	});
}

void EmitDefaultKilledPosition(EmitterState& state) {
	if (state.program.stage != ShaderType::Vertex ||
	    state.invalid_position_clip_distance == UINT32_MAX) {
		return;
	}
	const auto variable =
	    OutputVariableForExport(state, {.kind = IR::ExportTargetKind::Position, .index = 0});
	if (variable == 0) {
		return;
	}
	const auto stand_in = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpCompositeConstruct, TypeF32Vector(state, 4), stand_in,
	                          ConstantF32Value(state, 0.0f), ConstantF32Value(state, 0.0f),
	                          ConstantF32Value(state, 0.0f), ConstantF32Value(state, 1.0f));
	const auto position = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpAccessChain,
	                          TypePointer(state, spv::StorageClassOutput, TypeF32Vector(state, 4)),
	                          position, variable, ConstantU32(state, 0));
	state.builder.AddFunction(spv::OpStore, position, stand_in);
	const auto distance = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpAccessChain,
	                          TypePointer(state, spv::StorageClassOutput, TypeF32(state)), distance,
	                          state.clip_distance_variable,
	                          ConstantU32(state, state.invalid_position_clip_distance));
	state.builder.AddFunction(spv::OpStore, distance, ConstantF32Value(state, -1.0f));
}

uint32_t EmitIdentity(ValueEmitContext&, uint32_t value) {
	return value;
}

void EmitVoid(ValueEmitContext&) {}

void EmitBarrier(EmitterState& state) {
	const auto tessellation = state.program.stage == ShaderType::TessellationControl;
	if (!tessellation && ShaderWorkgroupInput(state.program.stage, state.input_info) == nullptr) {
		// Independent graphics invocations have no native workgroup left to synchronize.
		return;
	}
	const auto memory_scope = tessellation ? spv::ScopeInvocation : spv::ScopeWorkgroup;
	const auto semantics    = tessellation ? spv::MemorySemanticsMaskNone
	                                       : spv::MemorySemanticsAcquireReleaseMask |
	                                             spv::MemorySemanticsWorkgroupMemoryMask;
	state.builder.AddFunction(spv::OpControlBarrier, ConstantU32(state, spv::ScopeWorkgroup),
	                          ConstantU32(state, memory_scope), ConstantU32(state, semantics));
}

uint32_t EmitLaneId(EmitterState& state) {
	return state.program.stage == ShaderType::TessellationControl
	           ? EmitBuiltinU32(state, IR::StageInputKind::InvocationId, 0)
	           : EmitSubgroupLocalInvocationId(state);
}

uint32_t EmitMeshDrawParameter(ValueEmitContext& ctx, const IR::Inst& inst) {
	auto&      state  = ctx.state;
	const auto result = state.builder.AllocateId();
	const auto index  = inst.Arg(0).U32();
	if (state.program.stage != ShaderType::Mesh || index >= IR::PushData::MeshDrawDwordCount) {
		ctx.Fail(inst, "invalid mesh draw parameter");
	}
	const auto push_dword = [&](uint32_t dword) {
		const auto element = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpAccessChain, TypePushConstantElementPointer(state),
		                          element, state.push_constant_variable, ConstantU32(state, 0),
		                          ConstantU32(state, dword));
		return Unary(state, spv::OpLoad, TypeU32(state), element);
	};
	if (MeshDrawDataIndirect(state)) {
		// The parameters were written on the GPU from the guest's indirect arguments.
		const auto u64     = TypeScalarU64(state);
		const auto base    = Binary(
            state, spv::OpBitwiseOr, u64, Unary(state, spv::OpUConvert, u64, push_dword(0)),
            Binary(state, spv::OpShiftLeftLogical, u64,
		           Unary(state, spv::OpUConvert, u64, push_dword(1)),
		           state.builder.Constant(spv::OpConstant, u64, 32u, 0u)));
		const auto address = Binary(state, spv::OpIAdd, u64, base,
		                            state.builder.Constant(spv::OpConstant, u64, index * 4u, 0u));
		const auto pointer = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpConvertUToPtr, TypePhysicalU32Pointer(state), pointer,
		                          address);
		state.builder.AddFunction(spv::OpLoad, TypeU32(state), result, pointer,
		                          spv::MemoryAccessAlignedMask, 4u);
		return result;
	}
	const auto pointer = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpAccessChain, TypePushConstantElementPointer(state), pointer,
	                          state.push_constant_variable, ConstantU32(state, 0),
	                          ConstantU32(state, index));
	state.builder.AddFunction(spv::OpLoad, TypeU32(state), result, pointer);
	return result;
}

uint32_t EmitGetUserData(EmitterState& state, IR::ScalarReg reg) {

	uint32_t dword = 0;
	if (!UserDataDwordIndex(state, reg, dword)) {
		return ConstantU32(state, 0);
	} else {
		return EmitShaderDataDwordLoad(state, dword);
	}
}

uint32_t EmitGetBuiltin(ValueEmitContext& ctx, IR::Value kind, IR::Value index) {
	return EmitBuiltinU32(ctx.state, static_cast<IR::StageInputKind>(kind.U32()), index.U32());
}

uint32_t EmitUndefU1(EmitterState& state, const IR::Inst& inst) {
	const auto result = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpUndef, TypeId(state, inst.GetType()), result);
	return result;
}

uint32_t EmitDppMoveU32(ValueEmitContext& ctx, const IR::Inst& inst) {
	auto&      state    = ctx.state;
	const auto flags    = inst.Flags<IR::DppMoveFlags>();
	const auto target   = EmitDppTargetLane(state, flags);
	const auto shuffled = ctx.Shuffle(inst, 0, target.lane);
	if (flags.fetch_inactive) {
		return shuffled;
	}
	const auto ballot        = ctx.Ballot(inst.Arg(1));
	const auto source_active = EmitBallotLaneActiveBool(state, ballot, target.lane);
	const auto can_fetch     = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpLogicalAnd, TypeBool(state), can_fetch, target.valid,
	                          source_active);
	return EmitNative<spv::OpSelect, IR::Type::U32>(ctx.state, can_fetch, shuffled,
	                                                ConstantU32(state, 0));
}

// Research: DPP16 with FI=0 treats an inactive source lane like an out-of-range one, so with BC=0
// the destination lane is not written. A source lane the host has no invocation for (a pixel warp
// with uncovered quads) exists on the hardware and holds the identity a reduction seeded it with;
// leaving the destination unwritten has the same effect. A min-reduction combined those lanes as 0
// instead, its loop picked an item no lane held, and the GPU hung (PS 0x7dce3261e625949c).
uint32_t EmitDppUpdateU32(ValueEmitContext& ctx, const IR::Inst& inst) {
	auto&      state = ctx.state;
	const auto flags = inst.Flags<IR::DppMoveFlags>();
	auto       write = EmitDppWriteCondition(ctx, flags, ctx.Arg(inst, 2));
	if (!flags.dpp8) {
		const auto target = EmitDppTargetLane(state, flags);
		const auto host_lane =
		    ctx.other_half != nullptr
		        ? EmitBinaryU32(state, spv::OpBitwiseAnd, target.lane, ConstantU32(state, 31))
		        : target.lane;
		auto source_ok = EmitSubgroupLaneActiveBool(state, host_lane);
		if (!flags.bound_control && !flags.fetch_inactive) {
			source_ok = EmitNative<spv::OpLogicalAnd, IR::Type::U1>(
			    state, source_ok,
			    EmitBallotLaneActiveBool(state, ctx.Ballot(inst.Arg(2)), target.lane));
		}
		write = EmitNative<spv::OpLogicalAnd, IR::Type::U1>(state, write, source_ok);
	}
	return EmitNative<spv::OpSelect, IR::Type::U32>(state, write, ctx.Arg(inst, 0),
	                                                ctx.Arg(inst, 1));
}

uint32_t EmitConditionRef(ValueEmitContext& ctx, const IR::Inst& inst) {
	if (ctx.other_half == nullptr) return ctx.Arg(inst, 0);
	// A native scalar branch makes one decision for both emulated wave halves.
	if (ctx.half != 0) return ctx.other_half->Def(IR::Value(&inst));
	const auto kind = inst.Flags<CFG::BranchCondition>();
	if (kind == CFG::BranchCondition::ScalarInstruction) return ctx.Arg(inst, 0);
	const auto ballot = ctx.Ballot(inst.Arg(0));
	const auto low = ctx.state.builder.AllocateId();
	const auto high = ctx.state.builder.AllocateId();
	const auto combined = ctx.state.builder.AllocateId();
	const auto result = ctx.state.builder.AllocateId();
	ctx.state.builder.AddFunction(spv::OpCompositeExtract, TypeU32(ctx.state), low, ballot, 0);
	ctx.state.builder.AddFunction(spv::OpCompositeExtract, TypeU32(ctx.state), high, ballot, 1);
	const bool zero = kind == CFG::BranchCondition::ExecZero ||
	                  kind == CFG::BranchCondition::VccZero || kind == CFG::BranchCondition::SccZero;
	ctx.state.builder.AddFunction(zero ? spv::OpBitwiseAnd : spv::OpBitwiseOr,
	                              TypeU32(ctx.state), combined, low, high);
	ctx.state.builder.AddFunction(zero ? spv::OpIEqual : spv::OpINotEqual,
	                              TypeBool(ctx.state), result, combined,
	                              ConstantU32(ctx.state, zero ? ~0u : 0u));
	return result;
}

uint32_t EmitBallot(ValueEmitContext& ctx, IR::Value predicate) {
	return ctx.Ballot(predicate);
}

uint32_t EmitReadFirstLane(ValueEmitContext& ctx, const IR::Inst& inst) {
	const auto ballot = ctx.Ballot(inst.Arg(1));
	const auto lane   = ctx.FirstLane(ballot);
	return ctx.Shuffle(inst, 0, lane);
}

// The lanes' reduction, natively over the lanes the host subgroup has (IR::MatchLaneReduction).
static uint32_t EmitLaneReduction(ValueEmitContext& ctx, const IR::LaneReduction& reduction) {
	auto& state = ctx.state;
	// A wave64 on a 32-wide subgroup keeps lanes 32-63 in the second half.
	const auto host_lanes = state.lane_count == 2 ? 32u : state.program.wave_size;
	auto&      lane = reduction.first_lane / host_lanes == ctx.half ? ctx : *ctx.other_half;
	const auto subid = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpLoad, TypeU32(state), subid,
	                          state.subgroup_local_invocation_id_variable);
	const auto in_range =
	    Binary(state, spv::OpULessThan, TypeBool(state),
	           Binary(state, spv::OpISub, TypeU32(state), subid,
	                  ConstantU32(state, reduction.first_lane % host_lanes)),
	           ConstantU32(state, reduction.lanes));
	const auto contribution =
	    Select(state, TypeU32(state), in_range, lane.Def(reduction.source),
	           ConstantU32(state, IR::ReductionIdentity(reduction.operation)));
	spv::Op operation = spv::OpGroupNonUniformIAdd;
	switch (reduction.operation) {
		case IR::ValueOpcode::UMax32: operation = spv::OpGroupNonUniformUMax; break;
		case IR::ValueOpcode::UMin32: operation = spv::OpGroupNonUniformUMin; break;
		case IR::ValueOpcode::SMax32: operation = spv::OpGroupNonUniformSMax; break;
		case IR::ValueOpcode::SMin32: operation = spv::OpGroupNonUniformSMin; break;
		case IR::ValueOpcode::BitwiseOr32: operation = spv::OpGroupNonUniformBitwiseOr; break;
		case IR::ValueOpcode::BitwiseAnd32: operation = spv::OpGroupNonUniformBitwiseAnd; break;
		case IR::ValueOpcode::BitwiseXor32: operation = spv::OpGroupNonUniformBitwiseXor; break;
		default: break;
	}
	const auto result = state.builder.AllocateId();
	state.builder.AddFunction(operation, TypeU32(state), result,
	                          ConstantU32(state, spv::ScopeSubgroup), spv::GroupOperationReduce,
	                          contribution);
	return result;
}

// Research: V_READLANE may name a lane the host has no invocation for (a pixel-shader warp with
// uncovered quads), and OpGroupNonUniformShuffle then returns an undefined value. A wave
// reduction (DPP row shifts, then V_READLANE of lanes 15 and 31) read garbage there, and the loop
// it bounds never ended: a GPU hang in two pixel shaders. Such a lane reads the highest active
// lane at or below it in its 16-lane row, which holds the row prefix those reductions compute, or
// 0 when the row has no active lane.
uint32_t EmitReadLane(ValueEmitContext& ctx, const IR::Inst& inst) {
	if (const auto reduction = IR::MatchLaneReduction(inst, ctx.state.program.wave_size)) {
		return EmitLaneReduction(ctx, *reduction);
	}
	auto&      state    = ctx.state;
	const auto lane     = ctx.Arg(inst, 1);
	const auto physical = ctx.other_half != nullptr
	                          ? EmitBinaryU32(state, spv::OpBitwiseAnd, lane, ConstantU32(state, 31))
	                          : lane;
	const auto active   = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpGroupNonUniformBallot, TypeU32Vector(state, 4), active,
	                          ConstantU32(state, spv::ScopeSubgroup), ConstantBool(state, true));
	auto word = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpCompositeExtract, TypeU32(state), word, active, 0);
	if (ctx.other_half == nullptr && state.program.wave_size == 64u) {
		const auto high    = state.builder.AllocateId();
		const auto in_high = state.builder.AllocateId();
		const auto chosen  = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpCompositeExtract, TypeU32(state), high, active, 1);
		state.builder.AddFunction(spv::OpUGreaterThanEqual, TypeBool(state), in_high, physical,
		                          ConstantU32(state, 32));
		state.builder.AddFunction(spv::OpSelect, TypeU32(state), chosen, in_high, high, word);
		word = chosen;
	}
	const auto bit      = EmitBinaryU32(state, spv::OpBitwiseAnd, physical, ConstantU32(state, 31));
	const auto row_base = EmitBinaryU32(state, spv::OpBitwiseAnd, bit, ConstantU32(state, 16));
	// Bits row_base..bit: (2 << bit) - 1 wraps to all ones for bit 31.
	const auto up_to = EmitBinaryU32(
	    state, spv::OpISub,
	    EmitBinaryU32(state, spv::OpShiftLeftLogical, ConstantU32(state, 2), bit),
	    ConstantU32(state, 1));
	const auto below = EmitBinaryU32(
	    state, spv::OpISub,
	    EmitBinaryU32(state, spv::OpShiftLeftLogical, ConstantU32(state, 1), row_base),
	    ConstantU32(state, 1));
	const auto candidates = EmitBinaryU32(
	    state, spv::OpBitwiseAnd, word,
	    EmitBinaryU32(state, spv::OpBitwiseAnd, up_to,
	                  EmitNative<spv::OpNot, IR::Type::U32>(state, below)));
	const auto highest = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpExtInst, TypeU32(state), highest, GlslStd450(state),
	                          GLSLstd450FindUMsb, candidates);
	const auto any = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpINotEqual, TypeBool(state), any, candidates,
	                          ConstantU32(state, 0));
	const auto source = EmitBinaryU32(
	    state, spv::OpBitwiseOr,
	    EmitBinaryU32(state, spv::OpBitwiseAnd, lane, ConstantU32(state, ~31u)),
	    EmitNative<spv::OpSelect, IR::Type::U32>(state, any, highest, ConstantU32(state, 0)));
	const auto value = ctx.Shuffle(inst, 0, source);
	return EmitNative<spv::OpSelect, IR::Type::U32>(state, any, value, ConstantU32(state, 0));
}

uint32_t EmitWriteLane(ValueEmitContext& ctx, const IR::Inst& inst) {
	auto&      state = ctx.state;
	const auto hit   = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpIEqual, TypeBool(state), hit,
	                          EmitSubgroupLocalInvocationId(state), ctx.Arg(inst, 2));
	return EmitNative<spv::OpSelect, IR::Type::U32>(ctx.state, hit, ctx.Arg(inst, 1),
	                                                ctx.Arg(inst, 0));
}

uint32_t EmitPermlane16U32(ValueEmitContext& ctx, const IR::Inst& inst) {
	auto&      state     = ctx.state;
	const auto flags     = inst.Flags<IR::PermlaneFlags>();
	const auto subid     = EmitSubgroupLocalInvocationId(state);
	const auto row       = state.builder.AllocateId();
	const auto row_value = state.builder.AllocateId();
	const auto lane      = state.builder.AllocateId();
	const auto lane8     = state.builder.AllocateId();
	const auto shift     = state.builder.AllocateId();
	const auto upper     = state.builder.AllocateId();
	const auto selected  = state.builder.AllocateId();
	const auto shifted   = state.builder.AllocateId();
	const auto index     = state.builder.AllocateId();
	const auto target    = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpBitwiseAnd, TypeU32(state), row, subid,
	                          ConstantU32(state, 0xfffffff0u));
	if (flags.x16) {
		state.builder.AddFunction(spv::OpBitwiseXor, TypeU32(state), row_value, row,
		                          ConstantU32(state, 16));
	} else {
		state.builder.AddFunction(spv::OpCopyObject, TypeU32(state), row_value, row);
	}
	state.builder.AddFunction(spv::OpBitwiseAnd, TypeU32(state), lane, subid,
	                          ConstantU32(state, 15));
	state.builder.AddFunction(spv::OpBitwiseAnd, TypeU32(state), lane8, lane,
	                          ConstantU32(state, 7));
	state.builder.AddFunction(spv::OpShiftLeftLogical, TypeU32(state), shift, lane8,
	                          ConstantU32(state, 2));
	state.builder.AddFunction(spv::OpUGreaterThanEqual, TypeBool(state), upper, lane,
	                          ConstantU32(state, 8));
	state.builder.AddFunction(spv::OpSelect, TypeU32(state), selected, upper, ctx.Arg(inst, 2),
	                          ctx.Arg(inst, 1));
	state.builder.AddFunction(spv::OpShiftRightLogical, TypeU32(state), shifted, selected, shift);
	state.builder.AddFunction(spv::OpBitwiseAnd, TypeU32(state), index, shifted,
	                          ConstantU32(state, 15));
	state.builder.AddFunction(spv::OpBitwiseOr, TypeU32(state), target, row_value, index);
	const auto shuffled = ctx.Shuffle(inst, 0, target);
	uint32_t   result   = shuffled;
	if (!flags.fetch_inactive) {
		const auto source_exec = ctx.Shuffle(inst, 3, target);
		result                 = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpSelect, TypeU32(state), result, source_exec, shuffled,
		                          ConstantU32(state, 0));
	}
	return result;
}

uint32_t EmitGetAttribute(ValueEmitContext& ctx, const IR::Inst& inst) {
	return EmitAttribute(ctx.state, inst.Arg(0).U32(), inst.Arg(1).U32());
}

uint32_t EmitGetInterpolationParameter(ValueEmitContext& ctx, const IR::Inst& inst) {
	return EmitInterpolationParameter(ctx, inst.Arg(0).U32(), inst.Arg(1).U32(), inst.Arg(2).U32());
}

uint32_t EmitGetShaderBase(ValueEmitContext& ctx) {
	// Guest S_GETPC values stay shader-relative in SPIR-V, matching the runtime ABI. The
	// runtime descriptor evaluator supplies the mapped shader base for host-side planning.
	return ctx.Def(IR::Value(uint64_t {0}));
}

namespace {
std::atomic<uint32_t> g_shader_clock_divisor {8};
std::atomic<uint64_t> g_shader_clock_offset {0};
} // namespace

uint32_t EmitReadClockRealtime64(ValueEmitContext& ctx) {
	auto&      state   = ctx.state;
	const auto divisor = g_shader_clock_divisor.load(std::memory_order_relaxed);
	const auto clock   = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpReadClockKHR, TypeScalarU64(state), clock,
	                          ConstantU32(state, spv::ScopeDevice));
	// Rescale the device clock to the 100 MHz clock S_MEMREALTIME reads on the console.
	const auto scaled = divisor <= 1u ? clock
	                                  : Binary(state, spv::OpUDiv, TypeScalarU64(state), clock,
	                                           ConstantDeviceAddress(state, divisor));
	const auto offset = g_shader_clock_offset.load(std::memory_order_relaxed);
	const auto shifted = offset == 0u ? scaled
	                                  : Binary(state, spv::OpIAdd, TypeScalarU64(state), scaled,
	                                           ConstantDeviceAddress(state, offset));
	const auto result = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpBitcast, TypeU64(state), result, shifted);
	return result;
}

void EmitUnreachable(ValueEmitContext& ctx, const IR::Inst& inst) {
	ctx.Fail(inst, "must be lowered before SPIR-V emission");
}

// KYTY_LOCAL_HACK (debug probe): device address of the host-visible probe buffer, or 0.
// Layout: word 0 counts records; record r (8 words: sequence, 7 values) sits at byte
// 32 + 32 * (r % DebugProbeMaxRecords).
std::atomic<uint64_t> g_debug_probe_address {0};
constexpr uint32_t    DebugProbeMaxRecords = 65536;

void EmitDebugProbe(EmitterState& state, uint32_t pc, uint32_t v0, uint32_t v1, uint32_t v2,
                    uint32_t v3, uint32_t v4, uint32_t v5, uint32_t v6) {
	const auto base = g_debug_probe_address.load(std::memory_order_relaxed);
	if (base == 0) {
		return;
	}
	const auto pointer_at = [&](uint32_t address) {
		return Unary(state, spv::OpConvertUToPtr, TypePhysicalU32Pointer(state), address);
	};
	// KYTY_PROBE_FILTER=clip: only finite positions (v0..v3 = x, y, z, w) far outside the
	// view volume (w <= 0.001, |x| or |y| > 8 |w|) are recorded.
	static const bool clip_filter = [] {
		const char* filter = std::getenv("KYTY_PROBE_FILTER");
		return filter != nullptr && std::strcmp(filter, "clip") == 0;
	}();
	uint32_t condition = ConstantBool(state, true);
	if (clip_filter) {
		const auto f = [&](uint32_t value) { return Unary(state, spv::OpBitcast, TypeF32(state), value); };
		const auto x = f(v0), y = f(v1), z = f(v2), w = f(v3);
		const auto finite = [&](uint32_t value) {
			return Binary(state, spv::OpLogicalAnd, TypeBool(state),
			              Unary(state, spv::OpLogicalNot, TypeBool(state),
			                    Unary(state, spv::OpIsInf, TypeBool(state), value)),
			              Unary(state, spv::OpLogicalNot, TypeBool(state),
			                    Unary(state, spv::OpIsNan, TypeBool(state), value)));
		};
		const auto all_finite =
		    Binary(state, spv::OpLogicalAnd, TypeBool(state),
		           Binary(state, spv::OpLogicalAnd, TypeBool(state), finite(x), finite(y)),
		           Binary(state, spv::OpLogicalAnd, TypeBool(state), finite(z), finite(w)));
		const auto abs = [&](uint32_t value) { return EmitGlsl<GLSLstd450FAbs, IR::Type::F32>(state, value); };
		const auto limit = Binary(state, spv::OpFMul, TypeF32(state), abs(w),
		                          ConstantF32Value(state, 8.0f));
		const auto outside = Binary(
		    state, spv::OpLogicalOr, TypeBool(state),
		    Binary(state, spv::OpFOrdLessThanEqual, TypeBool(state), w,
		           ConstantF32Value(state, 0.001f)),
		    Binary(state, spv::OpLogicalOr, TypeBool(state),
		           Binary(state, spv::OpFOrdGreaterThan, TypeBool(state), abs(x), limit),
		           Binary(state, spv::OpFOrdGreaterThan, TypeBool(state), abs(y), limit)));
		condition = Binary(state, spv::OpLogicalAnd, TypeBool(state), all_finite, outside);
	}
	EmitIfCondition(state, condition, [&]() {
	const auto counter = pointer_at(ConstantDeviceAddress(state, base));
	const auto record  = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpAtomicIAdd, TypeU32(state), record, counter,
	                          ConstantU32(state, spv::ScopeDevice), ConstantU32(state, 0),
	                          ConstantU32(state, 1));
	// A ring: the newest records overwrite the oldest; word 0 of a record is its sequence.
	const auto slot = Binary(state, spv::OpBitwiseAnd, TypeU32(state), record,
	                         ConstantU32(state, DebugProbeMaxRecords - 1));
	{
		const auto offset = Binary(
		    state, spv::OpIAdd, TypeScalarU64(state), ConstantDeviceAddress(state, base + 32),
		    Unary(state, spv::OpUConvert, TypeScalarU64(state),
		          Binary(state, spv::OpShiftLeftLogical, TypeU32(state), slot,
		                 ConstantU32(state, 5))));
		(void)pc;
		const uint32_t values[8] {record, v0, v1, v2, v3, v4, v5, v6};
		for (uint32_t word = 0; word < 8; word++) {
			const auto address = Binary(state, spv::OpIAdd, TypeScalarU64(state), offset,
			                            ConstantDeviceAddress(state, word * 4u));
			state.builder.AddFunction(spv::OpStore, pointer_at(address), values[word],
			                          spv::MemoryAccessAlignedMask, 4u);
		}
	}
	});
}

} // namespace Libs::Graphics::ShaderRecompiler::Spirv::Emitter

namespace Libs::Graphics::ShaderRecompiler::Spirv {

void SetDebugProbeAddress(uint64_t address) {
	Emitter::g_debug_probe_address.store(address, std::memory_order_relaxed);
}

void SetShaderClockDivisor(uint32_t divisor) {
	Emitter::g_shader_clock_divisor.store(std::max(divisor, 1u), std::memory_order_relaxed);
}

void SetShaderClockOffset(uint64_t offset) {
	Emitter::g_shader_clock_offset.store(offset, std::memory_order_relaxed);
}

} // namespace Libs::Graphics::ShaderRecompiler::Spirv
