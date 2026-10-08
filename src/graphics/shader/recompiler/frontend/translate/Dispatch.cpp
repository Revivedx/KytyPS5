#include "common/assert.h"
#include "common/logging/log.h"
#include "graphics/shader/recompiler/frontend/translate/Translator.h"

namespace Libs::Graphics::ShaderRecompiler::Frontend {

void Translator::FailMissingTranslation(const Decoder::Instruction& inst) {
	// A non-fatal compile gives up on the shader instead: its draws or dispatches are skipped.
	// Every missing translation is named, so one log lists all a shader needs.
	if (TranslationNonFatalFlag()) {
		LOGF("shader translation: opcode %s at pc 0x%08x has no IR translation\n",
		     Decoder::InstructionToString(inst).c_str(), inst.pc);
		TranslationUnsupportedFlag() = true;
		return;
	}
	EXIT("opcode %s at pc 0x%08x has no IR translation",
	     Decoder::InstructionToString(inst).c_str(), inst.pc);
}

void Translator::TranslateInstruction(const Decoder::Instruction& inst) {
	current_opcode = inst.opcode;
	current_pc     = inst.pc;

	switch (inst.opcode) {
		case Decoder::Opcode::UNKNOWN:
		case Decoder::Opcode::COUNT:
			if (TranslationNonFatalFlag()) {
				LOGF("shader translation: decoded opcode at pc 0x%08x has no IR translation\n",
				     inst.pc);
				TranslationUnsupportedFlag() = true;
				return;
			}
			EXIT("decoded opcode has no IR translation at pc 0x%08x", inst.pc);
		case Decoder::Opcode::UNSUPPORTED:
			EXIT("unsupported decoded instruction: %s", Decoder::InstructionToString(inst).c_str());
		default: break;
	}

	switch (inst.family) {
		case Decoder::Family::SOP1:
		case Decoder::Family::SOP2:
		case Decoder::Family::SOPK:
		case Decoder::Family::SOPC:
		case Decoder::Family::SOPP: return EmitScalar(inst);
		case Decoder::Family::VOP1:
		case Decoder::Family::VOP2:
		case Decoder::Family::VOP3:
		case Decoder::Family::VOP3P:
		case Decoder::Family::VOPC: return EmitVector(inst);
		case Decoder::Family::SMEM:
		case Decoder::Family::MUBUF:
		case Decoder::Family::MTBUF:
		case Decoder::Family::FLAT:
		case Decoder::Family::DS:
		case Decoder::Family::MIMG: return EmitMemory(inst);
		case Decoder::Family::VINTRP: return EmitInterpolation(inst);
		case Decoder::Family::EXP: return EXP(inst);
		default: return FailMissingTranslation(inst);
	}
}

} // namespace Libs::Graphics::ShaderRecompiler::Frontend

namespace Libs::Graphics::ShaderRecompiler::Frontend {

void Translator::EmitDebugProbe(uint32_t pc, const std::array<uint32_t, 7>& vgprs) {
	// Numbers from 1000 name scalar operand codes (1000 + n: s<n>, 1106/1107: VCC halves).
	const auto reg = [&](size_t index) {
		if (vgprs[index] == UINT32_MAX) {
			return IR::Value(0u);
		}
		if (vgprs[index] >= 1000u) {
			return IR::Value(ReadScalarCode(vgprs[index] - 1000u));
		}
		return IR::Value(ir.GetVectorReg(static_cast<IR::VectorReg>(vgprs[index])));
	};
	ir.Emit(IR::ValueOpcode::DebugProbe,
	        {IR::Value(pc), reg(0), reg(1), reg(2), reg(3), reg(4), reg(5), reg(6)});
}

void Translator::StashDebugProbe(const std::array<uint32_t, 3>& vgprs) {
	for (uint32_t index = 0; index < vgprs.size(); index++) {
		if (vgprs[index] != UINT32_MAX) {
			ir.SetVectorReg(static_cast<IR::VectorReg>(253u + index),
			                ir.GetVectorReg(static_cast<IR::VectorReg>(vgprs[index])));
		}
	}
}

} // namespace Libs::Graphics::ShaderRecompiler::Frontend
