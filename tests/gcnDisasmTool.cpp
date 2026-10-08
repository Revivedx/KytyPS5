// Local debugging tool (not a test): print a dumped guest shader binary with the emulator's own
// decoder. Usage: gcn_disasm <shader.bin> — decoding stops after the first S_ENDPGM.
#include "graphics/shader/recompiler/frontend/decode/ShaderDecoder.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <vector>

using namespace Libs::Graphics::ShaderRecompiler::Decoder;

int main(int argc, char** argv) {
	if (argc < 2) {
		std::fprintf(stderr, "usage: %s <shader.bin>\n", argv[0]);
		return 1;
	}
	std::ifstream           in(argv[1], std::ios::binary);
	std::vector<char>       bytes((std::istreambuf_iterator<char>(in)), {});
	std::vector<uint32_t>   code(bytes.size() / 4);
	std::memcpy(code.data(), bytes.data(), code.size() * 4);
	for (uint32_t word = 0; word < code.size();) {
		Instruction inst;
		DecodeInstruction(code, word, inst);
		std::printf("%s\n", InstructionToString(inst).c_str());
		if (inst.opcode == Opcode::S_ENDPGM) {
			break;
		}
		word += inst.word_count == 0 ? 1 : inst.word_count;
	}
	return 0;
}
