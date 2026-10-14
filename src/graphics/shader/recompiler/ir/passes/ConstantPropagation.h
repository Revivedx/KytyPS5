#pragma once

#include "graphics/shader/recompiler/ir/Block.h"

namespace Libs::Graphics::ShaderRecompiler::IR {

void ConstantPropagationPass(const BlockList& blocks, uint32_t wave_size = 64);
// KYTY_WAVE64_MASKS (environment): lane mask projection in wave64 too (tests may set it).
extern bool g_wave64_mask_projection;
struct Program;
uint32_t SimplifyBoundedLoopRegisters(Program& program);

} // namespace Libs::Graphics::ShaderRecompiler::IR
