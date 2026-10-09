#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_RENDERDRAW_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_RENDERDRAW_H_

#include "graphics/guest_gpu/hardwareContext.h"

#include <array>
#include <cstdint>
#include <utility>

namespace Libs::Graphics {

struct ShaderVertexInputInfo;

// KYTY_DRAW_PREP scanner (drawPrepScanner.cpp).
[[nodiscard]] bool DrawPrepHasVertexShader(const HW::Shader& sh_ctx);
[[nodiscard]] bool DrawPrepPixelState(const HW::Context& ctx, const HW::Shader& sh_ctx,
                                      std::array<Prospero::ColorComponentMapping, 8>& target_export_mapping);

[[nodiscard]] std::pair<int32_t, uint32_t>
ResolveDrawOffsets(uint32_t index_offset, const ShaderVertexInputInfo& vs_input_info);

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_RENDERDRAW_H_
