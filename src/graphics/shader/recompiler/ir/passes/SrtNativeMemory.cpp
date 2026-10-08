// Executable memory for the native SRT walker (SrtNative.cpp). A file of its own so that the OS
// headers stay out of the IR headers, and so that the walker does not depend on the common
// library (its standalone test targets link only fmt).

#include <cstddef>
#include <cstdint>
#include <cstring>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <sys/mman.h>

#include <cstdlib>
#include <mutex>
#include <vector>
#endif

namespace Libs::Graphics::ShaderRecompiler::IR {

#if !defined(_WIN32)
// KYTY_LOCAL_HACK KYTY_JIT_ARENA (env, default 0): plans share 64 MiB read/write/execute arenas
// aligned to 2 MiB with MADV_HUGEPAGE, instead of one mmap per plan. Thousands of plans each on
// their own 4 KiB pages made the CP's JIT time an iTLB/i-cache miss storm (pf1 perf 2026-10-06:
// JIT 21% of the CP spread over ~2400 code pages). Arena code is never freed (plans live as long
// as the pipeline cache); a code block larger than an arena quarter takes the old path.
namespace {
constexpr size_t kArenaSize = size_t {64} << 20u;
constexpr size_t kHugePage  = size_t {2} << 20u;
struct CodeArena {
	uint8_t* base = nullptr;
	size_t   used = 0;
};
std::mutex             g_arena_mutex;
std::vector<CodeArena> g_arenas;

bool UseArena() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_JIT_ARENA");
		return value != nullptr && value[0] != '0';
	}();
	return enabled;
}

void* ArenaAllocate(const void* code, size_t size) {
	const size_t aligned = (size + 63u) & ~size_t {63};
	if (aligned > kArenaSize / 4u) {
		return nullptr;
	}
	std::lock_guard lock(g_arena_mutex);
	if (g_arenas.empty() || g_arenas.back().used + aligned > kArenaSize) {
		void* raw = mmap(nullptr, kArenaSize + kHugePage, PROT_READ | PROT_WRITE | PROT_EXEC,
		                 MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
		if (raw == MAP_FAILED) {
			return nullptr;
		}
		auto* base = reinterpret_cast<uint8_t*>(
		    (reinterpret_cast<uintptr_t>(raw) + kHugePage - 1u) & ~uintptr_t {kHugePage - 1u});
		(void)madvise(base, kArenaSize, MADV_HUGEPAGE);
		g_arenas.push_back({base, 0});
	}
	auto& arena  = g_arenas.back();
	auto* memory = arena.base + arena.used;
	std::memcpy(memory, code, size);
	arena.used += aligned;
	return memory;
}

bool InArena(const void* memory) {
	std::lock_guard lock(g_arena_mutex);
	const auto* p = static_cast<const uint8_t*>(memory);
	for (const auto& arena: g_arenas) {
		if (p >= arena.base && p < arena.base + kArenaSize) {
			return true;
		}
	}
	return false;
}
} // namespace
#endif

// Copies `size` bytes of code into fresh memory and makes it executable (read + execute only).
// Returns null on failure.
void* SrtNativeAllocateCode(const void* code, size_t size) {
#if !defined(_WIN32)
	if (UseArena()) {
		if (void* memory = ArenaAllocate(code, size); memory != nullptr) {
			return memory;
		}
	}
#endif
#if defined(_WIN32)
	void* memory = VirtualAlloc(nullptr, size, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
	if (memory == nullptr) {
		return nullptr;
	}
	std::memcpy(memory, code, size);
	DWORD old = 0;
	if (VirtualProtect(memory, size, PAGE_EXECUTE_READ, &old) == 0) {
		VirtualFree(memory, 0, MEM_RELEASE);
		return nullptr;
	}
	FlushInstructionCache(GetCurrentProcess(), memory, size);
	return memory;
#else
	void* memory = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
	if (memory == MAP_FAILED) {
		return nullptr;
	}
	std::memcpy(memory, code, size);
	if (mprotect(memory, size, PROT_READ | PROT_EXEC) != 0) {
		munmap(memory, size);
		return nullptr;
	}
	return memory;
#endif
}

void SrtNativeFreeCode(void* memory, size_t size) {
	if (memory == nullptr) {
		return;
	}
#if defined(_WIN32)
	(void)size;
	VirtualFree(memory, 0, MEM_RELEASE);
#else
	if (UseArena() && InArena(memory)) {
		return;
	}
	munmap(memory, size);
#endif
}

} // namespace Libs::Graphics::ShaderRecompiler::IR
