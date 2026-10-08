#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SRTWALKER_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SRTWALKER_H_

#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/recompiler/ir/passes/SrtNative.h"

#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <unordered_map>
#include <vector>

namespace Libs::Graphics::ShaderRecompiler::IR {

class Value;

using SrtMemoryReader = bool (*)(void* userdata, uint64_t address, std::span<uint32_t> values);

struct SrtRuntime {
	std::span<const uint32_t> user_data;
	uint64_t                  shader_base                = 0;
	SrtMemoryReader           read_memory                = nullptr;
	void*                     userdata                   = nullptr;
	SrtMemoryReader           read_specialization_memory = nullptr;
	// Accept image atomics on k32Float descriptors: upstream's float atomics, and integer atomics
	// run as uint atomics on the raw bits. On by default, as in the emulator; the emulator passes
	// --no-float-image-atomics through here.
	bool                      float_image_atomics        = true;
	// Compute: the dispatch's workgroup count (zero when the host does not know it, as for an
	// indirect dispatch) and workgroup size bound the invocation IDs in buffer write extents.
	std::array<uint32_t, 3>   workgroup_count            = {};
	std::array<uint32_t, 3>   workgroup_size             = {};
	// KYTY_LOCAL_HACK (Senaxx fc56ff3d): optional fast path for raw dword reads. map_clean_page
	// gives the host bytes of a guest page (page-aligned address) with no GPU-written byte, or
	// null; such a page is read directly instead of through read_memory. Valid for one refresh.
	// log_read records each direct read where read_memory would have (the resource memo's read
	// log), capture_ranges as the capture readers do. A walker that cannot read gets no page map.
	const uint8_t* (*map_clean_page)(void* userdata, uint64_t page)              = nullptr;
	void (*log_read)(void* userdata, uint64_t address, uint32_t word)            = nullptr;
	void*                                       page_userdata                     = nullptr;
	std::vector<std::pair<uint64_t, uint64_t>>* capture_ranges                    = nullptr;
};

enum class RuntimeValueType { Any, Integer };

// A ResourcePlan compiled for evaluation (KYTY_SRT_COMPILED): every value its roots reach,
// resolved once (identity chains, invariant phis, ReadConst's SRT read), with arguments as node
// indices. Evaluation stays lazy and uses SrtWalker's opcode semantics.
struct CompiledSrt {
	static constexpr int32_t None = -1;
	enum class Kind : uint8_t { Fail, Constant, Inst };
	struct Node {
		const Inst*            inst     = nullptr;
		uint64_t               value    = 0;
		Kind                   kind     = Kind::Fail;
		uint8_t                num_args = 0;
		std::array<int32_t, 8> args {};
	};
	std::vector<Node>                        nodes;
	std::unordered_map<const Inst*, int32_t> index;
	std::vector<int32_t>                     srt_reads;
	std::vector<std::array<int32_t, 8>>      descriptors;
	std::vector<int32_t>                     conditions;
};

bool ValidateRuntimeValue(const ResourcePlan& program, Value value,
                          RuntimeValueType type = RuntimeValueType::Any);
// Uses the strict reader for values that affect shader specialization.
SrtRuntime CleanRuntime(SrtRuntime runtime);

class SrtWalker;

// KYTY_LOCAL_HACK port of Senaxx/KytyPS5 598030de + 34e492b2 (SRT replay traces; KYTY_SRT_TRACE,
// live, default 1). A resource refresh evaluates one program's values through two walkers (clean
// and ordinary) with memoized, recursive, lazy evaluation (here: native code or compiled nodes);
// the walk was ~37% of the CP thread (pf5, 2026-10-06), most of it in the native code (i-cache).
// A trace records one refresh as a flat list of the instructions it evaluated, in order, with the
// operands each used, and the top-level requests (calls) it served. Later refreshes of the program
// replay the list in a loop: no memo, no recursion, no operand resolution, no per-plan code.
//
// Every data-dependent choice of the recording is a guard: an instruction that succeeded or failed
// must do the same, a select must pick the same arm, a logical AND/OR must take the same
// short-circuit path, and the refresh must request the same values in the same order. A guard
// that does not hold abandons the trace: that request and the rest of the refresh are evaluated
// by the interpreter (the memo is untouched by replay), and the program records again.
// KYTY_SRT_TRACE_VERIFY=1 evaluates every replayed refresh again without the trace and stops on a
// difference; KYTY_SRT_TRACE_STATS=1 reports how refreshes went.
struct SrtTraceOp {
	enum Kind : uint8_t {
		Pure,            // ApplyPureOp over the operands
		UserData,        // user data word `imm`
		ShaderBase,
		Pass,            // the operand (Phi, bit casts, ReadConst, ConditionRef, extract of a pair)
		Extract64,       // a 64-bit operand's dword `flags`
		AddCarryExtract, // IAddCarry32 of two operands, dword `flags`
		Select,          // predicate, then the arm `flags` (1 or 2) the recording took
		LogicalAnd,      // the interpreter's short-circuit rules
		LogicalOr,
		RawRead,         // EvaluateRawRead; `flags` 1 for a constant buffer, `imm` the offset
	};
	Kind        kind   = Pure;
	uint8_t     walker = 0; // 0 the clean walker, 1 the ordinary one
	uint8_t     count  = 0; // operands
	uint8_t     flags  = 0;
	bool        ok     = false; // what the recording got
	ValueOpcode opcode = ValueOpcode::Void;
	int32_t     imm    = 0;
	int32_t     args[5] {}; // SrtTraceSession::Event refs
	const Inst* inst = nullptr;
};

struct SrtTraceCall {
	Value       raw;                 // the requested value as given (matched without resolving)
	const Inst* inst      = nullptr; // null: an immediate
	uint64_t    immediate = 0;
	uint32_t    end       = 0; // ops up to here are evaluated for it
	int32_t     result    = 0; // an Event ref
	uint8_t     walker    = 0;
	bool        ok        = false;
};

struct SrtTrace {
	std::vector<SrtTraceOp>   ops;
	std::vector<uint64_t>     immediates;
	std::vector<SrtTraceCall> calls;
	uint8_t                   key = 0;
};

class SrtTraceSession {
public:
	SrtTraceSession(const ResourcePlan& program, SrtWalker& clean, SrtWalker& walker,
	                const SrtRuntime& runtime);
	~SrtTraceSession();
	SrtTraceSession(const SrtTraceSession&)            = delete;
	SrtTraceSession& operator=(const SrtTraceSession&) = delete;

	// The refresh succeeded: a recording becomes the program's trace.
	void Succeeded() { m_succeeded = true; }
	// The refresh was served from the trace from start to end.
	[[nodiscard]] bool FullyServed() const { return m_mode == Mode::Serve; }

	// Turns tracing off on this thread (verification re-runs).
	static bool& Suppressed();

private:
	friend class SrtWalker;
	enum class Mode : uint8_t { Off, Record, Serve, Passthrough };

	struct Event {
		int32_t  ref   = 0; // >= 0: an op; < 0: immediate -1 - ref; Failed: an operand that failed
		bool     ok    = false;
		uint64_t value = 0; // while recording
	};
	static constexpr int32_t Failed = INT32_MIN;
	struct Frame {
		std::array<Event, 6> events {};
		uint32_t             count  = 0;
		Event                result {Failed, false, 0};
		bool                 have   = false;
	};

	// A request the walkers make: answered inline from the trace when it is the next one and
	// needs no more ops (the common case), else through EvaluateSlow.
	inline bool Evaluate(SrtWalker& walker, Value value, uint64_t& result);
	bool EvaluateSlow(SrtWalker& walker, Value value, uint64_t& result);
	bool Serve(SrtWalker& walker, Value value, uint64_t& result);
	bool RunOps(uint32_t end);
	void Abandon(const char* reason);
	void Abort(const char* reason);
	// Recording hooks, called from SrtWalker::EvaluateWideImpl.
	void OnImmediate(bool ok, uint64_t bits);
	void OnMemoHit(const SrtWalker& walker, uint32_t index, uint64_t value);
	void OnInst(const SrtWalker& walker, const Inst& inst, uint32_t index, bool ok, uint64_t value);
	// Gives the walkers back their native code and compiled form.
	void RestoreEvaluators();

	const ResourcePlan&  m_program;
	SrtWalker*           m_walkers[2];
	const SrtNativeCode* m_native[2] {};
	const CompiledSrt*   m_compiled[2] {};
	Mode                 m_mode      = Mode::Off;
	bool                 m_succeeded = false;
	// Recording.
	std::unique_ptr<SrtTrace> m_recording;
	std::vector<Frame>        m_frames;
	std::vector<int32_t>      m_slots[2]; // evaluation index -> op, per walker
	const char*               m_abort = nullptr;
	// Serving.
	std::shared_ptr<SrtTrace> m_trace;
	uint64_t*                 m_values = nullptr; // thread-local scratch, as large as the trace
	uint8_t*                  m_status = nullptr;
	uint32_t                  m_cursor  = 0;
	uint32_t                  m_next_op = 0;
};

// One memoized evaluation session shared by the entire shader resource refresh.
class SrtWalker {
public:
	SrtWalker(const ResourcePlan& program, const SrtRuntime& runtime,
	          std::span<const uint8_t> clean_flat_slots = {}, SrtWalker* clean_evaluator = nullptr,
	          Value active_mask = {});
	~SrtWalker();
	SrtWalker(const SrtWalker&)            = delete;
	SrtWalker& operator=(const SrtWalker&) = delete;

	bool Evaluate(Value value, uint32_t& result);
	bool EvaluateDescriptor(uint32_t source, DescriptorValue& result);
	// Refreshes reachable scalar reads and active descriptor sources in one walk.
	bool RefreshFlatBuffer(std::vector<uint32_t>& flat);

private:
	friend struct SrtNativeHelpers;
	friend class SrtTraceSession;

	// Binds this walker to the plan's native code when the configuration is one it was compiled
	// for (SrtNative.h), compiling it once the plan is refreshed often enough.
	void BindNative();
	bool VerifyNative(Value value, bool native_ok, uint64_t native_result);
	bool FastFlatRefresh(std::vector<uint32_t>& flat);
	// A value from the native code's tables, evaluated with this walker's frame and mode.
	bool EvaluateNative(const SrtNativeValue& value, uint64_t& result);
	[[nodiscard]] bool UseNativeTables() const;

	static ResourcePlan::EvaluationContext& AcquireContext(const ResourcePlan& program);
	static float Float32(uint64_t bits);
	bool EvaluateWide(Value value, uint64_t& result) {
		if (m_trace != nullptr) [[unlikely]] {
			return m_trace->Evaluate(*this, value, result);
		}
		return EvaluateWideImpl(value, result);
	}
	bool EvaluateWideImpl(Value value, uint64_t& result);
	// KYTY_SRT_COMPILED: evaluates a compiled node (the IR value `value` for the comparison mode).
	bool EvaluateRoot(int32_t node, Value value, uint32_t& result);
	bool EvaluateNode(int32_t node, uint64_t& result);
	bool Arg(const Inst& inst, size_t index, uint64_t& result);
	// Argument `index` of the instruction that is argument `handle` of inst (a resolved handle).
	bool HandleArg(const Inst& inst, size_t handle, size_t index, uint64_t& result);
	// Argument `index` of inst as a compiled node, if inst is the node being evaluated.
	[[nodiscard]] int32_t CurrentNodeArg(const Inst& inst, size_t index) const;
	bool EvaluatePhi(const Inst& inst, uint64_t& result);
	bool EvaluateExtract(const Inst& inst, uint64_t& result);
	bool EvaluateRawRead(const Inst& inst, uint64_t& result);
	// EvaluateRawRead's read once the address is known.
	bool ReadRawWord(uint64_t address, uint64_t base, uint64_t& result);
	bool EvaluateBufferRead(const Inst& inst, uint64_t& result);
	bool EvaluateInst(const Inst& inst, uint64_t& result);

	const ResourcePlan&              m_program;
	SrtRuntime                      m_runtime;
	std::span<const uint8_t>         m_clean_flat_slots;
	SrtWalker*                      m_clean_evaluator = nullptr;
	Value                           m_active_mask;
	const Inst*                     m_failed_value = nullptr;
	ResourcePlan::EvaluationContext& m_context;
	// KYTY_SRT_COMPILED: the plan's compiled form (null: walk the IR), the node being evaluated,
	// and whether every root is evaluated both ways and compared (mode 2).
	const CompiledSrt*       m_compiled = nullptr;
	const CompiledSrt::Node* m_node     = nullptr;
	bool                     m_compare  = false;
	const SrtNativeCode*            m_native      = nullptr;
	SrtNativeMode                   m_native_mode = SrtNativeMode::Self;
	SrtNativeFrame                  m_native_frame;
	SrtTraceSession*                m_trace    = nullptr;
	uint8_t                         m_trace_id = 0;
	// ReadRawWord's fast path: the last page mapped (SrtRuntime::map_clean_page).
	uint64_t       m_mapped_page  = UINT64_MAX;
	const uint8_t* m_mapped_bytes = nullptr;
	// The last raw read that failed, for RefreshFlatBuffer's report.
	const char* m_read_failure         = nullptr;
	uint64_t    m_read_failure_address = 0;
	uint64_t    m_read_failure_offset  = 0;
	uint64_t    m_read_failure_size    = 0;
};

inline bool SrtTraceSession::Evaluate(SrtWalker& walker, Value value, uint64_t& result) {
	if (m_mode == Mode::Serve && m_frames.empty() && m_cursor < m_trace->calls.size()) {
		const auto& call = m_trace->calls[m_cursor];
		if (call.walker == walker.m_trace_id && call.raw.SameInstruction(value) &&
		    (call.end <= m_next_op || RunOps(call.end))) {
			m_cursor++;
			if (call.result == Failed) {
				return false;
			}
			if (call.result < 0) {
				result = m_trace->immediates[static_cast<size_t>(-1 - call.result)];
				return true;
			}
			result = m_values[static_cast<size_t>(call.result)];
			return m_status[static_cast<size_t>(call.result)] != 0u;
		}
	}
	// Not the expected request, or RunOps failed a guard (Serve then runs those ops again, fails
	// the same guard and abandons the trace).
	return EvaluateSlow(walker, value, result);
}

} // namespace Libs::Graphics::ShaderRecompiler::IR

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SRTWALKER_H_ */
