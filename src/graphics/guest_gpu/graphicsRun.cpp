#include "graphics/guest_gpu/graphicsRun.h"

#include "common/assert.h"
#include "common/emulatorConfig.h"
#include "common/liveSwitches.h"
#include "graphics/host_gpu/pipelineStats.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "common/stringUtils.h"
#include "common/threads.h"
#include "graphics/guest_gpu/command_processor/commandProcessor.h"
#include "graphics/guest_gpu/command_processor/pm4Dispatch.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/guest_gpu/pm4.h"
#include "graphics/host_gpu/renderer/commandRecorder.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/renderer/sync.h"
#include "graphics/presentation/videoOut.h"
#include "graphics/presentation/window.h"
#include "graphics/shader/shader.h"
#include "kernel/memory.h"
#include "libs/agc.h"
#include "libs/errno.h"

#include <cstring>
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <bit>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <memory>
#include <mutex>
#include <semaphore>
#include <thread>
#include <vector>

namespace Libs::Graphics {

// KYTY_OCCLUSION_STATS=1 (live): every 5 s, how the guest uses occlusion: counter dumps
// (ZPASS_DONE), SET_PREDICATION by op, and the packets (draws/dispatches among them) that obey
// predication, executed or skipped. Occlusion results are synthetic (always visible), so every
// occlusion-predicated draw runs; this measures how much an implementation could save.
namespace {
struct OcclusionStats {
	std::atomic<uint64_t> dumps {0};
	std::atomic<uint64_t> predication_zpass {0};
	std::atomic<uint64_t> predication_bool {0};
	std::atomic<uint64_t> predication_off {0};
	std::atomic<uint64_t> packets_executed {0};
	std::atomic<uint64_t> draws_executed {0};
	std::atomic<uint64_t> packets_skipped {0};
	std::atomic<uint64_t> draws_skipped {0};
	std::atomic<int64_t>  report_us {0};
};
OcclusionStats g_occlusion_stats;

bool OcclusionStatsEnabled() {
	static auto& enabled = Common::LiveSwitches::Get("KYTY_OCCLUSION_STATS", 0);
	return enabled.load(std::memory_order_relaxed) != 0;
}

bool IsDrawOrDispatchPacket(uint32_t opcode) {
	switch (opcode) {
		case Pm4::IT_DISPATCH_DIRECT:
		case Pm4::IT_DISPATCH_INDIRECT:
		case Pm4::IT_DRAW_INDIRECT:
		case Pm4::IT_DRAW_INDEX_INDIRECT:
		case Pm4::IT_DRAW_INDEX_2:
		case Pm4::IT_DRAW_INDIRECT_MULTI:
		case Pm4::IT_DRAW_INDEX_AUTO:
		case Pm4::IT_DRAW_INDEX_OFFSET_2: return true;
		default: return false;
	}
}

void ReportOcclusionStats() {
	const auto now_us = std::chrono::duration_cast<std::chrono::microseconds>(
	                        std::chrono::steady_clock::now().time_since_epoch())
	                        .count();
	auto last = g_occlusion_stats.report_us.load(std::memory_order_relaxed);
	if (last == 0) {
		g_occlusion_stats.report_us.compare_exchange_strong(last, now_us);
		return;
	}
	if (now_us - last < 5'000'000 ||
	    !g_occlusion_stats.report_us.compare_exchange_strong(last, now_us)) {
		return;
	}
	auto& s = g_occlusion_stats;
	::printf("Occlusion (5 s): dumps %llu, predication zpass %llu / bool %llu / off %llu, "
	         "predicated packets run %llu (draws %llu), skipped %llu (draws %llu)\n",
	         static_cast<unsigned long long>(s.dumps.exchange(0)),
	         static_cast<unsigned long long>(s.predication_zpass.exchange(0)),
	         static_cast<unsigned long long>(s.predication_bool.exchange(0)),
	         static_cast<unsigned long long>(s.predication_off.exchange(0)),
	         static_cast<unsigned long long>(s.packets_executed.exchange(0)),
	         static_cast<unsigned long long>(s.draws_executed.exchange(0)),
	         static_cast<unsigned long long>(s.packets_skipped.exchange(0)),
	         static_cast<unsigned long long>(s.draws_skipped.exchange(0)));
	std::fflush(stdout);
}
} // namespace

static thread_local CommandProcessor* g_current_processor = nullptr;
static thread_local Pm4Execution*     g_current_execution = nullptr;
static thread_local bool              g_gpu_mutex_owned   = false;
static thread_local bool              g_gpu_thread        = false;
static thread_local GuestGpu*         g_gpu_state         = nullptr;

struct DrawIndirectArgs {
	uint32_t vertex_count_per_instance;
	uint32_t instance_count;
	uint32_t start_vertex_location;
	uint32_t start_instance_location;
};

struct DrawIndexedIndirectArgs {
	uint32_t index_count_per_instance;
	uint32_t instance_count;
	uint32_t start_index_location;
	uint32_t base_vertex_location;
	uint32_t start_instance_location;
};

// KYTY_LOCAL_HACK KYTY_SUBMIT_LOCK_STATS=1 (live, research): every 5 s, per caller, the time spent
// waiting for and holding the submission mutex (the guest's submit threads queue on it).
struct SubmitLockStats {
	std::mutex                                          mutex;
	std::unordered_map<const char*, std::array<uint64_t, 4>> by_site; // wait ns, hold ns, calls, max wait ns
	std::chrono::steady_clock::time_point                 last = std::chrono::steady_clock::now();
};
static SubmitLockStats g_submit_lock_stats;
static bool SubmitLockStatsOn() {
	static auto& on = Common::LiveSwitches::Get("KYTY_SUBMIT_LOCK_STATS", 0);
	return on.load(std::memory_order_relaxed) != 0;
}

class GpuMutexLock final {
public:
	explicit GpuMutexLock(Common::Mutex& mutex, const char* site = "?"): m_mutex(mutex), m_site(site) {
		if (g_gpu_mutex_owned) {
			EXIT("recursive GPU mutex acquisition\n");
		}
		g_gpu_mutex_owned = true;
		m_stats = SubmitLockStatsOn();
		const auto start = m_stats ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point {};
		m_mutex.Lock();
		if (m_stats) {
			m_acquired = std::chrono::steady_clock::now();
			m_wait_ns  = static_cast<uint64_t>((m_acquired - start).count());
		}
	}
	~GpuMutexLock() {
		if (!g_gpu_mutex_owned) {
			EXIT("invalid GPU mutex release\n");
		}
		const auto released = m_stats ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point {};
		m_mutex.Unlock();
		g_gpu_mutex_owned = false;
		if (m_stats) {
			std::lock_guard lock(g_submit_lock_stats.mutex);
			auto& e = g_submit_lock_stats.by_site[m_site];
			e[0] += m_wait_ns;
			e[1] += static_cast<uint64_t>((released - m_acquired).count());
			e[2]++;
			e[3] = std::max(e[3], m_wait_ns);
			if (released - g_submit_lock_stats.last >= std::chrono::seconds(5)) {
				g_submit_lock_stats.last = released;
				::printf("Submit lock (5 s):");
				for (auto& [site, v]: g_submit_lock_stats.by_site) {
					::printf(" %s n=%" PRIu64 " wait %.1f ms (max %.1f) hold %.1f ms;", site, v[2],
					         static_cast<double>(v[0]) / 1e6, static_cast<double>(v[3]) / 1e6,
					         static_cast<double>(v[1]) / 1e6);
					v = {};
				}
				::printf("\n");
				std::fflush(stdout);
			}
		}
	}

private:
	Common::Mutex&                         m_mutex;
	const char*                            m_site;
	bool                                   m_stats   = false;
	uint64_t                               m_wait_ns = 0;
	std::chrono::steady_clock::time_point  m_acquired {};
};

static bool GraphicsRunDebugDumpEnabled() {
	return Config::GraphicsDebugDumpEnabled() &&
	       Config::GetPrintfDirection() != Config::LogDirection::Silent;
}

// Research: KYTY_LABELS_AFTER_GPU=1 publishes EOP labels, EOP timestamps and flips when the
// preceding GPU work has executed instead of when it was recorded. A label written at record time
// tells the guest its per-frame memory is free while the GPU has not read it yet; the next frame's
// data then mixes into this one. The scheduler's completion thread writes it, so nothing waits.
// On by default in this build (PR #937 runs Wolverine with it; the launcher sets no environment);
// KYTY_LABELS_AFTER_GPU=0 turns it off.
static bool LabelsAfterGpu() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_LABELS_AFTER_GPU");
		return value == nullptr || std::strcmp(value, "0") != 0;
	}();
	return enabled;
}

GuestGpu::GuestGpu(RenderContext& renderer): m_renderer(renderer) {
	EXIT_NOT_IMPLEMENTED(!Common::Thread::IsMainThread());
	GraphicsInitJmpTables();
	m_gfx_cp = std::make_unique<CommandProcessor>(renderer, 0);
	m_thread = std::jthread(ThreadRun, this);
}

GuestGpu::~GuestGpu() {
	Shutdown();
}

void GuestGpu::Shutdown() {
	std::lock_guard shutdown_lock(m_shutdown_mutex);
	if (m_shutdown_complete) {
		return;
	}
	{
		Common::LockGuard lock(m_queue_mutex);
		m_accepting = false;
		m_stopping  = true;
		m_work_available.SignalAll();
		m_suspend_point_done.SignalAll();
	}
	if (m_thread.joinable()) {
		m_thread.join();
	}
	m_shutdown_complete = true;
}

bool GuestGpu::IsStopping() {
	Common::LockGuard lock(m_queue_mutex);
	return m_stopping;
}

void GuestGpu::SendCommand(Common::UniqueFunction<void>&& command) {
	EXIT_IF(!command);
	if (IsGpuThread()) {
		command();
		return;
	}
	Common::LockGuard lock(m_queue_mutex);
	EXIT_IF(!m_accepting);
	m_commands.push_back(std::move(command));
	m_pending_commands.fetch_add(1, std::memory_order_release);
	m_work_available.Signal();
}

bool GuestGpu::TrySendCommand(Common::UniqueFunction<void>&& command) {
	EXIT_IF(!command);
	if (IsGpuThread()) {
		command();
		return true;
	}
	Common::LockGuard lock(m_queue_mutex);
	if (!m_accepting) {
		return false;
	}
	m_commands.push_back(std::move(command));
	m_pending_commands.fetch_add(1, std::memory_order_release);
	m_work_available.Signal();
	return true;
}

namespace {
std::atomic<bool> g_labels_deferred {false};
} // namespace

bool GuestGpu::LabelsDeferred() noexcept {
	return g_labels_deferred.load(std::memory_order_relaxed);
}

bool GuestGpu::LabelsAtCompletion() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_LABELS_AT_COMPLETION");
		return value != nullptr && value[0] == '1';
	}();
	return enabled;
}

void GuestGpu::DeferLabelWrite(uint64_t address, uint64_t value, uint32_t size) {
	EXIT_IF(!IsGpuThread() || (size != 4 && size != 8));
	g_labels_deferred.store(true, std::memory_order_relaxed);
	const auto sequence        = ++m_label_sequence;
	m_pending_labels[address] = {value, size, sequence};
	// The completion runs on the scheduler's priority thread, which must not fault on guest
	// memory the GPU wrote to; the write itself goes back to the GPU thread.
	m_renderer.GetCommandScheduler().DeferPriorityOperation([this, address, value, size,
	                                                         sequence] {
		(void)TrySendCommand([this, address, value, size, sequence] {
			std::memcpy(reinterpret_cast<void*>(address), &value, size);
			const auto pending = m_pending_labels.find(address);
			if (pending != m_pending_labels.end() && pending->second.sequence == sequence) {
				m_pending_labels.erase(pending);
			}
		});
	});
}

template <typename T>
T GuestGpu::ReadLabel(const volatile T* address) const {
	if (!m_pending_labels.empty()) {
		const auto pending = m_pending_labels.find(reinterpret_cast<uint64_t>(address));
		if (pending != m_pending_labels.end() && pending->second.size >= sizeof(T)) {
			return static_cast<T>(pending->second.value);
		}
	}
	return *address;
}

template uint32_t GuestGpu::ReadLabel<uint32_t>(const volatile uint32_t*) const;
template uint64_t GuestGpu::ReadLabel<uint64_t>(const volatile uint64_t*) const;

void GuestGpu::NoteRecordedLabel(uint64_t address, uint64_t value, uint32_t size, uint64_t tick) {
	if (!IsGpuThread()) {
		return; // only forwarded waits need it; they run on the GPU thread
	}
	m_recorded_labels[address] = {value, size, tick};
}

template <typename T>
bool GuestGpu::FindRecordedLabel(const volatile T* address, T* value) {
	EXIT_IF(!IsGpuThread());
	const auto recorded = m_recorded_labels.find(reinterpret_cast<uint64_t>(address));
	if (recorded == m_recorded_labels.end()) {
		return false;
	}
	if (recorded->second.size < sizeof(T) ||
	    m_renderer.GetCommandScheduler().IsFree(recorded->second.tick)) {
		// Published (or about to be): guest memory is the truth from here on.
		m_recorded_labels.erase(recorded);
		return false;
	}
	*value = static_cast<T>(recorded->second.value);
	return true;
}

template bool GuestGpu::FindRecordedLabel<uint32_t>(const volatile uint32_t*, uint32_t*);
template bool GuestGpu::FindRecordedLabel<uint64_t>(const volatile uint64_t*, uint64_t*);

void GuestGpu::ProcessCommands() {
	EXIT_IF(!IsGpuThread());
	while (m_pending_commands.load(std::memory_order_acquire) != 0) {
		Common::UniqueFunction<void> command;
		{
			Common::LockGuard lock(m_queue_mutex);
			EXIT_IF(m_commands.empty());
			command = std::move(m_commands.front());
			m_commands.pop_front();
			EXIT_IF(m_pending_commands.fetch_sub(1, std::memory_order_acq_rel) == 0);
		}
		KYTY_PROFILER_BLOCK("GuestGpu::ProcessCommands(command)");
		Lookahead::g_commands++;
		command();
	}
}

void GuestGpu::SendCommandSync(Common::UniqueFunction<void>&& command) {
	EXIT_IF(!command);
	if (IsGpuThread()) {
		command();
		return;
	}
	std::binary_semaphore done {0};
	SendCommand([operation = std::move(command), &done]() mutable {
		operation();
		done.release();
	});
	done.acquire();
}

void GuestGpu::Submit(std::span<const uint32_t> draw_commands,
                      std::span<const uint32_t> constant_commands) {
	if (draw_commands.empty()) {
		return;
	}
	GpuMutexLock lock(m_submission_mutex, "Submit");
	Submission   submission;
	submission.type              = SubmissionType::Graphics;
	submission.queue_id          = 0;
	submission.commands          = draw_commands;
	submission.constant_commands = constant_commands;
	submission.reset_processor   = m_graphics_done;
	m_graphics_done              = false;
	Enqueue(std::move(submission));
}

void GuestGpu::SubmitCompute(uint32_t queue, std::span<const uint32_t> commands) {
	EXIT_IF(commands.empty());
	GpuMutexLock lock(m_submission_mutex, "SubmitCompute");

	EXIT_NOT_IMPLEMENTED(queue < ComputeQueueBase || queue >= ComputeQueueBase + ComputeQueueCount);

	const auto compute_queue = queue - ComputeQueueBase;
	Submission submission;
	submission.type     = SubmissionType::Compute;
	submission.queue_id = 1 + compute_queue;
	submission.commands = commands;
	Enqueue(std::move(submission));
}

void GuestGpu::SubmitFlipPreparation(uint64_t request_id) {
	GpuMutexLock lock(m_submission_mutex, "SubmitFlipPreparation");
	Submission   submission;
	submission.type            = SubmissionType::FlipPreparation;
	submission.queue_id        = 0;
	submission.reset_processor = m_graphics_done;
	submission.flip_request_id = request_id;
	m_graphics_done            = false;
	Enqueue(std::move(submission));
}

void GuestGpu::Done() {
	GpuMutexLock lock(m_submission_mutex, "Done");
	if (!IsGpuThread()) {
		WaitForIdle();
	}
	m_graphics_done = true;
	m_done_num++;
}

// Research: sceAgcSuspendPoint inserts a drain of the graphics pipe into the graphics queue and
// blocks the caller only while the previous suspend point has not executed on the GPU; async
// compute keeps running (agc/baselayer.h, suspendPoint()). Done() waited for every queue to go
// idle while holding the submission mutex: the submit thread could not overlap the GPU, and a
// compute queue parked on a CPU write would never let it return. KYTY_SUSPEND_POINT_WAITS_IDLE=1
// restores it.
void GuestGpu::SuspendPoint() {
	static const bool waits_idle = std::getenv("KYTY_SUSPEND_POINT_WAITS_IDLE") != nullptr;
	if (waits_idle || IsGpuThread()) {
		Done();
		return;
	}
	uint64_t   gpu_tick       = 0;
	const bool prewait_stats  = SubmitLockStatsOn();
	const auto prewait_start  = prewait_stats ? std::chrono::steady_clock::now()
	                                          : std::chrono::steady_clock::time_point {};
	std::chrono::steady_clock::time_point prewait_mid {};
	{
		Common::LockGuard lock(m_queue_mutex);
		while (m_suspend_points_done < m_suspend_points_issued && !m_stopping) {
			m_suspend_point_done.Wait(&m_queue_mutex);
		}
		if (m_stopping) {
			return;
		}
		gpu_tick = m_suspend_point_gpu_tick;
	}
	// The previous drain must also have executed on the GPU, as on the console. Without this the
	// guest runs arbitrarily far ahead, and the uploads recorded for it pile up until an
	// allocation fails. The tick is already submitted, so this never waits on guest work.
	if (prewait_stats) prewait_mid = std::chrono::steady_clock::now();
	if (gpu_tick != 0) {
		m_renderer.GetCommandScheduler().GetMasterSemaphore().Wait(gpu_tick);
	}
	if (prewait_stats) {
		// Research: the suspend point's waits (previous drain processed by the CP / done on the GPU).
		const auto end = std::chrono::steady_clock::now();
		std::lock_guard stats_lock(g_submit_lock_stats.mutex);
		auto& cp_wait  = g_submit_lock_stats.by_site["SuspendPoint-waitCP"];
		auto& gpu_wait = g_submit_lock_stats.by_site["SuspendPoint-waitGPU"];
		cp_wait[0] += static_cast<uint64_t>((prewait_mid - prewait_start).count());
		cp_wait[2]++;
		cp_wait[3] = std::max<uint64_t>(cp_wait[3], static_cast<uint64_t>((prewait_mid - prewait_start).count()));
		gpu_wait[0] += static_cast<uint64_t>((end - prewait_mid).count());
		gpu_wait[2]++;
		gpu_wait[3] = std::max<uint64_t>(gpu_wait[3], static_cast<uint64_t>((end - prewait_mid).count()));
	}
	GpuMutexLock lock(m_submission_mutex, "SuspendPoint");
	Submission   submission;
	submission.type            = SubmissionType::SuspendPoint;
	submission.queue_id        = 0;
	submission.reset_processor = m_graphics_done;
	{
		Common::LockGuard queue_lock(m_queue_mutex);
		m_suspend_points_issued++;
	}
	Enqueue(std::move(submission));
	// The graphics state is reset after the drain: the next graphics submission resets it.
	m_graphics_done = true;
	m_done_num++;
}

int GuestGpu::GetFrameNum() const {
	return m_done_num;
}

CommandProcessor& GuestGpu::GetProcessor(uint32_t queue_id) {
	EXIT_IF(queue_id >= QueueCount);
	if (queue_id == 0) {
		return *m_gfx_cp;
	}
	auto& processor = m_compute_cp[queue_id - 1];
	if (processor == nullptr) {
		processor = std::make_unique<CommandProcessor>(m_renderer, ComputeQueueBase + queue_id - 1);
	}
	return *processor;
}

void CommandProcessor::Reset() {
	m_sh_ctx.Reset();
	m_ucfg.Reset();
	m_ctx.Reset();
	m_saved_ctx.Reset();
	m_context_state_pushed             = false;
	m_index_type_and_size              = 0;
	m_index_buffer_size                = 0;
	m_user_data_marker                 = HW::UserSgprType::Unknown;
	m_draw_indirect_args_base_addr     = 0;
	m_dispatch_indirect_args_base_addr = 0;

	std::memset(m_const_ram, 0, sizeof(m_const_ram));
}

void CommandProcessor::ApplyContextStateOperation(ContextStateOperation operation) {
	switch (operation) {
		case ContextStateOperation::Clear: m_ctx.Reset(); break;
		case ContextStateOperation::Push:
			EXIT_IF(m_context_state_pushed);
			m_saved_ctx            = m_ctx;
			m_context_state_pushed = true;
			break;
		case ContextStateOperation::Pop:
			EXIT_IF(!m_context_state_pushed);
			m_ctx                  = m_saved_ctx;
			m_saved_ctx            = {};
			m_context_state_pushed = false;
			break;
		case ContextStateOperation::PushClear:
			EXIT_IF(m_context_state_pushed);
			m_saved_ctx            = m_ctx;
			m_context_state_pushed = true;
			m_ctx.Reset();
			break;
		default: EXIT("unknown context state operation: %u\n", static_cast<uint32_t>(operation));
	}
}

void CommandProcessor::BufferInit() {
	KYTY_PROFILER_FUNCTION();
	GetScheduler().Begin(m_ctx, m_ucfg, m_sh_ctx);
}

void CommandProcessor::BufferFlush() {
	KYTY_PROFILER_FUNCTION();
	GetScheduler().Flush();
}

void CommandProcessor::BufferFlushAndWait() {
	GetScheduler().FlushAndWait();
}

void CommandProcessor::BufferWait() {
	BufferInit();
	GetScheduler().Finish();
}

void CommandProcessor::ResetDeCe() {
	m_de_count    = 0;
	m_ce_count    = 0;
	m_ce_complete = false;
}

void CommandProcessor::WaitCe() {
	if (m_ce_count <= m_de_count && !m_ce_complete) {
		KYTY_PROFILER_BLOCK("Pm4Suspend::WaitCe");
		SuspendPm4();
	}
}

void CommandProcessor::WaitDeDiff(uint32_t diff) {
	EXIT_IF(m_de_count > m_ce_count);
	if (m_ce_count - m_de_count >= diff) {
		KYTY_PROFILER_BLOCK("Pm4Suspend::WaitDeDiff");
		SuspendPm4();
	}
}

void CommandProcessor::WaitForRewind(bool valid) {
	if (!valid) {
		KYTY_PROFILER_BLOCK("Pm4Suspend::WaitForRewind");
		SuspendPm4();
	}
}

void CommandProcessor::IncrementDe() {
	m_de_count++;
}

void CommandProcessor::IncrementCe() {
	m_ce_count++;
}

void CommandProcessor::WriteConstRam(uint32_t offset, const uint32_t* src, uint32_t dw_num) {
	memcpy(m_const_ram + offset / 4, src, static_cast<size_t>(dw_num) * 4);
}

void CommandProcessor::DumpConstRam(uint32_t* dst, uint32_t offset, uint32_t dw_num) {
	Lookahead::g_cp_writes++;
	memcpy(dst, m_const_ram + offset / 4, static_cast<size_t>(dw_num) * 4);
}

bool TestWaitRegMemValue(uint64_t value, uint64_t ref, uint64_t mask, uint32_t func) {
	switch (func) {
		case 0: return true;
		case 1: return (value & mask) < ref;
		case 2: return (value & mask) <= ref;
		case 3: return (value & mask) == ref;
		case 4: return (value & mask) != ref;
		case 5: return (value & mask) >= ref;
		case 6: return (value & mask) > ref;
		default: EXIT("unknown wait compare function: %" PRIu32 "\n", func);
	}

	return false;
}

template <typename T>
T CommandProcessor::ReadLabel(const volatile T* addr) const {
	return GuestGpu::LabelsDeferred() ? m_renderer.GetGpu().ReadLabel(addr) : *addr;
}

template uint32_t CommandProcessor::ReadLabel<uint32_t>(const volatile uint32_t*) const;
template uint64_t CommandProcessor::ReadLabel<uint64_t>(const volatile uint64_t*) const;

template <typename T>
void CommandProcessor::WaitRegMem(uint32_t func, const T* addr, T ref, T mask, uint32_t poll,
                                  uint32_t wait_op) {
	EXIT_IF(addr == nullptr);
	if ((wait_op & ~1u) != 0) {
		EXIT("unsupported wait_reg_mem operation: 0x%08" PRIx32 "\n", wait_op);
	}

	(void)poll;
	if (TestWaitRegMemValue(ReadLabel(addr), ref, mask, func)) {
		return;
	}
	// Research: KYTY_LABEL_WAIT_FORWARD=1. With KYTY_LABELS_AFTER_GPU a label this GPU recorded
	// stays invisible until its tick completes, so a wait on it parked Thread_Gpu until the host
	// GPU caught up (Wolverine: ~28% of Thread_Gpu idle with every queue blocked). Everything
	// recorded before the label is ahead in the same Vulkan queue, so a full barrier gives the
	// commands after the wait the ordering the wait promised, and recording can go on.
	// Wolverine, standing still, same process: 0 → 283 ms/frame, 1 → 217 ms; claws and scene
	// unchanged.
	static auto&                 forward = Common::LiveSwitches::Get("KYTY_LABEL_WAIT_FORWARD", 1);
	static std::atomic<uint64_t> suspended {0};
	static std::atomic<uint64_t> forwarded {0};
	static std::atomic<uint64_t> forwardable {0};
	static auto                  report_time = std::chrono::steady_clock::now();
	T                            recorded    = 0;
	const bool can_forward = m_renderer.GetGpu().FindRecordedLabel(addr, &recorded) &&
	                         TestWaitRegMemValue(recorded, ref, mask, func);
	if (can_forward && forward.load(std::memory_order_relaxed) != 0) {
		forwarded++;
		EmitGlobalBarrier();
	} else {
		(can_forward ? forwardable : suspended)++;
		KYTY_PROFILER_BLOCK("Pm4Suspend::WaitRegMem");
		SuspendPm4();
	}
	const auto now = std::chrono::steady_clock::now();
	if (now - report_time >= std::chrono::seconds(5)) {
		report_time = now;
		::printf("Label waits (5 s): suspended %" PRIu64 ", forwardable %" PRIu64
		         ", forwarded %" PRIu64 "\n",
		         suspended.exchange(0), forwardable.exchange(0), forwarded.exchange(0));
		std::fflush(stdout);
	}
}

template void CommandProcessor::WaitRegMem<uint32_t>(uint32_t, const uint32_t*, uint32_t, uint32_t,
                                                     uint32_t, uint32_t);
template void CommandProcessor::WaitRegMem<uint64_t>(uint32_t, const uint64_t*, uint64_t, uint64_t,
                                                     uint32_t, uint32_t);

void CommandProcessor::WriteData(uint32_t* dst, const uint32_t* src, uint32_t dw_num,
                                 uint32_t write_control) {
	Lookahead::g_cp_writes++;
	const uint32_t dst_sel = ((write_control >> 30u) & 0x1u) | ((write_control >> 7u) & 0x1eu);
	const bool     write_one_address = ((write_control >> 16u) & 0x1u) != 0;

	switch (dst_sel) {
		case 0:
		case 2:
		case 4:
		case 5:
		case 6: break;
		default: EXIT("unsupported writeData destination selector 0x%02" PRIx32 "\n", dst_sel);
	}
	if (dw_num == 0) {
		return;
	}

	// Labels often share a page with bytes the GPU wrote. A plain write then faults and downloads
	// the page, draining the GPU; WriteClean writes both copies instead.
	auto& cache = m_renderer.GetBufferCache();
	if (write_one_address) {
		if (cache.WriteClean(reinterpret_cast<uint64_t>(dst), &src[dw_num - 1], sizeof(uint32_t))) {
			return;
		}
		for (uint32_t i = 0; i < dw_num; i++) {
			dst[0] = src[i];
		}
	} else {
		const auto bytes = static_cast<size_t>(dw_num) * sizeof(uint32_t);
		if (cache.WriteClean(reinterpret_cast<uint64_t>(dst), src, bytes)) {
			return;
		}
		memcpy(dst, src, bytes);
	}
}

void CommandProcessor::WriteReferenceClock(uint64_t dst_address, uint32_t num_bytes) {
	if (dst_address == 0 || (num_bytes != sizeof(uint32_t) && num_bytes != sizeof(uint64_t)) ||
	    (dst_address & (num_bytes - 1u)) != 0) {
		EXIT("invalid reference-clock copy, dst=0x%016" PRIx64 " size=%u\n", dst_address,
		     num_bytes);
	}
	const auto value = Sync::ReadReferenceClock();
	std::memcpy(reinterpret_cast<void*>(dst_address), &value, num_bytes);
	static std::atomic<uint32_t> clock_log_count {0};
	if (clock_log_count.fetch_add(1) < 64) {
		LOGF("\t copy_data reference clock: dst=0x%016" PRIx64 " value=0x%016" PRIx64
		     " size=%u\n",
		     dst_address, value, num_bytes);
	}
}

void CommandProcessor::DmaData(uint8_t engine, uint8_t dst_sel, uint8_t dst_cache_policy,
                               uint64_t dst_address_or_offset, uint8_t src_sel,
                               uint8_t  src_cache_policy,
                               uint64_t src_address_or_offset_or_immediate, uint32_t num_bytes,
                               uint8_t wait_for_previous, uint8_t write_confirm,
                               uint8_t block_engine) {
	Lookahead::g_cp_writes++;
	EXIT_NOT_IMPLEMENTED(engine > 1);
	if (num_bytes == 0) {
		return;
	}
	EXIT_NOT_IMPLEMENTED(dst_cache_policy > 3);
	EXIT_NOT_IMPLEMENTED(src_cache_policy > 3);
	EXIT_NOT_IMPLEMENTED(wait_for_previous > 1);
	EXIT_NOT_IMPLEMENTED(write_confirm > 1);
	EXIT_NOT_IMPLEMENTED(block_engine > 1);
	if (static_cast<uint32_t>(dst_address_or_offset) == 0x3022cu) {
		return;
	}
	auto decode_gds = [](uint8_t selector, bool& is_gds) {
		switch (selector) {
			case 0:
			case 3: is_gds = false; return true;
			case 1: is_gds = true; return true;
			default: return false;
		}
	};
	if (dst_sel == 2) {
		// kNowhere discards the GL2 prefetch destination without a guest-visible write.
		if (src_sel != 3) {
			EXIT("unsupported dmaData nowhere source selector 0x%02" PRIx8 "\n", src_sel);
		}
		return;
	}
	bool dst_gds = false;
	if (!decode_gds(dst_sel, dst_gds)) {
		EXIT("unsupported dmaData destination selector 0x%02" PRIx8 "\n", dst_sel);
	}
	auto& buffer_cache = m_renderer.GetBufferCache();
	if (src_sel == 2) {
		buffer_cache.FillBuffer(
		    dst_address_or_offset, num_bytes,
		    static_cast<uint32_t>(src_address_or_offset_or_immediate & 0xffffffffu), dst_gds);
		return;
	}
	bool src_gds = false;
	if (!decode_gds(src_sel, src_gds)) {
		EXIT("unsupported dmaData source selector 0x%02" PRIx8 "\n", src_sel);
	}
	if (src_gds && dst_gds) {
		EXIT("unsupported dmaData GDS-to-GDS copy\n");
	}
	buffer_cache.CopyBuffer(dst_address_or_offset, src_address_or_offset_or_immediate, num_bytes,
	                        dst_gds, src_gds);
}

void GuestGpu::Enqueue(Submission submission) {
	EXIT_IF(submission.queue_id >= QueueCount);
	Common::LockGuard lock(m_queue_mutex);
	EXIT_IF(!m_accepting);
	m_queues[submission.queue_id].push_back(std::move(submission));
	m_submission_count++;
	m_work_available.Signal();
}

void GuestGpu::Wake() {
	m_work_available.Signal();
}

void GuestGpu::WaitForIdle() {
	Common::LockGuard lock(m_queue_mutex);
	while (m_processing || !m_commands.empty() || m_submission_count != 0) {
		m_idle.Wait(&m_queue_mutex);
	}
}

// KYTY_LOCAL_HACK research, KYTY_CP_SPLIT_STATS=1 (live, default 0): every 5 s, the GPU thread's time
// in graphics-queue submissions, compute-queue submissions and forwarded commands, to size how much
// work a second processing thread could take.
namespace {
struct CpSplitStats {
	uint64_t ns[3] {};
	uint64_t calls[3] {};
	uint64_t compute_queues = 0; // bitmask of compute queues seen
	std::chrono::steady_clock::time_point last = std::chrono::steady_clock::now();
};
CpSplitStats g_cp_split;
bool CpSplitOn() {
	static auto& on = Common::LiveSwitches::Get("KYTY_CP_SPLIT_STATS", 0);
	return on.load(std::memory_order_relaxed) != 0;
}
void NoteCpSplit(int kind, std::chrono::steady_clock::time_point start, uint32_t queue_id) {
	const auto now = std::chrono::steady_clock::now();
	g_cp_split.ns[kind] += static_cast<uint64_t>(
	    std::chrono::duration_cast<std::chrono::nanoseconds>(now - start).count());
	g_cp_split.calls[kind]++;
	if (kind == 1) {
		g_cp_split.compute_queues |= uint64_t {1} << (queue_id % 64u);
	}
	if (now - g_cp_split.last >= std::chrono::seconds(5)) {
		::printf("CP split (5 s): graphics %.1f ms/%" PRIu64 ", compute %.1f ms/%" PRIu64
		         " (%d queues), commands %.1f ms/%" PRIu64 "\n",
		         static_cast<double>(g_cp_split.ns[0]) / 1e6, g_cp_split.calls[0],
		         static_cast<double>(g_cp_split.ns[1]) / 1e6, g_cp_split.calls[1],
		         std::popcount(g_cp_split.compute_queues),
		         static_cast<double>(g_cp_split.ns[2]) / 1e6, g_cp_split.calls[2]);
		g_cp_split      = {};
		g_cp_split.last = now;
	}
}
} // namespace

// KYTY_LOCAL_HACK KYTY_CP_IDLE_STATS=1 (live, research): every 5 s, the time the CP thread slept
// waiting for submissions (no work) and with every queue blocked on a label wait.
static bool CpIdleStatsOn() {
	static auto& on = Common::LiveSwitches::Get("KYTY_CP_IDLE_STATS", 0);
	return on.load(std::memory_order_relaxed) != 0;
}

static void NoteCpIdle(int kind, std::chrono::steady_clock::time_point start) {
	if (start == std::chrono::steady_clock::time_point {}) {
		return;
	}
	static uint64_t ns[2]    = {};
	static uint64_t count[2] = {};
	static auto     report   = std::chrono::steady_clock::now();
	const auto      now      = std::chrono::steady_clock::now();
	ns[kind] += static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(now - start).count());
	count[kind]++;
	if (now - report >= std::chrono::seconds(5)) {
		report = now;
		::printf("CP idle (5 s): no work %.1f ms (%" PRIu64 "), blocked queues %.1f ms (%" PRIu64 ")\n",
		         static_cast<double>(ns[0]) / 1e6, count[0], static_cast<double>(ns[1]) / 1e6, count[1]);
		std::fflush(stdout);
		ns[0] = ns[1] = count[0] = count[1] = 0;
	}
}

void GuestGpu::ThreadRun(void* data) {
	auto* gpu = static_cast<GuestGpu*>(data);
	EXIT_IF(gpu == nullptr);
	KYTY_PROFILER_THREAD("Thread_Gpu");
	g_gpu_thread = true;
	g_gpu_state  = gpu;

	for (;;) {
		Submission                   submission;
		Common::UniqueFunction<void> command;
		bool                         has_submission = false;
		bool                         should_stop    = false;
		// Commands a gated slice left unsubmitted go to the GPU before this thread waits.
		bool flush_pending = false;
		Common::Thread::ApplyGpuThreadPriority();
		{
			Common::LockGuard lock(gpu->m_queue_mutex);
			while (gpu->m_commands.empty() && gpu->m_submission_count == 0 && !gpu->m_stopping) {
				if (gpu->m_slice_flush_pending) {
					flush_pending = true;
					break;
				}
				KYTY_PROFILER_BLOCK("GuestGpu::WaitForWork");
				gpu->m_processing = false;
				gpu->m_idle.Signal();
				const auto idle_start = CpIdleStatsOn() ? std::chrono::steady_clock::now()
				                                        : std::chrono::steady_clock::time_point {};
				gpu->m_work_available.Wait(&gpu->m_queue_mutex);
				NoteCpIdle(0, idle_start);
			}
			if (flush_pending) {
				gpu->m_processing = true;
			} else if (gpu->m_stopping && gpu->m_commands.empty() && gpu->m_submission_count == 0) {
				gpu->m_processing = false;
				gpu->m_idle.SignalAll();
				gpu->m_suspend_point_done.SignalAll();
				should_stop = true;
			} else if (!gpu->m_commands.empty()) {
				command = std::move(gpu->m_commands.front());
				gpu->m_commands.pop_front();
				EXIT_IF(gpu->m_pending_commands.fetch_sub(1, std::memory_order_acq_rel) == 0);
				gpu->m_processing = true;
			} else {
				int selected_queue = -1;
				for (uint32_t offset = 0; offset < QueueCount; offset++) {
					const auto id = (gpu->m_next_queue + offset) % QueueCount;
					if (!gpu->m_queues[id].empty() && !gpu->m_queues[id].front().blocked) {
						selected_queue = static_cast<int>(id);
						break;
					}
				}
				if (selected_queue < 0 && gpu->m_slice_flush_pending) {
					flush_pending     = true;
					gpu->m_processing = true;
				} else if (selected_queue < 0) {
					gpu->m_processing = false;
					{
						KYTY_PROFILER_BLOCK("GuestGpu::WaitBlockedQueues");
						const auto idle_start = CpIdleStatsOn()
						                            ? std::chrono::steady_clock::now()
						                            : std::chrono::steady_clock::time_point {};
						gpu->m_work_available.WaitFor(&gpu->m_queue_mutex, 100);
						NoteCpIdle(1, idle_start);
					}
					for (auto& queue: gpu->m_queues) {
						if (!queue.empty()) {
							queue.front().blocked = false;
						}
					}
					continue;
				} else {
					auto& queue = gpu->m_queues[static_cast<uint32_t>(selected_queue)];
					submission  = std::move(queue.front());
					queue.pop_front();
					gpu->m_submission_count--;
					gpu->m_next_queue = (static_cast<uint32_t>(selected_queue) + 1) % QueueCount;
					gpu->m_processing = true;
					has_submission    = true;
				}
			}
		}
		if (flush_pending) {
			gpu->FlushPendingSlices();
			continue;
		}
		if (should_stop) {
			gpu->m_gfx_cp->BufferWait();
			g_gpu_state  = nullptr;
			g_gpu_thread = false;
			return;
		}

		if (command) {
			EXIT_IF(g_current_processor != nullptr);
			{
				KYTY_PROFILER_BLOCK("GuestGpu::RunCommand");
				const auto start = std::chrono::steady_clock::now();
				command();
				if (CpSplitOn()) {
					NoteCpSplit(2, start, 0);
				}
			}

			Common::LockGuard lock(gpu->m_queue_mutex);
			gpu->m_processing = false;
			if (gpu->m_commands.empty() && gpu->m_submission_count == 0) {
				gpu->m_idle.SignalAll();
			}
			continue;
		}

		EXIT_IF(!has_submission);
		const auto process_start = std::chrono::steady_clock::now();
		const auto queue_id      = submission.queue_id;
		const bool complete      = gpu->Process(submission);
		if (CpSplitOn()) {
			NoteCpSplit(queue_id == 0 ? 0 : 1, process_start, queue_id);
		}

		Common::LockGuard lock(gpu->m_queue_mutex);
		if (!complete) {
			submission.blocked = true;
			gpu->m_queues[submission.queue_id].push_front(std::move(submission));
			gpu->m_submission_count++;
		} else {
			for (auto& queue: gpu->m_queues) {
				if (!queue.empty()) {
					queue.front().blocked = false;
				}
			}
		}
		gpu->m_processing = false;
		if (gpu->m_commands.empty() && gpu->m_submission_count == 0) {
			gpu->m_idle.SignalAll();
		}
	}
}

// Research: KYTY_SLICE_FLUSH_US=N (live, 0 = off). Every processed slice of a guest submission
// ended with a submit: in Wolverine ~800 per second, most of them small async-compute
// submissions, each a ~50 us kernel call on Thread_Gpu. With label waits forwarded and labels
// batched, a completed slice no longer has to submit at once: it does when N microseconds have
// passed since the last submit, and otherwise its commands go with the next one. A slice that
// blocks always submits, and so does the GPU thread before it waits for work or for blocked
// queues (FlushPendingSlices), so nothing waits on commands that were never submitted.
void GuestGpu::FlushSlice(CommandProcessor& cp, bool complete) {
	// Wolverine run 19, same process: submits 8.8k -> 4.8k per 5 s; frames 138 vs 135-147 / 15 s
	// (frame time unchanged, vblank-bound); fewer, larger command buffers for the GPU.
	static auto& interval_us = Common::LiveSwitches::Get("KYTY_SLICE_FLUSH_US", 500);
	const auto   interval    = interval_us.load(std::memory_order_relaxed);
	if (complete && interval > 0) {
		const auto now_us = std::chrono::duration_cast<std::chrono::microseconds>(
		                        std::chrono::steady_clock::now().time_since_epoch())
		                        .count();
		if (now_us - m_renderer.GetCommandScheduler().LastSubmitUs() < interval) {
			m_slice_flush_pending = true;
			return;
		}
	}
	m_slice_flush_pending = false;
	cp.BufferFlush();
}

void GuestGpu::FlushPendingSlices() {
	EXIT_IF(!IsGpuThread());
	if (m_slice_flush_pending) {
		m_slice_flush_pending = false;
		m_gfx_cp->BufferFlush();
	}
}

static uint32_t g_stats_queue = 0; // KYTY_PIPELINE_STATS: queue of the submission being processed

bool GuestGpu::Process(Submission& submission) {
	const bool first_slice = !submission.started;
	g_stats_queue          = submission.queue_id;
	auto& cp = GetProcessor(submission.queue_id);

	if (first_slice) {
		g_guest_submission_seq.fetch_add(1, std::memory_order_relaxed);
	}
	if (first_slice && submission.reset_processor) {
		cp.Reset();
	}

	if (first_slice) {
		submission.started = true;
		cp.SetSubmitId(++m_submit_id);
		cp.ResetDeCe();
		cp.SetFlip({});
	}

	cp.BufferInit();
	bool complete = true;

	switch (submission.type) {
		case SubmissionType::Graphics: {
			bool progressed = false;
			submission.constant_complete |= submission.constant_commands.empty();
			for (;;) {
				bool round_progress = false;
				if (!submission.constant_complete) {
					submission.constant_complete =
					    cp.Process(submission.constant_execution, submission.constant_commands) ==
					    Pm4ProcessResult::Complete;
					round_progress |= submission.constant_execution.MadeProgress();
				}
				cp.SetCeComplete(submission.constant_complete);
				if (!submission.command_complete) {
					submission.command_complete =
					    cp.Process(submission.command_execution, submission.commands) ==
					    Pm4ProcessResult::Complete;
					round_progress |= submission.command_execution.MadeProgress();
				}
				progressed |= round_progress;
				complete = submission.command_complete && submission.constant_complete;
				if (complete || !round_progress) {
					break;
				}
			}
			if (progressed) {
				if (complete) {
					m_renderer.RunGarbageCollector();
				}
				FlushSlice(cp, complete);
			} else if (complete) {
				m_renderer.RunGarbageCollector();
			}
			break;
		}
		case SubmissionType::Compute: {
			const auto      num_dw = static_cast<uint32_t>(submission.commands.size());
			const auto*     buffer = submission.commands.data();
			static uint32_t compute_batch_log_count = 0;
			if (first_slice && num_dw <= 128 && compute_batch_log_count++ < 32) {
				LOGF("compute direct batch: data=0x%016" PRIx64 ", num_dw=%" PRIu32 "\n",
				     reinterpret_cast<uint64_t>(buffer), num_dw);
				for (uint32_t i = 0; i < std::min<uint32_t>(num_dw, 16); i++) {
					LOGF("\t compute[%02" PRIu32 "] = 0x%08" PRIx32 "\n", i, buffer[i]);
				}
			}
			if (first_slice) {
				GraphicsDbgDumpDcb("cc", num_dw, buffer);
			}
			complete = cp.Process(submission.command_execution, submission.commands) ==
			           Pm4ProcessResult::Complete;
			if (submission.command_execution.MadeProgress()) {
				if (complete) {
					m_renderer.RunGarbageCollector();
				}
				FlushSlice(cp, complete);
			} else if (complete) {
				m_renderer.RunGarbageCollector();
			}
			break;
		}
		case SubmissionType::FlipPreparation:
			m_renderer.RunGarbageCollector();
			cp.PrepareCpuFlip(submission.flip_request_id);
			break;
		case SubmissionType::SuspendPoint: {
			// Everything recorded before the marker goes to the GPU. The next suspend point waits
			// for that submission, so the guest runs at most one drain ahead of the GPU.
			cp.BufferFlush();
			const auto        tick = m_renderer.GetCommandScheduler().CurrentTick() - 1;
			Common::LockGuard lock(m_queue_mutex);
			m_suspend_point_gpu_tick = tick;
			m_suspend_points_done++;
			m_suspend_point_done.SignalAll();
			break;
		}
	}

	return complete;
}

Pm4ProcessResult CommandProcessor::Process(Pm4Execution&             execution,
                                           std::span<const uint32_t> commands) {
	KYTY_PROFILER_BLOCK("CommandProcessor::Process");
	EXIT_IF(g_current_execution != nullptr);
	EXIT_IF(commands.size() > UINT32_MAX);
	if (execution.m_buffer_stack.empty() && !commands.empty()) {
		execution.m_buffer_stack.push_back({commands});
	}
	execution.m_suspended     = false;
	execution.m_made_progress = false;

	struct ExecutionScope {
		ExecutionScope(CommandProcessor& processor, Pm4Execution& execution)
		    : previous_processor(g_current_processor), previous_execution(g_current_execution) {
			g_current_processor = &processor;
			g_current_execution = &execution;
		}
		~ExecutionScope() {
			g_current_processor = previous_processor;
			g_current_execution = previous_execution;
		}

		CommandProcessor* previous_processor;
		Pm4Execution*     previous_execution;
	} execution_scope(*this, execution);

	ProcessPm4(execution);
	return execution.m_buffer_stack.empty() ? Pm4ProcessResult::Complete
	                                        : Pm4ProcessResult::Blocked;
}

void CommandProcessor::ProcessIndirectBuffer(std::span<const uint32_t> commands, bool chain) {
	EXIT_IF(g_current_execution == nullptr);
	EXIT_IF(!g_current_execution->m_next_buffer.empty());
	g_current_execution->m_next_buffer = commands;
	g_current_execution->m_chain       = chain;
}

// KYTY_LOCAL_HACK KYTY_PIPELINE_STATS=1 (live, research; pipelineStats.h): packets by class for a
// pipelined command processor, the runs of draws/dispatches between sync packets, and the read-
// after-GPU-write hazards inside an 8-op window. Classes: 0 op (draw, dispatch), 1 state (register
// writes, index state, call into another command buffer, markers), 2 ordered (barriers, cache
// actions, end-of-pipe labels, CP writes and DMA, flips: a back thread can run them in order),
// 3 sync (waits on memory, conditional execution, anything unknown: the front thread drains).
static void PipelineStatsPacket(uint32_t opcode, uint32_t header, uint32_t queue) {
	static auto& on = Common::LiveSwitches::Get("KYTY_PIPELINE_STATS", 0);
	PipelineStats::g_on = on.load(std::memory_order_relaxed) != 0;
	if (!PipelineStats::g_on) return;
	int cls = 3;
	switch (opcode) {
		case Pm4::IT_DISPATCH_DIRECT:
		case Pm4::IT_DISPATCH_INDIRECT:
		case Pm4::IT_DRAW_INDIRECT:
		case Pm4::IT_DRAW_INDEX_INDIRECT:
		case Pm4::IT_DRAW_INDEX_2:
		case Pm4::IT_DRAW_INDIRECT_MULTI:
		case Pm4::IT_DRAW_INDEX_AUTO:
		case Pm4::IT_DRAW_INDEX_OFFSET_2:
		case Pm4::IT_DRAW_INDEX_INDIRECT_MULTI:
		case Pm4::IT_DISPATCH_DRAW: cls = 0; break;
		case Pm4::IT_SET_BASE:
		case Pm4::IT_INDEX_BUFFER_SIZE:
		case Pm4::IT_INDEX_BASE:
		case Pm4::IT_INDEX_TYPE:
		case Pm4::IT_NUM_INSTANCES:
		case Pm4::IT_DISPATCH_DRAW_PREAMBLE:
		case Pm4::IT_SET_SH_REG_INDIRECT:
		case Pm4::IT_SET_UCONFIG_REG_INDIRECT:
		case Pm4::IT_SET_CONFIG_REG:
		case Pm4::IT_SET_CONTEXT_REG:
		case Pm4::IT_SET_SH_REG:
		case Pm4::IT_SET_QUEUE_REG:
		case Pm4::IT_SET_UCONFIG_REG:
		case Pm4::IT_SET_UCONFIG_REG_INDEX:
		case Pm4::IT_SET_CONTEXT_REG_INDIRECT:
		case Pm4::IT_CONTEXT_CONTROL:
		case Pm4::IT_CLEAR_STATE:
		case Pm4::IT_INDIRECT_BUFFER:
		case Pm4::IT_INDIRECT_BUFFER_CNST:
		case Pm4::IT_PFP_SYNC_ME: cls = 1; break;
		case Pm4::IT_EVENT_WRITE:
		case Pm4::IT_EVENT_WRITE_EOP:
		case Pm4::IT_RELEASE_MEM:
		case Pm4::IT_ACQUIRE_MEM:
		case Pm4::IT_WRITE_DATA:
		case Pm4::IT_DMA_DATA: cls = 2; break;
		case Pm4::IT_NOP: {
			const auto r = KYTY_PM4_R(header);
			cls = (r == Pm4::R_ZERO || r == Pm4::R_PUSH_MARKER || r == Pm4::R_POP_MARKER ||
			       r == Pm4::R_DRAW_RESET || r == Pm4::R_DISPATCH_RESET || r == Pm4::R_CONTEXT_STATE)
			          ? 1
			      : (r == Pm4::R_ACQUIRE_MEM || r == Pm4::R_RELEASE_MEM || r == Pm4::R_WRITE_DATA ||
			         r == Pm4::R_DMA_DATA || r == Pm4::R_FLIP)
			          ? 2
			          : 3;
			break;
		}
		default: cls = 3; break;
	}
	static uint64_t counts[4] {};
	static uint64_t sync_by_op[512] {};
	static uint64_t run_ops = 0;          // ops since the last sync packet
	static uint64_t runs_hist[6] {};      // ops in runs of 1, 2-3, 4-7, 8-15, 16-63, 64+
	static uint64_t queue_ops[2] {};      // graphics, compute
	static auto     report = std::chrono::steady_clock::now();
	counts[cls]++;
	if (cls == 0) {
		PipelineStats::g_op++;
		run_ops++;
		queue_ops[queue == 0 ? 0 : 1]++;
	} else if (cls == 3) {
		sync_by_op[opcode == Pm4::IT_NOP ? 256u + KYTY_PM4_R(header) : opcode]++;
		if (run_ops != 0) {
			const int b = run_ops < 2 ? 0 : run_ops < 4 ? 1 : run_ops < 8 ? 2 : run_ops < 16 ? 3 : run_ops < 64 ? 4 : 5;
			runs_hist[b] += run_ops;
		}
		run_ops = 0;
		PipelineStats::g_sync_op = PipelineStats::g_op;
	}
	if (const auto now = std::chrono::steady_clock::now(); now - report > std::chrono::seconds(5)) {
		std::string sync;
		for (size_t i = 0; i < 512; i++) {
			if (sync_by_op[i] != 0) sync += fmt::format(" {}{:02x}={}", i >= 256 ? "R" : "", i & 255u, sync_by_op[i]);
		}
		std::printf("Pipeline stats (5 s): ops %llu (graphics %llu, compute %llu), state %llu, ordered %llu, "
		            "sync %llu; ops in runs 1/2-3/4-7/8-15/16-63/64+: %llu/%llu/%llu/%llu/%llu/%llu; "
		            "hazard reads %llu in %llu ops; sync:%s\n",
		            (unsigned long long)counts[0], (unsigned long long)queue_ops[0],
		            (unsigned long long)queue_ops[1], (unsigned long long)counts[1],
		            (unsigned long long)counts[2], (unsigned long long)counts[3],
		            (unsigned long long)runs_hist[0], (unsigned long long)runs_hist[1],
		            (unsigned long long)runs_hist[2], (unsigned long long)runs_hist[3],
		            (unsigned long long)runs_hist[4], (unsigned long long)runs_hist[5],
		            (unsigned long long)PipelineStats::g_hazard_reads,
		            (unsigned long long)PipelineStats::g_hazard_ops, sync.c_str());
		std::fflush(stdout);
		for (auto& c: counts) c = 0;
		for (auto& c: sync_by_op) c = 0;
		for (auto& c: runs_hist) c = 0;
		queue_ops[0] = queue_ops[1] = 0;
		PipelineStats::g_hazard_reads = PipelineStats::g_hazard_ops = 0;
		report = now;
	}
}

// KYTY_LOCAL_HACK KYTY_LOOKAHEAD_STATS=1 (live, research): after a draw, the shader registers the
// next draw will see are predicted by running the SET_SH_REG packets up to it on a saved copy of
// the state (other register writes, index state and markers are skipped; anything else ends the
// prediction), then compared at that draw with the real ones: how often a thread could prepare
// the next draw's shader resources while this one is still bound and recorded.
static bool LookaheadStatsOn() {
	static auto& on = Common::LiveSwitches::Get("KYTY_LOOKAHEAD_STATS", 0);
	return on.load(std::memory_order_relaxed) != 0;
}
// KYTY_LOCAL_HACK KYTY_LOOKAHEAD (live): the prediction is published (Lookahead::g_next) while
// the draw runs, when it is safe to prepare programs from: no context or user config register
// write was skipped, or the draw's shaders stay the same.
static bool LookaheadOn() {
	static auto& on = Common::LiveSwitches::Get("KYTY_LOOKAHEAD", 0);
	return on.load(std::memory_order_relaxed) != 0;
}

namespace {
struct LookaheadState {
	bool        valid = false;
	HW::Shader  predicted;
	bool        safe  = false;
	bool        trusted = false; // no skipped packet writes memory (RELEASE_MEM, EVENT_WRITE)
	uint64_t    published = 0;
	uint64_t    draws = 0, predicted_draws = 0, exact = 0, stopped = 0, distance = 0;
	std::array<uint64_t, 512> stops {};
	std::chrono::steady_clock::time_point report = std::chrono::steady_clock::now();
};
LookaheadState g_lookahead;
} // namespace

void CommandProcessor::LookaheadCheck() {
	auto& la = g_lookahead;
	la.draws++;
	if (la.valid) {
		la.predicted_draws++;
		if (std::memcmp(&la.predicted, &m_sh_ctx, sizeof(HW::Shader)) == 0) {
			la.exact++;
		}
	}
	la.valid = false;
	if (const auto now = std::chrono::steady_clock::now(); now - la.report > std::chrono::seconds(5)) {
		std::string stops;
		for (size_t i = 0; i < la.stops.size(); i++) {
			if (la.stops[i] != 0) stops += fmt::format(" {}{:02x}={}", i >= 256 ? "R" : "", i & 255u, la.stops[i]);
		}
		std::printf("Lookahead (5 s): draws %llu, predicted %llu, exact %llu, stopped %llu, avg packets %.1f "
		            "(sizeof Shader %zu, Context %zu), published %llu; stops:%s\n",
		            (unsigned long long)la.draws, (unsigned long long)la.predicted_draws,
		            (unsigned long long)la.exact, (unsigned long long)la.stopped,
		            la.predicted_draws != 0 ? double(la.distance) / double(la.predicted_draws) : 0.0,
		            sizeof(HW::Shader), sizeof(HW::Context), (unsigned long long)la.published, stops.c_str());
		std::fflush(stdout);
		const auto keep = std::move(la.predicted);
		la = {};
		la.predicted = keep;
		la.report = now;
	}
}

// The user config registers an indirect write may set while predicting: their handlers only set
// m_ucfg or the index type (both restored), or ignore the value.
static bool LookaheadUcRegisterPure(uint32_t offset) {
	switch (offset) {
		case Pm4::UC_NOP:
		case Pm4::VGT_PRIMITIVE_TYPE:
		case Pm4::VGT_INDEX_TYPE:
		case Pm4::VGT_OBJECT_ID:
		case Pm4::GE_MULTI_PRIM_IB_RESET_EN:
		case Pm4::IA_MULTI_VGT_PARAM:
		case Pm4::GE_CNTL:
		case Pm4::UC_PARAMETER_OVERSUBSCRIPTION:
		case Pm4::GE_USER_VGPR_EN:
		case Pm4::TEXTURE_GRADIENT_FACTORS:
		case Pm4::TEXTURE_GRADIENT_CONTROL: return true;
		default:
			return (offset >= Pm4::FSR_WINDOW_LEFT && offset < Pm4::FSR_WINDOW_LEFT + 2) ||
			       (offset >= Pm4::FSR_CONTROL_POINTS_LEFT_X && offset < Pm4::FSR_CONTROL_POINTS_LEFT_X + 4) ||
			       (offset >= Pm4::FSR_CONTROL_POINTS_LEFT_Y && offset < Pm4::FSR_CONTROL_POINTS_LEFT_Y + 4) ||
			       (offset >= Pm4::FSR_ALPHA_LEFT_X && offset < Pm4::FSR_ALPHA_LEFT_X + 2) ||
			       (offset >= Pm4::FSR_ALPHA_LEFT_Y && offset < Pm4::FSR_ALPHA_LEFT_Y + 2);
	}
}

void CommandProcessor::LookaheadPredict(std::span<const uint32_t> ahead) {
	auto&      la    = g_lookahead;
	const auto saved = m_sh_ctx;
	// Indirect register writes may also set context and user config registers: run on copies,
	// and a change of either makes the prediction unsafe unless the shaders stay the same.
	const auto saved_ctx    = m_ctx;
	const auto saved_ucfg   = m_ucfg;
	const auto saved_index  = m_index_type_and_size;
	const auto saved_marker = m_user_data_marker;
	uint32_t   pos   = 0;
	uint32_t   count = 0;
	bool       ok    = false;
	bool       touched = false;
	bool       writes  = false;
	while (pos < ahead.size()) {
		const auto* packet = ahead.data() + pos;
		const auto  header = packet[0];
		const auto  left   = static_cast<uint32_t>(ahead.size() - pos);
		if (header == 0x80000000u) {
			pos++;
			continue;
		}
		if ((header >> 30u) == 0u) {
			pos += ((header >> 16u) & 0x3fffu) + 2u;
			continue;
		}
		const auto opcode = (header >> 8u) & 0xffu;
		const auto len    = KYTY_PM4_LEN(header);
		if (len == 0 || len > left || (header & 1u) != 0) {
			la.stops[opcode]++;
			break;
		}
		count++;
		if (IsDrawOrDispatchPacket(opcode)) {
			ok = opcode != Pm4::IT_DISPATCH_DIRECT && opcode != Pm4::IT_DISPATCH_INDIRECT;
			if (!ok) la.stops[opcode]++;
			break;
		}
		writes  = writes || opcode == Pm4::IT_EVENT_WRITE ||
		          (opcode == Pm4::IT_NOP && KYTY_PM4_R(header) == Pm4::R_RELEASE_MEM);
		touched = touched || opcode == Pm4::IT_SET_CONTEXT_REG || opcode == Pm4::IT_SET_UCONFIG_REG ||
		          opcode == Pm4::IT_SET_CONTEXT_REG_INDIRECT || opcode == Pm4::IT_SET_UCONFIG_REG_INDEX;
		bool run = opcode == Pm4::IT_SET_SH_REG || (opcode == Pm4::IT_SET_SH_REG_INDIRECT && len == 5u);
		if (opcode == Pm4::IT_SET_UCONFIG_REG_INDIRECT && len == 5u) {
			const auto* regs = reinterpret_cast<const uint32_t*>(
			    (static_cast<uint64_t>(packet[1]) & 0xfffffffcu) | (static_cast<uint64_t>(packet[2]) << 32u));
			const auto num = packet[4] & 0x3fffu;
			run            = num == 0 || regs != nullptr;
			for (uint32_t i = 0; run && i < num; i++) {
				run = LookaheadUcRegisterPure(regs[i * 2] & ~0x70000000u);
			}
			if (!run) {
				la.stops[opcode]++;
				break;
			}
		}
		if (opcode == Pm4::IT_SET_SH_REG_INDIRECT && len == 5u && packet[4] != 0 &&
		    ((static_cast<uint64_t>(packet[1]) & 0xfffffffcu) | (static_cast<uint64_t>(packet[2]) << 32u)) == 0) {
			la.stops[opcode]++;
			break;
		}
		if (run) {
			(void)g_cp_op_func[opcode](*this, header & ~1u, packet + 1, left, static_cast<uint32_t>(ahead.size()));
		} else if (!(opcode == Pm4::IT_SET_CONTEXT_REG || opcode == Pm4::IT_SET_UCONFIG_REG ||
		             opcode == Pm4::IT_SET_CONTEXT_REG_INDIRECT || opcode == Pm4::IT_EVENT_WRITE ||
		             (opcode == Pm4::IT_NOP && (KYTY_PM4_R(header) == Pm4::R_RELEASE_MEM ||
		                                        KYTY_PM4_R(header) == Pm4::R_ACQUIRE_MEM)) ||
		             opcode == Pm4::IT_SET_UCONFIG_REG_INDEX || opcode == Pm4::IT_INDEX_TYPE ||
		             opcode == Pm4::IT_INDEX_BASE || opcode == Pm4::IT_INDEX_BUFFER_SIZE ||
		             opcode == Pm4::IT_NUM_INSTANCES || opcode == Pm4::IT_SET_BASE ||
		             (opcode == Pm4::IT_NOP && (KYTY_PM4_R(header) == Pm4::R_ZERO ||
		                                        KYTY_PM4_R(header) == Pm4::R_PUSH_MARKER ||
		                                        KYTY_PM4_R(header) == Pm4::R_POP_MARKER ||
		                                        KYTY_PM4_R(header) == Pm4::R_DRAW_RESET)))) {
			la.stops[opcode == Pm4::IT_NOP ? 256u + KYTY_PM4_R(header) : opcode]++;
			break;
		}
		pos += len;
	}
	if (ok) {
		la.predicted = m_sh_ctx;
		la.valid     = true;
		la.distance += count;
		la.trusted = !writes;
		touched = touched || std::memcmp(&saved_ctx, &m_ctx, sizeof(HW::Context)) != 0 ||
		          std::memcmp(&saved_ucfg, &m_ucfg, sizeof(HW::UserConfig)) != 0;
		la.safe = !touched || (saved.GetVs().es_regs.data_addr == m_sh_ctx.GetVs().es_regs.data_addr &&
		                       saved.GetVs().gs_regs.data_addr == m_sh_ctx.GetVs().gs_regs.data_addr &&
		                       saved.GetPs().ps_regs.data_addr == m_sh_ctx.GetPs().ps_regs.data_addr);
	} else {
		la.stopped++;
	}
	m_sh_ctx               = saved;
	m_ctx                  = saved_ctx;
	m_ucfg                 = saved_ucfg;
	m_index_type_and_size  = saved_index;
	m_user_data_marker     = saved_marker;
}

void CommandProcessor::SuspendPm4() {
	EXIT_IF(g_current_execution == nullptr);
	g_current_execution->m_suspended = true;
}

void CommandProcessor::ProcessPm4(Pm4Execution& execution) {
	while (!execution.m_buffer_stack.empty()) {
		if (g_gpu_state != nullptr) {
			g_gpu_state->ProcessCommands();
		}
		auto& cursor = execution.m_buffer_stack.back();
		EXIT_IF(cursor.offset_dw > cursor.commands.size());
		if (cursor.offset_dw == cursor.commands.size()) {
			execution.m_buffer_stack.pop_back();
			continue;
		}

		const auto* const packet        = cursor.commands.data() + cursor.offset_dw;
		const auto        total_dw      = static_cast<uint32_t>(cursor.commands.size());
		const auto        remaining_dw  = total_dw - cursor.offset_dw;
		const auto        packet_header = packet[0];
		const auto        opcode        = (packet_header >> 8u) & 0xffu;
		EXIT_NOT_IMPLEMENTED(remaining_dw > total_dw);

		if (packet_header == 0x80000000u) {
			cursor.offset_dw++;
			execution.m_made_progress = true;
			continue;
		}

		EXIT_NOT_IMPLEMENTED(remaining_dw < 2);

		if ((packet_header >> 30u) == 0u) {
			const auto packet_dw = ((packet_header >> 16u) & 0x3fffu) + 2u;
			EXIT_NOT_IMPLEMENTED(packet_dw > remaining_dw);
			cursor.offset_dw += packet_dw;
			execution.m_made_progress = true;
			continue;
		}

		if (GraphicsRunDebugDumpEnabled()) {
			LOGF("CP packet: offset=0x%05" PRIx32 " cmd_id=0x%08" PRIx32 " op=0x%02" PRIx32
			     " len=%" PRIu32 "\n",
			     total_dw - remaining_dw, packet_header, opcode, KYTY_PM4_LEN(packet_header));
		}

		if ((packet_header & 1u) != 0 && m_predication_op != 0 && OcclusionStatsEnabled()) {
			const bool draw = IsDrawOrDispatchPacket(opcode);
			if (ShouldSkipPredicatedPackets()) {
				g_occlusion_stats.packets_skipped.fetch_add(1, std::memory_order_relaxed);
				g_occlusion_stats.draws_skipped.fetch_add(draw ? 1 : 0, std::memory_order_relaxed);
			} else {
				g_occlusion_stats.packets_executed.fetch_add(1, std::memory_order_relaxed);
				g_occlusion_stats.draws_executed.fetch_add(draw ? 1 : 0, std::memory_order_relaxed);
			}
			ReportOcclusionStats();
		}
		if ((packet_header & 1u) != 0 && ShouldSkipPredicatedPackets()) {
			auto packet_dw = KYTY_PM4_LEN(packet_header);
			EXIT_NOT_IMPLEMENTED(packet_dw == 0 || packet_dw > remaining_dw);
			static std::atomic<uint32_t> skip_log_count {0};
			if (skip_log_count.fetch_add(1) < 2048) {
				LOGF("\t predicated skip: op=0x%02" PRIx32 ", r=0x%02" PRIx32 ", len=%" PRIu32
				     ", packet=0x%016" PRIx64 ", cmd_id=0x%08" PRIx32 "\n",
				     opcode, KYTY_PM4_R(packet_header), packet_dw,
				     reinterpret_cast<uint64_t>(packet), packet_header);
			}
			if (opcode == Pm4::IT_NOP && KYTY_PM4_R(packet_header) == Pm4::R_RELEASE_MEM &&
			    packet_dw >= 7) {
				static std::atomic<uint32_t> log_count {0};
				if (log_count.fetch_add(1) < 128) {
					const auto dst = packet[3] | (static_cast<uint64_t>(packet[4]) << 32u);
					const auto val = packet[5] | (static_cast<uint64_t>(packet[6]) << 32u);
					LOGF("\t predicated skip: R_RELEASE_MEM dst=0x%016" PRIx64
					     ", value=0x%016" PRIx64 ", action=0x%08" PRIx32
					     ", gcr/data/int=0x%08" PRIx32 "\n",
					     dst, val, packet[1], packet[2]);
				}
			}
			cursor.offset_dw += packet_dw;
			execution.m_made_progress = true;
			continue;
		}

		auto handler = g_cp_op_func[opcode];

		if (handler == nullptr) {
			const auto offset = total_dw - remaining_dw;
			LOGF("unknown PM4 packet: data=0x%016" PRIx64 ", num_dw=%" PRIu32
			     ", offset=0x%05" PRIx32 ", current=0x%016" PRIx64 "\n",
			     reinterpret_cast<uint64_t>(packet - offset), total_dw, offset,
			     reinterpret_cast<uint64_t>(packet));
			const auto  dump_begin = (offset > 8 ? offset - 8 : 0);
			const auto  dump_end   = std::min<uint32_t>(total_dw, offset + 16);
			auto* const base       = packet - offset;
			for (uint32_t i = dump_begin; i < dump_end; i++) {
				LOGF("\t%05" PRIx32 "%s %08" PRIx32 "\n", i, (i == offset ? ":" : " "), base[i]);
			}
			EXIT("unknown op\n\t%05" PRIx32 ":\n\tcmd_id = %08" PRIx32 "\n",
			     total_dw - remaining_dw, packet_header);
		}

		PipelineStatsPacket(opcode, packet_header, g_stats_queue);
		const bool lookahead_draw = (LookaheadStatsOn() || LookaheadOn()) && IsDrawOrDispatchPacket(opcode) &&
		                            opcode != Pm4::IT_DISPATCH_DIRECT &&
		                            opcode != Pm4::IT_DISPATCH_INDIRECT && (packet_header & 1u) == 0 &&
		                            KYTY_PM4_LEN(packet_header) <= remaining_dw;
		if (lookahead_draw) {
			LookaheadCheck();
			// The draw does not write shader registers: the state after it is the state now.
			LookaheadPredict(cursor.commands.subspan(cursor.offset_dw + KYTY_PM4_LEN(packet_header)));
			if (g_lookahead.valid && g_lookahead.safe && LookaheadOn()) {
				Lookahead::g_next         = &g_lookahead.predicted;
				Lookahead::g_next_trusted = g_lookahead.trusted;
				g_lookahead.published++;
			}
		}
		const auto packet_dw =
		    handler(*this, packet_header & ~1u, packet + 1, remaining_dw, total_dw) + 1;
		Lookahead::g_next = nullptr;
		EXIT_IF(packet_dw > remaining_dw);
		if (execution.m_suspended) {
			return;
		}
		cursor.offset_dw += packet_dw;
		execution.m_made_progress = true;
		if (!execution.m_next_buffer.empty()) {
			// Chains and taken branches reuse the fetcher; only calls retain a return cursor.
			if (execution.m_chain) {
				cursor = {execution.m_next_buffer};
			} else {
				execution.m_buffer_stack.push_back({execution.m_next_buffer});
			}
			execution.m_next_buffer = {};
		}
	}
}

void CommandProcessor::SetIndexType(uint32_t index_type_and_size) {
	m_index_type_and_size = index_type_and_size & 0x3u;
}

void CommandProcessor::SetIndexBaseAddress(uint64_t index_base_addr) {
	m_index_base_addr = index_base_addr;
}

void CommandProcessor::SetIndexBufferSize(uint32_t index_buffer_size) {
	m_index_buffer_size = index_buffer_size;
}

void CommandProcessor::SetDrawIndirectArgsBaseAddress(uint64_t draw_indirect_args_base_addr) {
	m_draw_indirect_args_base_addr = draw_indirect_args_base_addr;
}

void CommandProcessor::SetDispatchIndirectArgsBaseAddress(
    uint64_t dispatch_indirect_args_base_addr) {
	m_dispatch_indirect_args_base_addr = dispatch_indirect_args_base_addr;
}

void CommandProcessor::SetNumInstances(uint32_t num_instances) {
	if (num_instances == 0) {
		num_instances = 1;
	}

	m_num_instances = num_instances;
}

void CommandProcessor::SetPredication(uint32_t condition, uint32_t op, uint32_t wait_op,
                                      const volatile void* address, uint32_t count_in_dwords) {
	(void)count_in_dwords;
	uint64_t value = 0;
	m_predication_op = op;
	if (OcclusionStatsEnabled()) {
		auto& counter = op == 0x01   ? g_occlusion_stats.predication_zpass
		                : op == 0x03 ? g_occlusion_stats.predication_bool
		                             : g_occlusion_stats.predication_off;
		counter.fetch_add(1, std::memory_order_relaxed);
		ReportOcclusionStats();
	}

	switch (op) {
		case 0x00:
			m_predicate_skip = false;
			return;
		case 0x01: {
			EXIT_NOT_IMPLEMENTED(address == nullptr);
			// One begin/end pair per DB; bit 63 marks each counter ready.
			constexpr uint64_t ready_bit = 1ull << 63u;
			const auto* results = reinterpret_cast<const volatile uint64_t*>(address);
			for (uint32_t db = 0; db < 16u; db++) {
				const auto begin = results[db * 2u];
				const auto end   = results[db * 2u + 1u];
				if ((begin & end & ready_bit) == 0) {
					if (wait_op == 0) {
						KYTY_PROFILER_BLOCK("Pm4Suspend::Predication");
						SuspendPm4();
					} else {
						m_predicate_skip = false;
					}
					return;
				}
				value += end - begin;
			}
			// KYTY_OCCLUSION_CULL=1 (live, research only): treat every occlusion-predicated
			// packet as occluded, as if the test said "nothing visible". Objects disappear; this
			// is a ceiling for what real host occlusion queries could save.
			static auto& cull = Common::LiveSwitches::Get("KYTY_OCCLUSION_CULL", 0);
			if (cull.load(std::memory_order_relaxed) != 0) {
				value = condition == 0x00 ? 1 : 0;
			}
		} break;
		case 0x03: {
			// KYTY_BOOL_PREDICATION_WAIT (live, default 1): drain the GPU before reading a bool
			// predicate whose packet sets the wait bit. Upstream 840b9f57 dropped the drain: the
			// bit applies to Z-pass query readiness, and a value the GPU still owns is read
			// through the page tracker's readback anyway. 0 skips the drain (A/B).
			static auto& bool_wait = Common::LiveSwitches::Get("KYTY_BOOL_PREDICATION_WAIT", 1);
			if (wait_op != 0 && bool_wait.load(std::memory_order_relaxed) != 0) {
				BufferFlushAndWait();
			}
			EXIT_NOT_IMPLEMENTED(address == nullptr);
			value = *reinterpret_cast<const volatile uint64_t*>(address);
		} break;
		default: EXIT("unknown predication op: 0x%08" PRIx32 "\n", op);
	}
	switch (condition) {
		case 0x00: m_predicate_skip = (value != 0); break;
		case 0x01: m_predicate_skip = (value == 0); break;
		default: EXIT("unknown predication condition: 0x%08" PRIx32 "\n", condition);
	}
	if (op == 0x03) {
		static std::atomic<uint32_t> log_count {0};
		if (log_count.fetch_add(1) < 128) {
			LOGF("\t bool predication: addr=0x%016" PRIx64 ", value=0x%016" PRIx64
			     ", condition=%" PRIu32 ", skip=%u, wait_op=%" PRIu32 "\n",
			     reinterpret_cast<uint64_t>(address), value, condition,
			     m_predicate_skip ? 1u : 0u, wait_op);
		}
	}
}

void CommandProcessor::DrawIndex(DrawIndexArgs args) {
	args.index_type_and_size = m_index_type_and_size;
	if (args.instance_count == 0) {
		args.instance_count = m_num_instances;
	}
	if (GraphicsRunDebugDumpEnabled() && (args.base_vertex != 0 || args.first_instance != 0)) {
		LOGF("\t draw indexed offsets: base_vertex = %" PRId32 ", first_instance = %" PRIu32 "\n",
		     args.base_vertex, args.first_instance);
	}
	m_renderer.GetRenderExecutor().DrawIndex(m_submit_id, CurrentBuffer(), args);
}

void CommandProcessor::DrawIndexOffset(uint32_t index_offset, uint32_t index_count) {
	uint64_t index_size = 0;
	switch (m_index_type_and_size) {
		case 0: index_size = 2; break;
		case 1: index_size = 4; break;
		case 2: index_size = 1; break;
		default: EXIT("unknown index_type_and_size: %u\n", m_index_type_and_size);
	}

	auto* index_addr = reinterpret_cast<const void*>(
	    m_index_base_addr + static_cast<uint64_t>(index_offset) * index_size);

	DrawIndex({.index_count = index_count, .index_addr = index_addr});
}

void CommandProcessor::DrawIndirect(uint32_t data_offset, uint32_t draw_initiator, bool indexed) {
	EXIT_NOT_IMPLEMENTED((draw_initiator & ~0x20u) != 2u);
	EXIT_NOT_IMPLEMENTED(m_draw_indirect_args_base_addr == 0);

	const auto* args_addr =
	    reinterpret_cast<const void*>(m_draw_indirect_args_base_addr + data_offset);

	// Research: the GPU reads its own arguments. A CPU read of arguments an earlier GPU pass
	// wrote faults on the tracked page and drains the whole queue, once per draw. 8-bit indices
	// (expanded on the host) and an unknown index buffer size keep the CPU read.
	if (!indexed) {
		DrawIndexAuto({.vertex_count   = 1,
		               .instance_count = 1,
		               .offset_source  = DrawOffsetSource::IndirectArgs,
		               .gpu_args       = reinterpret_cast<uint64_t>(args_addr)});
		return;
	}
	if (m_index_buffer_size != 0 && (m_index_type_and_size == 0 || m_index_type_and_size == 1)) {
		DrawIndex({.index_count    = m_index_buffer_size,
		           .index_addr     = reinterpret_cast<const void*>(m_index_base_addr),
		           .instance_count = 1,
		           .offset_source  = DrawOffsetSource::IndirectArgs,
		           .gpu_args       = reinterpret_cast<uint64_t>(args_addr)});
		return;
	}

	if (!indexed) {
		DrawIndirectArgs args {};
		std::memcpy(&args, args_addr, sizeof(args));
		m_num_instances = args.instance_count;
		DrawIndexAuto({.vertex_count   = args.vertex_count_per_instance,
		               .instance_count = args.instance_count,
		               .first_vertex   = args.start_vertex_location,
		               .first_instance = args.start_instance_location,
		               .offset_source  = DrawOffsetSource::IndirectArgs});
		return;
	}

	DrawIndexedIndirectArgs args {};
	std::memcpy(&args, args_addr, sizeof(args));

	uint64_t index_size = 0;
	switch (m_index_type_and_size) {
		case 0: index_size = 2; break;
		case 1: index_size = 4; break;
		case 2: index_size = 1; break;
		default: EXIT("unknown index_type_and_size: %u\n", m_index_type_and_size);
	}

	auto* index_addr = reinterpret_cast<const void*>(
	    m_index_base_addr + static_cast<uint64_t>(args.start_index_location) * index_size);

	const uint32_t index_count =
	    (m_index_buffer_size != 0 ? std::min(args.index_count_per_instance, m_index_buffer_size)
	                              : args.index_count_per_instance);
	if (GraphicsRunDebugDumpEnabled() && index_count != args.index_count_per_instance) {
		static std::atomic<uint32_t> log_count {0};
		if (log_count.fetch_add(1, std::memory_order_relaxed) < 64) {
			LOGF("\t DrawIndexIndirect: clamped index_count from %" PRIu32 " to %" PRIu32
			     " using INDEX_BUFFER_SIZE\n",
			     args.index_count_per_instance, index_count);
		}
	}

	m_num_instances = args.instance_count;
	DrawIndex({.index_count    = index_count,
	           .index_addr     = index_addr,
	           .instance_count = args.instance_count,
	           .base_vertex    = static_cast<int32_t>(args.base_vertex_location),
	           .first_instance = args.start_instance_location,
	           .offset_source  = DrawOffsetSource::IndirectArgs});
}

void CommandProcessor::DrawIndirectMulti(uint32_t data_offset, uint32_t max_count_or_count,
                                         const volatile uint32_t* count_addr,
                                         uint32_t stride_in_bytes, uint32_t draw_initiator,
                                         bool indexed, uint32_t draw_index_register) {
	EXIT_NOT_IMPLEMENTED((draw_initiator & ~0x20u) != 2u);
	EXIT_NOT_IMPLEMENTED(m_draw_indirect_args_base_addr == 0);

	uint32_t draw_count = max_count_or_count;
	if (count_addr != nullptr) {
		draw_count = *count_addr;
		if (draw_count > max_count_or_count) {
			draw_count = max_count_or_count;
		}
	}

	if (draw_count == 0) {
		return;
	}

	const auto args_size = indexed ? sizeof(DrawIndexedIndirectArgs) : sizeof(DrawIndirectArgs);
	EXIT_NOT_IMPLEMENTED(stride_in_bytes < args_size);

	uint64_t index_size = 0;
	if (indexed) {
		switch (m_index_type_and_size) {
			case 0: index_size = 2; break;
			case 1: index_size = 4; break;
			case 2: index_size = 1; break;
			default: EXIT("unknown index_type_and_size: %u\n", m_index_type_and_size);
		}
	}

	for (uint32_t i = 0; i < draw_count; i++) {
		// DRAW_INDIRECT_MULTI writes DrawIndex before each draw, including draws
		// with zero vertices/instances. It starts at zero for each packet.
		if (draw_index_register != Pm4::SH_NOP) {
			EXIT_NOT_IMPLEMENTED(draw_index_register >= Pm4::SH_NUM ||
			                     g_hw_sh_indirect_func[draw_index_register] == nullptr);
			SetUserDataMarker(HW::UserSgprType::Unknown);
			g_hw_sh_indirect_func[draw_index_register](*this, draw_index_register, i);
		}
		const auto args_addr = m_draw_indirect_args_base_addr + data_offset +
		                       static_cast<uint64_t>(i) * stride_in_bytes;

		if (!indexed) {
			auto* args      = reinterpret_cast<const DrawIndirectArgs*>(args_addr);
			m_num_instances = args->instance_count;
			DrawIndexAuto({.vertex_count   = args->vertex_count_per_instance,
			               .instance_count = args->instance_count,
			               .first_vertex   = args->start_vertex_location,
			               .first_instance = args->start_instance_location,
			               .offset_source  = DrawOffsetSource::IndirectArgs});
			continue;
		}

		auto* args = reinterpret_cast<const DrawIndexedIndirectArgs*>(args_addr);

		auto* index_addr = reinterpret_cast<const void*>(
		    m_index_base_addr + static_cast<uint64_t>(args->start_index_location) * index_size);

		const uint32_t index_count =
		    (m_index_buffer_size != 0
		         ? std::min(args->index_count_per_instance, m_index_buffer_size)
		         : args->index_count_per_instance);
		if (GraphicsRunDebugDumpEnabled() && index_count != args->index_count_per_instance) {
			static std::atomic<uint32_t> log_count {0};
			if (log_count.fetch_add(1, std::memory_order_relaxed) < 64) {
				LOGF("\t DrawIndexIndirectMulti: clamped index_count from %" PRIu32 " to %" PRIu32
				     " using INDEX_BUFFER_SIZE\n",
				     args->index_count_per_instance, index_count);
			}
		}

		m_num_instances = args->instance_count;
		DrawIndex({.index_count    = index_count,
		           .index_addr     = index_addr,
		           .instance_count = args->instance_count,
		           .base_vertex    = static_cast<int32_t>(args->base_vertex_location),
		           .first_instance = args->start_instance_location,
		           .offset_source  = DrawOffsetSource::IndirectArgs});
	}
}

void CommandProcessor::DispatchDirect(uint32_t thread_group_x, uint32_t thread_group_y,
                                      uint32_t thread_group_z, uint32_t mode) {
	m_sh_ctx.SetCsWaveSize(Pm4::ComputeWaveSize(mode));

	uint32_t frame_num = 0;
	// uint32_t local_x   = 1;
	// uint32_t local_y   = 1;
	// uint32_t local_z   = 1;

	{
		frame_num = m_renderer.GetGpu().GetFrameNum();
		if (GraphicsRunDebugDumpEnabled()) {
			static std::atomic<uint32_t> log_count {0};
			if (log_count.fetch_add(1, std::memory_order_relaxed) < 1024) {
				const auto& cs = m_sh_ctx.GetCs().cs_regs;
				const auto& oa = m_ucfg.GetGdsOaCounter(m_ucfg.GetGdsOaState().GetIndex());
				LOGF("QueuePoint DispatchDirect: frame=%u submit=%" PRIu64
				     " groups=%ux%ux%u local=%ux%ux%u mode=0x%08" PRIx32 " wave=%u cs=0x%016" PRIx64
				     " oa_index=%u oa_enabled=%s oa_addr=0x%04" PRIx32 " oa_space=0x%08" PRIx32
				     "\n",
				     frame_num, m_submit_id, thread_group_x, thread_group_y, thread_group_z,
				     std::max(cs.num_thread_x, 1u), std::max(cs.num_thread_y, 1u),
				     std::max(cs.num_thread_z, 1u), mode, static_cast<uint32_t>(cs.wave_size),
				     cs.data_addr, m_ucfg.GetGdsOaState().GetIndex(),
				     oa.IsCounterEnabled() ? "true" : "false", oa.GetAddressBytes(),
				     oa.GetSpaceAvailable());
			}
		}

		const auto& cs = m_sh_ctx.GetCs().cs_regs;
		// local_x        = std::max(cs.num_thread_x, 1u);
		// local_y        = std::max(cs.num_thread_y, 1u);
		// local_z        = std::max(cs.num_thread_z, 1u);
		m_renderer.GetRenderExecutor().DispatchDirect(m_submit_id, CurrentBuffer(), thread_group_x,
		                                              thread_group_y, thread_group_z, mode);
	}

	/*constexpr uint32_t DispatchInitiatorUseThreadDimensions = 1u << 5u;
	auto               group_count = [](uint32_t threads, uint32_t group_size) {
	    return (threads == 0
	                ? 0u
	                : (threads + std::max(group_size, 1u) - 1u) / std::max(group_size, 1u));
	};

	auto groups_x = thread_group_x;
	auto groups_y = thread_group_y;
	auto groups_z = thread_group_z;
	if ((mode & DispatchInitiatorUseThreadDimensions) != 0) {
	    groups_x = group_count(thread_group_x, local_x);
	    groups_y = group_count(thread_group_y, local_y);
	    groups_z = group_count(thread_group_z, local_z);
	}

	const uint64_t invocations =
	    static_cast<uint64_t>(groups_x) * groups_y * groups_z * local_x * local_y * local_z;
	if (invocations != 0) {
	    BufferFlushAndWait();
	}*/
}

void CommandProcessor::DispatchIndirect(uint64_t args_addr, uint32_t mode) {
	EXIT_NOT_IMPLEMENTED(args_addr == 0 || (args_addr & 3u) != 0);
	// Thread counts are converted on the GPU. Reading them here drained the GPU for every
	// dispatch, since the previous dispatch writes them; KYTY_INDIRECT_THREADS_ON_CPU=1 restores
	// that path for A/B runs.
	static const bool threads_on_cpu = std::getenv("KYTY_INDIRECT_THREADS_ON_CPU") != nullptr;
	if (threads_on_cpu && (mode & Pm4::COMPUTE_DISPATCH_INITIATOR_USE_THREAD_DIMENSIONS) != 0) {
		const auto* args = reinterpret_cast<const vk::DispatchIndirectCommand*>(args_addr);
		DispatchDirect(args->x, args->y, args->z, mode);
		return;
	}
	m_sh_ctx.SetCsWaveSize(Pm4::ComputeWaveSize(mode));
	m_renderer.GetRenderExecutor().DispatchIndirect(m_submit_id, CurrentBuffer(), args_addr, mode);
}

void CommandProcessor::DrawIndexAuto(DrawAutoArgs args) {
	if (args.instance_count == 0) {
		args.instance_count = m_num_instances;
	}
	m_renderer.GetRenderExecutor().DrawAuto(m_submit_id, CurrentBuffer(), args);
}

void CommandProcessor::WaitFlipDone(uint32_t video_out_handle, uint32_t display_buffer_index) {
	BufferFlush();

	m_renderer.GetVideoOut().WaitFlipDone(static_cast<int>(video_out_handle),
	                                      static_cast<int>(display_buffer_index));
}

template <typename T>
void CommandProcessor::WriteAtEndOfPipe(uint32_t cache_policy, uint32_t event_write_dest,
                                        uint32_t eop_event_type, uint32_t cache_action,
                                        uint32_t event_index, uint32_t event_write_source,
                                        void* dst_gpu_addr, T value, uint32_t interrupt_selector,
                                        uint32_t interrupt_context_id) {
	static_assert(sizeof(T) == sizeof(uint32_t) || sizeof(T) == sizeof(uint64_t));

	auto& command = CurrentBuffer();

	if (GraphicsRunDebugDumpEnabled()) {
		const auto bits      = static_cast<unsigned>(sizeof(T) * 8u);
		const auto log_width = static_cast<int>(sizeof(T) * 2u);

		LOGF("CommandProcessor::WriteAtEndOfPipe%u()\n"
		     "\t cache_policy        = 0x%08" PRIx32 "\n"
		     "\t event_write_dest    = 0x%08" PRIx32 "\n"
		     "\t eop_event_type      = 0x%08" PRIx32 "\n"
		     "\t cache_action        = 0x%08" PRIx32 "\n"
		     "\t event_index         = 0x%08" PRIx32 "\n"
		     "\t event_write_source  = 0x%08" PRIx32 "\n"
		     "\t interrupt_selector  = 0x%08" PRIx32 "\n"
		     "\t interrupt_context   = 0x%08" PRIx32 "\n"
		     "\t dst_gpu_addr        = 0x%016" PRIx64 "\n"
		     "\t value               = 0x%0*" PRIx64 "\n",
		     bits, cache_policy, event_write_dest, eop_event_type, cache_action, event_index,
		     event_write_source, interrupt_selector, interrupt_context_id,
		     reinterpret_cast<uint64_t>(dst_gpu_addr), log_width, static_cast<uint64_t>(value));
	}

	EXIT_NOT_IMPLEMENTED(cache_policy != 0x00000000);
	EXIT_NOT_IMPLEMENTED(event_write_dest != 0x00000000);

	bool with_interrupt = false;
	switch (interrupt_selector) {
		case 0x00:
		case 0x03: with_interrupt = false; break;
		case 0x01:
			if (!IsAsyncComputeQueue()) {
				Sync::TriggerEopEventAtEndOfPipe(command, m_interrupt_event_id,
				                                 interrupt_context_id);
				return;
			}
			with_interrupt = true;
			break;
		case 0x02: with_interrupt = true; break;
		default: EXIT("unknown interrupt_selector\n");
	}

	const bool at_completion = !with_interrupt && GuestGpu::LabelsAtCompletion();

	auto write32 = [&](bool with_writeback) {
		auto* dst  = static_cast<uint32_t*>(dst_gpu_addr);
		auto  data = static_cast<uint32_t>(value);
		if (LabelsAfterGpu()) {
			// Published when the preceding GPU work has executed, not when it was recorded.
			PublishLabelAtCompletion(dst, data, sizeof(data), false);
		} else if (at_completion) {
			m_renderer.GetGpu().DeferLabelWrite(reinterpret_cast<uint64_t>(dst), data,
			                                    sizeof(data));
		} else {
			std::memcpy(dst, &data, sizeof(data));
		}

		if (with_interrupt) {
			if (with_writeback) {
				Sync::WriteAtEndOfPipeWithInterruptWriteBack32(m_submit_id, command, dst,
				                                               data, m_interrupt_event_id,
				                                               interrupt_context_id);
			} else {
				Sync::WriteAtEndOfPipeWithInterrupt32(m_submit_id, command, dst, data,
				                                      m_interrupt_event_id, interrupt_context_id);
			}
		} else if (with_writeback) {
			Sync::WriteAtEndOfPipeWithWriteBack32(m_submit_id, command, dst, data);
		} else {
			Sync::WriteAtEndOfPipe32(m_submit_id, command, dst, data);
		}
		if (LabelsAfterGpu()) {
			// Submit so the tick can complete; the label and any interrupt follow it.
			FlushForLabel();
		}
	};

	switch (event_write_source) {
		case 0x01:
			if constexpr (sizeof(T) == sizeof(uint32_t)) {
				if (eop_event_type == 0x2f && cache_action == 0x00 && event_index == 0x06) {
					auto* dst = static_cast<uint32_t*>(dst_gpu_addr);
					SynchronizeGpu();
					Sync::ReadGds(*m_renderer.GetBufferCache().GetGdsBuffer(), dst, value & 0xffffu,
					              value >> 16u);
					Sync::WriteAtEndOfPipeGds32(m_submit_id, command, dst, value & 0xffffu,
					                            value >> 16u);
					if (with_interrupt) {
						m_renderer.TriggerInterrupt(m_interrupt_event_id, interrupt_context_id);
					}
					return;
				}
			} else if (eop_event_type == 0x04 && cache_action == 0x00 && event_index == 0x05) {
				write32(false);
				return;
			}
			break;
		case 0x02:
		case 0x04:
			if constexpr (sizeof(T) == sizeof(uint32_t)) {
				if (event_write_source == 0x02 && eop_event_type == 0x2f && event_index == 0x06) {
					switch (cache_action) {
						case 0x00: write32(false); return;
						case 0x38: write32(true); return;
						default: break;
					}
				}
			} else {
				const bool clock_value = event_write_source == 0x04;
				if (clock_value && !LabelsAfterGpu()) {
					value = Sync::ReadReferenceClock();
				}
				auto write64 = [&](bool with_writeback) {
					auto* dst = static_cast<uint64_t*>(dst_gpu_addr);
					if (LabelsAfterGpu()) {
						// The label (or the timestamp, read then) lands when the GPU gets there.
						PublishLabelAtCompletion(dst, value, sizeof(value), clock_value);
					} else if (at_completion) {
						m_renderer.GetGpu().DeferLabelWrite(reinterpret_cast<uint64_t>(dst),
						                                    value, sizeof(value));
					} else {
						std::memcpy(dst, &value, sizeof(value));
					}

					if (with_interrupt) {
						if (with_writeback) {
							Sync::WriteAtEndOfPipeWithInterruptWriteBack64(
							    m_submit_id, command, dst, value, m_interrupt_event_id,
							    interrupt_context_id);
						} else {
							Sync::WriteAtEndOfPipeWithInterrupt64(m_submit_id, command, dst,
							                                      value, m_interrupt_event_id,
							                                      interrupt_context_id);
						}
					} else if (with_writeback) {
						Sync::WriteAtEndOfPipeWithWriteBack64(m_submit_id, command, dst,
						                                      value);
					} else {
						Sync::WriteAtEndOfPipe64(m_submit_id, command, dst, value);
					}
					if (LabelsAfterGpu()) {
						FlushForLabel();
					}
				};

				switch (cache_action) {
					case 0x00:
						switch (eop_event_type) {
							case 0x04:
								if (event_index == 0x05) {
									write64(false);
									return;
								}
								break;
							case 0x14:
							case 0x28:
							case 0x2f:
								if (event_index == 0x00) {
									write64(false);
									return;
								}
								break;
							case 0x2b:
							case 0x2d:
							case 0x30:
								if (event_index == 0x00 && !with_interrupt) {
									write64(false);
									return;
								}
								break;
							default: break;
						}
						break;
					case 0x38:
						switch (eop_event_type) {
							case 0x04:
							case 0x14:
							case 0x28:
								if (((eop_event_type == 0x04 || eop_event_type == 0x28) &&
								     event_index == 0x05) ||
								    (event_index == 0x00)) {
									write64(true);
									return;
								}
								break;
							case 0x2b:
							case 0x2d:
								if (event_index == 0x00 && !with_interrupt) {
									write64(true);
									return;
								}
								break;
							case 0x2f:
								if (event_index == 0x06 && !with_interrupt) {
									write64(true);
									return;
								}
								break;
							default: break;
						}
						break;
					case 0x3b:
						if (eop_event_type == 0x04 && event_index == 0x05 && with_interrupt) {
							write64(true);
							return;
						}
						break;
					default: break;
				}
			}
			break;
		default: break;
	}

	EXIT("unknown event type\n");
}

void CommandProcessor::WriteAtEndOfPipe32(uint32_t cache_policy, uint32_t event_write_dest,
                                          uint32_t eop_event_type, uint32_t cache_action,
                                          uint32_t event_index, uint32_t event_write_source,
                                          void* dst_gpu_addr, uint32_t value,
                                          uint32_t interrupt_selector,
                                          uint32_t interrupt_context_id) {
	WriteAtEndOfPipe(cache_policy, event_write_dest, eop_event_type, cache_action, event_index,
	                 event_write_source, dst_gpu_addr, value, interrupt_selector,
	                 interrupt_context_id);
}

void CommandProcessor::WriteAtEndOfPipe64(uint32_t cache_policy, uint32_t event_write_dest,
                                          uint32_t eop_event_type, uint32_t cache_action,
                                          uint32_t event_index, uint32_t event_write_source,
                                          void* dst_gpu_addr, uint64_t value,
                                          uint32_t interrupt_selector,
                                          uint32_t interrupt_context_id) {
	WriteAtEndOfPipe(cache_policy, event_write_dest, eop_event_type, cache_action, event_index,
	                 event_write_source, dst_gpu_addr, value, interrupt_selector,
	                 interrupt_context_id);
}

void CommandProcessor::EmitGlobalBarrier() {
	Common::LockGuard lock(m_renderer.GetMutex());
	// KYTY_DEFER_GLOBAL_BARRIER (live): inside a render pass, recorded when it ends
	// (CommandBuffer::DeferGlobalBarrier) instead of ending it here.
	static auto& defer = Common::LiveSwitches::Get("KYTY_DEFER_GLOBAL_BARRIER", 0);
	if (defer.load(std::memory_order_relaxed) != 0 && CurrentBuffer().IsRendering()) {
		CurrentBuffer().DeferGlobalBarrier();
		return;
	}

	vk::MemoryBarrier2 barrier {};
	barrier.srcStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
	barrier.srcAccessMask = vk::AccessFlagBits2::eMemoryWrite;
	barrier.dstStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
	barrier.dstAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite;

	vk::DependencyInfo dependency {};
	dependency.memoryBarrierCount = 1;
	dependency.pMemoryBarriers    = &barrier;
	GetScheduler().EndRendering();
	CurrentBuffer().Recorder().pipelineBarrier2(dependency);
}

void CommandProcessor::TriggerEopEventAtEndOfPipe(uint32_t interrupt_context_id) {
	Sync::TriggerEopEventAtEndOfPipe(CurrentBuffer(), m_interrupt_event_id, interrupt_context_id);
}

void CommandProcessor::TriggerEvent(uint32_t event_type, uint32_t event_index,
                                    uint64_t event_address) {
	if (GraphicsRunDebugDumpEnabled()) {
		LOGF("CommandProcessor::TriggerEvent()\n"
		     "\t event_type  = 0x%08" PRIx32 "\n"
		     "\t event_index = 0x%08" PRIx32 "\n"
		     "\t address     = 0x%016" PRIx64 "\n",
		     event_type, event_index, event_address);
	}

	const auto valid_cache_event_index = event_index == 0x00000000 || event_index == 0x00000007;
	switch (event_type) {
		// CsPartialFlush, GsPartialFlush, PsPartialFlush.
		case 0x00000007:
		case 0x0000000f:
		case 0x00000010: EmitGlobalBarrier(); break;
		// CbDbDataWritebackInvalidate, CbDataWritebackInvalidate.
		case 0x00000016:
		case 0x00000031:
			if (!valid_cache_event_index) {
				EXIT("unknown event type: 0x%08" PRIx32 ", 0x%08" PRIx32 "\n", event_type,
				     event_index);
			}
			EmitGlobalBarrier();
			break;
		// DbDataWritebackInvalidate, DbMetadataWritebackInvalidate, CbMetadataWritebackInvalidate.
		case 0x0000002a:
		case 0x0000002c:
		case 0x0000002e:
			if (!valid_cache_event_index) {
				EXIT("unknown event type: 0x%08" PRIx32 ", 0x%08" PRIx32 "\n", event_type,
				     event_index);
			}
			EmitGlobalBarrier();
			break;
		case 0x0000000d:
		case 0x0000000e:
		case 0x00000012:
		case 0x00000017:
		case 0x00000018:
		case 0x00000019:
		case 0x0000001a:
		case 0x0000001b:
		case 0x00000038:
		case 0x0000003a:
			LOGF("\t temporary: ignoring unsupported event_write type 0x%08" PRIx32
			     ", index 0x%08" PRIx32 "\n",
			     event_type, event_index);
			break;
		case 0x00000039: {
			if (event_index != 0x00000001 || event_address == 0 || (event_address & 0x7u) != 0) {
				EXIT("invalid occlusion-counter dump: index=0x%08" PRIx32 ", address=0x%016" PRIx64
				     "\n",
				     event_index, event_address);
			}
			if (OcclusionStatsEnabled()) {
				g_occlusion_stats.dumps.fetch_add(1, std::memory_order_relaxed);
				ReportOcclusionStats();
			}
			static std::once_flag warning_once;
			std::call_once(warning_once, [] {
				std::printf("Warning: game uses occlusion queries, which are currently treated as "
				            "always visible; GPU usage may be higher and FPS may be lower.\n");
			});

			// Until host occlusion queries are implemented, publish an always-visible result. The
			// PS5 layout contains one interleaved begin/end pair per DB, and bit 63 marks a result
			// ready.
			constexpr uint64_t ready_bit    = 1ull << 63u;
			constexpr uint64_t counter_mask = ready_bit - 1u;
			auto*              results      = reinterpret_cast<volatile uint64_t*>(event_address);
			const auto         value        = ready_bit | m_synthetic_occlusion_counter;
			for (uint32_t db = 0; db < 16u; db++) {
				results[db * 2u] = value;
			}
			m_synthetic_occlusion_counter = (m_synthetic_occlusion_counter + 1u) & counter_mask;
			break;
		}
		default:
			EXIT("unknown event type: 0x%08" PRIx32 ", 0x%08" PRIx32 "\n", event_type, event_index);
	}
}

void CommandProcessor::Flip() {
	if (GraphicsRunDebugDumpEnabled()) {
		LOGF("CommandProcessor::Flip()\n");
	}

	auto& command = CurrentBuffer();
	auto request = Sync::PrepareVideoOutFlip(command, m_flip.handle, m_flip.index, m_flip.flip_mode,
	                                         m_flip.flip_arg);
	Sync::WriteAtEndOfPipeOnlyFlip(m_submit_id, command, m_flip.handle, m_flip.index,
	                               m_flip.flip_mode, m_flip.flip_arg, request);
	GetScheduler().Flush();
}

void CommandProcessor::Flip(void* dst_gpu_addr, uint32_t value) {
	auto& command = CurrentBuffer();

	if (GraphicsRunDebugDumpEnabled()) {
		LOGF("CommandProcessor::Flip()\n"
		     "\t dst_gpu_addr = 0x%016" PRIx64 "\n"
		     "\t value        = 0x%08" PRIx32 "\n",
		     reinterpret_cast<uint64_t>(dst_gpu_addr), value);
	}

	if (LabelsAfterGpu()) {
		PublishLabelAtCompletion(dst_gpu_addr, value, sizeof(value), false);
	} else {
		std::memcpy(dst_gpu_addr, &value, sizeof(value));
	}
	auto request = Sync::PrepareVideoOutFlip(command, m_flip.handle, m_flip.index, m_flip.flip_mode,
	                                         m_flip.flip_arg);
	Sync::WriteAtEndOfPipeWithFlip32(m_submit_id, command, static_cast<uint32_t*>(dst_gpu_addr),
	                                 value, m_flip.handle, m_flip.index, m_flip.flip_mode,
	                                 m_flip.flip_arg, request);
	GetScheduler().Flush();
}

void CommandProcessor::FlipWithInterrupt(uint32_t eop_event_type, uint32_t cache_action,
                                         void* dst_gpu_addr, uint32_t value) {
	auto& command = CurrentBuffer();

	if (GraphicsRunDebugDumpEnabled()) {
		LOGF("CommandProcessor::FlipWithInterrupt()\n"
		     "\t eop_event_type      = 0x%08" PRIx32 "\n"
		     "\t cache_action        = 0x%08" PRIx32 "\n"
		     "\t dst_gpu_addr        = 0x%016" PRIx64 "\n"
		     "\t value               = 0x%08" PRIx32 "\n",
		     eop_event_type, cache_action, reinterpret_cast<uint64_t>(dst_gpu_addr), value);
	}

	if (eop_event_type != 0x00000004 || cache_action != 0x00000038) {
		EXIT("unknown event type\n");
	}
	if (LabelsAfterGpu()) {
		PublishLabelAtCompletion(dst_gpu_addr, value, sizeof(value), false);
	} else {
		std::memcpy(dst_gpu_addr, &value, sizeof(value));
	}
	auto request = Sync::PrepareVideoOutFlip(command, m_flip.handle, m_flip.index, m_flip.flip_mode,
	                                         m_flip.flip_arg);
	Sync::WriteAtEndOfPipeWithInterruptWriteBackFlip32(
	    m_submit_id, command, static_cast<uint32_t*>(dst_gpu_addr), value, m_flip.handle,
	    m_flip.index, m_flip.flip_mode, m_flip.flip_arg, request, m_interrupt_event_id);
	GetScheduler().Flush();
}

void CommandProcessor::PrepareCpuFlip(uint64_t request_id) {
	auto& command = CurrentBuffer();
	if (g_current_processor != nullptr) {
		EXIT("invalid graphics-thread CPU flip preparation\n");
	}
	struct ProcessorScope {
		explicit ProcessorScope(CommandProcessor& processor) { g_current_processor = &processor; }
		~ProcessorScope() { g_current_processor = nullptr; }
	};
	ProcessorScope processor_scope(*this);

	m_renderer.GetVideoOut().PrepareFlip(request_id, command);
	GetScheduler().DeferPriorityOperation(
	    [this, request_id] { m_renderer.GetVideoOut().CompleteFlip(request_id); });
	GetScheduler().Flush();
}

void CommandProcessor::SynchronizeGpu() {
	GetScheduler().Finish();
}

void CommandProcessor::PublishLabelAtCompletion(void* dst, uint64_t value, uint32_t bytes,
                                                bool clock) {
	auto& renderer = m_renderer;
	if (!clock) {
		renderer.GetGpu().NoteRecordedLabel(reinterpret_cast<uint64_t>(dst), value, bytes,
		                                    GetScheduler().CurrentTick());
	}
	GetScheduler().DeferPriorityOperation([&renderer, dst, value, bytes, clock] {
		const uint64_t data = clock ? Sync::ReadReferenceClock() : value;
		// The backing write cannot fault into the caches from this thread.
		LibKernel::Memory::WriteBacking(reinterpret_cast<uint64_t>(dst), &data, bytes);
		renderer.GetGpu().Wake();
	});
}

// Research: a submit per EOP label let each label complete as early as possible, but Wolverine
// writes thousands per frame and vkQueueSubmit took about a quarter of Thread_Gpu. The label is
// published by the completion thread whenever its tick completes, and every processed slice ends
// with a flush (a slice blocked on WAIT_REG_MEM too), so batching cannot deadlock: it only delays
// a label by up to KYTY_LABEL_FLUSH_US after the previous submit. 0 submits on every label.
// Wolverine, standing still: 0 → 200 ms/frame, 500 → 167 ms (submits 10k/s → 1.6k/s); 250 and
// 1000 measured the same as 500, 2000 slightly slower.
void CommandProcessor::FlushForLabel() {
	static auto& interval_us = Common::LiveSwitches::Get("KYTY_LABEL_FLUSH_US", 500);
	const auto   interval    = interval_us.load(std::memory_order_relaxed);
	auto&        scheduler   = GetScheduler();
	if (interval > 0) {
		const auto now_us = std::chrono::duration_cast<std::chrono::microseconds>(
		                        std::chrono::steady_clock::now().time_since_epoch())
		                        .count();
		if (now_us - scheduler.LastSubmitUs() < interval) {
			return;
		}
	}
	scheduler.Flush();
}

bool GuestGpu::IsGpuThread() noexcept {
	return g_gpu_thread;
}

} // namespace Libs::Graphics
