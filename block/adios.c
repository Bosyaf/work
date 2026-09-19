// SPDX-License-Identifier: GPL-2.0
/*
 * Adaptive Deadline I/O Scheduler (ADIOS)
 * Copyright (C) 2025 Masahito Suzuki
 */
#include <linux/bio.h>
#include <linux/blkdev.h>
#include <linux/compiler.h>
#include <linux/fs.h>
#include <linux/init.h>
#include <linux/kernel.h>
#include <linux/math.h>
#include <linux/module.h>
#include <linux/rbtree.h>
#include <linux/sbitmap.h>
#include <linux/mempool.h>
#include <linux/slab.h>
#include <linux/timekeeping.h>
#include <linux/percpu.h>
#include <linux/string.h>
#include <linux/list_sort.h>
#include <linux/rcupdate.h>
#include <linux/ioprio.h>
#include <linux/smp.h>
#include <linux/sched.h>
#include <linux/workqueue.h>

#include "elevator.h"
#include "blk.h"
#include "blk-mq.h"
#include "blk-mq-sched.h"

#define ADIOS_VERSION "3.3.5-SAGA"

/* Request Types:
 *
 * Tier 0 (Highest Priority): Emergency & System Integrity Requests
 * -----------------------------------------------------------------
 * - Target: Requests with the BLK_MQ_INSERT_AT_HEAD flag.
 * - Purpose: For critical, non-negotiable operations such as device error
 *   recovery or flush sequences that must bypass all other scheduling logic.
 * - Implementation: Placed in a dedicated, high-priority FIFO queue
 *   (`prio_queue[0]`) for immediate dispatch.
 *
 * Tier 1 (Medium Priority): Application Responsiveness
 * ----------------------------------------------------
 * - Target: Normal synchronous requests (e.g., from standard file reads).
 * - Purpose: To ensure correct application behavior for operations that
 *   depend on sequential I/O completion (e.g., file system mounts) and to
 *   provide low latency for interactive applications.
 * - Implementation: The deadline for these requests is set to their start
 *   time (`rq->start_time_ns`). This effectively enforces FIFO-like behavior
 *   within the deadline-sorted red-black tree, preventing out-of-order
 *   execution of dependent synchronous operations.
 *
 * Tier 2 (Normal Priority): Background Throughput
 * -----------------------------------------------
 * - Target: Asynchronous requests.
 * - Purpose: To maximize disk throughput for background tasks where latency
 *   is not critical.
 * - Implementation: These are the only requests where ADIOS's adaptive
 *   latency prediction model is used. A dynamic deadline is calculated based
 *   on the predicted I/O latency, allowing for aggressive reordering to
 *   optimize I/O efficiency.
 *
 * I/O barrier guarantees (REQ_PREFLUSH / REQ_FUA) are provided entirely by
 * the block layer's flush state machine (block/blk-flush.c), which is the
 * mechanism that replaced whole-queue barriers upstream in 2.6.37. No
 * queue-wide barrier is implemented or needed here:
 * - blk_insert_flush() always strips REQ_PREFLUSH (and REQ_FUA unless the
 *   device supports it in hardware) before any request ever reaches this
 *   elevator's ->insert_requests(), and always sets REQ_SYNC on it, so such
 *   a request lands in Tier 1 (immediate deadline) like any other
 *   synchronous request -- never delayed behind Tier 2 reordering.
 * - The data portion of a request that needed an actual PREFLUSH is
 *   resubmitted at the head of the queue once the flush completes, which
 *   lands in Tier 0 here, ahead of everything else.
 * - REQ_PREFLUSH/REQ_FUA and RQF_FLUSH_SEQ requests are already excluded
 *   from bio/request merging by the generic block layer (rq_mergeable(),
 *   blk_rq_merge_ok()) before this elevator's merge callbacks ever run.
 * - The synthesized pure REQ_OP_FLUSH command itself always bypasses this
 *   elevator entirely (blk_mq_insert_request() routes it straight to
 *   hctx->dispatch), so it never needs to be recognized or handled here.
 *
 * Dispatch Logic:
 * The scheduler always dispatches requests in strict priority order:
 * 1. prio_queue[0] (Tier 0)
 * 2. The deadline-sorted batch queue (which naturally prioritizes Tier 1
 *    over Tier 2 due to their calculated deadlines).
 */

// Global variable to control the latency
// SAGA: tuned defaults, validated empirically on UFS 2.2 (see SAGA-Build
// integration notes) — was 16000000ULL / 22000000ULL upstream.
static u64 default_global_latency_window            = 30000000ULL;
static u64 default_global_latency_window_rotational = 22000000ULL;
// Ratio below which batch queues should be refilled
// SAGA: was 20 upstream.
static u8  default_bq_refill_below_ratio = 50;
// Maximum latency sample to input
static u64 default_lat_model_latency_limit = 500 * NSEC_PER_MSEC;
// SAGA: default confidence threshold -- 64 small-bucket samples per
// optype before predictive reordering is trusted for it. Chosen as a
// conservative starting point (LM_SAMPLES_THRESHOLD upstream is 1024
// for a full model *update*, but ordering decisions can reasonably
// trust a smaller running sample than a full recalculation needs).
// NOT benchmarked -- a reasonable guess, tune via lm_confidence_samples.
static u32 default_lm_confidence_samples = 64;
// SAGA: defaults for self-detected degradation (see force_shrink_model).
// 150% = actual latency 1.5x what the model predicted counts as an
// overrun; 20 consecutive overruns before forcing an out-of-band shrink.
// Neither validated by benchmark -- reasonable starting guesses.
static u32 default_lm_overrun_ratio_pct = 150;
static u32 default_lm_overrun_trigger_count = 20;
// SAGA: interactive-boost defaults. 100ms idle gap to count as "was
// idle" (roughly: if you haven't touched storage in 1/10s, your next
// read is probably a fresh user-facing action, not a continuation of
// background work). Boost of 20ms shaves off the deadline for that one
// read -- enough to jump ahead of typical background async writes
// (latency_target[write] default is 2000ms) without needing it to be
// huge. Neither validated by benchmark.
static u32 default_lm_interactive_idle_ms = 100;
static u64 default_lm_interactive_boost_ns = 20 * NSEC_PER_MSEC;
// SAGA #6: adaptive depth controller defaults. Window of 128 completions
// per evaluation (independent per optype -- read and write each get
// their own window/multiplier). >=15% bad in a window throttles down;
// >=90% good opens up; the 15-90 range is the hysteresis dead zone.
// +/-10% (100 permille) step per adjustment, multiplier bounded to
// [0.25x, 2.0x] of the statically-scaled batch_limit from #2. All
// reasoned starting points, NOT validated by benchmark.
static u32 default_lm_adapt_window_samples = 128;
// SAGA: CORRECTED defaults after real telemetry showed the original
// 90%/15% thresholds (paired with the old shared-with-#3 bound) left
// the controller stuck at 1.000x. good_ratio_pct/bad_ratio_pct are the
// NEW per-sample classification bounds (see adapt_depth_controller's
// docstring); threshold_pct below are how much of a window needs to
// land in each bucket to actually move the multiplier -- also loosened
// from the original 90/15 since requiring 90% of ANY real-world sample
// window to be strictly good is an unreasonably high bar in practice.
static u32 default_lm_adapt_good_ratio_pct = 110;  // <=110% of prediction = good
static u32 default_lm_adapt_bad_ratio_pct = 130;   // >130% of prediction = bad
static u32 default_lm_adapt_bad_threshold_pct = 20;
static u32 default_lm_adapt_good_threshold_pct = 65;
static u32 default_lm_adapt_step_permille = 50;
// CHATGPT-REVIEW: asymmetric step (was symmetric +-100 permille, i.e.
// +-10%). Now defaults to +5%/-10% -- expand cautiously, contract fast.
// Standard control-theory technique to avoid overshoot; also directly
// motivated by the empirical volatility measured this session.
static u32 default_lm_adapt_step_down_permille = 100;
static u32 default_lm_adapt_cooldown_ms = 1000;
static u32 default_lm_adapt_min_permille = 250;
static u32 default_lm_adapt_max_permille = 2000;
// Batch ordering strategy
static u64 default_batch_order = 0;
// SAGA: modest read-priority bias. Nice-scale (-20..19), matches the
// CFS prio_to_wmult table reused for dl_bias weighting. Only matters
// when reads and writes are genuinely competing for dispatch at the
// same time -- which none of today's single-optype-at-a-time synthetic
// benchmarks (CPDT, miniio) ever exercised, so this is untested by
// anything in this session's data. Reasoning: real phone usage is
// dominated by foreground reads (opening an app, loading a resource)
// contending against background writes (autosave, sync, logging) --
// biasing dispatch toward reads under that real contention is a
// scheduling-order choice on top of already-good throughput, not a
// hardware capability change, so it shouldn't cost measured
// randwrite/randread numbers (which never see contention in this
// session's tests) while plausibly helping real-world responsiveness.
// Modest value chosen deliberately -- not maxed out, since writes
// still need to make progress and starving them creates its own
// problems (stalled autosave, GC/discard backing up, etc).
static s32 default_read_priority = -3;

/* Compliance Flags:
 * 0x1: Async requests will not be reordered based on the predicted latency
 */
enum adios_compliance_flags {
	ADIOS_CF_FIXORDER  = 1U << 0,
};

// Flags to control compliance with block layer constraints
// SAGA: CORRECTED. Was ADIOS_CF_FIXORDER (predictive reordering OFF)
// based on informal testing that turned out to be pointed at the
// wrong block device (sda, which never received real I/O -- see
// adios-tunable patch history). Retested with compliance_flags=0 on
// the correct device (sdc, /data's real disk) with a model that
// actually learns: ADIOS won meaningfully in seqwrite/seqread/
// randwrite, lost only slightly in randread. Reverted to upstream's
// own default (0x0, predictive reordering ON) based on that result.
// NOTE: comparison was against only 1 round at compliance_flags=0
// before Termux's scheduler-switch behavior reset it back to default
// -- directionally solid, but not the same 5-round protocol used for
// other conclusions in this series. Worth a fuller multi-round
// re-confirmation later if time allows.
static u64 default_compliance_flags = 0x0;

// Dynamic thresholds for shrinkage
static u32 default_lm_shrink_at_kreqs  =  5000;
static u32 default_lm_shrink_at_gbytes =    50;
static u32 default_lm_shrink_resist    =     2;

enum adios_optype {
	ADIOS_READ    = 0,
	ADIOS_WRITE   = 1,
	ADIOS_DISCARD = 2,
	ADIOS_OTHER   = 3,
	ADIOS_OPTYPES = 4,
};

// Latency targets for each operation type
/*
 * Starts at 0 (disabled). There's no meaningful "top-app" this early,
 * so activating immediately only ever risks catching boot-time
 * daemons that happen to share an elevated ioprio. Self-activates via
 * topapp_activate_timer below, topapp_activation_delay_ms after this
 * scheduler attaches to its queue -- fully in-kernel, no userspace
 * signal, no init.rc, no module needed.
 */
static u64 default_topapp_deadline_bonus;
static u64 default_topapp_deadline_bonus_target = 4 * NSEC_PER_MSEC;
static u64 default_topapp_activation_delay_ms = 90000;
static u64 default_latency_target[ADIOS_OPTYPES] = {
	[ADIOS_READ]    =     2ULL * NSEC_PER_MSEC,
	[ADIOS_WRITE]   =  2000ULL * NSEC_PER_MSEC,
	[ADIOS_DISCARD] =  8000ULL * NSEC_PER_MSEC,
	[ADIOS_OTHER]   =     0ULL * NSEC_PER_MSEC,
};

// Maximum batch size limits for each operation type
// SAGA: tuned defaults — was 36/72 upstream.
static u32 default_batch_limit[ADIOS_OPTYPES] = {
	[ADIOS_READ]    = 64,
	[ADIOS_WRITE]   = 128,
	// SAGA: modest bump from upstream's 1. This device's f2fs mount
	// uses real-time small-unit discard (mount opts include "discard,
	// discard_unit=block", confirmed via /proc/mounts on-device) --
	// background GC/discard traffic is live during normal use, not
	// just at unmount. batch_limit=1 means each discard is dispatched
	// almost synchronously; a small amount of batching room lets a
	// few coalesce instead of each one individually contending with
	// concurrent random writes. Kept deliberately small (discards
	// aren't latency-critical the way reads are, and this is NOT
	// auto-scaled by queue depth like read/write -- see the fix in
	// init_sched below).
	[ADIOS_DISCARD] =  4,
	[ADIOS_OTHER]   =  1,
};

enum adios_batch_order {
	ADIOS_BO_OPTYPE   = 0,
	ADIOS_BO_ELEVATOR = 1,
};

// Thresholds for latency model control
// NOTE: block size / outlier / interval thresholds were hardcoded #defines
// upstream (ADIOS 3.2.0). They are now per-model runtime fields, tunable
// via sysfs, defaulting to the original upstream values below.
#define LM_BLOCK_SIZE_THRESHOLD_DEFAULT 4096
#define LM_SAMPLES_THRESHOLD_DEFAULT    1024
#define LM_INTERVAL_THRESHOLD_DEFAULT   1500
#define LM_OUTLIER_PERCENTILE_DEFAULT     99
#define LM_LAT_BUCKET_COUNT       64

#define ADIOS_PQ_LEVELS 2
#define ADIOS_DL_TYPES  2
#define ADIOS_BQ_PAGES  2
// SAGA: fixed-size recency table for the interactive-boost heuristic.
// Small on purpose -- this is a hint mechanism, not a per-process
// accounting structure. Collisions just mean two recently-active tasks
// share a slot and both look "not idle" to each other; worst case is a
// missed boost, never incorrect ordering or a correctness issue.
// CHATGPT-REVIEW: bumped 16 -> 32. Real Android easily has more than
// 16 concurrently-relevant TGIDs (zygote, system_server,
// surfaceflinger, launcher, foreground app, GMS, media, vendor
// services...), so 16 slots evict often on a real device -- an evicted
// task's next read looks like "first sight" and gets boosted again,
// even though the mechanism is meant to reward genuine idle-then-active
// transitions. Doubling the table halves collision frequency for a
// small, fixed cost (still O(32) linear scan, still under a trylock --
// see task_was_idle_and_gets_boost). A full hash+probing redesign
// would scale better but is more invasive than this fix is urgent;
// left for a future revision.
#define ADIOS_IACTV_SLOTS 32

static u32 default_dl_prio[ADIOS_DL_TYPES] = {8, 0};

// Bit flags for the atomic state variable, indicating which queues have requests.
enum adios_state_flags {
	ADIOS_STATE_PQ_0      = 1U << 0,
	ADIOS_STATE_PQ_1      = 1U << 1,
	ADIOS_STATE_DL_0      = 1U << 2,
	ADIOS_STATE_DL_1      = 1U << 3,
	ADIOS_STATE_BQ_PAGE_0 = 1U << 4,
	ADIOS_STATE_BQ_PAGE_1 = 1U << 5,
};
#define ADIOS_STATE_PQ 0
#define ADIOS_STATE_DL 2
#define ADIOS_STATE_BQ 4

// Temporal granularity of the deadline tree node (dl_group)
#define ADIOS_QUANTUM_SHIFT 20

#define ADIOS_MAX_INSERTS_PER_LOCK 72
#define ADIOS_MAX_DELETES_PER_LOCK 24

// Structure to hold latency bucket data for small requests
struct latency_bucket_small {
	u64 weighted_sum_latency;
	u64 sum_of_weights;
};

// Structure to hold latency bucket data for large requests
struct latency_bucket_large {
	u64 weighted_sum_latency;
	u64 weighted_sum_block_size;
	u64 sum_of_weights;
};

// Structure to hold per-cpu buckets, improving data locality and code clarity.
struct lm_buckets {
	struct latency_bucket_small small_bucket[LM_LAT_BUCKET_COUNT];
	struct latency_bucket_large large_bucket[LM_LAT_BUCKET_COUNT];
};

// Structure to hold RCU-protected latency model parameters
struct latency_model_params {
	u64 base;
	u64 slope;
	u64 small_sum_delay;
	u64 small_count;
	u64 large_sum_delay;
	u64 large_sum_bsize;
	// CHATGPT-REVIEW FIX #2: was missing -- model_is_confident() had
	// no way to gate confidence for the LARGE bucket (only small_count
	// existed), so any workload dominated by large I/O (file copies,
	// APK installs, sequential restores) never built confidence and
	// predictive reordering stayed off for it indefinitely, even after
	// thousands of large-request completions. See model_is_confident().
	u64 large_count;
	u64 last_update_jiffies;
	struct rcu_head rcu;
};

// Structure to hold the latency model context data
struct latency_model {
	spinlock_t update_lock;
	struct latency_model_params __rcu *params;

	// Per-CPU buckets to avoid lock contention on the completion path
	struct lm_buckets __percpu *pcpu_buckets;
	// Per-CPU snapshots for delta-based aggregation (accessed under update_lock)
	struct lm_buckets __percpu *pcpu_snapshot;

	u32 lm_shrink_at_kreqs;
	u32 lm_shrink_at_gbytes;
	u8  lm_shrink_resist;

	// SAGA: self-detected degradation counter (see force_shrink_model
	// and its use in adios_completed_request). Counts consecutive
	// completions whose actual latency overran the model's own
	// prediction by more than lm_overrun_ratio_pct.
	atomic_t consec_overruns;
	// CHATGPT-REVIEW DEFERRED FIX, now applied: force_shrink_model()
	// allocates (kmemdup, GFP_ATOMIC) and does an RCU swap -- real work,
	// previously done directly inline from adios_completed_request()
	// under a spinlock, on the exact I/O completion path this whole
	// patch series is trying to keep fast. Deferred to the existing
	// update_timer (already fires within 100ms of any completion via
	// timer_reduce below) instead: completion just sets this flag,
	// the timer callback does the actual work in a context that isn't
	// the hot path.
	atomic_t shrink_pending;

	// Formerly hardcoded #defines, now tunable at runtime (see sysfs below)
	u32 lm_block_size_threshold;   // bytes; small vs large bucket boundary
	u8  lm_outlier_percentile;     // 1-100; percentile used to cut outliers
	u32 lm_interval_threshold_ms;  // ms; min interval between forced updates
	u32 lm_samples_threshold;      // sample count that forces an update
};

struct adios_pcpu_completion {
	u64      last_completed_time;
	sector_t last_completed_pos;
};

union adios_in_flight_rqs {
	atomic64_t	atomic;
	u64			scalar;
	struct {
		u64 	count:          16;
		u64 	total_pred_lat: 48;
	};
};

// Adios scheduler data
struct adios_data {
	spinlock_t pq_lock;
	struct list_head prio_queue[2];

	struct rb_root_cached dl_tree[2];
	spinlock_t lock;
	s64 dl_bias;
	s32 dl_prio[2];

	atomic_t state;
	u8  bq_state[ADIOS_BQ_PAGES];

	bool models_stable;

	u64 global_latency_window;
	u64 compliance_flags;
	u64 latency_target[ADIOS_OPTYPES];
	u64 topapp_deadline_bonus;
	u64 topapp_deadline_bonus_target;
	u64 topapp_activation_delay_ms;
	struct timer_list topapp_activate_timer;
	u32 batch_limit[ADIOS_OPTYPES];
	u32 batch_actual_max_size[ADIOS_OPTYPES];
	u32 batch_actual_max_total;
	u32 async_depth;
	u32 lat_model_latency_limit;
	// SAGA: confidence-gated predictive reordering. Below this many
	// accumulated small-bucket samples for a given optype's model,
	// predictive reordering auto-falls-back to FIXORDER-like behavior
	// for THAT optype only, regardless of the global compliance_flags
	// switch. See model_is_confident() and its use in add_to_dl_tree().
	u32 lm_confidence_samples;
	// SAGA: self-detected degradation thresholds (see force_shrink_model).
	u32 lm_overrun_ratio_pct;    // % of pred_lat that counts as "overrun"
	u32 lm_overrun_trigger_count; // consecutive overruns before forced shrink

	// SAGA: BFQ-inspired interactive-boost heuristic, scoped down.
	// BFQ's core insight for desktop/mobile responsiveness: a task that
	// was idle and just issued I/O is likely interactive (foreground
	// touch response, app cold-start reading resources) and deserves a
	// temporary priority bump. Full BFQ implements this via per-process
	// fair queuing (thousands of lines); this is the heuristic alone,
	// as a small fixed-size recency table -- not a queuing redesign.
	spinlock_t iactv_lock;
	struct {
		pid_t tgid;
		u64   last_seen_ns;
	} iactv_table[ADIOS_IACTV_SLOTS];
	u32 lm_interactive_idle_ms;   // gap since last I/O to count as "was idle"
	u64 lm_interactive_boost_ns;  // deadline reduction applied on boost

	// SAGA #6: Kyber-inspired adaptive depth controller. Unlike Kyber,
	// which compares against a fixed configured latency target, this
	// compares each completion against the model's OWN prediction
	// (rd->pred_lat) -- inheriting the model's per-device/per-workload
	// self-calibration instead of needing a second, separately-tuned
	// target. Only READ and WRITE are controlled (mirrors Kyber's
	// practical focus on those two domains); DISCARD/OTHER always run
	// at their statically-scaled batch_limit (from #2).
	// CHATGPT-REVIEW FIX #1 v2 (residual race after the first fix):
	// the cmpxchg guard added below only protected the evaluate+reset
	// step itself -- but atomic_inc(window_good/bad) happens BEFORE
	// that guard, so a writer on another CPU could still land a sample
	// in window_good/bad in the exact instant between another CPU's
	// atomic_xchg-reset of those counters and its atomic_set of
	// window_total, corrupting the count for one window without
	// double-applying an adjustment (the narrower bug the first fix
	// left behind). Switched to a double-buffered design: writers pick
	// whichever buffer window_buf_idx currently points to; the
	// evaluator atomically swaps the index (via cmpxchg) BEFORE
	// reading/resetting the old buffer, so new writers stop landing in
	// it from that instant on. One narrow residual race remains (a
	// writer that already read the old index microseconds before the
	// swap can still land one sample there) -- accepted as a
	// single-sample-per-window skew at most, standard tradeoff for
	// lock-free double-buffered kernel stats vs. a full seqlock.
	atomic_t window_good[2][ADIOS_OPTYPES];
	atomic_t window_bad[2][ADIOS_OPTYPES];
	atomic_t window_total[2][ADIOS_OPTYPES];
	atomic_t window_buf_idx[ADIOS_OPTYPES]; // which buffer (0/1) writers use now
	atomic_t depth_multiplier_permille[ADIOS_OPTYPES]; // 1000 == 1.0x
	// CHATGPT-REVIEW FIX (cooldown + asymmetric step): prevents the
	// multiplier from re-adjusting more than once per cooldown period,
	// and steps up more slowly than it steps down -- standard control-
	// theory technique to avoid overshoot/oscillation. Also directly
	// motivated by the same empirical volatility above.
	unsigned long last_adjust_jiffies[ADIOS_OPTYPES];
	u32 lm_adapt_window_samples;   // completions per evaluation window
	u32 lm_adapt_bad_threshold_pct;  // >= this %% bad -> throttle down
	// SAGA: decoupled from #3's lm_overrun_ratio_pct on purpose -- #3
	// stays conservative (rare, discards real history when it fires);
	// #6 needs its own, more sensitive classification bar to ever
	// actually move (see commit notes: 90%/150% reusing #3's threshold
	// left the controller permanently stuck at 1.000x in real testing).
	u32 lm_adapt_good_ratio_pct;   // latency <= pred_lat * this/100 -> good
	u32 lm_adapt_bad_ratio_pct;    // latency >  pred_lat * this/100 -> bad
	u32 lm_adapt_good_threshold_pct; // >= this %% good -> open up
	u32 lm_adapt_step_permille;    // adjustment step per evaluation (up)
	u32 lm_adapt_step_down_permille; // adjustment step when throttling down
	u32 lm_adapt_cooldown_ms;      // min time between adjustments, per optype
	u32 lm_adapt_min_permille;     // floor for the multiplier
	u32 lm_adapt_max_permille;     // ceiling for the multiplier
	// SAGA FIX (3rd review round): SYSFS_INT_DECL_PAIRED's cross-check
	// (see below) reads the paired field, then writes its own -- two
	// concurrent sysfs writers (one to good_ratio_pct, one to
	// bad_ratio_pct, say) can each read the OTHER's pre-write value,
	// both pass their own check, and still leave an invalid combo
	// (e.g. good >= bad) once both stores land. A plain spinlock
	// around the whole check+write closes this: sysfs writes are rare
	// admin actions, never on any hot path, so lock hold time here is
	// irrelevant to performance.
	spinlock_t lm_adapt_bounds_lock;

	u8  bq_refill_below_ratio;
	u8  is_rotational;
	u8  batch_order;
	u8  elv_direction;
	sector_t head_pos;

	bool bq_page;
	struct list_head batch_queue[ADIOS_BQ_PAGES][ADIOS_OPTYPES];
	u32 batch_count[ADIOS_BQ_PAGES][ADIOS_OPTYPES];
	u8  bq_batch_order[ADIOS_BQ_PAGES];
	spinlock_t bq_lock;

	struct lm_buckets *aggr_buckets;

	struct latency_model latency_model[ADIOS_OPTYPES];
	struct timer_list update_timer;
	// CHATGPT-REVIEW FIX #4: force_shrink_model() was deferred here
	// from the completion path to this timer's callback in the
	// previous revision -- but timer callbacks run in softirq context,
	// which still only allows GFP_ATOMIC (no sleeping). A workqueue
	// runs in real process context, allowing GFP_KERNEL -- more likely
	// to succeed under memory pressure, which is exactly when a shrink
	// is most likely to be needed. The timer now just schedules this
	// work instead of calling force_shrink_model() directly.
	struct work_struct shrink_work;

	union adios_in_flight_rqs in_flight_rqs;
	atomic64_t total_pred_lat;

	struct adios_pcpu_completion __percpu *pcpu_completion;

	struct kmem_cache *rq_data_cache;
	mempool_t *rq_data_pool;
	struct kmem_cache *dl_group_pool;

	struct request_queue *queue;
};

// List of requests with the same deadline in the deadline-sorted tree
struct dl_group {
	struct rb_node node;
	struct list_head rqs;
	u64 deadline;
} __attribute__((aligned(64)));

// Structure to hold scheduler-specific data for each request
struct adios_rq_data {
	struct list_head *dl_group;
	struct list_head dl_node;

	struct request *rq;
	u64 deadline;
	u64 pred_lat;
	u32 block_size;
	bool managed;
	/*
	 * REVIEW FIX: fill_batch_queues() used to infer sync/async by
	 * comparing rd->deadline against rq->start_time_ns -- but
	 * interactive-boost and/or the top-app deadline bonus both
	 * subtract from rd->deadline for genuinely sync requests too, so a
	 * boosted sync read ends up with rd->deadline != start_time_ns and
	 * gets misclassified as async (dest_idx=1, the "can be delayed
	 * freely" tier) -- the exact opposite of what the boost was for.
	 * Ground truth recorded explicitly here instead, set once in
	 * add_to_dl_tree() from rq->cmd_flags & REQ_SYNC before anything
	 * touches rd->deadline.
	 */
	bool is_sync;
} __attribute__((aligned(64)));

static const int adios_prio_to_wmult[40] = {
 /* -20 */     88761,     71755,     56483,     46273,     36291,
 /* -15 */     29154,     23254,     18705,     14949,     11916,
 /* -10 */      9548,      7620,      6100,      4904,      3906,
 /*  -5 */      3121,      2501,      1991,      1586,      1277,
 /*   0 */      1024,       820,       655,       526,       423,
 /*   5 */       335,       272,       215,       172,       137,
 /*  10 */       110,        87,        70,        56,        45,
 /*  15 */        36,        29,        23,        18,        15,
};

static inline bool compliant(struct adios_data *ad, u32 flag) {
	return ad->compliance_flags & flag;
}

// Count the number of entries in aggregated small buckets
static u64 lm_count_small_entries(struct latency_bucket_small *buckets) {
	u64 total_weight = 0;
	for (u8 i = 0; i < LM_LAT_BUCKET_COUNT; i++)
		total_weight += buckets[i].sum_of_weights;
	return total_weight;
}

// Update the small buckets in the latency model from aggregated data
static bool lm_update_small_buckets(struct latency_model *model,
		struct latency_model_params *params,
		struct latency_bucket_small *buckets,
		u64 total_weight, bool count_all) {
	u64 sum_latency = 0;
	u64 sum_weight = 0;
	u64 cumulative_weight = 0, threshold_weight = 0;
	u8  outlier_threshold_bucket = 0;
	u8  outlier_percentile = model->lm_outlier_percentile;
	u8  reduction;

	if (count_all)
		outlier_percentile = 100;

	// Calculate the threshold weight for outlier detection
	threshold_weight = (total_weight * outlier_percentile) / 100;

	// Identify the bucket that corresponds to the outlier threshold
	for (u8 i = 0; i < LM_LAT_BUCKET_COUNT; i++) {
		cumulative_weight += buckets[i].sum_of_weights;
		if (cumulative_weight >= threshold_weight) {
			outlier_threshold_bucket = i;
			break;
		}
	}

	// Calculate the average latency, excluding outliers
	for (u8 i = 0; i <= outlier_threshold_bucket; i++) {
		struct latency_bucket_small *bucket = &buckets[i];
		if (i < outlier_threshold_bucket) {
			sum_latency += bucket->weighted_sum_latency;
			sum_weight += bucket->sum_of_weights;
		} else {
			// The threshold bucket's contribution is proportional
			u64 remaining_weight =
				threshold_weight - (cumulative_weight - bucket->sum_of_weights);
			if (bucket->sum_of_weights > 0) {
				sum_latency += div_u64(bucket->weighted_sum_latency *
					remaining_weight, bucket->sum_of_weights);
				sum_weight += remaining_weight;
			}
		}
	}

	// Shrink the model if it reaches at the readjustment threshold
	if (params->small_count >= 1000ULL * model->lm_shrink_at_kreqs) {
		reduction = model->lm_shrink_resist;
		if (params->small_count >> reduction) {
			params->small_sum_delay -= params->small_sum_delay >> reduction;
			params->small_count     -= params->small_count     >> reduction;
		}
	}

	if (!sum_weight)
		return false;

	// Accumulate the average latency into the statistics
	params->small_sum_delay += sum_latency;
	params->small_count     += sum_weight;

	return true;
}

// Count the number of entries in aggregated large buckets
static u64 lm_count_large_entries(struct latency_bucket_large *buckets) {
	u64 total_weight = 0;
	for (u8 i = 0; i < LM_LAT_BUCKET_COUNT; i++)
		total_weight += buckets[i].sum_of_weights;
	return total_weight;
}

// Update the large buckets in the latency model from aggregated data
static bool lm_update_large_buckets(struct latency_model *model,
		struct latency_model_params *params,
		struct latency_bucket_large *buckets,
		u64 total_weight, bool count_all) {
	s64 sum_latency = 0;
	u64 sum_block_size = 0, intercept;
	u64 cumulative_weight = 0, threshold_weight = 0;
	u64 sum_weight = 0;
	u8  outlier_threshold_bucket = 0;
	u8  outlier_percentile = model->lm_outlier_percentile;
	u8  reduction;

	if (count_all)
		outlier_percentile = 100;

	// Calculate the threshold weight for outlier detection
	threshold_weight = (total_weight * outlier_percentile) / 100;

	// Identify the bucket that corresponds to the outlier threshold
	for (u8 i = 0; i < LM_LAT_BUCKET_COUNT; i++) {
		cumulative_weight += buckets[i].sum_of_weights;
		if (cumulative_weight >= threshold_weight) {
			outlier_threshold_bucket = i;
			break;
		}
	}

	// Calculate the average latency and block size, excluding outliers
	for (u8 i = 0; i <= outlier_threshold_bucket; i++) {
		struct latency_bucket_large *bucket = &buckets[i];
		if (i < outlier_threshold_bucket) {
			sum_latency += bucket->weighted_sum_latency;
			sum_block_size += bucket->weighted_sum_block_size;
			sum_weight += bucket->sum_of_weights;
		} else {
			// The threshold bucket's contribution is proportional
			u64 remaining_weight =
				threshold_weight - (cumulative_weight - bucket->sum_of_weights);
			if (bucket->sum_of_weights > 0) {
				sum_latency += div_u64(bucket->weighted_sum_latency *
					remaining_weight, bucket->sum_of_weights);
				sum_block_size += div_u64(bucket->weighted_sum_block_size *
					remaining_weight, bucket->sum_of_weights);
				sum_weight += remaining_weight;
			}
		}
	}

	if (!sum_weight)
		return false;

	// Shrink the model if it reaches at the readjustment threshold
	if (params->large_sum_bsize >= 0x40000000ULL * model->lm_shrink_at_gbytes) {
		reduction = model->lm_shrink_resist;
		if (params->large_sum_bsize >> reduction) {
			params->large_sum_delay -= params->large_sum_delay >> reduction;
			params->large_sum_bsize -= params->large_sum_bsize >> reduction;
			params->large_count     -= params->large_count     >> reduction;
		}
	}

	// Accumulate the average delay into the statistics
	intercept = params->base;
	if (sum_latency > intercept)
		sum_latency -= intercept;

	params->large_sum_delay += sum_latency;
	params->large_sum_bsize += sum_block_size;
	params->large_count     += sum_weight;

	return true;
}

static void reset_buckets(struct lm_buckets *buckets)
{ memset(buckets, 0, sizeof(*buckets)); }

/*
 * REVIEW FIX (per-CPU reset race): latency_model_input() -- the hot-path
 * writer -- only ever protects its own per-CPU bucket with
 * local_irq_save() on the CPU it's running on; it never takes
 * model->update_lock. A raw for_each_possible_cpu() + memset() here (as
 * before) can therefore race a live writer on another CPU while that
 * CPU is mid read-modify-write into the exact same bucket, corrupting
 * sum_of_weights/weighted_sum_latency (and, on the next delta-vs-
 * snapshot computation, potentially underflowing a u64 by a large
 * margin). Only ever called from the rare, explicit reset path
 * (sideload_latency_model(), sysfs-triggered) -- never per completion
 * -- so it can afford real cross-CPU exclusion without touching the
 * hot path's cost at all. on_each_cpu() runs the reset on each ONLINE
 * cpu with that cpu's local irqs disabled (SMP call-function context),
 * the same discipline the writer already uses on itself, so the two
 * can never interleave on the same CPU. CPUs that are offline can't be
 * running the writer either, so resetting their (still-allocated)
 * percpu storage without IPI'ing them is safe by construction.
 */
static void lm_reset_one_cpu_buckets(void *info)
{
	struct latency_model *model = info;

	reset_buckets(this_cpu_ptr(model->pcpu_buckets));
	reset_buckets(this_cpu_ptr(model->pcpu_snapshot));
}

static void lm_reset_pcpu_buckets(struct latency_model *model) {
	on_each_cpu(lm_reset_one_cpu_buckets, model, 1);
}

// Update the latency model parameters and statistics
static void latency_model_update(
		struct adios_data *ad, struct latency_model *model) {
	u64 now;
	u64 small_weight, large_weight;
	bool time_elapsed;
	bool small_processed = false, large_processed = false;
	struct lm_buckets *aggr = ad->aggr_buckets;
	struct latency_bucket_small *asb;
	struct latency_bucket_large *alb;
	struct lm_buckets *pcpu_b, *snap;
	unsigned long flags;
	int cpu;
	struct latency_model_params *old_params, *new_params;

	spin_lock_irqsave(&model->update_lock, flags);

	old_params = rcu_dereference_protected(model->params,
				lockdep_is_held(&model->update_lock));
	new_params = kmemdup(old_params, sizeof(*new_params), GFP_ATOMIC);
	if (!new_params) {
		spin_unlock_irqrestore(&model->update_lock, flags);
		return;
	}

	// Aggregate deltas from all CPUs using snapshot-delta method.
	// Per-CPU counters increase monotonically; we compute delta = current - snapshot.
	for_each_possible_cpu(cpu) {
		pcpu_b = per_cpu_ptr(model->pcpu_buckets, cpu);
		snap   = per_cpu_ptr(model->pcpu_snapshot, cpu);

		for (u8 i = 0; i < LM_LAT_BUCKET_COUNT; i++) {
			u64 sw  = pcpu_b->small_bucket[i].sum_of_weights;
			u64 sl  = pcpu_b->small_bucket[i].weighted_sum_latency;
			u64 dsw = sw - snap->small_bucket[i].sum_of_weights;
			u64 dsl = sl - snap->small_bucket[i].weighted_sum_latency;
			snap->small_bucket[i].sum_of_weights = sw;
			snap->small_bucket[i].weighted_sum_latency = sl;
			if (dsw) {
				asb = &aggr->small_bucket[i];
				asb->sum_of_weights += dsw;
				asb->weighted_sum_latency += dsl;
			}

			u64 lw  = pcpu_b->large_bucket[i].sum_of_weights;
			u64 ll  = pcpu_b->large_bucket[i].weighted_sum_latency;
			u64 lb  = pcpu_b->large_bucket[i].weighted_sum_block_size;
			u64 dlw = lw - snap->large_bucket[i].sum_of_weights;
			u64 dll = ll - snap->large_bucket[i].weighted_sum_latency;
			u64 dlb = lb - snap->large_bucket[i].weighted_sum_block_size;
			snap->large_bucket[i].sum_of_weights = lw;
			snap->large_bucket[i].weighted_sum_latency = ll;
			snap->large_bucket[i].weighted_sum_block_size = lb;
			if (dlw) {
				alb = &aggr->large_bucket[i];
				alb->sum_of_weights += dlw;
				alb->weighted_sum_latency += dll;
				alb->weighted_sum_block_size += dlb;
			}
		}
	}

	// Count the number of entries in aggregated buckets
	small_weight = lm_count_small_entries(aggr->small_bucket);
	large_weight = lm_count_large_entries(aggr->large_bucket);

	// Whether enough time has elapsed since the last update
	now = jiffies;
	time_elapsed = unlikely(!new_params->base) ||
		new_params->last_update_jiffies +
		msecs_to_jiffies(model->lm_interval_threshold_ms) <= now;

	// Update small buckets
	if (small_weight && (time_elapsed ||
			model->lm_samples_threshold <= small_weight || !new_params->base)) {
		small_processed = lm_update_small_buckets(model, new_params,
			aggr->small_bucket, small_weight, !new_params->base);
		memset(&aggr->small_bucket[0], 0, sizeof(aggr->small_bucket));
	}
	// Update large buckets
	if (large_weight && (time_elapsed ||
			model->lm_samples_threshold <= large_weight || !new_params->slope)) {
		large_processed = lm_update_large_buckets(model, new_params,
			aggr->large_bucket, large_weight, !new_params->slope);
		memset(&aggr->large_bucket[0], 0, sizeof(aggr->large_bucket));
	}

	// Update the base parameter if small bucket was processed
	if (small_processed && likely(new_params->small_count))
		new_params->base = div_u64(new_params->small_sum_delay,
			new_params->small_count);

	// Update the slope parameter if large bucket was processed
	if (large_processed && likely(new_params->large_sum_bsize))
		new_params->slope = div_u64(new_params->large_sum_delay,
			DIV_ROUND_UP_ULL(new_params->large_sum_bsize, 1024));

	// Update last updated jiffies if update happened or time has elapsed
	if (small_processed || large_processed || time_elapsed)
		new_params->last_update_jiffies = now;

	rcu_assign_pointer(model->params, new_params);
	spin_unlock_irqrestore(&model->update_lock, flags);

	kfree_rcu(old_params, rcu);
}

// Determine the bucket index for a given measured and predicted latency
static u8 lm_input_bucket_index(u64 measured, u64 predicted) {
	u32 bucket_index;

	if (measured < predicted * 2)
		bucket_index = div_u64((measured * 20), predicted);
	else if (measured < predicted * 5)
		bucket_index = div_u64((measured * 10), predicted) + 20;
	else
		bucket_index = div_u64((measured * 3), predicted) + 40;

	if (bucket_index >= LM_LAT_BUCKET_COUNT)
		bucket_index = LM_LAT_BUCKET_COUNT - 1;

	return (u8)bucket_index;
}

// Input latency data into the latency model
static void latency_model_input(struct adios_data *ad,
		struct latency_model *model,
		u32 block_size, u64 latency, u64 pred_lat, u32 weight) {
	unsigned long flags;
	u8 bucket_index;
	struct lm_buckets *buckets;
	u64 current_base;
	struct latency_model_params *params;

	local_irq_save(flags);
	buckets = per_cpu_ptr(model->pcpu_buckets, __smp_processor_id());

	rcu_read_lock();
	params = rcu_dereference(model->params);
	current_base = params->base;
	rcu_read_unlock();

	if (block_size <= READ_ONCE(model->lm_block_size_threshold)) {
		// Handle small requests
		bucket_index = lm_input_bucket_index(latency, current_base ?: 1);

		buckets->small_bucket[bucket_index].sum_of_weights += weight;
		buckets->small_bucket[bucket_index].weighted_sum_latency +=
			latency * weight;

		local_irq_restore(flags);

		if (unlikely(!current_base)) {
			latency_model_update(ad, model);
			return;
		}
	} else {
		// Handle large requests
		if (!current_base || !pred_lat) {
			local_irq_restore(flags);
			return;
		}

		bucket_index = lm_input_bucket_index(latency, pred_lat);

		buckets->large_bucket[bucket_index].sum_of_weights += weight;
		buckets->large_bucket[bucket_index].weighted_sum_latency +=
			latency * weight;
		buckets->large_bucket[bucket_index].weighted_sum_block_size +=
			block_size * weight;

		local_irq_restore(flags);
	}
}

// Predict the latency for a given block size using the latency model
static u64 latency_model_predict(struct latency_model *model, u32 block_size) {
	u64 result;
	struct latency_model_params *params;

	rcu_read_lock();
	params = rcu_dereference(model->params);

	result = params->base;
	u32 bs_threshold = READ_ONCE(model->lm_block_size_threshold);
	if (block_size > bs_threshold)
		result += params->slope *
			DIV_ROUND_UP_ULL(block_size - bs_threshold, 1024);

	rcu_read_unlock();

	return result;
}

// Determine the type of operation based on request flags
static u8 adios_optype(struct request *rq) {
	switch (rq->cmd_flags & REQ_OP_MASK) {
	case REQ_OP_READ:
		return ADIOS_READ;
	case REQ_OP_WRITE:
		return ADIOS_WRITE;
	case REQ_OP_DISCARD:
		return ADIOS_DISCARD;
	default:
		return ADIOS_OTHER;
	}
}

static inline u8 adios_optype_not_read(struct request *rq) {
	return (rq->cmd_flags & REQ_OP_MASK) != REQ_OP_READ;
}

// Helper function to retrieve adios_rq_data from a request
static inline struct adios_rq_data *get_rq_data(struct request *rq) {
	return rq->elv.priv[0];
}

static inline
void set_adios_state(struct adios_data *ad, u32 shift, u32 idx, bool flag) {
	if (flag)
		atomic_or(1U << (idx + shift), &ad->state);
	else
		atomic_andnot(1U << (idx + shift), &ad->state);
}

static inline u32 get_adios_state(struct adios_data *ad)
{ return atomic_read(&ad->state); }

static inline u32 eval_this_adios_state(u32 state, u32 shift)
{ return (state >> shift) & 0x3; }

static inline u32 eval_adios_state(struct adios_data *ad, u32 shift)
{ return eval_this_adios_state(get_adios_state(ad), shift); }

// SAGA: confidence gate for predictive reordering. Reads the current
// small_count (exact, already tracked upstream in latency_model_params)
// for the given optype's model via RCU and compares against the
// configured threshold. Returns true once the model has absorbed enough
// real samples for its prediction to be trusted for ordering decisions.
// This is the automatic, per-optype counterpart to the manual, global
// compliance_flags/FIXORDER switch: instead of the operator guessing
// when the model is "warm enough", each optype gates itself based on
// what it has actually learned so far.
// CHATGPT-REVIEW FIX #2: was small_count-only, meaning any workload
// dominated by large I/O (file copies, APK installs, sequential
// restores -- exactly the case a review of this patch flagged) could
// pass thousands of large-bucket completions through the model and
// STILL never be "confident", because small_count (an unrelated
// bucket) stayed near zero. Now checks the SAME bucket the request's
// own block_size will actually use for prediction (see
// latency_model_predict's identical size check), using the newly
// added large_count field.
static inline bool model_is_confident(struct adios_data *ad, u8 optype,
		u64 block_size) {
	struct latency_model *model = &ad->latency_model[optype];
	struct latency_model_params *params;
	u64 count;

	rcu_read_lock();
	params = rcu_dereference(model->params);
	count = (block_size <= READ_ONCE(model->lm_block_size_threshold)) ?
		params->small_count : params->large_count;
	rcu_read_unlock();

	return count >= READ_ONCE(ad->lm_confidence_samples);
}

// SAGA #6: effective admission limit for this optype right now --
// the statically-scaled batch_limit[] (base, from #2, sysfs-visible)
// times the live depth_multiplier_permille adjusted by the controller
// below. batch_limit[] itself is left untouched by the controller so
// a manual sysfs write to it always means what it says; the multiplier
// is the only thing the feedback loop moves.
static inline u32 effective_batch_limit(struct adios_data *ad, u8 optype) {
	u32 base = ad->batch_limit[optype];
	s32 mult = atomic_read(&ad->depth_multiplier_permille[optype]);
	u64 effective; // CHATGPT-REVIEW: u32*u32 could theoretically overflow
	               // if batch_limit were set to an extreme value via
	               // sysfs combined with a high multiplier -- unlikely
	               // in practice but free to do correctly.

	if (optype != ADIOS_READ && optype != ADIOS_WRITE)
		return base; // controller only manages read/write, see struct comment
	effective = ((u64)base * (u32)mult) / 1000;
	return (u32)max_t(u64, 1, effective);
}

// SAGA #6: windowed good/bad classification + hysteresis adjustment,
// the actual control loop. Called from adios_completed_request() for
// every completed READ/WRITE.
//
// CORRECTED after real telemetry: the first version reused #3's
// overrun definition (150% bound) for BAD and "latency <= pred_lat"
// (i.e. 100%) for GOOD. On the real device, depth_controller sat
// permanently at 1.000x across multiple full CPDT runs -- a
// well-calibrated regression naturally has ~50% of samples land above
// its own prediction, so a 100%-bound GOOD bar rarely reached the 90%
// threshold, and 150%-overrun BAD events are naturally rare too. The
// controller had no real chance to move. Now uses its OWN, more
// forgiving bounds (lm_adapt_good_ratio_pct default 110%,
// lm_adapt_bad_ratio_pct default 130%), decoupled from #3's threshold
// (#3 stays conservative on purpose -- it discards real history when
// it fires, so it should stay rare).
//   BAD:  actual > pred_lat * lm_adapt_bad_ratio_pct/100
//   GOOD: actual <= pred_lat * lm_adapt_good_ratio_pct/100
//   (anything between GOOD and BAD is the hysteresis dead zone --
//    counted in window_total but neither good nor bad, same 3-bucket
//    idea Kyber uses to avoid oscillating on borderline samples)
// Every lm_adapt_window_samples completions, evaluate the window's
// bad/good ratio and step the multiplier up, down, or leave it (dead
// zone), then reset the window. Step size, bounds, and floor/ceiling
// are all sysfs-tunable.
static void adapt_depth_controller(struct adios_data *ad, u8 optype,
		u64 latency, u64 pred_lat) {
	u32 good_ratio, bad_ratio;
	u64 good_bound, bad_bound;
	int buf, old_buf;
	u32 total;

	if (optype != ADIOS_READ && optype != ADIOS_WRITE)
		return;
	if (pred_lat == 0)
		return; // model not warmed up yet, nothing to compare against

	// SAGA: own classification bar, decoupled from #3 (see struct
	// comment on lm_adapt_good_ratio_pct/lm_adapt_bad_ratio_pct).
	good_bound = (pred_lat * READ_ONCE(ad->lm_adapt_good_ratio_pct)) / 100;
	bad_bound  = (pred_lat * READ_ONCE(ad->lm_adapt_bad_ratio_pct)) / 100;

	// CHATGPT-REVIEW FIX #1 v2: read the active buffer index ONCE, use
	// it for this sample's classification AND its total increment, so
	// both land in the same buffer consistently.
	buf = atomic_read(&ad->window_buf_idx[optype]);

	if (latency > bad_bound)
		atomic_inc(&ad->window_bad[buf][optype]);
	else if (latency <= good_bound)
		atomic_inc(&ad->window_good[buf][optype]);
	// else: dead zone, counts toward total only

	total = atomic_inc_return(&ad->window_total[buf][optype]);
	if (total < READ_ONCE(ad->lm_adapt_window_samples))
		return;

	// Try to become the evaluator by atomically swapping the active
	// buffer index away from `buf`. If this fails, another CPU already
	// swapped it (it's evaluating buf, or already moved past it) --
	// just bail, this thread's contribution is already safely counted.
	old_buf = buf;
	if (atomic_cmpxchg(&ad->window_buf_idx[optype], old_buf, 1 - old_buf) != old_buf)
		return;

	{
		// From this point on, new writers read window_buf_idx fresh and
		// land in the OTHER buffer -- old_buf is effectively ours now
		// (module the narrow residual race described in the struct
		// comment). xchg the final counts rather than trusting the
		// `total` read above, since a straggler write could still land
		// here between the swap and this read.
		u32 good  = atomic_xchg(&ad->window_good[old_buf][optype], 0);
		u32 bad   = atomic_xchg(&ad->window_bad[old_buf][optype], 0);
		u32 final_total = atomic_xchg(&ad->window_total[old_buf][optype], 0);
		unsigned long now_j = jiffies;
		unsigned long cooldown_j =
			msecs_to_jiffies(READ_ONCE(ad->lm_adapt_cooldown_ms));
		bool in_cooldown = time_before(now_j,
			READ_ONCE(ad->last_adjust_jiffies[optype]) + cooldown_j);

		if (final_total == 0)
			return; // shouldn't happen, but avoid a divide-by-zero if it does

		good_ratio = (good * 100) / final_total;
		bad_ratio  = (bad  * 100) / final_total;

		// CHATGPT-REVIEW FIX (cooldown): skip the actual adjustment if
		// we changed the multiplier too recently, even though the
		// window itself still resets on schedule (keeps sample
		// classification timely; only the ACTION is rate-limited).
		if (!in_cooldown) {
			if (bad_ratio >= READ_ONCE(ad->lm_adapt_bad_threshold_pct)) {
				s32 cur = atomic_read(&ad->depth_multiplier_permille[optype]);
				s32 next = cur - (s32)READ_ONCE(ad->lm_adapt_step_down_permille);
				if (next < (s32)READ_ONCE(ad->lm_adapt_min_permille))
					next = READ_ONCE(ad->lm_adapt_min_permille);
				atomic_set(&ad->depth_multiplier_permille[optype], next);
				WRITE_ONCE(ad->last_adjust_jiffies[optype], now_j);
			} else if (good_ratio >= READ_ONCE(ad->lm_adapt_good_threshold_pct)) {
				s32 cur = atomic_read(&ad->depth_multiplier_permille[optype]);
				s32 next = cur + (s32)READ_ONCE(ad->lm_adapt_step_permille);
				if (next > (s32)READ_ONCE(ad->lm_adapt_max_permille))
					next = READ_ONCE(ad->lm_adapt_max_permille);
				atomic_set(&ad->depth_multiplier_permille[optype], next);
				WRITE_ONCE(ad->last_adjust_jiffies[optype], now_j);
			}
			// else: dead zone, multiplier unchanged -- this is the
			// hysteresis that keeps the controller from oscillating on
			// every window when the ratio sits between the two thresholds.
		}
	}
}

// SAGA: forces an out-of-band shrink of the model's accumulated stats,
// same shift-based "forget a fraction of the history" logic the
// upstream volume-triggered shrink already uses (lm_shrink_resist),
// just triggered by adios_completed_request() noticing sustained
// misprediction instead of waiting for lm_shrink_at_kreqs/gbytes to be
// crossed. See its use below in adios_completed_request().
// SAGA FIX (3rd review round): force_shrink_model() is only ever called
// from shrink_work_fn() (see below), a workqueue callback -- real
// process context, not softirq, can sleep. The 2nd review round moved
// the CALL out of the completion hot path into this workqueue for
// exactly that reason, but left the allocation itself on GFP_ATOMIC
// "deliberately, to avoid restructuring around the spinlock" (see the
// old comment on shrink_pending). That tradeoff is no longer needed:
// allocate the copy with GFP_KERNEL *before* taking the lock (a fixed-
// size kmalloc, sized independent of old_params' content, so nothing
// about the copy depends on holding the lock first), then take the
// lock only for the short memcpy + shrink-math + RCU swap. This is
// strictly a reliability improvement -- GFP_KERNEL can wait for
// reclaim, GFP_ATOMIC can't, and a failed shrink under real memory
// pressure (precisely when this path fires most) previously just
// silently skipped that attempt.
static void force_shrink_model(struct latency_model *model) {
	struct latency_model_params *old_params, *new_params;
	unsigned long flags;
	u8 reduction;

	new_params = kmalloc(sizeof(*new_params), GFP_KERNEL);
	if (!new_params)
		return;

	spin_lock_irqsave(&model->update_lock, flags);
	reduction = model->lm_shrink_resist;
	old_params = rcu_dereference_protected(model->params,
				lockdep_is_held(&model->update_lock));
	memcpy(new_params, old_params, sizeof(*new_params));
	if (new_params->small_count >> reduction) {
		new_params->small_sum_delay -= new_params->small_sum_delay >> reduction;
		new_params->small_count     -= new_params->small_count     >> reduction;
	}
	if (new_params->large_sum_bsize >> reduction) {
		new_params->large_sum_delay -= new_params->large_sum_delay >> reduction;
		new_params->large_sum_bsize -= new_params->large_sum_bsize >> reduction;
		new_params->large_count    -= new_params->large_count     >> reduction;
	}
	rcu_assign_pointer(model->params, new_params);
	spin_unlock_irqrestore(&model->update_lock, flags);
	kfree_rcu(old_params, rcu);
}

// SAGA: interactive-boost check. Looks up (and updates) the current
// task's tgid in the small recency table. If its last I/O was longer
// ago than lm_interactive_idle_ms (or it has no entry at all -- first
// I/O ever seen from it), it's treated as "was idle, just woke up" and
// the caller applies a one-off deadline reduction to its SYNC read.
// Deliberately only for REQ_SYNC reads: that's the pattern of a UI
// thread blocking on a file/resource read, which is the case that
// actually shows up as jank if delayed. Async writes are unaffected --
// boosting them would fight the write-deprioritization the rest of
// ADIOS already does on purpose.
static bool task_was_idle_and_gets_boost(struct adios_data *ad) {
	pid_t tgid;
	u64 now;
	u64 idle_threshold_ns;
	unsigned long flags;
	int i, oldest_idx = 0;
	u64 oldest_seen = U64_MAX;
	bool boost = false;

	// CHATGPT-REVIEW FIX: exclude kernel threads. A kworker doing I/O
	// on behalf of some subsystem is not "user interaction" no matter
	// how long it's been since its last read -- boosting it fights the
	// whole point of this heuristic (prioritizing genuine foreground
	// activity). Real find from the review: without this, background
	// kernel work could occasionally look "interactive" and jump the
	// queue ahead of an actual foreground read.
	if (current->flags & PF_KTHREAD)
		return false;

	tgid = task_tgid_nr(current);
	now = ktime_get_ns();
	idle_threshold_ns = (u64)READ_ONCE(ad->lm_interactive_idle_ms) * NSEC_PER_MSEC;

	// SAGA: trylock instead of a blocking lock. This runs on EVERY sync
	// read insert -- the hottest path in the scheduler for a read-heavy
	// workload, which is exactly what we're trying to optimize. Under
	// real concurrent multi-thread I/O (not exercised by any of this
	// session's single-threaded synthetic benchmarks, but very real in
	// normal Android usage with many apps/threads reading at once), a
	// blocking spin_lock_irqsave here means one thread's read insert
	// can stall waiting for another CPU's read insert to finish updating
	// this same small table -- pure added latency with no benefit, on
	// the exact path we want fast. This mechanism is explicitly a
	// best-effort HINT already (see the struct/function comments above:
	// "missed boost, never incorrect ordering"), so skipping the boost
	// under contention instead of waiting for it is the correct
	// trade-off: never adds hot-path latency, occasionally misses a
	// boost it would have otherwise given. Uncontended case (by far the
	// common one) is unchanged -- trylock succeeds immediately, same as
	// before.
	if (!spin_trylock_irqsave(&ad->iactv_lock, flags))
		return false;
	for (i = 0; i < ADIOS_IACTV_SLOTS; i++) {
		if (ad->iactv_table[i].tgid == tgid) {
			if (now - ad->iactv_table[i].last_seen_ns >= idle_threshold_ns)
				boost = true;
			ad->iactv_table[i].last_seen_ns = now;
			spin_unlock_irqrestore(&ad->iactv_lock, flags);
			return boost;
		}
		if (ad->iactv_table[i].last_seen_ns < oldest_seen) {
			oldest_seen = ad->iactv_table[i].last_seen_ns;
			oldest_idx = i;
		}
	}
	// tgid not found: no history, evict the table's oldest entry for it.
	// Being unseen is itself "was idle" (nothing recorded = never
	// recently active), so this also boosts -- first read of a
	// freshly-launched process/thread.
	ad->iactv_table[oldest_idx].tgid = tgid;
	ad->iactv_table[oldest_idx].last_seen_ns = now;
	spin_unlock_irqrestore(&ad->iactv_lock, flags);
	return true;
}

/*
 * True when the request's IO priority marks it as belonging to a
 * latency-critical submitter -- RT class outright, or the top tiers
 * of the BE class (Android's task_profiles.json commonly assigns one
 * of these to the top-app cgroup via ioprio_set()/SetIoPriority when
 * configured to do so; harmless no-op otherwise, since default class
 * for unconfigured processes is IOPRIO_CLASS_NONE/BE-4).
 */
static inline bool adios_req_is_topapp(struct request *rq) {
	unsigned short ioprio = req_get_ioprio(rq);
	unsigned short class = IOPRIO_PRIO_CLASS(ioprio);

	if (class == IOPRIO_CLASS_RT)
		return true;
	if (class == IOPRIO_CLASS_BE && IOPRIO_PRIO_DATA(ioprio) <= 1)
		return true;
	return false;
}

// Add a request to the deadline-sorted red-black tree
static void add_to_dl_tree(
		struct adios_data *ad, bool dl_idx, struct request *rq) {
	struct rb_root_cached *root = &ad->dl_tree[dl_idx];
	struct rb_node **link = &(root->rb_root.rb_node), *parent = NULL;
	bool leftmost = true;
	struct adios_rq_data *rd = get_rq_data(rq);
	struct dl_group *dlg;
	u64 deadline;
	bool was_empty = RB_EMPTY_ROOT(&root->rb_root);

	rd->is_sync = !!(rq->cmd_flags & REQ_SYNC);

	/* Tier-2: Synchronous Requests
	 * - Needs to be FIFO within a same optype
	 * - Relaxed order between different optypes
	 * - basically needs to be processed in early time */
	rd->deadline = rq->start_time_ns;

	// SAGA: interactive-boost. Deliberately breaks the "FIFO within
	// optype" rule stated above, on purpose, only for this case: a
	// sync READ from a task that appears to have just woken up from
	// idle jumps ahead of other already-queued sync reads from
	// continuously-active tasks (e.g. a background scan). That's
	// exactly the BFQ-style bet -- the newly-active task is more
	// likely to be blocking a user-visible UI thread right now than
	// a task that's been steadily reading all along.
	if ((rq->cmd_flags & REQ_SYNC) && adios_optype(rq) == ADIOS_READ &&
			task_was_idle_and_gets_boost(ad)) {
		u64 boost_ns = READ_ONCE(ad->lm_interactive_boost_ns);
		if (rd->deadline > boost_ns)
			rd->deadline -= boost_ns;
		else
			rd->deadline = 0;
	}

	/* Tier-3: Aynchronous Requests
	 * - Can be reordered and delayed freely */
	if (!(rq->cmd_flags & REQ_SYNC)) {
		rd->deadline += READ_ONCE(ad->latency_target[adios_optype(rq)]);
		// SAGA: gate predictive reordering on both the manual switch
		// (compliance_flags/FIXORDER, operator-controlled) AND the
		// automatic per-optype confidence check (model_is_confident).
		// Either one alone can fall back to non-predictive ordering;
		// both must allow it for pred_lat to actually influence the
		// deadline. This directly targets the cold-start regressions
		// observed empirically: a freshly loaded/reset model no
		// longer gets to mis-order requests based on near-zero-sample
		// predictions -- it behaves like FIXORDER until it has earned
		// the right not to, per optype, independently.
		if (!compliant(ad, ADIOS_CF_FIXORDER) &&
				model_is_confident(ad, adios_optype(rq), blk_rq_bytes(rq)))
			rd->deadline += rd->pred_lat;
	}

	u64 topapp_bonus = READ_ONCE(ad->topapp_deadline_bonus);

	if (topapp_bonus && adios_req_is_topapp(rq))
		rd->deadline -= min_t(u64, rd->deadline, topapp_bonus);

	// Now quantize the deadline (-> dlg->deadline == RB-Tree key)
	deadline = rd->deadline & ~((1ULL << ADIOS_QUANTUM_SHIFT) - 1);

	while (*link) {
		dlg = rb_entry(*link, struct dl_group, node);
		s64 diff = deadline - dlg->deadline;

		parent = *link;
		if (diff < 0) {
			link = &((*link)->rb_left);
		} else if (diff > 0) {
			link = &((*link)->rb_right);
			leftmost = false;
		} else { // diff == 0
			goto found;
		}
	}

	dlg = rb_entry_safe(parent, struct dl_group, node);
	if (!dlg || dlg->deadline != deadline) {
		dlg = kmem_cache_zalloc(ad->dl_group_pool, GFP_ATOMIC);
		if (!dlg)
			return;
		dlg->deadline = deadline;
		INIT_LIST_HEAD(&dlg->rqs);
		rb_link_node(&dlg->node, parent, link);
		rb_insert_color_cached(&dlg->node, root, leftmost);
	}
found:
	list_add_tail(&rd->dl_node, &dlg->rqs);
	rd->dl_group = &dlg->rqs;

	if (was_empty)
		set_adios_state(ad, ADIOS_STATE_DL, dl_idx, true);
}

// Remove a request from the deadline-sorted red-black tree
static void del_from_dl_tree(
		struct adios_data *ad, bool dl_idx, struct request *rq) {
	struct rb_root_cached *root = &ad->dl_tree[dl_idx];
	struct adios_rq_data *rd = get_rq_data(rq);
	struct dl_group *dlg = container_of(rd->dl_group, struct dl_group, rqs);

	list_del_init(&rd->dl_node);
	if (list_empty(&dlg->rqs)) {
		rb_erase_cached(&dlg->node, root);
		kmem_cache_free(ad->dl_group_pool, dlg);
	}
	rd->dl_group = NULL;

	if (RB_EMPTY_ROOT(&ad->dl_tree[dl_idx].rb_root))
		set_adios_state(ad, ADIOS_STATE_DL, dl_idx, false);
}

// Remove a request from the scheduler
static void remove_request(struct adios_data *ad, struct request *rq) {
	bool dl_idx = adios_optype_not_read(rq);
	struct request_queue *q = rq->q;
	struct adios_rq_data *rd = get_rq_data(rq);

	list_del_init(&rq->queuelist);

	// We might not be on the rbtree, if we are doing an insert merge
	if (rd->dl_group)
		del_from_dl_tree(ad, dl_idx, rq);

	elv_rqhash_del(q, rq);
	if (q->last_merge == rq)
		q->last_merge = NULL;
}

// Convert a queue depth to the corresponding word depth for shallow allocation
static int to_word_depth(struct blk_mq_hw_ctx *hctx, unsigned int qdepth) {
	struct sbitmap_queue *bt = &hctx->sched_tags->bitmap_tags;
	const unsigned int nrr = hctx->queue->nr_requests;

	return ((qdepth << bt->sb.shift) + nrr - 1) / nrr;
}

// We limit the depth of request allocation for asynchronous and write requests
static void adios_limit_depth(blk_opf_t opf, struct blk_mq_alloc_data *data) {
	struct adios_data *ad = data->q->elevator->elevator_data;

	// Do not throttle synchronous reads
	if (op_is_sync(opf) && !op_is_write(opf))
		return;

	data->shallow_depth = to_word_depth(data->hctx, ad->async_depth);
}

// The number of requests in the queue was notified from the block layer
static void adios_depth_updated(struct blk_mq_hw_ctx *hctx) {
	struct request_queue *q = hctx->queue;
	struct adios_data *ad = q->elevator->elevator_data;
	struct blk_mq_tags *tags = hctx->sched_tags;

	ad->async_depth = q->nr_requests;

	sbitmap_queue_min_shallow_depth(&tags->bitmap_tags, 1);
}

// Handle request merging after a merge operation
static void adios_request_merged(struct request_queue *q, struct request *req,
				  enum elv_merge type) {
	bool dl_idx = adios_optype_not_read(req);
	struct adios_data *ad = q->elevator->elevator_data;

	// Reposition request in the deadline-sorted tree
	del_from_dl_tree(ad, dl_idx, req);
	add_to_dl_tree(ad, dl_idx, req);
}

// Handle merging of requests after one has been merged into another
static void adios_merged_requests(struct request_queue *q, struct request *req,
				   struct request *next) {
	struct adios_data *ad = q->elevator->elevator_data;

	lockdep_assert_held(&ad->lock);

	// kill knowledge of next, this one is a goner
	remove_request(ad, next);
}

// Try to merge a bio into an existing rq before associating it with an rq
static bool adios_bio_merge(struct request_queue *q, struct bio *bio,
		unsigned int nr_segs) {
	unsigned long flags;
	struct adios_data *ad = q->elevator->elevator_data;
	struct request *free = NULL;
	bool ret;

	if (!spin_trylock_irqsave(&ad->lock, flags))
		return false;

	ret = blk_mq_sched_try_merge(q, bio, nr_segs, &free);
	spin_unlock_irqrestore(&ad->lock, flags);

	if (free)
		blk_mq_free_request(free);

	return ret;
}

static bool merge_or_insert_to_dl_tree(struct adios_data *ad,
		struct request *rq, struct request_queue *q, struct list_head *free) {
	if (blk_mq_sched_try_insert_merge(q, rq, free))
		return true;

	bool dl_idx = adios_optype_not_read(rq);
	add_to_dl_tree(ad, dl_idx, rq);

	if (rq_mergeable(rq)) {
		elv_rqhash_add(q, rq);
		if (!q->last_merge)
			q->last_merge = rq;
	}

	return false;
}

static void insert_to_prio_queue(struct adios_data *ad,
		struct request *rq, bool pq_idx) {
	struct adios_rq_data *rd = get_rq_data(rq);

	/* We're sure that rd->managed == true */
	union adios_in_flight_rqs ifr = {
		.count          = 1,
		.total_pred_lat = rd->pred_lat,
	};
	atomic64_add(ifr.scalar, &ad->in_flight_rqs.atomic);

	scoped_guard(spinlock_irqsave, &ad->pq_lock) {
		bool was_empty = list_empty(&ad->prio_queue[pq_idx]);
		list_add_tail(&rq->queuelist, &ad->prio_queue[pq_idx]);
		if (was_empty)
			set_adios_state(ad, ADIOS_STATE_PQ, pq_idx, true);
	}
}

// Insert a request into the scheduler (after Read & Write models stabilized)
static void insert_request_post_stability(struct blk_mq_hw_ctx *hctx,
		struct request *rq, bool insert_flags, struct list_head *free) {
	struct request_queue *q = hctx->queue;
	struct adios_data *ad = q->elevator->elevator_data;
	struct adios_rq_data *rd = get_rq_data(rq);
	u8 optype = adios_optype(rq);

	rd->managed = true;
	rd->block_size = blk_rq_bytes(rq);
	rd->pred_lat =
		latency_model_predict(&ad->latency_model[optype], rd->block_size);
	if (unlikely(rd->pred_lat > ad->lat_model_latency_limit))
		rd->pred_lat = ad->lat_model_latency_limit;

	/* Tier-0: BLK_MQ_INSERT_AT_HEAD Requests */
	if (insert_flags) {
		insert_to_prio_queue(ad, rq, 0);
		return;
	}

	if (merge_or_insert_to_dl_tree(ad, rq, q, free))
		return;
}

// Insert a request into the scheduler (before Read & Write models stabilizes)
static void insert_request_pre_stability(struct blk_mq_hw_ctx *hctx,
		struct request *rq, bool insert_flags, struct list_head *free) {
	struct adios_data *ad = hctx->queue->elevator->elevator_data;
	struct adios_rq_data *rd = get_rq_data(rq);
	u8 optype = adios_optype(rq);
	u8 pq_idx = !insert_flags;
	bool stable = false;

	rd->managed = true;
	rd->block_size = blk_rq_bytes(rq);
	rd->pred_lat =
		latency_model_predict(&ad->latency_model[optype], rd->block_size);
	if (unlikely(rd->pred_lat > ad->lat_model_latency_limit))
		rd->pred_lat = ad->lat_model_latency_limit;

	insert_to_prio_queue(ad, rq, pq_idx);

	rcu_read_lock();
	if (rcu_dereference(ad->latency_model[ADIOS_READ].params)->base > 0 &&
		rcu_dereference(ad->latency_model[ADIOS_WRITE].params)->base > 0)
			stable = true;
	rcu_read_unlock();

	if (stable)
		ad->models_stable = true;
}

// Insert multiple requests into the scheduler
static void adios_insert_requests(struct blk_mq_hw_ctx *hctx,
				   struct list_head *list,
				   bool insert_flags) {
	struct request_queue *q = hctx->queue;
	struct adios_data *ad = q->elevator->elevator_data;
	struct request *rq;
	bool stop = false;
	LIST_HEAD(free);

	do {
	scoped_guard(spinlock_irqsave, &ad->lock)
	for (int i = 0; i < ADIOS_MAX_INSERTS_PER_LOCK; i++) {
		if (list_empty(list)) {
			stop = true;
			break;
		}
		rq = list_first_entry(list, struct request, queuelist);
		list_del_init(&rq->queuelist);
		if (likely(ad->models_stable))
			insert_request_post_stability(hctx, rq, insert_flags, &free);
		else
			insert_request_pre_stability(hctx, rq, insert_flags, &free);
	}} while (!stop);

	blk_mq_free_requests(&free);
}

// Prepare a request before it is inserted into the scheduler
static void adios_prepare_request(struct request *rq) {
	struct adios_data *ad = rq->q->elevator->elevator_data;
	struct adios_rq_data *rd;

	/*
	 * REVIEW FIX (partial -- see this patch's own changelog for the
	 * scope note): mempool_alloc() under GFP_ATOMIC is NOT guaranteed
	 * to succeed -- if the pool's reserved elements are all checked
	 * out AND the backing allocator also fails under atomic/no-wait
	 * conditions, it returns NULL. The old code fed that straight into
	 * memset(rd, 0, ...) with no check -- an immediate NULL-pointer
	 * deref/panic under exactly the memory-pressure conditions this
	 * scheduler's own shrink-model logic exists to survive. This check
	 * prevents that specific, guaranteed crash. It does NOT make the
	 * rest of the file NULL-safe: get_rq_data() is a raw
	 * `return rq->elv.priv[0];` with no NULL check, and every one of
	 * its ~8 call sites dereferences the result immediately -- so a
	 * request left with priv[0] == NULL here will still crash a few
	 * lines later in the insert path, just not right here. A complete
	 * fix needs either auditing/hardening every get_rq_data() call
	 * site, or restructuring this scheduler to pre-allocate per-tag
	 * private data at init time (bounded by q->nr_requests, so it
	 * could never fail here at all) instead of a mempool -- both are
	 * real, larger pieces of work, deliberately out of scope for this
	 * pass.
	 */
	rd = mempool_alloc(ad->rq_data_pool, GFP_ATOMIC);
	if (unlikely(!rd)) {
		rq->elv.priv[0] = NULL;
		return;
	}
	memset(rd, 0, sizeof(*rd));
	rd->rq = rq;
	rq->elv.priv[0] = rd;
}

static struct adios_rq_data *get_dl_first_rd(struct adios_data *ad, bool idx) {
	struct rb_root_cached *root = &ad->dl_tree[idx];
	struct rb_node *first = rb_first_cached(root);
	struct dl_group *dl_group = rb_entry(first, struct dl_group, node);

	return list_first_entry(&dl_group->rqs, struct adios_rq_data, dl_node);
}

// Comparison function for sorting requests by block address
static int cmp_rq_pos(void *priv,
		const struct list_head *a, const struct list_head *b) {
	struct request *rq_a = list_entry(a, struct request, queuelist);
	struct request *rq_b = list_entry(b, struct request, queuelist);
	u64 pos_a = blk_rq_pos(rq_a);
	u64 pos_b = blk_rq_pos(rq_b);

	return (int)(pos_a > pos_b) - (int)(pos_a < pos_b);
}

#ifndef list_last_entry_or_null
#define list_last_entry_or_null(ptr, type, member) \
	(!list_empty(ptr) ? list_last_entry(ptr, type, member) : NULL)
#endif

// Update the elevator direction
static void update_elv_direction(struct adios_data *ad) {
	if (!ad->is_rotational)
		return;

	bool page = ad->bq_page;
	struct list_head *q = &ad->batch_queue[page][1];
	if (ad->bq_batch_order[page] < ADIOS_BO_ELEVATOR || list_empty(q)) {
		ad->elv_direction = 0;
		return;
	}

	// Get first and last request positions in the queue
	struct request *rq_a = list_first_entry(q, struct request, queuelist);
	struct request *rq_b = list_last_entry (q, struct request, queuelist);
	u64 pos_a = blk_rq_pos(rq_a);
	u64 pos_b = blk_rq_pos(rq_b);
	u64 avg_rq_pos = (pos_a + pos_b) >> 1;

	ad->elv_direction = !!(ad->head_pos > avg_rq_pos);
}

// Fill the batch queues with requests from the deadline-sorted red-black tree
static bool fill_batch_queues(struct adios_data *ad, u64 tpl) {
	struct adios_rq_data *rd;
	struct request *rq;
	struct list_head *dest_q;
	u8  dest_idx;
	u64 added_lat = 0;
	u32 optype_count[ADIOS_OPTYPES] = {0};
	u32 count = 0;
	u8 optype;
	bool page = !ad->bq_page, dl_idx, bias_idx, update_bias;
	u32 dl_queued;
	u8 bq_batch_order;
	bool stop = false;

	// Reset batch queue counts for the back page
	memset(&ad->batch_count[page], 0, sizeof(ad->batch_count[page]));

	ad->bq_batch_order[page] =
		bq_batch_order = ad->batch_order;

	do {
	scoped_guard(spinlock_irqsave, &ad->lock)
	for (int i = 0; i < ADIOS_MAX_DELETES_PER_LOCK; i++) {
		bool has_base = false;

		dl_queued = eval_adios_state(ad, ADIOS_STATE_DL);
		// Check if there are any requests queued in the deadline tree
		if (!dl_queued) {
			stop = true;
			break;
		}

		// Reads if both queues have requests, otherwise pick the non-empty.
		dl_idx = dl_queued >> 1;

		// Get the first request from the deadline-sorted tree
		rd = get_dl_first_rd(ad, dl_idx);

		bias_idx = ad->dl_bias < 0;
		// If read and write requests are queued, choose one based on bias
		if (dl_queued == 0x3) {
			struct adios_rq_data *trd[2] = {get_dl_first_rd(ad, 0), rd};
			rd = trd[bias_idx];

			update_bias = (trd[bias_idx]->deadline > trd[!bias_idx]->deadline);
		} else
			update_bias = (bias_idx == dl_idx);

		rq = rd->rq;
		optype = adios_optype(rq);

		rcu_read_lock();
		has_base =
			!!rcu_dereference(ad->latency_model[optype].params)->base;
		rcu_read_unlock();

		// Check batch size and total predicted latency
		// SAGA NOTE (3rd review round): deliberately NOT confidence-gated,
		// unlike the dl_bias update below and add_to_dl_tree()'s own
		// deadline sum. Substituting 0 (or any other invented fallback)
		// here for an unconfident model would let MORE requests into the
		// batch than the real, unlearned latency cost justifies --
		// admission overshoot, the opposite failure mode from the
		// mis-ordering the confidence gate exists to prevent, and with
		// no benchmark evidence on this device to say which is worse.
		// The model's raw pred_lat (seeded via sideload/first samples,
		// never literally undefined once `has_base` is true, checked
		// just above) is the more conservative choice already in place
		// -- left as-is on purpose.
		if (count && (!has_base ||
				ad->batch_count[page][optype] >= effective_batch_limit(ad, optype) ||
				(tpl + added_lat + rd->pred_lat) > READ_ONCE(ad->global_latency_window))) {
			stop = true;
			break;
		}

		if (update_bias) {
			s64 sign = ((s64)bias_idx << 1) - 1;
			// SAGA FIX (3rd review round): the confidence gate added in
			// the 1st round only covered add_to_dl_tree()'s own deadline
			// sum (the ASYNC tier below) -- it never covered THIS spot,
			// where an unconfident model's near-arbitrary pred_lat could
			// still swing read-vs-write dispatch order every batch.
			// Reuse rd->pred_lat's OWN existing cold-start convention
			// (pred_lat==0 -> just alternate by `sign`, no magnitude-
			// based skew) for "not confident yet" too, via a local
			// substitution -- rd->pred_lat ITSELF is left completely
			// untouched (still real per latency_model_input()'s bucket
			// classification below, still real for the admission-window
			// check above -- see that check's own note for why it is
			// deliberately NOT gated the same way).
			u64 bias_pred_lat = model_is_confident(ad, optype, rd->block_size) ?
				rd->pred_lat : 0;
			if (unlikely(!bias_pred_lat))
				ad->dl_bias = sign;
			else
				// Adjust the bias based on the predicted latency
				ad->dl_bias += sign * (s64)((bias_pred_lat *
					adios_prio_to_wmult[ad->dl_prio[bias_idx] + 20]) >> 10);
		}

		remove_request(ad, rq);

		// Add request to the corresponding batch queue
		dest_idx = (bq_batch_order == ADIOS_BO_OPTYPE || optype == ADIOS_OTHER)?
			optype : !rd->is_sync;
		dest_q = &ad->batch_queue[page][dest_idx];
		list_add_tail(&rq->queuelist, dest_q);
		ad->bq_state[page] |= 1U << dest_idx;
		ad->batch_count[page][optype]++;
		optype_count[optype]++;
		added_lat += rd->pred_lat;
		count++;
	}} while (!stop);

	if (bq_batch_order == ADIOS_BO_ELEVATOR && ad->batch_count[page][1] > 1)
			list_sort(NULL, &ad->batch_queue[page][1], cmp_rq_pos);

	if (count) {
		/* We're sure that every request's rd->managed == true */
		union adios_in_flight_rqs ifr = {
			.count          = count,
			.total_pred_lat = added_lat,
		};
		atomic64_add(ifr.scalar, &ad->in_flight_rqs.atomic);

		set_adios_state(ad, ADIOS_STATE_BQ, page, true);

		for (optype = 0; optype < ADIOS_OPTYPES; optype++)
			if (ad->batch_actual_max_size[optype] < optype_count[optype])
				ad->batch_actual_max_size[optype] = optype_count[optype];
		if (ad->batch_actual_max_total < count)
			ad->batch_actual_max_total = count;
	}
	return count;
}

// Flip to the next batch queue page
static void flip_bq_page(struct adios_data *ad) {
	ad->bq_page = !ad->bq_page;
	update_elv_direction(ad);
}

// Pop a request from the specified index (optype or elevator tier)
static inline struct request *pop_bq_request(
		struct adios_data *ad, u8 idx, bool direction) {
	bool page = ad->bq_page;
	struct list_head *q = &ad->batch_queue[page][idx];
	struct request *rq = direction ?
		list_last_entry_or_null (q, struct request, queuelist):
		list_first_entry_or_null(q, struct request, queuelist);
	if (rq) {
		list_del_init(&rq->queuelist);
		if (list_empty(q))
			ad->bq_state[page] &= ~(1U << idx);
	}
	return rq;
}

static struct request *pop_next_bq_request_optype(struct adios_data *ad) {
	u32 bq_state = ad->bq_state[ad->bq_page];
	if (!bq_state) return NULL;

	struct request *rq;
	u32 bq_idx = __builtin_ctz(bq_state);

	// Dispatch based on optype (FIFO within each) or single-queue elevator
	rq = pop_bq_request(ad, bq_idx, false);
	return rq;
}

static struct request *pop_next_bq_request_elevator(struct adios_data *ad) {
	u32 bq_state = ad->bq_state[ad->bq_page];
	if (!bq_state) return NULL;

	struct request *rq;
	u32 bq_idx = __builtin_ctz(bq_state);
	bool direction = (bq_idx == 1) & ad->elv_direction;

	// Tier-2 (sync) is always high priority
	// Tier-3 (async) uses the pre-calculated elevator direction
	rq = pop_bq_request(ad, bq_idx, direction);

	/*
	 * REVIEW FIX: was `!(bq_state & 0x1)` -- bq_state here is the
	 * snapshot taken BEFORE the pop above. bq_idx == 0 is only reached
	 * when __builtin_ctz(bq_state) == 0, which requires bit 0 to
	 * already be set in that same pre-pop snapshot -- so
	 * `bq_state & 0x1` was always true whenever bq_idx == 0, making
	 * this condition impossible to satisfy: this update_elv_direction()
	 * call could never fire. pop_bq_request() clears the bit in
	 * ad->bq_state[page] directly (in place) when the pop empties the
	 * queue, so re-reading the live state after the pop is what
	 * actually reflects whether it just became empty.
	 */
	/* If batch queue for the sync requests just became empty */
	if (bq_idx == 0 && rq && !(ad->bq_state[ad->bq_page] & 0x1))
		update_elv_direction(ad);

	return rq;
}

// Returns the state of the batch queue page
static inline bool bq_page_has_rq(u32 bq_state, bool page) {
	return bq_state & (1U << page);
}

// Dispatch a request from the batch queues
static struct request *dispatch_from_bq(struct adios_data *ad) {
	struct request *rq;

	guard(spinlock_irqsave)(&ad->bq_lock);

	u32 state = get_adios_state(ad);
	u32 bq_state = eval_this_adios_state(state, ADIOS_STATE_BQ);
	u32 bq_curr_page_has_rq = bq_page_has_rq(bq_state, ad->bq_page);
	union adios_in_flight_rqs ifr;
	ifr.scalar = atomic64_read(&ad->in_flight_rqs.atomic);
	u64 tpl = ifr.total_pred_lat;

	// Refill the batch queues if the back page is empty, dl_tree has work, and
	// current page is empty or the total ongoing latency is below the threshold
	if (!bq_page_has_rq(bq_state, !ad->bq_page) &&
			(!bq_curr_page_has_rq || (!tpl || tpl < div_u64(
			READ_ONCE(ad->global_latency_window) * ad->bq_refill_below_ratio, 100))) &&
			eval_this_adios_state(state, ADIOS_STATE_DL))
		fill_batch_queues(ad, tpl);

	// If current batch queue page is empty, and the other page has work, flip
	if (!bq_curr_page_has_rq &&
			bq_page_has_rq(eval_adios_state(ad, ADIOS_STATE_BQ), !ad->bq_page))
		flip_bq_page(ad);

	// Use the per-page state to decide the dispatch logic, ensuring correctness
	rq = (ad->bq_batch_order[ad->bq_page] == ADIOS_BO_ELEVATOR) ?
		pop_next_bq_request_elevator(ad):
		pop_next_bq_request_optype(ad);

	if (rq) {
		bool page = ad->bq_page;
		bool is_empty = !ad->bq_state[page];
		if (is_empty)
			set_adios_state(ad, ADIOS_STATE_BQ, page, false);
		return rq;
	}

	return NULL;
}

// Dispatch a request from the priority queue
static struct request *dispatch_from_pq(struct adios_data *ad) {
	struct request *rq = NULL;

	guard(spinlock_irqsave)(&ad->pq_lock);
	u32 pq_state = eval_adios_state(ad, ADIOS_STATE_PQ);
	/*
	 * REVIEW FIX: was `pq_idx = pq_state >> 1`. pq_state's bit0 means
	 * prio_queue[0] (Tier 0, BLK_MQ_INSERT_AT_HEAD -- the highest
	 * priority queue) is non-empty, bit1 means prio_queue[1] (the
	 * pre-model-stability fallback queue, only ever used very early
	 * before both READ/WRITE models get their first sample) is
	 * non-empty. When BOTH are set (pq_state == 3), the old formula
	 * gave 3 >> 1 == 1 -- picking the lower-priority queue 1 while a
	 * genuine Tier-0 request sat waiting in queue 0. Correct rule: 0
	 * wins whenever it has anything, 1 only otherwise.
	 */
	u8  pq_idx = (pq_state & 1) ? 0 : 1;
	struct list_head *q = &ad->prio_queue[pq_idx];

	if (unlikely(list_empty(q))) return NULL;

	rq = list_first_entry(q, struct request, queuelist);
	list_del_init(&rq->queuelist);
	if (list_empty(q)) {
		set_adios_state(ad, ADIOS_STATE_PQ, pq_idx, false);
		update_elv_direction(ad);
	}
	return rq;
}

// Dispatch a request to the hardware queue
static struct request *adios_dispatch_request(struct blk_mq_hw_ctx *hctx) {
	struct adios_data *ad = hctx->queue->elevator->elevator_data;
	struct request *rq;

	rq = dispatch_from_pq(ad);
	if (rq)
		goto found;

	rq = dispatch_from_bq(ad);
	if (rq)
		goto found;

	return NULL;
found:
	if (ad->is_rotational)
		ad->head_pos = blk_rq_pos(rq) + blk_rq_sectors(rq);

	rq->rq_flags |= RQF_STARTED;
	return rq;
}

// Timer callback function to periodically update latency models
static void update_timer_callback(struct timer_list *t) {
	struct adios_data *ad = from_timer(ad, t, update_timer);
	bool need_shrink = false;

	for (u8 optype = 0; optype < ADIOS_OPTYPES; optype++) {
		struct latency_model *lm = &ad->latency_model[optype];

		latency_model_update(ad, lm);

		// CHATGPT-REVIEW FIX #4: don't call force_shrink_model()
		// directly from here -- this timer callback still runs in
		// softirq context. Just note that a shrink is pending; the
		// actual work happens in shrink_work_fn(), scheduled below,
		// which runs in real process context.
		if (atomic_read(&lm->shrink_pending))
			need_shrink = true;
	}
	if (need_shrink)
		schedule_work(&ad->shrink_work);
}

static void topapp_activate_timer_callback(struct timer_list *t) {
	struct adios_data *ad = from_timer(ad, t, topapp_activate_timer);

	WRITE_ONCE(ad->topapp_deadline_bonus, ad->topapp_deadline_bonus_target);
	pr_info("adios: topapp_deadline_bonus activated (%llu ns) after %llums\n",
		ad->topapp_deadline_bonus_target, ad->topapp_activation_delay_ms);
}

// CHATGPT-REVIEW FIX #4: force_shrink_model() moved here from the
// timer callback. Runs in a workqueue's own process context, not
// softirq and not the I/O completion hot path.
// REVIEW FIX (stale comment): this used to say the allocation inside
// force_shrink_model() was left on GFP_ATOMIC deliberately. That was
// true when this comment was written, but force_shrink_model() was
// since restructured (see its own comment, "SAGA FIX 3rd review
// round") to allocate with GFP_KERNEL *before* taking
// model->update_lock, precisely so it wouldn't need to stay on
// GFP_ATOMIC. This comment just never got updated to match -- fixed
// here instead of left contradicting the actual code. Either way,
// running in this workqueue instead of a timer callback still matters
// on its own: it no longer competes with timer/softirq processing for
// CPU time exactly when the device may already be under pressure (the
// situation that triggers a shrink in the first place).
static void shrink_work_fn(struct work_struct *work) {
	struct adios_data *ad = container_of(work, struct adios_data, shrink_work);

	for (u8 optype = 0; optype < ADIOS_OPTYPES; optype++) {
		struct latency_model *lm = &ad->latency_model[optype];

		if (atomic_xchg(&lm->shrink_pending, 0))
			force_shrink_model(lm);
	}
}

// Handle the completion of a request
static void adios_completed_request(struct request *rq, u64 now) {
	struct adios_data *ad = rq->q->elevator->elevator_data;
	struct adios_rq_data *rd = get_rq_data(rq);

	if (!rd)
		return;

	union adios_in_flight_rqs ifr = { .scalar = 0 };

	if (rd->managed) {
		union adios_in_flight_rqs ifr_to_sub = {
			.count          = 1,
			.total_pred_lat = rd->pred_lat,
		};
		ifr.scalar = atomic64_sub_return(
			ifr_to_sub.scalar, &ad->in_flight_rqs.atomic);
	}
	u8 optype = adios_optype(rq);

	unsigned long flags;
	struct adios_pcpu_completion *pc;

	local_irq_save(flags);
	pc = this_cpu_ptr(ad->pcpu_completion);

	if (optype == ADIOS_OTHER) {
		// Non-positional commands make the head position unpredictable.
		// Invalidate our knowledge of the last completed position.
		if (ad->is_rotational)
			pc->last_completed_pos = 0;
		local_irq_restore(flags);
		return;
	}

	u64 lct = pc->last_completed_time ?: rq->io_start_time_ns;
	pc->last_completed_time = (ifr.count) ? now : 0;

	if (!rq->io_start_time_ns || !rd->block_size || unlikely(now < lct)) {
		local_irq_restore(flags);
		return;
	}

	u64 latency = now - lct;

	// CHATGPT-REVIEW FIX #3 (real finding, upstream's own metric, not
	// something I introduced): `latency` above is the gap since the
	// last completion ON THIS SAME CPU (or this request's own start
	// time as a fallback) -- under continuous back-to-back load, that's
	// closer to "inter-completion gap in a saturated pipeline" than
	// true per-request service time. That's plausibly intentional for
	// ADIOS's OWN regression model (a queue-pressure-aware quantity is
	// arguably what you want for admission budgeting), so `latency`
	// itself and its use in latency_model_input() below are left
	// completely untouched. But #3/#6 (added in this patch series)
	// interpret deviation from prediction as "did service get worse",
	// which is a different question -- and under this metric, a change
	// in CONCURRENCY (fewer in-flight requests, not slower hardware)
	// can look identical to real degradation. Gets its own, unambiguous
	// per-request true latency instead, computed once here since
	// rq->io_start_time_ns is already confirmed non-zero above.
	u64 true_latency = now - rq->io_start_time_ns;

	u32 weight = 1;
	if (ad->is_rotational) {
		sector_t current_pos = blk_rq_pos(rq);
		// Only calculate seek distance if we have a valid last position.
		if (pc->last_completed_pos > 0) {
			u64 seek_distance = abs(
				(s64)current_pos - (s64)pc->last_completed_pos);
			if (seek_distance)
				weight = 65 - __builtin_clzll(seek_distance);
		}
		// Update (or re-synchronize) our knowledge of the head position.
		pc->last_completed_pos = current_pos + blk_rq_sectors(rq);
	}

	local_irq_restore(flags);

	// SAGA: self-detected degradation check -- runs BEFORE the outlier
	// exclusion below on purpose. A run of genuine bad-latency outliers
	// (excluded from polluting the smooth average, which is correct)
	// should still be allowed to trigger "something changed, forget
	// stale history faster" -- that's a different concern than average
	// accuracy, and outliers are exactly the signal we want here.
	// Also feeds the #6 depth controller below with the SAME overrun
	// classification, so both mechanisms agree on what "bad" means.
	// CHATGPT-REVIEW FIX #3: uses true_latency (this specific request's
	// own elapsed time), not the shared inter-completion-gap `latency`
	// -- see true_latency's own comment above for why.
	if (rd->pred_lat > 0) {
		struct latency_model *lm = &ad->latency_model[optype];
		u64 threshold = (rd->pred_lat * READ_ONCE(ad->lm_overrun_ratio_pct)) / 100;
		bool is_overrun = true_latency > threshold;

		if (is_overrun) {
			if (atomic_inc_return(&lm->consec_overruns) >=
					READ_ONCE(ad->lm_overrun_trigger_count)) {
				atomic_set(&lm->consec_overruns, 0);
				// CHATGPT-REVIEW: defer the actual shrink (allocation +
				// RCU swap) to the timer instead of doing it here, on
				// the completion hot path. timer_reduce below already
				// guarantees the timer fires within 100ms regardless.
				atomic_set(&lm->shrink_pending, 1);
			}
		} else {
			atomic_set(&lm->consec_overruns, 0);
		}

		adapt_depth_controller(ad, optype, true_latency, rd->pred_lat);
	}

	if (latency > ad->lat_model_latency_limit)
		return;

	latency_model_input(ad, &ad->latency_model[optype],
		rd->block_size, latency, rd->pred_lat, weight);
	timer_reduce(&ad->update_timer, jiffies + msecs_to_jiffies(100));
}

// Clean up after a request is finished
static void adios_finish_request(struct request *rq) {
	struct adios_data *ad = rq->q->elevator->elevator_data;

	if (rq->elv.priv[0]) {
		// Free adios_rq_data back to the memory pool
		mempool_free(get_rq_data(rq), ad->rq_data_pool);
		rq->elv.priv[0] = NULL;
	}
}

// Check if there are any requests available for dispatch
static bool adios_has_work(struct blk_mq_hw_ctx *hctx) {
	struct adios_data *ad = hctx->queue->elevator->elevator_data;

	return atomic_read(&ad->state) != 0;
}

// Initialize the scheduler-specific data for a hardware queue
static int adios_init_hctx(struct blk_mq_hw_ctx *hctx, unsigned int hctx_idx) {
	adios_depth_updated(hctx);
	return 0;
}

// Initialize the scheduler-specific data when initializing the request queue
static int adios_init_sched(struct request_queue *q, struct elevator_type *e) {
	struct adios_data *ad;
	struct elevator_queue *eq;
	int ret = -ENOMEM;
	u8 optype = 0;

	eq = elevator_alloc(q, e);
	if (!eq) {
		pr_err("adios: Failed to allocate the elevator\n");
		return ret;
	}

	ad = kzalloc_node(sizeof(*ad), GFP_KERNEL, q->node);
	if (!ad) {
		pr_err("adios: Failed to create adios_data\n");
		goto put_eq;
	}

	// Create a memory pool for adios_rq_data
	ad->rq_data_cache = kmem_cache_create("adios_rq_data",
						sizeof(struct adios_rq_data),
						0, SLAB_HWCACHE_ALIGN, NULL);
	if (!ad->rq_data_cache) {
		pr_err("adios: Failed to create rq_data_cache\n");
		goto free_ad;
	}

	ad->rq_data_pool = mempool_create_slab_pool(
						q->nr_requests, ad->rq_data_cache);
	if (!ad->rq_data_pool) {
		pr_err("adios: Failed to create rq_data_pool\n");
		goto destroy_rq_data_cache;
	}

	/* Create a memory pool for dl_group */
	ad->dl_group_pool = kmem_cache_create("dl_group_pool",
						sizeof(struct dl_group),
						0, SLAB_HWCACHE_ALIGN, NULL);
	if (!ad->dl_group_pool) {
		pr_err("adios: Failed to create dl_group_pool\n");
		goto destroy_rq_data_pool;
	}

	for (int i = 0; i < ADIOS_PQ_LEVELS; i++)
		INIT_LIST_HEAD(&ad->prio_queue[i]);

	for (u8 i = 0; i < ADIOS_DL_TYPES; i++) {
		ad->dl_tree[i] = RB_ROOT_CACHED;
		ad->dl_prio[i] = default_dl_prio[i];
	}
	ad->dl_bias = 0;

	for (u8 page = 0; page < ADIOS_BQ_PAGES; page++)
		for (optype = 0; optype < ADIOS_OPTYPES; optype++)
			INIT_LIST_HEAD(&ad->batch_queue[page][optype]);

	ad->aggr_buckets = kzalloc(sizeof(*ad->aggr_buckets), GFP_KERNEL);
	if (!ad->aggr_buckets) {
		pr_err("adios: Failed to allocate aggregation buckets\n");
		goto destroy_dl_group_pool;
	}

	ad->pcpu_completion = alloc_percpu(struct adios_pcpu_completion);
	if (!ad->pcpu_completion) {
		pr_err("adios: Failed to allocate per-CPU completion data\n");
		goto free_aggr_buckets;
	}

	for (optype = 0; optype < ADIOS_OPTYPES; optype++) {
		struct latency_model *model = &ad->latency_model[optype];
		struct latency_model_params *params;

		spin_lock_init(&model->update_lock);
		params = kzalloc(sizeof(*params), GFP_KERNEL);
		if (!params) {
			pr_err("adios: Failed to allocate latency_model_params\n");
			goto free_buckets;
		}
		params->last_update_jiffies = jiffies;
		RCU_INIT_POINTER(model->params, params);

		model->pcpu_buckets = alloc_percpu(struct lm_buckets);
		if (!model->pcpu_buckets) {
			pr_err("adios: Failed to allocate per-CPU buckets\n");
			kfree(params);
			goto free_buckets;
		}

		model->pcpu_snapshot = alloc_percpu(struct lm_buckets);
		if (!model->pcpu_snapshot) {
			pr_err("adios: Failed to allocate per-CPU snapshot\n");
			free_percpu(model->pcpu_buckets);
			kfree(params);
			goto free_buckets;
		}

		model->lm_shrink_at_kreqs  = default_lm_shrink_at_kreqs;
		model->lm_shrink_at_gbytes = default_lm_shrink_at_gbytes;
		model->lm_shrink_resist    = default_lm_shrink_resist;

		model->lm_block_size_threshold  = LM_BLOCK_SIZE_THRESHOLD_DEFAULT;
		model->lm_outlier_percentile    = LM_OUTLIER_PERCENTILE_DEFAULT;
		model->lm_interval_threshold_ms = LM_INTERVAL_THRESHOLD_DEFAULT;
		model->lm_samples_threshold     = LM_SAMPLES_THRESHOLD_DEFAULT;
		atomic_set(&model->consec_overruns, 0);
		atomic_set(&model->shrink_pending, 0);
	}

	// SAGA: auto-scale batch_limit defaults by the device's actual
	// configured queue depth (q->nr_requests, set by the block driver
	// at queue init -- portable across UFS/NVMe/SATA, no device-specific
	// probing needed). Baseline of 32 matches the original upstream
	// defaults (36 read / 72 write), which were tuned assuming roughly
	// that depth. Scale is clamped to [0.5x, 4x] so a very shallow or
	// very deep queue can't push batch_limit to a degenerate extreme.
	// This directly targets the batch_limit=128-on-a-QD32-device
	// regression observed empirically on UFS 2.2 today -- with this,
	// that specific misconfiguration can't happen from the default
	// path again (manual sysfs override still can, deliberately).
	{
		const u32 baseline_qd = 32;
		// CHATGPT-REVIEW: cast to u64 before multiplying -- q->nr_requests
		// is realistically in the tens-to-low-thousands range on any
		// real device, so this never actually overflows in practice, but
		// costs nothing to do correctly.
		u32 scale_permille =
			(u32)(((u64)q->nr_requests * 1000) / baseline_qd);

		if (scale_permille < 500)  scale_permille = 500;   // floor 0.5x
		if (scale_permille > 4000) scale_permille = 4000;  // ceil 4x

		for (optype = 0; optype < ADIOS_OPTYPES; optype++) {
			ad->latency_target[optype] = default_latency_target[optype];
			// SAGA: only READ/WRITE are auto-scaled by real queue depth --
			// DISCARD/OTHER keep their conservative upstream base value
			// unconditionally. Matches the same read/write-only scope
			// used by #6's depth controller; discards don't benefit from
			// deep batching (they're not latency-critical the way reads
			// are) and inflating batch_limit_discard on a very deep
			// queue device serves no purpose. Fixes an inconsistency
			// found on review -- the scaling loop previously applied to
			// all 4 optypes, silently harmless on this device only
			// because integer truncation happened to floor it back to 1
			// at this device's ~1.93x scale; would NOT stay harmless on
			// a device with a deeper queue (e.g. 4x ceiling -> discard
			// limit of 4 instead of 1).
			if (optype == ADIOS_READ || optype == ADIOS_WRITE) {
				ad->batch_limit[optype] = max_t(u32, 1,
					(default_batch_limit[optype] * scale_permille) / 1000);
			} else {
				ad->batch_limit[optype] = default_batch_limit[optype];
			}
		}
		pr_info("adios: nr_requests=%u -> batch_limit scale=%u.%03ux "
			"(read=%u write=%u)\n",
			q->nr_requests, scale_permille / 1000, scale_permille % 1000,
			ad->batch_limit[ADIOS_READ], ad->batch_limit[ADIOS_WRITE]);
	}

	eq->elevator_data = ad;

	ad->topapp_deadline_bonus = default_topapp_deadline_bonus;
	ad->topapp_deadline_bonus_target = default_topapp_deadline_bonus_target;
	ad->topapp_activation_delay_ms = default_topapp_activation_delay_ms;

	ad->is_rotational = !blk_queue_nonrot(q);
	ad->global_latency_window = (ad->is_rotational)?
		default_global_latency_window_rotational:
		default_global_latency_window;

	// SAGA: device-class heuristic adjustment of global_latency_window.
	// Honest scope note: this does NOT identify UFS generation (2.2 vs
	// 3.1 vs 4.0) -- doing that reliably would need SCSI/NVMe-specific
	// inquiry-string parsing that varies by driver stack and couldn't
	// be verified without the real device's kernel tree, so it's not
	// attempted here. What IS safely available and portable across
	// UFS/NVMe/SATA is the queue depth the block driver already
	// configured (q->nr_requests, same signal #2 uses) -- deeper queues
	// generally correlate with newer/faster interfaces (e.g. UFS with
	// MCQ), and can profitably use a *smaller* window (each batch page
	// fills with real requests faster, so it doesn't need to wait as
	// long to reach a useful size); shallow queues get more window to
	// compensate for higher per-op latency variance. Purely a shift on
	// top of the rotational-based default above, not a replacement.
	if (!ad->is_rotational) {
		if (q->nr_requests >= 128)
			ad->global_latency_window = ad->global_latency_window * 3 / 4;
		else if (q->nr_requests <= 16)
			ad->global_latency_window = ad->global_latency_window * 3 / 2;
	}

	ad->bq_refill_below_ratio = default_bq_refill_below_ratio;
	ad->lat_model_latency_limit = default_lat_model_latency_limit;
	ad->lm_confidence_samples = default_lm_confidence_samples;
	ad->lm_overrun_ratio_pct = default_lm_overrun_ratio_pct;
	ad->lm_overrun_trigger_count = default_lm_overrun_trigger_count;
	ad->lm_interactive_idle_ms = default_lm_interactive_idle_ms;
	ad->lm_interactive_boost_ns = default_lm_interactive_boost_ns;
	spin_lock_init(&ad->iactv_lock);
	spin_lock_init(&ad->lm_adapt_bounds_lock);
	memset(ad->iactv_table, 0, sizeof(ad->iactv_table));

	ad->lm_adapt_window_samples = default_lm_adapt_window_samples;
	ad->lm_adapt_bad_threshold_pct = default_lm_adapt_bad_threshold_pct;
	ad->lm_adapt_good_ratio_pct = default_lm_adapt_good_ratio_pct;
	ad->lm_adapt_bad_ratio_pct = default_lm_adapt_bad_ratio_pct;
	ad->lm_adapt_good_threshold_pct = default_lm_adapt_good_threshold_pct;
	ad->lm_adapt_step_permille = default_lm_adapt_step_permille;
	ad->lm_adapt_step_down_permille = default_lm_adapt_step_down_permille;
	ad->lm_adapt_cooldown_ms = default_lm_adapt_cooldown_ms;
	ad->lm_adapt_min_permille = default_lm_adapt_min_permille;
	ad->lm_adapt_max_permille = default_lm_adapt_max_permille;
	for (optype = 0; optype < ADIOS_OPTYPES; optype++) {
		atomic_set(&ad->window_good[0][optype], 0);
		atomic_set(&ad->window_good[1][optype], 0);
		atomic_set(&ad->window_bad[0][optype], 0);
		atomic_set(&ad->window_bad[1][optype], 0);
		atomic_set(&ad->window_total[0][optype], 0);
		atomic_set(&ad->window_total[1][optype], 0);
		atomic_set(&ad->window_buf_idx[optype], 0);
		atomic_set(&ad->depth_multiplier_permille[optype], 1000);
		WRITE_ONCE(ad->last_adjust_jiffies[optype], jiffies);
	}
	ad->batch_order = default_batch_order;
	ad->dl_prio[0] = default_read_priority;
	ad->compliance_flags = default_compliance_flags;

	ad->models_stable = false;

	atomic_set(&ad->state, 0);

	spin_lock_init(&ad->lock);
	spin_lock_init(&ad->pq_lock);
	spin_lock_init(&ad->bq_lock);

	timer_setup(&ad->update_timer, update_timer_callback, 0);

	timer_setup(&ad->topapp_activate_timer, topapp_activate_timer_callback, 0);
	if (ad->topapp_activation_delay_ms)
		mod_timer(&ad->topapp_activate_timer,
			jiffies + msecs_to_jiffies((unsigned int)ad->topapp_activation_delay_ms));

	INIT_WORK(&ad->shrink_work, shrink_work_fn);

	/* We dispatch from request queue wide instead of hw queue */
	blk_queue_flag_set(QUEUE_FLAG_SQ_SCHED, q);

	ad->queue = q;
	blk_stat_enable_accounting(q);

	q->elevator = eq;
	return 0;

free_buckets:
	pr_err("adios: Failed to allocate per-cpu buckets\n");
	while (optype-- > 0) {
		struct latency_model *prev_model = &ad->latency_model[optype];
		kfree(rcu_access_pointer(prev_model->params));
		free_percpu(prev_model->pcpu_snapshot);
		free_percpu(prev_model->pcpu_buckets);
	}
	free_percpu(ad->pcpu_completion);
free_aggr_buckets:
	kfree(ad->aggr_buckets);
destroy_dl_group_pool:
	kmem_cache_destroy(ad->dl_group_pool);
destroy_rq_data_pool:
	mempool_destroy(ad->rq_data_pool);
destroy_rq_data_cache:
	kmem_cache_destroy(ad->rq_data_cache);
free_ad:
	kfree(ad);
put_eq:
	kobject_put(&eq->kobj);
	return ret;
}

// Clean up and free resources when exiting the scheduler
static void adios_exit_sched(struct elevator_queue *e) {
	struct adios_data *ad = e->elevator_data;

	del_timer_sync(&ad->update_timer);
	del_timer_sync(&ad->topapp_activate_timer);
	cancel_work_sync(&ad->shrink_work);

	for (int i = 0; i < 2; i++)
		WARN_ON_ONCE(!list_empty(&ad->prio_queue[i]));

	/* REVIEW SUGGESTION: exit_sched previously only checked prio_queue.
	 * Scheduler switching (none <-> adios <-> mq-deadline, or the
	 * genhd.c boot enforcer forcing a switch) is exactly where lifetime
	 * bugs -- a request left behind in a structure this teardown
	 * doesn't look at -- would surface. These are diagnostic only
	 * (WARN_ON_ONCE, not a functional fix on their own); the belief is
	 * everything should already be empty by this point, so firing here
	 * means an insert/dispatch path is leaking a request somewhere. */
	for (int i = 0; i < 2; i++)
		WARN_ON_ONCE(!RB_EMPTY_ROOT(&ad->dl_tree[i].rb_root));
	for (int page = 0; page < ADIOS_BQ_PAGES; page++)
		for (u8 optype = 0; optype < ADIOS_OPTYPES; optype++)
			WARN_ON_ONCE(!list_empty(&ad->batch_queue[page][optype]));

	for (u8 i = 0; i < ADIOS_OPTYPES; i++) {
		struct latency_model *model = &ad->latency_model[i];
		struct latency_model_params *params = rcu_access_pointer(model->params);

		RCU_INIT_POINTER(model->params, NULL);
		kfree_rcu(params, rcu);

		free_percpu(model->pcpu_snapshot);
		free_percpu(model->pcpu_buckets);
	}

	synchronize_rcu();

	free_percpu(ad->pcpu_completion);
	kfree(ad->aggr_buckets);

	mempool_destroy(ad->rq_data_pool);
	kmem_cache_destroy(ad->rq_data_cache);

	if (ad->dl_group_pool)
		kmem_cache_destroy(ad->dl_group_pool);

	blk_stat_disable_accounting(ad->queue);

	kfree(ad);
}

static void sideload_latency_model(
		struct latency_model *model, u64 base, u64 slope) {
	struct latency_model_params *old_params, *new_params;
	unsigned long flags;

	new_params = kzalloc(sizeof(*new_params), GFP_KERNEL);
	if (!new_params)
		return;

	/*
	 * REVIEW FIX (found after the v3.3.5 per-CPU race fix -- that fix
	 * introduced this one): on_each_cpu(..., wait=1) must be called
	 * with local IRQs enabled -- kernel/smp.c's own
	 * smp_call_function_many_cond() asserts exactly this
	 * (lockdep_assert_irqs_enabled()) and documents why in its own
	 * comment: "Can deadlock when called with interrupts disabled".
	 * The previous version of this function called
	 * lm_reset_pcpu_buckets() (which does the on_each_cpu) *after*
	 * spin_lock_irqsave() below -- i.e. with this CPU's IRQs already
	 * off. Moved here, before the lock, so the per-CPU reset completes
	 * on ordinary IRQs-enabled task-context footing, same as any other
	 * on_each_cpu() caller. This does mean a completion racing this
	 * exact call could count against the about-to-be-replaced old
	 * params into a freshly-zeroed bucket instead of the also-about-
	 * to-be-replaced old bucket state -- a benign, one-sample-class
	 * ordering skew, not a correctness issue: this whole function only
	 * ever runs on an explicit, rare, sysfs-triggered reset, and every
	 * other timing skew in this file's stats paths already accepts the
	 * same order of imprecision on purpose (see the double-buffer
	 * depth controller's own comment for a documented example of that
	 * same tradeoff).
	 */
	lm_reset_pcpu_buckets(model);

	spin_lock_irqsave(&model->update_lock, flags);

	old_params = rcu_dereference_protected(model->params,
			lockdep_is_held(&model->update_lock));

	new_params->last_update_jiffies = jiffies;

	// Initialize base and its statistics as a single sample.
	new_params->base = base;
	new_params->small_sum_delay = base;
	new_params->small_count = 1;

	// Initialize slope and its statistics as a single sample.
	new_params->slope = slope;
	new_params->large_sum_delay = slope;
	new_params->large_sum_bsize = 1024; /* Corresponds to 1 KiB */
	new_params->large_count = 1;

	rcu_assign_pointer(model->params, new_params);
	spin_unlock_irqrestore(&model->update_lock, flags);

	kfree_rcu(old_params, rcu);
}

// Define sysfs attributes for operation types
#define SYSFS_OPTYPE_DECL(name, optype) \
static ssize_t adios_lat_model_##name##_show( \
		struct elevator_queue *e, char *page) { \
	struct adios_data *ad = e->elevator_data; \
	struct latency_model *model = &ad->latency_model[optype]; \
	struct latency_model_params *params; \
	ssize_t len = 0; \
	u64 base, slope; \
	rcu_read_lock(); \
	params = rcu_dereference(model->params); \
	base = params->base; \
	slope = params->slope; \
	rcu_read_unlock(); \
	len += sprintf(page,       "base : %llu ns\n", base); \
	len += sprintf(page + len, "slope: %llu ns/KiB\n", slope); \
	return len; \
} \
static ssize_t adios_lat_model_##name##_store( \
		struct elevator_queue *e, const char *page, size_t count) { \
	struct adios_data *ad = e->elevator_data; \
	struct latency_model *model = &ad->latency_model[optype]; \
	u64 base, slope; \
	int ret; \
	ret = sscanf(page, "%llu %llu", &base, &slope); \
	if (ret != 2) \
		return -EINVAL; \
	sideload_latency_model(model, base, slope); \
	reset_buckets(ad->aggr_buckets); \
	return count; \
} \
static ssize_t adios_lat_target_##name##_show( \
		struct elevator_queue *e, char *page) { \
	struct adios_data *ad = e->elevator_data; \
	return sprintf(page, "%llu\n", ad->latency_target[optype]); \
} \
static ssize_t adios_lat_target_##name##_store( \
		struct elevator_queue *e, const char *page, size_t count) { \
	struct adios_data *ad = e->elevator_data; \
	unsigned long nsec; \
	int ret; \
	ret = kstrtoul(page, 10, &nsec); \
	if (ret) \
		return ret; \
	sideload_latency_model(&ad->latency_model[optype], 0, 0); \
	WRITE_ONCE(ad->latency_target[optype], nsec); \
	return count; \
} \
static ssize_t adios_batch_limit_##name##_show( \
		struct elevator_queue *e, char *page) { \
	struct adios_data *ad = e->elevator_data; \
	return sprintf(page, "%u\n", ad->batch_limit[optype]); \
} \
static ssize_t adios_batch_limit_##name##_store( \
		struct elevator_queue *e, const char *page, size_t count) { \
	/* REVIEW FIX: was kstrtoul() into an unsigned long (64-bit on \
	 * arm64) with no upper-bound check, then assigned into batch_limit[]\
	 * (u32) -- a value like 4294967296 parsed fine and silently \
	 * truncated to 0 on assignment. kstrtou32() rejects anything that \
	 * doesn't fit in u32 outright (-ERANGE), matching the field's own \
	 * width instead of relying on an unwritten range check. */ \
	u32 max_batch; \
	int ret; \
	struct adios_data *ad = e->elevator_data; \
	ret = kstrtou32(page, 10, &max_batch); \
	if (ret || max_batch == 0) \
		return -EINVAL; \
	ad->batch_limit[optype] = max_batch; \
	return count; \
}

SYSFS_OPTYPE_DECL(read, ADIOS_READ);
SYSFS_OPTYPE_DECL(write, ADIOS_WRITE);
SYSFS_OPTYPE_DECL(discard, ADIOS_DISCARD);

// Show the maximum batch size actually achieved for each operation type
static ssize_t adios_batch_actual_max_show(
		struct elevator_queue *e, char *page) {
	struct adios_data *ad = e->elevator_data;
	u32 total_count, read_count, write_count, discard_count;

	/* REVIEW FIX: the hot-path writer (fill_batch_queues(), called only
	 * from dispatch_from_bq() under bq_lock) updates these under that
	 * lock; this show and the reset store below didn't take it, so a
	 * concurrent read/reset could observe a torn update. Low real-world
	 * severity (pure observability counters, never fed back into a
	 * scheduling decision), but the fix is free. */
	scoped_guard(spinlock_irqsave, &ad->bq_lock) {
		total_count = ad->batch_actual_max_total;
		read_count = ad->batch_actual_max_size[ADIOS_READ];
		write_count = ad->batch_actual_max_size[ADIOS_WRITE];
		discard_count = ad->batch_actual_max_size[ADIOS_DISCARD];
	}

	return sprintf(page,
		"Total  : %u\nDiscard: %u\nRead   : %u\nWrite  : %u\n",
		total_count, discard_count, read_count, write_count);
}

// SAGA #6: observability for the depth controller -- without this
// there's no way to tell from userspace whether it's ever actually
// adjusting anything. Shows live multiplier (permille) and the
// resulting effective limit for read/write.
static ssize_t adios_depth_controller_show(
		struct elevator_queue *e, char *page) {
	struct adios_data *ad = e->elevator_data;
	s32 rmult = atomic_read(&ad->depth_multiplier_permille[ADIOS_READ]);
	s32 wmult = atomic_read(&ad->depth_multiplier_permille[ADIOS_WRITE]);

	return sprintf(page,
		"read : multiplier=%d.%03dx effective_limit=%u (base=%u)\n"
		"write: multiplier=%d.%03dx effective_limit=%u (base=%u)\n",
		rmult / 1000, rmult % 1000, effective_batch_limit(ad, ADIOS_READ),
		ad->batch_limit[ADIOS_READ],
		wmult / 1000, wmult % 1000, effective_batch_limit(ad, ADIOS_WRITE),
		ad->batch_limit[ADIOS_WRITE]);
}

#define SYSFS_ULL_DECL(field, min_val, max_val) \
static ssize_t adios_##field##_show( \
		struct elevator_queue *e, char *page) { \
	struct adios_data *ad = e->elevator_data; \
	return sprintf(page, "%llu\n", ad->field); \
} \
static ssize_t adios_##field##_store( \
		struct elevator_queue *e, const char *page, size_t count) { \
	struct adios_data *ad = e->elevator_data; \
	unsigned long val; \
	int ret; \
	ret = kstrtoul(page, 10, &val); \
	if (ret || val < (min_val) || val > (max_val)) \
		return -EINVAL; \
	WRITE_ONCE(ad->field, val); \
	return count; \
}

SYSFS_ULL_DECL(global_latency_window, 0, ULLONG_MAX)
SYSFS_ULL_DECL(compliance_flags, 0, ULLONG_MAX)
SYSFS_ULL_DECL(topapp_deadline_bonus, 0, NSEC_PER_SEC)
SYSFS_ULL_DECL(topapp_deadline_bonus_target, 0, NSEC_PER_SEC)
SYSFS_ULL_DECL(topapp_activation_delay_ms, 0, 600000ULL)

#define SYSFS_INT_DECL(field, min_val, max_val) \
static ssize_t adios_##field##_show( \
		struct elevator_queue *e, char *page) { \
	struct adios_data *ad = e->elevator_data; \
	return sprintf(page, "%d\n", ad->field); \
} \
static ssize_t adios_##field##_store( \
		struct elevator_queue *e, const char *page, size_t count) { \
	struct adios_data *ad = e->elevator_data; \
	int val; \
	int ret; \
	ret = kstrtoint(page, 10, &val); \
	if (ret || val < (min_val) || val > (max_val)) \
		return -EINVAL; \
	WRITE_ONCE(ad->field, val); \
	return count; \
}

SYSFS_INT_DECL(bq_refill_below_ratio, 0, 100)
SYSFS_INT_DECL(lat_model_latency_limit, 0, 2*NSEC_PER_SEC)
SYSFS_INT_DECL(batch_order, ADIOS_BO_OPTYPE, !!ad->is_rotational)
// CHATGPT-REVIEW: minimum raised 0 -> 8. lm_confidence_samples=0 made
// model_is_confident()'s check ("count >= 0") always true, silently
// disabling the confidence gate entirely -- defeating its whole
// purpose (never trusting a near-empty model's prediction). Flagged in
// both review rounds; missed in the first round's fix pass.
SYSFS_INT_DECL(lm_confidence_samples, 8, 100000)
SYSFS_INT_DECL(lm_overrun_ratio_pct, 100, 1000)
SYSFS_INT_DECL(lm_overrun_trigger_count, 1, 10000)
SYSFS_INT_DECL(lm_interactive_idle_ms, 0, 60000)
// SAGA FIX (3rd review round): lm_interactive_boost_ns is a u64 field,
// but SYSFS_INT_DECL is built for int/u32 fields -- its store uses
// kstrtoint (32-bit parse) and its show does sprintf("%d", ad->field),
// reading only 4 of the field's 8 bytes through varargs. That's a
// format-string type mismatch (UB, triggers -Wformat, can be
// -Werror'd into a hard build failure on strict GKI toolchains) and
// an artificially narrowed store range. SYSFS_ULL_DECL is the correct,
// already-existing macro for u64 fields (see global_latency_window/
// compliance_flags above) -- kstrtoul + "%llu", no mismatch, and the
// full range up to ULLONG_MAX is available if the bound below is ever
// widened past 2*NSEC_PER_SEC.
SYSFS_ULL_DECL(lm_interactive_boost_ns, 0, 2*NSEC_PER_SEC)
SYSFS_INT_DECL(lm_adapt_window_samples, 4, 100000)
SYSFS_INT_DECL(lm_adapt_bad_threshold_pct, 1, 100)
// CHATGPT-REVIEW FIX: good_ratio_pct/bad_ratio_pct and min/max_permille
// were independently bounded but not cross-validated -- sysfs allowed
// e.g. good=150/bad=110 (overlapping, order-dependent classification)
// or min=2000/max=1000 (inverted clamp range, silently broken
// controller). These 4 custom setters replace the generic
// SYSFS_INT_DECL for just these fields, adding the cross-check.
// SAGA FIX (3rd review round): the cross-check below reads other_field,
// then writes field, with no synchronization between the two -- two
// concurrent sysfs writers (one per paired attribute) can each read the
// other's PRE-write value, both pass their own check against a value
// that's about to change, and still leave an invalid combo once both
// stores land (e.g. good=150 then bad=110 arriving in the window
// between each other's read and write). ad->lm_adapt_bounds_lock now
// wraps the read-check-write as one section, shared across all 4
// paired attributes (good/bad ratio, min/max permille) since a writer
// to any one of them must be serialized against writers to its actual
// pair. Sysfs writes are rare admin actions, never hot-path, so the
// hold time here is irrelevant to performance.
#define SYSFS_INT_DECL_PAIRED(field, min_val, max_val, other_field, is_lower) \
static ssize_t adios_##field##_show( \
		struct elevator_queue *e, char *page) { \
	struct adios_data *ad = e->elevator_data; \
	return sprintf(page, "%d\n", ad->field); \
} \
static ssize_t adios_##field##_store( \
		struct elevator_queue *e, const char *page, size_t count) { \
	struct adios_data *ad = e->elevator_data; \
	unsigned long flags; \
	int val; \
	int ret; \
	int rc = count; \
	ret = kstrtoint(page, 10, &val); \
	if (ret || val < (min_val) || val > (max_val)) \
		return -EINVAL; \
	spin_lock_irqsave(&ad->lm_adapt_bounds_lock, flags); \
	if (is_lower) { \
		if (val >= (int)ad->other_field) \
			rc = -EINVAL; \
	} else { \
		if (val <= (int)ad->other_field) \
			rc = -EINVAL; \
	} \
	if (rc >= 0) \
		WRITE_ONCE(ad->field, val); \
	spin_unlock_irqrestore(&ad->lm_adapt_bounds_lock, flags); \
	return rc; \
}

SYSFS_INT_DECL_PAIRED(lm_adapt_good_ratio_pct, 100, 500, lm_adapt_bad_ratio_pct, 1)
SYSFS_INT_DECL_PAIRED(lm_adapt_bad_ratio_pct, 100, 500, lm_adapt_good_ratio_pct, 0)
SYSFS_INT_DECL(lm_adapt_good_threshold_pct, 1, 100)
SYSFS_INT_DECL(lm_adapt_step_permille, 10, 1000)
SYSFS_INT_DECL(lm_adapt_step_down_permille, 10, 1000)
SYSFS_INT_DECL(lm_adapt_cooldown_ms, 0, 60000)
SYSFS_INT_DECL_PAIRED(lm_adapt_min_permille, 50, 1000, lm_adapt_max_permille, 1)
SYSFS_INT_DECL_PAIRED(lm_adapt_max_permille, 1000, 10000, lm_adapt_min_permille, 0)

// Show the read priority
static ssize_t adios_read_priority_show(
		struct elevator_queue *e, char *page) {
	struct adios_data *ad = e->elevator_data;
	return sprintf(page, "%d\n", ad->dl_prio[0]);
}

// Set the read priority
static ssize_t adios_read_priority_store(
		struct elevator_queue *e, const char *page, size_t count) {
	struct adios_data *ad = e->elevator_data;
	int prio;
	int ret;

	ret = kstrtoint(page, 10, &prio);
	if (ret || prio < -20 || prio > 19)
		return -EINVAL;

	guard(spinlock_irqsave)(&ad->lock);
	WRITE_ONCE(ad->dl_prio[0], prio);
	ad->dl_bias = 0;

	return count;
}

// Reset batch queue statistics
static ssize_t adios_reset_bq_stats_store(
		struct elevator_queue *e, const char *page, size_t count) {
	struct adios_data *ad = e->elevator_data;
	unsigned long val;
	int ret;

	ret = kstrtoul(page, 10, &val);
	if (ret || val != 1)
		return -EINVAL;

	scoped_guard(spinlock_irqsave, &ad->bq_lock) {
		for (u8 i = 0; i < ADIOS_OPTYPES; i++)
			ad->batch_actual_max_size[i] = 0;

		ad->batch_actual_max_total = 0;
	}

	return count;
}

// Reset the latency model parameters or load them from user input
static ssize_t adios_reset_lat_model_store(
		struct elevator_queue *e, const char *page, size_t count)
{
	struct adios_data *ad = e->elevator_data;
	struct latency_model *model;
	int ret;

	/*
	 * Differentiate between two modes based on input format:
	 * 1. "1": Fully reset the model (backward compatibility).
	 * 2. "R_base R_slope W_base W_slope D_base D_slope": Load values.
	 */
	if (!strchr(page, ' ')) {
		// Mode 1: Full reset.
		unsigned long val;

		ret = kstrtoul(page, 10, &val);
		if (ret || val != 1)
			return -EINVAL;

		for (u8 i = 0; i < ADIOS_OPTYPES; i++) {
			model = &ad->latency_model[i];
			sideload_latency_model(model, 0, 0);
		}
	} else {
		// Mode 2: Load initial values for all latency models.
		u64 params[3][2]; /* 0:base, 1:slope for R, W, D */

		ret = sscanf(page, "%llu %llu %llu %llu %llu %llu",
			&params[ADIOS_READ   ][0], &params[ADIOS_READ   ][1],
			&params[ADIOS_WRITE  ][0], &params[ADIOS_WRITE  ][1],
			&params[ADIOS_DISCARD][0], &params[ADIOS_DISCARD][1]);

		if (ret != 6)
			return -EINVAL;

		for (u8 i = ADIOS_READ; i <= ADIOS_DISCARD; i++) {
			model = &ad->latency_model[i];
			sideload_latency_model(model, params[i][0], params[i][1]);
		}
	}
	reset_buckets(ad->aggr_buckets);

	return count;
}

// Show the ADIOS version
static ssize_t adios_version_show(struct elevator_queue *e, char *page) {
	return sprintf(page, "%s\n", ADIOS_VERSION);
}

// Define sysfs attributes for dynamic thresholds
#define SHRINK_THRESHOLD_ATTR_RW(name, model_field, min_value, max_value) \
static ssize_t adios_shrink_##name##_store( \
		struct elevator_queue *e, const char *page, size_t count) { \
	struct adios_data *ad = e->elevator_data; \
	unsigned long val; \
	int ret; \
	ret = kstrtoul(page, 10, &val); \
	if (ret || val < min_value || val > max_value) \
		return -EINVAL; \
	for (u8 i = 0; i < ADIOS_OPTYPES; i++) { \
		struct latency_model *model = &ad->latency_model[i]; \
		unsigned long flags; \
		spin_lock_irqsave(&model->update_lock, flags); \
		model->model_field = val; \
		spin_unlock_irqrestore(&model->update_lock, flags); \
	} \
	return count; \
} \
static ssize_t adios_shrink_##name##_show( \
		struct elevator_queue *e, char *page) { \
	struct adios_data *ad = e->elevator_data; \
	u32 val = 0; \
	unsigned long flags; \
	struct latency_model *model = &ad->latency_model[0]; \
	spin_lock_irqsave(&model->update_lock, flags); \
	val = model->model_field; \
	spin_unlock_irqrestore(&model->update_lock, flags); \
	return sprintf(page, "%u\n", val); \
}

SHRINK_THRESHOLD_ATTR_RW(at_kreqs,  lm_shrink_at_kreqs,  1, 100000)
SHRINK_THRESHOLD_ATTR_RW(at_gbytes, lm_shrink_at_gbytes, 1,   1000)
SHRINK_THRESHOLD_ATTR_RW(resist,    lm_shrink_resist,    1,      3)

/* Formerly hardcoded #defines, now runtime-tunable per-model fields.
 * Reuses the same macro pattern as SHRINK_THRESHOLD_ATTR_RW: one sysfs
 * knob applies uniformly to all optype models (read/write/discard/other). */
#define LM_CONST_ATTR_RW(name, model_field, min_value, max_value) \
static ssize_t adios_##name##_store( \
		struct elevator_queue *e, const char *page, size_t count) { \
	struct adios_data *ad = e->elevator_data; \
	unsigned long val; \
	int ret; \
	ret = kstrtoul(page, 10, &val); \
	if (ret || val < (min_value) || val > (max_value)) \
		return -EINVAL; \
	for (u8 i = 0; i < ADIOS_OPTYPES; i++) { \
		struct latency_model *model = &ad->latency_model[i]; \
		unsigned long flags; \
		spin_lock_irqsave(&model->update_lock, flags); \
		model->model_field = val; \
		spin_unlock_irqrestore(&model->update_lock, flags); \
	} \
	return count; \
} \
static ssize_t adios_##name##_show( \
		struct elevator_queue *e, char *page) { \
	struct adios_data *ad = e->elevator_data; \
	u32 val = 0; \
	unsigned long flags; \
	struct latency_model *model = &ad->latency_model[0]; \
	spin_lock_irqsave(&model->update_lock, flags); \
	val = model->model_field; \
	spin_unlock_irqrestore(&model->update_lock, flags); \
	return sprintf(page, "%u\n", val); \
}

// CHATGPT-REVIEW FIX #2: lm_block_size_threshold is NOT like the other
// 3 LM constants generated by LM_CONST_ATTR_RW below -- changing it
// retroactively changes the MEANING of every small_count/large_count
// sample already accumulated (a request classified "small" under the
// old threshold might be "large" under the new one, or vice versa).
// lm_outlier_percentile/lm_interval_threshold_ms/lm_samples_threshold
// don't have this problem -- they only affect how FUTURE
// recalculations behave, not which bucket past samples already landed
// in. Custom setter: on an actual value change, resets the model for
// that optype the same way the native reset_lat_model does
// (sideload_latency_model(model, 0, 0)) so stale bucket history can't
// silently corrupt confidence/prediction under the new split point.
static ssize_t adios_lm_block_size_threshold_store(
		struct elevator_queue *e, const char *page, size_t count) {
	struct adios_data *ad = e->elevator_data;
	unsigned long val;
	int ret;

	ret = kstrtoul(page, 10, &val);
	if (ret || val < 512 || val > 1048576)
		return -EINVAL;

	for (u8 i = 0; i < ADIOS_OPTYPES; i++) {
		struct latency_model *model = &ad->latency_model[i];
		unsigned long flags;
		bool changed;

		spin_lock_irqsave(&model->update_lock, flags);
		changed = (model->lm_block_size_threshold != val);
		WRITE_ONCE(model->lm_block_size_threshold, val);
		spin_unlock_irqrestore(&model->update_lock, flags);

		if (changed)
			sideload_latency_model(model, 0, 0);
	}
	return count;
}
static ssize_t adios_lm_block_size_threshold_show(
		struct elevator_queue *e, char *page) {
	struct adios_data *ad = e->elevator_data;
	u32 val = 0;
	unsigned long flags;
	struct latency_model *model = &ad->latency_model[0];
	spin_lock_irqsave(&model->update_lock, flags);
	val = model->lm_block_size_threshold;
	spin_unlock_irqrestore(&model->update_lock, flags);
	return sprintf(page, "%u\n", val);
}

LM_CONST_ATTR_RW(lm_outlier_percentile,    lm_outlier_percentile,       1,     100)
LM_CONST_ATTR_RW(lm_interval_threshold_ms, lm_interval_threshold_ms,    50,   60000)
LM_CONST_ATTR_RW(lm_samples_threshold,     lm_samples_threshold,        16, 100000)

// NOTE: model save/restore across boots does NOT need a custom attr --
// lat_model_read / lat_model_write already exist upstream (see
// sideload_latency_model() below) and do exactly this. Removed the
// redundant lm_seed_read/lm_seed_write that duplicated them.

// Define sysfs attributes
#define AD_ATTR(name, show_func, store_func) \
	__ATTR(name, 0644, show_func, store_func)
#define AD_ATTR_RW(name) \
	__ATTR(name, 0644, adios_##name##_show, adios_##name##_store)
#define AD_ATTR_RO(name) \
	__ATTR(name, 0444, adios_##name##_show, NULL)
#define AD_ATTR_WO(name) \
	__ATTR(name, 0200, NULL, adios_##name##_store)

// Define sysfs attributes for ADIOS scheduler
static struct elv_fs_entry adios_sched_attrs[] = {
	AD_ATTR_RO(batch_actual_max),
	AD_ATTR_RW(bq_refill_below_ratio),
	AD_ATTR_RW(global_latency_window),
	AD_ATTR_RW(lat_model_latency_limit),
	AD_ATTR_RW(lm_confidence_samples),
	AD_ATTR_RW(lm_overrun_ratio_pct),
	AD_ATTR_RW(lm_overrun_trigger_count),
	AD_ATTR_RW(lm_interactive_idle_ms),
	AD_ATTR_RW(lm_interactive_boost_ns),

	AD_ATTR_RW(lm_adapt_window_samples),
	AD_ATTR_RW(lm_adapt_bad_threshold_pct),
	AD_ATTR_RW(lm_adapt_good_ratio_pct),
	AD_ATTR_RW(lm_adapt_bad_ratio_pct),
	AD_ATTR_RW(lm_adapt_good_threshold_pct),
	AD_ATTR_RW(lm_adapt_step_permille),
	AD_ATTR_RW(lm_adapt_step_down_permille),
	AD_ATTR_RW(lm_adapt_cooldown_ms),
	AD_ATTR_RW(lm_adapt_min_permille),
	AD_ATTR_RW(lm_adapt_max_permille),
	AD_ATTR_RO(depth_controller),
	AD_ATTR_RW(batch_order),
	AD_ATTR_RW(compliance_flags),

	AD_ATTR_RW(batch_limit_read),
	AD_ATTR_RW(batch_limit_write),
	AD_ATTR_RW(batch_limit_discard),

	AD_ATTR_RW(lat_model_read),
	AD_ATTR_RW(lat_model_write),
	AD_ATTR_RW(lat_model_discard),

	AD_ATTR_RW(lat_target_read),
	AD_ATTR_RW(lat_target_write),
	AD_ATTR_RW(lat_target_discard),

	AD_ATTR_RW(shrink_at_kreqs),
	AD_ATTR_RW(shrink_at_gbytes),
	AD_ATTR_RW(shrink_resist),

	AD_ATTR_RW(lm_block_size_threshold),
	AD_ATTR_RW(lm_outlier_percentile),
	AD_ATTR_RW(lm_interval_threshold_ms),
	AD_ATTR_RW(lm_samples_threshold),

	AD_ATTR_RW(read_priority),
	AD_ATTR_RW(topapp_deadline_bonus),
	AD_ATTR_RW(topapp_deadline_bonus_target),
	AD_ATTR_RW(topapp_activation_delay_ms),

	AD_ATTR_WO(reset_bq_stats),
	AD_ATTR_WO(reset_lat_model),
	AD_ATTR(adios_version, adios_version_show, NULL),

	__ATTR_NULL
};

// Define the ADIOS scheduler type
static struct elevator_type mq_adios = {
	.ops = {
		/*
		 * REVIEW FIX: these were elv_rb_latter_request/elv_rb_former_
		 * request -- generic block-layer helpers that operate directly
		 * on rq->rb_node, assuming one rb-tree node per request (how
		 * mq-deadline uses its tree). ADIOS's dl_tree doesn't work that
		 * way: nodes are struct dl_group (one per distinct deadline),
		 * each holding every same-deadline request on a plain list
		 * (rd->dl_node) -- rq->rb_node is never touched by ADIOS's own
		 * insert/remove code at all. These callbacks ARE reachable from
		 * a generic path independent of ADIOS's own .bio_merge/
		 * .request_merged/.requests_merged: block/blk-merge.c's
		 * attempt_back_merge()/attempt_front_merge() call
		 * elv_latter_request()/elv_former_request() directly, which
		 * would read whatever garbage/stale data happens to be sitting
		 * in rq->rb_node (uninitialized, or leftover from a prior
		 * scheduler's use of the same request slot) and walk it as if
		 * it were a real tree.
		 * Confirmed safe to just omit both: elv_latter_request()/
		 * elv_former_request() (block/elevator.c) already check
		 * `if (e->type->ops.next_request)` / `.former_request` before
		 * calling, so a NULL op here means that one generic fallback
		 * merge attempt becomes a no-op instead of undefined behavior --
		 * ADIOS's own bio_merge/request_merged/requests_merged remain
		 * fully in effect either way, since they're a separate,
		 * independent merge path.
		 */
		.limit_depth		= adios_limit_depth,
		.depth_updated		= adios_depth_updated,
		.request_merged		= adios_request_merged,
		.requests_merged	= adios_merged_requests,
		.bio_merge			= adios_bio_merge,
		.insert_requests	= adios_insert_requests,
		.prepare_request	= adios_prepare_request,
		.dispatch_request	= adios_dispatch_request,
		.completed_request	= adios_completed_request,
		.finish_request		= adios_finish_request,
		.has_work			= adios_has_work,
		.init_hctx			= adios_init_hctx,
		.init_sched			= adios_init_sched,
		.exit_sched			= adios_exit_sched,
	},
	.elevator_attrs = adios_sched_attrs,
	.elevator_name = "adios",
	.elevator_owner = THIS_MODULE,
};
MODULE_ALIAS("mq-adios-iosched");

#define ADIOS_PROGNAME "Adaptive Deadline I/O Scheduler"
#define ADIOS_AUTHOR   "Masahito Suzuki"

// Initialize the ADIOS scheduler module
static int __init adios_init(void) {
	printk(KERN_INFO "%s %s by %s\n",
		ADIOS_PROGNAME, ADIOS_VERSION, ADIOS_AUTHOR);
	return elv_register(&mq_adios);
}

// Exit the ADIOS scheduler module
static void __exit adios_exit(void) {
	elv_unregister(&mq_adios);
}

/*
 * REVIEW FIX: CONFIG_MQ_IOSCHED_ADIOS is `tristate` (can be built as
 * =m), but this used to call subsys_initcall(adios_init)
 * unconditionally with no #ifdef MODULE branch. subsys_initcall (like
 * every initcall level) only ever runs for code linked directly into
 * vmlinux -- the module loader does not process initcall sections at
 * all, it calls whatever module_init()'s argument function is. For a
 * =m build, that means adios_init() would never run: insmod would
 * "succeed" with elv_register() never called, and selecting "adios"
 * via sysfs would just fail with no explanation.
 *
 * subsys_initcall, not module_init (== device_initcall), for the
 * *built-in* (=y) case specifically: the UFS-MTK platform driver that
 * actually creates these block devices registers at device_initcall
 * too (module_platform_driver() in ufs-mediatek.c), so it's a
 * link-order coin flip which one wins the race to run first — and
 * elevator_init_mq() only ever calls elevator_get_default() once, at
 * queue creation, before the disk is even added. Lose that race and
 * elevator_get(q, "adios", false) below finds nothing registered yet,
 * elevator_init_mq() just returns with no scheduler set, and nothing
 * ever retries — that queue is stuck without a default forever. Same
 * reasoning as why CONFIG_MQ_IOSCHED_DEFAULT_ADIOS alone doesn't
 * reliably win: it only changes *which* elevator_get_default() asks
 * for, not *when* elevator_init_mq() runs relative to UFS-MTK probe.
 * subsys_initcall strictly precedes every device_initcall system-wide
 * (no exceptions, no probe-deferral timing games), so this closes the
 * race outright instead of leaving it to link order.
 *
 * None of that race-avoidance reasoning applies to a =m build at all:
 * a module loads long after every built-in driver's initcalls have
 * already run (UFS-MTK included, if it's built-in, which is the usual
 * case for core storage), so a loadable ADIOS could never win that
 * race regardless of initcall level -- it can only ever be selected
 * manually via sysfs after loading, a case elevator_init_mq()'s
 * default-selection timing doesn't even apply to. module_init() is
 * simply the correct, and only working, option there.
 */
#ifdef MODULE
module_init(adios_init);
#else
subsys_initcall(adios_init);
#endif
module_exit(adios_exit);

MODULE_AUTHOR(ADIOS_AUTHOR);
MODULE_LICENSE("GPL");
MODULE_DESCRIPTION(ADIOS_PROGNAME);
