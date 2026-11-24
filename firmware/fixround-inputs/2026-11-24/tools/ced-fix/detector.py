"""End-of-travel detector (CAL end detector spec), written the way the firmware runs it.

ced-fix: the design reference (ced-design/detector.py) plus the 2026-11-24 review fixes,
on by default: F1 (slow-arrival switch also <= 90% of iRef), F2 (a short curl's stall
with < 12 normal samples since arming and its last edge < 40 samples after arming ends
static, a retract only with med4 <= iRef), F4 (an unarmed curl that moved >= 12 edges
uses the 1000 ms window).

Integer arithmetic only. One sample every CAL_END_SAMPLE_MS (50 ms) of the driven
joint, counted from the start of the drive:
  e = hall drive edges since the drive started (hallDriveEdges(sign, count - startHall))
  i = motor current in tenths of an IS count (10 x the mean of avgIS over the sample period)
A ring of the last CAL_END_RING samples keeps both (firmware: int16 + uint16 each).

States after the PWM ramp has reached the move PWM:
  unarmed  -> static stall rules only (a drive with no running phase)
  armed    -> switch rule (rod retracting) / mechanical rule (rod extending), stuck rule
"""
from dataclasses import dataclass, field

SAMPLE_MS = 50


@dataclass
class Params:
    ring: int = 24                     # CAL_END_RING samples kept (>= pre_samples + sw_samples + 1 and > stall window)
    # arming
    learn_delay_ms: int = 200          # CAL_END_LEARN_DELAY_MS after the ramp
    min_run_edges_200: int = 4         # CAL_END_RUN_MIN_EDGES per 200 ms (20 edges/s)
    arm_samples: int = 6               # CAL_END_ARM_SAMPLES consecutive running samples (300 ms)
    arm_min_edges: int = 12            # CAL_HALL_CURL_EVIDENCE_EDGES
    # references while armed
    normal_speed_pct: int = 75         # CAL_END_NORMAL_SPEED_PCT of vRef: the sample updates iRef/vRef
    track_step: int = 3                # CAL_END_IREF_STEP tenths per sample
    # switch end (rod retracting)
    sw_samples: int = 3                # CAL_END_SWITCH_STOP_SAMPLES without an edge (150 ms)
    pre_samples: int = 20              # CAL_END_PRE_SAMPLES: reference = median of the 20 samples up to the last edge
    arrive_fast_pct: int = 50          # CAL_END_ARRIVE_FAST_PCT: v200 ending at the last edge >= 50% vRef
    sw_pct_fast: int = 88              # CAL_END_SWITCH_FAST_PCT: every stopped sample <= 88% of the reference
    sw_pct_slow: int = 70              # CAL_END_SWITCH_SLOW_PCT: after a slow arrival, <= 70%
    sw_below_min: bool = True          # and below a low percentile of the reference samples
    sw_below_rank: int = 0             # 0: the lowest reference sample; r: sorted[n*r/20] (5 = 25th pct)
    sw_below_slow_only: bool = True    # apply the below rule only after a slow arrival
    # mechanical end (rod extending)
    mech_speed_pct: int = 25           # CAL_END_MECH_SPEED_PCT: v200 <= 25% of vRef
    mech_cur_pct: int = 125            # CAL_END_MECH_CURRENT_PCT: median of last 4 currents >= 125% of iRef
    mech_hold: int = 3                 # CAL_END_MECH_HOLD_SAMPLES consecutive samples
    # stuck (armed, no signature)
    stall_ms: int = 1000               # CAL_STALL_WINDOW_MS
    stall_edges: int = 3               # CAL_HALL_STALL_MIN_EDGES
    stuck_still_ms: int = 400          # CAL_END_STUCK_STILL_MS: and no edge for this long
    # unarmed static stall
    static_curl_window_ms: int = 500   # CAL_END_STATIC_WINDOW_MS
    static_curl_min_drive_ms: int = 1000   # CAL_END_STATIC_MIN_DRIVE_MS
    static_uncurl_grace_ms: int = 1500     # CAL_START_GRACE_MS (after the ramp)
    static_uncurl_window_ms: int = 1000    # CAL_STALL_WINDOW_MS
    # review fixes (2026-11-24)
    f1_iref_pct: int = 90              # slow arrival: worst also <= this % of iRef (0 = off)
    short_samples: int = 40            # curl stall within this many samples of arming: plain stall (0 = off)
    short_retract_pct: int = 100       # ... on a retract drive only if med4 <= this % of iRef
    short_uncurl: bool = False         # also for uncurl drives
    short_extend_guard: bool = False   # extend fallback needs med4 >= iRef
    f4_moved_window: bool = True       # unarmed curl that moved >= arm_min_edges: stall_ms window
    short_normal: int = 12             # fallback when fewer normal samples than this since arming
    f1_use_min: bool = False           # F1 guard against the lowest iRef held this drive


def median(values):
    """Element n/2 of the sorted values (firmware: insertion sort; even n takes the upper middle)."""
    s = sorted(values)
    return s[len(s) // 2]


@dataclass
class Event:
    kind: str       # "switch", "mech", "stuck", "static"
    k: int          # sample index of the decision
    edges: int      # drive edges at the decision
    info: dict = field(default_factory=dict)


class EndDetector:
    def __init__(self, move_pwm, retract, curl, p=None):
        self.p = p or Params()
        self.move_pwm = move_pwm
        self.retract = retract          # rod retracting (-PWM): its end is the actuator's limit switch
        self.curl = curl
        self.e = []                     # full history for the replay; firmware keeps p.ring
        self.i = []
        self.k = -1
        self.ramp_done_k = None
        self.armed = False
        self.arm_k = None
        self.run_streak = 0
        self.i_ref = None
        self.v_ref = None
        self.mech_streak = 0
        self.last_edge_k = None
        self.e_max = None
        self.stop_ref = None
        self.stop_fast = False
        # diagnostics: closest approach to each rule while still running (>= 1 s before the end)
        self.samples = []

    def _ring_ok(self, back):
        if back >= self.p.ring:
            raise AssertionError(f"rule looks {back} samples back, ring holds {self.p.ring}")

    def v200(self, k):
        self._ring_ok(4)
        return self.e[k] - self.e[k - 4] if k >= 4 else 0

    def step(self, edges, cur10, pwm_mag):
        p = self.p
        self.k += 1
        k = self.k
        self.e.append(edges)
        self.i.append(cur10)
        # forward progress: a new maximum of the drive edges (a bounce back and forth at a
        # stop is not progress)
        if self.e_max is None or edges > self.e_max:
            if self.e_max is not None:
                self.last_edge_k = k
                self.stop_ref = None
            self.e_max = edges
        if self.ramp_done_k is None:
            if pwm_mag < self.move_pwm:
                return None
            self.ramp_done_k = k
        since_ramp_ms = (k - self.ramp_done_k) * SAMPLE_MS
        drive_ms = k * SAMPLE_MS
        v = self.v200(k)

        if not self.armed:
            if self.curl:
                w = p.static_curl_window_ms // SAMPLE_MS
                if p.f4_moved_window and edges >= p.arm_min_edges:
                    w = p.stall_ms // SAMPLE_MS
                self._ring_ok(w)
                if drive_ms >= p.static_curl_min_drive_ms and k >= w and self.e[k] - self.e[k - w] < p.stall_edges:
                    return Event("static", k, edges)
            else:
                w = p.static_uncurl_window_ms // SAMPLE_MS
                self._ring_ok(w)
                if since_ramp_ms >= p.static_uncurl_grace_ms + p.static_uncurl_window_ms and \
                        self.e[k] - self.e[k - w] < p.stall_edges:
                    return Event("static", k, edges)
            if since_ramp_ms < p.learn_delay_ms:
                return None
            self.run_streak = self.run_streak + 1 if v >= p.min_run_edges_200 else 0
            if self.run_streak >= p.arm_samples and edges >= p.arm_min_edges:
                n = p.arm_samples
                self.armed = True
                self.arm_k = k
                self.i_ref = median(self.i[k - n + 1:k + 1])
                self.v_ref = median([self.v200(j) for j in range(k - n + 1, k + 1)])
                self.i_ref_min = self.i_ref
            return None

        # armed: references follow normal running
        if v >= p.min_run_edges_200 and v * 100 >= p.normal_speed_pct * self.v_ref:
            self.i_ref += max(-p.track_step, min(p.track_step, cur10 - self.i_ref))
            self.normal_n = getattr(self, 'normal_n', 0) + 1
            self.v_ref = max(self.v_ref, v)
            self.i_ref_min = min(self.i_ref_min, self.i_ref)

        cur_med4 = median(self.i[k - 3:k + 1])
        rec = dict(k=k, v=v, v_ref=self.v_ref, i_ref=self.i_ref, cur_med4=cur_med4)

        if self.retract:
            n = p.sw_samples
            last = self.last_edge_k if self.last_edge_k is not None else self.arm_k
            if k - last >= n:
                if k - last == n or self.stop_ref is None:
                    # frozen once per stop, when the count has been still for n samples
                    # reference: the pre_samples samples before the last edge sample (its own
                    # current may already be falling)
                    lo = max(self.arm_k - p.arm_samples + 1, last - p.pre_samples)
                    hi = max(lo + 1, last)
                    self._ring_ok(k - lo)
                    self.stop_ref = median(self.i[lo:hi])
                    srt = sorted(self.i[lo:hi])
                    self.stop_min = srt[len(srt) * p.sw_below_rank // 20]
                    arrive = self.e[last] - self.e[last - 4]
                    self.stop_fast = arrive * 100 >= p.arrive_fast_pct * self.v_ref
                ref, fast = self.stop_ref, self.stop_fast
                pct = p.sw_pct_fast if fast else p.sw_pct_slow
                worst = max(self.i[k - n + 1:k + 1])
                rec.update(sw_ratio=worst / ref, sw_fast=fast)
                self.samples.append(rec)
                below_min = (not p.sw_below_min) or (p.sw_below_slow_only and fast) or worst < self.stop_min
                rec.update(sw_below_min=below_min)
                f1ref = self.i_ref_min if p.f1_use_min else self.i_ref
                f1ok = fast or p.f1_iref_pct == 0 or worst * 100 <= p.f1_iref_pct * f1ref
                if worst * 100 <= pct * ref and below_min and f1ok:
                    return Event("switch", k, edges, dict(ref=ref, worst=worst, ratio=round(worst / ref, 3),
                                                          fast=fast))
            else:
                self.samples.append(rec)
        else:
            slow = v * 100 <= p.mech_speed_pct * self.v_ref
            hot = cur_med4 * 100 >= p.mech_cur_pct * self.i_ref
            self.mech_streak = self.mech_streak + 1 if (slow and hot) else 0
            rec.update(mech_cur=cur_med4 / self.i_ref, mech_speed=v / self.v_ref)
            self.samples.append(rec)
            if self.mech_streak >= p.mech_hold:
                return Event("mech", k, edges, dict(i_ref=self.i_ref, cur_med=cur_med4, v=v, v_ref=self.v_ref,
                                                    ratio=round(cur_med4 / self.i_ref, 3)))
        w = p.stall_ms // SAMPLE_MS
        self._ring_ok(w)
        still = k - (self.last_edge_k if self.last_edge_k is not None else 0)
        if k >= w and self.e[k] - self.e[k - w] < p.stall_edges and still * SAMPLE_MS >= p.stuck_still_ms:
            lf = self.last_edge_k if self.last_edge_k is not None else self.arm_k
            if lf < self.arm_k: lf = self.arm_k
            nn = getattr(self, 'normal_n', 0)
            shortok = (not p.short_samples or lf - self.arm_k < p.short_samples) and (not p.short_normal or nn < p.short_normal) and (p.short_samples or p.short_normal)
            if shortok and (self.curl or p.short_uncurl) and                     (cur_med4 * 100 <= p.short_retract_pct * self.i_ref if self.retract else (not p.short_extend_guard or cur_med4 >= self.i_ref)):
                return Event("static", k, edges, dict(short=True, since_arm=lf - self.arm_k, nn=nn))
            return Event("stuck", k, edges, dict(i_ref=self.i_ref, cur_med=cur_med4, since_arm=lf - self.arm_k, nn=getattr(self, 'normal_n', 0)))
        return None
