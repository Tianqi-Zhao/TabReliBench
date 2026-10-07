import numpy as np
from scipy.linalg import norm as matrix_norm
import math
from collections import defaultdict

def compute_neff(K):
    """Effective sample size from an unnormalized kernel matrix (Equation 21).

    n_eff(h) = n * E[E[H(X,X')|X]^2] / E[H(X,X')^2]

    With n sample points this reduces to  sum_i R_i^2 / sum_ij K_ij^2
    where R_i = sum_j K_ij.
    """
    row_sums = K.sum(axis=1, keepdims=True)
    ntrain = K.shape[0]
    H = K / row_sums
    eff_val = ntrain / (matrix_norm(H, 'fro')**2) - 1
    return eff_val


def auto_bandwidth(kernel_fn, max_h, target_neff=50, rtol=1e-4, max_iter=100):
    """Find bandwidth *h* such that n_eff(kernel_fn(h)) ≈ *target_neff*.

    Parameters
    ----------
    kernel_fn : callable
        ``kernel_fn(h) -> K`` returns the **unnormalized** kernel weight
        matrix K of shape (n, n) for a given bandwidth *h*.  The caller is
        responsible for any numerical stabilization (e.g. global log-shift
        before exponentiating a Gaussian kernel).  Because n_eff is invariant
        to positive global scaling of K, a globally shifted K gives the same
        result.
    max_h : float
        Upper bound for the binary search.
    target_neff : float
        Desired effective sample size.
    rtol : float
        Relative tolerance on *h* for stopping.
    max_iter : int
        Maximum number of bisection steps.

    Returns
    -------
    h : float
    """
    def _neff(h):
        return compute_neff(kernel_fn(h))

    h_lo, h_hi = max_h * 1e-6, max_h
    if _neff(h_hi) <= target_neff:
        return h_hi
    if _neff(h_lo) >= target_neff:
        return h_lo
    # Invariant: _neff(h_lo) < target_neff <= _neff(h_hi).
    # We return h_hi so the result always satisfies n_eff >= target_neff.
    for _ in range(max_iter):
        h = (h_lo + h_hi) / 2
        if _neff(h) < target_neff:
            h_lo = h
        else:
            h_hi = h
        if (h_hi - h_lo) / max(h_hi, 1e-12) < rtol:
            break
    return h_hi


def solve_largest_x(T_cal_base, T_cal_jump, cal_scores, T_test_jump, alpha=0.05):
    """
    Finds the largest score threshold x per query such that T_test(x) <= Q_{1-alpha}
    of the multiset {T_cal_1(x), ..., T_cal_n(x)}.

    In the WCP paper (Section 3.3), for a fixed test point and threshold g:
        T_cal_j(g) = T_cal_base_j              if s_j < g   (calibration score j)
                   = T_cal_base_j + T_cal_jump_j  if s_j > g
        T_test(g)  = sum_{j : s_j < g} T_test_jump_j

    Goal: for every query i, find the LARGEST g such that
        T_test_i(g) <= Q_{1-alpha}(T_cal_1^(i)(g), ..., T_cal_n^(i)(g), T_test_i(g))

    The quantile is over n+1 values (the n calibration scores plus T_test itself),
    consistent with the conformal prediction guarantee.

    Arguments:
    ----------
    T_cal_base : 1D array of length n  OR  2D array of shape (m, n)
        Baseline value of each calibration T-score (value when s_j < g).
        If 1D, the same baseline is shared across all queries (used together
        with 1D T_cal_jump for the fast shared path).
        If 2D, each query has its own per-component baseline; must be paired
        with 2D T_cal_jump.
    T_cal_jump : 1D array of length n  OR  2D array of shape (m, n)
        Jump in each calibration T-score when g crosses s_j from above.
        If 1D, shared across all queries (one calibration order statistic).
        If 2D, per-query jumps; m heaps maintained in parallel.
    cal_scores : 1D array of length n
        Calibration nonconformity scores s_j (the breakpoints).
    T_test_jump : 2D array of shape (m, n)
        Per-query increment added to T_test when g crosses each s_j.
    alpha : float, default 0.05

    Returns:
    --------
    results : 1D numpy array of length m
        Largest valid g per query. np.inf if condition holds for all g;
        np.nan if it never holds.
    """
    T_cal_base = np.asarray(T_cal_base, dtype=np.float64)
    T_cal_jump = np.asarray(T_cal_jump, dtype=np.float64)
    T_test_jump_mat = np.asarray(T_test_jump, dtype=np.float64)
    cal_scores = np.asarray(cal_scores, dtype=np.float64)
    m, n = T_test_jump_mat.shape

    # The quantile is over n+1 values (n calibration T-scores + the test T-score
    # itself), matching the WCP paper condition:
    #   T_test <= Q_{1-alpha}(T_cal_1, ..., T_cal_n, T_test)
    # The (1-alpha)-quantile of n+1 values has rank ceil((1-alpha)(n+1)).
    rank_idx = min(n - 1, max(0, math.ceil((1.0 - alpha) * (n + 1)) - 1))

    events = defaultdict(list)
    for j in range(n):
        events[cal_scores[j]].append(j)
    sorted_scores = sorted(events.keys(), reverse=True)

    if T_cal_jump.ndim == 1:
        return _solve_shared_jump(T_cal_base, T_cal_jump, T_test_jump_mat,
                                  m, n, rank_idx, events, sorted_scores)
    else:
        return _solve_per_query_jump(T_cal_base, T_cal_jump, T_test_jump_mat,
                                     m, n, rank_idx, events, sorted_scores)


def _test_score_prefixes(weights, events):
    """Sum nonnegative weights forward, rather than subtracting from a total.

    At a score breakpoint s the test score is the weight of scores < s.
    Repeated subtraction loses tiny remaining weights and can leave a
    positive residual even when this set is empty. Prefix sums retain the
    small weights and give the empty set exactly zero, without a tolerance
    that could accept a genuinely positive score.
    """
    order = []
    before = {}
    for score in sorted(events):
        before[score] = len(order) - 1
        order.extend(events[score])
    prefix = np.cumsum(weights[:, order], axis=1)
    return prefix, before


def _solve_shared_jump(T_cal_base, T_cal_jump, T_test_jump_mat,
                       m, n, rank_idx, events, sorted_scores):
    """Shared calibration scores: select their order statistic directly.

    There is only one calibration vector in this path. Direct partitioning
    also avoids ambiguous value-based lazy deletion when its scores tie.
    """
    current_T_cal = T_cal_base.copy()
    test_prefix, before_score = _test_score_prefixes(T_test_jump_mat, events)
    results = np.full(m, np.nan)
    unsolved = np.ones(m, dtype=bool)
    p = np.partition(current_T_cal, rank_idx)[rank_idx]
    met = T_test_jump_mat.sum(axis=1) <= p
    results[met] = np.inf
    unsolved[met] = False
    for s in sorted_scores:
        if not unsolved.any():
            break
        idxs = events[s]
        current_T_cal[idxs] = T_cal_base[idxs] + T_cal_jump[idxs]
        k = before_score[s]
        current_T_test = test_prefix[:, k] if k >= 0 else np.zeros(m)
        p = np.partition(current_T_cal, rank_idx)[rank_idx]
        met = (current_T_test <= p) & unsolved
        results[met] = s
        unsolved[met] = False
    return results


def _solve_per_query_jump(T_cal_base, T_cal_jump, T_test_jump_mat,
                          m, n, rank_idx, events, sorted_scores):
    """
    Per-query T_cal_jump path: m pairs of heaps maintained in parallel.

    ``T_cal_base`` is a 2-D array of shape (m, n): ``T_cal_base[i, j]`` is the
    baseline T_cal_j value for query i (the value when s_j < g).

    Data layout
    -----------
    small_h[i, :] : max-heap of the bottom (rank_idx+1) T_cal values for query i.
                    Position 0 holds the maximum = percentile P[i].
    large_h[i, :] : min-heap of the remaining (n - rank_idx - 1) T_cal values.
                    Position 0 holds the minimum (next value above P[i]).
    small_comp[i, k] : which component occupies position k in small_h[i].
    large_comp[i, k] : which component occupies position k in large_h[i].
    comp_pos[i, j]   : position of component j in its current heap for query i.
    in_large[i, j]   : True if component j is currently in large_h[i].

    Sift operations sweep level-by-level using numpy gather/scatter, so each
    level costs O(m) and there are O(log n) levels per activation event.
    """
    small_size = rank_idx + 1
    large_size = n - small_size
    rows = np.arange(m)

    # ---- Build initial heaps (T_cal_base values, i.e. T_cal_j when s_j < g) ----
    # argsort rows ascending; sorted-descending prefix = valid max-heap,
    # sorted-ascending suffix = valid min-heap.
    sort_orders        = np.argsort(T_cal_base, axis=1)                          # (m, n) asc
    sorted_T_cal_base  = np.take_along_axis(T_cal_base, sort_orders, axis=1)     # (m, n)

    small_h    = sorted_T_cal_base[:, :small_size][:, ::-1].astype(np.float64).copy()
    small_comp = sort_orders[:, :small_size][:, ::-1].astype(np.int32).copy()

    if large_size > 0:
        large_h    = sorted_T_cal_base[:, small_size:].astype(np.float64).copy()
        large_comp = sort_orders[:, small_size:].astype(np.int32).copy()
    else:
        large_h    = np.empty((m, 0), dtype=np.float64)
        large_comp = np.empty((m, 0), dtype=np.int32)

    comp_pos = np.empty((m, n), dtype=np.int32)
    in_large = np.zeros((m, n), dtype=bool)
    for k in range(small_size):
        comp_pos[rows, small_comp[:, k]] = k
    if large_size > 0:
        for k in range(large_size):
            comp_pos[rows, large_comp[:, k]] = k
            in_large[rows, large_comp[:, k]] = True

    # ---- Vectorized sift helpers ----
    # Each function operates on a subset `act` of row indices.
    # pos_arr[i] and comp_arr[i] track the current position and component
    # for the element being sifted in row act[i].  comp_arr must be updated
    # alongside pos_arr so swapped components are tracked correctly.

    log_small = int(np.ceil(np.log2(small_size + 1))) + 1 if small_size > 1 else 1
    log_large = int(np.ceil(np.log2(large_size + 1))) + 1 if large_size > 1 else 1

    def sift_up_small(act, pos_arr, comp_arr):
        pos_arr = pos_arr.copy()
        comp_arr = comp_arr.copy()
        for _ in range(log_small):
            parent = (pos_arr - 1) // 2
            at_root = pos_arr == 0
            cur  = small_h[act, pos_arr]
            par  = small_h[act, parent]
            mask = ~at_root & (cur > par)
            if not mask.any():
                break
            r = act[mask]
            par_comp = small_comp[r, parent[mask]].copy()
            small_h[r, pos_arr[mask]]  = par[mask]
            small_h[r, parent[mask]]   = cur[mask]
            small_comp[r, pos_arr[mask]]  = par_comp
            small_comp[r, parent[mask]]   = comp_arr[mask]
            comp_pos[r, par_comp]       = pos_arr[mask]
            comp_pos[r, comp_arr[mask]] = parent[mask]
            pos_arr[mask] = parent[mask]

    def sift_down_small(act, pos_arr, comp_arr):
        pos_arr  = pos_arr.copy()
        comp_arr = comp_arr.copy()
        for _ in range(log_small):
            left, right = 2*pos_arr + 1, 2*pos_arr + 2
            lv = left  < small_size
            rv = right < small_size
            lc = np.minimum(left,  small_size - 1)
            rc = np.minimum(right, small_size - 1)
            lval = np.where(lv, small_h[act, lc], -np.inf)
            rval = np.where(rv, small_h[act, rc], -np.inf)
            cur  = small_h[act, pos_arr]
            use_l   = lval >= rval
            best    = np.where(use_l, left, right)
            bestval = np.where(use_l, lval, rval)
            mask = (lv | rv) & (bestval > cur)
            if not mask.any():
                break
            r = act[mask]
            ch_comp = small_comp[r, best[mask]].copy()
            small_h[r, pos_arr[mask]] = bestval[mask]
            small_h[r, best[mask]]    = cur[mask]
            small_comp[r, pos_arr[mask]] = ch_comp
            small_comp[r, best[mask]]    = comp_arr[mask]
            comp_pos[r, ch_comp]        = pos_arr[mask]
            comp_pos[r, comp_arr[mask]] = best[mask]
            pos_arr[mask] = best[mask]

    def sift_down_large(act, pos_arr, comp_arr):
        if large_size <= 1:
            return
        pos_arr  = pos_arr.copy()
        comp_arr = comp_arr.copy()
        for _ in range(log_large):
            left, right = 2*pos_arr + 1, 2*pos_arr + 2
            lv = left  < large_size
            rv = right < large_size
            lc = np.minimum(left,  large_size - 1)
            rc = np.minimum(right, large_size - 1)
            lval = np.where(lv, large_h[act, lc], np.inf)
            rval = np.where(rv, large_h[act, rc], np.inf)
            cur  = large_h[act, pos_arr]
            use_l   = lval <= rval
            best    = np.where(use_l, left, right)
            bestval = np.where(use_l, lval, rval)
            mask = (lv | rv) & (bestval < cur)
            if not mask.any():
                break
            r = act[mask]
            ch_comp = large_comp[r, best[mask]].copy()
            large_h[r, pos_arr[mask]] = bestval[mask]
            large_h[r, best[mask]]    = cur[mask]
            large_comp[r, pos_arr[mask]] = ch_comp
            large_comp[r, best[mask]]    = comp_arr[mask]
            comp_pos[r, ch_comp]        = pos_arr[mask]
            comp_pos[r, comp_arr[mask]] = best[mask]
            pos_arr[mask] = best[mask]

    # ---- Rebalance: swap roots when small max > large min ----
    def rebalance(act):
        """For rows in act where small_h[i,0] > large_h[i,0], swap roots."""
        if large_size == 0 or len(act) == 0:
            return
        sm_val  = small_h[act, 0].copy()
        lg_val  = large_h[act, 0].copy()
        mask    = sm_val > lg_val
        if not mask.any():
            return
        r = act[mask]
        sm_comp_r = small_comp[r, 0].copy()
        lg_comp_r = large_comp[r, 0].copy()
        # Swap values and component labels at roots
        small_h[r, 0]    = lg_val[mask]
        large_h[r, 0]    = sm_val[mask]
        small_comp[r, 0] = lg_comp_r
        large_comp[r, 0] = sm_comp_r
        comp_pos[r, sm_comp_r] = 0
        comp_pos[r, lg_comp_r] = 0
        in_large[r, sm_comp_r] = True
        in_large[r, lg_comp_r] = False
        # Restore heap properties at roots (values decreased in small, increased in large)
        sift_down_small(r,
                        np.zeros(len(r), dtype=np.int32),
                        lg_comp_r.copy())
        sift_down_large(r,
                        np.zeros(len(r), dtype=np.int32),
                        sm_comp_r.copy())

    # ---- T_test initialisation and g=+inf check ----
    P              = small_h[:, 0].copy()    # (1-alpha)-quantile of T_cal per query
    current_T_test = T_test_jump_mat.sum(axis=1)
    test_prefix, before_score = _test_score_prefixes(T_test_jump_mat, events)
    results        = np.full(m, np.nan)
    unsolved       = np.ones(m, dtype=bool)

    met = (current_T_test <= P) & unsolved
    results[met] = np.inf
    unsolved[met] = False

    # ---- Sweep right-to-left over sorted calibration scores ----
    for s in sorted_scores:
        if not unsolved.any():
            break

        for j in events[s]:
            # T_cal_j jumps from T_cal_base_j to T_cal_base_j + T_cal_jump_j
            new_vals   = T_cal_base[:, j] + T_cal_jump[:, j]  # (m,)
            j_in_large = in_large[:, j]                        # (m,) bool
            j_pos      = comp_pos[:, j]                        # (m,) int

            # --- Components currently in small_h: value increases → sift up ---
            act_small = np.where(~j_in_large)[0]
            if len(act_small):
                small_h[act_small, j_pos[act_small]] = new_vals[act_small]
                sift_up_small(act_small,
                              j_pos[act_small].copy(),
                              np.full(len(act_small), j, dtype=np.int32))
                rebalance(act_small)

            # --- Components currently in large_h: value increases → sift down ---
            act_large = np.where(j_in_large)[0]
            if len(act_large):
                large_h[act_large, j_pos[act_large]] = new_vals[act_large]
                sift_down_large(act_large,
                                j_pos[act_large].copy(),
                                np.full(len(act_large), j, dtype=np.int32))
                # No rebalance needed: large_h values only increase

        # Evaluate the remaining mass directly; do not subtract large weights
        # from the total and lose tiny local weights through cancellation.
        k = before_score[s]
        current_T_test = test_prefix[:, k] if k >= 0 else np.zeros(m)
        P = small_h[:, 0]
        met = (current_T_test <= P) & unsolved
        results[met] = s
        unsolved[met] = False

    return results
