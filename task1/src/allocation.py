"""P3 sparsity repair and the Q3 bit ladder.

P3 sets ``s_l`` proportional to ``1 / (sensitivity_l + eps) ** tau``. A
projection whose isolated prune distorts the output more keeps more of its
weights. Fractions are clipped to [0.10, 0.85], then one weight is added or
removed at a time until the total number of zeros equals the uniform per-row
budget (``_bulk_repair`` applies the long prefix of those unit moves at once). The weight moves on the projection whose current fraction is farthest
from the clip boundary in the direction of the move. Equal distance keeps the
earlier projection.

Q3 starts every projection at 2 bits. Each step promotes the most sensitive
projection that is not already at 8 bits, along 2 → 3 → 4 → 8. If that
promotion would exceed the packed-byte budget, that projection is left where
it is and the next most sensitive projection is tried. The search stops when
no remaining promotion fits.
"""

from __future__ import annotations

from dataclasses import dataclass

from task1.src.constants import Q3_BITS, SPARSITY_CLIP
from task1.src.pruning import fraction_in_clip, sparsity_bounds, uniform_zero_budget


class InfeasibleAllocation(ValueError):
    """The clip cannot represent the requested budget."""


@dataclass(frozen=True)
class ProjectionSize:
    name: str
    rows: int
    cols: int

    @property
    def numel(self) -> int:
        return self.rows * self.cols


def _proportional_fractions(
    sizes: list[ProjectionSize],
    sensitivities: list[float],
    target_zeros: int,
    tau: float,
) -> list[float]:
    # High output distortion keeps a lower sparsity. eps keeps a zero-error
    # projection finite and makes it the first place zeros are spent.
    raw = [(float(value) + 1e-12) ** (-float(tau)) for value in sensitivities]
    weighted = sum(raw[i] * sizes[i].numel for i in range(len(sizes)))
    if weighted == 0.0:
        # No measured distortion anywhere: start from the global rate. The
        # integer repair still lands on the exact uniform zero count.
        if target_zeros == 0:
            return [0.0 for _ in sizes]
        total = sum(size.numel for size in sizes)
        return [target_zeros / total for _ in sizes]
    return [raw[i] * target_zeros / weighted for i in range(len(sizes))]


def _move_value(z0: int, k: int, numel: int, direction: int, low: float, high: float) -> float:
    """Slack the unit-step repair sees before the k-th move (k >= 1) on one projection."""
    fraction = (z0 + direction * (k - 1)) / numel
    return (high - fraction) if direction > 0 else (fraction - low)


def _moves_above(z0: int, cap: int, numel: int, direction: int, low: float, high: float, threshold: float) -> int:
    """How many of the first ``cap`` moves have slack strictly above ``threshold``.

    Slack falls strictly with every move, so this is a prefix length.
    """
    lo, hi = 0, cap
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if _move_value(z0, mid, numel, direction, low, high) > threshold:
            lo = mid
        else:
            hi = mid - 1
    return lo


def _bulk_repair(
    sizes: list[ProjectionSize],
    zeros: list[int],
    bounds: list[tuple[int, int]],
    target: int,
    low: float,
    high: float,
) -> list[int]:
    """Apply, in one jump, a prefix of the moves the unit-step repair would make.

    The unit-step loop always moves the projection with the largest slack, and
    one projection's slack falls strictly with each move. So the loop takes
    moves in descending (slack, -index) order, and the set of moves whose
    slack lies above any threshold is a prefix of that order. We bisect for
    the lowest threshold whose prefix still fits in the gap and take it. The
    caller's unit-step loop finishes the few remaining moves, so the result
    is identical to running the unit-step loop alone, without its
    hundreds of millions of Python iterations.
    """
    gap = target - sum(zeros)
    if gap == 0:
        return list(zeros)
    direction = 1 if gap > 0 else -1
    need = abs(gap)
    caps = [
        (z_max - z) if direction > 0 else (z - z_min)
        for z, (z_min, z_max) in zip(zeros, bounds)
    ]

    def taken(threshold: float) -> list[int]:
        return [
            _moves_above(z, cap, size.numel, direction, low, high, threshold)
            for z, cap, size in zip(zeros, caps, sizes)
        ]

    values = [
        _move_value(z, 1, size.numel, direction, low, high)
        for z, cap, size in zip(zeros, caps, sizes)
        if cap > 0
    ]
    if not values:
        return list(zeros)
    hi = max(values)  # nothing lies strictly above the largest first move
    lo = min(values) - 2.0  # below every move: everything is taken
    if sum(taken(lo)) <= need:
        hi = lo
    else:
        for _ in range(200):
            mid = (lo + hi) / 2.0
            if mid <= lo or mid >= hi:
                break
            if sum(taken(mid)) > need:
                lo = mid
            else:
                hi = mid
    moves = taken(hi)
    return [z + direction * m for z, m in zip(zeros, moves)]


def allocate_p3_zeros(
    sizes: list[ProjectionSize],
    sensitivities: list[float],
    sparsity: float,
    tau: float,
) -> list[int]:
    """Integer zeros per projection for one tau and one global sparsity."""
    if len(sizes) != len(sensitivities):
        raise ValueError("sensitivities must align with projections")
    if len(sizes) == 0:
        raise ValueError("allocation needs at least one projection")
    if any(value < 0 for value in sensitivities):
        raise ValueError("sensitivity must be non-negative")
    low, high = SPARSITY_CLIP
    shapes = [(size.rows, size.cols) for size in sizes]
    target = uniform_zero_budget(shapes, sparsity)
    fractions = _proportional_fractions(sizes, sensitivities, target, tau)
    clipped = [min(high, max(low, fraction)) for fraction in fractions]
    zeros: list[int] = []
    bounds: list[tuple[int, int]] = []
    for size, fraction in zip(sizes, clipped):
        z_min, z_max = sparsity_bounds(size.numel, low, high)
        bounds.append((z_min, z_max))
        # Half away from zero. Fractions are non-negative, so this is floor(x+1/2).
        rounded = int(fraction * size.numel + 0.5)
        zeros.append(min(z_max, max(z_min, rounded)))
    feasible_min = sum(item[0] for item in bounds)
    feasible_max = sum(item[1] for item in bounds)
    if not feasible_min <= target <= feasible_max:
        raise InfeasibleAllocation(
            f"uniform budget {target} is outside the clip range [{feasible_min}, {feasible_max}]"
        )
    zeros = _bulk_repair(sizes, zeros, bounds, target, low, high)
    guard = 0
    limit = sum(size.numel for size in sizes) + 1
    while sum(zeros) != target:
        guard += 1
        if guard > limit:
            raise InfeasibleAllocation("integer repair did not reach the uniform budget")
        direction = 1 if sum(zeros) < target else -1
        candidates = []
        for index, (z_min, z_max) in enumerate(bounds):
            if direction > 0 and zeros[index] < z_max:
                candidates.append(index)
            elif direction < 0 and zeros[index] > z_min:
                candidates.append(index)
        if not candidates:
            raise InfeasibleAllocation("no projection can move without leaving the clip")

        def distance(index: int) -> tuple[float, int]:
            fraction = zeros[index] / sizes[index].numel
            slack = (high - fraction) if direction > 0 else (fraction - low)
            return (slack, -index)

        chosen = max(candidates, key=distance)
        zeros[chosen] += direction
    for size, count in zip(sizes, zeros):
        if not fraction_in_clip(count, size.numel, low, high):
            raise InfeasibleAllocation(f"{size.name} left the sparsity clip")
    return zeros


def next_q3_bit(bits: int) -> int | None:
    ladder = Q3_BITS
    if bits not in ladder:
        raise ValueError(f"bit width {bits} is not on the Q3 ladder {ladder}")
    index = ladder.index(bits)
    if index + 1 == len(ladder):
        return None
    return ladder[index + 1]


@dataclass(frozen=True)
class Q3Allocation:
    bits: dict[str, int]
    within_budget: bool
    steps: tuple[dict[str, int | str], ...]
    final_bytes: int
    budget_bytes: int


def allocate_q3_bits(
    names: list[str],
    sensitivities: list[float],
    byte_fn,
    budget_bytes: int,
) -> Q3Allocation:
    """Greedy mixed precision against one packed-byte budget.

    ``byte_fn`` accepts a ``name -> bits`` map and returns the total packed
    bytes of that assignment, including metadata and uncompressed tensors.
    """
    if len(names) != len(sensitivities):
        raise ValueError("sensitivities must align with projections")
    if len(names) == 0:
        raise ValueError("allocation needs at least one projection")
    bits = {name: Q3_BITS[0] for name in names}
    start_bytes = int(byte_fn(bits))
    steps: list[dict[str, int | str]] = []
    if start_bytes > budget_bytes:
        return Q3Allocation(
            bits=bits,
            within_budget=False,
            steps=tuple(steps),
            final_bytes=start_bytes,
            budget_bytes=int(budget_bytes),
        )
    ranked = [name for _, name in sorted(zip(sensitivities, names), key=lambda item: (-item[0], item[1]))]
    blocked: set[str] = set()
    guard = 0
    limit = len(names) * len(Q3_BITS) + 1
    while True:
        guard += 1
        if guard > limit:
            raise InfeasibleAllocation("Q3 upgrade did not terminate")
        candidate = None
        for name in ranked:
            if name in blocked:
                continue
            if next_q3_bit(bits[name]) is not None:
                candidate = name
                break
        if candidate is None:
            break
        promoted = next_q3_bit(bits[candidate])
        assert promoted is not None
        trial = dict(bits)
        trial[candidate] = promoted
        trial_bytes = int(byte_fn(trial))
        if trial_bytes > budget_bytes:
            blocked.add(candidate)
            continue
        bits = trial
        steps.append({"projection": candidate, "bits": promoted, "packed_bytes": trial_bytes})
    final_bytes = int(byte_fn(bits))
    return Q3Allocation(
        bits=bits,
        within_budget=final_bytes <= budget_bytes,
        steps=tuple(steps),
        final_bytes=final_bytes,
        budget_bytes=int(budget_bytes),
    )
