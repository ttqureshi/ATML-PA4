"""Q3m: greedy multiple-choice knapsack on calibration MSE per packed byte."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from task1.src.allocation import Q3M_RULE, allocate_q3_bits, allocate_q3_knapsack
from task1.src.constants import FULL_HELLASWAG_CONDITIONS, Q3_BITS
from task1.src.quantize import tensor_packed_bytes


def linear_bytes(sizes: dict[str, int], excluded: int = 0):
    def byte_fn(bits: dict[str, int]) -> int:
        return excluded + sum(bits[name] * sizes[name] for name in sizes)

    return byte_fn


def total(mse, bits) -> float:
    return float(sum(mse[name][bits[name]] for name in bits))


# Seven Qwen-like projection shapes, shrunk. Group size 8 keeps several groups per row.
SHAPES = {"q": (12, 16), "k": (2, 16), "v": (2, 16), "o": (12, 12), "g": (32, 12), "u": (32, 12), "d": (12, 32)}


def qwen_like(seed: int):
    rng = np.random.default_rng(seed)
    layers = int(rng.integers(3, 6))
    names = [f"l{layer}.{kind}" for layer in range(layers) for kind in SHAPES]
    shape = {name: SHAPES[name.split(".")[1]] for name in names}
    mse = {}
    for name in names:
        scale = rng.lognormal(0.0, 1.5)
        # Quantization error falls roughly 4x per extra bit; noise breaks the pattern.
        values = sorted((scale * 4.0 ** (-b) * np.exp(rng.normal(0.0, 0.3)) for b in Q3_BITS), reverse=True)
        mse[name] = dict(zip(Q3_BITS, values))

    def byte_fn(bits: dict[str, int]) -> int:
        return 7 + sum(tensor_packed_bytes(*shape[name], bits[name], 8) for name in names)

    return names, mse, byte_fn


class Q3KnapsackTests(unittest.TestCase):
    def test_hand_example_takes_the_best_ratio_and_may_jump_two_to_eight(self) -> None:
        sizes = {"a": 10, "b": 10}
        mse = {
            "a": {2: 100.0, 3: 99.0, 4: 98.0, 8: 0.0},  # only 2->8 pays: 100/60 per byte
            "b": {2: 10.0, 3: 5.0, 4: 4.0, 8: 3.0},  # 2->3 gives 5/10 per byte
        }
        allocation = allocate_q3_knapsack(["a", "b"], mse, linear_bytes(sizes), budget_bytes=40 + 60)
        self.assertEqual(allocation.bits, {"a": 8, "b": 2})
        self.assertEqual(allocation.steps[0]["from_bits"], 2)
        self.assertEqual(allocation.steps[0]["bits"], 8)
        self.assertEqual(allocation.final_bytes, 100)
        self.assertEqual(allocation.total_mse, 10.0)
        # Ten more bytes buy b 2->3 next.
        more = allocate_q3_knapsack(["a", "b"], mse, linear_bytes(sizes), budget_bytes=110)
        self.assertEqual(more.bits, {"a": 8, "b": 3})
        self.assertEqual([step["projection"] for step in more.steps], ["a", "b"])

    def test_exact_budget_edges(self) -> None:
        sizes = {"a": 3, "b": 5}
        mse = {"a": {2: 9.0, 3: 4.0, 4: 2.0, 8: 1.0}, "b": {2: 8.0, 3: 6.0, 4: 5.0, 8: 4.5}}
        byte_fn = linear_bytes(sizes, excluded=11)
        start = byte_fn({"a": 2, "b": 2})
        at_start = allocate_q3_knapsack(["a", "b"], mse, byte_fn, start)
        self.assertEqual(at_start.bits, {"a": 2, "b": 2})
        self.assertTrue(at_start.within_budget)
        self.assertEqual(at_start.steps, ())
        below = allocate_q3_knapsack(["a", "b"], mse, byte_fn, start - 1)
        self.assertFalse(below.within_budget)
        self.assertEqual(below.bits, {"a": 2, "b": 2})
        # a 2->3 costs exactly 3 bytes; a budget of start+3 must take it (<= budget, not <).
        exact = allocate_q3_knapsack(["a", "b"], mse, byte_fn, start + 3)
        self.assertEqual(exact.bits, {"a": 3, "b": 2})
        self.assertEqual(exact.final_bytes, start + 3)
        top = byte_fn({"a": 8, "b": 8})
        full = allocate_q3_knapsack(["a", "b"], mse, byte_fn, top)
        self.assertEqual(full.bits, {"a": 8, "b": 8})
        self.assertEqual(full.final_bytes, top)

    def test_budget_never_exceeded_and_monotone_in_budget(self) -> None:
        rng = np.random.default_rng(0)
        for _ in range(150):
            n = int(rng.integers(1, 7))
            names = [f"m{i}" for i in range(n)]
            sizes = {name: int(rng.integers(1, 12)) for name in names}
            mse = {
                name: dict(zip(Q3_BITS, sorted(rng.random(4) * rng.lognormal(0, 2), reverse=True)))
                for name in names
            }
            byte_fn = linear_bytes(sizes, excluded=int(rng.integers(0, 20)))
            low = byte_fn({name: 2 for name in names})
            high = byte_fn({name: 8 for name in names})
            previous = None
            for budget in range(low, high + 1):
                allocation = allocate_q3_knapsack(names, mse, byte_fn, budget)
                self.assertTrue(allocation.within_budget)
                self.assertLessEqual(allocation.final_bytes, budget)
                self.assertEqual(allocation.final_bytes, byte_fn(allocation.bits))
                self.assertTrue(set(allocation.bits.values()) <= set(Q3_BITS))
                self.assertAlmostEqual(allocation.total_mse, total(mse, allocation.bits))
                if previous is not None:
                    self.assertLessEqual(allocation.total_mse, previous + 1e-12)
                previous = allocation.total_mse

    def test_beats_v1_and_uniform_on_qwen_like_uniform_budgets(self) -> None:
        for seed in range(40):
            names, mse, byte_fn = qwen_like(seed)
            for uniform in (4, 3):
                budget = byte_fn({name: uniform for name in names})
                knapsack = allocate_q3_knapsack(names, mse, byte_fn, budget)
                ladder = allocate_q3_bits(names, [mse[name][4] for name in names], byte_fn, budget)
                self.assertLessEqual(knapsack.final_bytes, budget)
                self.assertLessEqual(knapsack.total_mse, total(mse, ladder.bits) * (1 + 1e-12))
                self.assertLessEqual(knapsack.total_mse, total(mse, {n: uniform for n in names}) * (1 + 1e-12))

    def test_greedy_is_not_exact_on_a_tiny_instance(self) -> None:
        # Documented limitation: the ladder wins here because the knapsack's last
        # bytes cannot buy a 4->8 step. The Qwen-like instances above do not show it.
        sizes = {"a": 3, "b": 3}
        mse = {"a": {2: 25, 3: 24, 4: 16, 8: 9}, "b": {2: 23, 3: 13, 4: 9, 8: 3}}
        byte_fn = linear_bytes(sizes)
        knapsack = allocate_q3_knapsack(["a", "b"], mse, byte_fn, 33)
        ladder = allocate_q3_bits(["a", "b"], [16, 9], byte_fn, 33)
        self.assertEqual(knapsack.bits, {"a": 4, "b": 4})
        self.assertEqual(ladder.bits, {"a": 8, "b": 3})
        self.assertGreater(knapsack.total_mse, total(mse, ladder.bits))

    def test_ties_are_broken_by_name_and_input_order_does_not_matter(self) -> None:
        table = {2: 4.0, 3: 2.0, 4: 1.0, 8: 0.5}
        names = ["zeta", "alpha", "mid"]
        mse = {name: dict(table) for name in names}
        sizes = {name: 5 for name in names}
        budget = linear_bytes(sizes)({name: 2 for name in names}) + 5
        forward = allocate_q3_knapsack(names, mse, linear_bytes(sizes), budget)
        backward = allocate_q3_knapsack(list(reversed(names)), mse, linear_bytes(sizes), budget)
        self.assertEqual(forward.bits, {"alpha": 3, "mid": 2, "zeta": 2})
        self.assertEqual(forward.bits, backward.bits)
        self.assertEqual(forward.steps, backward.steps)
        # Equal ratio for two widths of one projection: the smaller width wins.
        even = {"p": {2: 4.0, 3: 3.0, 4: 2.0, 8: -2.0}}  # 2->3: 1/1, 2->4: 2/2, 2->8: 6/6
        tied = allocate_q3_knapsack(["p"], even, linear_bytes({"p": 1}), 100)
        self.assertEqual([step["bits"] for step in tied.steps], [3, 4, 8])
        # A width that adds no reduction is never bought.
        flat = {"p": {2: 3.0, 3: 2.0, 4: 1.0, 8: 1.0}}
        stopped = allocate_q3_knapsack(["p"], flat, linear_bytes({"p": 1}), 100)
        self.assertEqual([step["bits"] for step in stopped.steps], [3, 4])

    def test_rejects_bad_inputs(self) -> None:
        mse = {"a": {2: 1.0, 3: 0.5, 4: 0.25, 8: 0.1}}
        with self.assertRaises(ValueError):
            allocate_q3_knapsack([], {}, linear_bytes({}), 10)
        with self.assertRaises(ValueError):
            allocate_q3_knapsack(["a"], {"a": {2: 1.0, 4: 0.5}}, linear_bytes({"a": 1}), 10)
        with self.assertRaises(ValueError):
            allocate_q3_knapsack(["a"], mse, lambda bits: 7, 10)  # bytes do not grow with bits

        def coupled(bits: dict[str, int]) -> int:  # not additive over projections
            return bits["a"] * bits["b"]

        with self.assertRaises(ValueError):
            allocate_q3_knapsack(["a", "b"], {"a": mse["a"], "b": mse["a"]}, coupled, 100)

    def test_rule_text_and_full_hellaswag(self) -> None:
        self.assertIn("2->8 directly allowed", Q3M_RULE)
        self.assertIn("largest reduction / extra bytes", Q3M_RULE)
        self.assertIn("q3m_b4", FULL_HELLASWAG_CONDITIONS)
        self.assertIn("q3m_b3", FULL_HELLASWAG_CONDITIONS)


class Q3mStageChecksTests(unittest.TestCase):
    """Budget and alpha checks of the GPU stage, on fake modules (no model)."""

    def setUp(self) -> None:
        self.modules = [
            ("model.layers.0.self_attn.q_proj", SimpleNamespace(weight=SimpleNamespace(shape=(8, 256)))),
            ("model.layers.0.self_attn.o_proj", SimpleNamespace(weight=SimpleNamespace(shape=(8, 300)))),
        ]
        self.census = {"n_excluded": 1000}

    def _q2(self, bits: int, modes: dict[str, str], alpha: float = 0.5) -> dict:
        from task1.src.pipeline import _packed_bytes_for

        names = [name for name, _module in self.modules]
        packed = _packed_bytes_for(self.modules, self.census, {name: bits for name in names}, modes)
        return {
            "packed_bytes": packed,
            "projections": [
                {"module": name, "alpha": alpha, "compensation": modes[name], "mse": 1.0} for name in names
            ],
        }

    def test_budgets_reproduce_q2_and_reject_mismatch(self) -> None:
        from task1.src.pipeline import _q3m_budgets

        names = [name for name, _module in self.modules]
        modes4 = {names[0]: "fold", names[1]: "store"}
        modes3 = {names[0]: "store", names[1]: "store"}  # INT3 alphas can change the plan
        q2 = {4: self._q2(4, modes4), 3: self._q2(3, modes3)}
        budgets = _q3m_budgets(self.modules, self.census, modes4, q2)
        self.assertEqual(budgets, {4: q2[4]["packed_bytes"], 3: q2[3]["packed_bytes"]})
        self.assertGreater(budgets[4], budgets[3])
        q2[3]["packed_bytes"] += 1
        with self.assertRaises(RuntimeError):
            _q3m_budgets(self.modules, self.census, modes4, q2)
        # The INT4 plan the stage recomputes must equal the one Q2 INT4 recorded.
        with self.assertRaises(RuntimeError):
            _q3m_budgets(self.modules, self.census, modes3, {4: self._q2(4, modes4), 3: self._q2(3, modes3)})

    def test_alphas_must_match_q2_int4(self) -> None:
        from task1.src.pipeline import _q3m_check_alphas

        names = [name for name, _module in self.modules]
        q2 = self._q2(4, {name: "store" for name in names}, alpha=0.5)
        _q3m_check_alphas(self.modules, {name: 0.5 for name in names}, q2)
        with self.assertRaises(RuntimeError):
            _q3m_check_alphas(self.modules, {names[0]: 0.5, names[1]: 1.0}, q2)


if __name__ == "__main__":
    unittest.main()
