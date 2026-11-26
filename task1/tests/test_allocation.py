"""P3 integer repair and the Q3 bit ladder."""

from __future__ import annotations

import unittest

import numpy as np

from task1.src.allocation import ProjectionSize, allocate_p3_zeros, allocate_q3_bits
from task1.src.pruning import fraction_in_clip, uniform_zero_budget
from task1.src.quantize import compensation_plan


class AllocationTests(unittest.TestCase):
    def test_p3_repair_hits_the_uniform_budget_inside_the_clip(self) -> None:
        sizes = [ProjectionSize("a", 10, 10), ProjectionSize("b", 10, 10), ProjectionSize("c", 10, 10)]
        budget = uniform_zero_budget([(10, 10), (10, 10), (10, 10)], 0.5)
        self.assertEqual(budget, 150)
        zeros = allocate_p3_zeros(sizes, [1.0, 100.0, 0.01], sparsity=0.5, tau=2.0)
        # Clipping puts b at 0.85 and a, c at 0.10 (105 zeros). The 45-zero
        # deficit alternates between a and c, the two projections farthest
        # from the upper clip, with index ties going to a first.
        self.assertEqual(zeros, [33, 85, 32])
        self.assertEqual(sum(zeros), budget)
        for count in zeros:
            self.assertTrue(fraction_in_clip(count, 100))

    def test_q3_stops_on_the_most_sensitive_projection(self) -> None:
        names = ["a", "b", "c"]
        weights = {name: 10 for name in names}

        def byte_fn(bits: dict[str, int]) -> int:
            return sum(bits[name] * weights[name] for name in names)

        # Uniform INT4 costs 120. Promoting b (the sensitive one) 2→3→4→8
        # lands on 120, and the next-most-sensitive step would pass 120.
        allocation = allocate_q3_bits(names, [1.0, 5.0, 2.0], byte_fn, budget_bytes=120)
        self.assertEqual(allocation.bits, {"a": 2, "b": 8, "c": 2})
        self.assertTrue(allocation.within_budget)
        self.assertLessEqual(allocation.final_bytes, 120)
        self.assertTrue(set(allocation.bits.values()) <= {2, 3, 4, 8})

        blocked = allocate_q3_bits(names, [1.0, 5.0, 2.0], byte_fn, budget_bytes=60)
        self.assertEqual(blocked.bits, {"a": 2, "b": 2, "c": 2})
        self.assertTrue(blocked.within_budget)

    def test_compensation_folds_only_matching_shared_inputs(self) -> None:
        scale = np.array([1.0, 2.0, 3.0])
        plan = compensation_plan(
            {
                "q_proj": scale,
                "k_proj": scale,
                "v_proj": scale,
                "o_proj": scale,
                "gate_proj": scale,
                "up_proj": scale * 2.0,
                "down_proj": scale,
            }
        )
        self.assertEqual(plan["q_proj"], "fold")
        self.assertEqual(plan["k_proj"], "fold")
        self.assertEqual(plan["v_proj"], "fold")
        self.assertEqual(plan["gate_proj"], "store")
        self.assertEqual(plan["up_proj"], "store")
        self.assertEqual(plan["o_proj"], "store")
        self.assertEqual(plan["down_proj"], "store")


if __name__ == "__main__":
    unittest.main()
