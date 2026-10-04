import unittest

import numpy as np

from bottleneck_distillation.environment import Certificate, Status, ToyTask
from bottleneck_distillation.method import readiness_target
from bottleneck_distillation.policy import LinearPolicy
from bottleneck_distillation.train import train


class ReadinessMethodTests(unittest.TestCase):
    def test_certificate_counterfactual_changes_only_target_status(self):
        from bottleneck_distillation.method import complete_certificate

        table = (
            Certificate("inspect", "record-1", "inspect", Status.UNSAT),
            Certificate("verify", "record-1", "verify", Status.INACTIVE),
        )
        changed = complete_certificate(table, 0)
        self.assertEqual(changed[0].status, Status.SAT)
        self.assertEqual(changed[0].binding, table[0].binding)
        self.assertEqual(changed[1], table[1])
        self.assertEqual(table[0].status, Status.UNSAT)

    def test_open_prerequisite_favors_preparation(self):
        policy = LinearPolicy(seed=0, teacher_strength=2.0)
        state = ToyTask().reset()
        task = ToyTask()
        actions = task.actions(state)
        result = readiness_target(
            policy, state, actions, task.certificates(state), top_k=4, tau=0.7, beta=0.05
        )
        self.assertGreater(result.target[actions.index("inspect")], result.student[actions.index("inspect")])
        self.assertGreater(result.js_by_condition["inspect"], 0.0)
        self.assertAlmostEqual(float(result.target.sum()), 1.0)

    def test_no_open_certificate_has_zero_readiness_loss(self):
        task = ToyTask()
        state = task.reset()
        state = task.step(state, "inspect").state
        state = task.step(state, "verify").state
        policy = LinearPolicy(seed=0)
        result = readiness_target(policy, state, task.actions(state), task.certificates(state))
        np.testing.assert_allclose(result.target, result.student)
        self.assertEqual(result.kl, 0.0)
        self.assertEqual(result.js_by_condition, {})

    def test_training_runs_and_updates_student(self):
        policy = LinearPolicy(seed=3)
        original = policy.weights.copy()
        history = train(policy, ToyTask(), updates=4, groups_per_update=8, group_size=8, seed=3)
        self.assertEqual(len(history), 4)
        self.assertTrue(np.isfinite(policy.weights).all())
        self.assertFalse(np.allclose(original, policy.weights))
        self.assertTrue(all(0.0 <= item["success_rate"] <= 1.0 for item in history))

    def test_custom_observation_encoder_is_accepted(self):
        policy = LinearPolicy(
            actions=("prepare", "commit"),
            feature_dim=2,
            feature_fn=lambda state: np.array([1.0, float(state)]),
        )
        probabilities = policy.probabilities(1, ("prepare", "commit"))
        self.assertEqual(probabilities.shape, (2,))
        self.assertAlmostEqual(float(probabilities.sum()), 1.0)


if __name__ == "__main__":
    unittest.main()
