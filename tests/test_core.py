"""Small mathematical tests; no downloads or training required."""
import unittest

import numpy as np
import pandas as pd
import torch

from src.metrics import optimizer_state_bytes, time_to_target_accuracy
from src.optimizers import Lion


class CoreTests(unittest.TestCase):
    """Check update ordering, persistent storage, and missing targets."""

    def test_lion_matches_equations(self) -> None:
        parameter = torch.nn.Parameter(torch.tensor([1., -2., 0.5], dtype=torch.float64))
        optimizer = Lion([parameter], lr=0.1, weight_decay=0.2)
        expected = parameter.detach().clone()
        momentum = torch.zeros_like(expected)
        for gradient in (torch.tensor([2., -1., 0.], dtype=torch.float64),
                         torch.tensor([-0.1, 0.2, 3.], dtype=torch.float64)):
            direction = (0.9 * momentum + 0.1 * gradient).sign()
            expected = expected - 0.1 * (direction + 0.2 * expected)
            momentum = 0.99 * momentum + 0.01 * gradient
            parameter.grad = gradient
            optimizer.step()
            torch.testing.assert_close(parameter, expected)
            torch.testing.assert_close(optimizer.state[parameter]["exp_avg"], momentum)
        self.assertEqual(set(optimizer.state[parameter]), {"exp_avg"})
        self.assertEqual(optimizer_state_bytes(optimizer), 3 * 8)

    def test_missing_grad_does_not_decay(self) -> None:
        parameter = torch.nn.Parameter(torch.tensor([1.]))
        optimizer = Lion([parameter], weight_decay=0.1)
        optimizer.step()
        self.assertEqual(parameter.item(), 1.)
        self.assertEqual(len(optimizer.state), 0)

    def test_time_to_target(self) -> None:
        history = pd.DataFrame({"validation_accuracy": [70., 80., 85.],
                                "cumulative_training_time": [5., 12., 20.]})
        self.assertEqual(time_to_target_accuracy(history, 80.), 12.)
        self.assertTrue(np.isnan(time_to_target_accuracy(history, 90.)))

    def test_optimizer_state_round_trip(self) -> None:
        first = torch.nn.Parameter(torch.tensor([1., -1.]))
        optimizer = Lion([first], lr=0.01, weight_decay=0.05)
        first.grad = torch.tensor([2., -3.])
        optimizer.step()
        second = torch.nn.Parameter(first.detach().clone())
        restored = Lion([second])
        import copy
        restored.load_state_dict(copy.deepcopy(optimizer.state_dict()))
        for parameter, instance in ((first, optimizer), (second, restored)):
            parameter.grad = torch.tensor([-4., 2.])
            instance.step()
        torch.testing.assert_close(first, second)
        torch.testing.assert_close(optimizer.state[first]["exp_avg"],
                                   restored.state[second]["exp_avg"])

    def test_closure_runs_with_grad(self) -> None:
        parameter = torch.nn.Parameter(torch.tensor([1.]))
        optimizer = Lion([parameter], lr=0.1)

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            loss = parameter.square().sum()
            loss.backward()
            return loss

        self.assertEqual(optimizer.step(closure).item(), 1.)
        torch.testing.assert_close(parameter, torch.tensor([0.9]))

    def test_sparse_grad_is_rejected(self) -> None:
        parameter = torch.nn.Parameter(torch.ones(3))
        parameter.grad = torch.sparse_coo_tensor([[0]], [1.], (3,),
                                                 check_invariants=True)
        with self.assertRaisesRegex(RuntimeError, "dense"):
            Lion([parameter]).step()

    def test_invalid_parameters_are_rejected(self) -> None:
        for options in ({"lr": -1}, {"weight_decay": -1},
                        {"betas": (0.9, 1.)}, {"betas": (-0.1, 0.99)}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                Lion([torch.nn.Parameter(torch.ones(1))], **options)


if __name__ == "__main__":
    unittest.main()
