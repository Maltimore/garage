"""Tests for DifferentialSGD optimizer."""
import torch

from garage.torch.optimizers import DiffSGD


def test_differentiable_sgd():
    """Test second order derivative after taking optimization step."""
    policy = torch.nn.Linear(10, 10, bias=False)
    lr = 0.01
    diff_sgd = DiffSGD(policy, lr=lr)

    theta = list(policy.parameters())[0]
    meta_loss = torch.sum(theta**2)
    meta_loss.backward(create_graph=True)

    diff_sgd.step()

    theta_prime = list(policy.parameters())[0]
    assert theta.ne(theta_prime).all()

    loss = torch.sum(theta_prime**2)
    result = torch.autograd.grad(loss, theta)[0]

    assert theta_prime.grad is not None

    dtheta_prime = 1 - 2 * lr  # dtheta_prime/dtheta
    dloss = 2 * theta_prime  # dloss/dtheta_prime
    expected_result = dloss * dtheta_prime  # dloss/dtheta

    assert torch.allclose(result, expected_result)
