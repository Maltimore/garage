"""PyTorch algorithms."""
from garage.torch.algos._utils import (  # noqa: F401
    _compute_advantages, _Default, _filter_valids, _make_optimizer,
    _pad_to_last)
from garage.torch.algos.ddpg import DDPG
from garage.torch.algos.vpg import VPG
from garage.torch.algos.ppo import PPO  # noqa: I100
from garage.torch.algos.trpo import TRPO

__all__ = ['DDPG', 'VPG', 'PPO', 'TRPO']
