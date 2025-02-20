"""
This script creates a test that fails when garage.tf.algos.PPO performance is
too low.
"""
import gym
import torch

from garage.envs import normalize
from garage.envs.base import GarageEnv
from garage.experiment import deterministic, LocalRunner
from garage.np.baselines import LinearFeatureBaseline
from garage.torch.algos import PPO
from garage.torch.policies import GaussianMLPPolicy
from tests.fixtures import snapshot_config


class TestPPO:

    def setup_method(self):
        self.env = GarageEnv(normalize(gym.make('InvertedDoublePendulum-v2')))
        self.policy = GaussianMLPPolicy(
            env_spec=self.env.spec,
            hidden_sizes=(64, 64),
            hidden_nonlinearity=torch.tanh,
            output_nonlinearity=None,
        )
        self.baseline = LinearFeatureBaseline(env_spec=self.env.spec)

    def teardown_method(self):
        self.env.close()

    def test_ppo_pendulum(self):
        """Test DDPG with Pendulum environment."""
        deterministic.set_seed(0)

        runner = LocalRunner(snapshot_config)
        algo = PPO(env_spec=self.env.spec,
                   policy=self.policy,
                   baseline=self.baseline,
                   optimizer=torch.optim.Adam,
                   max_path_length=100,
                   discount=0.99,
                   gae_lambda=0.97,
                   lr_clip_range=2e-1,
                   policy_lr=3e-4)

        runner.setup(algo, self.env)
        last_avg_ret = runner.train(n_epochs=10, batch_size=100)
        assert last_avg_ret > 0
