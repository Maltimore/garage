"""This script creates a test that fails when TRPO performance is too low."""
import gym
import torch

from garage.envs import normalize
from garage.envs.base import GarageEnv
from garage.experiment import deterministic, LocalRunner
from garage.np.baselines import LinearFeatureBaseline
from garage.torch.algos import TRPO
from garage.torch.policies import GaussianMLPPolicy
from tests.fixtures import snapshot_config


class TestTRPO:
    """Test class for TRPO."""

    def setup_method(self):
        """Setup method which is called before every test."""
        self.env = GarageEnv(normalize(gym.make('InvertedDoublePendulum-v2')))
        self.policy = GaussianMLPPolicy(
            env_spec=self.env.spec,
            hidden_sizes=(64, 64),
            hidden_nonlinearity=torch.tanh,
            output_nonlinearity=None,
        )
        self.baseline = LinearFeatureBaseline(env_spec=self.env.spec)

    def teardown_method(self):
        """Teardown method which is called after eevery test."""
        self.env.close()

    def test_trpo_pendulum(self):
        """Test TRPO with Pendulum environment."""
        deterministic.set_seed(0)

        runner = LocalRunner(snapshot_config)
        algo = TRPO(env_spec=self.env.spec,
                    policy=self.policy,
                    baseline=self.baseline,
                    max_path_length=100,
                    discount=0.99,
                    gae_lambda=0.98)

        runner.setup(algo, self.env)
        last_avg_ret = runner.train(n_epochs=10, batch_size=100)
        assert last_avg_ret > 50
