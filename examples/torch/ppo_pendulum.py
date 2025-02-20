#!/usr/bin/env python3
"""This is an example to train a task with PPO algorithm (PyTorch).

Here it runs InvertedDoublePendulum-v2 environment with 100 iterations.
"""
import torch

from garage.experiment import LocalRunner, run_experiment
from garage.np.baselines import LinearFeatureBaseline
from garage.tf.envs import TfEnv
from garage.torch.algos import PPO
from garage.torch.policies import GaussianMLPPolicy


def run_task(snapshot_config, *_):
    """Set up environment and algorithm and run the task.

    Args:
        snapshot_config (garage.experiment.SnapshotConfig): The snapshot
            configuration used by LocalRunner to create the snapshotter.
            If None, it will create one with default settings.
        _ : Unused parameters

    """
    env = TfEnv(env_name='InvertedDoublePendulum-v2')

    runner = LocalRunner(snapshot_config)

    policy = GaussianMLPPolicy(env.spec,
                               hidden_sizes=[64, 64],
                               hidden_nonlinearity=torch.tanh,
                               output_nonlinearity=None)

    baseline = LinearFeatureBaseline(env_spec=env.spec)

    algo = PPO(env_spec=env.spec,
               policy=policy,
               optimizer=torch.optim.Adam,
               baseline=baseline,
               max_path_length=100,
               discount=0.99,
               center_adv=False,
               policy_lr=3e-4)

    runner.setup(algo, env)
    runner.train(n_epochs=100, batch_size=10000)


run_experiment(
    run_task,
    snapshot_mode='last',
    seed=1,
)
