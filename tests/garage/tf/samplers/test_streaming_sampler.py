import gym
import numpy as np
import tensorflow as tf

from garage.experiment import LocalRunner
from garage.np.exploration_strategies import OUStrategy
from garage.replay_buffer import SimpleReplayBuffer
from garage.sampler.streaming_sampler import StreamingSampler
from garage.tf.algos import DDPG
from garage.tf.envs import TfEnv
from garage.tf.policies import ContinuousMLPPolicy
from garage.tf.q_functions import ContinuousMLPQFunction
from tests.fixtures import TfGraphTestCase


class TestStreamingSampler(TfGraphTestCase):
    def test_off_policy(self):
        with LocalRunner(self.sess) as runner:
            env = TfEnv(gym.make('InvertedDoublePendulum-v2'))
            action_noise = OUStrategy(env.spec, sigma=0.2)
            policy = ContinuousMLPPolicy(
                env_spec=env.spec,
                hidden_sizes=[64, 64],
                hidden_nonlinearity=tf.nn.relu,
                output_nonlinearity=tf.nn.tanh)
            qf = ContinuousMLPQFunction(
                env_spec=env.spec,
                hidden_sizes=[64, 64],
                hidden_nonlinearity=tf.nn.relu)
            replay_buffer = SimpleReplayBuffer(
                env_spec=env.spec,
                size_in_transitions=int(1e6),
                time_horizon=100)
            algo = DDPG(
                env_spec=env.spec,
                policy=policy,
                policy_lr=1e-4,
                qf_lr=1e-3,
                qf=qf,
                replay_buffer=replay_buffer,
                target_update_tau=1e-2,
                n_train_steps=50,
                discount=0.9,
                min_buffer_size=int(1e4),
                exploration_strategy=action_noise,
            )

            sampler = StreamingSampler.from_algo(algo)
            sampler.start_worker()

            runner.initialize_tf_vars()

            paths1 = sampler.obtain_samples(0, 5)
            paths2 = sampler.obtain_samples(0, 5)

            len1 = sum([len(path['rewards']) for path in paths1])
            len2 = sum([len(path['rewards']) for path in paths2])

            assert len1 == 5 and len2 == 5, 'Sampler should respect batch_size'

            assert (len(paths1[0]['rewards']) + len(paths2[0]['rewards'])
                    == paths2[0]['running_length']), \
                'Running length should be the length of full path'

            assert np.isclose(
                paths1[0]['rewards'].sum() + paths2[0]['rewards'].sum(),
                paths2[0]['undiscounted_return']
            ), 'Undiscounted_return should be the sum of rewards of full path'

            sampler.shutdown_worker()
