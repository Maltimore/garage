"""Collects samples in parallel using a stateful pool of workers."""

import tensorflow as tf

from garage.sampler import parallel_sampler
from garage.sampler.base import BaseSampler
from garage.sampler.stateful_pool import singleton_pool
from garage.sampler.utils import truncate_paths


def worker_init_tf(g):
    """Initialize the tf.Session on a worker.

    Args:
        g (LocalTFRunner): The Runner.

    """
    g.sess = tf.compat.v1.Session()
    g.sess.__enter__()


def worker_init_tf_vars(g):
    """Initialize the policy parameters on a worker.

    Args:
        g (LocalTFRunner): The Runner.

    """
    g.sess.run(tf.compat.v1.global_variables_initializer())


class BatchSampler(BaseSampler):
    """Collects samples in parallel using a stateful pool of workers.

    Args:
        algo (garage.np.algos.RLAlgorithm): The algorithm.
        env (gym.Env): The environment.
        n_envs (int): Number of environments.

    """

    def __init__(self, algo, env, n_envs):
        super().__init__(algo, env)
        self.n_envs = n_envs

    def start_worker(self):
        """Initialize the sampler."""
        assert singleton_pool.initialized, (
            'Use singleton_pool.initialize(n_parallel) to setup workers.')
        if singleton_pool.n_parallel > 1:
            singleton_pool.run_each(worker_init_tf)
        parallel_sampler.populate_task(self.env, self.algo.policy)
        if singleton_pool.n_parallel > 1:
            singleton_pool.run_each(worker_init_tf_vars)

    def shutdown_worker(self):
        """Terminate workers if necessary."""
        parallel_sampler.terminate_task(scope=self.algo.scope)

    # pylint: disable=arguments-differ
    def obtain_samples(self, itr, batch_size=None, whole_paths=True):
        """Collect samples for the given iteration number.

        Args:
            itr (int): number of iteration
            batch_size (int): number of batch size
            whole_paths (bool): whether to use whole path or truncated

        Returns:
            list[dict]: A list of paths.

        """
        if not batch_size:
            batch_size = self.algo.max_path_length * self.n_envs

        cur_policy_params = self.algo.policy.get_param_values()
        paths = parallel_sampler.sample_paths(
            policy_params=cur_policy_params,
            max_samples=batch_size,
            max_path_length=self.algo.max_path_length,
            scope=self.algo.scope,
        )

        return paths if whole_paths else truncate_paths(paths, batch_size)
