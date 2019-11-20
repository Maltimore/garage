"""NOP (no optimization performed) policy search algorithm."""
from garage.np.algos.base import RLAlgorithm


class NOP(RLAlgorithm):
    """NOP (no optimization performed) policy search algorithm."""

    def init_opt(self):
        """Initialize the optimization procedure."""

    def optimize_policy(self, itr, paths):
        """Optimize the policy using the samples.

        Args:
            itr (int): Iteration number.
            paths (list[dict]): A list of collected paths.

        """

    def train(self, runner):
        """Obtain samplers and start actual training for each epoch.

        Args:
            runner (LocalRunner): LocalRunner is passed to give algorithm
                the access to runner.step_epochs(), which provides services
                such as snapshotting and sampler control.

        """
