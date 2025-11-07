import abc

from torch.utils.data import Dataset


class TrajectoryDataset(Dataset, abc.ABC):
    """
    A dataset containing trajectories.
    TrajectoryDataset[i] returns: (observations, actions, mask)
        observations: Tensor[T, ...], T frames of observations
        actions: Tensor[T, ...], T frames of actions
        mask: Tensor[T]: 0: invalid; 1: valid
    """

    def __init__(
            self,
            device="cpu",
            obs_dim: int = 20,
            action_dim: int = 2,
            window_size: int = 1,
    ):

        self.device = device

        self.action_dim = action_dim
        self.obs_dim = obs_dim

        self.window_size = window_size

    @abc.abstractmethod
    def get_all_actions(self):
        """
        Returns all actions from all trajectories, concatenated on dim 0 (time).
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_all_observations(self):
        """
        Returns all actions from all trajectories, concatenated on dim 0 (time).
        """
        raise NotImplementedError