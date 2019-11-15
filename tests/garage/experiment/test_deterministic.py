import torch

from garage.experiment import deterministic


def test_deterministic_pytorch():
    """Test deterministic behavior of PyTorch"""
    deterministic.set_seed(111, pytorch=True)
    rand_tensor = torch.rand((5, 5))
    deterministic_tensor = torch.Tensor(
        [[0.715565920, 0.913992643, 0.281857729, 0.258099794, 0.631108642],
         [0.600053012, 0.931192935, 0.215290189, 0.603278518, 0.732785344],
         [0.185717106, 0.510067403, 0.754451334, 0.288391531, 0.577469587],
         [0.035843492, 0.102626860, 0.341910362, 0.439984798, 0.634111166],
         [0.622391582, 0.633447766, 0.857972443, 0.157199264, 0.785320759]])

    assert torch.all(torch.eq(rand_tensor, deterministic_tensor))
