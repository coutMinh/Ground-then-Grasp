import torch


def pytest_sessionstart(session):
    # Small CPU tensor operations are slower with the host's 20-thread default.
    torch.set_num_threads(2)
