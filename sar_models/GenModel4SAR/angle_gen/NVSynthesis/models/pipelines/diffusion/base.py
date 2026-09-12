import logging


class BasePiPL:
    """Shared state for diffusion pipelines."""

    def __init__(self, opt, device=None):
        self.opt = opt
        self.device = device
        self.T = opt["pipL"]["total_timestep"]
        self.logger = logging.getLogger("base")
        self.record = []

    def set_scheme(self, scheme):
        self.scheme = scheme
