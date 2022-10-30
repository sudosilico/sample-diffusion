import math
import torch


class ScheduleBase():
    def __init__(self, device):
        self.device = device
    
    def create(self, steps: int, first: int = 1, last: int = 0) -> torch.Tensor:
        raise NotImplementedError()


class LinearSchedule(ScheduleBase):
    def __init__(self, device):
        super().__init__(device)
    
    def create(self, steps: int, first: int = 1, last: int = 0) -> torch.Tensor:
        return torch.linspace(first, last, steps, device = self.device)


class CrashSchedule(ScheduleBase):
    def __init__(self, device):
        super().__init__(device)
    
    def create(self, steps: int, first: int = 1, last: int = 0) -> torch.Tensor:
        t = torch.linspace(first, last, steps, device = self.device)
        sigma = torch.sin(t * math.pi / 2) ** 2
        alpha = (1 - sigma**2) ** 0.5
        return torch.atan2(sigma, alpha) / math.pi * 2
    