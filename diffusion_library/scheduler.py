import math
import torch

class SchedulerBase():
    def __init__(self, device):
        self.device = device
    
    def create(self, steps: int, first: float = 1, last: float = 0, device: torch.device = None, scheduler_args = None) -> torch.Tensor:
        raise NotImplementedError()


class LinearSchedule(SchedulerBase):
    def __init__(self, device:torch.device = None):
        super().__init__(device)
    
    def create(self, steps: int, first: float = 1, last: float = 0, device: torch.device = None, scheduler_args = None) -> torch.Tensor:
        return torch.linspace(first, last, steps, device = device if (device != None) else self.device)


class DDPMSchedule(SchedulerBase):
    def __init__(self, device:torch.device = None):
        super().__init__(device)

    def create(self, steps: int, first: float = 1, last: float = 0, device: torch.device = None, scheduler_args = None) -> torch.Tensor:
        ramp = torch.linspace(first, last, steps, device = device if (device != None) else self.device)
        log_snr = -torch.special.expm1(1e-4 + 10 * ramp**2).log()
        alpha = log_snr.sigmoid().sqrt()
        sigma = log_snr.neg().sigmoid().sqrt()
        return torch.atan2(sigma, alpha) / math.pi * 2


class SplicedDDPMCosineSchedule(SchedulerBase):
    def __init__(self, device:torch.device = None):
        super().__init__(device)
    
    def create(self, steps: int, first: float = 1, last: float = 0, device: torch.device = None, scheduler_args = None) -> torch.Tensor:
        ramp = torch.linspace(first, last, steps, device = device if (device != None) else self.device)
        
        ddpm_crossover = 0.48536712
        cosine_crossover = 0.80074257
        big_t = ramp * (1 + cosine_crossover - ddpm_crossover)
        
        log_snr = -torch.special.expm1(1e-4 + 10 * (big_t + ddpm_crossover - cosine_crossover)**2).log()
        alpha = log_snr.sigmoid().sqrt()
        sigma = log_snr.neg().sigmoid().sqrt()
        ddpm_part = torch.atan2(sigma, alpha) / math.pi * 2

        return torch.where(big_t < cosine_crossover, big_t, ddpm_part)


class LogSchedule(SchedulerBase):
    def __init__(self, device:torch.device = None):
        super().__init__(device)
        
    def create(self, steps: int, first: float = 1, last: float = 0, device: torch.device = None, scheduler_args = {'min_log_snr': -10, 'max_log_snr': 10}) -> torch.Tensor:
        ramp = torch.linspace(first, last, steps, device = device if (device != None) else self.device)
        min_log_snr = scheduler_args.get('min_log_snr')
        max_log_snr = scheduler_args.get('max_log_snr')
        return self.get_log_schedule(
            ramp,
            min_log_snr if min_log_snr!=None else -10,
            max_log_snr if max_log_snr!=None else 10,
        )
        
    def get_log_schedule(self, t, min_log_snr=-10, max_log_snr=10):
        log_snr = t * (min_log_snr - max_log_snr) + max_log_snr
        alpha = log_snr.sigmoid().sqrt()
        sigma = log_snr.neg().sigmoid().sqrt()
        return torch.atan2(sigma, alpha) / math.pi * 2


class CrashSchedule(SchedulerBase):
    def __init__(self, device:torch.device = None):
        super().__init__(device)
    
    def create(self, steps: int, first: float = 1, last: float = 0, device: torch.device = None, scheduler_args = None) -> torch.Tensor:
        ramp = torch.linspace(first, last, steps, device = device if (device != None) else self.device)
        sigma = torch.sin(ramp * math.pi / 2) ** 2
        alpha = (1 - sigma**2) ** 0.5
        return torch.atan2(sigma, alpha) / math.pi * 2
    