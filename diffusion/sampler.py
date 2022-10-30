import torch
from tqdm.auto import trange
from dataclasses import dataclass
from typing import Callable, Tuple

from dance_diffusion.model import ModelBase
from diffusion.util import t_to_alpha_sigma


@dataclass
class TAS:
    """
    Combine timestep, alpha and sigma values.
    """
    t: torch.Tensor
    alpha: torch.Tensor
    sigma: torch.Tensor


class SamplerBase:
    """
    The base class for diffusion samplers.

    The following functions should be implemented in child types:
        
    ```
    def _sample(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()
    ```

    The `_sample` function is internally called by the public `sample` function.
    """
    
    def __init__(self, model_func: Callable):
        self.model_func = model_func
        
    def sample(self, x_t: torch.Tensor, ts: torch.Tensor, callback: Callable = None) -> torch.Tensor:
        # Apple Silicon MPS fix :)
        if(ts[0] == 1.0):
            _, sigma = t_to_alpha_sigma(ts[1])
            x_t = x_t * sigma
            ts = ts[1:]
        
        return self._sample(x_t, ts, callback)

    def _sample(self, x_t: torch.Tensor, ts: torch.Tensor, callback: Callable) -> torch.Tensor:
        raise NotImplementedError()


class ImprovedPseudoLinearMultiStep(SamplerBase):
    """
    Improved Pseudo Linear Multistep or "IPLMS" sampler.
    """
    def __init__(self, model: ModelBase):
        self.model = model
        super().__init__(model.v_function)

    def step(self, x_t: torch.Tensor, eps_cache: torch.Tensor, tas_now: TAS, tas_next: TAS, callback: Callable) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        v_t = self.model_func(x_t, tas_now.t)
        eps_t = x_t * tas_now.sigma + v_t * tas_now.alpha
        
        if len(eps_cache) == 0:
            eps_mt = eps_t
        elif len(eps_cache) == 1:
            eps_mt = (3/2 * eps_t - 1/2 * eps_cache[-1])
        elif len(eps_cache) == 2:
            eps_mt = (23/12 * eps_t - 16/12 * eps_cache[-1] + 5/12 * eps_cache[-2])
        else:
            eps_mt = (55/24 * eps_t - 59/24 * eps_cache[-1] + 37/24 * eps_cache[-2] - 9/24 * eps_cache[-3])

        pred_mt = (x_t - eps_mt * tas_now.sigma) / tas_now.alpha #torch.where(tas_now.alpha == 0, sys.float_info.epsilon, tas_now.alpha) 'orig mps fix'
        x_next = pred_mt * tas_next.alpha + eps_mt * tas_next.sigma
        
        if callback is not None:
            callback({'x': x_t, 't': tas_now.t, 'pred': (x_t - eps_t * tas_now.sigma) / tas_now.alpha})

        return x_next, eps_t

    def _sample(self, x_t: torch.Tensor, ts: torch.Tensor, callback: Callable) -> torch.Tensor:

        steps = ts.size(0) #timesteps
        batch_size = x_t.size(0)
        
        eps_cache = []
        alphas, sigmas = t_to_alpha_sigma(ts)
        
        for step in trange(steps - 1):
            
            x_t, eps_t = self.step(
                x_t, 
                eps_cache, 
                TAS(ts[step].expand(batch_size), alphas[step], sigmas[step]), 
                TAS(ts[step + 1].expand(batch_size), alphas[step + 1], sigmas[step + 1]), 
                callback
            )
            
            if len(eps_cache) >= 3:
                eps_cache.pop(0)
            eps_cache.append(eps_t)
        
        return x_t
       