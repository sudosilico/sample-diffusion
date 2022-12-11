import math
import torch

from tqdm.auto import trange
from typing import Callable, Tuple

from .util import t_to_alpha_sigma


class SamplerBase():
    
    def __init__(self, model_fn: Callable):
        self.model_fn = model_fn
        self.generator = None
        
    def set_model_fn(self, model_fn: Callable):
        self.model_fn = model_fn
        
    def get_model_fn(self):
        return self.model_fn
    
    def sample(
        self,
        x_t: torch.Tensor,
        ts: torch.Tensor,
        model_fn: Callable = None,
        callback: Callable = None,
        model_args = {},
        sampler_args = {}
    ) -> torch.Tensor:

        # Select model function to be used:
        model_fn = model_fn if (model_fn != None) else self.model_fn
        self.generator = torch.Generator(x_t.device if (x_t.device.type != 'mps') else torch.device('cpu'))
        
        return self._sample(model_fn, x_t, ts, callback, model_args, **sampler_args)
        
    def _sample(
        self,
        model_fn: Callable,
        x_t: torch.Tensor,
        ts: torch.Tensor,
        callback: Callable,
        model_args,
        **sampler_args
    ) -> torch.Tensor:
        raise NotImplementedError
    
    def inpaint(
        self,
        audio_source: torch.Tensor,
        mask: torch.Tensor,
        ts: torch.Tensor,
        resamples: int,
        model_fn: Callable = None,
        callback: Callable = None,
        model_args = {},
        sampler_args = {}
    ) -> torch.Tensor:
        
        # Select model function to be used:
        model_fn = model_fn if (model_fn != None) else self.model_fn
        self.generator = torch.Generator(audio_source.device if (audio_source.device.type != 'mps') else torch.device('cpu'))
        
        return self._inpaint(model_fn, audio_source, mask, ts, resamples, callback, model_args, **sampler_args)
        
    def _inpaint(
        self,
        model_fn: Callable,
        audio_source: torch.Tensor,
        mask: torch.Tensor,
        ts: torch.Tensor,
        resamples: int,
        callback: Callable,
        model_args,
        **sampler_args
    ) -> torch.Tensor:
        raise NotImplementedError('Inpainting not supported with this sampler (yet)')


class DDPM(SamplerBase):
    
    def __init__(self, model_fn: Callable = None):
        super().__init__(model_fn)
    
    def _step(
        self,
        model_fn: Callable,
        x_t: torch.Tensor,
        step: int,
        t_now: torch.Tensor,
        t_next: torch.Tensor,
        callback: Callable,
        model_args
    ) -> torch.Tensor:
        
        alpha_now, sigma_now = t_to_alpha_sigma(t_now) # Get alpha / sigma for current timestep.
        alpha_next, sigma_next = t_to_alpha_sigma(t_next) # Get alpha / sigma for next timestep.
        
        v_t = model_fn(x_t, t_now.expand(x_t.shape[0]), **model_args) # Expand t to match batch_size which corresponds to x_t.shape[0]
        
        eps_t = x_t * sigma_now + v_t * alpha_now
        pred_t = x_t * alpha_now - v_t * sigma_now
        
        if callback is not None:
            callback({'step': step, 'x': x_t, 't': t_now, 'pred': pred_t, 'eps': eps_t})
            
        return (pred_t * alpha_next + eps_t * sigma_next)
    
    def _sample(
        self,
        model_fn: Callable,
        x_t: torch.Tensor,
        ts: torch.Tensor,
        callback: Callable,
        model_args,
        **sampler_args
    ) -> torch.Tensor:
        
        print("Using DDPM Sampler.")
        steps = ts.size(0)
        
        use_tqdm = sampler_args.get('use_tqdm')
        use_range = trange if (use_tqdm if (use_tqdm != None) else False) else range
        
        for step in use_range(steps - 1):
            x_t = self._step(
                model_fn,
                x_t,
                step,
                ts[step],
                ts[step + 1],
                lambda kwargs: callback(**dict(kwargs, steps=steps)) if(callback != None) else None,
                model_args
            )
            
        return x_t
    
    def _inpaint(
        self,
        model_fn: Callable,
        audio_source: torch.Tensor,
        mask: torch.Tensor,
        ts: torch.Tensor,
        resamples: int,
        callback: Callable,
        model_args,
        **sampler_args
    ) -> torch.Tensor:
        
        steps = ts.size(0)
        batch_size = audio_source.size(0)
        alphas, sigmas = t_to_alpha_sigma(ts)

        x_t = audio_source
        
        use_tqdm = sampler_args.get('use_tqdm')
        use_range = trange if (use_tqdm if (use_tqdm != None) else False) else range
        
        for step in use_range(steps - 1):
            
            audio_source_noised = audio_source * alphas[step] + torch.randn_like(audio_source) * sigmas[step]
            sigma_dt = math.sqrt(sigmas[step] ** 2 - sigmas[step + 1] ** 2)
            
            for re in range(resamples):
                
                x_t = audio_source_noised * mask + x_t * ~mask
                
                v_t = model_fn(x_t, ts[step].expand(batch_size), **model_args)
        
                eps_t = x_t * sigmas[step] + v_t * alphas[step]
                pred_t = x_t * alphas[step] - v_t * sigmas[step]
                
                if callback is not None:
                    callback({'steps': steps, 'step': step, 'x': x_t, 't': ts[step], 'pred': pred_t, 'eps': eps_t, 'res': re})
                
                if(re < resamples - 1):
                    x_t = pred_t * alphas[step] + eps_t * sigmas[step + 1] + sigma_dt * torch.randn_like(x_t)
                else:
                    x_t = pred_t * alphas[step + 1] + eps_t * sigmas[step + 1]

        return (audio_source * mask + x_t * ~mask)
    
    
class DDIM(SamplerBase):

    def __init__(self, model_fn: Callable = None):
        super().__init__(model_fn)
    
    def _sample(
        self,
        model_fn: Callable,
        x_t: torch.Tensor,
        ts: torch.Tensor,
        callback: Callable,
        model_args,
        **sampler_args
    ) -> torch.Tensor:
        
        print("Using DDIM Sampler.")
        steps = ts.size(0)
        
        eta = sampler_args.get('eta')
        use_tqdm = sampler_args.get('use_tqdm')
        use_range = trange if (use_tqdm if (use_tqdm != None) else False) else range
        
        for step in use_range(steps - 1):
            
            alpha_now, sigma_now = t_to_alpha_sigma(ts[step]) # Get alpha / sigma for current timestep.
            alpha_next, sigma_next = t_to_alpha_sigma(ts[step + 1]) # Get alpha / sigma for next timestep.
            
            v_t = model_fn(x_t, ts[step].expand(x_t.shape[0]), **model_args) # Expand t to match batch_size which corresponds to x_t.shape[0]
            
            eps_t = x_t * sigma_now + v_t * alpha_now
            pred_t = x_t * alpha_now - v_t * sigma_now
            
            if callback is not None:
                callback({'step': step, 'x': x_t, 't': ts[step], 'pred': pred_t, 'eps': eps_t})
                
            if(step < steps - 1):
                ddim_sigma = eta * math.sqrt(sigma_next**2 / sigma_now**2) * math.sqrt(1 - alpha_now**2 / alpha_next**2)
                adjusted_sigma = math.sqrt(sigma_next**2 - ddim_sigma**2)
                x_t = (pred_t * alpha_next + eps_t * adjusted_sigma + torch.randn_like(x_t) * ddim_sigma)
            else:
                x_t = (pred_t * alpha_next + eps_t * adjusted_sigma)
            
        return x_t
    
    
class IPLMS(SamplerBase):
    
    def __init__(self, model_fn: Callable = None):
        super().__init__(model_fn)
        
    def _step(
        self,
        model_fn: Callable,
        x_t: torch.Tensor,
        eps_cache: torch.Tensor,
        step: int,
        t_now: torch.Tensor,
        t_next: torch.Tensor,
        callback: Callable,
        model_args
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        alpha_now, sigma_now = t_to_alpha_sigma(t_now)
        alpha_next, sigma_next = t_to_alpha_sigma(t_next)
        
        v_t = model_fn(x_t, t_now.expand(x_t.shape[0]), **model_args) # Expand t to match batch_size which corresponds to x_t.shape[0]
        eps_t = x_t * sigma_now + v_t * alpha_now
        
        if len(eps_cache) == 0:
            eps_mt = eps_t
        elif len(eps_cache) == 1:
            eps_mt = (3/2 * eps_t - 1/2 * eps_cache[-1])
        elif len(eps_cache) == 2:
            eps_mt = (23/12 * eps_t - 16/12 * eps_cache[-1] + 5/12 * eps_cache[-2])
        else:
            eps_mt = (55/24 * eps_t - 59/24 * eps_cache[-1] + 37/24 * eps_cache[-2] - 9/24 * eps_cache[-3])

        pred_mt = (x_t - eps_mt * sigma_now) / alpha_now
        
        if callback is not None:
            pred = (x_t - eps_t * sigma_now) / alpha_now
            callback({'step': step, 'x': x_t, 't': t_now, 'pred': pred, 'eps': eps_t})
            
        return (pred_mt * alpha_next + eps_mt * sigma_next), eps_t
    
    def _sample(
        self,
        model_fn: Callable,
        x_t: torch.Tensor,
        ts: torch.Tensor,
        callback: Callable,
        model_args,
        **sampler_args
    ) -> torch.Tensor:
        
        print("Using IPLMS Sampler.")
        
        if(ts[0] == 1.0):# Apple Silicon MPS fix :)
            _, sigma = t_to_alpha_sigma(ts[1])
            x_t = x_t * sigma
            ts = ts[1:]
        
        steps = ts.shape[0]
        eps_cache = []

        use_tqdm = sampler_args.get('use_tqdm')
        use_range = trange if(use_tqdm if(use_tqdm != None) else False) else range
        
        for step in use_range(steps - 1):
            
            x_t, eps_t = self._step(
                model_fn,
                x_t,
                eps_cache,
                step,
                ts[step],
                ts[step + 1],
                lambda kwargs: callback(**dict(kwargs, steps=steps)) if(callback != None) else None,
                model_args
            )
            
            if len(eps_cache) >= 3:
                eps_cache.pop(0)
            eps_cache.append(eps_t)
        
        return x_t
    