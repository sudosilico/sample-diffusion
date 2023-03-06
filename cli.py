import json
import torch
import argparse

from util.util import load_audio, save_audio, cropper
from util.platform import get_torch_device_type
from dance_diffusion.api import RequestHandler, Request, Response, RequestType, SamplerType, SchedulerType, ModelType


def main():
    args = parse_cli_args()
    
    device_type_accelerator = args.device_accelerator if(args.device_accelerator != None) else get_torch_device_type()
    device_accelerator = torch.device(device_type_accelerator)
    device_offload = torch.device(args.device_offload)
    
    autocrop = cropper(args.chunk_size, args.crop_randomly) if(args.use_autocrop==True) else lambda audio: audio
    
    request_handler = RequestHandler(device_accelerator, device_offload, optimize_memory_use=False, use_autocast=args.use_autocast)
    
    seed = args.seed if(args.seed!=-1) else torch.randint(0, 4294967294, [1], device=device_type_accelerator).item()
    print(f"Using accelerator: {device_type_accelerator}, Seed: {seed}.")
    
    request = Request(
        request_type=args.mode,
        model_path=args.model,
        model_type=ModelType.DD,
        model_chunk_size=args.chunk_size,
        model_sample_rate=args.sample_rate,
        
        seed=seed,
        batch_size=args.batch_size,
        
        audio_source=autocrop(load_audio(device_accelerator,args.audio_source, args.sample_rate)) if(args.audio_source != None) else None,
        audio_target=autocrop(load_audio(device_accelerator,args.audio_target, args.sample_rate)) if(args.audio_target != None) else None,
        mask=torch.load(args.mask) if(args.mask != None) else None,
        
        noise_level=args.noise_level,
        interpolation_positions=args.interpolations if(args.interpolations_linear == None) else torch.linspace(0, 1, args.interpolations_linear, device=device_accelerator),
        resamples=args.resamples,
        keep_start=args.keep_start,
                
        steps=args.steps,
        
        sampler_type=args.sampler,
        sampler_args=args.sampler_args,
        
        scheduler_type=args.schedule,
        scheduler_args=args.schedule_args
    )
    
    response = request_handler.process_request(request)#, lambda **kwargs: print(f"{kwargs['step'] / kwargs['x']}"))
    save_audio((0.5 * response.result).clamp(-1,1) if(args.tame == True) else response.result, f"audio/Output/{ModelType.DD.__str__()}/{args.mode.__str__()}/", args.sample_rate, f"{seed}")


def str2bool(value):
    if value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_cli_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use_autocast",
        type=str2bool,
        default=True,
        help="Use autocast."
    )
    parser.add_argument(
        "--use_autocrop",
        type=str2bool,
        default=True,
        help="Use autocrop(automatically crops audio provided to chunk_size)."
    )
    parser.add_argument(
        "--crop_randomly",
        type=str2bool,
        default=False,
        help="Whether autocrop should crop randomly."
    )
    parser.add_argument(
        "--device_accelerator",
        type=str,
        default=None,
        help="Device of execution."
    )
    parser.add_argument(
        "--device_offload",
        type=str,
        default="cpu",
        help="Device to store models when not in use."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/dd/model.ckpt",
        help="Path to the model checkpoint file to be used (default: models/dd/model.ckpt)."
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=48000,
        help="The samplerate the model was trained on."
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=65536,
        help="The native chunk size of the model."
    )
    parser.add_argument(
        "--mode",
        type=RequestType,
        choices=RequestType,
        default=RequestType.Generation,
        help="The mode of operation (Generation, Variation, Interpolation, Inpainting or Extension)."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=-1,
        help="The seed used for reproducable outputs. Leave empty for random seed."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="The maximal number of samples to be produced per batch."
    )
    parser.add_argument(
        "--audio_source",
        type=str,
        default=None,
        help="Path to the audio source."
    )   
    parser.add_argument(
        "--audio_target",
        type=str,
        default=None,
        help="Path to the audio target (used for interpolations)."
    )
    parser.add_argument(
        "--mask",
        type=str,
        default=None,
        help="Path to the mask tensor (used for inpainting)."
    )  
    parser.add_argument(
        "--noise_level",
        type=float,
        default=0.7,
        help="The noise level used for variations & interpolations."
    )
    parser.add_argument(
        "--interpolations_linear",
        type=int,
        default=None,
        help="The number of interpolations, even spacing."
    )
    parser.add_argument(
        "--interpolations",
        nargs='+',
        type=float,
        default=None,
        help="The interpolation positions."
    )
    parser.add_argument(
        "--resamples",
        type=int,
        default=4,
        help="Number of resampling steps in conventional samplers for inpainting."
    )
    parser.add_argument(
        "--keep_start",
        type=str2bool,
        default=True,
        help="Keep beginning of audio provided(only applies to mode Extension)."
    )
    parser.add_argument(
        "--tame",
        type=str2bool,
        default=True,
        help="Decrease output by 3db, then clip."
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="The number of steps for the sampler."
    )
    parser.add_argument(
        "--sampler",
        type=SamplerType,
        choices=SamplerType,
        default=SamplerType.IPLMS,
        help="The sampler used for the diffusion model."
    )
    parser.add_argument(
        "--sampler_args",
        type=json.loads,
        default={'use_tqdm': True},
        help="Additional arguments of the DD sampler."
    )
    parser.add_argument(
        "--schedule",
        type=SchedulerType,
        choices=SchedulerType,
        default=SchedulerType.CrashSchedule,
        help="The schedule used for the diffusion model."
    )
    parser.add_argument(
        "--schedule_args",
        type=json.loads,
        default={},
        help="Additional arguments of the DD schedule."
    )
    return parser.parse_args()


if __name__ == '__main__':
    main()