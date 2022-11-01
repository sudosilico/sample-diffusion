import io
import argparse
import numpy as np
import onnx
import onnxruntime
import torch

from torch import nn
import torch.onnx

from typing import Tuple
from audio_diffusion.models import DiffusionAttnUnet1D
from yaml import parse
from dance_diffusion.model import DanceDiffusionInference

def main():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--ckpt",
        metavar="CHECKPOINT",
        type=str,
        default="models/model.ckpt",
        help="path to the model checkpoint file to be used (default: models/model.ckpt)",
    )
    parser.add_argument(
        "--onnx",
        metavar="ONNX",
        type=str,
        default="models/model.onnx",
        help="path to the model output (default: models/model.onnx)",
    )
    parser.add_argument(
        "--cs", 
        metavar="CHUNKSIZE",
        type=int, 
        default=65536, 
        help="the samples per chunk of the model (default: 65536)"
    )
    parser.add_argument(
        "--sr", 
        metavar="SAMPLERATE",
        type=int, 
        default=48000, 
        help="the samplerate of the model (default: 48000)"
    )
    
    args = parser.parse_args()
    
    module = DanceDiffusionInference(
        sample_size=args.cs,
        sample_rate=args.sr,
        latent_dim=0
    )
    
    map_location = lambda storage, loc: storage
    if torch.cuda.is_available():
        map_location = None
    
    module.load_state_dict(
        torch.load(args.ckpt, map_location=map_location)["state_dict"], 
        strict=False,
    )
    
    module.eval()
    
    batch_size = 2
    
    x_t = torch.randn(batch_size, 2, args.cs, requires_grad=True)
    t = torch.ones([batch_size])
    
    #torch_out = module.diffusion_ema(x_t, t)
    #print(torch_out.size())

    # Export the model
    torch.onnx.export(
        module.diffusion_ema,
        (x_t, t),
        args.onnx,
        export_params=True,
        opset_version=10,
        do_constant_folding=True,
        input_names = ['x_t', 't'],
        output_names = ['v'],
        dynamic_axes={'x_t' : {0 : 'batch_size'}, 't' : {0 : 'batch_size'}, 'v' : {0 : 'batch_size'}}
    )
    
    onnx_model = onnx.load(args.onnx)
    onnx.checker.check_model(onnx_model)

    ort_session = onnxruntime.InferenceSession("super_resolution.onnx")

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x_t)}
    ort_outs = ort_session.run(None, ort_inputs)

    # compare ONNX Runtime and PyTorch results
    np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)

    print("Exported model has been tested with ONNXRuntime, and the result looks good!")

if __name__ == "__main__":
    main()