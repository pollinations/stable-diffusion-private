import sys

sys.path.append("/CLIP")
sys.path.append("/taming-transformers")
sys.path.append("/k-diffusion")
# Slightly modified version of: https://github.com/CompVis/stable-diffusion/blob/main/scripts/txt2img.py
import os
import sys
import time
# Code to turn kwargs into Jupyter widgets
from collections import OrderedDict
from contextlib import contextmanager, nullcontext

import numpy as np
import torch
from cog import BasePredictor, Input, Path
from einops import rearrange
from k_diffusion import sampling
from k_diffusion.external import CompVisDenoiser
from omegaconf import OmegaConf
from PIL import Image
from pytorch_lightning import seed_everything
from torch import autocast
#from tqdm.auto import tqdm, trange  # NOTE: updated for notebook
from tqdm import tqdm, trange  # NOTE: updated for notebook

from helpers import sampler_fn, save_samples
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.util import instantiate_from_config
from scripts.txt2img import chunk, load_model_from_config


class Predictor(BasePredictor):


    @torch.inference_mode()
    def setup(self):
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        options = get_default_options()
        self.options = options

        self.model = load_model(self.options, self.device)
        self.model_wrap = CompVisDenoiser(self.model)



    @torch.inference_mode()
    def predict(
        self,
        prompts: str = Input(
            default="Apple by magritte\nBanana by magritte",
            description="model will try to generate this text.",
        ),
        prompt_scale: float = Input(
            default=5.0,
            description="Determines influence of your prompt on generation.",
        ),
        num_frames_per_prompt: int = Input(
            default=15,
            description="Number of frames to generate per prompt (limited to a maximum of 35 for now because we are experiencing heavy use).",
        ),
        random_seed: int = Input(
            default=42,
            description="Each seed generates a different image",
        ),
        width: int = Input(
            default=512,
            description="Width of the generated image. The model was really only trained on 512x512 images. Other sizes tend to create less coherent images.",
        ),
        height: int = Input(
            default=512,
            description="Height of the generated image. The model was really only trained on 512x512 images. Other sizes tend to create less coherent images.",
        ),
    ) -> Path:
        
        num_frames_per_prompt = abs(min(num_frames_per_prompt, 35))

        
        options = self.options
        options['prompts'] = prompts.split("\n")
        options['prompts'] = [prompt.strip() for prompt in options['prompts'] if prompt.strip()]
        
        options['num_interpolation_steps'] = num_frames_per_prompt
        options['scale'] = prompt_scale
        options['seed'] = random_seed
        options['H'] = height
        options['W'] = width

        run_inference(options, self.model, self.model_wrap, self.device)

        #if num_frames_per_prompt == 1:
        #    return Path(options['output_path'])     
        encoding_options = "-c:v libx264 -crf 20 -preset slow -vf format=yuv420p -c:a aac -movflags +faststart"
        os.system("ls -l /outputs")

        # calculate the frame rate of the video so that the length is always 8 seconds
        frame_rate = num_frames_per_prompt / 8

        os.system(f'ffmpeg -y -r {frame_rate} -i {options["outdir"]}/%*.png {encoding_options} /tmp/z_interpollation.mp4')
        
        return Path("/tmp/z_interpollation.mp4")

def load_model(opt,device):
    """Seperates the loading of the model from the inference"""
    
    # if opt.laion400m:
    #     print("Falling back to LAION 400M model...")
    #     opt.config = "configs/latent-diffusion/txt2img-1p4B-eval.yaml"
    #     opt.ckpt = "models/ldm/text2img-large/model.ckpt"
    #     opt.outdir = "outputs/txt2img-samples-laion400m"

    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")

    model = model.to(device)
    
    return model

def slerp(t, v0, v1, DOT_THRESHOLD=0.9995):
    """ helper function to spherically interpolate two arrays v1 v2 """

    if not isinstance(v0, np.ndarray):
        inputs_are_torch = True
        input_device = v0.device
        v0 = v0.cpu().numpy()
        v1 = v1.cpu().numpy()

    dot = np.sum(v0 * v1 / (np.linalg.norm(v0) * np.linalg.norm(v1)))
    if np.abs(dot) > DOT_THRESHOLD:
        v2 = (1 - t) * v0 + t * v1
    else:
        theta_0 = np.arccos(dot)
        sin_theta_0 = np.sin(theta_0)
        theta_t = theta_0 * t
        sin_theta_t = np.sin(theta_t)
        s0 = np.sin(theta_0 - theta_t) / sin_theta_0
        s1 = sin_theta_t / sin_theta_0
        v2 = s0 * v0 + s1 * v1

    if inputs_are_torch:
        v2 = torch.from_numpy(v2).to(input_device)

    return v2

def diffuse(count_start, start_code, c, batch_size, opt, model, model_wrap, outpath, device):
    #print("diffusing with batch size", batch_size)
    uc = None
    if opt.scale != 1.0:
        uc = model.get_learned_conditioning(batch_size * [""])
    shape = [opt.C, opt.H // opt.f, opt.W // opt.f]

    #if args.sampler in ["klms","dpm2","dpm2_ancestral","heun","euler","euler_ancestral"]:
    samples = sampler_fn(
        c=c,
        uc=uc,
        args=opt,
        model_wrap=model_wrap,
        init_latent=start_code,
        t_enc=0,
        device=device,
        # cb=callback
        )
    # samples, _ = sampler.sample(S=opt.ddim_steps,
    #                                 conditioning=c,
    #                                 batch_size=batch_size,
    #                                 shape=shape,
    #                                 verbose=False,
    #                                 unconditional_guidance_scale=opt.scale,
    #                                 unconditional_conditioning=uc,
    #                                 eta=opt.ddim_eta,
    #                                 x_T=start_code)
    print("samples_ddim", samples.shape)
    x_samples = model.decode_first_stage(samples)
    x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
    if not opt.skip_save:
        count = count_start
        for x_sample in x_samples:
            x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
            Image.fromarray(x_sample.astype(np.uint8)).save(
                os.path.join(outpath, f"{count:05}.png"))
            count += 1
    



def run_inference(opt, model, model_wrap, device):
    """Seperates the loading of the model from the inference
    
    Additionally, slightly modified to display generated images inline
    """
    seed_everything(opt.seed)

    # if opt.plms:
    #     sampler = PLMSSampler(model)
    # else:
    #     sampler = DDIMSampler(model)

    outpath = opt.outdir
    os.makedirs(outpath, exist_ok=True)

    batch_size = opt.n_samples

    
    prompts = opt.prompts

    
    # add first prompt to end to create a video for single prompts
    single_prompt = False
    if len(prompts) == 1:
        single_prompt = True
        prompts = prompts + [prompts[0]]


    print("embedding prompts")
    cs = [model.get_learned_conditioning(prompt) for prompt in prompts]

    datas = [[batch_size * c] for c in cs] 

    run_count = len(os.listdir(outpath)) + 1

    os.makedirs(outpath, exist_ok=True)
    
    base_count = len(os.listdir(outpath))
    
    start_code_a = None
    start_code_b = None
    
    start_code_a = torch.randn([opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)
    start_code_b = torch.randn([opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)

    # If more than one prompt we only interpolate the text conditioning
    if not single_prompt:
        start_code_b = start_code_a
        
    precision_scope = autocast if opt.precision=="autocast" else nullcontext
    audio_intensity = 0

    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                tic = time.time()
                for n in trange(opt.n_iter, desc="Sampling"):
                    for data_a,data_b in zip(datas,datas[1:]):          
                        for t in np.linspace(0, 1, opt.num_interpolation_steps):
                            #print("data_a",data_a)

                            data = [slerp(float(t), data_a[0], data_b[0])]
                            #audio_intensity = (audio_intensity * opt.audio_smoothing) + (opt.audio_keyframes[base_count] * (1 - opt.audio_smoothing))
                            
                            # calculate interpolation for init noise. this only applies if we have only on text prompt
                            # otherwise noise stays constant for now (due to start_code_a == start_code_b)
                            
                            t_max = min((0.5, opt.num_interpolation_steps / 10))
                            noise_t = t * t_max                         
                    
                            start_code = slerp(float(noise_t), start_code_a, start_code_b) #slerp(audio_intensity, start_code_a, start_code_b)
                            for c in tqdm(data, desc="data"):
                                diffuse(base_count, start_code, c, batch_size, opt, model, model_wrap, outpath, device)
                                base_count += 1



                toc = time.time()

    print(f"Your samples have been saved to: \n{outpath} \n"
          f" \nEnjoy.")




class WidgetDict2(OrderedDict):
    def __getattr__(self,val):
        return self[val]


def get_default_options():
    options = WidgetDict2()
    options['outdir'] ="/outputs"
    options['sampler'] = "euler"
    options['skip_save'] = False
    options['ddim_steps'] = 50
    options['steps'] = 15
    options['plms'] = True
    options['laion400m'] = False
    options['ddim_eta'] = 0.0
    options['n_iter'] = 1
    options['C'] = 4
    options['f'] = 8
    options['n_samples'] = 1
    options['n_rows'] = 0
    options['from_file'] = None
    options['config'] = "configs/stable-diffusion/v1-inference.yaml"
    options['ckpt'] ="/stable-diffusion-checkpoints/sd-v1-4.ckpt"
    options['precision'] = "full"  # or "full" "autocast"
    options['use_init'] = True
    # Extra option for the notebook
    options['display_inline'] = False
    options['audio_smoothing'] = 0.7
    return options
