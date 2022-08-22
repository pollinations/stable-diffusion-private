import sys
sys.path.append("/CLIP")
sys.path.append("/taming-transformers")
# Code to turn kwargs into Jupyter widgets
from collections import OrderedDict

# Slightly modified version of: https://github.com/CompVis/stable-diffusion/blob/main/scripts/txt2img.py
import os, sys 
import torch    
import numpy as np    
from omegaconf import OmegaConf    
from PIL import Image    
#from tqdm.auto import tqdm, trange  # NOTE: updated for notebook
from tqdm import tqdm, trange  # NOTE: updated for notebook
from einops import rearrange     
import time    
from pytorch_lightning import seed_everything    
from torch import autocast    
from contextlib import contextmanager, nullcontext    
    
from ldm.util import instantiate_from_config    
from ldm.models.diffusion.ddim import DDIMSampler    
from ldm.models.diffusion.plms import PLMSSampler
from scripts.txt2img import chunk, load_model_from_config



from cog import BasePredictor, Input, Path
from omegaconf import OmegaConf
import torch

# @lru_cache(maxsize=None)  # cache the model, so we don't have to load it every time
# def load_clip(clip_model="ViT-L/14", use_jit=True, device="cpu"):
#     clip_model, preprocess = clip.load(clip_model, device=device, jit=use_jit)
#     return clip_model, preprocess


class Predictor(BasePredictor):


    @torch.inference_mode()
    def setup(self):
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        options = get_default_options()
        self.options = options

        self.model = load_model(self.options, self.device)




    @torch.inference_mode()
    def predict(
        self,
        prompts: str = Input(
            default="Lust by magritte\nEnvy by magritte",
            description="model will try to generate this text.",
        ),
        prompt_scale: float = Input(
            default=5.0,
            description="Determines influence of your prompt on generation.",
        ),
        num_frames_per_prompt: int = Input(
            default=10,
            description="Number of frames to generate per prompt.",
        ),
        random_seed: int = Input(
            default=42,
            description="Each seed generates a different image",
        ),
    ) -> Path:
        
        options = self.options
        options['prompts'] = prompts.split("\n")
        options['num_interpolation_steps'] = num_frames_per_prompt
        options['scale'] = prompt_scale
        options['seed'] = random_seed

        run_inference(options, self.model, self.device)
                
        encoding_options = "-c:v libx264 -crf 20 -preset slow -vf format=yuv420p -c:a aac -movflags +faststart"
        os.system("ls -l /outputs")
        os.system(f'ffmpeg -y -r 5 -i {options["outdir"]}/%*.png {encoding_options} /outputs/z_interpollation.mp4')
        return Path("/outputs/z_interpollation.mp4")

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

def diffuse(count_start, start_code, c, batch_size, opt, model, sampler,  outpath):
    #print("diffusing with batch size", batch_size)
    uc = None
    if opt.scale != 1.0:
        uc = model.get_learned_conditioning(batch_size * [""])
    shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
    samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
                                    conditioning=c,
                                    batch_size=batch_size,
                                    shape=shape,
                                    verbose=False,
                                    unconditional_guidance_scale=opt.scale,
                                    unconditional_conditioning=uc,
                                    eta=opt.ddim_eta,
                                    x_T=start_code)
    print("samples_ddim", samples_ddim.shape)
    x_samples_ddim = model.decode_first_stage(samples_ddim)
    x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
    if not opt.skip_save:
        count = count_start
        for x_sample in x_samples_ddim:
            x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
            Image.fromarray(x_sample.astype(np.uint8)).save(
                os.path.join(outpath, f"{count:05}.png"))
            count += 1
    



def run_inference(opt, model, device):
    """Seperates the loading of the model from the inference
    
    Additionally, slightly modified to display generated images inline
    """
    seed_everything(opt.seed)

    if opt.plms:
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    outpath = opt.outdir
    os.makedirs(outpath, exist_ok=True)

    batch_size = opt.n_samples

    
    prompts = opt.prompts

    print("embedding prompts")
    cs = [model.get_learned_conditioning(prompt) for prompt in prompts]

    datas = [[batch_size * c] for c in cs]

    run_count = len(os.listdir(outpath)) + 1

    os.makedirs(outpath, exist_ok=True)
    
    base_count = len(os.listdir(outpath))
    
    start_code_a = None
    start_code_b = None
    if opt.fixed_code:
        start_code_a = torch.randn([opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)
        start_code_b = torch.randn([opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)

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
                            start_code = start_code_a #slerp(audio_intensity, start_code_a, start_code_b)
                            for c in tqdm(data, desc="data"):
                                diffuse(base_count, start_code, c, batch_size, opt, model, sampler, outpath)
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
    options['skip_save'] = False
    options['ddim_steps'] = 50
    options['plms'] = True
    options['laion400m'] = False
    options['fixed_code'] = True
    options['ddim_eta'] = 0.0
    options['n_iter'] = 1
    options['H'] = 512
    options['W'] = 512
    options['C'] = 4
    options['f'] = 8
    options['n_samples'] = 1
    options['n_rows'] = 0
    options['from_file'] = None
    options['config'] = "configs/stable-diffusion/v1-inference.yaml"
    options['ckpt'] ="/stable-diffusion-checkpoints/sd-v1-3-full-ema.ckpt"
    options['precision'] = "full"  # or "full" "autocast"
    # Extra option for the notebook
    options['display_inline'] = False
    options['audio_smoothing'] = 0.7
    return options
