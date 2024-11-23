import argparse, os, sys, glob
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import cv2
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext
import shutil
import logging
import torchvision.utils as tvu
print(sys.path)

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.models.diffusion.dpm_solver import DPMSolverSampler
from ldm.models.diffusion.uni_pc import UniPCSampler
from ldm.models.diffusion.dpm_solver_v3 import DPMSolverv3Sampler
from uni_sampler.uni_sampler import Uni_Sampler
from evaluate.fid_score import calculate_fid_given_paths
def get_logger(args):
    log_format = "%(asctime)s %(message)s"
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format=log_format,
        datefmt='%m/%d %I:%M:%S %p')
    logger = logging.getLogger()
    if not os.path.exists(args.exp):
        os.makedirs(args.exp)
    file_handler = logging.FileHandler(os.path.join(args.exp, "log"))
    file_handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(file_handler)
    logger.info("Conducting Command: %s", " ".join(sys.argv))
    return logger

def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


def load_replacement(x):
    try:
        hwc = x.shape
        y = Image.open("assets/rick.jpeg").convert("RGB").resize((hwc[1], hwc[0]))
        y = (np.array(y) / 255.0).astype(x.dtype)
        assert y.shape == x.shape
        return y
    except Exception:
        return x


def main():
    parser = argparse.ArgumentParser()

    # for fid evaluation and experiment setting
    parser.add_argument(
        "--gpu",
        type=str,
        default="0",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--eval_config",
        type=str,
        help="config about FID evaluation",
        default="eval_config.yml",
    )
    parser.add_argument(
        "--exp",
        type=str,
        default="./exp/default",
    )
    # for sampling
    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="",
        help="the prompt to render",
    )
    parser.add_argument(
        "--outdir", type=str, nargs="?", help="dir to write results to", default="outputs/txt2img-samples"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="number of sampling steps",
    )
    parser.add_argument("--method", default="unisampler", choices=["ddim", "plms", "dpm_solver++", "uni_pc", "dpm_solver_v3","unisampler"])
    # parser.add_argument(
    #     "--fixed_noise",
    #     action="store_true",
    #     help="if enabled, uses the same starting code across samples ",
    # )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling)",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=2,
        help="sample this often",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1000,
        help="how many samples to produce for each given prompt. A.k.a. batch size",
    )
    parser.add_argument(
        "--n_rows",
        type=int,
        default=0,
        help="rows in the grid (default: n_samples)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=7.5,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--from-file",
        type=str,
        help="if specified, load prompts from this file",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument("--statistics_dir", type=str, default="statistics/lsun_beds256/120_1024", help="Statistics path for DPM-Solver-v3.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/latent-diffusion/lsun_bedrooms-ldm-vq-4.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="models/ldm/lsun_beds256/model.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=256,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=256,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=3,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=4,
        help="downsampling factor",
    )
    parser.add_argument(
        "--precision", type=str, help="evaluate at this precision", choices=["full", "autocast"], default="autocast"
    )
    
    # for dpm-solver sampler
    parser.add_argument(
        "--dpm_solver_sampler_type", type=str, default="dpmsolver++", help="prediction type of dpm-solver"
    )
    parser.add_argument(
        "--dpm_solver_order", type=int, default=3, help="order of dpm-solver"
    )
    parser.add_argument(
        "--dpm_solver_atol", type=float, default=0.0078, help="atol for adaptive step size algorithm"
    )
    parser.add_argument(
        "--dpm_solver_rtol", type=float, default=0.05, help="rtol for adaptive step size algorithm"
    )
    parser.add_argument(
        "--dpm_solver_method",
        type=str,
        default="singlestep",
        help="method of dpm_solver ('adaptive' or 'singlestep' or 'multistep' or 'singlestep_fixed')",
    )
    parser.add_argument(
        "--dpm_solver_type",
        type=str,
        default="dpmsolver",
        help="type of dpm_solver ('dpmsolver' or 'taylor')",
    )
    
    # for uni-pc sampler
    parser.add_argument(
        "--uni_pc_order", type=int, default=3, help="order of uni-pc"
    )
    parser.add_argument(
        "--uni_pc_prediction_type", type=str, default="data_prediction", help="prediction type of uni-pc"
    )
    parser.add_argument(
        "--uni_pc_variant", type=str, default='bh1', help="B(h) of uni-pc"
    )
    parser.add_argument(
        "--uni_pc_disable_corrector", action="store_true",
    )
    
    # for Uni-Sampler
    parser.add_argument(
        "--uni_sampler_decision_type", type=str, default=None, help="type of decision for uni-sampler, can be chosen from ['dpmsolver', 'unipc' , 'from_search_space']"
    )
    parser.add_argument(
        "--uni_sampler_method", type=str, default="multistep", help="singlestep or multistep, only works if the decision is not got in advance"
    )
    parser.add_argument(
        "--uni_sampler_order", type=int, default=3, help="the taylor order of uni-sampler, only works if the decision is not got in advance"
    )
    parser.add_argument(
        "--t_start", type=float, default=1.0, help="t_T"
    )
    parser.add_argument(
        "--t_end", type=float, default=0.001, help="t_0"
    )
    parser.add_argument(
        "--load_decision", type=str, default=None, help="load a decision"
    )
    parser.add_argument(
        "--skip_type",
        type=str,
        default="time_uniform",
        help="skip according to ('uniform' or 'quadratic' for DDIM/DDPM; 'logSNR' or 'time_uniform' or 'time_quadratic' for DPM-Solver)",
    )
    parser.add_argument("--dpmsolver_v3_order", type=int, default=2, help="Order of DPM Solver v3.")
    parser.add_argument("--p_pseudo", action="store_true", help="Use P-pseudo if set.")
    parser.add_argument("--use_corrector", action="store_true", help="Use corrector if set.")
    parser.add_argument("--c_pseudo", action="store_true", help="Use C-pseudo if set.")
    parser.add_argument("--degenerated", action="store_true", help="Use degenerated mode if set.")
    parser.add_argument("--dpmsolver_v3_t_start", type=float, default=1, help="Start time for DPM Solver v3 EMS.")
    parser.add_argument("--dpmsolver_v3_t_end", type=float, default=1e-4, help="End time for DPM Solver v3 EMS.")
    parser.add_argument("--number_of_samples", type=int, default=None, help="Number of samples to generate.")
    parser.add_argument("--afs", type=str, default="no_afs", help="AFS method to use. support [no_afs, zero_x0,average_x0]")
    parser.add_argument("--denoise", action="store_true", default=False)
    parser.add_argument("--lower_order_final", action="store_true", default=False)
    parser.add_argument("--thresholding", action="store_true", default=False)
    parser.add_argument("--return_intermediate", action="store_true")
    parser.add_argument("--baseline",type=str,default="dpmsolver_v3",choices=["dpmsolver","dpmsolver++","unipc","dpmsolver_v3"])
    opt = parser.parse_args()

    seed_everything(opt.seed)

    # prepare logger
    logging = get_logger(opt)
    if opt.gpu and torch.cuda.is_available():
        opt.gpu_flag = True
        # device = torch.device('cuda')
        gpus = [int(d) for d in opt.gpu.split(',')]
        opt.gpu = gpus
        device = torch.device(f"cuda:{gpus[0]}")
        torch.cuda.set_device(gpus[0]) # currently only training & inference on single card is supported.
        logging.info("Using GPU(s). Available gpu count: {}".format(torch.cuda.device_count()))
    else:
        device = torch.device('cpu')
        logging.info("Using cpu!")
    config = OmegaConf.load(f"{opt.config}")
    
    model = load_model_from_config(config, f"{opt.ckpt}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    samplers = {"ddim": DDIMSampler, "plms": PLMSSampler, "dpm_solver++": DPMSolverSampler, "uni_pc": UniPCSampler}

    if opt.method in samplers.keys():
        sampler = samplers[opt.method](model)
    elif opt.method == "dpm_solver_v3":
        sampler = DPMSolverv3Sampler(opt.ckpt, opt.statistics_dir, model, steps=opt.steps, guidance_scale=opt.scale)
    elif opt.method == "unisampler":
        sampler = Uni_Sampler(
            model=model,
            statistics_dir=opt.statistics_dir,
            guidance_scale=opt.scale,
            args=opt
        )
    else:
        raise ValueError(f"Unsupported sampling method {opt.method}")

    os.makedirs(opt.exp, exist_ok=True)

    batch_size = config["sampling"]["batch_size"]
    
    sample_path = os.path.join(opt.exp, "samples")
    os.makedirs(sample_path, exist_ok=True)

    if config["fixed_noise"]["enable"]:
        fixed_noise = torch.load(config["fixed_noise"]["fixed_noise_path"]).to(device)
    def eval_decision(opt):
        if os.path.exists(sample_path):
            shutil.rmtree(sample_path)
            os.makedirs(sample_path,exist_ok=True)
        else:
            os.makedirs(sample_path,exist_ok=True)
        img_id = 0
        noise_idx = 0
        precision_scope = autocast if opt.precision == "autocast" else nullcontext
        with torch.no_grad():
            with precision_scope("cuda"):
                with model.ema_scope():
                    while img_id < opt.n_samples:
                        uc = None
                        c = None
                        shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                        n = min(batch_size, opt.n_samples - img_id)
                        if config["fixed_noise"]["enable"]:
                            start_code = fixed_noise[noise_idx:noise_idx+n].to(device)
                            noise_idx += n
                        else:
                            start_code = torch.randn([n, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)
                        if opt.method == "dpm_solver_v3":
                            # batch_size, shape, conditioning, x_T, unconditional_conditioning
                            samples, _ = sampler.sample(
                                conditioning=c,
                                batch_size=n,
                                shape=shape,
                                unconditional_conditioning=uc,
                                x_T=start_code,
                                use_corrector=opt.scale < 5.0,
                            )
                        elif opt.method == "unisampler":
                            decisions = sampler.get_empirical_decisions(opt)
                            samples, _ = sampler.sample(
                                conditioning=c,
                                batch_size=n,
                                shape=shape,
                                unconditional_conditioning=uc,
                                x_T=start_code,
                                decisions=decisions
                            )
                        else:
                            samples, _ = sampler.sample(
                                S=opt.steps,
                                conditioning=c,
                                batch_size=n,
                                shape=shape,
                                verbose=False,
                                unconditional_guidance_scale=opt.scale,
                                unconditional_conditioning=uc,
                                eta=opt.ddim_eta,
                                x_T=start_code,
                            )

                        x_samples = model.decode_first_stage(samples)
                        x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
                        for i in range(x_samples.size(0)):
                            path = os.path.join(sample_path, f"{img_id}.png")
                            tvu.save_image(x_samples[i],path)
                            img_id += 1
                        print(f"Already generate {len(os.listdir(sample_path))} samples")
        print("Begin to compute FID...")
        fid = calculate_fid_given_paths(
            (None, sample_path), 
            batch_size=config["sampling"]["fid_batch_size"], 
            device=device, 
            dims=2048, 
            num_workers=8, 
            load_act=[config["sampling"]["fid_stats_dir"], None],
            save_act=[config["sampling"]["fid_stats_dir"], None],
        )
        logging.info("FID: {}".format(fid))
        if not config["sampling"]["keep_samples"]:
            print("Delete all samples...")
            shutil.rmtree(sample_path)
            os.makedirs(sample_path,exist_ok=True)
        decisions = sampler.get_empirical_decisions(opt)
        return decisions, fid
    
    opt.uni_sampler_decision_type = opt.baseline
    output_dir = "./search_data/initialize"
    skip_type = "time_uniform"
    for step in range(3,13):
        for afs in ["no_afs","zero_x0"]:
            if opt.baseline in ["dpmsolver","dpmsolver++"]:
                for dpmsolver_order in [2,3]:
                    opt.dpm_solver_type = "dpmsolver"
                    opt.skip_type = skip_type
                    opt.dpm_solver_method = "multistep"
                    opt.dpm_solver_order = dpmsolver_order
                    opt.steps = step
                    opt.afs = afs
                    opt.lower_order_final = True
                    # 生成前缀
                    prefix = f"{opt.baseline}_{step}_{afs}_{dpmsolver_order}"
                    # 检查是否有以 prefix 开头的文件
                    if any(filename.startswith(prefix) for filename in os.listdir(output_dir)):
                        print(f"已有文件以 {prefix} 开头，跳过此次 fid 评估。")
                        continue
                    decision, fid = eval_decision(opt)
                    torch.save([decision, fid], os.path.join(output_dir, f"{opt.baseline}_{step}_{afs}_{dpmsolver_order}_{fid}.pth"))
            elif opt.baseline == "unipc":
                for uni_pc_prediction_type in ["noise_prediction","data_prediction"]:
                    for uni_pc_variant in ["bh1","bh2"]:
                        # 生成前缀
                        prefix = f"{opt.baseline}_{step}_{afs}_{uni_pc_prediction_type}_{uni_pc_variant}"
                        # 检查是否有以 prefix 开头的文件
                        if any(filename.startswith(prefix) for filename in os.listdir(output_dir)):
                            print(f"已有文件以 {prefix} 开头，跳过此次 fid 评估。")
                            continue
                        opt.uni_pc_prediction_type=uni_pc_prediction_type
                        opt.uni_pc_disable_corrector=False
                        opt.skip_type=skip_type
                        opt.uni_pc_order=2
                        opt.steps=step
                        opt.afs=afs
                        opt.lower_order_final = True
                        opt.uni_pc_variant = uni_pc_variant
                        decision, fid = eval_decision(opt)
                        torch.save([decision, fid], os.path.join(output_dir, f"{opt.baseline}_{step}_{afs}_{uni_pc_prediction_type}_{uni_pc_variant}_{fid}.pth"))
            elif opt.baseline == "dpmsolver_v3":
                for p_pseudo in [True, False]:
                    for c_pseudo in [True, False]:
                        # 生成前缀
                        prefix = f"{opt.baseline}_{step}_{afs}_p{p_pseudo}_c{c_pseudo}"
                        # 检查是否有以 prefix 开头的文件
                        if any(filename.startswith(prefix) for filename in os.listdir(output_dir)):
                            print(f"已有文件以 {prefix} 开头，跳过此次 fid 评估。")
                            continue
                        opt.use_corrector = False # set by v3 codebase
                        opt.skip_type=skip_type
                        opt.c_pseudo = c_pseudo
                        opt.p_pseudo = p_pseudo
                        opt.afs=afs
                        opt.lower_order_final = True
                        opt.dpmsolver_v3_order = 2 # set by v3 codebase
                        opt.steps=step
                        decision, fid = eval_decision(opt)
                        torch.save([decision, fid], os.path.join(output_dir, f"{opt.baseline}_{step}_{afs}_p{p_pseudo}_c{c_pseudo}_{fid}.pth"))
            else:
                raise NotImplementedError(f"don't support baseline method {opt.baseline}")
    
     
if __name__ == "__main__":
    main()