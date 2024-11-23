import os
import shutil
import logging
import time
import glob
import yaml
from tkinter import E

import blobfile as bf

import numpy as np
import tqdm
import torch
import torch.utils.data as data

from models.ema import EMAHelper
from models.diffusion import Model
from functions import get_optimizer
from functions.losses import loss_registry
from datasets import get_dataset, data_transform, inverse_data_transform
from functions.ckpt_util import load_model_for_sample
from evaluate.fid_score import calculate_fid_given_paths
# from evaluate.fid_score_reduced import calculate_fid_given_paths_reduced
import torchvision.utils as tvu

def picture_puzzle(x, p=4):
    assert x.shape[0]>=p*p
    h = x.shape[2]
    w = x.shape[3]
    puzzled = torch.zeros([x.shape[1], h*p, w*p])
    for i in range(p):
        for j in range(p):
            puzzled[:,i*h:(i+1)*h, j*w:(j+1)*w] = x[j+i*p,:,:,:] 
    return puzzled


def load_data_for_worker(base_samples, batch_size, cond_class):
    with bf.BlobFile(base_samples, "rb") as f:
        obj = np.load(f)
        image_arr = obj["arr_0"]
        if cond_class:
            label_arr = obj["arr_1"]
    buffer = []
    label_buffer = []
    while True:
        for i in range(len(image_arr)):
            buffer.append(image_arr[i])
            if cond_class:
                label_buffer.append(label_arr[i])
            if len(buffer) == batch_size:
                batch = torch.from_numpy(np.stack(buffer)).float()
                batch = batch / 127.5 - 1.0
                batch = batch.permute(0, 3, 1, 2)
                res = dict(low_res=batch)
                if cond_class:
                    res["y"] = torch.from_numpy(np.stack(label_buffer))
                yield res
                buffer, label_buffer = [], []


def torch2hwcuint8(x, clip=False):
    if clip:
        x = torch.clamp(x, -1, 1)
    x = (x + 1.0) / 2.0
    return x


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].
    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == 'cosine':
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: np.cos((t + 0.008) / 1.008 * np.pi / 2) ** 2,
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


class Diffusion(object):
    def __init__(self, args, config, logger):
        self.args = args
        self.config = config
        device = (
            torch.device("cuda")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        self.device = device

        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0
        )
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        if self.model_var_type == "fixedlarge":
            self.logvar = betas.log()
            # torch.cat(
            # [posterior_variance[1:2], betas[1:]], dim=0).log()
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log()
            
        self.logger = logger
        self.model = None
        self.classifier = None
        self.img_id = 0
        # possible samplers
        self.dpm_solver = None
        self.uni_pc = None
        self.dpm_solver_v3 = None
        self.uni_sampler = None
        # decisions
        self.decisions = None
        self.NFE = self.compute_nfe()
        

    def train(self):
        args, config = self.args, self.config
        dataset, test_dataset = get_dataset(args, config)
        train_loader = data.DataLoader(
            dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
        )
        model = Model(config)

        model = model.to(self.device)
        model = torch.nn.DataParallel(model)

        optimizer = get_optimizer(self.config, model.parameters())

        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(model)
        else:
            ema_helper = None

        start_epoch, step = 0, 0
        if self.args.resume_training:
            states = torch.load(os.path.join(self.args.log_path, "ckpt.pth"))
            model.load_state_dict(states[0])

            states[1]["param_groups"][0]["eps"] = self.config.optim.eps
            optimizer.load_state_dict(states[1])
            start_epoch = states[2]
            step = states[3]
            if self.config.model.ema:
                ema_helper.load_state_dict(states[4])

        for epoch in range(start_epoch, self.config.training.n_epochs):
            data_start = time.time()
            data_time = 0
            for i, (x, y) in enumerate(train_loader):
                n = x.size(0)
                data_time += time.time() - data_start
                model.train()
                step += 1

                x = x.to(self.device)
                x = data_transform(self.config, x)
                e = torch.randn_like(x)
                b = self.betas

                # antithetic sampling
                t = torch.randint(
                    low=0, high=self.num_timesteps, size=(n // 2 + 1,)
                ).to(self.device)
                t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]
                loss = loss_registry[config.model.type](model, x, t, e, b)

                logging.info(
                    f"step: {step}, loss: {loss.item()}, data time: {data_time / (i+1)}"
                )

                optimizer.zero_grad()
                loss.backward()

                try:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.optim.grad_clip
                    )
                except Exception:
                    pass
                optimizer.step()

                if self.config.model.ema:
                    ema_helper.update(model)

                if step % self.config.training.snapshot_freq == 0 or step == 1:
                    states = [
                        model.state_dict(),
                        optimizer.state_dict(),
                        epoch,
                        step,
                    ]
                    if self.config.model.ema:
                        states.append(ema_helper.state_dict())

                    torch.save(
                        states,
                        os.path.join(self.args.log_path, "ckpt_{}.pth".format(step)),
                    )
                    torch.save(states, os.path.join(self.args.log_path, "ckpt.pth"))

                data_start = time.time()
    
    def sample(self, number_of_samples=None, decisions=None):
        
        if self.model is None or self.classifier is None:
            self.model, self.classifier = load_model_for_sample(self.config, self.device, self.args.gpu)

        if self.args.fid:
            if number_of_samples is None:
                number_of_samples = self.args.number_of_samples
            
            self.sample_fid(self.model, classifier=self.classifier, total_n_samples=number_of_samples, decisions=decisions)
            
            print("Begin to compute FID...")
            fid = calculate_fid_given_paths(
                (self.config.sampling.fid_image_dir, self.args.image_folder), 
                batch_size=self.config.sampling.fid_batch_size, 
                device=self.device, 
                dims=2048, 
                num_workers=8, 
                load_act=[self.config.sampling.fid_stats_dir, None],
                save_act=[self.config.sampling.fid_stats_dir, None],
            )
            self.logger.info("FID: {}".format(fid))
            if decisions is not None:
                nfe = self.compute_nfe(decisions)
            else:
                nfe = self.NFE
            self.logger.info("NFE: {}".format(nfe))
            if not self.config.sampling.keep_samples:
                print("Delete all samples...")
                shutil.rmtree(self.args.image_folder)
            return fid
        # elif self.args.interpolation:
        #     self.sample_interpolation(model)
        # elif self.args.sequence:
        #     self.sample_sequence(model)
        else:
            raise NotImplementedError("Sample procedeure not defined")

    def sample_different_sampler_nfe(self):

        nfe_list = [4, 5, 6, 7, 8, 9, 10]
        # nfe_list = [10, 20]
        sampler_list = [
            # {"sample_type":"dpmsolver++", "skip_type":"logSNR", "method":"multistep", "dpm_solver_order": 2}, 
            # {"sample_type":"dpmsolver++", "skip_type":"logSNR", "method":"multistep", "dpm_solver_order": 3}, 
            {"sample_type":"dpmsolver", "skip_type":"logSNR", "method":"singlestep", "dpm_solver_order": 2}, 
            # {"sample_type":"dpmsolver", "skip_type":"logSNR", "method":"singlestep", "dpm_solver_order": 3}, 
            # {"sample_type":"unipc", "uni_pc_variant":"bh1", "skip_type": "logSNR", "uni_pc_prediction_type":"data_prediction", "uni_pc_order": 2}, 
            # {"sample_type":"unipc", "uni_pc_variant":"bh2", "skip_type": "logSNR", "uni_pc_prediction_type":"data_prediction", "uni_pc_order": 2}, 
            # {"sample_type":"unipc", "uni_pc_variant":"vary_coeff", "skip_type": "logSNR", "uni_pc_prediction_type":"data_prediction", "uni_pc_order": 3}, 
            # {"sample_type":"dpmsolver", "skip_type":"logSNR", "method":"singlestep"}, 
            # {"sample_type":"generalized", "skip_type":"quad"}, 
        ]
        result = []
        
        model, classifier = load_model_for_sample(self.config, self.device, self.args.gpu)
        
        for index, sampler_type in enumerate(sampler_list):
            result.append({})
            
            self.args.sample_type = sampler_type.get("sample_type")
            self.args.skip_type = sampler_type.get("skip_type")
            self.args.dpm_solver_method = sampler_type.get("method", "singlestep")
            self.args.dpm_solver_order = sampler_type.get("dpm_solver_order", 2)
        
            self.args.uni_pc_prediction_type = sampler_type.get("uni_pc_prediction_type", "data_prediction")
            self.args.uni_pc_order = sampler_type.get("uni_pc_order", 2)
            self.args.uni_pc_variant = sampler_type.get("uni_pc_variant", "vary_coeff")
                
            for nfe in nfe_list:
                self.args.timesteps = nfe
                self.sample_fid(model, classifier=classifier)
                print("Begin to compute FID...")
                fid = calculate_fid_given_paths(
                    (self.config.sampling.fid_image_dir, self.args.image_folder), 
                    batch_size=self.config.sampling.fid_batch_size, 
                    device=self.device, 
                    dims=2048, 
                    num_workers=8, 
                    load_act=[self.config.sampling.fid_stats_dir, None],
                    save_act=[self.config.sampling.fid_stats_dir, None],
                )
                self.logger.info(f"Sampler type: {sampler_type} | NFE: {nfe} | FID: {fid}")
                result[index][nfe] = fid
                print("Delete all samples...")
                shutil.rmtree(self.args.image_folder)
        self.logger.info(sampler_list)
        self.logger.info(result)
        torch.save({"samper_list":sampler_list, "result":result}, os.path.join(self.args.exp, "result.pth"))

    def sample_fid(self, model, classifier=None, total_n_samples=None, decisions=None):
        config = self.config
        if total_n_samples is None:
            total_n_samples = config.sampling.fid_total_samples
        if os.path.exists(self.args.image_folder) and not config.sampling.continuous:
            shutil.rmtree(self.args.image_folder)
        if not os.path.exists(self.args.image_folder):
            os.makedirs(self.args.image_folder)
        if len(glob.glob(f"{self.args.image_folder}/*.png")) == total_n_samples:
            return
        self.img_id = len(os.listdir(self.args.image_folder))

        if self.config.model.is_upsampling:
            base_samples_total = load_data_for_worker(self.args.base_samples, config.sampling.batch_size, config.sampling.cond_class)
        
        # use fixed noise for evaluation to reduce randomness
        if self.config.fixed_noise.enable:
            if getattr(self, "fixed_noise", None) is None:
                print(f"Load noise from {self.config.fixed_noise.path}")
                self.fixed_noise = torch.load(self.config.fixed_noise.path)
            assert total_n_samples<=len(self.fixed_noise), "The length of noise must larger than the number of samples"
            noise_idx = self.img_id
            
            # use fixed classes for evaluation to reduce randomness
            if classifier is not None:
                if self.config.fixed_classes.enable:
                    if getattr(self, "fixed_classes", None) is None:
                        print(f"Load class lables from {self.config.fixed_classes.path}")
                        self.fixed_classes = torch.load(self.config.fixed_classes.path)
                    assert total_n_samples<=len(self.fixed_classes), "The length of noise must larger than the number of samples"

        with torch.no_grad():
            while(self.img_id<total_n_samples):

                n = min(config.sampling.batch_size, total_n_samples-self.img_id)
                if self.config.fixed_noise.enable:
                    x = self.fixed_noise[noise_idx:noise_idx+n].to(self.device)
                    if classifier is not None:
                        if self.config.fixed_classes.enable:
                            classes = self.fixed_classes[noise_idx:noise_idx+n]
                        else:
                            classes = torch.randint(low=0, high=self.config.data.num_classes, size=(n,))
                    else:
                        classes = None
                    noise_idx += n
                else:
                    x = torch.randn(
                        n,
                        config.data.channels,
                        config.data.image_size,
                        config.data.image_size,
                        device=self.device,
                    )
                    if classifier is not None:
                        classes = torch.randint(low=0, high=self.config.data.num_classes, size=(n,))
                    else:
                        classes = None

                if self.config.model.is_upsampling:
                    base_samples = next(base_samples_total)
                else:
                    base_samples = None

                # import ipdb; ipdb.set_trace()
                x, classes = self.sample_image(x, model, classifier=classifier, base_samples=base_samples, decisions=decisions, guided_classes=classes)
                # end.record()
                # torch.cuda.synchronize()
                # t_list.append(start.elapsed_time(end))
                x = inverse_data_transform(config, x)
                # puzzled_x = picture_puzzle(x, p=8)
                # tvu.save_image(puzzled_x, os.path.join(self.args.image_folder, f"puzzled.png"))
                # import ipdb; ipdb.set_trace()
                self.save_samples(x, classes)
        # # Remove the time evaluation of the first batch, because it contains extra initializations
        # print('time / batch', np.mean(t_list[1:]) / 1000., 'std', np.std(t_list[1:]) / 1000.)

    def sample_sequence(self, model, classifier=None):
        config = self.config

        x = torch.randn(
            8,
            config.data.channels,
            config.data.image_size,
            config.data.image_size,
            device=self.device,
        )

        # NOTE: This means that we are producing each predicted x0, not x_{t-1} at timestep t.
        with torch.no_grad():
            _, x = self.sample_image(x, model, last=False, classifier=classifier)

        x = [inverse_data_transform(config, y) for y in x]

        for i in range(len(x)):
            for j in range(x[i].size(0)):
                tvu.save_image(
                    x[i][j], os.path.join(self.args.image_folder, f"{j}_{i}.png")
                )

    def sample_interpolation(self, model):
        config = self.config

        def slerp(z1, z2, alpha):
            theta = torch.acos(torch.sum(z1 * z2) / (torch.norm(z1) * torch.norm(z2)))
            return (
                torch.sin((1 - alpha) * theta) / torch.sin(theta) * z1
                + torch.sin(alpha * theta) / torch.sin(theta) * z2
            )

        z1 = torch.randn(
            1,
            config.data.channels,
            config.data.image_size,
            config.data.image_size,
            device=self.device,
        )
        z2 = torch.randn(
            1,
            config.data.channels,
            config.data.image_size,
            config.data.image_size,
            device=self.device,
        )
        alpha = torch.arange(0.0, 1.01, 0.1).to(z1.device)
        z_ = []
        for i in range(alpha.size(0)):
            z_.append(slerp(z1, z2, alpha[i]))

        x = torch.cat(z_, dim=0)
        xs = []

        # Hard coded here, modify to your preferences
        with torch.no_grad():
            for i in range(0, x.size(0), 8):
                xs.append(self.sample_image(x[i : i + 8], model))
        x = inverse_data_transform(config, torch.cat(xs, dim=0))
        for i in range(x.size(0)):
            tvu.save_image(x[i], os.path.join(self.args.image_folder, f"{i}.png"))

    def sample_image(self, x, model, last=True, classifier=None, base_samples=None, decisions=None, guided_classes=None):
        assert last
        try:
            skip = self.args.skip
        except Exception:
            skip = 1

        classifier_scale = self.config.sampling.classifier_scale if self.args.scale is None else self.args.scale
        if guided_classes is None or self.args.fixed_class is not None:
            if self.config.sampling.cond_class:
                if self.args.fixed_class is None:
                    classes = torch.randint(low=0, high=self.config.data.num_classes, size=(x.shape[0],)).to(x.device)
                else:
                    classes = torch.randint(low=self.args.fixed_class, high=self.args.fixed_class + 1, size=(x.shape[0],)).to(x.device)
            else:
                classes = None
        else:
            assert classifier is not None
            classes = guided_classes.to(x.device)
        
        if base_samples is None:
            if classes is None:
                model_kwargs = {}
            else:
                model_kwargs = {"y": classes}
        else:
            model_kwargs = {"y": base_samples["y"], "low_res": base_samples["low_res"]}

        if self.args.sample_type == "generalized":
            if self.args.skip_type == "uniform":
                skip = self.num_timesteps // self.args.timesteps
                seq = range(0, self.num_timesteps, skip)
            elif self.args.skip_type == "quad":
                seq = (
                    np.linspace(
                        0, np.sqrt(self.num_timesteps * 0.8), self.args.timesteps
                    )
                    ** 2
                )
                seq = [int(s) for s in list(seq)]
            else:
                raise NotImplementedError
            from functions.denoising import generalized_steps
            def model_fn(x, t, **model_kwargs):
                print("Call the model...")
                out = model(x, t, **model_kwargs)
                if "out_channels" in self.config.model.__dict__.keys():
                    if self.config.model.out_channels == 6:
                        return torch.split(out, 3, dim=1)[0]
                return out
            xs, _ = generalized_steps(x, seq, model_fn, self.betas, eta=self.args.eta, classifier=classifier, is_cond_classifier=self.config.sampling.cond_class, classifier_scale=classifier_scale, **model_kwargs)
            x = xs[-1]
        elif self.args.sample_type == "ddpm_noisy":
            if self.args.skip_type == "uniform":
                skip = self.num_timesteps // self.args.timesteps
                seq = range(0, self.num_timesteps, skip)
            elif self.args.skip_type == "quad":
                seq = (
                    np.linspace(
                        0, np.sqrt(self.num_timesteps * 0.8), self.args.timesteps
                    )
                    ** 2
                )
                seq = [int(s) for s in list(seq)]
            else:
                raise NotImplementedError
            from functions.denoising import ddpm_steps
            def model_fn(x, t, **model_kwargs):
                out = model(x, t, **model_kwargs)
                if "out_channels" in self.config.model.__dict__.keys():
                    if self.config.model.out_channels == 6:
                        return torch.split(out, 3, dim=1)[0]
                return out
            xs, _ = ddpm_steps(x, seq, model_fn, self.betas, classifier=classifier, is_cond_classifier=self.config.sampling.cond_class, classifier_scale=classifier_scale, **model_kwargs)
            x = xs[-1]
        elif self.args.sample_type in ["dpmsolver", "dpmsolver++", "unipc"]:
            from dpm_solver.sampler import NoiseScheduleVP, model_wrapper
            def model_fn(x, t, **model_kwargs):
                print("Call the model...")
                out = model(x, t, **model_kwargs)
                # If the model outputs both 'mean' and 'variance' (such as improved-DDPM and guided-diffusion),
                # We only use the 'mean' output for DPM-Solver, because DPM-Solver is based on diffusion ODEs.
                if "out_channels" in self.config.model.__dict__.keys():
                    if self.config.model.out_channels == 6:
                        out = torch.split(out, 3, dim=1)[0]
                return out

            def classifier_fn(x, t, y, **classifier_kwargs):
                logits = classifier(x, t)
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                return log_probs[range(len(logits)), y.view(-1)]
            noise_schedule = NoiseScheduleVP(schedule=self.config.diffusion.beta_schedule, betas=self.betas)
            # noise_schedule = NoiseScheduleVP(schedule="discrete", betas=self.betas)
            model_fn_continuous = model_wrapper(
                model_fn,
                noise_schedule,
                time_input_type=self.config.sampling.time_input_type,
                model_type="noise",
                model_kwargs=model_kwargs,
                guidance_type="uncond" if classifier is None else "classifier",
                condition=model_kwargs["y"] if "y" in model_kwargs.keys() else None,
                guidance_scale=classifier_scale,
                classifier_fn=classifier_fn,
                classifier_kwargs={},
            )
            
            if self.config.sampling.adaptive_tend:
                t_end = 1e-3 if self.args.timesteps<15 else 1e-4
            else:
                t_end = None
            
            if self.args.sample_type in ["dpmsolver", "dpmsolver++"]:
                if self.dpm_solver is None:
                    from dpm_solver.sampler import DPM_Solver
                    self.dpm_solver = DPM_Solver(
                        model_fn_continuous,
                        noise_schedule,
                        algorithm_type=self.args.sample_type,
                        correcting_x0_fn="dynamic_thresholding" if self.args.thresholding else None,
                    )
                else:
                    self.dpm_solver.model = lambda x, t: model_fn_continuous(x, t.expand((x.shape[0])))
            
                x = self.dpm_solver.sample(
                    x,
                    steps=(self.args.timesteps - 1 if self.args.denoise else self.args.timesteps),
                    order=self.args.dpm_solver_order,
                    skip_type=self.args.skip_type,
                    method=self.args.dpm_solver_method,
                    lower_order_final=self.args.lower_order_final,
                    denoise_to_zero=self.args.denoise,
                    solver_type=self.args.dpm_solver_type,
                    atol=self.args.dpm_solver_atol,
                    rtol=self.args.dpm_solver_rtol,
                    t_end=t_end,
                    return_intermediate=self.args.return_intermediate,
                )

            elif self.args.sample_type in ["unipc"]:
                if self.uni_pc is None:
                    from unipc.uni_pc import UniPC
                    self.uni_pc = UniPC(
                        model_fn_continuous,
                        noise_schedule,
                        algorithm_type=self.args.uni_pc_prediction_type,
                        variant=self.args.uni_pc_variant,
                        correcting_x0_fn="dynamic_thresholding" if self.args.thresholding else None,
                    )
                else:
                    self.uni_pc.model = lambda x, t: model_fn_continuous(x, t.expand((x.shape[0])))

                x = self.uni_pc.sample(
                    x,
                    steps=(self.args.timesteps - 1 if self.args.denoise else self.args.timesteps),
                    order=self.args.uni_pc_order,
                    skip_type=self.args.skip_type,
                    method="multistep", # currently only "multistep" is supported
                    lower_order_final=self.args.lower_order_final,
                    disable_corrector=self.args.uni_pc_disable_corrector,
                    denoise_to_zero=self.args.denoise,
                    t_end=t_end,
                    return_intermediate=self.args.return_intermediate,
                )
        elif self.args.sample_type == "dpmsolver_v3":
            from dpm_solver_v3.sampler import NoiseScheduleVP, model_wrapper, DPM_Solver_v3
            def model_fn(x, t, **model_kwargs):
                print("Call the model...")
                out = model(x, t, **model_kwargs)
                # If the model outputs both 'mean' and 'variance' (such as improved-DDPM and guided-diffusion),
                # We only use the 'mean' output for DPM-Solver, because DPM-Solver is based on diffusion ODEs.
                if "out_channels" in self.config.model.__dict__.keys():
                    if self.config.model.out_channels == 6:
                        out = torch.split(out, 3, dim=1)[0]
                return out

            def classifier_fn(x, t, y, **classifier_kwargs):
                logits = classifier(x, t)
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                return log_probs[range(len(logits)), y.view(-1)]
            noise_schedule = NoiseScheduleVP(schedule=self.config.diffusion.beta_schedule, betas=self.betas)
            model_fn_continuous = model_wrapper(
                model_fn,
                noise_schedule,
                time_input_type=self.config.sampling.time_input_type,
                model_type="noise",
                model_kwargs=model_kwargs,
                guidance_type="uncond" if classifier is None else "classifier",
                condition=model_kwargs["y"] if "y" in model_kwargs.keys() else None,
                guidance_scale=classifier_scale,
                classifier_fn=classifier_fn,
                classifier_kwargs={},
            )
            if self.dpm_solver_v3 is None:
                self.dpm_solver_v3 = DPM_Solver_v3(
                    self.args.statistics_dir,
                    model_fn_continuous,
                    noise_schedule,
                    steps = (self.args.timesteps - 1 if self.args.denoise else self.args.timesteps),
                    skip_type=self.args.skip_type,
                    correcting_x0_fn="dynamic_thresholding" if self.args.thresholding else None,
                    degenerated=self.args.degenerated,
                    device=self.device,
                    dpmsolver_v3_t_start=self.args.dpmsolver_v3_t_start,
                    dpmsolver_v3_t_end=self.args.dpmsolver_v3_t_end,
                    t_end=self.args.t_end,
                )
            else:
                # self.model = lambda x, t: model_fn(x, t.expand((x.shape[0])))
                self.dpm_solver_v3.model = lambda x, t: model_fn_continuous(x, t.expand((x.shape[0])))
            
            x = self.dpm_solver_v3.sample(
                x,
                order=self.args.dpmsolver_v3_order,
                p_pseudo=self.args.p_pseudo,
                use_corrector=self.args.use_corrector,
                denoise_to_zero=self.args.denoise,
                c_pseudo=self.args.c_pseudo,
                lower_order_final=self.args.lower_order_final,
                return_intermediate=self.args.return_intermediate,
            )

        elif self.args.sample_type == "unisampler":
            from uni_sampler.uni_sampler import NoiseScheduleVP, model_wrapper
            def model_fn(x, t, **model_kwargs):
                # print("Call the model...")
                out = model(x, t, **model_kwargs)
                # If the model outputs both 'mean' and 'variance' (such as improved-DDPM and guided-diffusion),
                # We only use the 'mean' output for DPM-Solver, because DPM-Solver is based on diffusion ODEs.
                if "out_channels" in self.config.model.__dict__.keys():
                    if self.config.model.out_channels == 6:
                        out = torch.split(out, 3, dim=1)[0]
                return out

            def classifier_fn(x, t, y, **classifier_kwargs):
                logits = classifier(x, t)
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                return log_probs[range(len(logits)), y.view(-1)]
            noise_schedule = NoiseScheduleVP(schedule=self.config.diffusion.beta_schedule, betas=self.betas)
            # noise_schedule = NoiseScheduleVP(schedule="discrete", betas=self.betas)
            model_fn_continuous = model_wrapper(
                model_fn,
                noise_schedule,
                time_input_type=self.config.sampling.time_input_type,
                model_type="noise",
                model_kwargs=model_kwargs,
                guidance_type="uncond" if classifier is None else "classifier",
                condition=model_kwargs["y"] if "y" in model_kwargs.keys() else None,
                guidance_scale=classifier_scale,
                classifier_fn=classifier_fn,
                classifier_kwargs={},
            )
            if self.uni_sampler is None:
                from uni_sampler.uni_sampler import uni_sampler
                self.uni_sampler = uni_sampler(
                    model_fn_continuous,
                    noise_schedule,
                    correcting_x0_fn="dynamic_thresholding" if self.args.thresholding else None,
                    statistics_dir=self.args.statistics_dir,
                )
            else:
                self.uni_sampler.model = lambda x, t: model_fn_continuous(x, t.expand((x.shape[0])))
            
            # get decisions
            if decisions is None:
                from uni_sampler.utils import print_decisions
                if self.args.load_decision is not None:
                    self.decisions = torch.load(self.args.load_decision)
                    if "decisions" in self.decisions.keys():
                        self.decisions = self.decisions["decisions"]
                else: # no decision, no load decision, generate decision
                    from uni_sampler.utils import get_empirical_decisions
                    self.decisions = get_empirical_decisions(self.args, self.uni_sampler, self.device)
                    self.logger.info("using empirical decisions:")
                    print_decisions(self.logger, self.decisions)
                    torch.save(self.decisions, os.path.join(self.args.exp, f"{self.args.uni_sampler_decision_type}_decisions.pth"))
            else:
                self.decisions = decisions

            # compute NFE
            self.NFE = self.compute_nfe(decisions)
            # print(f"NFE: {self.NFE}")

            x = self.uni_sampler.sample(
                x,
                self.decisions,
                steps=(self.args.timesteps - 1 if self.args.denoise else self.args.timesteps),
                order=self.args.uni_sampler_order,
                skip_type=self.args.skip_type,
                method=self.args.uni_sampler_method,
                lower_order_final=self.args.lower_order_final,
                denoise_to_zero=self.args.denoise,
                return_intermediate=self.args.return_intermediate,
            )
        else:
            print(self.args.sample_type)
            raise NotImplementedError
        if self.args.return_intermediate:
            import ipdb; ipdb.set_trace()
        return x, classes

    def save_samples(self, x, classes=None, path=None):   
        if path is None:
            for i in range(x.size(0)):
                if classes is None:
                    path = os.path.join(self.args.image_folder, f"{self.img_id}.png")
                else:
                    path = os.path.join(self.args.image_folder, f"{self.img_id}_{int(classes.cpu()[i])}.png")
                tvu.save_image(x[i], path)
                self.img_id += 1
            print(f"Already generate {len(os.listdir(self.args.image_folder))} samples")
        else:
            for i in range(x.size(0)):
                if classes is None:
                    path = os.path.join(path, f"{self.img_id}.png")
                else:
                    path = os.path.join(path, f"{self.img_id}_{int(classes.cpu()[i])}.png")
                tvu.save_image(x[i], path)
                self.img_id += 1
            print(f"Already generate {len(os.listdir(path))} samples")
        
    
    def compute_nfe(self, decisions=None):
        decisions = self.decisions if decisions is None else decisions
        # if use implicit corrector, the NFE is not equal to the NFE here
        if decisions is None:
            assert self.args.timesteps is not None, "Please provide the number of timesteps"
            steps = self.args.timesteps
            nfe = steps
            return nfe
        if decisions.get("orders", None) is not None:
            steps = len(self.decisions["orders"])
            afs = decisions.get("afs", "no_afs")
            if afs != "no_afs":
                nfe = steps - 1
            else:
                nfe = steps
            return nfe
        else:
            assert self.args.timesteps is not None, "Please provide the number of timesteps"
            steps = self.args.timesteps
            afs = decisions.get("afs", "no_afs")
            if afs != "no_afs":
                nfe = steps - 1
            else:
                nfe = steps
            return nfe
    
    def get_model_function_continuous(self):
        classifier_scale = self.config.sampling.classifier_scale if self.args.scale is None else self.args.scale

        model, classifier = load_model_for_sample(self.config, self.device, self.args.gpu)
        from uni_sampler.uni_sampler import NoiseScheduleVP, model_wrapper
        def model_fn(x, t, **model_kwargs):
            print("Call the model...")
            out = model(x, t, **model_kwargs)
            # If the model outputs both 'mean' and 'variance' (such as improved-DDPM and guided-diffusion),
            # We only use the 'mean' output for DPM-Solver, because DPM-Solver is based on diffusion ODEs.
            if "out_channels" in self.config.model.__dict__.keys():
                if self.config.model.out_channels == 6:
                    out = torch.split(out, 3, dim=1)[0]
            return out

        def classifier_fn(x, t, y, **classifier_kwargs):
            logits = classifier(x, t)
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            return log_probs[range(len(logits)), y.view(-1)]

        noise_schedule = NoiseScheduleVP(schedule=self.config.diffusion.beta_schedule, betas=self.betas)
        model_fn_continuous = model_wrapper(
            model_fn,
            noise_schedule,
            time_input_type=self.config.sampling.time_input_type,
            model_type="noise",
            guidance_type="uncond" if classifier is None else "classifier",
            guidance_scale=classifier_scale,
            classifier_fn=classifier_fn,
            classifier_kwargs={},
        )
        return model_fn_continuous, noise_schedule