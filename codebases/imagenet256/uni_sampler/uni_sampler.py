import torch
import torch.nn.functional as F
import math
import numpy as np
import os
import copy


class NoiseScheduleVP:
    def __init__(
            self,
            schedule='discrete',
            betas=None,
            alphas_cumprod=None,
            continuous_beta_0=0.1,
            continuous_beta_1=20.,
            dtype=torch.float32,
        ):
        """Create a wrapper class for the forward SDE (VP type).

        ***
        Update: We support discrete-time diffusion models by implementing a picewise linear interpolation for log_alpha_t.
                We recommend to use schedule='discrete' for the discrete-time diffusion models, especially for high-resolution images.
        ***

        The forward SDE ensures that the condition distribution q_{t|0}(x_t | x_0) = N ( alpha_t * x_0, sigma_t^2 * I ).
        We further define lambda_t = log(alpha_t) - log(sigma_t), which is the half-logSNR (described in the DPM-Solver paper).
        Therefore, we implement the functions for computing alpha_t, sigma_t and lambda_t. For t in [0, T], we have:

            log_alpha_t = self.marginal_log_mean_coeff(t)
            sigma_t = self.marginal_std(t)
            lambda_t = self.marginal_lambda(t)

        Moreover, as lambda(t) is an invertible function, we also support its inverse function:

            t = self.inverse_lambda(lambda_t)

        ===============================================================

        We support both discrete-time DPMs (trained on n = 0, 1, ..., N-1) and continuous-time DPMs (trained on t in [t_0, T]).

        1. For discrete-time DPMs:

            For discrete-time DPMs trained on n = 0, 1, ..., N-1, we convert the discrete steps to continuous time steps by:
                t_i = (i + 1) / N
            e.g. for N = 1000, we have t_0 = 1e-3 and T = t_{N-1} = 1.
            We solve the corresponding diffusion ODE from time T = 1 to time t_0 = 1e-3.

            Args:
                betas: A `torch.Tensor`. The beta array for the discrete-time DPM. (See the original DDPM paper for details)
                alphas_cumprod: A `torch.Tensor`. The cumprod alphas for the discrete-time DPM. (See the original DDPM paper for details)

            Note that we always have alphas_cumprod = cumprod(1 - betas). Therefore, we only need to set one of `betas` and `alphas_cumprod`.

            **Important**:  Please pay special attention for the args for `alphas_cumprod`:
                The `alphas_cumprod` is the \hat{alpha_n} arrays in the notations of DDPM. Specifically, DDPMs assume that
                    q_{t_n | 0}(x_{t_n} | x_0) = N ( \sqrt{\hat{alpha_n}} * x_0, (1 - \hat{alpha_n}) * I ).
                Therefore, the notation \hat{alpha_n} is different from the notation alpha_t in DPM-Solver. In fact, we have
                    alpha_{t_n} = \sqrt{\hat{alpha_n}},
                and
                    log(alpha_{t_n}) = 0.5 * log(\hat{alpha_n}).


        2. For continuous-time DPMs:

            We support the linear VPSDE for the continuous time setting. The hyperparameters for the noise
            schedule are the default settings in Yang Song's ScoreSDE:

            Args:
                beta_min: A `float` number. The smallest beta for the linear schedule.
                beta_max: A `float` number. The largest beta for the linear schedule.
                T: A `float` number. The ending time of the forward process.

        ===============================================================

        Args:
            schedule: A `str`. The noise schedule of the forward SDE. 'discrete' for discrete-time DPMs,
                    'linear' for continuous-time DPMs.
        Returns:
            A wrapper object of the forward SDE (VP type).
        
        ===============================================================

        Example:

        # For discrete-time DPMs, given betas (the beta array for n = 0, 1, ..., N - 1):
        >>> ns = NoiseScheduleVP('discrete', betas=betas)

        # For discrete-time DPMs, given alphas_cumprod (the \hat{alpha_n} array for n = 0, 1, ..., N - 1):
        >>> ns = NoiseScheduleVP('discrete', alphas_cumprod=alphas_cumprod)

        # For continuous-time DPMs (VPSDE), linear schedule:
        >>> ns = NoiseScheduleVP('linear', continuous_beta_0=0.1, continuous_beta_1=20.)

        """

        if schedule not in ['discrete', 'linear', 'cosine']:
            raise ValueError("Unsupported noise schedule {}. The schedule needs to be 'discrete' or 'linear'".format(schedule))

        self.schedule = schedule
        if schedule == 'discrete':
            if betas is not None:
                log_alphas = 0.5 * torch.log(1 - betas).cumsum(dim=0)
            else:
                assert alphas_cumprod is not None
                log_alphas = 0.5 * torch.log(alphas_cumprod)
            self.T = 1.
            self.log_alpha_array = self.numerical_clip_alpha(log_alphas).reshape((1, -1,)).to(dtype=dtype)
            self.total_N = self.log_alpha_array.shape[1]
            self.t_array = torch.linspace(0., 1., self.total_N + 1)[1:].reshape((1, -1)).to(dtype=dtype)
        elif schedule == "linear":
            self.T = 1.
            self.total_N = 1000
            self.beta_0 = continuous_beta_0
            self.beta_1 = continuous_beta_1
        elif schedule == "cosine":
            self.T = 0.9946
            self.cosine_s = 0.008
            self.cosine_beta_max = 999.
            self.cosine_t_max = math.atan(self.cosine_beta_max * (1. + self.cosine_s) / math.pi) * 2. * (1. + self.cosine_s) / math.pi - self.cosine_s
            self.cosine_log_alpha_0 = math.log(math.cos(self.cosine_s / (1. + self.cosine_s) * math.pi / 2.))
            

    def numerical_clip_alpha(self, log_alphas, clipped_lambda=-5.1):
        """
        For some beta schedules such as cosine schedule, the log-SNR has numerical isssues. 
        We clip the log-SNR near t=T within -5.1 to ensure the stability.
        Such a trick is very useful for diffusion models with the cosine schedule, such as i-DDPM, guided-diffusion and GLIDE.
        """
        log_sigmas = 0.5 * torch.log(1. - torch.exp(2. * log_alphas))
        lambs = log_alphas - log_sigmas  
        idx = torch.searchsorted(torch.flip(lambs, [0]), clipped_lambda)
        if idx > 0:
            log_alphas = log_alphas[:-idx]
        return log_alphas

    def marginal_log_mean_coeff(self, t):
        """
        Compute log(alpha_t) of a given continuous-time label t in [0, T].
        """
        if self.schedule == 'discrete':
            return interpolate_fn(t.reshape((-1, 1)), self.t_array.to(t.device), self.log_alpha_array.to(t.device)).reshape((-1))
        elif self.schedule == 'linear':
            return -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        elif self.schedule == 'cosine':
            log_alpha_fn = lambda s: torch.log(torch.cos((s + self.cosine_s) / (1. + self.cosine_s) * math.pi / 2.))
            log_alpha_t =  log_alpha_fn(t) - self.cosine_log_alpha_0
            return log_alpha_t


    def marginal_alpha(self, t):
        """
        Compute alpha_t of a given continuous-time label t in [0, T].
        """
        return torch.exp(self.marginal_log_mean_coeff(t))

    def marginal_std(self, t):
        """
        Compute sigma_t of a given continuous-time label t in [0, T].
        """
        return torch.sqrt(1. - torch.exp(2. * self.marginal_log_mean_coeff(t)))

    def marginal_lambda(self, t):
        """
        Compute lambda_t = log(alpha_t) - log(sigma_t) of a given continuous-time label t in [0, T].
        """
        log_mean_coeff = self.marginal_log_mean_coeff(t)
        log_std = 0.5 * torch.log(1. - torch.exp(2. * log_mean_coeff))
        return log_mean_coeff - log_std

    def inverse_lambda(self, lamb):
        """
        Compute the continuous-time label t in [0, T] of a given half-logSNR lambda_t.
        """
        if self.schedule == 'linear':
            tmp = 2. * (self.beta_1 - self.beta_0) * torch.logaddexp(-2. * lamb, torch.zeros((1,)).to(lamb))
            Delta = self.beta_0**2 + tmp
            return tmp / (torch.sqrt(Delta) + self.beta_0) / (self.beta_1 - self.beta_0)
        elif self.schedule == 'discrete':
            log_alpha = -0.5 * torch.logaddexp(torch.zeros((1,)).to(lamb.device), -2. * lamb)
            t = interpolate_fn(log_alpha.reshape((-1, 1)), torch.flip(self.log_alpha_array.to(lamb.device), [1]), torch.flip(self.t_array.to(lamb.device), [1]))
            return t.reshape((-1,))
        elif self.schedule == "cosine":
            log_alpha = -0.5 * torch.logaddexp(-2. * lamb, torch.zeros((1,)).to(lamb))
            t_fn = lambda log_alpha_t: torch.arccos(torch.exp(log_alpha_t + self.cosine_log_alpha_0)) * 2. * (1. + self.cosine_s) / math.pi - self.cosine_s
            t = t_fn(log_alpha)
            return t


def model_wrapper(
    model,
    noise_schedule,
    time_input_type='1',
    model_type="noise",
    model_kwargs={},
    guidance_type="uncond",
    condition=None,
    unconditional_condition=None,
    guidance_scale=1.,
    classifier_fn=None,
    classifier_kwargs={},
):
    """Create a wrapper function for the noise prediction model.

    DPM-Solver needs to solve the continuous-time diffusion ODEs. For DPMs trained on discrete-time labels, we need to
    firstly wrap the model function to a noise prediction model that accepts the continuous time as the input.

    We support four types of the diffusion model by setting `model_type`:

        1. "noise": noise prediction model. (Trained by predicting noise).

        2. "x_start": data prediction model. (Trained by predicting the data x_0 at time 0).

        3. "v": velocity prediction model. (Trained by predicting the velocity).
            The "v" prediction is derivation detailed in Appendix D of [1], and is used in Imagen-Video [2].

            [1] Salimans, Tim, and Jonathan Ho. "Progressive distillation for fast sampling of diffusion models."
                arXiv preprint arXiv:2202.00512 (2022).
            [2] Ho, Jonathan, et al. "Imagen Video: High Definition Video Generation with Diffusion Models."
                arXiv preprint arXiv:2210.02303 (2022).
    
        4. "score": marginal score function. (Trained by denoising score matching).
            Note that the score function and the noise prediction model follows a simple relationship:
            ```
                noise(x_t, t) = -sigma_t * score(x_t, t)
            ```

    We support three types of guided sampling by DPMs by setting `guidance_type`:
        1. "uncond": unconditional sampling by DPMs.
            The input `model` has the following format:
            ``
                model(x, t_input, **model_kwargs) -> noise | x_start | v | score
            ``

        2. "classifier": classifier guidance sampling [3] by DPMs and another classifier.
            The input `model` has the following format:
            ``
                model(x, t_input, **model_kwargs) -> noise | x_start | v | score
            `` 

            The input `classifier_fn` has the following format:
            ``
                classifier_fn(x, t_input, cond, **classifier_kwargs) -> logits(x, t_input, cond)
            ``

            [3] P. Dhariwal and A. Q. Nichol, "Diffusion models beat GANs on image synthesis,"
                in Advances in Neural Information Processing Systems, vol. 34, 2021, pp. 8780-8794.

        3. "classifier-free": classifier-free guidance sampling by conditional DPMs.
            The input `model` has the following format:
            ``
                model(x, t_input, cond, **model_kwargs) -> noise | x_start | v | score
            `` 
            And if cond == `unconditional_condition`, the model output is the unconditional DPM output.

            [4] Ho, Jonathan, and Tim Salimans. "Classifier-free diffusion guidance."
                arXiv preprint arXiv:2207.12598 (2022).
        

    The `t_input` is the time label of the model, which may be discrete-time labels (i.e. 0 to 999)
    or continuous-time labels (i.e. epsilon to T).

    We wrap the model function to accept only `x` and `t_continuous` as inputs, and outputs the predicted noise:
    ``
        def model_fn(x, t_continuous) -> noise:
            t_input = get_model_input_time(t_continuous)
            return noise_pred(model, x, t_input, **model_kwargs)         
    ``
    where `t_continuous` is the continuous time labels (i.e. epsilon to T). And we use `model_fn` for DPM-Solver.

    ===============================================================

    Args:
        model: A diffusion model with the corresponding format described above.
        noise_schedule: A noise schedule object, such as NoiseScheduleVP.
        model_type: A `str`. The parameterization type of the diffusion model.
                    "noise" or "x_start" or "v" or "score".
        model_kwargs: A `dict`. A dict for the other inputs of the model function.
        guidance_type: A `str`. The type of the guidance for sampling.
                    "uncond" or "classifier" or "classifier-free".
        condition: A pytorch tensor. The condition for the guided sampling.
                    Only used for "classifier" or "classifier-free" guidance type.
        unconditional_condition: A pytorch tensor. The condition for the unconditional sampling.
                    Only used for "classifier-free" guidance type.
        guidance_scale: A `float`. The scale for the guided sampling.
        classifier_fn: A classifier function. Only used for the classifier guidance.
        classifier_kwargs: A `dict`. A dict for the other inputs of the classifier function.
    Returns:
        A noise prediction model that accepts the noised data and the continuous time as the inputs.
    """

    def get_model_input_time(t_continuous):
        """
        Convert the continuous-time `t_continuous` (in [epsilon, T]) to the model input time.
        For discrete-time DPMs, we convert `t_continuous` in [1 / N, 1] to `t_input` in [0, 1000 * (N - 1) / N].
        For continuous-time DPMs, we just use `t_continuous`.
        """
        total_N = 1000
        if time_input_type == '0':
            # discrete_type == '0' means that the model is continuous-time model.
            # For continuous-time DPMs, the continuous time equals to the discrete time.
            return t_continuous
        elif time_input_type == '1':
            # Type-1 discrete label, as detailed in the Appendix of DPM-Solver.
            return 1000. * torch.max(t_continuous - 1. / total_N, torch.zeros_like(t_continuous).to(t_continuous))
        elif time_input_type == '2':
            # Type-2 discrete label, as detailed in the Appendix of DPM-Solver.
            max_N = (total_N - 1) / total_N * 1000.
            return max_N * t_continuous
        elif time_input_type == '3':
            # Type-3 discrete label, for score based model from https://github.com/yang-song/score_sde_pytorch.
            return 999 * t_continuous
        else:
            raise ValueError("Unsupported time input type {}, must be '0' or '1' or '2'".format(time_input_type))

    def noise_pred_fn(x, t_continuous, cond=None):
        t_input = get_model_input_time(t_continuous)
        if cond is None:
            output = model(x, t_input, **model_kwargs)
        else:
            output = model(x, t_input, cond, **model_kwargs)
        if model_type == "noise":
            return output
        elif model_type == "x_start":
            alpha_t, sigma_t = noise_schedule.marginal_alpha(t_continuous), noise_schedule.marginal_std(t_continuous)
            return (x - alpha_t * output) / sigma_t
        elif model_type == "v":
            alpha_t, sigma_t = noise_schedule.marginal_alpha(t_continuous), noise_schedule.marginal_std(t_continuous)
            return alpha_t * output + sigma_t * x
        elif model_type == "score":
            sigma_t = noise_schedule.marginal_std(t_continuous)
            return -sigma_t * output

    def cond_grad_fn(x, t_input):
        """
        Compute the gradient of the classifier, i.e. nabla_{x} log p_t(cond | x_t).
        """
        with torch.enable_grad():
            x_in = x.detach().requires_grad_(True)
            log_prob = classifier_fn(x_in, t_input, condition, **classifier_kwargs)
            return torch.autograd.grad(log_prob.sum(), x_in)[0]

    def model_fn(x, t_continuous):
        """
        The noise predicition model function that is used for DPM-Solver.
        """
        if guidance_type == "uncond":
            return noise_pred_fn(x, t_continuous)
        elif guidance_type == "classifier":
            assert classifier_fn is not None
            t_input = get_model_input_time(t_continuous)
            cond_grad = cond_grad_fn(x, t_input)
            sigma_t = noise_schedule.marginal_std(t_continuous)
            noise = noise_pred_fn(x, t_continuous)
            return noise - guidance_scale * expand_dims(sigma_t, x.dim()) * cond_grad
        elif guidance_type == "classifier-free":
            if guidance_scale == 1. or unconditional_condition is None:
                return noise_pred_fn(x, t_continuous, cond=condition)
            else:
                x_in = torch.cat([x] * 2)
                t_in = torch.cat([t_continuous] * 2)
                c_in = torch.cat([unconditional_condition, condition])
                noise_uncond, noise = noise_pred_fn(x_in, t_in, cond=c_in).chunk(2)
                return noise_uncond + guidance_scale * (noise - noise_uncond)

    assert model_type in ["noise", "x_start", "v", "score"]
    assert guidance_type in ["uncond", "classifier", "classifier-free"]
    return model_fn

class uni_sampler:
    def __init__(
        self,
        model_fn,
        noise_schedule,
        correcting_x0_fn=None,
        correcting_xt_fn=None,
        thresholding_max_val=1.,
        dynamic_thresholding_ratio=0.995,
        statistics_dir=None,
        device="cuda",
        dpmsolver_v3_t_start=1,
        dpmsolver_v3_t_end=1e-3,
        average_x0_dir=None,
    ):
        self.model = lambda x, t: model_fn(x, t.expand((x.shape[0]))) if model_fn is not None else None
        self.device = device
        self.noise_schedule = noise_schedule
        if correcting_x0_fn == "dynamic_thresholding":
            self.correcting_x0_fn = self.dynamic_thresholding_fn
        else:
            self.correcting_x0_fn = correcting_x0_fn
        self.correcting_xt_fn = correcting_xt_fn
        self.dynamic_thresholding_ratio = dynamic_thresholding_ratio
        self.thresholding_max_val = thresholding_max_val
        self.statistics_init_torch(statistics_dir, t_start=dpmsolver_v3_t_start, t_end=dpmsolver_v3_t_end)
        self.exp_coeffs = {} # store high-order exponential coefficients (lazy)
        if average_x0_dir is not None:
            self.average_x0 = torch.load(average_x0_dir)
        else:
            try:
                self.average_x0 = torch.load("average_x0.pth")
            except:
                self.average_x0 = None
                print("No average_x0.pth found. Please provide the average_x0_dir.")
    
    def dynamic_thresholding_fn(self, x0, t):
        """
        The dynamic thresholding method. 
        """
        dims = x0.dim()
        p = self.dynamic_thresholding_ratio
        s = torch.quantile(torch.abs(x0).reshape((x0.shape[0], -1)), p, dim=1)
        s = expand_dims(torch.maximum(s, self.thresholding_max_val * torch.ones_like(s).to(s.device)), dims)
        try:
            x0 = torch.clamp(x0, -s, s) / s
        except:
            for i in range(len(x0)): # some low torch version doesn't support the second and third args of torch.clamp being torch
                x0[i] = torch.clamp(x0[i], -s[i].item(), s[i].item()) / s[i].item()
        return x0

    def noise_prediction_fn(self, x, t):
        """
        Return the noise prediction model.
        """
        noise = self.model(x, t)
        return noise

    def data_prediction_from_noise(self, x, noise, t):
        """
        Return the data prediction from the noise
        """
        alpha_t, sigma_t = self.noise_schedule.marginal_alpha(t), self.noise_schedule.marginal_std(t)
        x0 = (x - sigma_t * noise) / alpha_t
        if self.correcting_x0_fn is not None:
            # print("Use thresholding for data prediction")
            x0 = self.correcting_x0_fn(x0, t)
        return x0

    def noise_prediction_from_data(self, x, x0, t):
        """
        Return the noise prediction from the data
        """
        alpha_t, sigma_t = self.noise_schedule.marginal_alpha(t), self.noise_schedule.marginal_std(t)
        noise = (x - x0 * alpha_t) / sigma_t
        return noise

    def model_fn(self, x, t):
        """
        Get noise prediction model and the data prediction from the noise. 
        """
        noise = self.noise_prediction_fn(x, t)
        data = self.data_prediction_from_noise(x, noise, t)
        if self.correcting_x0_fn is not None:
            # print("Use thresholding for noise prediction")
            alpha_t, sigma_t = self.noise_schedule.marginal_alpha(t), self.noise_schedule.marginal_std(t)
            noise = (x - data * alpha_t) / sigma_t
        return noise, data
        
    def get_time_steps(self, skip_type, t_T, t_0, N, device):
        """Compute the intermediate time steps for sampling.

        Args:
            skip_type: A `str`. The type for the spacing of the time steps. We support three types:
                - 'logSNR': uniform logSNR for the time steps.
                - 'time_uniform': uniform time for the time steps. (**Recommended for high-resolutional data**.)
                - 'time_quadratic': quadratic time for the time steps. (Used in DDIM for low-resolutional data.)
            t_T: A `float`. The starting time of the sampling (default is T).
            t_0: A `float`. The ending time of the sampling (default is epsilon).
            N: A `int`. The total number of the spacing of the time steps.
            device: A torch device.
        Returns:
            A pytorch tensor of the time steps, with the shape (N + 1,).
        """
        if skip_type == 'logSNR':
            lambda_T = self.noise_schedule.marginal_lambda(torch.tensor(t_T).to(device))
            lambda_0 = self.noise_schedule.marginal_lambda(torch.tensor(t_0).to(device))
            logSNR_steps = torch.linspace(lambda_T.cpu().item(), lambda_0.cpu().item(), N + 1).to(device)
            return self.noise_schedule.inverse_lambda(logSNR_steps)
        elif skip_type == 'time_uniform':
            return torch.linspace(t_T, t_0, N + 1).to(device)
        elif skip_type == 'time_quadratic':
            t_order = 2
            t = torch.linspace(t_T**(1. / t_order), t_0**(1. / t_order), N + 1).pow(t_order).to(device)
            return t
        elif skip_type == "reverse_time_quadratic":
            t_order = 2
            t = torch.linspace(t_T**(1. / t_order), t_0**(1. / t_order), N + 1).pow(t_order).to(device)
            t = (t[0] - t + t[-1]).flip(dims=(0,))
            return t
        elif skip_type == "reverse_1.5":
            t_order = 1.5
            t = torch.linspace(t_T**(1. / t_order), t_0**(1. / t_order), N + 1).pow(t_order).to(device)
            t = (t[0] - t + t[-1]).flip(dims=(0,))
            return t
        else:
            raise ValueError("Unsupported skip_type {}, need to be 'logSNR' or 'time_uniform' or 'time_quadratic'".format(skip_type))
        
    def get_orders(self, order, method, lower_order_final, steps):
        if method=="multistep":
            orders = [order] * steps
            for i in range(order):
                orders[i] = i + 1
            if lower_order_final:
                for i in range(1, order):
                    orders[-i] = i
        elif method=="singlestep":
            K = steps // order
            outer_orders = [order] * K
            for i in range(steps % order):
                outer_orders.append(i + 1)
            orders = []
            for i in outer_orders:
                for j in range(i):
                    orders.append(j + 1)
        else:
            raise ValueError(f"Method must be multistep or singlestep. Got {method}")
        
        return orders
    
    def get_start_points(self, method, orders):
        if method == "multistep":
            start_points = [-1] * len(orders)
        elif method == "singlestep":
            start_points = [-p for p in orders]
        else:
            raise NotImplementedError(f"method must be in multistep or singlestep. Got {method}")
        
        return start_points
    
    def get_derivatives_estimating_method(self, orders):
        derivative_types = []
        for order in orders:
            derivative_type = {}
            for i in range(1, order):
                derivative_type[i] = {"estimate":f"Difference_{order - 1}", "relaxation":None, "active_points":None}
            derivative_types.append(derivative_type)
            
        return derivative_types
        
    def get_corrector_types(self, use_corrector, corrector_type, steps):
        if use_corrector:
            assert corrector_type in ["pseudo", "implicit"], "Corrector type must be pseudo or implicit"
            corrector_types = [corrector_type] * steps
        else:
            corrector_types = ["no_corrector"] * steps
            
        return corrector_types
    
    def get_diff_derivative(self, lambda_prev_list, model_prev_list, estimate_order, derivative_order, start_point, active_points=None, get_active_point_method="single", corrector=False):
        assert derivative_order <= estimate_order < len(lambda_prev_list) == len(model_prev_list)
        
        # get the index of function evaluations which can be exploited

        if active_points is None:
            active_points = []
            if corrector:
                active_points.append(-1)
            right_offset = start_point + 1
            left_offset = start_point - 1
            if get_active_point_method == "single":
                while len(active_points) < estimate_order:
                    if right_offset <= -1 and right_offset not in active_points:
                        active_points.append(right_offset)
                        right_offset += 1
                    else:
                        break
                while len(active_points) < estimate_order:
                    if left_offset >= -len(model_prev_list) and left_offset not in active_points:
                        active_points.append(left_offset)
                        left_offset -= 1
            elif get_active_point_method == "around":
                while len(active_points) < estimate_order:
                    if right_offset < -1 and right_offset not in active_points:
                        active_points.append(right_offset)
                        if len(active_points) >= estimate_order:
                            break
                        right_offset += 1
                    if left_offset >= -len(model_prev_list) and left_offset not in active_points:
                        active_points.append(left_offset)
                        left_offset -= 1   
            # print(f"Got None active points. Now get active points {active_points} by method {get_active_point_method}")
                        
        assert len(active_points) == estimate_order
        assert start_point not in active_points
        if corrector:
            assert -1 in active_points and start_point != -1
        
        Delta_Fs = []
        Delta_hs = []    
        for index in active_points:
            Delta_hs.append(lambda_prev_list[index] - lambda_prev_list[start_point])
            Delta_Fs.append(model_prev_list[index] - model_prev_list[start_point])
        Delta_Fs = torch.stack(Delta_Fs, dim=1)
        Delta_hs = torch.stack(Delta_hs)
    
        # calculate derivative through Taylor expansion to the start point
        # 1. construct Taylor expansion coefficients matrix and augmented column vector
        taylor_coff = []
        fac = 1
        for i in range(1, estimate_order + 1):
            fac *= i
            taylor_coff.append(Delta_hs.pow(i) / fac)
        taylor_coff = torch.stack(taylor_coff)
        b = torch.zeros(estimate_order, device=taylor_coff.device)
        b[derivative_order - 1] = 1.
        # 2. solve the system of linear equations
        a = torch.linalg.solve(taylor_coff, b)
        # 3. calculate the derivative
        d = torch.einsum("k,bkchw->bchw", a, Delta_Fs)

        return d        
    
    def update(self, x_prev_list, model_prev_list, t_prev_list, t, taylor_order, prediction_type, start_point, derivative_types, corrector=False):
        ns = self.noise_schedule
        
        try:
            assert len(derivative_types) == taylor_order - 1 or taylor_order == 1
        except:
            import ipdb; ipdb.set_trace()
        
        # # print information for debug
        # t_start = t_prev_list[start_point]
        # print(f"Corrector: {corrector} | Target timestep: {t} | Start timestep: {t_start} | Start point: {start_point} | Taylor order: {taylor_order} | Prediction type: {prediction_type} | Derivative type: {derivative_types}")
        
        # get lambda
        lambda_prev_list = []
        for t_prev in t_prev_list:
            lambda_prev_list.append(ns.marginal_lambda(t_prev).squeeze() if t_prev is not None else None)
        lambda_t = ns.marginal_lambda(t).squeeze()
        h = lambda_t - lambda_prev_list[start_point]
        
        # get alpha_s, alpha_t, sigma_s, sigma_t
        x_s = x_prev_list[start_point]
        t_s = t_prev_list[start_point]
        log_alpha_s, log_alpha_t = ns.marginal_log_mean_coeff(t_s), ns.marginal_log_mean_coeff(t)
        sigma_s, sigma_t = ns.marginal_std(t_s), ns.marginal_std(t)
        alpha_t = ns.marginal_alpha(t)
        
        if prediction_type in ["noise_prediction", "data_prediction"]:
            # calculate phi
            hh = h if prediction_type == "noise_prediction" else -h
            prev_phi = torch.exp(hh)
            phis = []
            fac = 1
            for i in range(taylor_order):
                fac *= max(i, 1)
                phi_i = (prev_phi - 1 / fac) / hh
                phis.append(phi_i)
                prev_phi = phi_i
    
        if prediction_type == "noise_prediction":
            
            # DDIM update
            x_t = (torch.exp(log_alpha_t - log_alpha_s)) * x_s - sigma_t * h * phis[0] * model_prev_list[start_point]

            # Utilize higher-order terms
            assert len(derivative_types) >= taylor_order - 1
            for i in range(1, taylor_order):
                
                # estimate the derivatives
                # 1. calculate the derivatives
                derivative_type = derivative_types[i - 1]
                if "Difference" in derivative_type["estimate"]:
                    estimate_order = int(derivative_type["estimate"].split("_")[1]) # the number of Taylor expansions which can be utilized to calculate the derivative
                    active_points = derivative_type.get("active_points", None) # the indices of points which are Taylor expanded to the start point
                    derivative_i = self.get_diff_derivative(lambda_prev_list, model_prev_list, estimate_order, i, start_point, active_points, corrector=corrector)
                else:
                    raise NotImplementedError
                # 2. relax the estimated derivative by multiple a relaxation term (1 + O(h^p))
                if i == 1: # currently only support relaxation for the first-order derivative
                    relaxation_type = derivative_type["relaxation"]
                    if relaxation_type is None or relaxation_type == "no_relaxation": # no relaxation
                        pass
                    elif relaxation_type == "linear": # linear relaxation
                        relaxation_coefficient = derivative_type.get("relaxation_coefficient", 0)
                        derivative_i = self.relax_derivative(derivative_i, relaxation_type, h, relaxation_coefficient)
                    else: # empriical relaxation
                        derivative_i = self.relax_derivative(derivative_i, relaxation_type, h)
                
                # # print information 
                # relaxation_type = relaxation_type if relaxed else "No relaxation"
                # print(f"Estimate {i}-order derivative | active points: {active_points} | estimate order: {estimate_order} | relaxation type: {relaxation_type}")

                x_t -= sigma_t * (h ** (i + 1)) * phis[i] * derivative_i
                
        elif prediction_type == "data_prediction":
            
            # DDIM update
            x_t = (sigma_t / sigma_s) * x_s + alpha_t * h * phis[0] * model_prev_list[start_point]
            
            # Utilize higher-order terms
            for i in range(1, taylor_order):
                
                # estimate the derivatives
                # 1. calculate the derivatives
                derivative_type = derivative_types[i - 1]
                if "Difference" in derivative_type["estimate"]:
                    estimate_order = int(derivative_type["estimate"].split("_")[1]) # the accuracy order of estimation
                    active_points = derivative_type.get("active_points", None)
                    derivative_i = self.get_diff_derivative(lambda_prev_list, model_prev_list, estimate_order, i, start_point, active_points,corrector=corrector)
                else:
                    raise NotImplementedError
                # 2. relax the estimated derivative by multiple a relaxation term (1 + O(h^p))
                if i == 1: # currently only support relaxation for the first-order derivative
                    relaxation_type = derivative_type["relaxation"]
                    if relaxation_type is None or relaxation_type == "no_relaxation": # no relaxation
                        pass
                    elif relaxation_type == "linear": # linear relaxation
                        relaxation_coefficient = derivative_type.get("relaxation_coefficient", 0)
                        derivative_i = self.relax_derivative(derivative_i, relaxation_type, h, relaxation_coefficient)
                    else: # empriical relaxation
                        h_temp = -h # in previous practice, when relaxing the derivative of data prediction, the sign of h should be reversed (this won't be applied to linear relaxation)
                        derivative_i = self.relax_derivative(derivative_i, relaxation_type, h_temp)

                # # print information 
                # relaxation_type = relaxation_type if relaxed else "No relaxation"
                # print(f"Estimate {i}-order derivative | active points: {active_points} | estimate order: {estimate_order} | relaxation type: {relaxation_type}")
                
                x_t += alpha_t * (h ** (i + 1)) * phis[i] * derivative_i

        elif prediction_type == "dpmsolver_v3_prediction":
            index_prev_list = self.get_indexes(t_prev_list)
            lambda_prev_list = []
            alpha_prev_list = []
            for t_prev in t_prev_list:
                if t_prev is None:
                    lambda_prev_list.append(None)
                    alpha_prev_list.append(None)
                else:
                    lambda_prev_list.append(ns.marginal_lambda(t_prev).squeeze())
                    alpha_prev_list.append(ns.marginal_alpha(t_prev).squeeze())
            # sigma_prev_list = ns.marginal_std(t_prev_list)
            alpha_s = alpha_prev_list[start_point]
            index_s = index_prev_list[start_point]
            index_t = self.get_index(t)
            g_prev_list = self.get_g_prev_list(index_s,model_prev_list,index_prev_list)
            
            # 1.first order update, different from DDIM
            x_t = (
            alpha_t / alpha_s * torch.exp(self.L[index_s] - self.L[index_t]) * x_s
            - alpha_t * torch.exp(-self.L[index_t] - self.S[index_s]) * (self.I[index_t] - self.I[index_s]) * g_prev_list[start_point]
            - alpha_t
            * torch.exp(-self.L[index_t])
            * (self.C[index_t] - self.C[index_s] - self.B[index_s] * (self.I[index_t] - self.I[index_s]))
            )
            # 2. higher order update
            for i in range(1, taylor_order):
                # estimate the derivatives
                # 1. calculate the derivatives
                derivative_type = derivative_types[i - 1]
                if "Difference" in derivative_type["estimate"]:
                    estimate_order = int(derivative_type["estimate"].split("_")[1]) # the accuracy order of estimation
                    active_points = derivative_type.get("active_points", None)
                    derivative_i = self.get_diff_derivative(lambda_prev_list, g_prev_list, estimate_order, i, start_point, active_points, corrector=corrector)
                else:
                    raise NotImplementedError
                # 2. relax the estimated derivative by multiple a relaxation term (1 + O(h^p))
                if i == 1: # currently only support relaxation for the first-order derivative
                    relaxation_type = derivative_type["relaxation"]
                    if relaxation_type is None or relaxation_type == "no_relaxation": # no relaxation
                        pass
                    elif relaxation_type == "linear": # linear relaxation
                        relaxation_coefficient = derivative_type.get("relaxation_coefficient", 0)
                        derivative_i = self.relax_derivative(derivative_i, relaxation_type, h, relaxation_coefficient)
                    else: # empriical relaxation
                        derivative_i = self.relax_derivative(derivative_i, relaxation_type, h)
                x_t = (
                    x_t
                    - alpha_t
                    * torch.exp(self.L[index_s] - self.L[index_t])
                    * self.compute_exponential_coefficients_high_order(index_s, index_t, order=i)   
                    * derivative_i
                )
        return x_t
        
    def sample(self, x, decisions, steps=20, t_start=None, t_end=None, order=2, skip_type='time_uniform', prediction_type="noise_prediction", 
        method='multistep', lower_order_final=True, use_corrector=False, corrector_type="pseudo", denoise_to_zero=False, return_intermediate=False, update_statistics=False
    ):
        device = x.device
        
        # get timesteps, parameter "steps" is used only when "timesteps" is not provided
        if decisions.get("timesteps", None) is not None:
            timesteps = decisions["timesteps"].to(device)
        else:
            t_0 = 1. / self.noise_schedule.total_N if t_end is None else t_end
            t_T = self.noise_schedule.T if t_start is None else t_start
            assert t_0 > 0 and t_T > 0, "Time range needs to be greater than 0. For discrete-time DPMs, it needs to be in [1 / N, 1], where N is the length of betas array"
            timesteps = self.get_time_steps(skip_type=skip_type, t_T=t_T, t_0=t_0, N=steps, device=device)
            assert timesteps.shape[0] - 1 == steps
        steps = len(timesteps) - 1
        # analytical first step
        afs = decisions.get("afs", "no_afs")
        
        # get orders of taylor expansion
        if decisions.get("orders", None) is not None:
            orders = decisions["orders"]
        else:
            orders = self.get_orders(order, method, lower_order_final, steps)
            
        # get prediction type
        if decisions.get("prediction_types", None) is not None:
            prediction_types = decisions["prediction_types"]
        else:
            prediction_types = [prediction_type] * steps
            
        # get start point of each step
        if decisions.get("start_points", None) is not None:
            start_points = decisions["start_points"]
        else:
            start_points = self.get_start_points(method, orders)
            
        # get methods for estimating derivatives
        if decisions.get("derivative_types", None) is not None:
            derivative_types = decisions["derivative_types"]
        else:
            derivative_types = self.get_derivatives_estimating_method(orders)
            
        # get corrector
        if decisions.get("corrector_types", None) is not None:
            correctors = decisions["corrector_types"]
        else:
            correctors = self.get_corrector_types(use_corrector, corrector_type, steps)
        if decisions.get("skip_coefficients", None) is not None:
            skip_coefficients = decisions["skip_coefficients"]
            pid = os.getpid()
            torch.save(skip_coefficients, f"./temp/skip_coefficients_{pid}.pth")
        intermediates = []
        
        noise_prev_list = [None] * (steps + 1)
        data_prev_list = [None] * (steps + 1)
        f_prev_list = [None] * (steps + 1)

        t_prev_list = [None] * (steps + 1)
        x_prev_list = [None] * (steps + 1)
        
        # dpm_solver_v3
        # index_prev_list = [None] * (steps + 1)
        
        
        
        with torch.no_grad():
            step = 0
            t = timesteps[step]
            t_prev_list[-1] = t
            if afs == "no_afs":
                noise_prev_list[-1], data_prev_list[-1] = self.model_fn(x, t)
                f_prev_list[-1] = self.f_prediction_from_noise(x, noise_prev_list[-1], t)
            elif afs == "zero_x0":
                noise_prev_list[-1] = x
                data_prev_list[-1] = self.data_prediction_from_noise(x, x, t)
                f_prev_list[-1] = self.f_prediction_from_noise(x, x, t)
            elif afs == "average_x0":
                data_prev_list[-1] = self.average_x0.to(device)
                noise_prev_list[-1] = self.noise_prediction_from_data(x, data_prev_list[-1], t)
                f_prev_list[-1] = self.f_prediction_from_noise(x, noise_prev_list[-1], t)
            else:
                raise ValueError(f"Unsupported afs {afs}")
            
            if self.correcting_xt_fn is not None:
                x = self.correcting_xt_fn(x, t, step)
            x_prev_list[-1] = x
            
                
            if return_intermediate:
                intermediates.append(x)
                
            for step in range(steps):
                # get all decisions of the current step
                t = timesteps[step + 1]
                taylor_order = orders[step]
                prediction_type = prediction_types[step]
                start_point = start_points[step]
                derivative_type = derivative_types[step]
                corrector_type = correctors[step] 
                
                # get model list
                assert prediction_type in ["noise_prediction", "data_prediction", "dpmsolver_v3_prediction"]
                model_prev_list = data_prev_list if prediction_type == "data_prediction" else noise_prev_list if prediction_type == "noise_prediction" else f_prev_list
                
                # update x
                x = self.update(x_prev_list, model_prev_list, t_prev_list, t, taylor_order, prediction_type, start_point, derivative_type)
                if self.correcting_xt_fn is not None:
                    x = self.correcting_xt_fn(x, t, step + 1)
                if return_intermediate:
                    intermediates.append(x)
                
                # update the list of previous information
                for i in range(len(t_prev_list) - 1):
                    t_prev_list[i] = t_prev_list[i + 1]
                    noise_prev_list[i] = noise_prev_list[i + 1]
                    data_prev_list[i] = data_prev_list[i + 1]
                    x_prev_list[i] = x_prev_list[i + 1]
                    f_prev_list[i] = f_prev_list[i + 1]
                x_prev_list[-1] = x
                t_prev_list[-1] = t
                if corrector_type["type"] == "no_corrector":
                    if step != steps - 1:
                        noise_prev_list[-1], data_prev_list[-1] = self.model_fn(x, t)
                        f_prev_list[-1] = self.f_prediction_from_noise(x, noise_prev_list[-1], t)
                else: # use corrector
                    noise_prev_list[-1], data_prev_list[-1] = self.model_fn(x, t)
                    f_prev_list[-1] = self.f_prediction_from_noise(x, noise_prev_list[-1], t)
                    
                    model_prev_list = data_prev_list if prediction_type == "data_prediction" else noise_prev_list if prediction_type == "noise_prediction" else f_prev_list
                    correct_start_point, correct_taylor_order, correct_derivative_types = corrector_type["start_point"], corrector_type["taylor_order"], corrector_type["derivative_type"]

                    x = self.update(x_prev_list, model_prev_list, t_prev_list, t, correct_taylor_order, prediction_type, correct_start_point, correct_derivative_types)
                    x_prev_list[-1] = x
                    if corrector_type == "implicit" and step != steps - 1:
                        noise_prev_list[-1], data_prev_list[-1] = self.model_fn(x, t)
                        f_prev_list[-1] = self.f_prediction_from_noise(x, noise_prev_list[-1], t)


            if denoise_to_zero:
                t = torch.ones((1,)).to(device) * t_0
                x = self.denoise_to_zero_fn(x, t)
                if self.correcting_xt_fn is not None:
                    x = self.correcting_xt_fn(x, t, step + 1)
                if return_intermediate:
                    intermediates.append(x)
        # delete the temporary file
        if decisions.get("skip_coefficients", None) is not None:
            os.remove(f"./temp/skip_coefficients_{pid}.pth")
        if return_intermediate:
            return x, intermediates, t_prev_list, noise_prev_list, data_prev_list
        else:
            return x

    def relax_derivative(self, derivative, relaxation_type, h, relaxation_coefficient=0):
        if relaxation_type == "dpmsolver-2":
            derivative *= (0.5 * h * torch.expm1(h)) / (torch.expm1(h) - h)
        elif relaxation_type == "unipc-bh1":
            derivative *= (0.5 * h * h) / (torch.expm1(h) - h)
        elif relaxation_type == "unipc-bh2":
            derivative *= (0.5 * h * torch.expm1(h)) / (torch.expm1(h) - h) # The same as 'dpmsolver-2' relaxation
        elif relaxation_type == "0.5exp":
            derivative *= 1 + 0.5 * torch.expm1(h)
        elif relaxation_type == "-0.5exp":
            derivative *= 1 - 0.5 * torch.expm1(h)
        elif relaxation_type == "linear":
            derivative *= 1 + relaxation_coefficient * h
        else:
            raise ValueError(f"Unsupported relaxation type {relaxation_type}")
        return derivative
        

    # DPM-Solver-v3 functions
    def statistics_init_torch(self, statistics_dir, degenerated=False, t_start=1, t_end=1e-4):
        assert statistics_dir is not None, "The statistics_dir must be provided for DPM-Solver-v3."
        l = np.load(os.path.join(statistics_dir, "l.npz"))["l"]
        sb = np.load(os.path.join(statistics_dir, "sb.npz"))
        s, b = sb["s"], sb["b"]
        if degenerated:
            l = np.ones_like(l)
            s = np.zeros_like(s)
            b = np.zeros_like(b)
        self.statistics_steps = l.shape[0] - 1
        timesteps = self.get_time_steps("logSNR", t_start, t_end, self.statistics_steps, "cpu")
        ts = self.noise_schedule.marginal_lambda(timesteps).numpy()[:, None, None, None]
        self.ts = torch.from_numpy(ts).cuda()
        self.lambda_T = ts[0].item()
        self.lambda_0 = ts[-1].item()
        l = torch.from_numpy(l).to(self.device)
        s = torch.from_numpy(s).to(self.device)
        b = torch.from_numpy(b).to(self.device)
        ts = torch.from_numpy(ts).to(self.device)
        z = torch.zeros_like(l)
        o = torch.ones_like(l)
        L = weighted_cumsumexp_trapezoid_torch(z, ts, l)
        S = weighted_cumsumexp_trapezoid_torch(z, ts, s)

        I = weighted_cumsumexp_trapezoid_torch(L + S, ts, o)
        B = weighted_cumsumexp_trapezoid_torch(-S, ts, b)
        C = weighted_cumsumexp_trapezoid_torch(L + S, ts, B)
        self.l = l
        self.s = s
        self.b = b
        self.L = L
        self.S = S
        self.I = I
        self.B = B
        self.C = C


    def statistics_init(self, statistics_dir, degenerated=False, t_start=1, t_end=1e-4):
        assert statistics_dir is not None, "The statistics_dir must be provided for DPM-Solver-v3."
        l = np.load(os.path.join(statistics_dir, "l.npz"))["l"]
        sb = np.load(os.path.join(statistics_dir, "sb.npz"))
        s, b = sb["s"], sb["b"]
        if degenerated:
            l = np.ones_like(l)
            s = np.zeros_like(s)
            b = np.zeros_like(b)
        self.statistics_steps = l.shape[0] - 1
        timesteps = self.get_time_steps("logSNR", t_start, t_end, self.statistics_steps, "cpu")
        ts = self.noise_schedule.marginal_lambda(timesteps).numpy()[:, None, None, None]
        self.ts = torch.from_numpy(ts).cuda()
        self.lambda_T = ts[0].item()
        self.lambda_0 = ts[-1].item()
        z = np.zeros_like(l)
        o = np.ones_like(l)
        L = weighted_cumsumexp_trapezoid(z, ts, l)
        S = weighted_cumsumexp_trapezoid(z, ts, s)

        I = weighted_cumsumexp_trapezoid(L + S, ts, o)
        B = weighted_cumsumexp_trapezoid(-S, ts, b)
        C = weighted_cumsumexp_trapezoid(L + S, ts, B)
        self.l = torch.from_numpy(l).cuda()
        self.s = torch.from_numpy(s).cuda()
        self.b = torch.from_numpy(b).cuda()
        self.L = torch.from_numpy(L).cuda()
        self.S = torch.from_numpy(S).cuda()
        self.I = torch.from_numpy(I).cuda()
        self.B = torch.from_numpy(B).cuda()
        self.C = torch.from_numpy(C).cuda()

    def get_g(self, f_t, i_s, i_t):
        return torch.exp(self.S[i_s] - self.S[i_t]) * f_t - torch.exp(self.S[i_s]) * (self.B[i_t] - self.B[i_s])
    
    def get_g_prev_list(self, index_s, model_prev_list, index_prev_list):
        assert len(model_prev_list) == len(index_prev_list)
        g_prev_list = []
        for f_t,index_t in zip(model_prev_list,index_prev_list):
            if f_t is None or index_t is None:
                g_prev_list.append(None)
            else:
                g_prev_list.append(self.get_g(f_t, index_s, index_t))
        return g_prev_list

    def f_prediction_from_noise(self, x, noise, t):
        sigma_t = self.noise_schedule.marginal_std(t)
        index = self.get_index(t)
        l_t = self.l[index]
        alpha_t = self.noise_schedule.marginal_alpha(t)
        f_t = (sigma_t*noise - l_t *x) / alpha_t
        return f_t
        
    def get_index(self, t):
        logSNR = self.noise_schedule.marginal_lambda(t)
        index = int((self.statistics_steps*(logSNR - self.lambda_T)/(self.lambda_0 - self.lambda_T)).round().cpu().numpy().astype(np.int64))
        return index

    def get_indexes(self, timesteps):
        indexes = []
        for t in timesteps:
            indexes.append(self.get_index(t) if t is not None else None)
        return indexes
            
    def convert_to_ems_indexes(self, timesteps):
        logSNR_steps = self.noise_schedule.marginal_lambda(timesteps)
        indexes = list(
            (self.statistics_steps * (logSNR_steps - self.lambda_T) / (self.lambda_0 - self.lambda_T))
            .round()
            .cpu()
            .numpy()
            .astype(np.int64)
        )
        return indexes

    def convert_to_timesteps(self, indexes, device):
        logSNR_steps = (
            self.lambda_T + (self.lambda_0 - self.lambda_T) * torch.Tensor(indexes).to(device) / self.statistics_steps
        )
        return self.noise_schedule.inverse_lambda(logSNR_steps)

    def compute_exponential_coefficients_high_order(self, i_s, i_t, order=1):
        key = (i_s, i_t, order)
        if key in self.exp_coeffs.keys():
            coeffs = self.exp_coeffs[key]
        else:
            n = order
            a = self.L[i_s : i_t + 1] + self.S[i_s : i_t + 1] - self.L[i_s] - self.S[i_s]
            x = self.ts[i_s : i_t + 1]
            b = (self.ts[i_s : i_t + 1] - self.ts[i_s]) ** n / math.factorial(n)
            coeffs = weighted_cumsumexp_trapezoid_torch(a, x, b, cumsum=False)
            self.exp_coeffs[key] = coeffs
        return coeffs
#############################################################
# other utility functions
#############################################################

def interpolate_fn(x, xp, yp):
    """
    A piecewise linear function y = f(x), using xp and yp as keypoints.
    We implement f(x) in a differentiable way (i.e. applicable for autograd).
    The function f(x) is well-defined for all x-axis. (For x beyond the bounds of xp, we use the outmost points of xp to define the linear function.)

    Args:
        x: PyTorch tensor with shape [N, C], where N is the batch size, C is the number of channels (we use C = 1 for DPM-Solver).
        xp: PyTorch tensor with shape [C, K], where K is the number of keypoints.
        yp: PyTorch tensor with shape [C, K].
    Returns:
        The function values f(x), with shape [N, C].
    """
    N, K = x.shape[0], xp.shape[1]
    all_x = torch.cat([x.unsqueeze(2), xp.unsqueeze(0).repeat((N, 1, 1))], dim=2)
    sorted_all_x, x_indices = torch.sort(all_x, dim=2)
    x_idx = torch.argmin(x_indices, dim=2)
    cand_start_idx = x_idx - 1
    start_idx = torch.where(
        torch.eq(x_idx, 0),
        torch.tensor(1, device=x.device),
        torch.where(
            torch.eq(x_idx, K), torch.tensor(K - 2, device=x.device), cand_start_idx,
        ),
    )
    end_idx = torch.where(torch.eq(start_idx, cand_start_idx), start_idx + 2, start_idx + 1)
    start_x = torch.gather(sorted_all_x, dim=2, index=start_idx.unsqueeze(2)).squeeze(2)
    end_x = torch.gather(sorted_all_x, dim=2, index=end_idx.unsqueeze(2)).squeeze(2)
    start_idx2 = torch.where(
        torch.eq(x_idx, 0),
        torch.tensor(0, device=x.device),
        torch.where(
            torch.eq(x_idx, K), torch.tensor(K - 2, device=x.device), cand_start_idx,
        ),
    )
    y_positions_expanded = yp.unsqueeze(0).expand(N, -1, -1)
    start_y = torch.gather(y_positions_expanded, dim=2, index=start_idx2.unsqueeze(2)).squeeze(2)
    end_y = torch.gather(y_positions_expanded, dim=2, index=(start_idx2 + 1).unsqueeze(2)).squeeze(2)
    cand = start_y + (x - start_x) * (end_y - start_y) / (end_x - start_x)
    return cand


def expand_dims(v, dims):
    """
    Expand the tensor `v` to the dim `dims`.

    Args:
        `v`: a PyTorch tensor with shape [N].
        `dim`: a `int`.
    Returns:
        a PyTorch tensor with shape [N, 1, 1, ..., 1] and the total dimension is `dims`.
    """
    return v[(...,) + (None,)*(dims - 1)]

# DPM-solver_v3 utility functions

def weighted_cumsumexp_trapezoid(a, x, b, cumsum=True):
    # ∫ b*e^a dx
    # Input: a,x,b: shape (N+1,...)
    # Output: y: shape (N+1,...)
    # y_0 = 0
    # y_n = sum_{i=1}^{n} 0.5*(x_{i}-x_{i-1})*(b_{i}*e^{a_{i}}+b_{i-1}*e^{a_{i-1}}) (n from 1 to N)

    assert x.shape[0] == a.shape[0] and x.ndim == a.ndim
    if b is not None:
        assert a.shape[0] == b.shape[0] and a.ndim == b.ndim

    a_max = np.amax(a, axis=0, keepdims=True)

    if b is not None:
        b = np.asarray(b)
        tmp = b * np.exp(a - a_max)
    else:
        tmp = np.exp(a - a_max)

    out = 0.5 * (x[1:] - x[:-1]) * (tmp[1:] + tmp[:-1])
    if not cumsum:
        return np.sum(out, axis=0) * np.exp(a_max)
    out = np.cumsum(out, axis=0)
    out *= np.exp(a_max)
    return np.concatenate([np.zeros_like(out[[0]]), out], axis=0)


def weighted_cumsumexp_trapezoid_torch(a, x, b, cumsum=True):
    assert x.shape[0] == a.shape[0] and x.ndim == a.ndim
    if b is not None:
        assert a.shape[0] == b.shape[0] and a.ndim == b.ndim

    a_max = torch.amax(a, dim=0, keepdims=True)

    if b is not None:
        tmp = b * torch.exp(a - a_max)
    else:
        tmp = torch.exp(a - a_max)

    out = 0.5 * (x[1:] - x[:-1]) * (tmp[1:] + tmp[:-1])
    if not cumsum:
        return torch.sum(out, dim=0) * torch.exp(a_max)
    out = torch.cumsum(out, dim=0)
    out *= torch.exp(a_max)
    return torch.concat([torch.zeros_like(out[[0]]), out], dim=0)

def index_list(lst, index):
    new_lst = []
    for i in index:
        new_lst.append(lst[i])
    return new_lst