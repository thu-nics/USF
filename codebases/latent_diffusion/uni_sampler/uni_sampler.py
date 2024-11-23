import torch
from .USF import usf, NoiseScheduleVP, model_wrapper

class Uni_Sampler:
    def __init__(self, statistics_dir, model, guidance_scale, args=None):
        super().__init__()
        self.model = model
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(model.device)
        self.alphas_cumprod = to_torch(model.alphas_cumprod)
        self.device = self.model.betas.device
        self.guidance_scale = guidance_scale
        self.args = args
        self.ns = NoiseScheduleVP("discrete", alphas_cumprod=self.alphas_cumprod)
        self.usf = usf(
            self.ns,
            device=self.device,
            statistics_dir=statistics_dir
        )

    @torch.no_grad()
    def sample(
        self,
        batch_size,
        shape,
        conditioning=None,
        x_T=None,
        unconditional_conditioning=None,
        decisions=None,
        # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
        **kwargs
    ):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)

        if x_T is None:
            img = torch.randn(size, device=self.device)
        else:
            img = x_T

        if conditioning is None:
            model_fn = model_wrapper(
                lambda x, t, c: self.model.apply_model(x, t, c),
                self.ns,
                model_type="noise",
                guidance_type="uncond",
            )

        else:
            model_fn = model_wrapper(
                lambda x, t, c: self.model.apply_model(x, t, c),
                self.ns,
                model_type="noise",
                guidance_type="classifier-free",
                condition=conditioning,
                unconditional_condition=unconditional_conditioning,
                guidance_scale=self.guidance_scale,
            )
        from uni_sampler.utils import print_decisions
        if decisions is None:
            from uni_sampler.utils import print_decisions
            if self.args.load_decision is not None:
                self.decisions = torch.load(self.args.load_decision)
                if "decisions" in self.decisions.keys():
                    self.decisions = self.decisions["decisions"]
            else: # no decision, no load decision, generate decision
                from uni_sampler.utils import get_empirical_decisions
                self.decisions = get_empirical_decisions(self.args, self.usf, self.device)
                # self.logger.info("using empirical decisions:")
                # print_decisions(self.logger, self.decisions)
        else:
            self.decisions = decisions

        x = self.usf.sample(
            img,
            model_fn,
            self.decisions,
            steps=(self.args.steps - 1 if self.args.denoise else self.args.steps),
            order=self.args.uni_sampler_order,
            skip_type=self.args.skip_type,
            method=self.args.uni_sampler_method,
            lower_order_final=self.args.lower_order_final,
            denoise_to_zero=self.args.denoise,
            return_intermediate=self.args.return_intermediate,
        )
        
        return x, None
    
    def get_empirical_decisions(self, args):
        from uni_sampler.utils import get_empirical_decisions
        decisions = get_empirical_decisions(args, self.usf, self.device)
        return decisions