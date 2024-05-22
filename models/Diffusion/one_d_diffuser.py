import torch.nn as nn
import torch
import numpy as np
from functools import partial
from .block import default
from utils.monitoring import plot_img

def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def noise_like(shape, device, repeat=False):
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()

def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    if schedule == "linear":
        betas = (
                torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64) ** 2
        )

    elif schedule == "cosine":
        timesteps = (
                torch.arange(n_timestep + 1, dtype=torch.float64) / n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * np.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = np.clip(betas, a_min=0, a_max=0.999)

    elif schedule == "sqrt_linear":
        betas = torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64)
    elif schedule == "sqrt":
        betas = torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64) ** 0.5
    else:
        raise ValueError(f"schedule '{schedule}' unknown.")
    return betas.numpy()


class DDPM_ELBO(nn.Module):
    def __init__(self, nn_model, betas, n_T, device, clip_denoised=False, drop_prob=0.1):
        super(DDPM_ELBO, self).__init__()

        # register_buffer allows accessing dictionary produced by ddpm_schedules
        # e.g. can access self.sqrtab later

        self.v_posterior = 0
        self.original_elbo_weight = 0

        self.model = nn_model.to(device)
        self.n_T = n_T
        self.device = device
        self.drop_prob = drop_prob
        self.clip_denoised = clip_denoised

        self.register_schedule(betas_range=betas, timesteps=n_T)

    def get_loss(self, pred, target, mean=True):
        if mean:
            loss = torch.nn.functional.mse_loss(target, pred)
        else:
            loss = torch.nn.functional.mse_loss(target, pred, reduction='none')

        return loss

    def p_losses(self, x_start, t, c=None, noise=None, guidance_mask=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_out = self.model(x_noisy, t, proto=c, guidance_mask=guidance_mask)

        target = noise
        loss = self.get_loss(model_out, target, mean=False)
        if len(loss.size()) == 4:
            loss = loss.mean(dim=[1,2,3])
        elif len(loss.size()) == 2:
            loss = loss.mean(dim=[1])
        else:
            raise Exception("data should be either 4 or 2 dimensional")
        loss_simple = loss.mean()

        loss_vlb = (self.lvlb_weights[t] * loss).mean()

        loss = loss_simple + self.original_elbo_weight * loss_vlb
        return loss

    def forward(self, x, c=None):
        """
        this method is used in training, so samples t and noise randomly
        """
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
        #t = torch.linspace(0, self.num_timesteps, x.shape[0]).to(self.device).long()
        if self.drop_prob != 0:
            #
            if len(c.size()) == 4:
                guidance_mask = torch.bernoulli(torch.zeros_like(c[:, 0, 0, 0]) + self.drop_prob).to(self.device)
            elif len(c.size()) == 2:
                guidance_mask = torch.bernoulli(torch.zeros_like(c[:, 0]) + self.drop_prob).to(self.device)
        else:
            guidance_mask = None

        return self.p_losses(x, t, c=c, guidance_mask=guidance_mask)

    def test_noise(self, x_start, nb_step, noise=None):
        t = torch.linspace(0, self.num_timesteps-1, nb_step).to(self.device).long()
        x_start = x_start.repeat(nb_step, 1)
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        return x_noisy

    def register_schedule(self, betas_range, timesteps, beta_schedule="linear"):

        linear_start = betas_range[0]
        linear_end = betas_range[1]
        betas = make_beta_schedule(schedule=beta_schedule, n_timestep=timesteps, linear_start=linear_start,
                                   linear_end=linear_end)
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert alphas_cumprod.shape[0] == self.num_timesteps, 'alphas have to be defined for each timestep'

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (1 - self.v_posterior) * betas * (1. - alphas_cumprod_prev) / (
                1. - alphas_cumprod) + self.v_posterior * betas
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

        lvlb_weights = self.betas ** 2 / (2 * self.posterior_variance * to_torch(alphas) * (1 - self.alphas_cumprod))
        lvlb_weights[0] = lvlb_weights[1]
        #self.register_buffer('lvlb_weights', lvlb_weights, persistent=False)
        self.register_buffer('lvlb_weights', lvlb_weights)
        assert not torch.isnan(self.lvlb_weights).all()

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def predict_start_from_noise(self, x_t, t, noise):
        return (
                extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def p_mean_variance(self, x, t,clip_denoised, c=None, guidance_mask=None, w_guidance=1):
        bs = x.size(0)
        if self.drop_prob != 0:
            # double batch
            if len(x.size()) == 4:
                x = x.repeat(2, 1, 1, 1)
            elif len(x.size()) == 2:
                x = x.repeat(2, 1)
            else:
                raise NotImplementedError
            # t_is = t_is.repeat(2, 1, 1, 1)
            t = t.repeat(2)
        model_out = self.model(x, t, proto=c, guidance_mask=guidance_mask)
        if self.drop_prob != 0:
            x = x[:bs]
            t = t[:bs]
            eps1 = model_out[:bs]
            eps2 = model_out[bs:]
            eps = (1 + w_guidance) * eps1 - w_guidance * eps2
        else:
            eps = model_out
        x_recon = self.predict_start_from_noise(x, t=t, noise=eps)
        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    def p_mean_variance_with_guidance_score(self, x, t,clip_denoised, c=None, guidance_mask=None, w_guidance=1):
        bs = x.size(0)
        if self.drop_prob != 0:
            # double batch
            if len(x.size()) == 4:
                x = x.repeat(2, 1, 1, 1)
            elif len(x.size()) == 2:
                x = x.repeat(2, 1)
            else:
                raise NotImplementedError
            # t_is = t_is.repeat(2, 1, 1, 1)
            t = t.repeat(2)
        model_out = self.model(x, t, proto=c, guidance_mask=guidance_mask)
        if self.drop_prob != 0:
            x = x[:bs]
            t = t[:bs]
            eps1 = model_out[:bs]
            eps2 = model_out[bs:]
            eps = (1 + w_guidance) * eps1 - w_guidance * eps2
        else:
            eps = model_out
        x_recon = self.predict_start_from_noise(x, t=t, noise=eps)
        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance, eps1, eps2

    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True, repeat_noise=False, c=None, guidance_mask=None, w_guidance=1, return_guidance_score=False):
        b, *_, device = *x.shape, x.device
        if return_guidance_score:
            model_mean, _, model_log_variance, cond_score, uncond_score = self.p_mean_variance_with_guidance_score(x=x, t=t, clip_denoised=clip_denoised, c=c,
                                                                     guidance_mask=guidance_mask,
                                                                     w_guidance=w_guidance)
        else:
            model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, clip_denoised=clip_denoised, c=c,
                                                                 guidance_mask = guidance_mask,
                                                                 w_guidance = w_guidance)
        noise = noise_like(x.shape, device, repeat_noise)

        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        if return_guidance_score:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise, cond_score, uncond_score
        else:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return ((extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape)) * x_start +
                (extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)) * noise)

    @torch.no_grad()
    def sample_c(self, image_size, batch_size, cond=None, w_guidance=1):
        device = self.device
        shape = (batch_size, *image_size)
        b = batch_size
        img = torch.randn(shape, device=device)
        if self.drop_prob != 0:
            c_i = cond
            if len(cond.size()) == 4:
                guidance_mask = torch.zeros_like(c_i[:, 0, 0, 0]).to(self.device)
                c_i = c_i.repeat(2, 1, 1, 1)
            elif len(cond.size()) == 2:
                guidance_mask = torch.zeros_like(c_i[:, 0]).to(self.device)
                c_i = c_i.repeat(2, 1)
            else:
                raise NotImplementedError()
            guidance_mask = guidance_mask.repeat(2)
            guidance_mask[batch_size:] = 1.
        else:
            guidance_mask = None
            c_i = cond

        for i in reversed(range(0, self.n_T)):

            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long),
                                c=c_i,
                                guidance_mask=guidance_mask,
                                clip_denoised=self.clip_denoised,
                                w_guidance=w_guidance)
        return img

    @torch.no_grad()
    def partial_sample_c(self, image, cond=None, nb_it=300, w_guidance=1):
        device = self.device
        shape = image.size()
        b = shape[0]
        t = (torch.ones(b)*nb_it).long().to(device)
        img = self.q_sample(image, t)
        noisy_latent = img.clone()
        if self.drop_prob != 0:
            c_i = cond
            if len(cond.size()) == 4:
                guidance_mask = torch.zeros_like(c_i[:, 0, 0, 0]).to(self.device)
                c_i = c_i.repeat(2, 1, 1, 1)
            elif len(cond.size()) == 2:
                guidance_mask = torch.zeros_like(c_i[:, 0]).to(self.device)
                c_i = c_i.repeat(2, 1)
            else:
                raise NotImplementedError()
            guidance_mask = guidance_mask.repeat(2)
            guidance_mask[b:] = 1.
        else:
            guidance_mask = None
            c_i = cond

        for i in reversed(range(0, nb_it)):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long),
                                c=c_i,
                                guidance_mask=guidance_mask,
                                w_guidance=w_guidance,
                                clip_denoised=self.clip_denoised)

        return img, noisy_latent

    @torch.no_grad()
    def compute_guidance_score(self, image_size, batch_size, cond=None, w_guidance=1):
        device = self.device
        shape = (batch_size, *image_size)
        b = batch_size
        img = torch.randn(shape, device=device)
        all_cond_score = []
        all_uncond_score = []
        all_img = []
        total_score = []
        if self.drop_prob != 0:
            c_i = cond
            if len(cond.size()) == 4:
                guidance_mask = torch.zeros_like(c_i[:, 0, 0, 0]).to(self.device)
                c_i = c_i.repeat(2, 1, 1, 1)
            elif len(cond.size()) == 2:
                guidance_mask = torch.zeros_like(c_i[:, 0]).to(self.device)
                c_i = c_i.repeat(2, 1)
            else:
                raise NotImplementedError()
            guidance_mask = guidance_mask.repeat(2)
            guidance_mask[batch_size:] = 1.
        else:
            guidance_mask = None
            c_i = cond

        for i in reversed(range(0, self.n_T)):
            img, cond_score, uncond_score = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long),
                                c=c_i,
                                guidance_mask=guidance_mask,
                                clip_denoised=self.clip_denoised,
                                w_guidance=w_guidance,
                                return_guidance_score=True)
            all_img.append(img)
            all_cond_score.append(cond_score)
            all_uncond_score.append(uncond_score)
        all_cond_score = torch.stack(all_cond_score, dim=0)
        all_uncond_score = torch.stack(all_uncond_score, dim=0)
        all_img = torch.stack(all_img, dim=0)
        return img, all_cond_score, all_uncond_score, all_img