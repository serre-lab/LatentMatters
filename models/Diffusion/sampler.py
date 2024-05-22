import torch.nn as nn
import torch
import numpy as np
from functools import partial
from .block import default

def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def noise_like(shape, device, repeat=False):
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()

def ddpm_schedules(beta1, beta2, T):
    """
    Returns pre-computed schedules for DDPM sampling, training process.
    """
    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

    beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1
    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

    sqrtab = torch.sqrt(alphabar_t)
    oneover_sqrta = 1 / torch.sqrt(alpha_t)

    sqrtmab = torch.sqrt(1 - alphabar_t)
    mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

    return {
        "alpha_t": alpha_t,  # \alpha_t
        "oneover_sqrta": oneover_sqrta,  # 1/\sqrt{\alpha_t}
        "sqrt_beta_t": sqrt_beta_t,  # \sqrt{\beta_t}
        "alphabar_t": alphabar_t,  # \bar{\alpha_t}
        "sqrtab": sqrtab,  # \sqrt{\bar{\alpha_t}}
        "sqrtmab": sqrtmab,  # \sqrt{1-\bar{\alpha_t}}
        "mab_over_sqrtmab": mab_over_sqrtmab_inv,  # (1-\alpha_t)/\sqrt{1-\bar{\alpha_t}}
    }


class DDPM(nn.Module):
    def __init__(self, nn_model, betas, n_T, device, drop_prob=0.1, loss_type='MSE'):
        super(DDPM, self).__init__()
        self.nn_model = nn_model.to(device)

        # register_buffer allows accessing dictionary produced by ddpm_schedules
        # e.g. can access self.sqrtab later
        for k, v in ddpm_schedules(betas[0], betas[1], n_T).items():
            self.register_buffer(k, v)

        self.n_T = n_T
        self.device = device
        self.drop_prob = drop_prob
        if loss_type == 'MSE':
            self.loss_mse = nn.MSELoss()


    def forward(self, x, c):
        """
        this method is used in training, so samples t and noise randomly
        """

        _ts = torch.randint(1, self.n_T, (x.shape[0],)).to(self.device).long()  # t ~ Uniform(0, n_T)
        noise = torch.randn_like(x)  # eps ~ N(0, 1)

        x_t = (
                self.sqrtab[_ts, None, None, None] * x
                + self.sqrtmab[_ts, None, None, None] * noise
        )  # This is the x_t, which is sqrt(alphabar) x_0 + sqrt(1-alphabar) * eps
        # We should predict the "error term" from this x_t. Loss is what we return.

        # dropout context with some probability
        if self.nn_model.embedding_model is not None and self.nn_model.embedding_model != 'stack':
            c = self.nn_model.embedding_model(c)
            context_mask = torch.bernoulli(torch.zeros_like(c[:, 0]) + self.drop_prob).to(self.device)
        elif self.nn_model.embedding_model == 'stack':
            if self.drop_prob != 0:
                context_mask = torch.bernoulli(torch.zeros_like(c[:, 0, 0, 0]) + self.drop_prob).to(self.device)
            else:
                context_mask = None
        else:
            c, context_mask = None, None

        # return MSE between added noise, and our predicted noise
        return self.loss_mse(noise, self.nn_model(x_t, (_ts / self.n_T).float(), c=c, context_mask=context_mask))

    @torch.no_grad()
    def sample_c(self, image_size, batch_size, cond=None, guide_w=1.0):
        # we follow the guidance sampling scheme described in 'Classifier-Free Diffusion Guidance'
        # to make the fwd passes efficient, we concat two versions of the dataset,
        # one with context_mask=0 and the other context_mask=1
        # we then mix the outputs with the guidance scale, w
        # where w>0 means more guidance
        size = (batch_size, *image_size)
        x_i = torch.randn(*size).to(self.device)  # x_T ~ N(0, 1), sample initial noise
        if self.nn_model.embedding_model is not None and self.nn_model.embedding_model != 'stack':
            c_i = self.nn_model.embedding_model(cond)
            context_mask = torch.zeros_like(c_i[:, 0]).to(self.device)
            c_i = c_i.repeat(2, 1)
            context_mask = context_mask.repeat(2)
            context_mask[batch_size:] = 1.
        elif self.nn_model.embedding_model is not None and self.nn_model.embedding_model == 'stack':
            if self.drop_prob != 0:
                c_i = cond
                context_mask = torch.zeros_like(c_i[:, 0, 0, 0]).to(self.device)
                c_i = c_i.repeat(2, 1, 1, 1)
                context_mask = context_mask.repeat(2)
                context_mask[batch_size:] = 1.
            else:
                context_mask = None
                c_i = cond

        else:
            c_i, context_mask = None, None

        # don't drop context at test time

        for i in range(self.n_T, 0, -1):
            print(f'sampling timestep {i}', end='\r')
            t_is = torch.tensor([i / self.n_T]).to(self.device)
            # t_is = t_is.repeat(batch_size, 1, 1, 1)

            t_is = t_is.repeat(batch_size)

            if self.drop_prob != 0:
                # double batch
                x_i = x_i.repeat(2, 1, 1, 1)
                # t_is = t_is.repeat(2, 1, 1, 1)
                t_is = t_is.repeat(2)

            z = torch.randn(*size).to(self.device) if i > 1 else 0
            # split predictions and compute weighting
            eps = self.nn_model(x_i, t_is, c=c_i, context_mask=context_mask)

            if self.drop_prob != 0:

                eps1 = eps[:batch_size]
                eps2 = eps[batch_size:]
                eps = (1 + guide_w) * eps1 - guide_w * eps2

                # eps1 = eps[:batch_size]
                # eps = (1 + guide_w) * eps1
                # eps2 = eps[batch_size:]
                # eps -= guide_w * eps2
                x_i = x_i[:batch_size]
            else:
                eps1 = eps[:batch_size]
                eps = eps1
                x_i = x_i[:batch_size]
            x_i = (
                    self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
                    + self.sqrt_beta_t[i] * z
            )
        return x_i


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
    def __init__(self, nn_model, betas, n_T, device, clip_denoised=True, drop_prob=0.1):
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
        model_out = self.model(x_noisy, t, c, guidance_mask=guidance_mask)

        target = noise
        loss = self.get_loss(model_out, target, mean=False).mean(dim=[1, 2, 3])

        loss_simple = loss.mean()

        loss_vlb = (self.lvlb_weights[t] * loss).mean()

        loss = loss_simple + self.original_elbo_weight * loss_vlb
        return loss

    def forward(self, x, c):
        """
        this method is used in training, so samples t and noise randomly
        """
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()

        if self.drop_prob != 0:
            guidance_mask = torch.bernoulli(torch.zeros_like(c[:, 0, 0, 0]) + self.drop_prob).to(self.device)
        else:
            guidance_mask = None

        return self.p_losses(x, t, c, guidance_mask=guidance_mask)

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

    def p_mean_variance(self, x, t, c, clip_denoised, guidance_mask=None, guide_w=1):
        bs = x.size(0)
        if self.drop_prob != 0:
            # double batch
            x = x.repeat(2, 1, 1, 1)
            # t_is = t_is.repeat(2, 1, 1, 1)
            t = t.repeat(2)
        model_out = self.model(x, t, c, guidance_mask=guidance_mask)
        if self.drop_prob != 0:
            x = x[:bs]
            t = t[:bs]
            eps1 = model_out[:bs]
            eps2 = model_out[bs:]
            eps = (1 + guide_w) * eps1 - guide_w * eps2
        else:
            eps = model_out
        x_recon = self.predict_start_from_noise(x, t=t, noise=eps)
        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, c, clip_denoised=True, repeat_noise=False, guidance_mask=None):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, c=c, clip_denoised=clip_denoised, guidance_mask=guidance_mask)
        noise = noise_like(x.shape, device, repeat_noise)

        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return ((extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape)) * x_start +
                (extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)) * noise)

    @torch.no_grad()
    def sample_c(self, image_size, batch_size, cond=None):
        device = self.device
        shape = (batch_size, *image_size)
        b = batch_size
        img = torch.randn(shape, device=device)
        if self.drop_prob != 0:
            c_i = cond
            guidance_mask = torch.zeros_like(c_i[:, 0, 0, 0]).to(self.device)
            c_i = c_i.repeat(2, 1, 1, 1)
            guidance_mask = guidance_mask.repeat(2)
            guidance_mask[batch_size:] = 1.
        else:
            guidance_mask = None
            c_i = cond

        for i in reversed(range(0, self.n_T)):

            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long), c_i,
                                clip_denoised=self.clip_denoised, guidance_mask=guidance_mask)
        return img
