import torch.nn as nn
import torch
from einops import rearrange

class identity_enc(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, image):
        #return torch.flatten(image, start_dim=1)
        return image

class identity_dec(nn.Module):
    def __init__(self, image_size=(1, 48,48)):
        super().__init__()
        self.image_size = image_size

    def forward(self, image):
        #return image.view(-1, *self.image_size)
        return image

def soft_prob(dist, smooth):
    prob = torch.exp(-dist*0.5*smooth)/torch.sqrt(smooth)
    #prob = torch.exp(-dist**2/ (smooth**2))
    probs = prob/torch.sum(prob, dim=1, keepdim=True)
    #probs = prob/torch.sum(prob, dim=2, keepdim=True)
    return probs

class VectorQuantizer1d(nn.Module):
    def __init__(self,
                 latent_size,
                 nb_word,
                 word_size,
                 vq_reg=None,
                 commit_vq=2.5):
        super().__init__()

        self.latent_size = latent_size
        self.nb_word = nb_word # number of word in the dictionary
        self.word_size = word_size # size of each word in the dictionary
        self.embedding = nn.Embedding(self.nb_word, self.word_size)
        self.embedding.weight.data.uniform_(-1.0 / self.nb_word, 1.0 / self.nb_word)
        self.vq_reg = vq_reg
        self.commit_vq = commit_vq

    def forward(self, inputs):
        assert len(inputs) == 2, 'should have mean and logvar in the quantizer'
        z_mean, z_log_var = inputs[0], inputs[1]
        input_shape = z_mean.size()
        z = z_mean.view(-1, self.latent_size//self.word_size, self.word_size)
        z_flattened = z.view(-1, self.word_size)

        distances = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
                torch.sum(self.embedding.weight ** 2, dim=1) - 2 * \
                torch.einsum('bd,dn->bn', z_flattened, rearrange(self.embedding.weight, 'n d -> d n'))

        #distances = torch.sum(z_mean ** 2, dim=1, keepdim=True) + \
        #    torch.sum(self.embedding.weight ** 2, dim=1) - 2 * \
        #    torch.einsum('bd,dn->bn', z_mean, rearrange(self.embedding.weight, 'n d -> d n'))

        #distances = torch.sum(z_mean**2, dim=1, keepdim=True) \
        #            - 2*torch.matmul(z_mean, self.embedding.weight) \
        #            + torch.sum(self.embedding.weight**2, dim=0, keepdim=True)
        if self.vq_reg == 'soft_vq':
            #std = torch.exp(0.5 * z_log_var).unsqueeze(-1)
            z_log_var = torch.zeros_like(z_log_var)
            std = (1.0/z_log_var.exp()**2).unsqueeze(-1)
            #smooth = 1.0/z_log_var.exp()**2
            #probs = soft_prob(distances, smooth)
            original_size = distances.size()
            probs = soft_prob(distances.view(-1, self.latent_size, self.nb_word), std)
            probs = probs.view(original_size)
            #quantize_vect = torch.einsum('bd,dn->bn', probs, rearrange(self.embedding.weight, 'n d -> d n'))
            z_q = torch.einsum('bd,dn->bn', probs, self.embedding.weight).view(z.size())
            #probs = torch.unsqueeze(1)
            #codebook = self.embedding.weight.unsqueeze(0)
            #quantize_vect = torch.sum(code_book*probs, 2)

        elif self.vq_reg == 'normal_vq':
            min_encoding_indices = torch.argmin(distances, dim=1)
            z_q = self.embedding(min_encoding_indices).view(z.size())

        loss =  torch.mean((z_q.detach()-z)**2) + self.commit_vq * \
                   torch.mean((z_q - z.detach()) ** 2)
        z_q = z + (z_q - z).detach()
        return z_q.view(input_shape), loss

class AE(nn.Module):
    def __init__(self,
                 latent_size=128,
                 tanh=False,
                 k=0,
                 w_kl=0,
                 vq_reg=None,
                 word_size=4,
                 n_words=512,
                 commit_vq=2.5):
        super().__init__()

        self.tanh = tanh
        self.k = k
        self.n_words = n_words
        if self.k is None:
            self.k = 0
        self.w_kl = w_kl
        self.commit_vq = commit_vq
        self.vq_reg = vq_reg
        if self.vq_reg is not None:
            #self.quantize = VectorQuantizer1d(512, latent_size, vq_reg=vq_reg)
            self.quantize = VectorQuantizer1d(latent_size,
                                              nb_word=n_words,
                                              word_size=word_size,
                                              vq_reg=vq_reg,
                                              commit_vq=commit_vq)
        #self.encoder = nn.Identity()
        self.latent_size = latent_size
        self.encoder = identity_enc()
        #self.decoder = nn.Identity()
        self.decoder = identity_dec()

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        print('freezed')

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, image):
        latent = self.encode(image)
        loss = None
        if self.vq_reg is not None:
            quantized, loss = self.quantize(latent)
            latent = quantized
            decoded_img = self.decode(quantized)
        elif self.w_kl != 0 :
            decoded_img = self.decode(latent[-1])
        else:
            decoded_img = self.decode(latent)
        return latent, decoded_img, loss

    def encode(self, image):

        if self.w_kl != 0:
            q_mean, q_logvar = self.encoder(image)
            latent = (q_mean, q_logvar, self.reparametrize(q_mean, q_logvar))
        else:
            latent = self.encoder(image)

            if self.tanh:
                latent = torch.tanh(latent)
            if self.k != 0:
                values, indices = torch.kthvalue(-latent, self.k, keepdim=True)
                mask = latent.clone() < -values
                latent[mask] = 0

        return latent

    def decode(self, latent):
        decoded_img = self.decoder(latent)
        return decoded_img

    def scale(self, latent, mean=0.0, std=1.0, reverse=False):
        if not reverse:
            return (latent - mean)/std
        else:
            return (latent*std) + mean

    def estimate_stats(self, image):
        with torch.no_grad():
            latent = self.encode(image)
            if self.w_kl != 0:
                (_, _, latent) = latent
            if self.vq_reg is not None:
                latent, _ = self.quantize(latent)
        return latent.mean(), latent.std()

    def get_latent_from_img(self, image, mean, std, scale=True):
        latent = self.encode(image)
        if self.w_kl != 0:
            (_, _, latent) = latent
        if self.vq_reg is not None:
            latent, _ = self.quantize(latent)
        if scale:
            latent = self.scale(latent, mean=mean, std=std, reverse=False)
        return latent

    def get_img_from_latent(self, latent, mean, std, scale=True):
        if scale:
            latent = self.scale(latent, mean=mean, std=std, reverse=True)
        img = self.decode(latent)
        return img


class One_d_AE(AE):
    # Learned perceptual metric
    def __init__(self, latent_size=128, tanh=False, k=None):
        super().__init__(latent_size, tanh, k)

        self.encoder = nn.Sequential(
          nn.Linear(784, 512),
          nn.ReLU(),
          nn.Linear(512,256),
          nn.ReLU(),
          nn.Linear(256, self.latent_size),
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.latent_size, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 784),
            nn.Sigmoid()
        )

    def encode(self, image):
        image = torch.flatten(image, start_dim=1)
        latent = self.encoder(image)
        if self.tanh:
            latent = torch.tanh(latent)
        if self.k is not None:
            values, indices = torch.kthvalue(-latent, self.k, keepdim=True)
            mask = latent.clone() < -values
            latent[mask] = 0
        return latent

    def decode(self, latent):
        decoded_img = self.decoder(latent)
        decoded_img = decoded_img.view(-1, 1, 28, 28)
        return decoded_img


class RAE_28x28(AE):
    def __init__(self,
                 latent_size=128,
                 ae_dim=128,
                 tanh=False,
                 k=0,
                 use_batchnorm=True,
                 **kwargs):

        super().__init__(latent_size, tanh, k)
        self.encoder = RAEEncoder28x28(num_channels=1,
                                       ae_dim=ae_dim,
                                       z_dim=latent_size,
                                       use_batchnorm=use_batchnorm)

        self.decoder = RAEDecoder28x28(num_channels=1,
                                       ae_dim=ae_dim,
                                       z_dim=latent_size,
                                       use_batchnorm=use_batchnorm)


class RAEEncoder28x28(nn.Module):
    def __init__(
        self,
        num_channels,
        ae_dim=128,
        z_dim=128,
        use_batchnorm=True,
    ):
        super().__init__()

        self.dims = [
            ae_dim // (2**i) for i in reversed(range(4))
        ]  # 1 2 4 8 # 32 64 128 256

        self.dims[-1] = ae_dim
        self.use_batchnorm = use_batchnorm

        self.out_dim = self.dims[3] * 2 * 2
        self.z_dim = z_dim

        self.layers = nn.Sequential(
            nn.Conv2d(
                num_channels,
                self.dims[0],
                kernel_size=4,
                stride=2,
                padding=1,
                bias=not self.use_batchnorm,
            ),
            nn.BatchNorm2d(self.dims[0]) if self.use_batchnorm else nn.Identity(),
            nn.ReLU(),
            nn.Conv2d(
                self.dims[0],
                self.dims[1],
                kernel_size=4,
                stride=2,
                padding=1,
                bias=not self.use_batchnorm,
            ),
            nn.BatchNorm2d(self.dims[1]) if self.use_batchnorm else nn.Identity(),
            nn.ReLU(),
            nn.Conv2d(
                self.dims[1],
                self.dims[2],
                kernel_size=4,
                stride=2,
                padding=2,
                bias=not self.use_batchnorm,
            ),
            nn.BatchNorm2d(self.dims[2]) if self.use_batchnorm else nn.Identity(),
            nn.ReLU(),
            nn.Conv2d(
                self.dims[2],
                self.dims[3],
                kernel_size=4,
                stride=2,
                padding=1,
                bias=not self.use_batchnorm,
            ),
            nn.BatchNorm2d(self.dims[3]) if self.use_batchnorm else nn.Identity(),
            nn.ReLU(),
        )

        #self.q_mean = nn.Linear(self.out_dim, self.z_dim)
        #self.q_logvar = nn.Linear(self.out_dim, self.z_dim)
        self.to_latent = nn.Linear(self.out_dim, self.z_dim)

    def forward(self, x):
        out = self.layers(x)#.view(x[0].size(0), -1)
        out = out.view(out.size(0), -1)

        return self.to_latent(out)


class RAEDecoder28x28(nn.Module):
    def __init__(
        self,
        num_channels,
        z_dim=128,
        ae_dim=128,
        use_batchnorm=True,
    ):
        super().__init__()

        self.num_channels = num_channels
        self.dims = [ae_dim // (2**i) for i in reversed(range(3))]
        self.dims[-1] = ae_dim

        #self.kernel_dim = kernel_dim
        #self.padding = padding
        self.use_batchnorm = use_batchnorm
        #self.
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(
                z_dim,
                self.dims[2],
                kernel_size=7,
                stride=1,
                bias=not self.use_batchnorm,
            ),
            nn.BatchNorm2d(self.dims[2]) if self.use_batchnorm else nn.Identity(),
            nn.ReLU(),
            nn.ConvTranspose2d(
                self.dims[2],
                self.dims[1],
                kernel_size=4,
                stride=2,
                padding=1,
                bias=not self.use_batchnorm,
            ),
            nn.BatchNorm2d(self.dims[1]) if self.use_batchnorm else nn.Identity(),
            nn.ReLU(),
            nn.ConvTranspose2d(
                self.dims[1],
                self.dims[0],
                kernel_size=4,
                stride=2,
                padding=1,
                bias=not self.use_batchnorm,
            ),
            nn.BatchNorm2d(self.dims[0]) if self.use_batchnorm else nn.Identity(),
            nn.ReLU(),
        )

        self.out_layer = nn.Sequential(
            nn.ZeroPad2d((0, 1, 0, 1)),
            nn.Conv2d(
                self.dims[0],
                self.num_channels,
                kernel_size=4,
                stride=1,
                padding=1,
            ),
            nn.Sigmoid(),
        )

    def forward(self, z):
        out1 = self.layers(z[:, :, None, None])
        image_output = self.out_layer(out1)
        return image_output


class RAE_48x48(AE):
    def __init__(self,
                 latent_size=128,
                 ae_dim=128,
                 tanh=False,
                 k=0,
                 use_batchnorm=True,
                 w_kl = 0,
                 w_cons = 0,
                 m=0.99,
                 vq_reg=None,
                 word_size=4,
                 n_words=512,
                 commit_vq=2.5,
                 **kwargs):

        super().__init__(latent_size,
                         tanh,
                         k=k,
                         w_kl=w_kl,
                         vq_reg=vq_reg,
                         word_size=word_size,
                         n_words=n_words,
                         commit_vq=commit_vq)
        self.w_cons = w_cons
        #if self.vq_reg is not None:
        #    lt_size = word_size*latent_size
        #else:
        #lt_size = latent_size
        self.encoder = RAEEncoder48x48(num_channels=1,
                                       ae_dim=ae_dim,
                                       z_dim=latent_size,
                                       w_kl=self.w_kl,
                                       vq_reg=self.vq_reg,
                                       n_words=n_words,
                                       use_batchnorm=use_batchnorm)

        self.decoder = RAEDecoder48x48(num_channels=1,
                                       ae_dim=ae_dim,
                                       z_dim=latent_size,
                                       use_batchnorm=use_batchnorm)
        self.m = m

        if self.w_cons != 0:
            self.encoder_target = RAEEncoder48x48(num_channels=1,
                                       ae_dim=ae_dim,
                                       z_dim=latent_size,
                                       w_kl=self.w_kl,
                                       use_batchnorm=use_batchnorm)

            for param_q, param_k in zip(self.encoder.parameters(), self.encoder_target.parameters()):
                param_k.data.copy_(param_q.data)
                param_k.requires_grad = False

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder.parameters(), self.encoder_target.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

class NoneLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return None

class RAEEncoder48x48(nn.Module):
    def __init__(
        self,
        num_channels,
        ae_dim=128,
        z_dim=128,
        use_batchnorm=True,
        w_kl=0,
        n_words=512,
        vq_reg=None
    ):
        super().__init__()

        self.w_kl = w_kl
        self.vq_reg = vq_reg
        self.dims = [
            ae_dim // (2**i) for i in reversed(range(4))
        ]  # 1 2 4 8 # 32 64 128 256

        self.dims[-1] = ae_dim
        self.use_batchnorm = use_batchnorm

        self.out_dim = self.dims[3] * 3 * 3
        self.z_dim = z_dim

        self.layers = nn.Sequential(
            nn.Conv2d(
                num_channels,
                self.dims[0],
                kernel_size=4,
                stride=2,
                padding=1,
                bias=not self.use_batchnorm,
            ),
            nn.BatchNorm2d(self.dims[0]) if self.use_batchnorm else nn.Identity(),
            nn.ReLU(),
            nn.Conv2d(
                self.dims[0],
                self.dims[1],
                kernel_size=4,
                stride=2,
                padding=1,
                bias=not self.use_batchnorm,
            ),
            nn.BatchNorm2d(self.dims[1]) if self.use_batchnorm else nn.Identity(),
            nn.ReLU(),
            nn.Conv2d(
                self.dims[1],
                self.dims[2],
                kernel_size=4,
                stride=2,
                padding=2,
                bias=not self.use_batchnorm,
            ),
            nn.BatchNorm2d(self.dims[2]) if self.use_batchnorm else nn.Identity(),
            nn.ReLU(),
            nn.Conv2d(
                self.dims[2],
                self.dims[3],
                kernel_size=4,
                stride=2,
                padding=1,
                bias=not self.use_batchnorm,
            ),
            nn.BatchNorm2d(self.dims[3]) if self.use_batchnorm else nn.Identity(),
            nn.ReLU(),
        )
        if self.w_kl != 0:
            self.q_mean = nn.Linear(self.out_dim, self.z_dim)
            self.q_logvar = nn.Linear(self.out_dim, self.z_dim)

        if self.vq_reg is not None:
            self.q_mean = nn.Linear(self.out_dim, self.z_dim)
            if self.vq_reg == 'soft_vq':
                self.q_logvar = nn.Linear(self.out_dim, 256)
            else:
                self.q_logvar = NoneLayer()
                #self.q_logvar = nn.Linear(self.out_dim, num_embeddings)

        if (self.w_kl == 0) or (self.vq_reg is None):
            self.to_latent = nn.Linear(self.out_dim, self.z_dim)

    def forward(self, x):
        out = self.layers(x)#.view(x[0].size(0), -1)
        out = out.view(out.size(0), -1)
        if self.w_kl != 0 or self.vq_reg is not None:
            return self.q_mean(out), self.q_logvar(out)
        else:
            return self.to_latent(out)


class RAEDecoder48x48(nn.Module):
    def __init__(
        self,
        num_channels,
        z_dim=128,
        ae_dim=128,
        use_batchnorm=True,
    ):
        super().__init__()

        self.num_channels = num_channels
        self.dims = [ae_dim // (2**i) for i in reversed(range(4))]
        self.dims[-1] = ae_dim

        #self.kernel_dim = kernel_dim
        #self.padding = padding
        self.use_batchnorm = use_batchnorm
        #self.
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(
                z_dim,
                self.dims[3],
                kernel_size=6,
                stride=1,
                bias=not self.use_batchnorm,
            ),
            nn.BatchNorm2d(self.dims[3]) if self.use_batchnorm else nn.Identity(),
            nn.ReLU(),
            nn.ConvTranspose2d(
                self.dims[3],
                self.dims[2],
                kernel_size=4,
                stride=2,
                padding=1,
                bias=not self.use_batchnorm,
            ),
            nn.BatchNorm2d(self.dims[2]) if self.use_batchnorm else nn.Identity(),
            nn.ReLU(),
            nn.ConvTranspose2d(
                self.dims[2],
                self.dims[1],
                kernel_size=4,
                stride=2,
                padding=1,
                bias=not self.use_batchnorm,
            ),
            nn.BatchNorm2d(self.dims[1]) if self.use_batchnorm else nn.Identity(),
            nn.ReLU(),
            nn.ConvTranspose2d(
                self.dims[1],
                self.dims[0],
                kernel_size=4,
                stride=2,
                padding=1,
                bias=not self.use_batchnorm,
            ),
            nn.BatchNorm2d(self.dims[0]) if self.use_batchnorm else nn.Identity(),
            nn.ReLU(),
        )

        self.out_layer = nn.Sequential(
            nn.ZeroPad2d((0, 1, 0, 1)),
            nn.Conv2d(
                self.dims[0],
                self.num_channels,
                kernel_size=4,
                stride=1,
                padding=1,
            ),
            nn.Sigmoid(),
        )

    def forward(self, z):
        out1 = self.layers(z[:, :, None, None])
        image_output = self.out_layer(out1)
        return image_output