import torch
import argparse
import random
from .utils import load_weights
import torch.nn as nn


def load_ori_critic_network(model_name='', path_to_embedding='', device='cpu', image_size=48):
    if path_to_embedding == '' and model_name == '':
        def to_embedding(x):
            return x
    else:
        model_args, model_weights = load_weights(path_to_embedding,
                                                 model_name,
                                                 mode='end')
        if model_args.model_name == "simclr" and image_size == 50:
            embedding_model = ImageEmbedding(model_args.z_size, model_args.input_shape[0]).to(device)
            print('Loading ImageEmbedding50 Omniglot')
            embedding_model.load_state_dict(model_weights)
            embedding_model.eval()

            def to_embedding(x):
                features, last_layer = embedding_model(x)
                return features

        elif model_args.model_name == "simclr" and image_size == 48:
            embedding_model = ImageEmbedding48(model_args.z_size, model_args.input_shape[0]).to(device)
            print('Loading ImageEmbedding48')
            embedding_model.load_state_dict(model_weights)
            embedding_model.eval()

            def to_embedding(x):
                features, last_layer = embedding_model(x)
                return features

        else:
            raise NotImplementedError()

    return to_embedding

class ImageEmbedding(nn.Module):
    def __init__(self, embedding_size, num_input_channels=1):
        super().__init__()
        self.embedding_size = embedding_size
        self.embedding = nn.Sequential(
                        conv_block(num_input_channels, 64),
                        conv_block(64, 64),
                        conv_block(64, 64),
                        conv_block(64, 64),
                        Flatten(),
                        nn.ReLU(),
                        nn.Linear(576, self.embedding_size)
    )

        self.projection = nn.Sequential(
            #nn.Linear(576, self.embedding_size),
            #nn.ReLU(),
            nn.Linear(self.embedding_size, 128)
        )


    def calulate_embedding(self, image):
        return self.embedding(image)

    def forward(self, x):
        embedding = self.calulate_embedding(x)
        projection = self.projection(embedding)
        return embedding, projection

class ImageEmbedding48(nn.Module):
    def __init__(self, embedding_size, num_input_channels=1):
        super().__init__()
        self.embedding_size = embedding_size

        self.embedding = nn.Sequential(
            conv_block(num_input_channels, 64),
            conv_block(64, 128),
            conv_block(128, 128),
            Flatten(),
            nn.ReLU(),
            nn.Linear(6*6*128, embedding_size)
        )

        self.projection = nn.Sequential(
            #nn.Linear(576, self.embedding_size),
            #nn.ReLU(),
            nn.Linear(self.embedding_size, 128)
        )


    def calulate_embedding(self, image):
        return self.embedding(image)

    def forward(self, x):
        embedding = self.calulate_embedding(x)
        projection = self.projection(embedding)
        return embedding, projection

def conv_block(in_channels: int, out_channels: int) -> nn.Module:
    """Returns a Module that performs 3x3 convolution, ReLu activation, 2x2 max pooling.

    # Arguments
        in_channels:
        out_channels:
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2)
    )

class Flatten(nn.Module):
    """Converts N-dimensional Tensor of shape [batch_size, d1, d2, ..., dn] to 2-dimensional Tensor
    of shape [batch_size, d1*d2*...*dn].

    # Arguments
        input: Input tensor
    """
    def forward(self, input):
        return input.view(input.size(0), -1)