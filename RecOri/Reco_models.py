from .utils import load_weights
from .Ori_models import Flatten
import torch.nn as nn
import torch
import math
import numpy as np
import torch.nn.functional as F
from RecOri.Ori_models import conv_block

def load_reco_critic_network(path_to_classifier, classifier_name, image_size, device, args):
    few_shot_args, few_shot_weights = load_weights(
            path_to_classifier,
            classifier_name,
            mode='end')
    if (few_shot_args.model_name == 'res_net' and image_size == 48):
        few_shot_model = ResNet48(z_size=few_shot_args.z_size, num_input_channels=1).to(device)
        check_args = (args.k_test == few_shot_args.k_test) and \
                     (args.n_test == few_shot_args.n_test) and \
                     (args.q_test == few_shot_args.q_test)
        if not check_args:
            print('Carreful, the Input args and the network args are not the same ... \n ' 
                  'Erasing the network args in favor of the Input args')
            few_shot_args.k_test = args.k_test
            few_shot_args.n_test = args.n_test
            few_shot_args.q_test = args.q_test

        few_shot_model.load_state_dict(few_shot_weights)
        few_shot_model.eval()

        def predict(support, queries):
            with torch.no_grad():
                _, metric_layer_sup = few_shot_model(support)
                _, metric_layer_que = few_shot_model(queries)
            logit_size = metric_layer_que.size(-1)
            metric_layer_sup = metric_layer_sup.view(-1, few_shot_args.n_test * few_shot_args.k_test, logit_size)
            metric_layer_que = metric_layer_que.view(-1, few_shot_args.q_test * few_shot_args.k_test,
                                                     logit_size)

            prototypes = compute_prototypes_fast(metric_layer_sup, few_shot_args.k_test, few_shot_args.n_test)
            distances = pairwise_distances_fast(metric_layer_que, prototypes, few_shot_args.distance)
            y_pred = (-distances).softmax(dim=2)
            return y_pred, distances

    elif (few_shot_args.model_name == 'proto_net' and image_size == 50):
        few_shot_model = ProtoNet(z_size=few_shot_args.z_size, num_input_channels=1).to(device)
        check_args = (args.k_test == few_shot_args.k_test) and \
                     (args.n_test == few_shot_args.n_test) and \
                     (args.q_test == few_shot_args.q_test)
        if not check_args:
            raise NameError('the dataset args and the network args are not the same')
        few_shot_model.load_state_dict(few_shot_weights)
        few_shot_model.eval()

        def predict(support, queries):
            with torch.no_grad():
                _, metric_layer_sup = few_shot_model(support)
                _, metric_layer_que = few_shot_model(queries)
            logit_size = metric_layer_que.size(-1)
            metric_layer_sup = metric_layer_sup.view(-1, few_shot_args.n_test * few_shot_args.k_test, logit_size)
            metric_layer_que = metric_layer_que.view(-1, few_shot_args.q_test * few_shot_args.k_test,
                                                     logit_size)

            prototypes = compute_prototypes_fast(metric_layer_sup, few_shot_args.k_test, few_shot_args.n_test)
            distances = pairwise_distances_fast(metric_layer_que, prototypes, few_shot_args.distance)
            y_pred = (-distances).softmax(dim=2)
            return y_pred, distances

    return predict


class ProtoNet(nn.Module):
    def __init__(self, z_size: int, num_input_channels = 1):
        super(ProtoNet, self).__init__()
        self.embedding = nn.Sequential(
            conv_block(num_input_channels, 64),
            conv_block(64, 64),
            conv_block(64, 64),
            conv_block(64, 64),
            Flatten(),
            nn.ReLU(),
            nn.Linear(576, z_size)
        )
        self.metric_layer = nn.Linear(z_size, 128)

    def forward(self, x):
        features = self.embedding(x)
        #output = self.metric_layer(features)
        output = self.metric_layer(features)
        return features, output

class ResNet48(nn.Module):
    def __init__(self, z_size: int, num_input_channels=1):
        super(ResNet48, self).__init__()
        self.in_planes = 64
        self.embedding = nn.Sequential(
            nn.Conv2d(num_input_channels, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            self._make_layer(BasicBlock, 64, 2, stride=1), #48
            self._make_layer(BasicBlock, 128, 2, stride=2), #24
            self._make_layer(BasicBlock, 256, 2, stride=2), #12
            self._make_layer(BasicBlock, 512, 2, stride=2), #6
            nn.AvgPool2d(6),
            Flatten(),
            nn.Linear(512, z_size)

        )

        self.metric_layer = nn.Linear(z_size, 128)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        features = self.embedding(x)
        # output = self.metric_layer(features)
        output = self.metric_layer(features)
        return features, output

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


def compute_prototypes_fast(support: torch.Tensor, k: int, n: int) -> torch.Tensor:
    """Compute class prototypes from support samples.

    # Arguments
        support: torch.Tensor. Tensor of shape (n * k, d) where d is the embedding
            dimension.
        k: int. "k-way" i.e. number of classes in the classification task
        n: int. "n-shot" of the classification task

    # Returns
        class_prototypes: Prototypes aka mean embeddings for each class
    """
    # Reshape so the first dimension indexes by class then take the mean
    # along that dimension to generate the "prototypes" for each class
    n_features = support.size(-1)
    class_prototypes = support.reshape(-1, k, n, n_features).mean(dim=2)
    return class_prototypes

def pairwise_distances_fast(x: torch.Tensor,
                       y: torch.Tensor,
                       matching_fn: str) -> torch.Tensor:
    """Efficiently calculate pairwise distances (or other similarity scores) between
    two sets of samples.

    # Arguments
        x: Query samples. A tensor of shape (n_x, d) where d is the embedding dimension
        y: Class prototypes. A tensor of shape (n_y, d) where d is the embedding dimension
        matching_fn: Distance metric/similarity score to compute between samples
    """
    n_x = x.shape[1]
    n_y = y.shape[1]
    nb_block = y.shape[0]
    if matching_fn == 'l2':
        #x = x/x.norm(dim=-1, keepdim=True)
        #y = y/y.norm(dim=-1, keepdim=True)
        distances = (
                x.unsqueeze(2).expand(nb_block, n_x, n_y, -1) -
                y.unsqueeze(1).expand(nb_block, n_x, n_y, -1)
        ).pow(2).sum(dim=3)
        return distances
    elif matching_fn == 'cosine':
        raise NotImplementedError()
        normalised_x = x / (x.pow(2).sum(dim=1, keepdim=True).sqrt() + EPSILON)
        normalised_y = y / (y.pow(2).sum(dim=1, keepdim=True).sqrt() + EPSILON)

        expanded_x = normalised_x.unsqueeze(1).expand(n_x, n_y, -1)
        expanded_y = normalised_y.unsqueeze(0).expand(n_x, n_y, -1)

        cosine_similarities = (expanded_x * expanded_y).sum(dim=2)
        return 1 - cosine_similarities
    elif matching_fn == 'dot':
        raise NotImplementedError()
        expanded_x = x.unsqueeze(1).expand(n_x, n_y, -1)
        expanded_y = y.unsqueeze(0).expand(n_x, n_y, -1)

        return -(expanded_x * expanded_y).sum(dim=2)
    else:
        raise(ValueError('Unsupported similarity function'))


def generate_evaluation_task(batch_stop= None, nb_test=500, nb_class=150,
                             seed=44, nb_block=1, k_test=20):
    if seed is not None:
        torch.manual_seed(seed)

    all_label = torch.arange(nb_class)
    if batch_stop is not None:
        all_label = all_label[:batch_stop]

    fixed_task = []

    label_extended = all_label.repeat(nb_test)
    range1 = (len(all_label) // k_test) * k_test
    nb_range_1 = int(math.ceil(label_extended.size(0) / range1))

    for i_r_1 in range(nb_range_1):
        if i_r_1 == nb_range_1 - 1:
            int_lab = label_extended[i_r_1 * range1:]
        else:
            int_lab = label_extended[i_r_1 * range1: (i_r_1 + 1) * range1]

        int_lab = int_lab[torch.randperm(int_lab.size(0))]
        if int_lab.size(0) % k_test != 0:
            raise Exception("Dint lab not divisible size ({0}) by k_way ({1})".format(int_lab.size(0), k_test))
        nb_range_2 = int_lab.size(0) // k_test

        for i_r_2 in range(nb_range_2):
            fixed_task.append(int_lab[i_r_2 * k_test: (i_r_2 + 1) * k_test].numpy())

    if nb_block != 1:
        new_task = []
        nb_batch = len(fixed_task) // nb_block
        for i in range(nb_batch):
            new_task.append(np.hstack(fixed_task[i*nb_block:(i+1)*nb_block]))
        if len(fixed_task) % nb_block != 0:
            new_task.append(np.hstack(fixed_task[(i+1)*nb_block:]))
        fixed_task = new_task
    return fixed_task
