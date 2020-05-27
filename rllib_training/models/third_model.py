from abc import ABC

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils import try_import_torch

torch, nn = try_import_torch()


class ActorCriticModel(nn.Module, TorchModelV2, ABC):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, in_channels, feature_dim):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        self.shared_layers = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=32,
                kernel_size=3,
                padding=1,
                stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1,
                stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            Flatten(),
            nn.Linear(9 * 9 * 128, 256),
            nn.ReLU(),
            nn.Linear(256, feature_dim),
            nn.ReLU()
        )

        self.actor_layers = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, action_space.n)
        )

        self.critic_layers = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=1)
        )

        self._shared_layer_out = None

    def forward(self, input_dict, state, seq_lens):
        x = input_dict["obs"]
        self._shared_layer_out = self.shared_layers(x)
        logits = self.actor_layers(self._shared_layer_out)

        return logits, []

    def value_function(self):
        return torch.reshape(self.critic_layers(self._shared_layer_out), [-1])


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
