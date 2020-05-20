from abc import ABC

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils import try_import_torch
from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.models.model import restore_original_dimensions

torch, nn = try_import_torch()
from pommerman import constants


class ActorCriticModel(nn.Module, TorchModelV2, ABC):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        self.preprocessor = get_preprocessor(obs_space.original_space)(obs_space.original_space)

        self.shared_layers = None
        self.actor_layers = None
        self.critic_layers = None
        self._value_out = None

    def forward(self, input_dict, state, seq_lens):
        x = input_dict["obs"]
        print(x)
        x = self.shared_layers(x)
        # actor outputs
        logits = self.actor_layers(x)

        # compute value
        self._value_out = self.critic_layers(x)
        return logits, None

    def value_function(self):
        return self._value_out


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class ConvNetModel(ActorCriticModel):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, in_channels, feature_dim):
        ActorCriticModel.__init__(self, obs_space, action_space, num_outputs, model_config, name)

        self.shared_layers = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.ReLU(),
            Flatten(),
            nn.Linear(9 * 9 * 128, 256),
            nn.ReLU(),
            nn.Linear(256, feature_dim),
            nn.ReLU(),
        )

        self.actor_layers = nn.Sequential(
            nn.Linear(in_features=feature_dim, out_features=action_space.n)
        )

        self.critic_layers = nn.Sequential(
            nn.Linear(in_features=feature_dim, out_features=1)
        )

        self._value_out = None
