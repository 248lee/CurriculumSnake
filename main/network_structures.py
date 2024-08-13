import torch.nn as nn
import torch as th
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym
from stable_baselines3.common.preprocessing import get_action_dim, is_image_space, maybe_transpose, preprocess_obs

conv_stage1_kernels = 32
conv_stage2_kernels = 64
conv_stage3_kernels = 64

class CustomFeatureExtractorCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim=512):
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.NatureCNN = nn.Sequential(
            nn.Conv2d(n_input_channels, 64, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(64, 96, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(96, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Flatten(),
        )
        with th.no_grad():
            n_flatten = self.NatureCNN(th.as_tensor(observation_space.sample()[None]).float()).shape[1]
        self.Linea = nn.Sequential(
            nn.BatchNorm1d(num_features=n_flatten, affine=False),
            nn.Linear(in_features=n_flatten, out_features=features_dim)
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.Linea(self.NatureCNN(observations))
    

class Stage2CustomFeatureExtractorCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        self.avg = nn.AvgPool2d(2, stride=2)
        self.cnn_stage2 = nn.Sequential(
            nn.Conv2d(n_input_channels, conv_stage2_kernels, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
        )
        self.cnn_stage1 = nn.Sequential(
            nn.Conv2d(conv_stage2_kernels, conv_stage1_kernels, kernel_size=5, stride=1, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
        )
        self.cnn = nn.Sequential(
            nn.Conv2d(conv_stage1_kernels, 64, kernel_size=5, stride=1, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(64, 64, kernel_size=4, stride=1, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Flatten(),
        )
        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(self.cnn_stage1(self.cnn_stage2(self.avg(
                th.as_tensor(observation_space.sample()[None]).float()
            )))).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

        with th.no_grad():
            self.latent_output_shape = self.forward(th.as_tensor(observation_space.sample()[None]).float()).shape

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(self.cnn_stage1(self.cnn_stage2(self.avg(observations)))))
    
class Stage3CustomFeatureExtractorCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        self.avg = nn.AvgPool2d(2, stride=2)
        self.cnn_stage3 = nn.Sequential(
            nn.Conv2d(n_input_channels, conv_stage3_kernels, kernel_size=8, stride=1, padding='same'),
            nn.PReLU(),
            nn.MaxPool2d(2, stride=2),
        )
        self.cnn_stage2 = nn.Sequential(
            nn.Conv2d(conv_stage3_kernels, conv_stage2_kernels, kernel_size=5, stride=1, padding='same'),
            nn.PReLU(),
            nn.MaxPool2d(2, stride=2),
        )
        self.bn = nn.BatchNorm2d(conv_stage2_kernels)
        self.cnn_stage1 = nn.Sequential(
            nn.Conv2d(conv_stage2_kernels, conv_stage1_kernels, kernel_size=5, stride=1, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
        )
        self.cnn = nn.Sequential(
            # nn.Conv2d(conv_stage1_kernels, 64, kernel_size=4, stride=1, padding='same'),
            # nn.ReLU(),
            # nn.MaxPool2d(2, stride=2),
            # nn.Conv2d(64, 64, kernel_size=3, stride=1, padding='same'),
            # nn.ReLU(),
            # nn.MaxPool2d(2, stride=2),
            nn.Flatten(),
        )
        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(self.cnn_stage1(self.bn(self.cnn_stage2(self.cnn_stage3(self.avg(
                th.as_tensor(observation_space.sample()[None]).float()
            )))))).shape[1]
        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

        with th.no_grad():
            self.latent_output_shape = self.forward(th.as_tensor(observation_space.sample()[None]).float()).shape

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(self.cnn_stage1(self.bn(self.cnn_stage2(self.cnn_stage3(self.avg(observations)))))))
    
class WhiteFeatureExtractorCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box):
        super().__init__(observation_space, features_dim=87)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return observations


def init_weights(m):
    if isinstance(m, nn.Linear):
        th.nn.init.uniform_(m.weight, -0.1, 0.1)
        m.bias.data.fill_(0)

class DVNNetwork(nn.Module):
    def __init__(self, old_model_name: str, features_dim: int = 512):
        from sb3_contrib import MaskablePPO
        super(DVNNetwork, self).__init__()
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        self.feature_extractor = MaskablePPO.load(old_model_name).policy
        for name, param in self.feature_extractor.named_parameters():
            param.requires_grad = False
        # self.feature_extractor.set_training_mode(False)
        self.batch_norm = nn.BatchNorm1d(features_dim)
        self.hidden = nn.Sequential(nn.Linear(features_dim, 256), 
                                    nn.ReLU(),
                                    nn.Linear(256, 128),
                                    nn.ReLU())
        self.hidden.apply(init_weights)
        self.value = nn.Linear(128, 1)
        th.nn.init.uniform_(self.value .weight, -0.1, 0.1)

    def forward(self, observations):
        feature = self.feature_extractor.extract_features(observations, self.feature_extractor.features_extractor)
        y = self.batch_norm(feature)
        y = self.hidden(y)
        output = self.value(y)
        return output
    
    def set_unfreeze(self):
        for name, param in self.feature_extractor.named_parameters():
            param.requires_grad = True
