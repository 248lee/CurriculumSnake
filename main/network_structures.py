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

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        self.NatureCNN = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Flatten(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.NatureCNN(observations)
    

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
    
class DVNNetwork(nn.Module):
    def __init__(self, old_model_name: str, features_dim: int = 512):
        from sb3_contrib import MaskablePPO
        super(DVNNetwork, self).__init__()
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        self.feature_extractor = MaskablePPO.load(old_model_name).policy
        self.batch_norm = nn.BatchNorm1d(features_dim)
        self.hidden = nn.Sequential(nn.Linear(features_dim, 256), nn.ReLU())
        self.value = nn.Linear(256, 1)
    
    def forward(self, observations):
        with th.no_grad():
            feature = self.feature_extractor.extract_features(observations, self.feature_extractor.features_extractor)
        y = self.batch_norm(feature)
        y = self.hidden(y)
        output = self.value(y)
        return output
