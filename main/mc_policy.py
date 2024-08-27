import numpy as np
from sb3_contrib import MaskablePPO
from vppo import VMaskablePPO
from stable_baselines3.common.type_aliases import GymEnv
from typing import List, Optional
import torch as th
import torch.nn.functional as F

class MultiPolicy():
    def __init__(self, env: GymEnv, policy_model_paths: List[str], mc_value_model_paths: List[str]):
        self.env = env
        self.policy_models = []
        self.mc_value_models = []
        for policy_model_path in policy_model_paths:
            policy_model = MaskablePPO.load(policy_model_path, env=self.env)
            policy_model.policy.set_training_mode(False)
            self.policy_models.append(policy_model)
        for mc_value_model_path in mc_value_model_paths:
            mc_value_model = VMaskablePPO.load(mc_value_model_path, env=self.env)
            mc_value_model.policy.set_training_mode(False)
            self.mc_value_models.append(mc_value_model)

    def select_coaches(self, observations: th.Tensor):
        with th.no_grad():
            valueses = []
            for mc_value_model in self.mc_value_models:
                values = mc_value_model.policy.predict_values(observations)
                valueses.append(values)
            valueses = th.concat(valueses, dim=-1)  # shape: (batchsize, num_of_mc_value_models)
            logits = valueses  # rename valueses to logits
            probabilities = F.softmax(logits, dim=1)  # Convert logits to probabilities using softmax

            chosen_models = th.multinomial(probabilities, num_samples=1)
            chosen_models = chosen_models.squeeze()

        return chosen_models
    
    def get_distributions(self, observations: th.Tensor, action_masks: Optional[np.ndarray]):
        with th.no_grad():
            logits = []
            # probses = []  # check-used
            for policy_model in self.policy_models:
                policy_model.policy.set_training_mode(False)
                features = policy_model.policy.extract_features(observations, policy_model.policy.pi_features_extractor)
                latent_pi = policy_model.policy.mlp_extractor.forward_actor(features)
                action_logits = policy_model.policy.action_net(latent_pi)            
                logits.append(action_logits)
                # probs = policy_model.policy.get_distribution(observations, action_masks).distribution.probs  # check-used
                # probses.append(probs)  # check-used
            logits = th.stack(logits, dim=0)  # shape: (num_of_models, batchsize, actions_num=4)
            # probses = th.stack(probses, dim=0)  # check-used
            chosen_model_indices = self.select_coaches(observations)
            chosen_logits = logits[chosen_model_indices, th.tensor(range(len(chosen_model_indices))), :]  # shape: (batchsize, actions_num=4)
            # chosen_probs = probses[chosen_model_indices, th.tensor(range(len(chosen_model_indices))), :]  # check-used
            distributions = self.policy_models[0].policy.action_dist.proba_distribution(action_logits=chosen_logits)
            if action_masks is not None:
                distributions.apply_masking(action_masks)

            # print(distributions.distribution.probs)
            # print("================================")
            # print(chosen_probs)  # check-used
            # input("PLEASE CHECK ME!!!")  # check-used and PASS!!

            return distributions
    
    def predict(self, observations: th.Tensor, action_masks: Optional[np.ndarray], deterministic: bool =False):
        with th.no_grad():
            distributions = self.get_distributions(observations, action_masks)
            actions = distributions.get_actions(deterministic=deterministic)
            actions = actions.cpu().numpy()
            return actions
    
    def log_prob(self, observations: th.Tensor, action_masks: Optional[np.ndarray], deterministic: bool =False):
        with th.no_grad():
            distributions = self.get_distributions(observations, action_masks)
            actions = distributions.get_actions(deterministic=deterministic)
            return distributions.log_prob(actions)
