# This file was copied from parkour/scripts/rsl_rl/modules/feature_extractors/estimator.py
# Original source: https://github.com/CAI23sbP/Isaaclab_Parkour.git
# Base model: https://extreme-parkour.github.io/
#
# Licensed under GNU General Public License v3.0
# See parkour/LICENSE for full license text
#
# This file was copied to avoid import dependencies on isaaclab_rl in the test environment.

import torch.nn as nn
import torch
from rsl_rl.utils import resolve_nn_activation

class DefaultEstimator(nn.Module):
    def __init__(self,  
                 num_prop,
                 num_priv_explicit,
                 hidden_dims=[256, 128, 64],
                 activation="elu",
                **kwargs):
        super(DefaultEstimator, self).__init__()

        self.input_dim = num_prop
        self.output_dim = num_priv_explicit
        activation = resolve_nn_activation(activation)
        estimator_layers = []
        estimator_layers.append(nn.Linear(self.input_dim, hidden_dims[0]))
        estimator_layers.append(activation)
        for l in range(len(hidden_dims)):
            if l == len(hidden_dims) - 1:
                estimator_layers.append(nn.Linear(hidden_dims[l], num_priv_explicit))
            else:
                estimator_layers.append(nn.Linear(hidden_dims[l], hidden_dims[l + 1]))
                estimator_layers.append(activation)
        self.estimator = nn.Sequential(*estimator_layers)
    
    def forward(self, input):
        return self.estimator(input)
    
    def inference(self, input):
        with torch.no_grad():
            return self.estimator(input)



