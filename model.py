import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

class RecurrentMAPPOModel(nn.Module):
    def __init__(self, local_dim=513, global_dim=332, hidden_dim=512, action_dim=38):
        super(RecurrentMAPPOModel, self).__init__()
        
        # ACTOR: Evaluates Imperfect Local Information
        self.actor_feature_extractor = nn.Sequential(
            nn.Linear(local_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.actor = nn.Linear(hidden_dim, action_dim)
        
        # CRITIC: Evaluates Perfect Global Information (Stateless CTDE)
        self.critic = nn.Sequential(
            nn.Linear(global_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, local_obs, global_obs, hc, action_mask=None, is_sequence=False):
        # --- ACTOR PASS ---
        if not is_sequence:
            # Single Step Rollout Mode
            actor_features = self.actor_feature_extractor(local_obs).unsqueeze(1) 
            lstm_out, new_hc = self.lstm(actor_features, hc)
            actor_out = lstm_out.squeeze(1) 
        else:
            # Batch Sequence Training Mode
            actor_features = self.actor_feature_extractor(local_obs) 
            actor_out, new_hc = self.lstm(actor_features, hc)
            
        logits = self.actor(actor_out)
        
        if action_mask is not None:
            bool_mask = action_mask.bool()
            logits = logits.masked_fill(~bool_mask, -1e9)
            
        dist = Categorical(logits=logits)
        
        # --- CRITIC PASS ---
        # nn.Linear automatically handles (Batch, Seq_Len, Dim) arrays flawlessly
        value = self.critic(global_obs)
        
        return dist, value, new_hc