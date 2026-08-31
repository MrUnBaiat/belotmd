"""
agent.py — the reference agent: a recurrent MAPPO policy trained by self-play.

This is the only place in the repository that imports torch or knows the
observation layout. Everything it needs — the 513/332 vectors, the LSTM hidden
state, the checkpoint — is private to this package. Swapping in a different
architecture means writing a sibling of this file, not touching `bot.py`.

The observation contract is documented in docs/PPO_AGENT.md. It is fixed by
the trained weights: change the layout and the checkpoint becomes meaningless.
"""

import os

import numpy as np
import torch

from .observation import build_observation, verify_observation_contract
from .policy import RecurrentMAPPOModel

# The trainer writes to CHECKPOINT_DIR/latest_model.pt (default
# "checkpoints/"), so a bare filename silently falls back to RANDOM WEIGHTS.
DEFAULT_SEARCH = [
    os.path.join("checkpoints", "latest_model.pt"),
    os.path.join("checkpoints", "best_model.pt"),
    "latest_model.pt",
]

HIDDEN_DIM = 512


class PPOAgent:
    name = "ppo"

    def __init__(self, checkpoint=None, device=None, deterministic=True):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.deterministic = deterministic
        self.model = RecurrentMAPPOModel().to(self.device)

        candidates = ([checkpoint] if checkpoint else []) + [
            p for p in DEFAULT_SEARCH if p != checkpoint
        ]
        found = next((p for p in candidates if p and os.path.exists(p)), None)
        if found:
            ckpt = torch.load(found, map_location=self.device)
            self.model.load_state_dict(ckpt["model_state_dict"])
            print(f"[Agent] Loaded trained weights from {found} "
                  f"(epoch {ckpt.get('epoch', '?')})")
        else:
            print("=" * 68)
            print("[Agent][CRITICAL] No checkpoint found — PLAYING WITH RANDOM "
                  "WEIGHTS.\n                  Searched: "
                  + ", ".join(candidates) + "\n"
                  "                  Any conclusion about play quality is "
                  "meaningless until this is fixed.")
            print("=" * 68)

        self.model.eval()
        self.hidden_state = self._zero_lstm_state()

        # The belief matrix depends on observation.py excluding known cards
        # from every candidate row. Without that, declared and swapped cards
        # leak phantom probability mass onto the wrong opponents.
        verify_observation_contract()

    def _zero_lstm_state(self):
        return (torch.zeros(1, 1, HIDDEN_DIM, device=self.device),
                torch.zeros(1, 1, HIDDEN_DIM, device=self.device))

    def reset(self):
        """New hand: zero the recurrent state, as at the start of every
        training episode."""
        self.hidden_state = self._zero_lstm_state()

    def act(self, state, seat, match_scores, legal_mask):
        local_obs, global_obs, _ = build_observation(state, seat, match_scores)

        # NOTE: `build_observation` embeds its own full legal mask as an
        # observation feature — that is what the network was trained on, so it
        # must NOT be narrowed. The mask passed in here may be narrower (the
        # server refused an action earlier this turn) and is applied only to
        # the logits, at selection time.
        local_t = torch.from_numpy(local_obs).unsqueeze(0).to(self.device)
        glob_t = torch.from_numpy(global_obs).unsqueeze(0).to(self.device)
        mask_t = torch.from_numpy(
            np.ascontiguousarray(legal_mask, dtype=np.int8)
        ).unsqueeze(0).to(self.device)

        with torch.no_grad():
            dist, _, new_hidden = self.model(
                local_t, glob_t, self.hidden_state, mask_t, is_sequence=False
            )
            if self.deterministic:
                action = int(torch.argmax(dist.probs, dim=-1).item())
            else:
                action = int(dist.sample().item())

        self.hidden_state = new_hidden
        return action

    # -------------------------------------------------------------- retries
    # A refused action has to be retried from the SAME pre-turn hidden state,
    # so the LSTM advances once per game decision (as in training) rather than
    # once per network message. `bot.py` drives this around a retry.
    def snapshot(self):
        return self.hidden_state

    def restore(self, hidden):
        self.hidden_state = hidden
