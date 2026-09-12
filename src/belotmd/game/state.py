"""
state.py — Belot hand state and rules.

Holds one hand (bidding + eight tricks) and answers the question the live
bot actually needs answered: which actions are legal right now.

This is deliberately NOT a Gym/PettingZoo environment. In this repository the
state is driven by `belotmd.platform.sync`, which writes the fields directly
from observed server frames rather than stepping a simulation — the belot.md
server is the authority on what happened. The trick-resolution, reward and
`step()` machinery that belongs to the self-play trainer lives in the training
repository, not here.

Card ids are `suit * 8 + rank`, with suits ordered diamonds, hearts, clubs,
spades and ranks ordered 7 < 8 < 9 < 10 < J < Q < K < A.
"""

import numpy as np

from .actions import ACTION_ACCEPT, ACTION_PASS, ACTION_SPACE_SIZE, ACTION_SUIT_BASE

RANK_JACK = 4


class BelotState:
    """State of a single hand. `reset()` starts a new one; bolts persist."""

    def __init__(self):
        self.action_space_size = ACTION_SPACE_SIZE
        self.num_players = 4

        self.dealer = 0
        self.bolts_by_team = [0, 0]   # persists across hands until a match reset
        self.graveyard = []
        self.last_trick = []
        self.reset()

    def reset(self):
        """Start a fresh hand. Bolt counters survive; everything else does not."""
        # Deck setup: 32 cards, ids 0-31.
        self.deck = np.random.permutation(32).tolist()
        self.hands = [[] for _ in range(self.num_players)]

        # Initial deal: 5 cards each.
        for p in range(self.num_players):
            self.hands[p] = self.deck[:5]
            self.deck = self.deck[5:]

        self.face_up_card = self.deck.pop(0)
        self.face_up_suit = self.face_up_card // 8
        self.face_up_rank = self.face_up_card % 8

        self.current_player = (self.dealer + 1) % self.num_players
        self.phase = "BIDDING"
        self.bidding_round = 1
        self.passes_in_round = 0

        self.trump = None
        self.declarer = None
        self.declaring_team = None
        self.defending_team = None
        self.declarer_has_played_trump = False

        # Trick tracking
        self.tricks_played = 0
        self.current_trick = []        # [(player_id, card), ...]
        self.trick_history = []
        self.tricks_won_by_team = [0, 0]
        self.raw_points_by_team = [0, 0]

        # Belief state
        self.graveyard = []
        self.last_trick = []
        self.impossible_cards = np.zeros((4, 32), dtype=bool)
        self.known_cards = np.zeros((4, 32), dtype=bool)

        self.done = False

        # Turn clock, filled in from live frames by the synchronizer (see
        # belotmd.platform.sync). None offline, where there is no clock.
        #   time_left_s    seconds remaining for the seat that must act
        #   turn_budget_s  seconds that seat was given in total
        #   deadline       time.monotonic() stamp to act by
        # An agent that searches should budget against `deadline`: overrunning
        # loses the seat to the platform bot permanently, not just the turn.
        # True when we joined a hand already in progress: past tricks are
        # unrecoverable, so the belief state is incomplete. Set by the
        # synchronizer; always False offline.
        self.beliefs_degraded = False

        # Each seat's declared combinations, verbatim wire tokens ("2l|5k"),
        # as the server's per-player `combinations` field stands. Always empty
        # offline. Filled by the synchronizer during play, by which point the
        # server has settled which declarations score -- an agent that scores
        # the hand the way belot.md does (bolts on trick + combination points)
        # reads it through `combinations.team_points`.
        self.combinations = ["", "", "", ""]

        self.time_left_s = None
        self.turn_budget_s = None
        self.deadline = None

        # Forced-trump exception ("BIZON"): a face-up Jack settles trump with
        # no bidding at all, and the player left of the dealer becomes
        # declarer. Live, this is why some hands jump phase 5 -> 9 and skip
        # the seven-swap window entirely.
        if self.face_up_rank == RANK_JACK:
            self.trump = self.face_up_suit
            self.declarer = (self.dealer + 1) % self.num_players
            self._finalize_bidding()

    def _finalize_bidding(self):
        """A trump has been chosen: deal the remaining cards and start play."""
        self.phase = "PLAYING"
        self.declaring_team = self.declarer % 2
        self.defending_team = 1 - self.declaring_team

        # Round 1: the declarer takes the face-up card. Round 2: the dealer does.
        face_up_recipient = self.declarer if self.bidding_round == 1 else self.dealer

        # The recipient provably holds it — the one belief pin that exists
        # before a single card is played.
        self.known_cards[face_up_recipient, self.face_up_card] = True

        for p in range(self.num_players):
            if p == face_up_recipient:
                self.hands[p].append(self.face_up_card)
                self.hands[p].extend(self.deck[:2])
                self.deck = self.deck[2:]
            else:
                self.hands[p].extend(self.deck[:3])
                self.deck = self.deck[3:]

        self.current_player = self.declarer      # declarer leads the first trick

    def get_legal_actions(self):
        """Boolean array of length 38: what the current player may do now."""
        legal = np.zeros(self.action_space_size, dtype=bool)
        if self.done:
            return legal

        if self.phase == "BIDDING":
            # The dealer is not allowed to pass out round 2 — someone must
            # name a trump.
            if not (self.bidding_round == 2 and self.current_player == self.dealer):
                legal[ACTION_PASS] = True

            if self.bidding_round == 1:
                legal[ACTION_ACCEPT] = True
            elif self.bidding_round == 2:
                # Any suit except the one that was turned up.
                for suit in range(4):
                    if suit != self.face_up_suit:
                        legal[ACTION_SUIT_BASE + suit] = True

        elif self.phase == "PLAYING":
            hand = self.hands[self.current_player]

            # --- leading a trick ---
            if len(self.current_trick) == 0:
                # Trump may not be led until the declarer has played one,
                # unless the leader holds nothing else.
                can_lead_trump = True
                if not self.declarer_has_played_trump and self.current_player != self.declarer:
                    if not all(c // 8 == self.trump for c in hand):
                        can_lead_trump = False
                for card in hand:
                    if card // 8 == self.trump and not can_lead_trump:
                        continue
                    legal[card] = True
                return legal

            # --- following ---
            led_suit = self.current_trick[0][1] // 8
            has_led_suit = any(c // 8 == led_suit for c in hand)
            has_trump = any(c // 8 == self.trump for c in hand)

            # Highest trump already on the table, in trick-power terms.
            trick_trumps = [c for _, c in self.current_trick if c // 8 == self.trump]
            highest_trump = max(
                (self.card_value(t, is_trump=True)[1] for t in trick_trumps),
                default=-1,
            )

            # Whether we *could* beat it — computed once, not per card.
            can_overruff = has_trump and any(
                c // 8 == self.trump and self.card_value(c, is_trump=True)[1] > highest_trump
                for c in hand
            )

            for card in hand:
                card_suit = card // 8
                card_val = self.card_value(card, is_trump=(card_suit == self.trump))[1]

                if has_led_suit:
                    if card_suit != led_suit:
                        continue
                    # Overruffing is compulsory even when following a led trump.
                    if led_suit == self.trump and can_overruff:
                        if card_val > highest_trump:
                            legal[card] = True
                    else:
                        legal[card] = True
                elif has_trump:
                    # Void in the led suit but holding trump: must ruff, and
                    # must overruff if able.
                    if card_suit != self.trump:
                        continue
                    if can_overruff:
                        if card_val > highest_trump:
                            legal[card] = True
                    else:
                        legal[card] = True
                else:
                    legal[card] = True       # void in both: free discard

        return legal

    def card_value(self, card, is_trump):
        """-> (card points, trick power rank). Trump reorders both."""
        rank = card % 8
        # Ranks: 0=7, 1=8, 2=9, 3=10, 4=J, 5=Q, 6=K, 7=A
        if is_trump:
            points_map = {0: 0, 1: 0, 2: 14, 3: 10, 4: 20, 5: 3, 6: 4, 7: 11}
            # Trump hierarchy: 7 < 8 < Q < K < 10 < A < 9 < J
            rank_map = {0: 0, 1: 1, 5: 2, 6: 3, 3: 4, 7: 5, 2: 6, 4: 7}
        else:
            points_map = {0: 0, 1: 0, 2: 0, 3: 10, 4: 2, 5: 3, 6: 4, 7: 11}
            # Plain hierarchy: 7 < 8 < 9 < J < Q < K < 10 < A
            rank_map = {0: 0, 1: 1, 2: 2, 4: 3, 5: 4, 6: 5, 3: 6, 7: 7}
        return points_map[rank], rank_map[rank]
