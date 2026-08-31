import numpy as np

class BelotEnv:
    def __init__(self):
        # This is the env for 1 episode (8 tricks) till Done = True. After which reset is called
        # Action Space: 0-31 for playing cards, 32=Pass, 33=Accept, 34-37=Pick Suit
        self.action_space_size = 38
        self.num_players = 4
        
        # Internal state
        self.dealer = 0
        self.bolts_by_team = [0, 0] # Persists across hands (reset() does not wipe this)
        self.graveyard = []
        self.last_trick = []
        
        # New: Track ongoing dense rewards to calculate the final true-up
        self.accumulated_dense_rewards = [0.0, 0.0, 0.0, 0.0]
        self.reset()

    def reset(self):
        """Called at the end of the episode (after 8 tricks) to start a new hand. Bolts persist across hands until a match reset."""
        # Deck setup: 32 cards. IDs 0-31
        # Suit = id // 8, Rank = id % 8 (0=7, 1=8, 2=9, 3=10, 4=J, 5=Q, 6=K, 7=A)
        self.deck = np.random.permutation(32).tolist()
        self.hands = [[] for _ in range(self.num_players)]
        
        # Initial Deal: 5 cards each
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
        
        # Tricks tracking
        self.tricks_played = 0
        self.current_trick = []  # List of (player_id, card)
        self.trick_history = []
        self.tricks_won_by_team = [0, 0] 
        self.raw_points_by_team = [0, 0]
        
        # New State Tracking
        self.graveyard = []
        self.last_trick = []
        self.impossible_cards = np.zeros((4, 32), dtype=bool)
        self.known_cards = np.zeros((4, 32), dtype=bool)
        
        # Reset episode accumulated rewards
        self.accumulated_dense_rewards = [0.0, 0.0, 0.0, 0.0]
        self.done = False

        # Forced Jack Exception
        if self.face_up_rank == 4: # Jack
            self.trump = self.face_up_suit
            # The player next to the dealer becomes declarer
            self.declarer = (self.dealer + 1) % self.num_players
            self._finalize_bidding()
            
        return self._get_observation()

    def _finalize_bidding(self):
        """Called when a trump is chosen. Deals remaining cards."""
        self.phase = "PLAYING"
        self.declaring_team = self.declarer % 2
        self.defending_team = 1 - self.declaring_team
        
        # If bidding round 1: Declarer gets face up card.
        # If bidding round 2: Dealer gets face up card.
        face_up_recipient = self.declarer if self.bidding_round == 1 else self.dealer
        
        # Track that the recipient definitely holds the face-up card
        self.known_cards[face_up_recipient, self.face_up_card] = True

        for p in range(self.num_players):
            if p == face_up_recipient:
                self.hands[p].append(self.face_up_card)
                self.hands[p].extend(self.deck[:2])
                self.deck = self.deck[2:]
            else:
                self.hands[p].extend(self.deck[:3])
                self.deck = self.deck[3:]
                
        # Declarer leads the first trick
        self.current_player = self.declarer

    def get_legal_actions(self):
        """Returns a boolean array of length 38 indicating legal actions for the current player."""
        legal = np.zeros(self.action_space_size, dtype=bool)
        if self.done: return legal
            
        if self.phase == "BIDDING":
            if self.bidding_round == 2 and self.current_player == self.dealer:
                legal[32] = False # Dealer cannot pass in round 2
            else:
                legal[32] = True # Pass
                
            if self.bidding_round == 1:
                legal[33] = True # Accept
            elif self.bidding_round == 2:
                # Can choose any suit EXCEPT the face up suit
                for suit in range(4):
                    if suit != self.face_up_suit:
                        legal[34 + suit] = True
        
        elif self.phase == "PLAYING":
            hand = self.hands[self.current_player]
            
            # If leading the trick
            if len(self.current_trick) == 0:
                # Check Trump restriction for non-declarers
                can_lead_trump = True
                if not self.declarer_has_played_trump and self.current_player != self.declarer:
                    # Exception: If player has ONLY trumps
                    if not all(c // 8 == self.trump for c in hand):
                        can_lead_trump = False
                for card in hand:
                    if card // 8 == self.trump and not can_lead_trump: continue # Illegal to lead trump right now
                    legal[card] = True
                return legal
                
            # Following trick logic
            led_card = self.current_trick[0][1]
            led_suit = led_card // 8
            has_led_suit = any(c // 8 == led_suit for c in hand)
            has_trump = any(c // 8 == self.trump for c in hand)
            
            # Find the highest trump currently in the trick
            trick_trumps = [c[1] for c in self.current_trick if c[1] // 8 == self.trump]
            highest_trick_trump_val = max([self._get_card_value(t, is_trump=True)[1] for t in trick_trumps]) if trick_trumps else -1
            
            # Pre-calculate overruff capability ONCE
            can_overruff = False
            if has_trump:
                can_overruff = any(c // 8 == self.trump and self._get_card_value(c, is_trump=True)[1] > highest_trick_trump_val for c in hand)
            
            for card in hand:
                card_suit = card // 8
                card_val = self._get_card_value(card, is_trump=(card_suit == self.trump))[1]
                
                if has_led_suit:
                    if card_suit == led_suit:
                        if led_suit == self.trump:
                            # Overruff applies even when following a led trump!
                            if can_overruff:
                                if card_val > highest_trick_trump_val:
                                    legal[card] = True
                            else:
                                legal[card] = True
                        else:
                            legal[card] = True
                elif has_trump:
                    if card_suit == self.trump:
                        # Overruff rule when Ruffing
                        if can_overruff:
                            if card_val > highest_trick_trump_val:
                                legal[card] = True
                        else:
                            legal[card] = True # Have to play trump, but can't overruff
                else:
                    # Discard
                    legal[card] = True
        return legal

    def step(self, action):
        if not self.get_legal_actions()[action]:
            raise ValueError(f"Illegal action {action} chosen by Player {self.current_player}")

        step_rewards = [0.0, 0.0, 0.0, 0.0]

        if self.phase == "BIDDING":
            self._handle_bidding_action(action)
        elif self.phase == "PLAYING":
            trick_resolved, points, winner = self._handle_playing_action(action)
            
            if trick_resolved:
                # 1. Provide Dense Zero-Sum Reward (Normalized to Raw Points Max: 162.0)
                winning_team = winner % 2
                losing_team = 1 - winning_team
                
                dense_w = points / 162.0
                dense_l = -points / 162.0
                
                step_rewards[winning_team] = dense_w
                step_rewards[winning_team + 2] = dense_w
                step_rewards[losing_team] = dense_l
                step_rewards[losing_team + 2] = dense_l
                
                for i in range(4):
                    self.accumulated_dense_rewards[i] += step_rewards[i]

        info = {}
        if self.done:
            # 2. Final True-Up Process
            game_points = self._calculate_final_rewards()
            info["game_points"] = game_points # Export for global tensorboard tracking
            
            # Zero-sum Match Points target (Normalized to Max Match Points: 16.0)
            target_0 = (game_points[0] - game_points[1]) / 16.0
            target_1 = (game_points[1] - game_points[0]) / 16.0
            targets = [target_0, target_1, target_0, target_1]
            
            # Correct the final trick's reward so episode sum exactly equals the zero-sum strategic Target
            for i in range(4):
                true_up = targets[i] - self.accumulated_dense_rewards[i]
                step_rewards[i] += true_up
                
            self.dealer = (self.dealer + 1) % self.num_players

        return self._get_observation(), step_rewards, self.done, info

    def _handle_bidding_action(self, action):
        if action == 32: # Pass
            self.passes_in_round += 1
            if self.passes_in_round == 4 and self.bidding_round == 1:
                self.bidding_round = 2
                self.passes_in_round = 0
                
        elif action == 33: # Accept
            self.trump = self.face_up_suit
            self.declarer = self.current_player
            self._finalize_bidding()
            
        elif 34 <= action <= 37: # Choose Suit
            self.trump = action - 34
            self.declarer = self.current_player
            self._finalize_bidding()
            
        if self.phase == "BIDDING":
            self.current_player = (self.current_player + 1) % self.num_players

    def _handle_playing_action(self, action):
        card = action
        
        # Track if declarer breaks trump
        if self.current_player == self.declarer and (card // 8 == self.trump):
            self.declarer_has_played_trump = True

        # Remove card from hands and known beliefs
        self.hands[self.current_player].remove(card)
        self.known_cards[self.current_player, card] = False 
        
        # Track Impossible Cards (Failure to follow suit/ruff)
        if len(self.current_trick) > 0:
            led_suit = self.current_trick[0][1] // 8
            played_suit = card // 8
            if played_suit != led_suit:
                # Did not follow suit
                for c in range(led_suit*8, led_suit*8+8):
                    self.impossible_cards[self.current_player, c] = True
                
                # If they didn't ruff (and trump exists), they don't have trump
                if self.trump is not None and played_suit != self.trump:
                    for c in range(self.trump*8, self.trump*8+8):
                        self.impossible_cards[self.current_player, c] = True

        self.current_trick.append((self.current_player, card))
        
        if len(self.current_trick) < 4:
            self.current_player = (self.current_player + 1) % self.num_players
            return False, 0, None # Not resolved yet
        else:
            # Evaluate trick
            winner, points = self._evaluate_trick()
            winning_team = winner % 2
            
            # Populate Graveyard and Last Trick memory before wiping
            self.last_trick = self.current_trick.copy()
            self.graveyard.extend([c for _, c in self.current_trick])
            
            self.tricks_won_by_team[winning_team] += 1
            self.trick_history.append(self.current_trick)
            self.tricks_played += 1
            
            if self.tricks_played == 8: # Why are we doing Pasledu handling here?
                # Add Pasledu directly into trick raw points for correct dense reward scaling
                points += 10
                self.done = True

            self.raw_points_by_team[winning_team] += points
            self.current_player = winner
            self.current_trick = []

            return True, points, winner

    def _evaluate_trick(self):
        led_suit = self.current_trick[0][1] // 8
        best_player, best_rank_val = None, -1
        best_is_trump = False
        points = 0
        
        for player, card in self.current_trick:
            suit = card // 8
            is_trump = (suit == self.trump)
            pts, rank_val = self._get_card_value(card, is_trump)
            points += pts
            
            if not is_trump and not best_is_trump and suit == led_suit:
                if rank_val > best_rank_val:
                    best_rank_val, best_player = rank_val, player
            elif is_trump and best_is_trump:
                if rank_val > best_rank_val:
                    best_rank_val, best_player = rank_val, player
            elif is_trump and not best_is_trump:
                best_is_trump = True
                best_rank_val, best_player = rank_val, player
                    
        return best_player, points

    def _get_card_value(self, card, is_trump):
        """Returns (Points, Internal Trick Power Rank)"""
        rank = card % 8
        # Ranks: 0=7, 1=8, 2=9, 3=10, 4=J, 5=Q, 6=K, 7=A
        if is_trump:
            points_map = {0:0, 1:0, 2:14, 3:10, 4:20, 5:3, 6:4, 7:11}
            # Trick hierarchy in trump: 7 < 8 < Q < K < 10 < A < 9 < J
            rank_map = {0:0, 1:1, 5:2, 6:3, 3:4, 7:5, 2:6, 4:7} 
        else:
            points_map = {0:0, 1:0, 2:0, 3:10, 4:2, 5:3, 6:4, 7:11}
            # Trick hierarchy non-trump: 7 < 8 < 9 < J < Q < K < 10 < A
            rank_map = {0:0, 1:1, 2:2, 4:3, 5:4, 6:5, 3:6, 7:7}
        return points_map[rank], rank_map[rank]

    def _calculate_final_rewards(self):
        """This is run at the end of the episode (8 tricks)"""
        game_points = [0, 0]
        bolt_occurred = False
        
        # 1. Zero Tricks Condition Check
        for team in range(2):
            if self.tricks_won_by_team[team] == 0: game_points[team] = -10 

        # 2. Score Computation (with Bolt Logic)
        if game_points[0] != -10 and game_points[1] != -10:
            raw_dec = self.raw_points_by_team[self.declaring_team]
            raw_def = self.raw_points_by_team[self.defending_team]
            
            # Bolt Condition Check: Total raw points is 162. If declarer gets <= 80, they fail.
            if raw_dec <= 80:
                game_points[self.declaring_team] = 0
                game_points[self.defending_team] = 16
                bolt_occurred = True
            else:
                # Standard calculation: Find Bile for defending team
                remainder = raw_def % 10
                def_bile = (raw_def // 10) + (1 if remainder > 5 else 0)
                
                game_points[self.defending_team] = def_bile
                game_points[self.declaring_team] = 16 - def_bile
        
        # Handing the Zero Trick Edge Cases vs Bolt
        elif game_points[self.defending_team] == -10:
            # Defending team got 0 tricks
            game_points[self.declaring_team] = 16
        elif game_points[self.declaring_team] == -10:
            # Declaring team got 0 tricks. They take the -10 zero-trick penalty, but it is ALSO a Bolt!
            game_points[self.defending_team] = 16
            bolt_occurred = True

        # 3. Apply the 3rd Bolt Penalty
        if bolt_occurred:
            self.bolts_by_team[self.declaring_team] += 1
            if self.bolts_by_team[self.declaring_team] == 3:
                game_points[self.declaring_team] -= 10
                self.bolts_by_team[self.declaring_team] = 0 

        return [game_points[0], game_points[1], game_points[0], game_points[1]]

    def _get_observation(self):
        return {
            "current_player": self.current_player,
            "hand": self.hands[self.current_player],
            "trump": self.trump,
            "trick_history": self.trick_history,
            "current_trick": self.current_trick,
            "bolts_by_team": self.bolts_by_team.copy() 
        }
        
'''
TODO:
- Implement less than 14 game cancellation
'''