# Use policy that sticks if the player's sum is 20 or 21 and otherwise hists
import numpy as np

class Blackjack:
    def __init__(self):
        # All possible states where player hand(12-21), dealer hand(ace-10), boolean usable_ace
        self._states = []
        for i in range(12, 22):
            for j in range(1, 11):
                self._states.append((i, j, False))
                self._states.append((i, j, True))
        # Hit = 1, stay = 0
        self._actions = {0, 1}
        # infinite deck w/ replacement (prob, card value)
        self.cards = np.array([1,2,3,4,5,6,7,8,9,10])
        self.probs = np.array([
            1/13, 1/13, 1/13, 1/13, 1/13, 1/13, 1/13, 1/13, 1/13, # Ace through 9
            4/13 # 10, J, K, Q
        ])

    def states(self):
        return self._states
    
    # Added so generic MC can be used w/ environment
    def actions(self, s):
        return self._actions
    
    def draw_card(self, rng):
        return int(rng.choice(self.cards, p=self.probs))
    
    def add_card(self, total, usable_ace, card):
        if card == 1: # Ace drawn
            if total + 11 <= 21:
                return total + 11, True
            else:
                return total + 1, usable_ace
        
        total += card
        if total > 21 and usable_ace:
            return total - 10, False
        
        return total, usable_ace
    
    def is_natural(self, card_1, card_2):
        return((card_1 == 1 and card_2 == 10) or (card_1 == 10 and card_2 == 1))

    # Generic reset naming for use with generic MC
    def reset(self, rng):
        player_sum = 0
        player_usable_ace = False
        self.player_stay = False
        self.dealer_stay = False

        # Explicit first two draws to check for natural
        player_card_1 = self.draw_card(rng)
        player_sum, player_usable_ace = self.add_card(player_sum, player_usable_ace, player_card_1)
        player_card_2 = self.draw_card(rng)
        player_sum, player_usable_ace = self.add_card(player_sum, player_usable_ace, player_card_2)
        player_natural = self.is_natural(player_card_1, player_card_2)

        if not player_natural:
            while player_sum < 12:
                card = self.draw_card(rng)
                player_sum, player_usable_ace = self.add_card(player_sum, player_usable_ace, card)
        
        dealer_sum = 0
        dealer_usable_ace = False
        dealer_card_shown = self.draw_card(rng)
        dealer_sum, dealer_usable_ace = self.add_card(dealer_sum, dealer_usable_ace, dealer_card_shown)
        self.dealer_card_hidden = self.draw_card(rng)
        dealer_sum, dealer_usable_ace = self.add_card(dealer_sum, dealer_usable_ace, self.dealer_card_hidden)
        dealer_natural = self.is_natural(dealer_card_shown, self.dealer_card_hidden)

        # Handle naturals
        done = False
        r = 0

        if dealer_natural and player_natural:
            r = 0
            done = True
        elif dealer_natural:
            r = -1
            done = True
        elif player_natural:
            r = 1
            done= True

        return (player_sum, dealer_card_shown, player_usable_ace), r, done
        
    def step(self, s, a, rng):
        if a not in self.actions:
            raise ValueError(f"Invalid action {a}. Must be one of {self.actions}")
        
        player_sum, dealer_card_shown, usable_ace = s
        s_prime = [0, 0, usable_ace]
    
        # Players turn
        # Handles adjusting ace according to new draw, making ace draws 11 if sum below 21, and other draws
        if a == 1 and not self.player_stay: # Player hit
            card = self.draw_card(rng)
            player_sum, usable_ace = self.add_card(player_sum, usable_ace, card)
        # Handle stays like else: run dealer till he stays
        elif a == 0 and not self.player_stay: self.player_stay = True

        # Dealer turn
        dealer_hand = dealer_hand_shown + self.dealer_card_hidden
        if dealer_hand >= 17 and not self.dealer_stay: 
            s_prime[1] = dealer_hand_shown # Predetermined dealer policy
            self.dealer_stay = True
        else: 
            draw = int(rng.choice(self.cards, p=self.prob))
            if (draw == 1) and (dealer_hand + 11 < 21): draw = 11
            s_prime[1] = dealer_hand_shown + draw

        # Comparison
        if self.player_stay == self.dealer_stay == True:
            if dealer_hand > 21 and player_hand > 21 or dealer_hand == player_hand == 21: r = 0 # Both bust or both win: draw
            elif dealer_hand == 21 or player_hand > 21: r = -1  # lose
            elif player_hand == 21 or dealer_hand > 21: r = 1 # win
            else: 
                if player_hand > dealer_hand: r = 1
                elif dealer_hand > player_hand: r =-1
                else: r = 0
        else: r = 0

        return (s_prime), r, done
    
    # add transitions function if needed for DP methods
        
        

        


                