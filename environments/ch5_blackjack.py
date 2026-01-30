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
            if total + 11 <= 21: return total + 11, True
            else: return total + 1, usable_ace
        
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
        if a not in self._actions:
            raise ValueError(f"Invalid action {a}. Must be one of {self._actions}")
        
        done = False
        player_sum, dealer_card_shown, player_usable_ace = s
        r = 0
    
        # Players turn
        if a == 1: # Player hit
            card = self.draw_card(rng)
            player_sum, player_usable_ace = self.add_card(player_sum, player_usable_ace, card)
            if player_sum > 21: r = -1; done = True
            return (player_sum, dealer_card_shown, player_usable_ace), r, done
        elif a == 0: # player stay
            # Dealer turn
            dealer_sum, dealer_usable_ace = self.add_card(0, False, dealer_card_shown)
            dealer_sum, dealer_usable_ace = self.add_card(dealer_sum, dealer_usable_ace, self.dealer_card_hidden)
            while dealer_sum < 17:
                card = self.draw_card(rng)
                dealer_sum, dealer_usable_ace = self.add_card(dealer_sum, dealer_usable_ace, card)
            if dealer_sum > 21: 
                r = 1
                done = True
                return (player_sum, dealer_card_shown, player_usable_ace), r, done

            # Comparison
            done = True
            if player_sum > dealer_sum: r = 1
            elif dealer_sum > player_sum: r = -1

        return (player_sum, dealer_card_shown, player_usable_ace), r, done
    
    # add transitions function if needed for DP methods
        
        

        


                