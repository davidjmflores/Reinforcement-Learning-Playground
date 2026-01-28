# Use policy that sticks if the player's sum is 20 or 21 and otherwise hists

class Blackjack:
    def __init__(self):
        # All possible states where player hand(12-21), dealer hand(ace-10), boolean usable_ace
        self._states = []
        for i in range(12, 21):
            for j in range(1, 10):
                self._states.append((i, j, False))
                self._states.append((i, j, True))
        # Hit = 1, stay = 0
        self.actions = {0, 1}
        # infinite deck w/ replacement (prob, card value)
        self.cards = [(1/13, 1),(1/13, 2),(1/13, 3),(1/13, 4),(1/13, 5),(1/13, 6),(1/13, 7),(1/13, 8),(1/13, 9),(3/13, 10)]

    def states(self):
        return self._states
    
    def deal(self, rng):
        player_hand = 0
        dealer_card_shown = 0
        self.dealer_card_hidden = 0
        usable_ace = False
        # implement 
        # If player initially has ace and face card, its called a natural.
        # If has natural they win immediately unless dealer also has natural in which the game is a draw

        while player_hand < 12:
            draw += int(rng.choice(self.cards[0], p=self.cards[1]))
            if draw == 1:
                usable_ace = True
                if player_hand + 11 < 21: player_hand += 11
                else: player_hand += 1
            else: player_hand += draw
            # restart game if in winning state
            if player_hand == 21: player_hand = 0
        
        while True:
            dealer_card_shown = int(rng.choice(self.cards[0], p=self.cards[1]))
            self.dealer_card_hidden = int(rng.choice(self.cards[0], p=self.cards[1]))
            if  dealer_card_shown == 1 and not self.dealer_card_hidden == 10: break
            elif self.dealer_card_hidden == 1 and not dealer_card_shown == 10: break
            # otherwise, we have [a, 10] or [10, a] and need to redeal

        return (player_hand, dealer_card_shown, usable_ace)
        
    def step(self, s, a, rng):
        player_hand, dealer_hand_shown, usable_ace = s
        s_prime = []
        
        # Players turn
        if a == 1: # Player hit
            draw = int(rng.choice(self.cards[1], p=self.cards[0]))
            if draw == 1: usable_ace == True
            player_hand += draw

        # Dealer turn
        dealer_hand = dealer_hand_shown + self.dealer_card_hidden
        if dealer_hand >= 17: s_prime[1] = dealer_hand_shown # Predetermined dealer policy
        else: 
            draw = int(rng.choice(self.cards[0], p=self.cards[1]))
            if (draw == 1) and (dealer_hand + 11 < 21): draw = 11
            else: draw = 1
            dealer_hand += draw

        # Comparison
        if dealer_hand > 21 and player_hand > 21 or dealer_hand == player_hand == 21: r = 0 # BOth bust or both win: draw
        elif dealer_hand == 21 or player_hand > 21: r = -1  # lose
        elif player_hand == 21 or dealer_hand > 21: r = 1 # win
        else: r = 0 # transient states

        # implement
        # When dealer's turn, he hits or sticks according to a fixed  strategy w/o choice
        # dealer sticks on any sum of 17 or greater and hits otherwise
        # if dealer goes bust, player wins
        # otherwise outcome determine by who's closer to 21

        # if player gets ace that they can count as 11 w/o going bust, then ace is usable(always counted as 11)

        s_prime = (player_hand, dealer_hand, usable_ace)

        return (s_prime), r
        
        

        


                