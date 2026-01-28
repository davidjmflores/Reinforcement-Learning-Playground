# Obtain cards the sum of whose numerical values is as great as possible w/o exceeding 21
# J, K, Q = 10, A = 1 or 11
# version where players independently compete against dealer
# Game begins with two cards dealt to both dealer and player
# (One of dealer's cards is face up and one is face down)
# If player initially has ace and face card, its called a natural.
# If has natural they win immediately unless dealer also has natural in which the game is a draw
# If player hits, they get a card from the stack
# goes until they stop(sticks) or exceeds 21 (goes bust)

# Assumption: State = two tuples. [player_card_1, player_card_2,dealer_total]
# or State = (player_sum, dealer_sum+expectation over card values)
# action = 1 or 0 hit or stay

# When dealer's turn, he hits or sticks according to a fixed  strategy w/o choice
# dealer sticks on any sum of 17 or greater and hits otherwise
# if dealer goes bust, player wins
# otherwise outcome determine by who's closer to 21

# R = +1 -1 for losing and 0 for draws, game rewards are zero, no discounting
# terminal rewards are returns
# cards drawn from infinite deck(w/ replacement)

# if player gets ace that they can count as 11 w/o going bust, then ace is usable(always counted as 11)

# Use policy that sticks if the player's sum is 20 or 21 and otherwise hists

class Blackjack:
    def __init__(self):

    def states(self):
        
        for i in range