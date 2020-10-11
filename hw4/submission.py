import util, math, random
from collections import defaultdict
from util import ValueIteration
from typing import List, Callable, Tuple, Any

############################################################
# Problem 3a

class BlackjackMDP(util.MDP):
    def __init__(self, cardValues: List[int], multiplicity: int, threshold: int, peekCost: int):
        """
        cardValues: list of integers (face values for each card included in the deck)
        multiplicity: single integer representing the number of cards with each face value
        threshold: maximum number of points (i.e. sum of card values in hand) before going bust
        peekCost: how much it costs to peek at the next card
        """
        self.cardValues = cardValues
        self.multiplicity = multiplicity
        self.threshold = threshold
        self.peekCost = peekCost

    # Return the start state.
    # Look closely at this function to see an example of state representation for our Blackjack game.
    # Each state is a tuple with 3 elements:
    #   -- The first element of the tuple is the sum of the cards in the player's hand.
    #   -- If the player's last action was to peek, the second element is the index
    #      (not the face value) of the next card that will be drawn; otherwise, the
    #      second element is None.
    #   -- The third element is a tuple giving counts for each of the cards remaining
    #      in the deck, or None if the deck is empty or the game is over (e.g. when
    #      the user quits or goes bust).
    def startState(self) -> Tuple:
        return (0, None, (self.multiplicity,) * len(self.cardValues))

    # Return set of actions possible from |state|.
    # You do not need to modify this function.
    # All logic for dealing with end states should be placed into the succAndProbReward function below.
    def actions(self, state: Tuple) -> List[str]:
        return ['Take', 'Peek', 'Quit']

    # Given a |state| and |action|, return a list of (newState, prob, reward) tuples
    # corresponding to the states reachable from |state| when taking |action|.
    # A few reminders:
    # * Indicate a terminal state (after quitting, busting, or running out of cards)
    #   by setting the deck to None.
    # * If |state| is an end state, you should return an empty list [].
    # * When the probability is 0 for a transition to a particular new state,
    #   don't include that state in the list returned by succAndProbReward.
    # Note: The grader expects the outputs follow the same order as the cards.
    # For example, if the deck has face values: 1, 2, 3. You should order your corresponding
    # tuples in the same order.
    def succAndProbReward(self, state: Tuple, action: str) -> List[Tuple]:
        # BEGIN_YOUR_CODE (our solution is 38 lines of code, but don't worry if you deviate from this)
        totalCardValueInHand, nextCardIndexIfPeeked, deckCardCounts = state
        if deckCardCounts == None:
            return []
        if action == 'Quit':
            return [((totalCardValueInHand, None, None), 1., totalCardValueInHand)]
        totalCardCount = sum(cardCount for cardCount in deckCardCounts)
        if action == 'Take':
            results = []
            for cardI, cardCount in enumerate(deckCardCounts):
                if cardCount == 0 or (nextCardIndexIfPeeked is not None and nextCardIndexIfPeeked != cardI):
                    continue
                nextCardValue = totalCardValueInHand + self.cardValues[cardI]
                prob = cardCount / totalCardCount if nextCardIndexIfPeeked is None else 1
                if nextCardValue > self.threshold:
                    results.append(((nextCardValue, None, None), prob, 0))
                else:
                    nextCardCounts = tuple(count - (1 if c == cardI else 0) for c, count in enumerate(deckCardCounts))
                    if sum(nextCardCounts) == 0:
                        results.append(((nextCardValue, None, None), prob, nextCardValue))
                    else:
                        results.append(((nextCardValue, None, nextCardCounts), prob, 0))
            return results
        elif action == 'Peek':
            if nextCardIndexIfPeeked is not None:
                return []
            results = []
            for cardI, cardCount in enumerate(deckCardCounts):
                if cardCount == 0:
                    continue
                prob = cardCount / totalCardCount
                nextState = (totalCardValueInHand, cardI, deckCardCounts)
                results.append((nextState, prob, -1 * self.peekCost))
            return results
        # END_YOUR_CODE

    def discount(self):
        return 1

############################################################
# Problem 4a: Q learning

# Performs Q-learning.  Read util.RLAlgorithm for more information.
# actions: a function that takes a state and returns a list of actions.
# discount: a number between 0 and 1, which determines the discount factor
# featureExtractor: a function that takes a state and action and returns a list of (feature name, feature value) pairs.
# explorationProb: the epsilon value indicating how frequently the policy
# returns a random action
class QLearningAlgorithm(util.RLAlgorithm):
    def __init__(self, actions: Callable, discount: float, featureExtractor: Callable, explorationProb=0.2):
        self.actions = actions
        self.discount = discount
        self.featureExtractor = featureExtractor
        self.explorationProb = explorationProb
        self.weights = defaultdict(float)
        self.numIters = 0

    # Return the Q function associated with the weights and features
    def getQ(self, state: Tuple, action: Any) -> float:
        score = 0
        for f, v in self.featureExtractor(state, action):
            score += self.weights[f] * v
        return score

    # This algorithm will produce an action given a state.
    # Here we use the epsilon-greedy algorithm: with probability
    # |explorationProb|, take a random action.
    def getAction(self, state: Tuple) -> Any:
        self.numIters += 1
        if random.random() < self.explorationProb:
            return random.choice(self.actions(state))
        else:
            return max((self.getQ(state, action), action) for action in self.actions(state))[1]

    # Call this function to get the step size to update the weights.
    def getStepSize(self) -> float:
        return 1.0 / math.sqrt(self.numIters)

    # We will call this function with (s, a, r, s'), which you should use to update |weights|.
    # Note that if s is a terminal state, then s' will be None.  Remember to check for this.
    # You should update the weights using self.getStepSize(); use
    # self.getQ() to compute the current estimate of the parameters.
    def incorporateFeedback(self, state: Tuple, action: Any, reward: int, newState: Tuple) -> None:
        # BEGIN_YOUR_CODE (our solution is 9 lines of code, but don't worry if you deviate from this)
        VOptSp = max(self.getQ(newState, newAction) for newAction in self.actions(newState)) if newState is not None else 0   # V_opt(s')
        factor = self.getStepSize() * (self.getQ(state, action) - (reward + self.discount * VOptSp))
        for f, v in self.featureExtractor(state, action):
            self.weights[f] -= factor * v
        # END_YOUR_CODE

# Return a single-element list containing a binary (indicator) feature
# for the existence of the (state, action) pair.  Provides no generalization.
def identityFeatureExtractor(state: Tuple, action: Any) -> List[Tuple[Tuple, int]]:
    featureKey = (state, action)
    featureValue = 1
    return [(featureKey, featureValue)]

############################################################
# Problem 4b: convergence of Q-learning
# As noted in the comments/documentation, util.simulate() is a function that takes as inputs an MDP and a particular RL algorithm you wish to run on the MDP. 
# The RL algorithm will be an instance of the RLAlgorithm abstract class defined in util.py. 
# In this case, you’ll want to use the Q-learning algorithm that you implemented in 4(a). 
# Once you’re done calling simulate, your RL will have explored and learned a policy from the MDP. 
# You will also want to run value iteration on the same MDP to get a policy pi
# Now that you have your trained Q-learning policy and value iteration policy, you can examine/explore the two and see where/how they differ. 
# You’ll want to think about how you can extract/query the policy from your trained Q-learning algorithm object. 
# Note that you should be careful that when you’re examining the policy, this is the final, “optimal” policy (i.e. your algorithm should only exploit, not explore). 

# Small test case
smallMDP = BlackjackMDP(cardValues=[1, 5], multiplicity=2, threshold=10, peekCost=1)

# Large test case
largeMDP = BlackjackMDP(cardValues=[1, 3, 5, 8, 10], multiplicity=3, threshold=40, peekCost=1)

def simulate_QL_over_MDP(mdp: BlackjackMDP, featureExtractor: Callable):
    # NOTE: adding more code to this function is totally optional, but it will probably be useful
    # to you as you work to answer question 4b (a written question on this assignment).  We suggest
    # that you add a few lines of code here to run value iteration, simulate Q-learning on the MDP,
    # and then print some stats comparing the policies learned by these two approaches. Remember to 
    # set your explorationProb to zero after simulate.
    # BEGIN_YOUR_CODE
    valueIteration = ValueIteration()
    valueIteration.solve(mdp)
    qLearn = QLearningAlgorithm(mdp.actions, mdp.discount(), featureExtractor)
    util.simulate(mdp, qLearn, 30000)
    qLearn.explorationProb = 0
    totalDiff = 0
    for vState, vAction in valueIteration.pi.items():
        qAction =  qLearn.getAction(vState)
        if vAction != qAction:
            totalDiff += 1
            # print(f"differing action at state {vState}: vAction={vAction} and qLearn={qAction}")
    # print(f"totalDiff={totalDiff} with {len(valueIteration.pi)} # of states")
    # END_YOUR_CODE
"""
# UNCOMMENT FOR SIMULATION OF 4B
print("Simulating for smallMDP:")
simulate_QL_over_MDP(smallMDP, identityFeatureExtractor)
print("Simulating for largeMDP:")
simulate_QL_over_MDP(largeMDP, identityFeatureExtractor)
"""

############################################################
# Problem 4c: features for Q-learning.

# You should return a list of (feature key, feature value) pairs.
# (See identityFeatureExtractor() above for a simple example.)
# Include only the following features in the list you return:
# -- Indicator for the action and the current total (1 feature).
#       The feature should be (('total', totalCardValueInHand, action),1). Feel free to use a different name.
# -- Indicator for the action and the presence/absence of each face value in the deck.
#       Example: if the deck is (3, 4, 0, 2), then your indicator on the presence of each card is (1, 1, 0, 1)
#       The feature will be (('bitmask', (1, 1, 0, 1), action), 1). Feel free to use a different name. 
#       Note: only add this feature if the deck is not None.
# -- Indicators for the action and the number of cards remaining with each face value.
#       Example: if the deck is (3, 4, 0, 2), you should have four features (one for each face value).
#       The first feature will be ((0, 3, action), 1)
#       Note: only add these features if the deck is not None.
def blackjackFeatureExtractor(state: Tuple, action: str) -> List[tuple]:
    total, nextCard, counts = state

    # BEGIN_YOUR_CODE (our solution is 7 lines of code, but don't worry if you deviate from this)
    features = []
    features.append((('total', total, action), 1))
    if counts is not None:
        features.append((('bitmask', tuple(1 if count > 0 else 0 for count in counts), action), 1))
        for cardI, count in enumerate(counts):
            features.append(((cardI, count, action), 1))
    return features
    # END_YOUR_CODE

############################################################
# Problem 4d: What happens when the MDP changes underneath you?!

# Original mdp
originalMDP = BlackjackMDP(cardValues=[1, 5], multiplicity=2, threshold=10, peekCost=1)

# New threshold
newThresholdMDP = BlackjackMDP(cardValues=[1, 5], multiplicity=2, threshold=15, peekCost=1)

def compare_changed_MDP(original_mdp: BlackjackMDP, modified_mdp: BlackjackMDP, featureExtractor: Callable):
    # NOTE: as in 4b above, adding more code to this function is completely optional, but we've added
    # this partial function here to help you figure out the answer to 4d (a written question).
    # Consider adding some code here to simulate two different policies over the modified MDP
    # and compare the rewards generated by each. You should try to run at least 30000 trails so that 
    # the model can converge.
    # BEGIN_YOUR_CODE
    valueIteration = ValueIteration()
    valueIteration.solve(originalMDP)
    fixedRL = util.FixedRLAlgorithm(valueIteration.pi)
    fixedRLReward = util.simulate(modified_mdp, fixedRL, 30000)
    print(f"Expected fixedRLReward={sum(fixedRLReward)/len(fixedRLReward)}")
    qLearn = QLearningAlgorithm(modified_mdp.actions, modified_mdp.discount(), featureExtractor)
    qLearnReward = util.simulate(modified_mdp, qLearn, 30000)
    print(f"Expected qLearnReward={sum(qLearnReward)/len(qLearnReward)}")
    # END_YOUR_CODE

"""
# UNCOMMENT FOR SIMULATION OF 4D
compare_changed_MDP(originalMDP, newThresholdMDP, blackjackFeatureExtractor)
"""