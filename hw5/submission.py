from util import manhattanDistance
from game import Directions
import math, random, util
from typing import Any, DefaultDict, List, Set, Tuple
from collections import defaultdict

from game import Agent
from pacman import GameState


class ReflexAgent(Agent):
  """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
  """
  def __init__(self):
    self.lastPositions = []
    self.dc = None


  def getAction(self, gameState: GameState):
    """
    getAction chooses among the best options according to the evaluation function.

    getAction takes a GameState and returns some Directions.X for some X in the set {North, South, West, East}
    ------------------------------------------------------------------------------
    Description of GameState and helper functions:

    A GameState specifies the full game state, including the food, capsules,
    agent configurations and score changes. In this function, the |gameState| argument
    is an object of GameState class. Following are a few of the helper methods that you
    can use to query a GameState object to gather information about the present state
    of Pac-Man, the ghosts and the maze.

    gameState.getLegalActions(agentIndex):
        Returns the legal actions for the agent specified. Returns Pac-Man's legal moves by default.

    gameState.generateSuccessor(agentIndex, action):
        Returns the successor state after the specified agent takes the action.
        Pac-Man is always agent 0.

    gameState.getPacmanState():
        Returns an AgentState object for pacman (in game.py)
        state.configuration.pos gives the current position
        state.direction gives the travel vector

    gameState.getGhostStates():
        Returns list of AgentState objects for the ghosts

    gameState.getNumAgents():
        Returns the total number of agents in the game

    gameState.getScore():
        Returns the score corresponding to the current state of the game


    The GameState class is defined in pacman.py and you might want to look into that for
    other helper methods, though you don't need to.
    """
    # Collect legal moves and successor states
    legalMoves = gameState.getLegalActions()

    # Choose one of the best actions
    scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
    bestScore = max(scores)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices) # Pick randomly among the best


    return legalMoves[chosenIndex]

  def evaluationFunction(self, currentGameState: GameState, action: str) -> float:
    """
    The evaluation function takes in the current and proposed successor
    GameStates (pacman.py) and returns a number, where higher numbers are better.

    The code below extracts some useful information from the state, like the
    remaining food (oldFood) and Pacman position after moving (newPos).
    newScaredTimes holds the number of moves that each ghost will remain
    scared because of Pacman having eaten a power pellet.
    """
    # Useful information you can extract from a GameState (pacman.py)
    successorGameState = currentGameState.generatePacmanSuccessor(action)
    newPos = successorGameState.getPacmanPosition()
    oldFood = currentGameState.getFood()
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    return successorGameState.getScore()


def scoreEvaluationFunction(currentGameState: GameState):
  """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
  """
  return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
  """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
  """

  def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
    self.index = 0 # Pacman is always agent index 0
    self.evaluationFunction = util.lookup(evalFn, globals())
    self.depth = int(depth)

######################################################################################
# Problem 1b: implementing minimax

class MinimaxAgent(MultiAgentSearchAgent):
  """
    Your minimax agent (problem 1)
  """

  def getAction(self, gameState: GameState) -> str:
    """
      Returns the minimax action from the current gameState using self.depth
      and self.evaluationFunction. Terminal states can be found by one of the following:
      pacman won, pacman lost or there are no legal moves.

      Here are some method calls that might be useful when implementing minimax.

      gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

      gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

      gameState.getNumAgents():
        Returns the total number of agents in the game

      gameState.getScore():
        Returns the score corresponding to the current state of the game

      gameState.isWin():
        Returns True if it's a winning state

      gameState.isLose():
        Returns True if it's a losing state

      self.depth:
        The depth to which search should continue

    """

    # BEGIN_YOUR_CODE (our solution is 20 lines of code, but don't worry if you deviate from this)
    def getVal(s, d, agentIndex, evalFn = self.evaluationFunction):
      nextAgentIndex = 0 if agentIndex == s.getNumAgents() - 1 else agentIndex + 1
      actions = s.getLegalActions(agentIndex)
      if len(actions) == 0:
        return s.getScore()
      elif d == 0:
        if agentIndex != 0:
          raise Exception(f"Unexpected agentIndex {agentIndex} != {0}")
        return evalFn(s)
      elif agentIndex == 0:
        return max(getVal(s.generateSuccessor(agentIndex, a), d, nextAgentIndex) for a in actions)
      else:
        nextD = d - (1 if agentIndex == s.getNumAgents() - 1 else 0)
        return min(getVal(s.generateSuccessor(agentIndex, a), nextD, nextAgentIndex) for a in actions)

    targetVal = getVal(gameState, self.depth, 0)
    # print(f"MinimaxAgent value of state = {targetVal}")
    legalActions = gameState.getLegalActions(0)
    actions = [a for a in legalActions if getVal(gameState.generateSuccessor(0, a), self.depth, 1) == targetVal]

    return random.choice(actions)
    # END_YOUR_CODE

######################################################################################
# Problem 2a: implementing alpha-beta

class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your minimax agent with alpha-beta pruning (problem 2)
    You may reference the pseudocode for Alpha-Beta pruning here:
    en.wikipedia.org/wiki/Alpha%E2%80%93beta_pruning#Pseudocode
  """

  def getAction(self, gameState: GameState) -> str:
    """
      Returns the minimax action using self.depth and self.evaluationFunction
    """

    # BEGIN_YOUR_CODE (our solution is 36 lines of code, but don't worry if you deviate from this)
    def getVal(s, d, agentIndex, alpha = float('-inf'), beta = float('inf'), evalFn = self.evaluationFunction):
      nextAgentIndex = 0 if agentIndex == s.getNumAgents() - 1 else agentIndex + 1
      actions = s.getLegalActions(agentIndex)
      if len(actions) == 0:
        return s.getScore()
      elif d == 0:
        if agentIndex != 0:
          raise Exception(f"Unexpected agentIndex {agentIndex} != {0}")
        return evalFn(s)
      elif agentIndex == 0:
        maxVal = float('-inf')
        # actions.sort(key=lambda a: evalFn(s.generateSuccessor(agentIndex, a)), reverse=True)
        for a in actions:
          maxVal = max(maxVal, getVal(s.generateSuccessor(agentIndex, a), d, nextAgentIndex, alpha, beta))
          alpha = max(alpha, maxVal)
          if alpha >= beta:
            break
        return maxVal
      else:
        nextD = d - (1 if agentIndex == s.getNumAgents() - 1 else 0)
        minVal = float('inf')
        # actions.sort(key=lambda a: evalFn(s.generateSuccessor(agentIndex, a)), reverse=False)
        for a in actions:
          minVal = min(minVal, getVal(s.generateSuccessor(agentIndex, a), nextD, nextAgentIndex, alpha, beta))
          beta = min(beta, minVal)
          if alpha >= beta:
            break
        return minVal

    targetVal = getVal(gameState, self.depth, 0)
    # print(f"AlphaBetaAgent value of state = {targetVal}")
    legalActions = gameState.getLegalActions(0)
    actions = [a for a in legalActions if getVal(gameState.generateSuccessor(0, a), self.depth, 1) == targetVal]

    return random.choice(actions)
    # END_YOUR_CODE

######################################################################################
# Problem 3b: implementing expectimax

class ExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your expectimax agent (problem 3)
  """

  def getAction(self, gameState: GameState) -> str:
    """
      Returns the expectimax action using self.depth and self.evaluationFunction

      All ghosts should be modeled as choosing uniformly at random from their
      legal moves.
    """

    # BEGIN_YOUR_CODE (our solution is 20 lines of code, but don't worry if you deviate from this)
    def getVal(s, d, agentIndex, evalFn = self.evaluationFunction):
      nextAgentIndex = 0 if agentIndex == s.getNumAgents() - 1 else agentIndex + 1
      actions = s.getLegalActions(agentIndex)
      if len(actions) == 0:
        return s.getScore()
      elif d == 0:
        if agentIndex != 0:
          raise Exception(f"Unexpected agentIndex {agentIndex} != {0}")
        return evalFn(s)
      elif agentIndex == 0:
        return max(getVal(s.generateSuccessor(agentIndex, a), d, nextAgentIndex) for a in actions)
      else:
        nextD = d - (1 if agentIndex == s.getNumAgents() - 1 else 0)
        return sum((1 / len(actions)) * getVal(s.generateSuccessor(agentIndex, a), nextD, nextAgentIndex) for a in actions)

    targetVal = getVal(gameState, self.depth, 0)
    # print(f"MinimaxAgent value of state = {targetVal}")
    legalActions = gameState.getLegalActions(0)
    actions = [a for a in legalActions if getVal(gameState.generateSuccessor(0, a), self.depth, 1) == targetVal]

    return random.choice(actions)
    # END_YOUR_CODE

######################################################################################
# Problem 4a (extra credit): creating a better evaluation function

def betterEvaluationFunction(currentGameState: GameState) -> float:
  """
    Your extreme, unstoppable evaluation function (problem 4). Note that you can't fix a seed in this function.
  """

  # BEGIN_YOUR_CODE (our solution is 13 lines of code, but don't worry if you deviate from this)
  def getDistFromPacman(x, y):
    pacmanPos = currentGameState.getPacmanPosition()
    return abs(pacmanPos[0] - x) + abs(pacmanPos[1] - y)
  def exponentiallyWeightedScore(objectives):
    return sum(2 ** (-1 * getDistFromPacman(objectivePos[0], objectivePos[1])) for objectivePos in objectives)
  def getFoodScore(foodGrid, pacmanPosition):
    foodsPos = [(x, y) for y in range(foodGrid.height) for x in range(foodGrid.width) if foodGrid[x][y] == True]
    return exponentiallyWeightedScore(foodsPos)
  food = 9 * getFoodScore(currentGameState.getFood(), currentGameState.getPacmanPosition())
  ghostStates = currentGameState.getGhostStates()
  scary = sum(ghostState.scaredTimer for ghostState in ghostStates) > 0
  capsule = 100 if scary else 200 * exponentiallyWeightedScore(currentGameState.getCapsules())
  scaredGhostsPos = [ghostState.getPosition() for ghostState in ghostStates if ghostState.scaredTimer > 0]
  ghost = 200 * exponentiallyWeightedScore(scaredGhostsPos)
  score = currentGameState.getScore()
  return food + capsule + ghost + score
  # END_YOUR_CODE

# Abbreviation
better = betterEvaluationFunction
