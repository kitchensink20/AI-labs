from util import manhattanDistance
from game import Directions
import random, util
from typing import Any, DefaultDict, List, Set, Tuple

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

  def getAction(self, gameState):
    depth = 0
    return self.maxValue(gameState, depth)[1]

  def maxValue (self, state: GameState, depth, agent=0):
    actions = self.getActions(state, agent)

    if not actions or self.isEnd(state) or self.depth <= depth:
      return self.eval(state), Directions.STOP
    
    max_cost = float('-inf')
    max_action = Directions.STOP

    for action in actions:
      successorState = state.generateSuccessor(agent, action)
      cost = self.minValue(successorState, depth, agent+1)[0]
      if cost > max_cost:
        max_cost = cost
        max_action = action
    
    return max_cost, max_action

  def minValue(self, state: GameState, depth, agent):
    actions = self.getActions(state, agent)
    numAgents = state.getNumAgents()

    if not actions or self.isEnd(state) or depth >= self.depth:
      return self.eval(state), Directions.STOP
    
    min_cost = float('inf')
    min_action = Directions.STOP

    for action in actions:
      successorState = state.generateSuccessor(agent, action)
      cost = 0
      if(numAgents - 1) != agent:
        cost = self.minValue(successorState, depth + 1, agent + 1)[0]
      else:
        cost = self.maxValue(successorState, depth + 1)[0]

      if cost < min_cost:
        min_cost = cost
        min_action = action

    return min_cost, min_action
    
  def getActions(self, state: GameState, agent_index):
    return state.getLegalActions(agent_index)

  def eval(self, state: GameState):
    return betterEvaluationFunction(state)
  
  def isEnd(self, state: GameState):
    return state.isWin() or state.isLose()

######################################################################################
# Problem 4a (extra credit): creating a better evaluation function

def betterEvaluationFunction(currentGameState: GameState) -> float:
  currentPacmanPosition = currentGameState.getPacmanPosition()
  food = currentGameState.getFood().asList()
  capsules = currentGameState.getCapsules()
  ghostStates = currentGameState.getGhostStates()

  huntingGhosts = [ghost for ghost in ghostStates if not ghost.scaredTimer]
  scaredGhosts = [ghost for ghost in ghostStates if ghost.scaredTimer > 0]
  leftScaredTimes = [ghost.scaredTimer for ghost in scaredGhosts]
  remainingFood = len(food)
  remainingCapsules = len(capsules)
  distToClosestFood = float("inf")
  for foodPos in food:
    dist = util.manhattanDistance(currentPacmanPosition, foodPos)
    if dist < distToClosestFood:
      distToClosestFood = dist

  distToClosestCapsule = float("inf")
  invDistToClosestCapsule = 0
  if remainingCapsules == 0:
    distToClosestCapsule = 0
  for capsulePos in capsules:
    dist = util.manhattanDistance(currentPacmanPosition, capsulePos)
    if dist < distToClosestCapsule:
      distToClosestCapsule = dist

  distToClosestHuntingGhost = float("inf")
  if len(huntingGhosts) != 0:
    for huntingGhost in huntingGhosts:
      huntingGhostPos = huntingGhost.getPosition()
      dist = util.manhattanDistance(currentPacmanPosition, huntingGhostPos)
      if dist < distToClosestHuntingGhost:
        distToClosestHuntingGhost = dist

  distToClosestScaredGhost = float("inf")
  leftScaredTime = 0
  for ghost, time in zip(scaredGhosts, leftScaredTimes):
   dist = util.manhattanDistance(currentPacmanPosition, ghost.getPosition())
   if dist < distToClosestScaredGhost:
     distToClosestScaredGhost = dist
     leftScaredTime = time

  if distToClosestHuntingGhost > 0:
    inverseDistToClosestHuntingGhost = 1.0 / distToClosestHuntingGhost 
  else:
    inverseDistToClosestHuntingGhost = 0

  if distToClosestScaredGhost > 0:
    inverseDistToClosestScaredGhost = 1.0 / distToClosestScaredGhost
  else:
    inverseDistToClosestScaredGhost = 0

  evaluation = currentGameState.getScore() \
         - 2 * inverseDistToClosestHuntingGhost \
         + 15 * leftScaredTime * inverseDistToClosestScaredGhost \
         - 2 * remainingFood \
         - 3 * invDistToClosestCapsule \
         - 1 * distToClosestFood

  return evaluation 
  
# Abbreviation
better = betterEvaluationFunction
