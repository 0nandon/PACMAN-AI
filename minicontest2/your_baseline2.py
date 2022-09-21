# QapproximateAgents.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from captureAgents import CaptureAgent
import random, time, util
from game import Directions, Actions
from util import nearestPoint
import game
import copy
import math

from baseline import OffensiveReflexAgent
from your_baseline5 import AstarAttacker
import numpy as np
import os

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first='Attacker', second='DefensiveQAgent'):
  """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.
  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """

  # The following line is an example only; feel free to change it.
  return [eval(first)(firstIndex), eval(second)(secondIndex)]


##########
# Agents #
##########

class BaseAgent(CaptureAgent):
  def registerInitialState(self, gameState):
    self.start = gameState.getAgentPosition(self.index)
    CaptureAgent.registerInitialState(self, gameState)
    
    # maze의 size에 관한 정보
    self.height = gameState.data.layout.height
    self.width = gameState.data.layout.width
    self.midWidth = int(self.width / 2)
    
    self.util = Util(self) # 각종 feature 계산을 위한 utility function들이 들어있다.
    self.search = Search(self) # BFS, A-star
    self.boundaryPos = self.util.getValidBoundaryPosition(gameState) # 상대팀과 우리팀 사이의 경계선 좌표
    self.walls = gameState.getWalls() # 벽의 위치
    
    # 음식 관련 정보 processing
    self._initFoodsInfo(gameState)

  # ================= 음식을 쉬운 음식과 어려운 음식으로 구분한다. ================= #
  def _initFoodsInfo(self, gameState):
    foodsWithInfo = self.preprocessFood(gameState)
    self.getEasyFood(foodsWithInfo)
    self.getHardFood()
  
  def preprocessFood(self, gameState):
    foodsWithInfo = []
    self.foodsPos = self.getFood(gameState).asList()
    
    for food in self.foodsPos:
      validPos = []
      for moveX, moveY in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        if gameState.hasWall(food[0]+moveX, food[1]+moveY):
          continue
        
        validPos.append((food[0]+moveX, food[1]+moveY))
      foodsWithInfo.append((food, validPos))
      
    return foodsWithInfo
    
  def getEasyFood(self, foodsWithInfo):
    self.easyFood = []
    
    for foodInfo in foodsWithInfo:
      foodPos = foodInfo[0]
      validPos = foodInfo[1]
      
      if len(validPos) < 2:
        continue
      
      count = 0
      for pos in validPos:
        if self.search.BFS(pos, [foodPos], self.boundaryPos):
          count += 1
      
      if count > 1:
        self.easyFood.append(foodPos)
          
  def getHardFood(self):
    self.hardFood = []
    for food in self.foodsPos:
      if food in self.easyFood:
        continue
      
      self.hardFood.append(food)
  # ==================================================================== #
  
  def updateEasyFood(self, gameState):
    foodsPos = self.getFood(gameState).asList()
    updatedEasyFood = []
    
    for pos in foodsPos:
      if pos in self.easyFood:
        updatedEasyFood.append(pos)
        
    self.easyFood = updatedEasyFood
  
  def updateHardFood(self, gameState):
    foodsPos = self.getFood(gameState).asList()
    updatedHardFood = []
    
    for pos in foodsPos:
      if pos in self.hardFood:
        updatedHardFood.append(pos)
        
    self.hardFood = updatedHardFood
        
  def heuristicForAttacker(self, pos, gameState):
    dist2Ghost, _ = self.util.dist2Ghost(gameState)
    
    if dist2Ghost is not None:
      if dist2Ghost < 3:
        return (5 - dist2Ghost) * 9999
      return 0
    else:
      return 0

class Attacker(BaseAgent):  
  def FoodProblem(self, gameState):
    return PositionSearchProblem(self, gameState, option='food')
  
  def EasyFoodProblem(self, gameState):
    return PositionSearchProblem(self, gameState, option='easy_food')
  
  def HardFoodProblem(self, gameState):
    return PositionSearchProblem(self, gameState, option='hard_food')
  
  def CapsuleProblem(self, gameState):
    return PositionSearchProblem(self, gameState, option='capsule')
  
  def BackhomeProblem(self, gameState):
    return PositionSearchProblem(self, gameState, option='back_home')
  
  def EmergencyProblem(self, gameState):
    return PositionSearchProblem(self, gameState, option='emergency')
  
  def chooseAction(self, gameState):
    self.updateEasyFood(gameState)
    self.updateHardFood(gameState)
    
    dist2Ghost, ghostState = self.util.dist2Ghost(gameState)
    myState = gameState.getAgentState(self.index)
    opponentScaredTime = self.util.getOpponentScaredTime(gameState)
    
    if dist2Ghost is not None and dist2Ghost < 5:
        problem = self.EmergencyProblem(gameState)
    elif myState.numCarrying > 15 or gameState.data.timeleft < self.util.dist2Home(gameState) + 50:
      problem = self.BackhomeProblem(gameState)
    elif opponentScaredTime is not None and opponentScaredTime > 15 and len(self.hardFood) > 0:
        problem = self.HardFoodProblem(gameState)
    elif len(self.hardFood) > 0 and len(self.getCapsules(gameState)) != 0 and opponentScaredTime is not None and opponentScaredTime < 10:
      problem = self.CapsuleProblem(gameState)
    elif len(self.easyFood) > 0 and myState.numCarrying <3:
      problem = self.EasyFoodProblem(gameState)
    else:
      problem = self.FoodProblem(gameState)
        
    actions = self.search.aStarSearch(problem, self.heuristicForAttacker)
    if len(actions) == 0:
      return Directions.STOP
    return actions[0]

#######################
#   DefensiveQAgent   #
#######################

class QapproximateAgents(CaptureAgent):

  def __init__(self, index, timeForComputing=.1, numTraining=0, epsilon=0.5, lr=0.5, gamma=1, update='AdaGrad', **args):
    CaptureAgent.__init__(self, index, timeForComputing)    
    self.numTraining = int(numTraining)
    self.epsilon = float(epsilon)
    self.lr = float(lr)
    self.discount = float(gamma)
    
    self.qValues = util.Counter()
    self.nValues = util.Counter() # for exploration function
    
    self.updatePolicy = eval(update)(lr) # GD, AdaGrad, Adam, Newton
    self.util = Util(self) # features를 계산할 때 사용되는 여러 함수들
    
  def _initMaze(self, gameState):
    # maze size 관련 변수
    self.width = gameState.data.layout.width
    self.midWidth = self.width / 2
    self.height = gameState.data.layout.height
    self.boundaryPos = self.util.getValidBoundaryPosition(gameState)

  def registerInitialState(self, gameState):
    self.start = gameState.getAgentPosition(self.index)
    self._initMaze(gameState)
    
    self.episodeLoss = []
    CaptureAgent.registerInitialState(self, gameState)
    self.startEpisode()

  # 에피소드 시작
  def startEpisode(self):
    self.lastState = None
    self.lastAction = None

  def final(self, state):
    CaptureAgent.final(self, state)
    deltaReward = state.getScore() - self.lastState.getScore()
    self.update(self.lastState, self.lastAction, state, deltaReward)

class DefensiveQAgent(QapproximateAgents):

  def __init__(self, index, timeForComputing=.1, **args):
    QapproximateAgents.__init__(self, index, timeForComputing, **args)
    self.lastEatenFoodPosition = None
    self.weights = self.getInitWeights()
    self.search = Search(self)
    
    # for training
    self.loss = []
    
    # # initialize weights
    if self.numTraining == 0:
      self.epsilon = 0.0 # no exploration
      self.lr = 0.01 # no learning
      self.weights = self.getInitWeights() # 가중치를 일정 숫자로 초기화하여 학습 속도를 늘린다.

  # 가중치 반환
  def getInitWeights(self):
    return util.Counter({'numInvaders': -1000.0,
                         'onDefense': 100.0,
                         'invaderDistance': 500.0,
                         'stop': -100.0,
                         'reverse': -2.0,
                         'DistToCapsules': 200.0,})
  
  # 학습된 가중치
  def getGoodWeights(self):
    
    """
    dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(dir, 'weights/weight.npz')
    trained_weights = np.load(path)
    print(dict(trained_weights))
    """
    return util.Counter({'numInvaders': -967.81473287,
                         'onDefense': 117.10239798,
                         'invaderDistance': 529.21394498,
                         'stop': -98.64601033,
                         'reverse': 11.7778266,
                         'DistToCapsules': 207.47689023,})
  
  def load_weights(self, path):
    trained_weights = np.load(path)
    return dict(trained_weights)

  def save_weights(self, path, weights):
    if os.path.exists(path):
        os.remove(path)
  
    np.savez_compressed(path, **weights)

  def storeActionCash(self, gameState, action):
    self.lastState = gameState
    self.lastAction = action

  # 매 액션마다 호출되어 업데이트를 수행한다.
  def observationFunction(self, state):
    if not self.lastState is None:
      reward = 0.0
      self.update(self.lastState, self.lastAction, state, reward)
    return CaptureAgent.observationFunction(self, state)
  
  # ================== exploration function ================== #
  def epsilonGreedy(self, gameState):
    actions = gameState.getLegalActions(self.index)
    action = None
    
    if util.flipCoin(self.epsilon):
      action = random.choice(actions)
    else:
      _, action = self.computeActionFromQValues(gameState)
    return action
  
  def exploreFunction(self, gameState, k=3):
    qValues, _ = self.computeActionFromQValues(gameState)
    actions = gameState.getLegalActions(self.index)
    
    exploreValues = [qValue + k / (self.nValues[(gameState, action)]+1e-7) for qValue, action in zip(qValues, actions)]
    maxValue = max(exploreValues)
    bestActions = [action for action, value in zip(actions, exploreValues) if value == maxValue]
    return random.choice(bestActions)
  # ========================================================= #
  
  def FoodProblem(self, gameState):
    return PositionSearchProblem(self, gameState, option='food')
  
  def BackhomeProblem(self, gameState):
    return PositionSearchProblem(self, gameState, option='back_home')
  
  def EmergencyProblem(self, gameState):
    return PositionSearchProblem(self, gameState, option='emergency')
  
  def heuristicForAttacker(self, pos, gameState):
    dist2Ghost, _ = self.util.dist2Ghost(gameState)
    
    if dist2Ghost is not None:
      if dist2Ghost < 3:
        return (5 - dist2Ghost) * 9999
      return 0
    else:
      return 0
  
  # choose action
  def chooseAction(self, gameState, option='explore_function'):
    opponents = [gameState.getAgentState(agent) for agent in self.getOpponents(gameState)]
    pacman = [opponent for opponent in opponents if opponent.isPacman]
    opponentScaredTime = self.util.getOpponentScaredTime(gameState)
    myState = gameState.getAgentState(self.index)
    
    problem = None
    if opponentScaredTime is not None and opponentScaredTime > 20:
      problem = self.FoodProblem(gameState)
    elif opponentScaredTime is not None and opponentScaredTime < self.util.dist2Home(gameState) + 15:
      problem = self.BackhomeProblem(gameState)
    elif len(pacman) < 1 and myState.numCarrying < 3:
      dist2Ghost, _ = self.util.dist2Ghost(gameState)
      if self.util.dist2Home(gameState) > 7 or myState.numCarrying > 4 or (dist2Ghost is not None and dist2Ghost < 5):
        problem = self.EmergencyProblem(gameState)
      else:
        problem = self.FoodProblem(gameState)
    
    # Q learning으로 다음 액션을 결정
    if problem is not None:
      actions = self.search.aStarSearch(problem, self.heuristicForAttacker)
      if len(actions) == 0:
        return Directions.STOP
      return actions[0]
    else:
      if option == 'epsilon_greedy':
        action = self.epsilonGreedy(gameState)
      elif option == 'explore_function':
        action = self.exploreFunction(gameState)
    
      self.storeActionCash(gameState, action)
      self.nValues[(gameState, action)] += 1
      return action
  
  def computeActionFromQValues(self, gameState):
    actions = gameState.getLegalActions(self.index)
    values = [self.getQvalue(gameState, action) for action in actions]

    maxValue = max(values)
    bestActions = [action for action, value in zip(actions, values) if value == maxValue]
    return values, random.choice(bestActions)

  def getSuccessor(self, gameState,  action):
    """Finds the next successor which is a grid position (location tuple)."""
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != nearestPoint(pos):
      # Only half a grid position was covered
      return successor.generateSuccessor(self.index, action)
    else:
      return successor

  @property
  def getWeights(self):
    return self.weights

  def getQvalue(self, gameState, action):
      weights = self.getWeights
      features = self.getFeatures(gameState, action)
      return weights * features
    
  # features 값 초기화
  def initializeFeatures(self):
    features = util.Counter()
    features['invaderDistance'] = 0.0
    features['onDefense'] = 1.0
    features['numInvaders'] = 0.0
    features['stop'] = 0.0
    features['reverse'] = 0.0
    features['DistToCapsules'] = 0.0
    return features
  
  def getFeatures(self, gameState, action):
    features = self.initializeFeatures()
    successor = self.getSuccessor(gameState, action)
    
    myState = successor.getAgentState(self.index)
    myPosition = myState.getPosition()
    
    if myState.isPacman:
      features['onDefense'] = 0.0

    # baseline.py의 getFeatures() 함수의 코드를 사용
    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
    features['numInvaders'] = float(len(invaders))
    if len(invaders) > 0:
      dists = [self.getMazeDistance(myPosition, a.getPosition()) for a in invaders]
      features['invaderDistance'] = float(1/min(dists))
      if gameState.getAgentState(self.index).scaredTimer > 0:
        features['invaderDistance'] = float(-1/min(dists))
        
    if action == Directions.STOP: features['stop'] = 1
    rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
    if action == rev: features['reverse'] = 1
    
    # 자기 구역의 캡슐까지의 거리를 계산하여 feature로 사용
    dist2Capsules = self.util.dist2CapsulesForDefender(gameState)
    if dist2Capsules is not None:
      if dist2Capsules == 0:
        features['DistToCapsules'] = 1.0
      else:
        features['DistToCapsules'] = float(1/dist2Capsules)
    
    return features

  # 가중치 업데이트
  def update(self, state, action, nextState, reward):
    actions = nextState.getLegalActions(self.index)
    values = [self.getQvalue(nextState, action) for action in actions]
    maxValue = max(values)
    weights = self.getWeights
    features = self.getFeatures(state, action)
    
    for feature in features:
      y = (reward + self.discount * maxValue)
      yhat = self.getQvalue(state, action)
      difference = y - yhat
      
      # scared일 때는 가중치를 업데이트 하지 않는다.
      scared = self.util.getOpponentScaredTime(nextState)
      if scared is None:
        self.updatePolicy.update(weights, features, feature, difference)
      
      self.episodeLoss.append(self.mse(y, yhat))
    
  # loss function
  def mse(self, y, yhat):
    return (y - yhat) ** 2
      
  def final(self, state):
    QapproximateAgents.final(self, state)
    
    # 가중치 저장
    """
    dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(dir, 'weights/weight.npz')
    self.save_weights(path, self.getWeights)
    """


########
# util #
########

class Util:
  def __init__(self, agent):
    self.agent = agent
    
  def getValidBoundaryPosition(self, gameState):
    boundaryPosition = []
    
    if self.agent.red:
      xPos = int(self.agent.midWidth - 1)
    else:
      xPos = int(self.agent.midWidth + 1)
      
    for yPos in range(self.agent.height):
      if gameState.hasWall(xPos, yPos):
        continue
      
      boundaryPosition.append((xPos, yPos))
    return boundaryPosition
  
  def dist2Invader(self, state):
    # 내 위치 추출
    myState = state.getAgentState(self.agent.index)
    myPosition = myState.getPosition()
    
    # 상대방 invader 위치 추출
    enemies = [state.getAgentState(enemy) for enemy in self.agent.getOpponents(state)]
    invaders = [enemy for enemy in enemies if (enemy.isPacman and enemy.getPosition() != None)]
    
    # 나와 invader 사이의 거리를 배열로 저장하여 반환
    dists = [self.agent.getMazeDistance(myPosition, invader.getPosition()) for invader in invaders]
    if len(dists) == 0:
      return None

    return dists
  
  def dist2Home(self, gameState):
    myState = gameState.getAgentState(self.agent.index)
    myPosition = myState.getPosition()
    
    dist2Home = [self.agent.getMazeDistance(pos, myPosition) for pos in self.agent.boundaryPos]
    return min(dist2Home)
  
  def dist2Food(self, gameState):
    myState = gameState.getAgentState(self.agent.index)
    myPosition = myState.getPosition()
    remainFoods = self.agent.getFood(gameState).asList() # 남아있는 음식 좌표
    
    if len(self.agent.foodsPos) > 0:
      dist2Food = [self.agent.getMazeDistance(pos, myPosition) for pos in remainFoods]
      return min(dist2Food)
    else:
      return None
  
  def dist2Capsules(self, gameState):
    myState = gameState.getAgentState(self.agent.index)
    myPosition = myState.getPosition()
    remainCapsules = self.agent.getCapsules(gameState) # 남아있는 음식 좌표
    
    if len(remainCapsules) > 0:
      dist2Capsules = [self.agent.getMazeDistance(pos, myPosition) for pos in remainCapsules]
      return min(dist2Capsules)
    return None
  
  # 자신의 구역에 있는 가장 가까운 캡슐까지의 거리 반환
  def dist2CapsulesForDefender(self, gameState):
    myState = gameState.getAgentState(self.agent.index)
    myPosition = myState.getPosition()
    
    if self.agent.red:
      remainCapsules = gameState.getRedCapsules()
    else:
      remainCapsules = gameState.getBlueCapsules()
    
    if len(remainCapsules) > 0:
      dist2Capsules = [self.agent.getMazeDistance(pos, myPosition) for pos in remainCapsules]
      return min(dist2Capsules)
    return None
  
  def dist2Ghost(self, gameState):
    myState = gameState.getAgentState(self.agent.index)
    myPosition = myState.getPosition()
    
    opponents = [gameState.getAgentState(opponent) for opponent in self.agent.getOpponents(gameState)]
    ghosts = [agent for agent in opponents if not agent.isPacman and agent.getPosition() != None and agent.scaredTimer < 5]
    
    if len(ghosts) > 0:
      dist2Ghost = [self.agent.getMazeDistance(myPosition, ghost.getPosition()) for ghost in ghosts]
      minDist = min(dist2Ghost)
      state = [ghosts[idx] for idx, dist in enumerate(dist2Ghost) if dist == minDist]
      return (minDist, state)
    else:
      return None, None
  
  def getOpponentScaredTime(self, gameState):
    opponents = self.agent.getOpponents(gameState)
    
    for opponent in opponents:
      if gameState.getAgentState(opponent).scaredTimer > 1:
        return gameState.getAgentState(opponent).scaredTimer
      
    return None

  
#################
# problem class #
#################

class SearchProblem:
  def getStartState(self):
    util.raiseNotDefined()

  def isGoalState(self, state):
    util.raiseNotDefined()

  def getSuccessors(self, state):
    util.raiseNotDefined()

  def getCostOfActions(self, actions):
    util.raiseNotDefined()
        
class PositionSearchProblem(SearchProblem):
  def __init__(self, agent, gameState, costFn = lambda x: 1, option="food"):
    self.agent = agent
    self.gameState = gameState
    self.walls = gameState.getWalls()
    self.startState = gameState.getAgentPosition(self.agent.index)
    self.costFn = costFn
    
    opponents = [gameState.getAgentState(enemy) for enemy in self.agent.getOpponents(gameState)]
    self.invaderPos = [opponent.getPosition() for opponent in opponents if opponent.isPacman and opponent.getPosition() is not None]
    self.setGoalState(gameState, option)
      
  def setGoalState(self, gameState, option="food"):
    if option == 'food':
      self.goalState = self.agent.getFood(gameState).asList() + self.agent.getCapsules(gameState)
    elif option == 'easy_food':
      self.goalState = self.agent.easyFood
    elif option == 'hard_food':
      self.goalState = self.agent.hardFood
    elif option == 'capsule':
      self.goalState = self.agent.getCapsules(gameState)
    elif option == 'back_home':
      self.goalState = self.agent.boundaryPos
    elif option == 'emergency':
      self.goalState = self.agent.boundaryPos + self.agent.getCapsules(gameState)
    elif option == 'invader':
      self.goalState = self.invaderPos
        
  def getStartState(self):
    return self.startState
  
  def isGoalState(self, state):
    return state in self.goalState

  def getSuccessors(self, state):
    successors = []
    for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
      x,y = state
      dx, dy = Actions.directionToVector(action)
      nextx, nexty = int(x + dx), int(y + dy)
      if not self.walls[nextx][nexty]:
        nextState = (nextx, nexty)
        cost = self.costFn(nextState)
        successors.append( ( nextState, action, cost) )

    return successors

  def getCostOfActions(self, actions):
    """
    Returns the cost of a particular sequence of actions. If those actions
    include an illegal move, return 999999.
    """
    if actions == None: return 999999
    x,y= self.getStartState()
    cost = 0
    for action in actions:
      # Check figure out the next state and see whether its' legal
      dx, dy = Actions.directionToVector(action)
      x, y = int(x + dx), int(y + dy)
      if self.walls[x][y]: return 999999
      cost += self.costFn((x,y))
    return cost

    
####################
# search algorithm #
####################

class Search:
  def __init__(self, agent):
    self.agent = agent
    
  # code from first assignment
  def getSuccessors(self, state):
    successors = []
    for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
      x,y = state
      
      from game import Actions
      dx, dy = Actions.directionToVector(action)
      nextx, nexty = int(x + dx), int(y + dy)
      if not self.agent.walls[nextx][nexty]:
        nextState = (nextx, nexty)
        successors.append(nextState)
      
    return successors
  
  # basic BFS
  def BFS(self, pos, visited, goalState):
    from util import Queue
    
    queue = Queue()
    queue.push(pos)
    while not queue.isEmpty():
      pos = queue.pop()
      visited.append(pos)
      
      if pos in goalState:
        return True
      
      for successor in self.getSuccessors(pos):
        if successor not in visited:
          visited.append(successor)
          queue.push(successor)
      
    return False
  
  def nullHeuristic(self, pos, gameState):
    return 0
  
  # astar code from my first assignment
  def aStarSearch(self, problem, heuristic=nullHeuristic):
    heap = NewPriorityQueue()
    state = problem.getStartState()
    heap.push(state, 0)
    
    found, distance = {}, {}
    distance[state] = 0
    actions = []
    while not heap.isEmpty():
      state = heap.pop()
        
      if problem.isGoalState(state):
        break
        
      nexts = problem.getSuccessors(state)
      for next in nexts:
        if next[0] not in distance or distance[next[0]] > distance[state] + next[2]:
          distance[next[0]] = distance[state] + next[2]
          heap.update(next[0], distance[next[0]] + heuristic(next[0], problem.gameState))
          found[next[0]] = (state, next[1])
    
    state = (state,)
    while state[0] != problem.getStartState():
      state = found[state[0]]
      actions.append(state[1])
        
    return list(reversed(actions))
  

#####################
# Training strategy #
#####################

# 여러 가중치 업데이트 전략을 코드로 구현
class GD:
  def __init__(self, lr):
    self.lr = lr
    
  def update(self, weights, features, feature, difference):
    grad = difference * features[feature]
    weights[feature] = weights[feature] + self.lr * grad
    
class AdaGrad:
  def __init__(self, lr):
    self.lr = lr
    self.h = None
    
  def update(self, weights, features, feature, difference):
    if self.h is None:
      self.h = util.Counter()
      
      for feature in features:
        self.h[feature] = 0.0
    
    grad = difference * features[feature]
    self.h[feature] += grad * grad
    weights[feature] = weights[feature] + self.lr * grad / (math.sqrt(self.h[feature]) + 1e-7)

class Adam:
  def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
      self.lr = lr
      self.beta1 = beta1
      self.beta2 = beta2
      self.iter = 0
      self.m = None
      self.v = None
        
  def update(self, weights, features, feature, difference):
    if self.m is None:
      self.m, self.v = util.Counter(), util.Counter()
      for feature in features:
        self.m[feature] = 0.0
        self.v[feature] = 0.0
    
    if feature == 'DistToBoundary':
      self.iter += 1
      
    lr_t  = self.lr * np.sqrt(1.0 - self.beta2**self.iter) / (1.0 - self.beta1**self.iter)
    
    grad = -1 * difference * features[feature]
    self.m[feature] += (1 - self.beta1) * (grad - self.m[feature])
    self.v[feature] += (1 - self.beta2) * (grad**2 - self.v[feature])
    weights[feature] -= lr_t * self.m[feature] / (np.sqrt(self.v[feature]) + 1e-7)

  
class Newton:
  def __init__(self, lr):
    pass
  
  def update(self, weights, features, feature, difference):
    grad = -1 * difference * features[feature]
    weights[feature] = weights[feature] - grad / (features[feature] ** 2 + 1e-7)
    
##################
# data structure #
##################

from util import PriorityQueue
import heapq

class NewPriorityQueue(PriorityQueue):

  def update(self, item, priority):
    for index, (p, c, i) in enumerate(self.heap):
      if i == item:
        if p <= priority:
          break
        del self.heap[index]
        self.heap.append((priority, c, item))
        heapq.heapify(self.heap)
        break
    else:
      self.push(item, priority)