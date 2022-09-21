# myTeam.py
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
from game import Directions
from game import Actions

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'Attacker', second = 'Defender'):
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
  
  # 쉬운 음식 추출
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
  
  # 어려운 음식 추출 
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
  
  # a-star 알고리즘을 위한 휴리스틱
  # ghost와 일정 거리 이하로 가까워지면 굉장히 높은 숫자를 반환하여, ghost를 피하도록 유도
  def heuristicForAttacker(self, pos, gameState):
    dist2Ghost, _ = self.util.dist2Ghost(gameState)
    
    if dist2Ghost is not None:
      if dist2Ghost < 3:
        return (5 - dist2Ghost) ** 3
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
    # 매 액션마다 남아있는 쉬운 음식과 어려운 음식을 업데이트 해준다.
    self.updateEasyFood(gameState)
    self.updateHardFood(gameState)
    
    dist2Ghost, _ = self.util.dist2Ghost(gameState)
    myState = gameState.getAgentState(self.index)
    opponentScaredTime = self.util.getOpponentScaredTime(gameState)
    
    # 상황에 따라 맞는 문제를 할당
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

class Defender(BaseAgent):
  def FoodProblem(self, gameState):
    return PositionSearchProblem(self, gameState, option='food')
  
  def BackhomeProblem(self, gameState):
    return PositionSearchProblem(self, gameState, option='back_home')
  
  def EmergencyProblem(self, gameState):
    return PositionSearchProblem(self, gameState, option='emergency')
  
  def InvaderProblem(self, gameState):
    return PositionSearchProblem(self, gameState, option='invader')
  
  def chooseAction(self, gameState):
    opponents = [gameState.getAgentState(agent) for agent in self.getOpponents(gameState)]
    pacman = [opponent for opponent in opponents if opponent.isPacman]
    opponentScaredTime = self.util.getOpponentScaredTime(gameState)
    myState = gameState.getAgentState(self.index)
    
    # 상황에 따라 맞는 문제를 할당
    if len(pacman) < 1 and opponentScaredTime is not None and opponentScaredTime > 20:
      problem = self.FoodProblem(gameState) # attacker 역할을 한다.
    elif len(pacman) < 1 and opponentScaredTime is not None and opponentScaredTime < self.util.dist2Home(gameState) + 15:
      problem = self.BackhomeProblem(gameState)
    elif len(pacman) < 1 and myState.numCarrying < 3:
      dist2Ghost, _ = self.util.dist2Ghost(gameState)
      if self.util.dist2Home(gameState) > 7 or myState.numCarrying > 4 or (dist2Ghost is not None and dist2Ghost < 5):
        problem = self.EmergencyProblem(gameState)
      else:
        problem = self.FoodProblem(gameState) # attacker 역할을 한다.
    else:
      problem = self.InvaderProblem(gameState) # defender 역할을 한다.
      
    actions = self.search.aStarSearch(problem, self.heuristicForAttacker)
    if len(actions) == 0:
      return Directions.STOP
    return actions[0]
      
#################
# problem class #
#################

# a-star 알고리즘에 할당하기 위한 problem 객체

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

# BFS, A-star 알고리즘 등이 구현되어 있다.

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
  

#################
# util function #
#################

# 맵 내의 상황을 파악하기 위한 여러 함수들이 구현되어 있다.

class Util:
  def __init__(self, agent):
    self.agent = agent
  
  # red team과 blue team의 경계면 좌표 반환
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
  
  # 자신의 구역까지의 최소 거리
  def dist2Home(self, gameState):
    myState = gameState.getAgentState(self.agent.index)
    myPosition = myState.getPosition()
    
    dist2Home = [self.agent.getMazeDistance(pos, myPosition) for pos in self.agent.boundaryPos]
    return min(dist2Home)
  
  # 가장 가까이에 있는 음식까지의 거리
  def dist2Food(self, gameState):
    myState = gameState.getAgentState(self.agent.index)
    myPosition = myState.getPosition()
    remainFoods = self.agent.getFood(gameState).asList() # 남아있는 음식 좌표
    
    if len(self.agent.foodsPos) > 0:
      dist2Food = [self.agent.getMazeDistance(pos, myPosition) for pos in remainFoods]
      return min(dist2Food)
    else:
      return None
  
  # 가장 가까이에 있는 캡슐까지의 거리
  def dist2Capsules(self, gameState):
    myState = gameState.getAgentState(self.agent.index)
    myPosition = myState.getPosition()
    remainCapsules = self.agent.getCapsules(gameState) # 남아있는 음식 좌표
    
    if len(remainCapsules) > 0:
      dist2Capsules = [self.agent.getMazeDistance(pos, myPosition) for pos in remainCapsules]
      return min(dist2Capsules)
    return None
  
  # 가장 가까이에 있는 ghost까지의 거리
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
  
  # 상대편의 남은 scared time
  def getOpponentScaredTime(self, gameState):
    opponents = self.agent.getOpponents(gameState)
    
    for opponent in opponents:
      if gameState.getAgentState(opponent).scaredTimer > 1:
        return gameState.getAgentState(opponent).scaredTimer
      
    return None
    
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