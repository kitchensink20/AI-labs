# mdpAgents.py
# parsons/15-oct-2017
#
# Version 1
#
# The starting point for CW2.
#
# Intended to work with the PacMan AI projects from:
#
# http://ai.berkeley.edu/
#
# These use a simple API that allow us to control Pacman's interaction with
# the environment adding a layer on top of the AI Berkeley code.
#
# As required by the licensing agreement for the PacMan AI we have:
#
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

# The agent here is was written by Simon Parsons, based on the code in
# pacmanAgents.py

from pacman import Directions
from game import Agent
import api
import random
import game
import util
import sys
import copy

class MDPAgent(Agent):

    def __init__(self):
        self.__TOP_RIGHT_WALL_CORNER_smallGrid = (6, 6)
        self.__TOP_RIGHT_WALL_CORNER_mediumClassic = (19, 10)
        self.__CAPSULE = ("CP", 50.0)
        self.__FOOD = ("FD", 10.0)
        self.__WALL = ("WL", 0.0)
        self.__FREE = ("FR", 0.0)
        self.__EDIBLE = ("EG", 400.0)
        self.__HOSTILE = ("HG", -500.0)
        self.__CONVERGENT = ("CV", None)
        self.__SAFETY_DISTANCE = 4
        self.__THREAT_DECAY_RATE = 50.0
        self.__DISCOUNT_FACTOR = 0.7
        self.__CONVERGENCE_TOLERANCE = 0.1
        self.__NORMAL_EARLY_STOPPING_POINT = 100
        self.__SPARSE_EARLY_STOPPING_POINT = 200
        self.__INACTIVE_GHOSTBUSTER_MODE = "inactive"
        self.__DEFENSIVE_GHOSTBUSTER_MODE = "defensive"
        self.__OFFENSIVE_GHOSTBUSTER_MODE = "offensive"
        self.__GHOSTBUSTER_MODE = self.__INACTIVE_GHOSTBUSTER_MODE
        self.__states = None
        self.__capsules = None
        self.__foods = None
        self.__walls = None
        self.__corners = None
        self.__floors = None
        self.__neighbors = None
        self.__rewards = None
        self.__utilities = None
        self.__counter = 0
        self.__round = 1

   
    def registerInitialState(self, state):
        print("Round " + str(self.__round) + " running...")
        corners = api.corners(state)
        # optimal parameter setting for smallGrid
        if self.__TOP_RIGHT_WALL_CORNER_smallGrid in corners:
            self.__SAFETY_DISTANCE = 2
            self.__DISCOUNT_FACTOR = 0.6
            self.__GHOSTBUSTER_MODE = self.__INACTIVE_GHOSTBUSTER_MODE
        # optimal parameter setting for mediumClassic
        elif self.__TOP_RIGHT_WALL_CORNER_mediumClassic in corners:
            self.__SAFETY_DISTANCE = 4
            self.__DISCOUNT_FACTOR = 0.7
            self.__GHOSTBUSTER_MODE = self.__INACTIVE_GHOSTBUSTER_MODE


    def final(self, state, log_mode=False, filename="./testing_data_1/log_1.csv"):
        self.__states = None
        self.__capsules = None
        self.__foods = None
        self.__walls = None
        self.__corners = None
        self.__floors = None
        self.__neighbors = None
        self.__rewards = None
        self.__utilities = None
        self.__counter = 0
        self.__round += 1
        if log_mode:
            with open(filename, "a") as log_file:
                entry = ""
                entry += str(self.__SAFETY_DISTANCE) + ", "
                entry += str(self.__DISCOUNT_FACTOR) + ", "
                entry += str(self.__CONVERGENCE_TOLERANCE) + ", "
                entry += str(self.__NORMAL_EARLY_STOPPING_POINT) + ", "
                entry += str(self.__SPARSE_EARLY_STOPPING_POINT) + ", "
                entry += str(self.__GHOSTBUSTER_MODE) + ", "
                if state.isWin():
                    entry += "win, "
                elif state.isLose():
                    entry += "lose, "
                else:
                    entry += ", "
                entry += str(state.getScore()) + "\n"
                log_file.write(entry)

    
    def __mapping_operation(self, state):
        # first step of each round: populate internal memories
        if self.__states == None:
            self.__states = []
        if self.__capsules == None:
            self.__capsules = set(api.capsules(state))
        if self.__foods == None:
            self.__foods = set(api.food(state))
        if self.__walls == None:
            self.__walls = api.walls(state)
        if self.__corners == None:
            self.__corners = api.corners(state)
            # calculate the real corners that the agent can move to
            for i in range(len(self.__corners)):
                x = self.__corners[i][0]
                y = self.__corners[i][1]
                if x == 0:
                    x += 1
                else:
                    x -= 1
                if y == 0:
                    y += 1
                else:
                    y -= 1
                self.__corners[i] = (x, y)
        if self.__floors == None:
            self.__floors = []
            x_coordinates = []
            y_coordinates = []
            for wall in self.__walls:
                x_coordinates.append(wall[0])
                y_coordinates.append(wall[1])
            x_minimum, x_maximum = min(x_coordinates), max(x_coordinates)
            y_minimum, y_maximum = min(y_coordinates), max(y_coordinates)
            for x in range(x_minimum, x_maximum + 1):
                for y in range(y_minimum, y_maximum + 1):
                    if (x, y) not in self.__walls:
                        self.__floors.append((x, y))
        if self.__neighbors == None:
            self.__neighbors = dict()
            # displacements: [East, West, North, South]
            displacements = {(1, 0): Directions.EAST, (-1, 0): Directions.WEST, (0, 1): Directions.NORTH, (0, -1): Directions.SOUTH}
            for floor in self.__floors:
                self.__neighbors[floor] = dict()
                for displacement, direction in displacements.items():
                    neighbor = (floor[0] + displacement[0], floor[1] + displacement[1])
                    self.__neighbors[floor][direction] = neighbor
        # log game state history
        self.__states.append(state)
        # the location of agent
        agent_location = api.whereAmI(state)
        # this location won't have capsule any more in the future
        if agent_location in self.__capsules:
            self.__capsules.remove(agent_location)
        # this location won't have food any more in the future
        if agent_location in self.__foods:
            self.__foods.remove(agent_location)
        # if at a corner: set the target corner to another one
        if agent_location in self.__corners:
            self.__counter = self.__corners.index(agent_location)
            self.__counter += random.choice([1, 2, 3])

    
    def __initialize_data_structures(self, targets, target_value, walls, floors, ghosts, safety_distance=4, threat_decay_rate=50.0):
        # initialize empty data structures
        rewards = dict()
        utilities = dict()
        # assign 0.0 reward value to each wall
        # assign 0.0 utility value to each wall
        for wall in walls:
            rewards[wall] = self.__WALL
            utilities[wall] = self.__WALL
        # assign 0.0 reward value to each free space
        # assign +10.0 reward value to each target
        # assign 0.0 utility value each floor
        for floor in floors:
            rewards[floor] = self.__FREE
            if floor in targets:
                rewards[floor] = target_value
            utilities[floor] = self.__FREE
        # assign -500.0 reward value to each ghost
        # assign decaying threat reward values recursively to locations surrounding each ghost
        for ghost in ghosts:
            ghost = (int(ghost[0]), int(ghost[1]))
            rewards[ghost] = self.__HOSTILE
            queue = util.Queue()
            queue.push(ghost)
            visited = set([ghost])
            while not queue.isEmpty():
                curr_location = queue.pop()
                if rewards[curr_location][1] >= (self.__HOSTILE[1] + threat_decay_rate * (safety_distance - 1)) or rewards[curr_location][1] >= (0.0 - threat_decay_rate):
                    continue
                displacements = [(1, 0), (-1, 0), (0, 1), (0, -1)]
                for displacement in displacements:
                    next_location = (curr_location[0] + displacement[0], curr_location[1] + displacement[1])
                    if next_location in walls:
                        continue
                    if next_location in visited:
                        continue
                    if rewards[next_location][1] <= rewards[curr_location][1]:
                        continue
                    rewards[next_location] = (self.__HOSTILE[0], rewards[curr_location][1] + threat_decay_rate)
                    queue.push(next_location)
                    visited.add(next_location)
        return (rewards, utilities)

    def __update_utilities(self, walls, neighbors, rewards, utilities, discount_factor=1.0, convergence_tolerance=0.0001, ignoring_walls=False, maximum_mode=True, noise=0):
        previous_utilities = copy.deepcopy(utilities)
        chance = random.random()
        fully_convergent = True
        total_entropy = 0.0
        for location in utilities.keys():
            if utilities[location][0] == self.__FREE[0]:
                fully_convergent = False
                east_utility = 0.0
                west_utility = 0.0
                north_utility = 0.0
                south_utility = 0.0
                if ignoring_walls:
                    east_utility += 0.8 * previous_utilities[neighbors[location][Directions.EAST]][1] + 0.1 * previous_utilities[neighbors[location][Directions.NORTH]][1] + 0.1 * previous_utilities[neighbors[location][Directions.SOUTH]][1]
                    west_utility += 0.8 * previous_utilities[neighbors[location][Directions.WEST]][1] + 0.1 * previous_utilities[neighbors[location][Directions.SOUTH]][1] + 0.1 * previous_utilities[neighbors[location][Directions.NORTH]][1]
                    north_utility += 0.8 * previous_utilities[neighbors[location][Directions.NORTH]][1] + 0.1 * previous_utilities[neighbors[location][Directions.WEST]][1] + 0.1 * previous_utilities[neighbors[location][Directions.EAST]][1]
                    south_utility += 0.8 * previous_utilities[neighbors[location][Directions.SOUTH]][1] + 0.1 * previous_utilities[neighbors[location][Directions.EAST]][1] + 0.1 * previous_utilities[neighbors[location][Directions.WEST]][1]
                else:
                    # east
                    if neighbors[location][Directions.EAST] not in walls:
                        east_utility += 0.8 * previous_utilities[neighbors[location][Directions.EAST]][1]
                    else:
                        east_utility += 0.8 * previous_utilities[location][1]
                    # left of east: north
                    if neighbors[location][Directions.NORTH] not in walls:
                        east_utility += 0.1 * previous_utilities[neighbors[location][Directions.NORTH]][1]
                    else:
                        east_utility += 0.1 * previous_utilities[location][1]
                    # right of east: south
                    if neighbors[location][Directions.SOUTH] not in walls:
                        east_utility += 0.1 * previous_utilities[neighbors[location][Directions.SOUTH]][1]
                    else:
                        east_utility += 0.1 * previous_utilities[location][1]
                    # west
                    if neighbors[location][Directions.WEST] not in walls:
                        west_utility += 0.8 * previous_utilities[neighbors[location][Directions.WEST]][1]
                    else:
                        west_utility += 0.8 * previous_utilities[location][1]
                    # left of west: south
                    if neighbors[location][Directions.SOUTH] not in walls:
                        west_utility += 0.1 * previous_utilities[neighbors[location][Directions.SOUTH]][1]
                    else:
                        west_utility += 0.1 * previous_utilities[location][1]
                    # right of west: north
                    if neighbors[location][Directions.NORTH] not in walls:
                        west_utility += 0.1 * previous_utilities[neighbors[location][Directions.NORTH]][1]
                    else:
                        west_utility += 0.1 * previous_utilities[location][1]
                    # north
                    if neighbors[location][Directions.NORTH] not in walls:
                        north_utility += 0.8 * previous_utilities[neighbors[location][Directions.NORTH]][1]
                    else:
                        north_utility += 0.8 * previous_utilities[location][1]
                    # left of north: west
                    if neighbors[location][Directions.WEST] not in walls:
                        north_utility += 0.1 * previous_utilities[neighbors[location][Directions.WEST]][1]
                    else:
                        north_utility += 0.1 * previous_utilities[location][1]
                    # right of north: east
                    if neighbors[location][Directions.EAST] not in walls:
                        north_utility += 0.1 * previous_utilities[neighbors[location][Directions.EAST]][1]
                    else:
                        north_utility += 0.1 * previous_utilities[location][1]
                    # south
                    if neighbors[location][Directions.SOUTH] not in walls:
                        south_utility += 0.8 * previous_utilities[neighbors[location][Directions.SOUTH]][1]
                    else:
                        south_utility += 0.8 * previous_utilities[location][1]
                    # left of south: east
                    if neighbors[location][Directions.EAST] not in walls:
                        south_utility += 0.1 * previous_utilities[neighbors[location][Directions.EAST]][1]
                    else:
                        south_utility += 0.1 * previous_utilities[location][1]
                    # right of south: west
                    if neighbors[location][Directions.WEST] not in walls:
                        south_utility += 0.1 * previous_utilities[neighbors[location][Directions.WEST]][1]

                if maximum_mode:
                    if chance < noise:
                        utilities[location] = (previous_utilities[location][0], rewards[location][1] +  discount_factor * random.choice([east_utility, west_utility, north_utility, south_utility]))
                    else:
                        utilities[location] = (previous_utilities[location][0], rewards[location][1] +  discount_factor * max([east_utility, west_utility, north_utility, south_utility]))
                    
                else:
                    utilities[location] = (previous_utilities[location][0], rewards[location][1] + discount_factor * min([east_utility, west_utility, north_utility, south_utility]))
                total_entropy += abs(utilities[location][1] - previous_utilities[location][1])
            if total_entropy < convergence_tolerance:
                fully_convergent = True
        return (utilities, fully_convergent, total_entropy)

    def __print_data_structure(self, walls, grid):
        x_coordinates = []
        y_coordinates = []
        for wall in walls:
            x_coordinates.append(wall[0])
            y_coordinates.append(wall[1])
        x_minimum, x_maximum = min(x_coordinates), max(x_coordinates)
        y_minimum, y_maximum = min(x_coordinates), max(y_coordinates)
        for y in range(y_maximum, y_minimum - 1, -1):
            line = "{:4s}:".format("y=" + str(y))
            for x in range(x_minimum, x_maximum + 1, 1):
                grid_value = grid[(x, y)]
                line += "({:2s},{:>13s})".format(grid_value[0], "{:+9.3f}".format(grid_value[1]))
            print(line)
        x_axis = "     "
        for x in range(x_minimum, x_maximum + 1, 1):
            x_axis += "{:18s}".format(" x=" + str(x))
        print(x_axis)

    def __value_iteration(self, state, ghostbuster_mode):
        ghosts_states = api.ghostStates(state)
        edible_ghosts = []
        hostile_ghosts = []
        early_stopping_point = self.__NORMAL_EARLY_STOPPING_POINT
        for ghost_state in ghosts_states:
            if ghost_state[1] == 1:
                edible_ghosts.append((int(ghost_state[0][0]), int(ghost_state[0][1])))
            else:
                hostile_ghosts.append(ghost_state[0])
        self.__rewards, self.__utilities = self.__initialize_data_structures(self.__foods, self.__FOOD, self.__walls, self.__floors, hostile_ghosts, safety_distance=self.__SAFETY_DISTANCE, threat_decay_rate=self.__THREAT_DECAY_RATE)
        early_stopping_point = self.__NORMAL_EARLY_STOPPING_POINT
        if len(self.__foods) < 10:
            early_stopping_point = self.__SPARSE_EARLY_STOPPING_POINT
        if ghostbuster_mode == self.__DEFENSIVE_GHOSTBUSTER_MODE:
            if len(edible_ghosts) > 0:
                self.__rewards, self.__utilities = self.__initialize_data_structures(edible_ghosts, self.__EDIBLE, self.__walls, self.__floors, hostile_ghosts, safety_distance=self.__SAFETY_DISTANCE, threat_decay_rate=self.__THREAT_DECAY_RATE)
                early_stopping_point = self.__SPARSE_EARLY_STOPPING_POINT
        if ghostbuster_mode == self.__OFFENSIVE_GHOSTBUSTER_MODE:
            if len(edible_ghosts) > 0:
                self.__rewards, self.__utilities = self.__initialize_data_structures(edible_ghosts, self.__EDIBLE, self.__walls, self.__floors, hostile_ghosts, safety_distance=self.__SAFETY_DISTANCE, threat_decay_rate=self.__THREAT_DECAY_RATE)
                early_stopping_point = self.__SPARSE_EARLY_STOPPING_POINT
            elif len(self.__capsules) > 0:
                self.__rewards, self.__utilities = self.__initialize_data_structures(self.__capsules, self.__CAPSULE, self.__walls, self.__floors, hostile_ghosts, safety_distance=self.__SAFETY_DISTANCE, threat_decay_rate=self.__THREAT_DECAY_RATE)
                early_stopping_point = self.__SPARSE_EARLY_STOPPING_POINT
        stopping_point = None
        for i in range(early_stopping_point):
            stopping_point = i + 1
            self.__utilities, fully_convergent, total_entropy = self.__update_utilities(self.__walls, self.__neighbors, self.__rewards, self.__utilities, discount_factor=self.__DISCOUNT_FACTOR, convergence_tolerance=self.__CONVERGENCE_TOLERANCE, ignoring_walls=False, maximum_mode=True)
            if fully_convergent:
                break
   
    def __maximum_expected_utility(self, state):
        # the location of agent
        agent_location = api.whereAmI(state)
        # discover the legal actions
        legal = api.legalActions(state)
        # remove STOP to increase mobility
        legal.remove(Directions.STOP)
        # decide next move based on maximum expected utility
        action, maximum_expected_utility = None, None
        for direction in legal:
            utility = self.__utilities[self.__neighbors[agent_location][direction]][1]
            if action == None or maximum_expected_utility == None:
                action = direction
                maximum_expected_utility = utility
            expected_utility = utility
            if expected_utility > maximum_expected_utility:
                action = direction
                maximum_expected_utility = expected_utility
        return action

    def getAction(self, state):
        self.__mapping_operation(state)
        self.__value_iteration(state, ghostbuster_mode=self.__GHOSTBUSTER_MODE)
        meu_action = self.__maximum_expected_utility(state)
        return api.makeMove(meu_action, api.legalActions(state))
