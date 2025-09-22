import random
import networkx as nx
import timeout_decorator
from agent_baselines import Agent
import pprint
from game import copy_game
from itertools import product
from tqdm import tqdm
import time
import heapq
from functools import cache
from enum import Enum
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

class ActionType(Enum):
    HOLD = 1
    MOVE = 2
    SUPPORT = 3
    CONVOY = 4
    BUILD = 5
    DISBAND = 6
    RETREAT = 7

# Used to generate some candidate actions to search through
@dataclass
class CandidateAction():
    unit_location: str 
    unit_type: str
    order_string: str

    action_type: ActionType = ActionType.HOLD
    target: Optional[str] = None # HOLD will have no target

    supported_unit: Optional[str] = None 
    support_target: Optional[str] = None # The location the supported unit is going to

    via_convoy: bool = False # Movin via convoy

    score: float = 0 

class StudentAgent(Agent):
    @timeout_decorator.timeout(1)
    def __init__(self, agent_name='wining bot'):
        super().__init__(agent_name)

        self.map_graph_army = None
        self.map_graph_navy = None

    @timeout_decorator.timeout(1)
    def new_game(self, game, power_name):
        self.game = game
        self.power_name = power_name

        self.build_map_graphs()

    @timeout_decorator.timeout(1)
    def build_map_graphs(self): # Grabbed from the greedy agent implementation
        if not self.game:
            raise Exception('Game Not Initialised. Cannot Build Map Graphs.')

        self.map_graph_army = nx.Graph()
        self.map_graph_navy = nx.Graph()

        locations = list(self.game.map.loc_type.keys()) # locations with '/' are not real provinces

        for i in locations:
            if self.game.map.loc_type[i] in ['LAND', 'COAST']:
                self.map_graph_army.add_node(i.upper())
            if self.game.map.loc_type[i] in ['WATER', 'COAST']:
                self.map_graph_navy.add_node(i.upper())

        locations = [i.upper() for i in locations]

        for i in locations:
            for j in locations:
                if self.game.map.abuts('A', i, '-', j):
                    self.map_graph_army.add_edge(i, j)
                if self.game.map.abuts('F', i, '-', j):
                    self.map_graph_navy.add_edge(i, j)

    def build_interaction_graph(our_unit_locations, candidates, neighbours):
        # Nodes are units and edges are if actions interact, like same target/supports etc.
        graph = nx.Graph()

        for unit in our_unit_locations:
            graph.add_node(unit)

        # Join units if nay pair of them interact
        # Lots of loops but not too many units so its ok
        for i, unit in enumerate(our_unit_locations):
            for other_unit in our_unit_locations[i+1:]:
                interacting = False

                # Get all of their candidate actions
                for action in candidates.get(unit, []):
                    for other_action in candidates.get(other_unit, []):
                        if action.target == other_action.target: # Gonna bounce
                            interacting = True; break # Python is so cool for this
                        
                        # Supporting eachother
                        if action.action_type == ActionType.SUPPORT and action.supported_unit == other_unit:
                            interacting = True; break
                        if other_action.action_type == ActionType.SUPPORT and other_action.supported_unit == unit:
                            interacting = True; break
                        
                        # Gonna start some beef (are next to eachother)
                        target1 = action.target or action.unit_location
                        target2 = other_action.target or other_action.unit_location

                        if target1 in neighbours.get(target2, []) or target2 in neighbours.get(target1, []):
                            interacting = True; break
                        
                if interacting:
                    graph.add_edge(unit, other_unit)

        # TODO: Enemies
        return graph

    def is_valid_order_set(order_set):
        actions_by_unit = {a.unit_location: a for a in order_set} # Dictionary comprehension go crazy
        # For example just gives {"PAR": "A PAR H"}

        # Make sure actions make sense
        for action in order_set:
            if action.action_type == ActionType.SUPPORT:
                # Supportting a hold
                if action.support_target is None:
                    if action.supported_unit not in actions_by_unit: return False # This guy dont exist
                # We got a move
                else:
                    supported_action = actions_by_unit.get(action.supported_unit) # Action of the supported guy

                    if supported_action is None: # Bro does not exist
                        return False
                    if supported_action.action_type != ActionType.MOVE: 
                        return False
                    if supported_action.target != action.support_target: # Where he goin
                        return False

        # TODO: Make sure we aren't self-bouncing

    # Turn the order into an CandidateAction
    def parse_order(order):
        s = order.upper() # Hate that I have to do this but some come in as lowercase

        unit_type = s[0]
        unit_location = s[2:5]
        order_string = s

        candidate = CandidateAction(
            unit_location = unit_location,
            unit_type = unit_type,
            order_string = order_string
        )

        # F NTH S A EDI - YOR
        # F IRI - MAO VIA
        # 0123456789012345678

        if len(s) > 6:
            if s[6] == "H":
                candidate.order_type = ActionType.HOLD
            elif s[6] == "S":
                candidate.order_type = ActionType.SUPPORT
                candidate.supported_unit = str(s[10:13])

                if len(s) > 16:
                    candidate.support_target = str(s[16:19])

            elif s[6] == "-":
                candidate.order_type = ActionType.MOVE
                candidate.target = str(s[8:11])

                if len(s) == 15:
                    candidate.via_convoy = True
            elif s[6] == "C":
                candidate.order_type = ActionType.CONVOY
                candidate.supported_unit = str(s[10:13])
                candidate.support_target = str(s[16:19])

            elif s[6] == "D":
                candidate.order_type = ActionType.DISBAND
            elif s[6] == "B":
                candidate.order_type = ActionType.BUILD
            elif s[6] == "R":
                candidate.order_type = ActionType.RETREAT
            else:
                raise ValueError(s)
        else:
            raise ValueError(s)

        return candidate

    @timeout_decorator.timeout(1)
    def update_game(self, all_power_orders):
        for power_name in all_power_orders.keys():
            self.game.set_orders(power_name, all_power_orders[power_name])
        self.game.process()

    # Super quick and basic eval of a candidate action
    def evaluate_candidate_action(action, game, our_centres, enemy_centres, unit_locations, neighbours):
        score = 0

        occupied_centres = {centre for centres in game.get_centers().values() for centre in centres}
        unoccupied_centres = set(game.map.scs) - occupied_centres

        if action.action_type == ActionType.MOVE:
            if action.target in unoccupied_centres: # Grabbing free centre
                score += 40 
            if action.target in enemy_centres: # Attacking enemy centre
                score += 20
            if action.target in neighbours.get(action.unit_location, []): # Not moving too far away
                score += 2
        elif action.action_type == ActionType.HOLD:
            if action.unit_location in our_centres: # Defending centre
                score += 10 
        elif action.action_type == ActionType.SUPPORT:
            if action.supported_unit and action.supported_unit in unit_locations:
                if action.support_target and action.support_target in enemy_centres | unoccupied_centres: # Supporting a move into enemy 
                    score += 15
                else:
                    score += 5
            else:
                score -= 5  # Supporting no one

        return score
    
    def generate_candidates(amount, game, unit_location, our_centres, enemy_centres, unit_locations, neighbours):
        all_orders = game.get_all_possible_orders()
        unit_orders = all_orders.get(unit_location, [])

        # Evaluate all the orders individually
        candidates = []
        for order in unit_orders:
            c = StudentAgent.parse_order(order)
            c.score = StudentAgent.evaluate_candidate_action(c, game, our_centres, enemy_centres, unit_locations, neighbours)
            candidates.append(c)
        
        # Return the best ones
        candidates.sort(key=lambda x: x.score, reverse=True)
        return candidates[:amount]

    # Return a quick estimate of the board state in the favour of the agent
    @timeout_decorator.timeout(1)
    def eval(power_name, game):
        # Using dictionary to weight different properties
        propertyWeightings = {
            "our_centres": 10,
            "enemy_centres_lead": 5
        }

        # Initialise properties
        stateProperties = {}
        for property in propertyWeightings:
            stateProperties[property] = 0

        # How many supply centres we own
        centres = game.get_centers()
        stateProperties["our_centres"] = len(centres[power_name])

        # We win yay
        if len(centres[power_name]) >= 18: return float('inf')

        # Check if anyone has won/is close based on supply centres
        for power, scs in centres.items():
            if power == power_name: continue

            # Terminal state
            if len(scs) >= 18:
                return float('-inf')

            # If someone is ahead of us that is not good
            scsLead = len(scs) - len(centres[power_name])
            if scsLead > 3:
                stateProperties["enemy_centres_lead"] += scsLead 

        # Calculate the weighted evaluation
        evaluation = 0

        for property, value in stateProperties.items():
            evaluation += value * propertyWeightings[property]

        return evaluation
    
    def get_product(candidate_actions, unit_locations):
        # TODO: Beam search for when many units involved

        valid_orders = [candidate_actions[x] for x in unit_locations]

        order_sets = []
        for order_set in product(*valid_orders):
            o = list(order_set)

            if not StudentAgent.is_valid_order_set(o): continue

            # Rank them
            score = sum(action.score for action in o)
            order_sets.append((score, o))

        order_sets.sort(key=lambda x: x[0], reverse=True)

        return [x for _, x in order_sets]
    
    @timeout_decorator.timeout(1)
    def get_random_orders(game, power_name):
        possible_orders = game.get_all_possible_orders()
        
        orderable_locations = game.get_orderable_locations(power_name)
        power_orders = [random.choice(possible_orders[loc]) for loc in orderable_locations if possible_orders[loc]]
        
        return power_orders 
    
    def generate_our_orders(self):
        our_units = self.game.get_units(self.power_name)
        our_units = [x[2:5] for x in our_units] # Strip the "A " or "F " at the start

        neighbours = self.game.map.loc_abut

        # Make them uppercase cause this is stupid
        for loc in list(neighbours.keys()):
            neighbours[loc] = [x.upper() for x in neighbours[loc]]
            neighbours[loc.upper()] = neighbours[loc]

        # Check which centres are unoccupied
        occupied_centres = {centre for centres in self.game.get_centers().values() for centre in centres}
        unoccupied_centres = set(self.game.map.scs)

        all_centres = set(unoccupied_centres)
        our_centres = set(self.game.get_centers(self.power_name))
        enemy_centres = all_centres.difference(our_centres)

        for centre in occupied_centres:
            unoccupied_centres.remove(centre)

        unit_locations = {}
        for power, units in self.game.get_units().items():
            for unit in units:
                unit_locations[unit[2:5]] = (power, unit)

        # Generate our candidate actions
        CANDIDATE_ORDER_COUNT = 3

        candidate_actions = {}
        for unit in our_units:
            candidate_actions[unit] = StudentAgent.generate_candidates(CANDIDATE_ORDER_COUNT, self.game, unit, our_centres, enemy_centres, unit_locations, neighbours)

        # Get the interaction graph, and get the connected components
        graph = StudentAgent.build_interaction_graph(our_units, candidate_actions, neighbours)

        components = []
        for comp in nx.connected_components(graph):
            components.append(comp)

        # Generate order sets for each component separately
        component_order_sets = []

        for units in components:
            order_sets = StudentAgent.get_product(candidate_actions, units)

            order_lists = [[action.order_string for action in order_set] for order_set in order_sets]

            # TODO: If we got nothin
            component_order_sets.append(order_lists)

        # Cartesian product over each component
        full_orders = []
        for prod_tuple in product(*component_order_sets):
            # flatten list of lists
            combined = []
            for chunk in prod_tuple:
                combined.extend(chunk)
            full_orders.append(combined)
        
        full_orders.sort(key=lambda o: len(o), reverse=True)

        print(f"Our orders are: {full_orders}")

        return full_orders
    
    # Generate order sets for all powers
    @timeout_decorator.timeout(1)
    def generate_joint_order_sets(self):
        our_order_sets = self.generate_our_orders()

        enemy_order_sets = {}
        ENEMY_ORDER_SET_COUNT = 1

        for power in self.game.powers.keys():
            if power == self.power_name: continue

            enemy_order_sets[power] = []

            for i in range(ENEMY_ORDER_SET_COUNT):
                enemy_order_sets[power].append(StudentAgent.get_random_orders(self.game, power))

        all_power_order_sets = enemy_order_sets
        all_power_order_sets[self.power_name] = our_order_sets

        keys = all_power_order_sets.keys()
        value_lists = all_power_order_sets.values()
        
        # Generate cartesian product of all orders
        combinations = list(product(*value_lists))
        
        # Create all combinations of orders using cartesian product again
        joint_order_sets : list[dict] = [dict(zip(keys, combo)) for combo in combinations]

        print(f"Generated {len(joint_order_sets)} orders.\n")
        quit()

        return joint_order_sets

    @timeout_decorator.timeout(1)
    def get_actions(self):
        '''
            Generate own orders
            Generate enemy orders
                - Sample based on heuristic (hold stable, advance to border, defend etc.)
                - Limit to some amount
            Prune orders
            Join to get entire order set
            Eval each one
        '''

        # Whats the game state
        phase = self.game.get_current_phase()
        year = phase[1:5]
        season = phase[0]

        # Get all possible orders
        print(f"Getting actions for phase {phase}")
        now = time.time_ns()
        order_sets = self.generate_joint_order_sets()
        finish = time.time_ns()
        print(f"Took {round((finish - now)/1000000)}ms.")

        # Search 1 move ahead and get the best eval
        best_eval = float('-inf')
        best_order_set : dict = {}

        print(f"Checking {len(order_sets)} states")

        with tqdm(total=len(order_sets)) as pbar:
            for order_set in order_sets:
                newGame = copy_game(self.game)

                for p in newGame.powers.keys():
                    newGame.set_orders(p, order_set[p])

                newGame.process()

                evaluation = StudentAgent.eval(self.power_name, newGame)

                if evaluation > best_eval:
                    best_eval = evaluation
                    best_order_set = order_set[self.power_name]

                pbar.update(1)

        print(f"Finished phase {phase}")

        return best_order_set