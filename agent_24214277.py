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

class StudentAgent(Agent):
    '''
    Implement your agent here. 

    Please read the abstract Agent class from baseline_agents.py first.
    
    You can add/override attributes and methods as needed.
    '''

    class MTCSNode:
        def __init__(self, order_set):
            # Orders for all powers
            self.order_set = order_set

            self.visit_count = 0
            self.evaluation = 0

            self.children = []

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


    @timeout_decorator.timeout(1) # This is only for updating the game engine and other states if any. Do not implement heavy strategy here.
    def update_game(self, all_power_orders):
        # do not make changes to the following codes
        for power_name in all_power_orders.keys():
            self.game.set_orders(power_name, all_power_orders[power_name])
        self.game.process()

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
    
    @timeout_decorator.timeout(1)
    def get_random_orders(game, power_name):
        possible_orders = game.get_all_possible_orders()
        
        orderable_locations = game.get_orderable_locations(power_name)
        power_orders = [random.choice(possible_orders[loc]) for loc in orderable_locations if possible_orders[loc]]
        
        return power_orders 
    
    # Generate order sets for all powers
    @timeout_decorator.timeout(1)
    def generate_joint_order_sets(self):
        # Get some info about the game state
        all_possible_orders = self.game.get_all_possible_orders()
        troop_locations = self.game.get_orderable_locations(self.power_name)

        valid_orders = [all_possible_orders[location] for location in troop_locations]

        neighbours = self.game.map.loc_abut

        # Check which centres are unoccupied
        occupied_centres = self.game.get_centers()
        unoccupied_centres = set(self.game.map.scs)

        for power, centres in occupied_centres.items():
            for centre in centres:
                unoccupied_centres.remove(centre)

        # Check which locations are able to move into those centres
        locations_next_to_unoccupied_scs = set()

        for sc in unoccupied_centres:
            for neighbour in neighbours[sc]:
                locations_next_to_unoccupied_scs.add(neighbour)

        # Same for occupied ones
        locations_next_to_occupied_scs = set()

        for sc in occupied_centres:
            for neighbour in neighbours[sc]:
                locations_next_to_occupied_scs.add(neighbour)

        # Evaluate how good each order might be 
        order_qualities = []
        for order in valid_orders:
            evaluation = 0

            # Evaluate it individually
            unit_type = order[0]
            unit_location = order[2:5]

            if unit_location in locations_next_to_unoccupied_scs:
                evaluation = 10

            if unit_location in locations_next_to_occupied_scs and 

            heapq.heapppush(order_qualities, (evaluation, order))

        # Get only the top good ones
        CANDIDATE_ORDER_COUNT = 3
        candidate_orders = []
        for i in range(CANDIDATE_ORDER_COUNT):
            order, _ = heapq.heapppop(order_qualities)
            
            candidate_orders.append(order)

        # Make 3 random order sets for each enemy agent
        enemy_order_sets = {}

        for power in self.game.powers.keys():
            if power == self.power_name: continue

            enemy_order_sets[power] = []

            for i in range(3):
                enemy_order_sets[power].append(StudentAgent.get_random_orders(self.game, power))

        # Find all sets of our orders using cartesian product
        prospective_order_sets = [list(x) for x in product(*candidate_orders)]
        our_order_sets = []

        # Pruning
        for order_set in prospective_order_sets:
            is_good = True

            # Go through individual orders
            for order in order_set:
                if len(order) < 13: continue

                isSupport = order[6] == "S"

                # Make sure the support makes sense
                if isSupport:
                    if len(order) > 14: # Must be supporting a move
                        # Make sure that move exists in the other
                        if order[8:19] not in order_set: 
                            is_good = False
                            break

            if is_good:
                our_order_sets.append(order_set)

        print(f"Pruned {len(prospective_order_sets) - len(our_order_sets)} orders. Taking the top ")
        #pprint.pprint(our_order_sets)

        all_power_order_sets = enemy_order_sets
        all_power_order_sets[self.power_name] = our_order_sets

        keys = all_power_order_sets.keys()
        value_lists = all_power_order_sets.values()
        
        # Generate cartesian product of all orders
        combinations = product(*value_lists)
        
        # Create all combinations of orders using cartesian product again
        joint_order_sets : list[dict] = [dict(zip(keys, combo)) for combo in combinations]

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
        now = time.time_ns()
        order_sets = self.generate_joint_order_sets()
        finish = time.time_ns()
        print(f"Took {round((finish - now)/1000000)}ms.")

        # Search 1 move ahead and get the best eval
        best_eval = float('-inf')
        best_order_set : dict = {}

        print(f"Checking {len(order_sets)} states")

        for order_set in order_sets:
            newGame = copy_game(self.game)

            for p in self.game.powers.keys():
                newGame.set_orders(p, order_set[p])

            evaluation = StudentAgent.eval(self.power_name, newGame)

            if evaluation > best_eval:
                best_eval = evaluation
                best_order_set = order_set[self.power_name]

        print(f"Finished phase {phase}")

        return best_order_set