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

class StudentAgent(Agent):
    '''
    Implement your agent here. 

    Please read the abstract Agent class from baseline_agents.py first.
    
    You can add/override attributes and methods as needed.
    '''

    # class MTCSNode:
    #     def __init__(self, order_set):
    #         # Orders for all powers
    #         self.order_set = order_set

    #         self.visit_count = 0
    #         self.evaluation = 0

    #         self.children = []

    class OrderType(Enum):
        HOLD = 1
        MOVE = 2
        SUPPORT = 3
        CONVOY = 4
        BUILD = 5
        DISBAND = 6

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
        #return []
    
    # Generate order sets for all powers
    @timeout_decorator.timeout(1)
    def generate_joint_order_sets(self, season):
        # Get some info about the game state
        all_possible_orders = self.game.get_all_possible_orders()
        troop_locations = self.game.get_orderable_locations(self.power_name)

        valid_orders = [all_possible_orders[location] for location in troop_locations]

        #print(valid_orders)

        num_troops = len(valid_orders)
        #valid_orders = [order for location_orders in valid_orders for order in location_orders]

        #print(valid_orders)

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

        #pprint.pprint(occupied_centres)
        #pprint.pprint(unoccupied_centres)

        # Get where different units are
        unit_locations = {}
        for power, units in self.game.get_units().items():
            for unit in units:
                unit_locations[unit[2:5]] = (power, unit)

        

        # How many opps at each location
        opp_counts = defaultdict(int)
        support_counts = defaultdict(int)

        for loc in troop_locations:
            opp_count = 0
            support_count = 0

            for neighbour in neighbours[loc]:
                if neighbour not in unit_locations: continue

                troop_owner = unit_locations[neighbour][0]
                if troop_owner != self.power_name: opp_count += 1
                else: support_count += 1

                opp_counts[loc] = opp_count
                support_counts[loc] = support_count

        # Evaluate how good each individual order might be 
        order_qualities = [list() for _ in range(num_troops)]
        for i in range(num_troops):
            # Get shortest parth to each centre
            paths = nx.shortest_path(self.map_graph_army, source=loc)
            closest_center = None
            closest_distance = 1000
            for center in enemy_centres:
                if center not in paths.keys():
                    continue
                dis = len(paths[center])
                if dis < closest_distance:
                    closest_distance = dis
                    closest_center = center

            for order in valid_orders[i]:
                evaluation = 0

                # Evaluate it individually
                unit_type = order[0]
                unit_location = str(order[2:5])

                #print(unit_location)
            # print(order)

                order_type = StudentAgent.OrderType.HOLD
                target_location = None
                supportee = None

                if len(order) == 15: continue # We ain't convoying yet

                if len(order) > 6:
                    if order[6] == "S":
                        order_type = StudentAgent.OrderType.SUPPORT
                        supportee = str(order[10:13])

                        if len(order) > 16:
                            target_location = str(order[16:19])

                    elif order[6] == "-":
                        order_type = StudentAgent.OrderType.MOVE
                        target_location = str(order[8:11])
                    elif order[6] == "C":
                        order_type = StudentAgent.OrderType.CONVOY
                        continue
                    elif order[6] == "D":
                        order_type = StudentAgent.OrderType.DISBAND
                        #evaluation = -100
                        #continue
                    elif order[6] == "B":
                        order_type = StudentAgent.OrderType.BUILD
                        #evaluation = 100
                        #continue

                # If we got support we are happy
                if support_counts[unit_location] > 0:
                    evaluation += 5 * support_counts[unit_location]

                # If we defending a centre
                if unit_location in our_centres and order_type == StudentAgent.OrderType.HOLD:
                    # Check if we got opps
                    if opp_counts[unit_location] == 0:
                        evaluation = -10
                    else:
                        evaluation = 40

                if order_type == StudentAgent.OrderType.MOVE:
                    # If we can grab a free centre
                    if target_location in unoccupied_centres:
                        evaluation += 10

                        # No opps is good
                        if opp_counts[target_location] == 0:
                            evaluation += 50
                        # Probably not good if theres hella opps
                        if opp_counts[target_location] >= 2:
                            evaluation -= 5

                    # Might be good to move to the close guy
                    if closest_center is not None:
                        if len(paths[closest_center]) < 2: continue
                        target = paths[closest_center][1]

                        if target_location == target: 
                            evaluation += 20

                # Supports are alright
                if order_type == StudentAgent.OrderType.SUPPORT:
                    # Supporting move into empty spot
                    if target_location not in unit_locations:
                        if opp_counts[target_location] == 0: evaluation = -10
                        elif opp_counts[target_location] == 1: evaluation = min(30, evaluation * 2)
                        else: evaluation += 10
                    else:
                        if opp_counts[target_location] == 0: evaluation = min(30, evaluation * 2)
                        elif opp_counts[target_location] == 1: 
                            if support_counts[target_location] < 2: evaluation = -10
                            else: evaluation *= min(20, evaluation * 1.5)
                        else: evaluation -= 10

                heapq.heappush(order_qualities[i], (-evaluation, order))

        print(order_qualities)
        
        # Get only the top 3 orders per unit
        CANDIDATE_ORDER_COUNT = 3
        candidate_orders = [list() for _ in range(num_troops)]
        for j in range(num_troops):
            for i in range(CANDIDATE_ORDER_COUNT):
                if len(order_qualities) <= j: break
                if len(order_qualities[j]) == 0: continue
                evaluation, order = heapq.heappop(order_qualities[j])

                print(-evaluation, order)
                
                candidate_orders[j].append(order)

        print(candidate_orders)
        

        # Make 3 random order sets for each enemy agent
        enemy_order_sets = {}
        ENEMY_ORDER_SET_COUNT = 1

        for power in self.game.powers.keys():
            if power == self.power_name: continue

            enemy_order_sets[power] = []

            for i in range(ENEMY_ORDER_SET_COUNT):
                enemy_order_sets[power].append(StudentAgent.get_random_orders(self.game, power))

        # Find all sets of our orders using cartesian product
        prospective_order_sets = [list(x) for x in product(*candidate_orders)]
        our_order_sets = []

        #print("Prospective order sets:")
        #pprint.pprint(prospective_order_sets)
        #print()
        #quit()

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
                            print(f"Pruned order set {order_set}")
                            break

            if is_good:
                our_order_sets.append(order_set)

        print(f"Pruned {len(prospective_order_sets) - len(our_order_sets)} orders.")

        all_power_order_sets = enemy_order_sets
        all_power_order_sets[self.power_name] = our_order_sets

        keys = all_power_order_sets.keys()
        value_lists = all_power_order_sets.values()
        
        # Generate cartesian product of all orders
        combinations = list(product(*value_lists))
        
        # Create all combinations of orders using cartesian product again
        joint_order_sets : list[dict] = [dict(zip(keys, combo)) for combo in combinations]

        print(f"Generated {len(joint_order_sets)} orders.\n")

        print(f"We have {len(troop_locations)} units.")
        #quit()

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
        order_sets = self.generate_joint_order_sets(season)
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