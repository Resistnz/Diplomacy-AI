import random
import networkx as nx
import timeout_decorator
from agent_baselines import Agent
import pprint
from game import copy_game
from itertools import product
from tqdm import tqdm

class StudentAgent(Agent):
    '''
    Implement your agent here. 

    Please read the abstract Agent class from baseline_agents.py first.
    
    You can add/override attributes and methods as needed.
    '''

    @timeout_decorator.timeout(1)
    def __init__(self, agent_name='Give a nickname'):
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

    @timeout_decorator.timeout(1)
    def get_actions(self):

        '''Implement your agent here.'''

        '''
        Return a list of orders. Each order is a string, with specific format. For the format, read the game rule and game engine documentation.
        
        Expected format:
        A LON H                  # Army at LON holds
        F IRI - MAO              # Fleet at IRI moves to MAO (and attack)
        A WAL S F LON            # Army at WAL supports Fleet at LON (and hold)
        F NTH S A EDI - YOR      # Fleet at NTH supports Army at EDI to move to YOR
        F NWG C A NWY - EDI      # Fleet at NWG convoys Army at NWY to EDI
        A NWY - EDI VIA          # Army at NWY moves to EDI via convoy
        A WAL R LON              # Army at WAL retreats to LON
        A LON D                  # Disband Army at LON
        A LON B                  # Build Army at LON
        F EDI B                  # Build Fleet at EDI

        Note: If an invalid order is sent to the engine, it will be accepted but with a result of 'void' (no effect).
        Note: For a 'support' action, two orders are needed, one for the supporter and one for the supportee. (Same for 'convoy')
        Note: For each unit, if no order is given, it will 'hold' by default.

        Useful Functions:
        
        # This is a dict of all the possible orders for each unit at each location (for all powers).
        possible_orders = self.game.get_all_possible_orders()

        # This is a list of all orderable locations for the power you control.
        orderable_locations = self.game.get_orderable_locations(self.power_name)
    
        # Combining these two, you can have the full action space for the power you control.

        # You can re-use the build_map_graphs function in the GreedyAgent to build the connection graph of the map if needed.
        
        '''

        # Whats the game state
        phase = self.game.get_current_phase()
        year = phase[1:5]
        season = phase[0]

        # Get me my possible orders
        all_possible_orders = self.game.get_all_possible_orders()
        troop_locations = self.game.get_orderable_locations(self.power_name)

        valid_orders = [all_possible_orders[location] for location in troop_locations]
        flat = [x for sublist in valid_orders for x in sublist] # Flatten the list
        #valid_orders = flat

        # Make a random order set for each other agent
        orders = {}
        for power in self.game.powers.keys():
            if power == self.power_name: continue

            orders[power] = StudentAgent.get_random_orders(self.game, power)

        # Find all sets of orders using cartesian product of the valid orders
        order_sets = [list(x) for x in product(*valid_orders)]

        # Search 1 move ahead and get the best eval
        best_eval = float('-inf')
        best_order_set = []

        with tqdm(total=len(order_sets)) as pbar:
            for order_set in order_sets:
                newGame = copy_game(self.game)

                orders[self.power_name] = order_set

                for p in self.game.powers.keys():
                    #print(orders[p])
                    newGame.set_orders(p, orders[p])

                evaluation = StudentAgent.eval(self.power_name, newGame)

                if evaluation > best_eval:
                    best_eval = evaluation
                    best_order_set = order_set

                pbar.update(1)

        print(f"Finished phase {phase}")

        return best_order_set