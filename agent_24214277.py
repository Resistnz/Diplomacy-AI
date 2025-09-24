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
from collections import defaultdict, deque
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
    WAIVE = 8

@dataclass
class Unit():
    unit_location: str 
    unit_type: str
    power_name: str = None

# Used to generate some candidate actions to search through
@dataclass
class CandidateAction():
    unit: Unit

    unit_location: str 
    unit_type: str
    order_string: str
    power_name: str = None

    action_type: ActionType = ActionType.HOLD
    target: Optional[str] = None # HOLD will have no target

    supported_unit: Optional[str] = None 
    support_target: Optional[str] = None # The location the supported unit is going to

    via_convoy: bool = False # Movin via convoy
    build_type: str = None

    score: float = 0 

class Game:
    def __init__(self, adjacency, supply_centres_by_power, units, phase):
        self.adjacency = adjacency

        # Clean up supply centres input
        self.supply_centres_by_power = {
            power: [c.upper() for c in centres]
            for power, centres in supply_centres_by_power.items()
        }

        # Who owns each supply centre 
        self.who_owns_supply_centres = {}
        all_scs = set()
        for centres in self.supply_centres_by_power.values():
            all_scs.update(centres)
        for sc in all_scs:
            self.who_owns_supply_centres[sc] = None
        for power, centres in self.supply_centres_by_power.items():
            for c in centres:
                self.who_owns_supply_centres[c] = power

        # Units by power (list of "A PAR" etc)
        self.units_by_power = {
            power: [ustr.strip().upper() for ustr in ulist]
            for power, ulist in units.items()
        }

        # Units by location {location: (power, unit_type)}
        self.units_by_location = {}
        for power, ulist in self.units_by_power.items():
            for s in ulist:
                unit_type = s[0]  # A or F
                unit_location = s[2:5] 
                self.units_by_location[unit_location] = (power, unit_type)

        self.phase = phase

    def get_centers(self):
        centres_by_power = defaultdict(list)
        for location, owner in self.who_owns_supply_centres.items():
            if owner:
                centres_by_power[owner].append(location)

        # Ensure every power is there even if it owns no centres
        for p in self.units_by_power.keys():
            centres_by_power.setdefault(p, [])
        return dict(centres_by_power)

    def get_units(self):
        out = defaultdict(list)
        for location, (power, unit_type) in self.units_by_location.items():
            out[power].append(f"{unit_type} {location}")
        for p in self.units_by_power.keys():
            out.setdefault(p, [])
        return dict(out)

    def resolve_orders(self, orders_by_unit):
        orders_by_location = {}
        for power, order_list in orders_by_unit.items():
            for order in order_list:
                if order.action_type == ActionType.BUILD:
                    self.units_by_location[order.unit_location] = (power, order.build_type)
                    self.units_by_power[power].append(f"{order.build_type} {order.unit_location}")

                    continue
                orders_by_location[order.unit_location] = order

        moves = {}   
        holds = set()
        supports = defaultdict(list)
        convoys = defaultdict(set)

        for unit_location, order in orders_by_location.items():
            order_type = order.action_type

            if order_type == ActionType.MOVE:
                moves[unit_location] = order.target
            elif order_type == ActionType.HOLD:
                holds.add(unit_location)
            elif order_type == ActionType.SUPPORT:
                supports[(order.supported_unit, order.support_target)].append(unit_location)
            elif order_type == ActionType.CONVOY:
                convoys[(order.supported_unit, order.support_target)].add(unit_location)
            else:
                holds.add(unit_location)

        # Fun nested function
        def support_count_for(attacker_loc: str, target_loc: str) -> int:
            key = (attacker_loc, target_loc)
            supporters_list = supports.get(key, [])
            if attacker_loc in moves:
                if moves[attacker_loc] == target_loc:
                    return len(supporters_list)
                else:
                    return 0
            else:
                if attacker_loc == target_loc:
                    return len(supporters_list)
                return 0

        contests = defaultdict(list)

        # Add attackers
        for mover_loc, dest in moves.items():
            if mover_loc not in self.units_by_location:
                continue
            mover_power, mover_unit_type = self.units_by_location[mover_loc]
            sup_count = support_count_for(mover_loc, dest)
            contests[dest].append((mover_loc, mover_power, mover_unit_type, 1 + sup_count))

        # Add defenders
        for dest in list(contests.keys()) + list(self.units_by_location.keys()):
            if dest in self.units_by_location:
                occupant_power, occupant_unit_type = self.units_by_location[dest]
                if dest in moves:
                    continue
                sup_count = support_count_for(dest, dest)
                contests[dest].append((dest, occupant_power, occupant_unit_type, 1 + sup_count))

        new_positions = {}
        removed_origins = set()

        for dest, candidates in contests.items():
            if not candidates:
                continue
            strengths = [c[3] for c in candidates]
            max_strength = max(strengths)
            winners = [c for c in candidates if c[3] == max_strength]
            if len(winners) > 1:
                continue
            winner_loc, winner_power, winner_unit_type, winner_strength = winners[0]
            if winner_loc == dest:
                new_positions[dest] = (winner_power, winner_unit_type)
            else:
                new_positions[dest] = (winner_power, winner_unit_type)
                removed_origins.add(winner_loc)
                if dest in self.units_by_location:
                    removed_origins.add(dest)

        for orig_loc, (power, unit_type) in self.units_by_location.items():
            if orig_loc in removed_origins:
                continue
            if orig_loc not in new_positions:
                new_positions[orig_loc] = (power, unit_type)

        # Update units
        new_units_by_power = defaultdict(list)
        new_units_by_location = {}
        for location, (power, unit_type) in new_positions.items():
            new_units_by_location[location] = (power, unit_type)
            new_units_by_power[power].append(f"{unit_type} {location}")

        self.units_by_location = new_units_by_location
        self.units_by_power = {p: lst[:] for p, lst in new_units_by_power.items()}

        # Update supply centre ownership in Fall
        if self.phase[0] == "F":
            for location, (power, _) in new_positions.items():
                if location in self.who_owns_supply_centres:
                    self.who_owns_supply_centres[location] = power
            new_scs_by_power = defaultdict(list)
            for sc, owner in self.who_owns_supply_centres.items():
                if owner:
                    new_scs_by_power[owner].append(sc)
            self.supply_centres_by_power = dict(new_scs_by_power)

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

        locations = list(self.game.map.loc_type.keys()) # locations with '/' are not real places

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

    def is_valid_order_set(self, order_set):
        actions_by_unit = {a.unit_location: a for a in order_set} # Dictionary comprehension go crazy
        # For example just gives {"PAR": "A PAR H"}

        if len(order_set) == 0: return False

        build_count = 0

        move_targets = set()

        # Make sure actions make sense
        for action in order_set:
            if action.action_type == ActionType.MOVE:
                #print(move_targets, action.target)
                if action.target in move_targets: 
                    return False

                move_targets.add(action.target)
            elif action.action_type == ActionType.SUPPORT:
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
            
            elif action.action_type == ActionType.BUILD:
                build_count += 1

        # Make sure we build the right amount
        if build_count != self.amount_to_build: return False

        return True

    # Turn the order into an CandidateAction
    def parse_order(order):
        s = order.upper() # Hate that I have to do this but some come in as lowercase

        unit_type = s[0]
        unit_location = s[2:5]
        order_string = s

        candidate = CandidateAction(
            unit = Unit(unit_location, unit_type),

            unit_location = unit_location,
            unit_type = unit_type,
            order_string = order_string
        )

        # F NTH S A EDI - YOR
        # F IRI - MAO VIA
        # 0123456789012345678

        if s == "WAIVE":
            candidate.action_type = ActionType.WAIVE
            return candidate

        if len(s) > 6:
            if s[5] == "/": # Fix coasts
                s = s[:5] + s[8:]

            if s[6] == "H":
                candidate.action_type = ActionType.HOLD
            elif s[6] == "S":
                candidate.action_type = ActionType.SUPPORT
                candidate.supported_unit = str(s[10:13])

                if len(s) > 16:
                    candidate.support_target = str(s[16:19])

            elif s[6] == "-":
                candidate.action_type = ActionType.MOVE
                candidate.target = str(s[8:11])

                if len(s) == 15:
                    candidate.via_convoy = True
            elif s[6] == "C":
                candidate.action_type = ActionType.CONVOY
                candidate.supported_unit = str(s[10:13])
                candidate.support_target = str(s[16:19])

            elif s[6] == "D":
                candidate.action_type = ActionType.DISBAND
            elif s[6] == "B":
                candidate.action_type = ActionType.BUILD
                candidate.build_type = s[0]
            elif s[6] == "R":
                candidate.action_type = ActionType.RETREAT
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

    def calculate_map_properties(self):
        # Centres
        self.occupied_centres = {centre for centres in self.game.get_centers().values() for centre in centres}
        self.unoccupied_centres = set(self.game.map.scs) - self.occupied_centres

        self.all_centres = set(self.game.map.scs)
        self.our_centres = set(self.game.get_centers(self.power_name))

        #print(f"Our centres are: {self.our_centres}")

        self.enemy_centres = self.occupied_centres - self.our_centres
        self.unoccupied_or_enemy_centres = self.unoccupied_centres.union(self.enemy_centres)

        #print(self.our_centres)
        #print(self.enemy_centres)
        #print(self.unoccupied_or_enemy_centres)

        self.all_unit_locations = {}
        for power, units in self.game.get_units().items():
            for unit in units:
                self.all_unit_locations[unit[2:5]] = (power, unit)

        # Neighbours
        self.neighbours = self.game.map.loc_abut

        # Make them uppercase cause this is stupid
        for location in list(self.neighbours.keys()):
            self.neighbours[location] = [x.upper() for x in self.neighbours[location]]
            self.neighbours[location.upper()] = self.neighbours[location]

        # Units
        self.our_units = self.game.get_units(self.power_name)
        self.our_units = [x[2:5] for x in self.our_units] # Strip the "A " or "F " at the start

        self.all_units = self.game.get_units()
        self.enemy_units = []
        for power, units in self.all_units.items():
            if power == self.power_name: continue
            self.enemy_units.extend(units)

        #pprint.pprint(F"Our units: {self.our_units}")

        # How many opps at each location
        self.opp_counts = defaultdict(int)
        self.support_counts = defaultdict(int)

        for location in self.game.map.locs:
            opp_count = 0
            support_count = 0

            for neighbour in self.neighbours[location]:
                if neighbour not in self.all_unit_locations: continue

                troop_owner = self.all_unit_locations[neighbour][0]
                if troop_owner != self.power_name: opp_count += 1
                else: support_count += 1

            self.opp_counts[location] = opp_count
            self.support_counts[location] = support_count

        self.season = self.game.phase[0]
        self.amount_to_build = 0

        self.expanding = True

        if self.season == "W":
            self.amount_to_build = len(self.our_centres) - len(self.our_units)

        # Distance from any node to any other
        locations = list(self.game.map.loc_abut.keys())
        self.distance_cache = {l: {} for l in locations}

        for start in locations:
            queue = deque([(start, 0)])
            seen = {start}
            while queue:
                current, d = queue.popleft()
                self.distance_cache[start][current] = d
                for neigh in self.game.map.loc_abut[current]:
                    if neigh not in seen:
                        seen.add(neigh)
                        queue.append((neigh, d + 1))

        self.locations_to_build_fleets = {"POR", "SPA", "NAP", "TUN", "BRE", "LON", "EDI", "LVR", "SMY", "GRE", "ANK"}

    # Super quick and basic eval of a candidate action
    def evaluate_candidate_action(self, action : CandidateAction):
        def season():
            if self.season == "S": return 0.8
            if self.season == "F": return 1.5

            return 1

        score = 0

        target = action.target
        unit_location = action.unit_location
        action_type = action.action_type
        support_target = action.support_target
        supported_unit = action.supported_unit

        if action_type == ActionType.BUILD:
            if unit_location in self.locations_to_build_fleets and action.build_type == "F":
                return 200
            return 100
        
        elif action_type == ActionType.CONVOY:
            return -100
        
        elif action_type == ActionType.WAIVE:
            return -100
        
        elif action_type == ActionType.RETREAT:            
            return 0

        elif action_type == ActionType.MOVE:
            score += 2 # Encourage moving a lil bit

            if target in self.unoccupied_centres: score += 20 * season() # New centre
            elif target in self.enemy_centres: score += 15 * season() # Enemy centre
            elif target in self.our_centres: score += 5 # Defending our centre

            if unit_location in self.our_centres: # Leaving it empty
                if self.opp_counts[unit_location] > 0: score -= 20 # Opps around

            #if target == "VEN":
            #    print("Attack!")

            # Attacking an enemy
            #print(action.order_string, self.all_unit_locations.get(target, (self.power_name, False)))
            if self.all_unit_locations.get(target, (self.power_name, False))[0] != self.power_name:
                if self.support_counts[target] > 1: 
                    #print(f"We gonna attack {target} with support")
                    #quit()
                    score += 20 # Be more aggressive
                else:
                    score -= 40

            # Check if we can win the fight
            #score += (self.support_counts[target] - self.opp_counts[target] - 1) * 6

            # Expand
            

            # Check the proximity to other supply centres
            proximity = 0
            for sc in self.unoccupied_or_enemy_centres:
                d = self.distance_cache[target].get(sc, 99)

                if d < 99:
                    proximity += max(0, 5 - d)
            score += (proximity//2)
            
            # Move away from self
            # if self.expanding:
            #     proximity = 0
            #     for sc in self.our_centres:
            #         d = self.distance_cache[target].get(sc, 99)

            #         if d < 99:
            #             proximity += max(0, 2 - d)

            #     score -= proximity//5

        elif action_type == ActionType.HOLD:
            if unit_location in self.our_centres:  # Defending our centre
                if self.opp_counts[unit_location] == 0: score = -10 # No opps around, just move
                else: score = 30

                score *= season()
            else: # Grabbing a new centre is always good
                if unit_location in self.unoccupied_or_enemy_centres:
                    score += 100
                else:
                    score += self.opp_counts[unit_location]*10

        elif action_type == ActionType.SUPPORT:
            if supported_unit and self.all_unit_locations.get(supported_unit, (False, False))[0] == self.power_name: # Supporting our own guy
                #print(f"Support! {action.order_string}")
                #print(self.all_unit_locations)
                #print(self.all_unit_locations.get(support_target, (self.power_name, False)))
                #quit()
                # Supporting a direct attack
                if self.all_unit_locations.get(support_target, (self.power_name, False))[0] != self.power_name: 
                    score += 20
                    #print("Supporting direct attack! {action.order_string}")
                    #quit()
                
                if support_target and support_target in self.unoccupied_or_enemy_centres: # Supporting a move into new centre 
                    score += 20
                    #print(f"Move into centre {support_target}")
                else:
                    score += 5

        #print(action.order_string, score)
        
        return score
    
    def generate_candidates(self, amount, unit_location):
        all_orders = self.game.get_all_possible_orders()
        unit_orders = all_orders.get(unit_location, [])

        order_status = self.game.get_order_status()

        #print(order_status)

        # Evaluate all the orders individually
        candidates = []
        for order in unit_orders:
            action = StudentAgent.parse_order(order)

            # Make sure it actually is us and not the enemy
            if action.action_type == ActionType.RETREAT or action.action_type == ActionType.DISBAND:
                # print(self.our_units)
                # pprint.pprint(order_status)
                # pprint.pprint(action)
                # print(order_status[self.power_name].get(f"{action.unit_type} {action.unit_location}", [False]))
                
                if self.power_name in order_status and f"{action.unit_type} {action.unit_location}" not in order_status[self.power_name]: 
                    #print(f"Not us! {order}")
                    continue
                #quit()

            action.score = self.evaluate_candidate_action(action)
            candidates.append(action)
        
        # Return the best ones
        candidates.sort(key=lambda x: x.score, reverse=True)
        #pprint.pprint(candidates[:amount])
        return candidates[:amount]

    # Return a quick estimate of the board state in the favour of the agent
    @timeout_decorator.timeout(1)
    def eval(self, game: Game):
        # Using dictionary to weight different properties
        propertyWeightings = {
            "our_centres": 10,
            "enemy_centres_lead": 5,
            "next_turn_centres": 20,
            "threat_to_our_centres": 5,
            "support_for_our_units": 10
        }

        # Initialise properties
        stateProperties = {}
        for property in propertyWeightings:
            stateProperties[property] = 0

        # How many supply centres we own
        centres = game.get_centers()
        stateProperties["our_centres"] = len(centres[self.power_name])

        # We win yay
        if len(centres[self.power_name]) >= 18: return float('inf')

        # How many supply centres we are about to get
        occupied_centres = {centre for centres_list in centres.values() for centre in centres_list}
        unoccupied_centres = set(self.game.map.scs) - occupied_centres

        for sc in unoccupied_centres:
            if game.units_by_location.get(sc, (False, False))[0] == self.power_name:
                stateProperties["next_turn_centres"] += 1
                #print(f"We getting {sc} next turn yay")

        # Check if anyone has won/is close based on supply centres
        for power, scs in centres.items():
            if power == self.power_name: continue

            # Terminal state
            if len(scs) >= 18:
                return float('-inf')

            # If someone is ahead of us that is not good
            scsLead = len(scs) - len(centres[self.power_name])
            if scsLead > 3:
                stateProperties["enemy_centres_lead"] += scsLead 

        # Evaluate the threat level to our supply centres
        stateProperties["threat_to_our_centres"] = 0
        for centre in self.our_centres:
            stateProperties["threat_to_our_centres"] += self.opp_counts[centre]

        # Evaluate the support level around our units
        stateProperties["support_for_our_units"] = 0
        for unit in self.our_units:
            stateProperties["support_for_our_units"] += self.support_counts[unit]

        # Calculate the weighted evaluation
        evaluation = 0

        for property, value in stateProperties.items():
            evaluation += value * propertyWeightings[property]

        return evaluation
    
    def get_product(self, candidate_actions, unit_locations):
        # TODO: Beam search for when many units involved

        # valid_orders = [candidate_actions[x] for x in unit_locations]

        # print("\nValid orders:")
        # print(len(list(product(*valid_orders))))

        # ORDER_COUNT = 100

        # order_sets = []
        # for order_set in product(*valid_orders):
        #     o = list(order_set)

        #     if not self.is_valid_order_set(o): 
        #         #print(f"{[x.order_string for x in order_set]} is bad, continuing")
        #         continue

        #     # Rank them
        #     score = sum(action.score for action in o)

        #     order_sets.append((score, o))

        # order_sets.sort(key=lambda x: x[0], reverse=True)

        # #print(len(order_sets))

        # #pprint.pprint(order_sets)

        # return [x for _, x in order_sets[:ORDER_COUNT]]

        beams = [ (0, []) ]  # (score, [actions])

        for unit in unit_locations:
            new_beams = []
            for score, action_list in beams:
                for action in candidate_actions[unit]:
                    new_score = score + action.score
                    new_action_list = action_list + [action]
                    if self.is_valid_order_set(new_action_list):  # optional pruning
                        new_beams.append((new_score, new_action_list))

            # keep top beam_width only
            new_beams.sort(key=lambda x: x[0], reverse=True)
            beams = new_beams[:100]

        return [actions for _, actions in beams]
    
    @timeout_decorator.timeout(1)
    def get_random_orders(game, power_name):
        possible_orders = game.get_all_possible_orders()
        
        orderable_locations = game.get_orderable_locations(power_name)
        
        #power_orders = [random.choice(possible_orders[location]) for location in orderable_locations if possible_orders[location]]
        # Always holdin
        power_orders = [f"{possible_orders[location][0][0:5]} H" for location in orderable_locations if possible_orders[location]]

        return power_orders 
    
    def generate_our_orders(self):
        start = time.time_ns()
        # Generate our candidate actions
        CANDIDATE_ORDER_COUNT = 3
        candidate_actions = {}

        for unit in self.our_units:
            #print(unit)
            candidate_actions[unit] = self.generate_candidates(CANDIDATE_ORDER_COUNT, unit)

        #print(f"Our units: {self.our_units}")

        # If we in a build phase
        if self.season == "W":
            for sc in self.our_centres:
                candidate_actions[sc] = self.generate_candidates(CANDIDATE_ORDER_COUNT, sc)

        # print(f"Candidate actions took {round((time.time_ns() - start)/1000000)}ms.")
        # start = time.time_ns()

        #print(f"Our candidate actions are: ")
        #pprint.pprint(candidate_actions)

        # Get the interaction graph, and get the connected components
        graph = StudentAgent.build_interaction_graph(self.our_units, candidate_actions, self.neighbours)

        if self.season == "W":
            graph = StudentAgent.build_interaction_graph(list(self.our_centres), candidate_actions, self.neighbours)

        components = []
        for comp in nx.connected_components(graph):
            components.append(comp)

        # print(f"Building interaction graphs took {round((time.time_ns() - start)/1000000)}ms.")
        # start = time.time_ns()

        # print("\nComponents of the interaction graph:")
        # pprint.pprint(components)

        # Generate order sets for each component separately
        component_order_sets = []
        #print(components)

        list_time = 0

        for units in components:
            
            #print(f"Checking component: {units}")
            s = time.time_ns()
            order_sets = self.get_product(candidate_actions, units)
            list_time += time.time_ns() - s
            #pprint.pprint(candidate_actions)

            #print(f"Resulting order sets: {order_sets}")
            
            order_lists = [[action.order_string for action in order_set] for order_set in order_sets]
            

            #if len(order_lists) == 0: 
            #    pprint.pprint(order_sets)
            #    pprint.pprint(components)

            if not order_lists:
                fallback = [[candidate_actions[u][0].order_string for u in units if candidate_actions[u]]]
                order_lists = fallback
                #pprint.pprint(f"Fallback is {fallback}")
                #quit()

            component_order_sets.append(order_lists)

        # print(f"Generating orders for each component took {round((time.time_ns() - start)/1000000)}ms")
        # print(f"get_product() took {round((list_time)/1000000)}ms")
        # start = time.time_ns()

        # print(f"Final order sets has: {len(component_order_sets)}\n\n")

        # Cartesian product over each component
        full_orders = []
        for prod_tuple in product(*component_order_sets):
            #print(prod_tuple)

            # flatten list of lists
            combined = []
            for chunk in prod_tuple:
                combined.extend(chunk)
            full_orders.append(combined)
        
        full_orders.sort(key=lambda o: len(o), reverse=True)

        #print(full_orders)

        #if self.season == "W": quit()

        #print(f"Our orders are: {full_orders}")
        #quit()

       # print(f"Final product took {round((time.time_ns() - start)/1000000)}ms. We have {len(full_orders)}")

        return full_orders
    
    # Generate order sets for all powers
    #@timeout_decorator.timeout(1)
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

        #print(f"Generated {len(joint_order_sets)} order sets.")
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

        self.calculate_map_properties()

        #print("-------------------------")

        # Get all possible orders
        #print(f"Getting actions for phase {phase}")
        now = time.time_ns()
        order_sets = self.generate_joint_order_sets()
        order_finish = time.time_ns()
        #print(f"Took {round((order_finish - now)/1000000)}ms to generate orders.")

        # Search 1 move ahead and get the best eval
        best_eval = float('-inf')
        best_order_set : dict = {}

        #print(f"Checking {len(order_sets)} states.")

        #if self.season == "W":
        #    pprint.pprint(order_sets)
        

        #if len(order_sets) == 1:
        #pprint.pprint(f"Our units: {self.game.get_units(self.power_name)} ({len(self.game.get_units(self.power_name))})")
        #pprint.pprint(f"Our centres: {self.game.get_centers(self.power_name)} ({len(self.game.get_centers(self.power_name))})")

            #quit()

        #with tqdm(total=len(order_sets)) as pbar:
        for order_set in order_sets:

            # newGameEngine = copy_game(self.game)

            # for p in newGameEngine.powers.keys():
            #     newGameEngine.set_orders(p, order_set[p])

            # newGameEngine.process()


            orders = {}


            for power, order in order_set.items():
                orders[power] = []
                for o in order:
                    o = StudentAgent.parse_order(o)
                    o.power_name = power
                    orders[power].append(o)

            adjacency = self.game.map.loc_abut
            supply_centres_by_power = self.game.get_centers() 
            units = self.game.get_units()                     
            phase = phase                 
            
            newGame = Game(adjacency, supply_centres_by_power, units, phase)
            newGame.resolve_orders(orders)

            # Assert that the units and centers are equivalent
            # assert all(
            #     sorted(newGameEngine.get_units()[power]) == sorted(newGame.get_units()[power])
            #     for power in newGameEngine.get_units()
            # ), f"Units mismatch between newGameEngine and newGame \n{newGameEngine.get_units()}\n{newGame.get_units()}"

            # assert all(
            #     sorted(newGameEngine.get_centers()[power]) == sorted(newGame.get_centers()[power])
            #     for power in newGameEngine.get_centers()
            # ), f"Centers mismatch between newGameEngine and newGame: \n{newGameEngine.get_centers()}\n{newGame.get_centers()}"

            #quit()

            evaluation = self.eval(newGame)

            #print(f"Orderset {order_set} had evaluation {evaluation}")

            if evaluation > best_eval:
                best_eval = evaluation
                best_order_set = order_set[self.power_name]

                #print("This is best!")

                #pbar.update(1)

        finish = time.time_ns()
        #print(f"Took {round((finish - now)/1000000)}ms.")
        #print(f"Finished phase {phase} with orders {best_order_set}\n")

        #if season == "W": 
            
        #    quit()
        #quit()

        self.game.render(output_path='img.svg')

        return best_order_set