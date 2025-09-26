# ZERO generative AI was used in this project. All features have been hand coded, tested and refined over countless hours.

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
    #unit: Unit

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

    def is_valid_order_set(self, order_set):
        actions_by_unit = {a.unit_location: a for a in order_set} # Dictionary comprehension go crazy
        # For example just gives {"PAR": "A PAR H"}

        if len(order_set) == 0: return False

        build_count = 0

        for action in order_set:
            if action.action_type == ActionType.MOVE:
                if action.via_convoy: return False
            elif action.action_type == ActionType.BUILD:
                build_count += 1
            
        if build_count < min(len(order_set), self.amount_to_build): return False

        return True

        # move_targets = set()
        # support_targets = set()

        # #print(f"Checking orderset: {[x.order_string for x in order_set]}")

        # # Prepass to get all suports
        # for action in order_set:
        #     if action.action_type == ActionType.SUPPORT:
        #         #print("We have a support")
        #         if action.support_target is None:
        #             #print("we make it here first")
        #             if action.supported_unit not in actions_by_unit: 
        #                 #print("bye bye")
        #                 return False # This guy dont exist

        #             support_targets.add(action.supported_unit)
        #         else:
        #             supported_action = actions_by_unit.get(action.supported_unit) # Action of the supported guy
        #             #print("we make it here")

        #             if supported_action is None: # Bro does not exist
        #                 return False
        #             if supported_action.action_type != ActionType.MOVE: 
        #                 #print("Very bad!!2")
        #                 return False
        #             if supported_action.target != action.support_target: # Where he goin
        #                 #print("Very bad!!")
        #                 return False
                    
        #             support_targets.add(action.support_target)
        
        # # Make sure actions make sense
        # for action in order_set:
        #     # Gonna self bounce
        #     if action.action_type == ActionType.MOVE:
        #         if action.target in move_targets: 
        #             return False
                
        #         # Ignore convoys for now
        #         if action.via_convoy: return False

        #         move_targets.add(action.target)

        #         # If this is an attack, make sure we have support
        #         #if action.target not in self.all_unit_locations: continue
        #         #if self.all_unit_locations[action.target][0] != self.power_name: # Gonna bump into an opp
        #             #if action.target not in support_targets: 
        #                 #print(f"No support! {action.order_string} {support_targets}")
        #                 #return False # Got no support
        #             #else:
        #                 #print("Gonna get through!")
        #                 #print([x.order_string for x in order_set])

        #     elif action.action_type == ActionType.SUPPORT: continue
            
        #     elif action.action_type == ActionType.BUILD:
        #         build_count += 1

        # # Make sure we build the right amount
        # if build_count < min(len(order_set), self.amount_to_build): return False

        # return True

    # Turn the order into an CandidateAction
    def parse_order(order):
        s = order.upper() # Hate that I have to do this but some come in as lowercase

        unit_type = s[0]
        unit_location = s[2:5]
        order_string = s

        candidate = CandidateAction(
            #unit = Unit(unit_location, unit_type),

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
        self.our_units = []
        self.unit_string_by_location = {}

        self.fleet_count = 0
        self.army_count = 0

        for u in self.game.get_units(self.power_name):
            if u[0] == "F": self.fleet_count += 1
            else: self.army_count += 1

            self.our_units.append(u[2:5])  # Strip the "A " or "F " at the start
            

        self.all_units = self.game.get_units()
        self.enemy_units = []
        for power, units in self.all_units.items():
            for u in units:
                self.unit_string_by_location[u[2:5]] = u

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

                # Make sure theres actually an edge here
                if self.unit_string_by_location[neighbour][0] == 'A':
                    graph = self.map_graph_army
                else:
                    graph = self.map_graph_navy

                if not graph.has_edge(neighbour, location): continue

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

        self.locations_to_build_fleets = {"POR", "SPA", "NAP", "TUN", "BRE", "LON", "EDI", "LON", "SMY", "GRE", "ANK"}

    def distance_to_enemy(self, start_location):
        visited = set()
        queue = deque([(start_location, 0)])

        while queue:
            loc, dist = queue.popleft()
            if loc in self.enemy_units:
                return dist
            if loc in visited:
                continue
            visited.add(loc)

            for neighbour in self.neighbours[loc]:
                if neighbour not in visited:
                    queue.append((neighbour, dist + 1))

        return float('inf')

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
            if action.build_type == "F":
                if self.fleet_count >= self.army_count - 1: 
                    return -50
                if self.army_count > self.fleet_count + 3:
                    return -50
                if unit_location in self.locations_to_build_fleets:
                    return 200
                
            return 100
        
        elif action_type == ActionType.CONVOY:
            return -100
        
        elif action_type == ActionType.WAIVE:
            return -100
        
        elif action_type == ActionType.RETREAT:            
            return 0

        elif action_type == ActionType.MOVE:
            if target in self.unoccupied_centres: score += 50 * season() # New centre
            elif target in self.enemy_centres: score += 50 * season() # Enemy centre
            #elif target in self.our_centres: score += 5 # Defending our centre

            if unit_location in self.our_centres: # Leaving it empty
                if self.opp_counts[unit_location] > 0: score -= 50 * season() # Opps around

            support_diff = self.opp_counts[target] - self.support_counts[target]

            if support_diff <= -3:  # We stuck in our city
                score -= 200
            else:
                score += 50

            #if target == "VEN":
            #    print("Attack!")

            if action.unit_type == "F" and self.game.map.loc_type.get(target, None) == "WATER":
                score += 30 # Move into some water

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
                    proximity += max(0, 3 - d)
            score += proximity
            
            # Move closer to enemy
            if self.distance_to_enemy(target) > self.distance_to_enemy(unit_location):
                score -= 20

        elif action_type == ActionType.HOLD:
            if unit_location in self.our_centres:  # Defending our centre
                if self.opp_counts[unit_location] == 0: score = -30 # No opps around, just move
                #else: score = 30

                score *= season()
            else: # Grabbing a new centre is always good
                if unit_location in self.unoccupied_or_enemy_centres:
                    score += 100
                    #print("grabbing new one!")
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
                    score += 40
                    #print("Supporting direct attack! {action.order_string}")
                    #quit()
                
                if support_target:
                    if support_target in self.unoccupied_or_enemy_centres: # Supporting a move into new centre 
                    
                        score += 50
                    else: # Just a random move
                        support_diff = self.opp_counts[support_target] - self.support_counts[support_target]

                        if support_diff <= -2:  # We stuck in our city
                            score -= 100
                        else:
                            score += 20
                    #print(f"Move into centre {support_target}")
                elif supported_unit in self.unoccupied_or_enemy_centres: # Supporting a new hold
                    score += 50
                else:
                    score += 5

        #print(action.order_string, score)
        
        return score
    
    def evaluate_partial_order_set(self, order_set):
        move_targets = set()
        support_target_counts = defaultdict(int)

        score = 0

        for action in order_set:
            if action.action_type == ActionType.SUPPORT:
                if action.support_target:
                    support_target_counts[action.support_target] += 1
                else:
                    support_target_counts[action.supported_unit] += 1

            if action.action_type == ActionType.MOVE:
                move_targets.add(action.target)

        for target, count in support_target_counts.items():
            if count >= 2 and target not in move_targets: return -1000
            
        score += sum([x.score for x in order_set])

        return score
    
    def generate_candidates(self, amount, unit_location):
        all_orders = self.game.get_all_possible_orders()
        unit_orders = all_orders.get(unit_location, [])

        order_status = self.game.get_order_status()

        # Evaluate all the orders individually
        candidates = []
        for order in unit_orders:
            action = StudentAgent.parse_order(order)

            # Skip supports for now
            if action.action_type == ActionType.SUPPORT: continue

            # Skip convoys too
            if action.via_convoy: continue

            # Make sure it actually is us and not the enemy
            if action.action_type == ActionType.RETREAT or action.action_type == ActionType.DISBAND:
                if self.power_name in order_status and f"{action.unit_type} {action.unit_location}" not in order_status[self.power_name]: 
                    continue

            action.score = self.evaluate_candidate_action(action)
            candidates.append(action)

        # Also build some support moves
        
        # Return the best ones
        candidates.sort(key=lambda x: x.score, reverse=True)
        
        return candidates[:amount]

    # Return a quick estimate of the board state in the favour of the agent
    @timeout_decorator.timeout(1)
    def eval(self, game: Game):
        # Using dictionary to weight different properties
        propertyWeightings = {
            "our_centres": 30,
            "enemy_centres_lead": 5,
            "next_turn_centres": 20,
            "threat_to_our_centres": -5,
            "support_for_our_units": 5
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
            if scsLead >= 2:
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
    
    def synergy_bonus(self, action_list, new_action):
        bonus = 0.0
        penalty = 0.0

        # Self bounce
        if new_action.action_type == ActionType.MOVE:
            for a in action_list:
                if a.action_type == ActionType.MOVE and a.target == new_action.target:
                    # Conflict: two units moving to the same place
                    penalty -= 100.

        # --- Check synergies: support matches a move in the set ---
        if new_action.action_type == ActionType.SUPPORT:
            for a in action_list:
                # If a is a move, and the support is for this unit to its target
                if (a.action_type == ActionType.MOVE and
                    a.unit_location == new_action.supported_unit and
                    a.target == new_action.support_target):
                    bonus += 100

        # --- Also check the reverse: if new_action is a move, existing support actions match it ---
        if new_action.action_type == ActionType.MOVE:
            for a in action_list:
                if (a.action_type == ActionType.SUPPORT and
                    a.supported_unit == new_action.unit_location and
                    a.support_target == new_action.target):
                    bonus += 100

        randomness = random.uniform(-20, 20)

        return bonus + penalty + randomness

    def joint_synergy_bonus(self, partial_assignment, new_power, new_order_set):
        bonus = 0.0
        penalty = 0.0

        # Compare new_order_set’s orders to already chosen ones
        for other_power, order_set in partial_assignment.items():
            for order in order_set:
                for new_order in new_order_set:
                    # e.g. check for move+support synergy across powers
                    if (new_order.action_type == ActionType.SUPPORT and
                        new_order.supported_unit == order.unit_location and
                        new_order.support_target == order.target):
                        bonus += 200
                    if (new_order.action_type == ActionType.MOVE and
                        order.action_type == ActionType.SUPPORT and
                        order.supported_unit == new_order.unit_location and
                        order.support_target == new_order.target):
                        bonus += 200

                    # penalize conflicting moves
                    if (order.action_type == ActionType.MOVE and
                        new_order.action_type == ActionType.MOVE and
                        order.target == new_order.target):
                        penalty -= 100

        noise = random.uniform(-20, 20)
        return bonus + penalty + noise
    
    def get_product(self, candidate_actions, unit_locations):
        BEAM_WIDTH = 100

        EXACT_THRESHOLD = 5000

        # # Skipping empty guys
        candidate_actions_no_empty = {k: v for k, v in candidate_actions.items() if v}
        unit_locations_fixed = [u for u in unit_locations if u in candidate_actions_no_empty]

        #print(len(candidate_actions) - len(candidate_actions_no_empty))

        # # # Compute the total combinations
        # total_combos = 1
        # for unit in unit_locations_fixed:
        #     total_combos *= len(candidate_actions[unit])
        #     if total_combos > EXACT_THRESHOLD:
        #         break 

        # # # If small we can brute force it
        # if total_combos <= EXACT_THRESHOLD:
        #     all_combos = []
        #     unit_candidates = [candidate_actions[u] for u in unit_locations_fixed]
        #     for combo in product(*unit_candidates):
        #         if self.is_valid_order_set(combo):
        #             all_combos.append(list(combo))
        #     return all_combos

        beams = [(0, [])]  # (score, [actions])

        for i, unit in enumerate(unit_locations_fixed):
            new_beams = []
            for score, action_list in beams:
                for action in candidate_actions[unit]:
                    new_score = score + action.score + self.synergy_bonus(action_list, action)
                    
                    new_action_list = action_list + [action]
                    new_beams.append((new_score, new_action_list))

            # Keep top amount only
            new_beams.sort(key=lambda x: x[0], reverse=True)
            top_beams = new_beams[:BEAM_WIDTH]

            if len(new_beams) > BEAM_WIDTH:
                remainder = new_beams[BEAM_WIDTH:]
                random_beams = random.sample(remainder, min(BEAM_WIDTH//10, len(remainder)))
            else:
                random_beams = []

            beams = top_beams + random_beams

        #print(f"First beam generated {len(beams)} beams:")
        # if self.season == "W":
        #     for s, a in beams:
        #         print([action.order_string for action in a])

        # after full expansion, evaluate full order sets
        valid_beams = []
        for s, a in beams:
            if self.is_valid_order_set(a):
                #full_score = self.evaluate_partial_order_set(a)
                valid_beams.append((s, a))

        #valid_beams = [(s, a) for s, a in beams if self.is_valid_order_set(a)]
        #print(f"{len(valid_beams)} beams are valid.")

       # if len(valid_beams) == 0:
         #   for s, a in beams:
        #        print([action.order_string for action in a])

           # quit()

        all_actions = [actions for _, actions in valid_beams]

        final = all_actions[:100]

        # Get some randos in there too
        if len(all_actions) > 300:
            final.extend(all_actions[300:330])

        return final
    
    def get_joint_product(self, candidate_orders_per_power, power_list, BEAM_WIDTH):
        beams = [(0, {})]  # (score, {power: order_set})

        for power in power_list:
            new_beams = []
            for score, partial_assignment in beams:
                for order_set in candidate_orders_per_power[power]:
                    new_score = score + self.joint_synergy_bonus(partial_assignment, power, order_set) 
                    new_assignment = partial_assignment.copy()
                    new_assignment[power] = order_set
                    new_beams.append((new_score, new_assignment))

            # keep top BEAM_WIDTH only
            new_beams.sort(key=lambda x: x[0], reverse=True)
            beams = new_beams[:BEAM_WIDTH]


        #pprint.pprint(f"We have {len(beams)} joint beams.")

        return [assignment for _, assignment in beams]
    
    @timeout_decorator.timeout(1)
    def get_random_orders(self, power_name):
        possible_orders = self.game.get_all_possible_orders()
        
        orderable_locations = self.game.get_orderable_locations(power_name)
        
        #power_orders = [StudentAgent.parse_order(random.choice(possible_orders[location])) for location in orderable_locations if possible_orders[location]]

        power_orders = [StudentAgent.parse_order(f"{possible_orders[location][0][0:5]} H") for location in orderable_locations if possible_orders[location]]

        return power_orders 
    
    def get_likely_enemy_orders(self, power_name):
        return self.get_random_orders(power_name)
    
    def generate_supports(self, amount, unit_location, candidate_actions):
        all_orders = self.game.get_all_possible_orders()
        unit_orders = all_orders.get(unit_location, [])

        new_candidates = defaultdict(list)
        unit_type = self.unit_string_by_location[unit_location][0]

        if unit_type == 'A':
            graph = self.map_graph_army
        else:
            graph = self.map_graph_navy

        #print('Generating supports')

        for other_unit, unit_candidates in candidate_actions.items():
            if other_unit == unit_location: continue

            for action in unit_candidates:
                order_string = f"{unit_type} {unit_location} S {action.order_string}"

                # Build a support if we can reach
                if action.action_type != ActionType.MOVE: continue

                #if action.target in self.neighbours[unit_location]:
                if graph.has_edge(unit_location, action.target):
                    # Make sure fleet can reach it
                    #print(f"Yes we can support {action.target} from {unit_location} ")

                    new = CandidateAction(
                        #unit = Unit(unit_location, unit_type),

                        unit_location = unit_location,
                        unit_type = unit_type,
                        order_string = order_string,

                        action_type = ActionType.SUPPORT,
                        supported_unit = other_unit,
                        support_target = action.target
                    )

                    new.score = self.evaluate_candidate_action(new)

                    new_candidates[unit_location].append(new)

                    #print(f"Added {new.order_string}")

        for unit in new_candidates:
            new_candidates[unit].sort(key = lambda x: x.score, reverse=True)
            candidate_actions[unit].extend(new_candidates[unit][:amount])
            candidate_actions[unit].sort(key = lambda x: x.score, reverse=True)
    
    def generate_our_orders(self):
        start = time.time_ns()
        # Generate our candidate actions
        CANDIDATE_ORDER_COUNT = 3
        candidate_actions = {}

        for unit in self.our_units:
            #print(unit)
            candidate_actions[unit] = self.generate_candidates(CANDIDATE_ORDER_COUNT, unit)

        #pprint.pprint(candidate_actions)

        # Generate some supports to back up other agent moves
        for unit in self.our_units:
            self.generate_supports(CANDIDATE_ORDER_COUNT, unit, candidate_actions)

        
        #quit()

        #print(f"Our units: {self.our_units}")

        # If we in a build phase
        if self.season == "W":
            for sc in self.our_centres:
                candidate_actions[sc] = self.generate_candidates(CANDIDATE_ORDER_COUNT, sc)

        #pprint.pprint(candidate_actions)

        # print(f"Candidate actions took {round((time.time_ns() - start)/1000000)}ms.")
        # start = time.time_ns()

        #print(f"Our candidate actions are: ")
        pprint.pprint(candidate_actions)

        # Get the interaction graph, and get the connected components
        # graph = StudentAgent.build_interaction_graph(self.our_units, candidate_actions, self.neighbours)

        # if self.season == "W":
        #     graph = StudentAgent.build_interaction_graph(list(self.our_centres), candidate_actions, self.neighbours)

        # components = []
        # for comp in nx.connected_components(graph):
        #     components.append(comp)

        # # print(f"Building interaction graphs took {round((time.time_ns() - start)/1000000)}ms.")
        # start = time.time_ns()

        # print("\nComponents of the interaction graph:")
        # pprint.pprint(components)

        # Generate order sets for each component separately
        #component_order_sets = []
        #print(components)

        #list_time = 0

        #for units in components:
            
            #print(f"Checking component: {units}")
        #s = time.time_ns()
        unit_locations = self.our_units
        if self.season == "W":
            unit_locations = self.our_centres

        order_sets = self.get_product(candidate_actions, unit_locations)

        #quit()
        #list_time += time.time_ns() - s
        #pprint.pprint(candidate_actions)

        #print(f"Resulting order sets: {order_sets}")
        
        # order_lists = [order_set for order_set in order_sets]

        # if not order_lists:
        #     fallback = [[candidate_actions[u][0] for u in units if candidate_actions[u]]]
        #     order_lists = fallback
        #     pprint.pprint(f"Fallback is {fallback}")
        #     quit()

        # component_order_sets.append(order_lists)

        # # print(f"Generating orders for each component took {round((time.time_ns() - start)/1000000)}ms")
        # # print(f"get_product() took {round((list_time)/1000000)}ms")
        # # start = time.time_ns()

        # # print(f"Final order sets has: {len(component_order_sets)}\n\n")

        # # Cartesian product over each component
        # full_orders = []
        # for prod_tuple in product(*component_order_sets):
        #     #print(prod_tuple)

        #     # flatten list of lists
        #     combined = []
        #     for chunk in prod_tuple:
        #         combined.extend(chunk)
        #     full_orders.append(combined)
        
        order_sets.sort(key=lambda o: len(o), reverse=True)

        #print(full_orders)

        

        #print(f"Our orders are: {order_sets}")
        #if self.season == "W": quit()
        #quit()

        #print(f"Final product took {round((time.time_ns() - start)/1000000)}ms. We have {len(full_orders)}")

        return order_sets
    
    # Generate order sets for all powers
    #@timeout_decorator.timeout(1)
    def generate_joint_order_sets(self):
        our_order_sets = self.generate_our_orders()

        enemy_order_sets = {}
        ENEMY_ORDER_SET_COUNT = 3

        for power in self.game.powers.keys():
            if power == self.power_name: continue

            enemy_order_sets[power] = []

            for i in range(ENEMY_ORDER_SET_COUNT):
                enemy_order_sets[power].append(self.get_likely_enemy_orders(power))

        all_power_order_sets = enemy_order_sets
        all_power_order_sets[self.power_name] = our_order_sets

        power_list = list(all_power_order_sets.keys())

        #pprint.pprint(all_power_order_sets)
        joint_order_sets = self.get_joint_product(all_power_order_sets, power_list, BEAM_WIDTH=100)

        return joint_order_sets

    @timeout_decorator.timeout(1)
    def get_actions(self):
        # Whats the game state
        phase = self.game.get_current_phase()
        year = phase[1:5]
        season = phase[0]

        self.calculate_map_properties()

        #print("-------------------------")

        # Get all possible orders
        print(f"Getting actions for phase {phase}")
        now = time.time_ns()
        order_sets = self.generate_joint_order_sets()
        order_finish = time.time_ns()
        print(f"Took {round((order_finish - now)/1000000)}ms to generate orders.")

        # Search 1 move ahead and get the best eval
        best_eval = float('-inf')
        best_order_set : dict = {}

        #print(f"Checking {len(order_sets)} states.")

        #if self.season == "W":
        #    pprint.pprint(order_sets)
        
        #with tqdm(total=len(order_sets)) as pbar:
        for order_set in order_sets:
            orders = {}

            for power, order in order_set.items():
                orders[power] = []
                for o in order:
                    o.power_name = power
                    orders[power].append(o)

            adjacency = self.game.map.loc_abut
            supply_centres_by_power = self.game.get_centers() 
            units = self.game.get_units()                     
            phase = phase                 
            
            newGame = Game(adjacency, supply_centres_by_power, units, phase)
            newGame.resolve_orders(orders)

            evaluation = self.eval(newGame)

            #print(f"Orderset {order_set} had evaluation {evaluation}")

            if evaluation > best_eval:
                best_eval = evaluation
                best_order_set = order_set[self.power_name]

        best_order_set_strings = [x.order_string for x in best_order_set]

        finish = time.time_ns()
        print(f"Took {round((finish - now)/1000000)}ms.")
        print(f"Finished phase {phase} with orders {best_order_set_strings}\n")

        self.game.render(output_path='img.svg')

        return best_order_set_strings