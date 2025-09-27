# ZERO generative AI was used in this project. All features have been hand coded, tested and refined over countless hours.

import random
import networkx as nx
import timeout_decorator
from agent_baselines import Agent
from itertools import product
from tqdm import tqdm
from functools import cache
from enum import Enum
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Optional
import math

class ActionType(Enum):
    HOLD = 1
    MOVE = 2
    SUPPORT = 3
    CONVOY = 4
    BUILD = 5
    DISBAND = 6
    RETREAT = 7
    WAIVE = 8

class MCTSNode:
    def __init__(self, game_state, parent=None, action=None):
        self.game_state = game_state        
        self.parent = parent
        self.action = action                # CandidateAction or order set that led here
        self.children = []
        self.visits = 0
        self.value = 0.0                    # Sum of rollout results

    def ucb_score(self, exploration=1.4):
        if self.visits == 0:
            return float('inf')
        mean_value = self.value / self.visits
        return mean_value + exploration * math.sqrt(
            math.log(self.parent.visits + 1) / self.visits
        )

    def best_child(self):
        return max(self.children, key=lambda c: c.visits)

# Used to generate some candidate actions to search through
@dataclass
class CandidateAction():
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
        self.units_by_power = defaultdict(list, {
            power: [ustr.strip().upper() for ustr in ulist]
            for power, ulist in units.items()
        })

        # Units by location {location: (power, unit_type)}
        self.units_by_location = {}
        for power, ulist in self.units_by_power.items():
            for s in ulist:
                unit_type = s[0]  # A or F
                unit_location = s[2:5] 
                self.units_by_location[unit_location] = (power, unit_type)

        self.phase = phase

        self.orders_by_power = {}
        self.orders_by_location = {}

    def copy_game(game):
        return Game(
            adjacency=game.adjacency,
            supply_centres_by_power=defaultdict(list, {p: centres[:] for p, centres in game.supply_centres_by_power.items()}),
            units={p: lst[:] for p, lst in game.units_by_power.items()},
            phase=game.phase
        )

    def get_centers(self, power_name=None):
        if power_name:
            return self.supply_centres_by_power[power_name] 

        return self.supply_centres_by_power

    def get_units(self):
        out = defaultdict(list)
        for location, (power, unit_type) in self.units_by_location.items():
            out[power].append(f"{unit_type} {location}")
        for p in self.units_by_power.keys():
            out.setdefault(p, [])
        return dict(out)

    def resolve_orders(self, orders_by_unit):
        orders_by_location = defaultdict(list)
        for power, order_list in orders_by_unit.items():
            if power not in self.units_by_power: continue
            units_by_power = self.units_by_power[power]
            for order in order_list:
                if order.action_type == ActionType.BUILD:
                    self.units_by_location[order.unit_location] = (power, order.build_type)
                    units_by_power.append(f"{order.build_type} {order.unit_location}")
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

        def support_count_for(attacker_loc: str, target_loc: str) -> int:
            key = (attacker_loc, target_loc)
            supporters_list = supports.get(key)
            if not supporters_list:
                return 0
            if attacker_loc in moves:
                return len(supporters_list) if moves[attacker_loc] == target_loc else 0
            return len(supporters_list) if attacker_loc == target_loc else 0

        contests = defaultdict(list)

        for mover_loc, dest in moves.items():
            unit_info = self.units_by_location.get(mover_loc)
            if not unit_info:
                continue
            mover_power, mover_unit_type = unit_info
            sup_count = support_count_for(mover_loc, dest)
            contests[dest].append((mover_loc, mover_power, mover_unit_type, 1 + sup_count))

        units_keys = self.units_by_location.keys()
        for dest in set(contests.keys()) | set(units_keys):
            occupant_info = self.units_by_location.get(dest)
            if not occupant_info or dest in moves:
                continue
            occupant_power, occupant_unit_type = occupant_info
            sup_count = support_count_for(dest, dest)
            contests[dest].append((dest, occupant_power, occupant_unit_type, 1 + sup_count))

        new_positions = {}
        removed_origins = set()

        for dest, candidates in contests.items():
            if not candidates:
                continue
            max_strength = max(c[3] for c in candidates)
            winners = [c for c in candidates if c[3] == max_strength]
            if len(winners) > 1:
                continue
            winner_loc, winner_power, winner_unit_type, _ = winners[0]
            new_positions[dest] = (winner_power, winner_unit_type)
            if winner_loc != dest:
                removed_origins.add(winner_loc)
                if dest in self.units_by_location:
                    removed_origins.add(dest)

        for orig_loc, (power, unit_type) in self.units_by_location.items():
            if orig_loc not in removed_origins and orig_loc not in new_positions:
                new_positions[orig_loc] = (power, unit_type)

        new_units_by_power = defaultdict(list)
        new_units_by_location = {}
        for location, (power, unit_type) in new_positions.items():
            new_units_by_location[location] = (power, unit_type)
            new_units_by_power[power].append(f"{unit_type} {location}")

        self.units_by_location = new_units_by_location
        self.units_by_power = defaultdict(list, {p: lst[:] for p, lst in new_units_by_power.items()})

        if self.phase[0] == "F":
            for location, (power, _) in new_positions.items():
                if location in self.who_owns_supply_centres:
                    self.who_owns_supply_centres[location] = power
            new_scs_by_power = defaultdict(list)
            for sc, owner in self.who_owns_supply_centres.items():
                if owner:
                    new_scs_by_power[owner].append(sc)
            self.supply_centres_by_power = defaultdict(list, new_scs_by_power)

        orders_by_power = defaultdict(list)
        orders_by_location = defaultdict(list)

        adjacency = self.adjacency
        units_by_location = self.units_by_location
        supply_centres_by_power = self.supply_centres_by_power

        for location, (power, unit_type) in units_by_location.items():
            orders_by_power[power].append(f"{unit_type} {location} H")
            orders_by_location[location].append(f"{unit_type} {location} H")

            for adj in adjacency.get(location, ()):
                orders_by_power[power].append(f"{unit_type} {location} - {adj}")
                orders_by_location[location].append(f"{unit_type} {location} - {adj}")

            for adj in adjacency.get(location, ()):
                adj_info = units_by_location.get(adj)
                if adj_info:
                    adj_power, adj_unit_type = adj_info
                    orders_by_power[power].append(f"{unit_type} {location} S {adj_unit_type} {adj}")
                    orders_by_location[location].append(f"{unit_type} {location} S {adj_unit_type} {adj}")

            if unit_type == "F":
                for adj in adjacency.get(location, ()):
                    adj_info = units_by_location.get(adj)
                    if adj_info:
                        adj_power, adj_unit_type = adj_info

            if self.phase[0] == "B":
                for sc in supply_centres_by_power.get(power, ()):
                    if sc not in units_by_location:
                        orders_by_power[power].append(f"A {sc} B")
                        orders_by_power[power].append(f"F {sc} B")
                        orders_by_location[location].append(f"A {sc} B")
                        orders_by_location[location].append(f"F {sc} B")

        self.orders_by_power = orders_by_power
        self.orders_by_location = orders_by_location

    def get_all_possible_orders(self):
        return self.orders_by_location
    
    def get_orderable_locations(self, power_name):
        return [x[2:5] for x in self.units_by_power[power_name]]

class StudentAgent(Agent):
    def __init__(self, agent_name='wining bot'):
        super().__init__(agent_name)

        self.map_graph_army = None
        self.map_graph_navy = None

        self.last_unit_position_by_power = defaultdict(list)
        self.hold_counts = defaultdict(int)

        self.root = None

    def new_game(self, game, power_name):
        self.game = game
        self.power_name = power_name

        self.build_map_graphs()

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

    def start_mcts(self, top_order_sets, rollout_depth):
        adjacency = self.game.map.loc_abut
        supply_centres_by_power = self.game.get_centers() 
        units = self.game.get_units()                     
        phase = self.game.phase                 

        # Initialize root game and node
        root_game = Game(adjacency, supply_centres_by_power, units, phase)
        self.root = MCTSNode(root_game)

        # Seed root children with top joint order sets
        for joint_orders in top_order_sets:
            child_game = Game(adjacency, supply_centres_by_power, units, phase)
            child_game.resolve_orders(joint_orders)
            child_node = MCTSNode(child_game, parent=self.root, action=joint_orders)
            self.root.children.append(child_node)

        # Standard MCTS iterations
        # Will timeout automatically
        while True:
            node = self.mcts_selection(self.root)
            if node.visits == 0:
                reward = self.mcts_rollout(node.game_state, depth=rollout_depth)
                self.mcts_backpropagate(node, reward)
            else:
                expanded = self.mcts_expand(node)
                if expanded is not None:
                    reward = self.mcts_rollout(expanded.game_state, depth=rollout_depth)
                    self.mcts_backpropagate(expanded, reward)

    def mcts_selection(self, node):
        while node.children:
            node = max(node.children, key=lambda c: c.ucb_score())
        return node

    def mcts_expand(self, node):
        # Clone the game apply orders for all powers
        new_game = node.game_state.copy_game()
        all_orders = {}
        for power in self.game.powers.keys():
            all_orders[power] = self.get_likely_enemy_orders(new_game, power)

        new_game.resolve_orders(all_orders)

        child = MCTSNode(new_game, parent=node, action=all_orders[self.power_name])
        node.children.append(child)
        return child

    def mcts_rollout(self, game_state, depth):
        rollout_game = game_state.copy_game()
        for _ in range(depth):
            # Add random orders for all powers
            all_orders = {}
            for power in self.game.powers.keys():
                all_orders[power] = self.get_likely_enemy_orders(rollout_game, power)

            rollout_game.resolve_orders(all_orders)

        # evaluate: sum of candidate action scores
        evaluation = self.eval(game_state)
        return evaluation  # crude but works; can combine with evaluate_partial_order_set

    def mcts_backpropagate(self, node, reward):
        while node is not None:
            node.visits += 1
            node.value += reward
            node = node.parent

    def is_valid_order_set(self, order_set):
        actions_by_unit = {a.unit_location: a for a in order_set} # Dictionary comprehension go crazy

        if len(order_set) == 0: return False

        build_count = 0

        convoy_move_targets = set()
        convoy_targets = set()

        for action in order_set:
            if action.action_type == ActionType.MOVE:
                if action.via_convoy: # There has to also be someone doing the convoying
                    convoy_move_targets.add(action.target)

            elif action.action_type == ActionType.BUILD:
                build_count += 1
            elif action.action_type == ActionType.CONVOY:
                convoy_targets.add(action.support_target)

        for target in convoy_move_targets:
            if target not in convoy_targets:
                return False
            
        if build_count < min(len(order_set), self.amount_to_build): return False

        return True

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

        self.enemy_centres = self.occupied_centres - self.our_centres
        self.unoccupied_or_enemy_centres = self.unoccupied_centres.union(self.enemy_centres)

        self.full_phase = self.game.get_current_phase()

        self.all_unit_locations = {}
        for power, units in self.game.get_units().items():
            for unit in units:
                self.all_unit_locations[unit[2:5]] = (power, unit)

            if power == self.power_name: continue

            if not (self.full_phase[0] == "F" or self.full_phase[0] == "S") and self.full_phase[0] == "M": continue

            # See if theres any difference since last time
            old_spots = self.last_unit_position_by_power.get(power, set())
            units_set = set(units)
            fixed_set = set()

            all_holding = True
            for unit in units_set:
                fixed_unit = unit

                if unit[0] == "*": 
                   fixed_unit = unit[1:]
                
                fixed_set.add(fixed_unit)

                if fixed_unit not in old_spots: all_holding = False

            if all_holding:
                self.hold_counts[power] += 1
            else:
                self.hold_counts[power] = 0

            self.last_unit_position_by_power[power] = fixed_set

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
            self.enemy_units.extend([u[2:5] for u in units])

        self.enemy_units = set(self.enemy_units)
            
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

        self.locations_to_build_fleets = {"POR", "SPA", "NAP", "TUN", "BRE", "SMY", "GRE", "ANK", "EDI", "LVR"}
        self.locations_to_convoy = {"LON", "WAL", "EDI", "CLY"}
        self.not_in_england = {"PIC", "BEL", "HOL", "NWY", "BRE"}

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
    
    def fast_action_evaluation(self, game : Game, action : CandidateAction, power_name):
        score = 0

        target = action.target
        unit_location = action.unit_location
        action_type = action.action_type
        support_target = action.support_target
        supported_unit = action.supported_unit

        if action_type == ActionType.BUILD:
            return 100
        
        elif action_type == ActionType.WAIVE:
            return -100
        
        elif action_type == ActionType.MOVE:
            if target not in game.get_centers(power_name):
                score += 50

        elif action_type == ActionType.HOLD:
            if unit_location in self.all_centres: # Holding
                score += 20
                if unit_location not in game.get_centers(power_name): # Grabbing enemy
                    score += 50

        elif action_type == ActionType.SUPPORT:
            if support_target:
                if support_target in self.all_centres: # New sc move
                    score += 50
            else:
                if unit_location in self.all_centres: # New sc hold
                    score += 30

        return score

    # Evaluation of a candidate action for this power
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
                    score = -300
                if unit_location in self.locations_to_build_fleets:
                    score = 200
            if action.build_type == "A":
                if self.army_count >= self.fleet_count + 2:
                    score = -300
                
            # Put this guy where there are more opps
            score += 300 - self.distance_to_enemy(unit_location)*100
            score -= self.support_counts[action.unit_location]*100

        elif action_type == ActionType.CONVOY:
            if support_target in self.unoccupied_or_enemy_centres: # Convoying a move into new centre 
                score += 20 * season()
            else: # Get tf out of england
                if supported_unit in self.locations_to_convoy or unit_location in self.locations_to_convoy:
                    score += 200
                if target in self.not_in_england:
                    score += 100
        
        elif action_type == ActionType.WAIVE:
            return -100
        
        elif action_type == ActionType.RETREAT:            
            return 100
        
        elif action_type == ActionType.DISBAND:
            if unit_location in self.our_centres: score -= 200

            score -= self.opp_counts[unit_location]*100

        elif action_type == ActionType.MOVE:
            if target in self.unoccupied_centres: 
                score += 50 * season() # New centre
                if action.unit_type == "F": score += 5
            elif target in self.enemy_centres: score += 50 * season() # Enemy centre

            if unit_location in self.our_centres: # Leaving it empty
                if self.opp_counts[unit_location] > 0: 
                    if target in self.unoccupied_or_enemy_centres:
                        score -= 30 * season() 
                    else:
                        score -= 1000 # Opps around
            if target in self.our_centres and self.opp_counts[target] > 1:
                score += 40

            # Please move into empty spots
            if target not in self.our_units:
                score += 20
            
            support_count = self.support_counts[target]
            if action.via_convoy: 
                if self.map_graph_army.has_edge(unit_location, target): # Bro just walk
                    score -= 1000

                elif unit_location in self.locations_to_convoy and target in self.not_in_england:
                    score += 200

                support_count -= 1

            support_diff = self.opp_counts[target] - support_count

            if support_diff <= -3:  # We stuck in our middle
                score -= 200

            if action.unit_type == "F" and self.game.map.loc_type.get(target, None) == "WATER":
                # Move next to other fleets
                for n in self.neighbours[target]:
                    if self.unit_string_by_location.get(n, "A")[0] == "F": score += 50

                score += 20 # Move into some water

            # Attacking an enemy
            if self.all_unit_locations.get(target, (self.power_name, False))[0] != self.power_name:
                if support_count > 1: 
                    score += 50 * support_count # Be more aggressive
                else:
                    score -= 100
            
            # Check the proximity to other supply centres
            proximity = 0
            for sc in self.unoccupied_or_enemy_centres:
                d = self.distance_cache[target].get(sc, 99)

                if d < 99:
                    proximity += max(0, 3 - d)
            score += proximity
            
            # Don't move away from enemy
            if self.distance_to_enemy(target) > self.distance_to_enemy(unit_location):
                score -= 20

        elif action_type == ActionType.HOLD:
            if unit_location in self.our_centres:  # Defending our centre
                if self.opp_counts[unit_location] == 0: score = -30 # No opps around, just move

                score *= season()
            else: # Grabbing a new centre is always good
                if unit_location in self.unoccupied_or_enemy_centres:
                    score += 100
                else:
                    score += self.opp_counts[unit_location]*10

        elif action_type == ActionType.SUPPORT:
            if supported_unit and self.all_unit_locations.get(supported_unit, (False, False))[0] == self.power_name: # Supporting our own guy
                # Supporting a direct attack
                if self.all_unit_locations.get(support_target, (self.power_name, False))[0] != self.power_name: 
                    score += 30
                
                if support_target:
                    if support_target in self.unoccupied_or_enemy_centres: # Supporting a move into new centre 
                    
                        score += 30
                    else: # Just a random move
                        support_diff = self.opp_counts[support_target] - self.support_counts[support_target]

                        if support_diff <= -2:  # We stuck in our city
                            score -= 100

                elif supported_unit in self.unoccupied_or_enemy_centres: # Supporting a new hold
                    score += 50
                else:
                    score += 5

        return score
    
    ##@timeout_decorator.timeout(1)
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
                if action.target in move_targets: 
                    score -= 300

                move_targets.add(action.target)

        for target, count in support_target_counts.items():
            if count >= 1 and target in move_targets:
                score += 200 * count

        # penalise leaving our own centres empty
        for action in order_set:
            if action.action_type == ActionType.MOVE and action.unit_location in self.our_centres:
                if self.opp_counts[action.unit_location] > 0:
                    score -= 30
            
        score += sum([x.score for x in order_set])

        #print((score, [x.order_string for x in order_set]))

        return score
    
    ##@timeout_decorator.timeout(1)
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

            # Make sure we can actually perform the convoy - limit to length of 1 (unfortunately)
            if action.via_convoy: 
                valid_convoy = False
                for neighbour in self.neighbours[action.unit_location]: # CHeck neighbours
                    if self.unit_string_by_location.get(neighbour, "A")[0] == "F": # If fleet, check its neighbpours
                        if action.target in self.neighbours[neighbour]: # If target, we good
                            valid_convoy = True

                if not valid_convoy: continue

            # Make sure it actually is us and not the enemy
            if action.action_type == ActionType.RETREAT or action.action_type == ActionType.DISBAND:
                if self.power_name in order_status and f"{action.unit_type} {action.unit_location}" not in order_status[self.power_name]: 
                    continue

            action.score = self.evaluate_candidate_action(action)
            candidates.append(action)
        
        # Return the best ones
        candidates.sort(key=lambda x: x.score, reverse=True)
        
        return candidates[:amount]

    # Return a quick estimate of the board state in the favour of the agent
    def eval(self, game: Game):
        # Using dictionary to weight different properties
        propertyWeightings = {
            "our_centres": 50,
            "enemy_centres_lead": 20,
            "next_turn_centres": 20
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

        # Calculate the weighted evaluation
        evaluation = 0

        for property, value in stateProperties.items():
            evaluation += value * propertyWeightings[property]

        return evaluation

    def synergy_bonus(self, action_list, new_action):
        bonus = 0
        penalty = 0

        # Self bounce
        if new_action.action_type == ActionType.MOVE:
            for a in action_list:
                if a.action_type == ActionType.MOVE and a.target == new_action.target:
                    # Conflict two units moving to the same place gonna self bounce
                    penalty -= 300

        #Check synergies support matches a move in the set
        if new_action.action_type == ActionType.SUPPORT or new_action.action_type == ActionType.CONVOY:
            for a in action_list:
                # If a is a move, and the support is for this unit to its target
                if (a.action_type == ActionType.MOVE and
                    a.unit_location == new_action.supported_unit and
                    a.target == new_action.support_target):
                    bonus += 100

        #Also check the reverse
        if new_action.action_type == ActionType.MOVE:
            for a in action_list:
                if ((a.action_type == ActionType.SUPPORT or a.action_type == ActionType.CONVOY) and
                    a.supported_unit == new_action.unit_location and
                    a.support_target == new_action.target):
                    bonus += 100

        randomness = random.uniform(-10, 10)

        return bonus + penalty + randomness
    
    def get_product(self, candidate_actions, unit_locations):
        BEAM_WIDTH = 100

        # Skipping empty guys
        candidate_actions_no_empty = {k: v for k, v in candidate_actions.items() if v}
        unit_locations_fixed = [u for u in unit_locations if u in candidate_actions_no_empty]

        # Sort unit_locations_fixed by the highest scored action associated with each location
        unit_locations_fixed.sort(
            key=lambda loc: (
            max(candidate_actions[loc], key=lambda action: action.score).score -
            (sorted(candidate_actions[loc], key=lambda action: action.score, reverse=True)[1].score 
             if len(candidate_actions[loc]) > 1 else float('-inf'))
            ),
            reverse=True
        )

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

        # after full expansion, evaluate full order sets
        valid_beams = []
        for s, a in beams:
            if self.is_valid_order_set(a):
                full_score = self.evaluate_partial_order_set(a)
                valid_beams.append((full_score, a))

        valid_beams.sort(key = lambda x: x[0], reverse=True)

        all_actions = [actions for _, actions in valid_beams]

        final = all_actions[:20]

        return final
    
    ##@timeout_decorator.timeout(1)
    def get_joint_product(self, all_power_orders):
        AMOUNT = 20

        power_list = list(all_power_orders.keys())

        our_orders = all_power_orders[self.power_name]
        enemy_powers = [p for p in power_list if p != self.power_name]

        joint_assignments = []

        for order_set in our_orders:
            assignment = {self.power_name: order_set}
            for enemy in enemy_powers:
                assignment[enemy] = random.choice(all_power_orders[enemy])
            joint_assignments.append(assignment)

        return joint_assignments[:AMOUNT]

    # Get random one of two top moves per unit
    def get_likely_enemy_orders(self, game, power_name):
        possible_orders = game.get_all_possible_orders()
        orderable_locations = game.get_orderable_locations(power_name)

        power_orders = []

        for location in orderable_locations:
            if not possible_orders.get(location, None): continue

            orders_at_location = []

            for action in possible_orders[location]:
                parsed = StudentAgent.parse_order(action)

                # If they just gonna hold
                if self.hold_counts.get(power_name, 0) >= 3:
                    if parsed.action_type != ActionType.HOLD: continue

                score = self.fast_action_evaluation(game, parsed,power_name)

                orders_at_location.append((score, parsed))

            orders_at_location.sort(key = lambda x: x[0], reverse = True)
            
            choice = random.randint(0, 1)

            if len(orders_at_location) == 0: continue
            elif len(orders_at_location) == 1: power_orders.append(orders_at_location[0][1])
            else: power_orders.append(orders_at_location[choice][1])

        return power_orders
    
    def generate_supports(self, amount, unit_location, candidate_actions):
        all_orders = self.game.get_all_possible_orders()
        unit_orders = all_orders.get(unit_location, [])

        new_candidates = defaultdict(list)
        unit_type = self.unit_string_by_location[unit_location][0]

        if unit_type == 'A':
            graph = self.map_graph_army
        else:
            graph = self.map_graph_navy

        for other_unit, unit_candidates in candidate_actions.items():
            if other_unit == unit_location: continue

            for action in unit_candidates:
                order_string = f"{unit_type} {unit_location} S {action.order_string}"

                # Build a support if we can reach
                if action.action_type != ActionType.MOVE: continue

                if action.via_convoy:
                    order_string = order_string[:-4] # Remove the VIA at the end of the order

                    if unit_type == "F" and action.unit_type == "A" and self.game.map.loc_type.get(unit_location, None) == "WATER": # Also generate a convoy for this move if we can help
                        if graph.has_edge(unit_location, action.unit_location) and graph.has_edge(unit_location, action.target):
                            new = CandidateAction(
                                unit_location = unit_location,
                                unit_type = unit_type,
                                order_string = f"{unit_type} {unit_location} C {action.order_string[:-4]}",

                                action_type = ActionType.CONVOY,
                                supported_unit = other_unit,
                                support_target = action.target
                            )

                            new.score = self.evaluate_candidate_action(new)

                            new_candidates[unit_location].append(new)

                # Make sure fleet can reach it
                if graph.has_edge(unit_location, action.target):
                    new = CandidateAction(
                        unit_location = unit_location,
                        unit_type = unit_type,
                        order_string = order_string,

                        action_type = ActionType.SUPPORT,
                        supported_unit = other_unit,
                        support_target = action.target
                    )

                    new.score = self.evaluate_candidate_action(new)

                    new_candidates[unit_location].append(new)

        for unit in new_candidates:
            new_candidates[unit].sort(key = lambda x: x.score, reverse=True)
            candidate_actions[unit].extend(new_candidates[unit][:amount])
            candidate_actions[unit].sort(key = lambda x: x.score, reverse=True)
    
    def generate_our_orders(self):
        CANDIDATE_ORDER_COUNT = 3
        candidate_actions = {}

        for unit in self.our_units:
            candidate_actions[unit] = self.generate_candidates(CANDIDATE_ORDER_COUNT, unit)

        # Generate some supports to back up other agent moves
        for unit in self.our_units:
            self.generate_supports(CANDIDATE_ORDER_COUNT, unit, candidate_actions)

        # If we in a build phase
        if self.season == "W":
            for sc in self.our_centres:
                candidate_actions[sc] = self.generate_candidates(CANDIDATE_ORDER_COUNT, sc)

        unit_locations = self.our_units
        if self.season == "W":
            unit_locations = self.our_centres

        order_sets = self.get_product(candidate_actions, unit_locations)

        return order_sets
    
    # Generate order sets for all powers
    ##@timeout_decorator.timeout(1)
    def generate_joint_order_sets(self):
        our_order_sets = self.generate_our_orders()

        enemy_order_sets = {}
        ENEMY_ORDER_SET_COUNT = 3

        for power in self.game.powers.keys():
            if power == self.power_name: continue

            enemy_order_sets[power] = []

            for i in range(ENEMY_ORDER_SET_COUNT):
                enemy_order_sets[power].append(self.get_likely_enemy_orders(self.game, power))

        all_power_order_sets = enemy_order_sets
        all_power_order_sets[self.power_name] = our_order_sets

        joint_order_sets = self.get_joint_product(all_power_order_sets)

        return joint_order_sets

    @timeout_decorator.timeout(1)
    def get_actions(self):
        self.root = None

        try:
            self.get_actions_timed()
        except timeout_decorator.TimeoutError:
            pass

        # pick the most visited child of the root
        if not self.root or not self.root.children:
            return []
        best_orders = self.root.best_child().action
        best_order_set = best_orders[self.power_name]

        best_order_set_strings = [x.order_string for x in best_order_set]

        self.game.render(output_path='img.svg')

        return best_order_set_strings

    @timeout_decorator.timeout(0.95)
    def get_actions_timed(self):
        self.calculate_map_properties()

        # Get all possible orders
        order_sets = self.generate_joint_order_sets()

        if len(order_sets) == 0:
            return
        else:
            self.start_mcts(order_sets, rollout_depth=2)