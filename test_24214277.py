# https://github.com/Resistnz/Diplomacy-AI

"""
Scenario 1:
----- The Per-Power and the Overall Performance of the Player Agent -----
AUSTRIA: SCs - 17.33±0.94, Wins - 66.67%, Survives - 33.33%, Defeats - 0.0%
ENGLAND: SCs - 17.0±1.41, Wins - 66.67%, Survives - 33.33%, Defeats - 0.0%
FRANCE: SCs - 15.67±3.3, Wins - 66.67%, Survives - 33.33%, Defeats - 0.0%
GERMANY: SCs - 18.0±0.0, Wins - 100.0%, Survives - 0.0%, Defeats - 0.0%
ITALY: SCs - 16.33±2.36, Wins - 66.67%, Survives - 33.33%, Defeats - 0.0%
RUSSIA: SCs - 18.0±0.0, Wins - 100.0%, Survives - 0.0%, Defeats - 0.0%
TURKEY: SCs - 18.0±0.0, Wins - 100.0%, Survives - 0.0%, Defeats - 0.0%
ALL: SCs - 17.19±1.87, Wins - 80.95%, Survives - 19.05%, Defeats - 0.0%
-------------------------------------------------------------------------

Scenario 2:
----- The Per-Power and the Overall Performance of the Player Agent -----
AUSTRIA: SCs - 12.0±8.49, Wins - 66.67%, Survives - 0.0%, Defeats - 33.33%
ENGLAND: SCs - 9.67±6.94, Wins - 0.0%, Survives - 66.67%, Defeats - 33.33%
FRANCE: SCs - 16.0±1.63, Wins - 33.33%, Survives - 66.67%, Defeats - 0.0%
GERMANY: SCs - 15.0±4.24, Wins - 66.67%, Survives - 33.33%, Defeats - 0.0%
ITALY: SCs - 13.0±4.55, Wins - 33.33%, Survives - 66.67%, Defeats - 0.0%
RUSSIA: SCs - 16.67±1.89, Wins - 66.67%, Survives - 33.33%, Defeats - 0.0%
TURKEY: SCs - 10.0±0.82, Wins - 0.0%, Survives - 100.0%, Defeats - 0.0%
ALL: SCs - 13.19±5.52, Wins - 38.1%, Survives - 52.38%, Defeats - 9.52%
-------------------------------------------------------------------------
"""

""" Final Agent Workings:

As in the report, the final agent works by generating candidate actions per unit with a local evaluation score,
combining these candidate actions into 20 complete candidate order sets via a beam search, which are 
then handed to a Monte-Carlo Tree Search (MCTS) to complete as many rollouts using lightweight
and slightly randomised likely move generator to a depth of 3 turns ahead, to ultimately return
the node with the best results.

Final Evaluation Function:
The evaluation function handles orders by their order type:

    Builds - Certain coastal locations should have fleets built rather than armies, and the agent should not have too great
    an imbalance between fleets and armies. Additionally, place in centres closer to enemies.
    Convoys - Mainly just used to get units out of England. Convoys are generated specifically for moves
    Retreat - Higher evaluation than disband
    Disband - Avoid giving up centres, disband in normal territories instead
    Moves - Moving into unoccupied centres, attacking enemies with possible support or moving toward enemy centres are all favoured,
    while leaving centres empty is penalised
    Supports - Essentially the same as moves, better quality moves recieve better evaluation
    Holds - Hold when enemies are close, does not hold when no enemies around

Final Position Evaluation:
The evaluation of an overall position is simple, just counting the number of supply centres for each unit. More complex evaluation
such as connectedness of troops was tested, but due to its use in the MCTS it was far too time intensive (removing it decreased execution 
time by 11x)

Beam Search:
Beam search was a part that I agonised over tweaking settings for, until a beam width of 100 was used to return 20 candidate orders.
Beam width win that I tested (scenario 1):
    20 (53%)
    100 (80%)
    200 (78%), since order generation took twice as long and MCTS had less time to rollout

I also found it beneficial to include up to 10 random beams on each iteration. I did not measure the performance directly, however I did not see 
any downside, as it most likely gives small performance boost but at minimal extra cost.

Likely Enemy Move Generation:
This simply used a faster, more stripped down version of the order evaluation, checking only centres and good supports.
Employing this over random rollouts had a noticeable effect in only scenario 2 (as scenario 1 was just holds):

    Random - 13% 
    Likely Moves - 25% (at the time)

Additionally, detecting which agents are likely to be HOLD agents dramatically increased scenario 1 win rate:

    No hold detection - 67%
    Hold detection - 88% (at the time)

Scenario 1 was never at 100%, as increasing performance in scenario 1 often led to the agent overextending and underperforming in 
scenario 2.

MCTS Depths in scenario 2:
    rollout_depth: 0 (i.e just 1-ply) - 14%
    rollout_depth: 1 - 23%
    rollout_depth: 2 - 25%
    rollout_depth: 3 - 24%

    Random rollouts - 17%
    Heuristic rollouts - ~25%

It is highly possible these results are just the result of variance, however it does make sense that larger rollout depths
decreases the number of iterations that can be performed.

More statistics can be found in the code snippets below, in mostly chornological order.

"""

# My initial single-ply search
# Using cartesian product for all combinations worked extremely well for small amount of units, took upwards of 30 seconds
# for ~8 agents. With infinite time, this would have won 100% of scenario 1 games, I just would stop it at year ~1907 and by then
# it had picked up around 10 scs. 
"""
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
            newGame.set_orders(p, orders[p])

        evaluation = StudentAgent.eval(self.power_name, newGame)

        if evaluation > best_eval:
            best_eval = evaluation
            best_order_set = order_set

        pbar.update(1)

print(f"Finished phase {phase}")

return best_order_set
"""

# Initial move eval stats:
"""
    Stats on scenario 1 with single ply search:

    Differentiating enemy/our centres (4.6 scs without vs 11.1 scs with)
    Preferencing distance to closest centre (~10 scs vs ~13 scs)
    Leaving owned scs emtpy when enemies around (~10 vs ~9)*
    Moving to places with possible support (no change)**

    *this was only noticeable in scenario 2, but I did not run any specific tests with it
    **supports were not properly implemented yet
"""

# Interaction Graph:
# This unfortunately ended up having no real benefit, as decribed in the report
"""
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

    return graph

...

graph = StudentAgent.build_interaction_graph(our_units, candidate_actions, neighbours)

components = []
for comp in nx.connected_components(graph):
    components.append(comp)

# Generate order sets for each component separately
component_order_sets = []

for units in components:
    order_sets = StudentAgent.get_product(candidate_actions, units)

    order_lists = [[action.order_string for action in order_set] for order_set in order_sets]

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

"""

# Order Set Validation with full cartesian product
# This was one of the most impactful changes, as it drastically reduced cartesian product

# >30s with single-ply search to ~5s 
# is_valid_order_set() was the same as it is now

"""
def get_product(candidate_actions, unit_locations):
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
"""

# Better move eval
# This function combined with the old one
# Stats in scenario 2: 
#   Weighting supply centres based on season (5% without vs 7% with), most likely variance but still useful
#   Leaving centres empty when enemies around (~6% vs 13%)
#   Only attacking when with support (8.3% vs ~11%)
#   
# 
# Stats in scenario 1:
#   Ensuring the correct amount of units were built if possible (~25% vs ~40%)
#   !Proximity to other supply centres (43% vs 84.3%)*
#
# *The agent finally started to expand rather than clump together. This had poor results in scenario 2.
"""
def evaluate_candidate_action(self, action : CandidateAction):
    def season(): 
        if self.season == "S": return 0.5
        if self.season == "F": return 1.5

        return 1

    score = 0

    target = action.target
    unit_location = action.unit_location
    action_type = action.action_type
    support_target = action.support_target
    supported_unit = action.supported_unit

    if action_type == ActionType.MOVE:
        score += 2 # Encourage moving a lil bit

        if target in self.unoccupied_centres: score += 20 # New centre
        if target in self.enemy_centres: score += 12 # Enemy centre
        if target in self.our_centres: score += 5 # Defending our centre

        if unit_location in self.our_centres: # Leaving it empty
            if self.opp_counts[unit_location] > 0: score -= 10 # Opps around

        # Check if we can win the fight
        score += (self.support_counts[target] - self.opp_counts[target] - 1) * 6

        # Check the proximity to other supply centres
        proximity = 0
        for sc in self.unoccupied_or_enemy_centres:
            d = self.distance_cache[target].get(sc, 99)

            if d < 99:
                proximity += max(0, 5 - d)

        score += proximity

        # Compensate for seasons
        score *= season()

    elif action_type == ActionType.HOLD:
        if unit_location in self.our_centres:  # Defending our centre
            if self.opp_counts[unit_location] == 0: score = -10 # No opps around
            else: score = 20

            score *= season()
"""

# Beam Search!
# Beam search finally made generating movesets viable
"""
    Beam search to generate candidate order sets (~10s vs ~50ms)
    
    Order Set Quality (subjective):
    Restricting supports to only be added with matching move (poor)
    Summing action weights (mediocre)
    Initial sorting of actions by score (better)
    Adding partial_order_set evaluation and checking validity only at end (good)
    Adding random orders at each pruning iteration (also good)
    Replacing partial_order_set eval with synergy calculation (great)
    Sorting by unit with greatest eval (good)
    Sorting by unit with greatest sum of evals (good)
    Sorting by unit with largest candidate eval differential (great)
    Evaluating entire only at the end and sorting (best)

    Synergy calculation:
    If new action matches support (good)
    If new support matches action (better), can't believe I forgot to check this for so long
    Convoy matching move (best)
"""

# Second beam search
"""
    I was using a second beam search to generate full likely order sets for enemies:

    Order Set Quality (subjective):
    Beam width 100 (bad)
    Beam width 300 (bad and slow)
    Beam width 20 (worse)
    Sorting by most valuable action (bad) 
    Prioritising our power by * 1.5 (ok)
    Prioritising our power by * 10 (ok)
    Removing this second beam search (best!)
"""

# Better position eval
# Added properties with weighting"next_turn_centres": 20, "threat_to_our_centres": 5, "support_for_our_units": 10
# Testing showed negligible improvements, but not worse so I kept it.

# My Game Engine:
# I found the built in game engine extremely slow to copy, (~2ms), and this meant I simply could
# not simulate many moves. I am very unsure as to what we were supposed to do, so I just made my
# own simple version to resolve basic orders and provide fast copying.

"""
    Timings:
    Copying my game class: 0.03ms.
    Copying built in game class: 1.47ms.
    Copying built in game class with deep copy: 2.64ms.

    Resolving orders in my game class: 0.16ms (still rather slow unfortunately)
    Resolving built in game class orders: 0.12ms (slightly faster, but not worth extra copy time)
"""

# Monto Carlo Tree Search
"""
    Results at the top of the page.
"""