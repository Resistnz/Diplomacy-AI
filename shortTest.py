from test import *
import cProfile

AI_AGENT = "FRANCE"
OPPONENT_POOL = [StaticAgent] # [RandomAgent, AttitudeAgent, AttitudeAgent, GreedyAgent, GreedyAgent]

if __name__ == "__main__":
    agents = {}

    print(f"Running 1 simulation with the AI as {AI_AGENT}")

    # Assign each type of agent
    for p in ALL_POWERS:
        if p == AI_AGENT:
            agents[p] = StudentAgent()
        else:
            opponentAgent = random.choice(OPPONENT_POOL)
            agents[p] = opponentAgent()

    #print("\nProfiling the simulation...")
    #profiler = cProfile.Profile()
    #profiler.enable()
    results, _ = run_one_game(agents)
    #profiler.disable()
    #profiler.print_stats(sort='time')

    scores, wins = scoring(results)

    print("Game Finished!\n")
    for power in results:
        print(f"{power} ended with {results[power]} supply centres - {wins[power]}" + (' (AI Agent)' if power == AI_AGENT else ''))
'''
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