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

    print("\nProfiling the simulation...")
    #profiler = cProfile.Profile()
    #profiler.enable()
    results, _ = run_one_game(agents)
    #profiler.disable()
    #profiler.print_stats(sort='time')

    scores, wins = scoring(results)

    print("Game Finished!\n")
    for power in results:
        print(f"{power} ended with {results[power]} supply centres - {wins[power]}" + (' (AI Agent)' if power == AI_AGENT else ''))

