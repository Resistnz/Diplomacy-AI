# Github link

""" Final Agent Workings:

The final agent works by generating candidate actions per unit with a local evaluation score,
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

It is highly possible these results are just the result of variance, however it does make sense that larger rollout depths
decreases the number of iterations that can be performed.

"""

