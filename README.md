# CITS3011 Intelligent Agent Project

This was a project that I completed for university to create a bot to play the game Diplomacy. The report on the project can be found as `report_24214277.pdf`. Unfortunately, due to the complexity of the game and lack of relevance to any of the actual material we learned in the course, all of the cool AI features I had researched such as hierarchical task networks and goal based planning were infeasible to implement within the project timeline, but perhaps I will one day revisit them. 

Below is the task description:

# Task Description

This project is to be completed individually.

This project is marked out of a total of 40 marks and is worth 40% of your unit mark.

This project is due at 11:59 pm on October 1st (Wednesday), 2025. The unit coordinator reserves the right to extend this deadline in rare but necessary cases. 

You are strongly encouraged to submit earlier to avoid any urgent submission issues, and you may receive a late penalty if you are unable to submit by the deadline.


# Description
In this project you are tasked to research, design, develop, and evaluate an agent for playing the game Diplomacy.

You will be assessed on the performance of the agent you develop and on your written report of the techniques you investigated and developed.

As always, the purpose of this project is to simulate you encountering this as a novel problem. Do not look up or use any existing code, solutions, or computational strategies for Diplomacy. Reusing any existing code rather than working on the problem yourself is plagiarism and misconduct.

# Introduction to Diplomacy

Diplomacy is a strategic board game with seven players competing and cooperating to capture supply centres on a map of Europe. This game is very challenging for AI due to the large action space and state space. You need to be creative.

Please read carefully the following information that is necessary for your project.

- Game Rules: https://en.wikibooks.org/wiki/Diplomacy/Rules

The following is the game engine used in this project and its documentation. You will need to read the documentation to find the correct way to retrieve the state information from the engine and interact with the engine: 

- Game Engine: https://github.com/diplomacy/diplomacy
- Documentation: https://diplomacy.readthedocs.io/en/stable/

The following are statistics and databases of the game. You do not need to rely on them to complete the project, but it can help you understand the game:

- Game Statistics: https://vdiplomacy.net/variants.php?variantID=1
- Game Database: https://world-diplomacy-database.com/php/commun/index.php

You can also try playing the game yourself on the online platform below:

- Online Playing: https://webdiplomacy.net/


# Getting Started

Download the attached `the_diplomacy.zip`. You should build your agent in `agent_studentnumber.py`, replacing with your student number.
That is, you should subclass the Agent class (defined in `agent_baselines.py`) and override the methods and/or implement new methods in your agent.

1. Create a virtual environment using `conda` or `venv` (optional but highly recommended).

2. Install the game engine and other required packages:
```
pip3 install diplomacy tqdm networkx numpy timeout-decorator
```

Or install the packages using the provided `requirements.txt`:
```
pip3 install -r requirements.txt
```

3. Running a test will by default run a large number of games and report the performance of the agent:
```
python3 test.py
```

Your objective is to maximize your expected performance when your agent is dropped into complex (maybe unknown) scenarios.

You can reuse or adapt the testing code `test.py` during the development of your agent. After submission, your agent will be tested under multiple scenarios against multiple baseline agents.


# Game Setup
- Stanard Map will be used.
- No Press mode will be used, i.e., no messages among agents.
- Game ends in the year 1920, if there is no winner before 1920.


# Agent Rules

- Your agent must implement the provided Python interface to take part in the game.
- Agents are time-limited and all actions they take must be completed within 1 second.
- Agents are memory-limited and a maximum of 512MB memory can be used.
- Agents are not allowed to save files.
- Agents must not attempt to circumvent or hack the simulation.
- Agents have no connection to the internet and cannot use API calls, e.g., to LLMs.
- Agents have no access to GPUs.

If we believe your agent attempts to violate any of these rules or otherwise undermine the assessment, it may be disqualified and you may receive no mark.


# Report
You are required to write a report detailing the techniques you researched/investigated, your reasoning behind your choice of design and technique, and your assessment of the effectiveness of your agent.

Your report should be no more than three A4 pages.

Your report should be submitted as a PDF.
- If it is not submitted as a PDF, it may receive no mark.
- If it is over length, it may receive no mark, or be truncated and only partly marked.
- If it is illegibly formatted (tiny font, for example) or otherwise unintelligible, it may receive no mark.


# Baseline Agents

Your agent will play the game against baseline agents developed by the teaching staff. There are five different baseline agents.

- **Static Agent**: This is an agent that always takes the default actions, i.e., hold.
- **Random Agent**: This is an agent that always takes random actions.
- **Attitude Agent**: This is an agent that takes random actions, but has attitudes towards other powers, including being friendly, neutral, or hostile. The attitude depends on other players' actions and can change during the game. A friendly agent will never attack you, a hostile one will never support you, and a neutral one can do anything.
- **Greedy Agent**: This is an agent that always takes greedy actions, without long-term planning. Each unit controlled by the agent will move towards and attack the closest supply centre, or support other units if having the same target.
- **Hidden Agent**: This is an unknown agent.


# Scenarios

- **Scenario 1**: Your agent will control a random power. Other powers are all controlled by copies of **Static Agent**.
- **Scenario 2**: Your agent will control a random power. Other powers are controlled by copies of agents randomly chosen from **Random Agent**, **Attitude Agent**, and **Greedy Agent**. The **Random Agent** is less likely to appear than the other two.
- **Scenario 3**: Your agent will control a random power. Other powers are controlled by copies of agents randomly chosen from **Random Agent**, **Attitude Agent**, **Greedy Agent**, and **Hidden Agent**. There will be exactly one **Hidden Agent** in each game. The **Hidden Agent** is a reasonably strong agent with a ~50% win rate in **Scenario 2**.
- **Scenario 4**: All the student agents will be put together to play a multi-round tournament. 


# Marking Rubrics

The marking of the agent and the report will be independent of each other. 
The marking of the agent will focus on the performance. 
The marking of the report will focus on the knowledge, thinking, reasoning, and presentation.

**Agent Rubrics (18 pts)[^1]**:
* **Scenario 1 (6 pts)**  
- The agent achieves >2% win rate, or captures >7 supply centres on average. (2 pts)   
- The agent achieves >20% win rate, or captures >12 supply centres on average. (4 pts) 
- The agent achieves >90% win rate, or captures >16 supply centres on average. (6 pts) 
* **Scenario 2 (6 pts)**  
- The agent achieves >2% win rates, or captures >7 supply centres on average. (2 pts)   
- The agent achieves >25% win rates, or captures >10 supply centres on average. (4 pts) 
- The agent achieves >45% win rates, or captures >13 supply centres on average. (6 pts) 
* **Scenario 3 (6 pts)**  
- The agent achieves >2% win rate, or <20% defeat rate, or captures >7 supply centres on average. (2 pts)  
- The agent achieves >20% win rate, or captures >9 supply centres on average. (4 pts)
- The agent achieves >30% win rate, or captures >11 supply centres on average. (6 pts)
* **Scenario 3 (Bonus)[^2]**  
- The agent ranks top 3 among all student agents in **Scenario 3**. (3 pts)
* **Scenario 4 (Bonus)[^2]**  
- The agent ranks top 7 among all student agents in **Scenario 4**. (3 pts)

[^1]: The baseline agents have been provided to enable you to assess your code. It is allowed to look at the codes of the baseline agents for better understanding. However, if we have reason to believe that you have plagiarized from these agents (for example, directly copying them to get points in some scenarios), you may receive no mark. Your agent must be your own original work, as always. 

[^2]: The total points after receiving bonus will not exceed 40 points.


**Report Rubrics (22 pts)**:

* **Basic Techniques (14 pts)**
- Considers and describes at least two basic techniques [^4]. Basic techniques can be from those taught in the lectures, or other existing techniques [^8]. (4 pts, 2 for each)
- Discusses the merits/motivations of these basic techniques and justifies the choices. (6 pts, 3 for each)
- Evaluates the effectiveness of these basic techniques and reports experimental results [^7]. (4 pts, 2 for each)

[^4]: You need to implement the basic techniques in the submitted code, and refer to the implementation in your report. At least two basic techniques should be implemented in the code, even if you do not use all of them in the your final version of the agent.</em>
[^8]: If you use other existing techniques, references must be provided in the report. Besides, you must implement the techniques yourself.

* **New Techniques (8 pts)**
- Creates and describes at least two new techniques [^5] [^6] designed by yourself for improvements. (2 pts, 1 for each)
- Discusses the merits/motivations of these new techniques and justifies the designs. (4 pts, 2 for each)
- Evaluates the effectiveness of these new techniques and reports experimental results [^7]. (2 pts, 1 for each)

[^5]: The new techniques do not necessarily need to be very substantial. For example, a reasonable new heuristic function, a variant of the search process, or a modification of a basic technique, can be considered as a new technique if well-motivated and justified. </em> 

[^6]: You need to implement the new techniques in the submitted code, and refer to the implementation in your report. At least two new techniques should be implemented in the code, even if you do not use all of them in the your final version of the agent.</em>

[^7]: Experimental results can be either positive or negative. The marking will focus on whether the results are presented and analysed meaningfully, not the exact numbers.


# Submission

You should submit the three following files to LMS:
- `agent_studentnumber.py`: Your agent. (Maximum 100KB)
- `report_studentnumber.pdf`: Your report as a PDF. (Maximum 3 pages)
- `test_studentnumber.py`: Your experiments. (Maximum 100KB)

In above, "studentnumber" should be replaced with your student number.


# Allowed Packages:

The agent is allowed to use built-in packages for Python 3 and the following external packages. Use the provided `requirements.txt` to install the consistent version of the following packages:
- diplomacy
- tqdm
- random
- networkx
- numpy
- scipy
- scikit-learn
- simpleai: https://pypi.org/project/simpleai/

The implementation of the textbook is also allowed to use as reference for your coding: https://github.com/aimacode/aima-python.

# LLM/GenAI Policy:
- You are allowed to use LLM for brainstorming and designing.
- You are NOT allowed to use LLM for report writing. 
- If LLMs are used, you must submit an extra PDF document, recording all your prompts and the responses of LLMs, named as `llm_usage_studentnumber.pdf`.
- The agent itself cannot use any LLM.
