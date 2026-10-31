### 4.7 Hypotheses, forks, and planning

**Built modules**

- `arcagent/mind/goals.py` (built, changes). Goal guesses (reach, clear, count, cover, match, sequence) and the ranking formula. Changes: the ranking becomes path probability (below); the goal kinds stay.
- `arcagent/mind/search.py` (built, changes): see 4.6c.
- `arcagent/mind/fluid.py` (built, changes). The seven-step decision ladder. Changes: the ladder is replaced by the planning loop (3.2), with the same logging of branch and reason.
- `arcagent/mind/tester.py` (built, changes). Fact tests become curiosity checks (4.11).

**Not yet built**

- Hypothesis list ranked by path probability.
- Expected-value choice (3.4).
- Plan tree with forks, and the return route (3.6).
- Horizon rule (below).

**M4.7a Hypothesis list**

- Purpose: the ways to complete the level task, best first.
- Inputs: the rule store, the scene, the simulator (4.8).
- Outputs: a list of hypothesis records, each with its path probability and its path.
- Logic:
  1. Build goal hypotheses from the goal kinds that match the scene (goals.py guesses).
  2. For each goal, simulate the best path up to the horizon (4.8).
  3. Set the hypothesis's path probability (M4.4c).
  4. Sort by path probability, best first. Ties are rare by construction (four decimal places).
  5. A hypothesis with path probability below the floor (0.3, PLACEHOLDER) stays on the list but is not expanded.
  6. The list is rebuilt on each replan and not kept across replans, except its fork records.
- Kameron, 2026-10-31 (Q5 on the earlier spec): path probability alone ranks the list.

**M4.7b Expected-value choice**

- Purpose: choose which hypothesis to act on, when the best probability is not the best value.
- Logic: per hypothesis, from the simulated path:
  1. Expected moves = the expected number of moves to reach the goal along the path (the simulator returns the moves to each state, weighted by probability).
  2. Reward = the value of the goals it reaches. Goal kinds carry value from the level's score: completing the level has the largest value. Collectibles carry a value estimated from observed gains.
  3. Adverse cost = the probability of harm on the path times the cost of restarting from the last fork (moves to re-reach it).
  4. Expected value = reward times the path probability, minus the expected moves cost, minus the adverse cost.
  5. Expected moves cost uses the level's typical moves per completion (horizon rule below). Kameron, 2026-10-31 (Q1): "It should change to an expected value scoring because it will allow for less steps to identify the best plan."
  6. Choose the highest expected value. Ties by four-decimal path probability, then by id.
- Score link: the reward uses the scoring formula, `min(human/agent, 1)^2`. The human count is unknown at runtime, so the reward uses the typical moves per completion as the reference.

**Horizon rule**

- Kameron, 2026-10-31 (Q3): "it should follow each level's typical moves per completion."
- Logic:
  1. Horizon for a level = the mean number of moves the agent took on its completed attempts of that level, rounded up.
  2. Before the first completion of a level, horizon = PLACEHOLDER cap (initially 200), plus the time gate (4.10).
  3. A plan segment (3.5) is never longer than the horizon.
  4. The horizon is recorded per level in the long-term store.

**M4.7c Fork and plan tree**

- Kameron, 2026-10-31 (Q2): "a fork would be considered a plan or path that could go two ways toward the same goal. The agent would need to take the shortest path or least dangerous path, whichever is more favorable for the score."
- Logic:
  1. A fork is recorded when two or more paths to the same goal exist from the same state (the simulator finds more than one path).
  2. Each fork holds its state key, its paths ordered by expected value, and each path's status (untried, in progress, failed, achieved).
  3. Moves that led to the fork are kept on the fork record (Kameron, 2026-10-31 (Q3): "it should keep the moves that got there"). They are the return route.
  4. The return route to a fork is planned on the known moves (breadth-first search, 4.6a) and executed with the same check as any move.
  5. Replan order (3.6): newest fork with an untried path first, then older forks, then the root.
  6. Kameron, 2026-10-31 (Q4): "replan step" is the explorer's backtracking. So the explorer's backtracking is the rule for the return route, not a separate method.
  7. A failed path is marked failed for this goal instance (level instance). It is not retried in the same instance.

**Tests**

- Two-path fixture: the choice follows the expected value, not the probability alone.
- Fork return: after a dead end, the agent returns to the fork over known moves and takes the second path.
- A path below the floor is listed and not expanded.

### 4.8 Simulator (game model)

**Built modules**

- `arcagent/mind/predictor.py` (built, keep). `predict` is read-only (Kameron, 2026-10-30, answering Q1 on the earlier spec: "yes").
- `arcagent/mind/game_model.py` (built, off path, reuse). `play_out` simulates a plan step by step with `predict` and stops at a goal or at an unknown step. It is the LLM path's simulator. Reused as the base for 4.8's rollout.
- `arcagent/mind/kinematics.py` (built, changes): reachable positions and distances (4.6).

**Not yet built**

- Learned rules for uncontrolled movers (enemies and similar), for collectibles, and for adverse events.
- Probability on every predicted effect.
- Continuous-movement check.
- Rollout that returns probability, expected moves, and harm.

**M4.8a Rollout**

- Purpose: predict a path move by move, with probabilities, and stop at a goal, a harm, or the horizon.
- Inputs: the current scene and state (health, counters), a candidate move list, the rule store, the horizon.
- Outputs: the predicted states, each step's probability, the reached goal (if any), the harm events on the path, the stop reason.
- Logic:
  1. Start from the current scene and state.
  2. For each move i up to the horizon:
     1. Find the rules that predict this move from the current state (4.3).
     2. Apply the rule with the highest probability to each thing it covers. Things with no rule are predicted to stay still, and this is logged as an assumption (rule 4).
     3. Record the step's probability (M4.4c).
     4. Apply harm: if the state's health goes to zero, stop with a death. If an adverse event is predicted, record it with its probability.
     5. If a goal holds in the predicted scene, stop with goal reached.
  3. Return the outputs.
- Logs: every assumption and every harm is logged with the step number.

**M4.8b Learned movers (enemies and similar)**

- Purpose: predict things that move on their own.
- Kameron, 2026-10-30 (Q4): "I have been using that as an example and you are supposed to draw from that concept and not explicitly. This way if it has enemies then it can avoid them but if it has collections like pacman collects fruit then it can go for those before it completes the level."
- Logic:
  1. A thing that moves when the agent does nothing is a mover of its own. Its motion is learned as a rule with the thing's kind as subject.
  2. A mover's motion is predicted by its learned rule. Until a rule exists, the mover is predicted still, and each such prediction is logged as an assumption.
  3. A mover that the agent meets and that reduces health on contact is given the harm role (4.10).
  4. A thing that disappears on contact and changes a counter is given the collectible role. Its value is estimated from observed gains.
  5. The role is inferred from observed effects. No role is named after a game.
- Kameron, 2026-10-31, answering Q4 on the earlier spec (whether enemies exist in the games): "I have no idea." So the first build does not assume either exists.

**M4.8c Continuous movement check**

- Purpose: find out whether a game moves things by amounts that are not the learned step.
- Logic:
  1. For each moved thing, compare its displacement with the learned step size of its moves.
  2. IF displacements are not multiples of the step, for a run of moves THEN flag continuous movement for this game. The flag is logged and the theta-star question is reopened (4.6e).
- Kameron, 2026-10-30 (Q2): "I have no idea because I don't see them, only the agent does." So this check is what answers the question.

**M4.8d Horizon and assumptions in the log**

- Every rollout logs its horizon, the number of assumed-still predictions, the number of harm events predicted, and the number of predictions that later missed.

Tests:
- Rollout on a fixture with a known mover: the predicted scene matches after each step.
- A mover with no rule is predicted still, with an assumption logged.
- A predicted death stops the rollout with the death as the stop reason.

### 4.9 Planner and performance (model-first)

This is the change Kameron asked for. The planner is a language model with access to
skills and tools. It writes the next short segment of moves, and the executor checks each
move against the simulator.

**Built modules**

- `arcagent/mind/contract.py` (built, off path, reuse). Parses the model's reply into goals and plan steps, with checks for legal action ids. Reused for the planner's reply format.
- `arcagent/mind/sandbox.py` (built, off path, keep). Runs a generated program with time and memory limits. Kameron, 2026-10-30 (Q4 on the earlier spec): the code path "should replace the action part of it." So the sandbox is kept as the runner for planner programs.
- `arcagent/mind/turn.py`, `vchat.py` (built, off path). Prompt building and the vision chat. Reuse: the briefing builder in 4.9 takes its formatting and length limits from turn.py. vchat is kept only if the planner is given the screen image (decision in section 7).
- `arcagent/mind/assign.py` (built, off path). Hungarian assignment, which the planner does not need as written here. Kept as reference.
- `arcagent/mind/executive.py` (built, off path, not reviewed for this spec). Its role is not assessed here. Review before any reuse.
- `arcagent/mind/llm_drive.py`, `drive.py`, `toolbox.py`, `tool_fix.py` (built, off path). The DRIVE path. Its lessons (repair turns, toolbox fixes) are reference for the skill library. Not reused as code in this spec until reviewed.

**Not yet built**

- Briefing (the lists the planner reads).
- Tool API (what the planner may call).
- Question library.
- Skill library with deterministic strategy skills.
- Executor check (each act checked against prediction).
- Token and time gates for the planner (4.10).

**M4.9a Briefing**

- Purpose: give the planner the state in fixed lists, so it does not reread the game.
- Lists, in this order (each list has a fixed length limit, PLACEHOLDER per list, and is truncated from the least recent end):
  1. Current state list: each thing on screen, with its kind, its known rules, and its role (harm, collectible, blocker, mover, controlled).
  2. Action list: the valid action ids, and whether click is present (4.5a).
  3. Goal list: the level goal and the game goal, with status.
  4. Task list: the ordered tasks from the chosen hypothesis.
  5. Level rule list: rules for this level, by object, event, experience.
  6. Game rule list: rules for this game.
  7. Hypothesis list: top hypotheses with probability and expected value.
  8. Sequence list: the current segment and its status.
  9. Fork list: open forks and their untried paths.
  10. Flags: items marked check, with value and cost.
  11. Self record and focus record (4.5).
  12. Recent events: the last few move results with predicted and observed.
- Logic: the briefing is built from the stores (4.2) and the simulator (4.8). It is never built from the planner's own past reply, so the planner cannot repeat its own error as fact.

**M4.9b Tool API**

- Purpose: the only things the planner may call.
- Tools:
  - `simulate(moves)`: rollout from the current state (4.8a). Returns probability, expected moves, harm, stop reason.
  - `probability(rule_id)`: the probability and status of a rule.
  - `query(features)`: retrieval (4.3).
  - `route(targets)`: order of visits (4.6b).
  - `shortest(from, to)`: breadth-first search over known moves (4.6a).
  - `skill(name, args)`: a strategy skill (below).
  - `ask(question_id)`: a question from the question library (below).
  - `act(action)`: send one move. Only through the executor (M4.9d).
- Rule: the planner may simulate as often as it needs. The time gate (4.10) and the token gate bound the total.

**M4.9c Question library**

- Purpose: Kameron, 2026-10-30 ("You can propose questions to it so it decides quicker instead of telling it what to do"). The planner answers questions; the code does not tell it what to do.
- Questions (each returns a short structured answer from the stores or the simulator):
  - Q-BLOCK: what blocks the path to the goal, and is it harmful, a blocker, or a mover?
  - Q-REWARD: which things in view are goals or collectibles, and what is each one's value?
  - Q-HARM: what harm does the path carry, and at what step, with what probability?
  - Q-FORK: which open forks exist, and which untried path is best by expected value?
  - Q-CHECK: which flagged item gives the most information for the fewest moves?
  - Q-SAFE: can the agent survive the path to the next fork with the current health?
  - Q-UNKNOWN: which rule or thing has no rule, and how many times has it been seen?

**M4.9d Strategy skills (deterministic approaches)**

- Kameron, 2026-10-30 (Q1 on the sequence): part of a sequence "could be deterministic approaches like 'enemy blocks path, then kill it' or 'pitfall blocks path, then replan route around it'."
- A skill is a named approach with a trigger on kinds and events, and a fixed set of steps the executor can run. A skill never names a game.
- Skill list (first version):
  - S-ATTACK: trigger: a harm-role thing blocks the path. Steps: simulate the attack path; if the health after the attack stays above zero with margin, take the attack path; else go to S-AROUND.
  - S-AROUND: trigger: a harm-role or blocker thing blocks the path. Steps: find a route to the goal that avoids the thing's cells (breadth-first search, 4.6a), simulate it, take the best by expected value.
  - S-REPLAN-PIT: trigger: a hazard that causes a reset lies on the path. Steps: mark the hazard cells as harm, replan the route around them.
  - S-COLLECT: trigger: collectible things are in view and the goal is not closer. Steps: order collectibles by route (4.6b), simulate, add them to the path if the expected value rises.
  - S-WAIT: trigger: a mover blocks the path. Steps: wait the number of moves its learned rule predicts for it to clear; if no rule, S-AROUND.
  - S-EXPLORE: trigger: no hypothesis has a path. Steps: take the flagged check with the highest value per cost (4.11).
- Logic: the planner may pick a skill or write its own segment. A skill's steps still pass through the simulator and the executor. A skill that fails is marked failed for this level instance.

**M4.9e Executor**

- Purpose: send moves to the game, and check each one.
- Logic:
  1. Take the next move of the segment.
  2. Simulate it from the current state (one step).
  3. Send it to the game.
  4. Observe. Compare the observed scene with the predicted one (4.12a).
  5. IF they match THEN continue the segment.
  6. IF they miss and the miss matters (4.10 rule on mattering) THEN stop the segment and go to fork handling (3.6).
  7. IF they miss and it does not matter THEN log it, learn from it (4.12a), and continue.
- A move that is not in the segment is not sent. That is what keeps the planner from oscillating between two tiles: the segment is written once, checked, and executed in full before the next planner call (Kameron, 2026-10-31).

**M4.9f Planner call**

- Logic:
  1. A planner call is made when a segment is needed (replan, level start, segment done).
  2. The briefing (M4.9a) is built, and the planner is asked for one segment: up to horizon moves, or one skill call, plus one question if it needs more.
  3. The sandbox runs the reply's program with the tool API only. The sandbox stops at the time gate or the token gate (4.10).
  4. The segment is passed to the simulator (3.5). If the simulator finds it unsafe, the planner is asked once more with the failed step noted.
  5. A second failure drops the hypothesis (3.5).
  6. Each call's briefing, reply, and outcome are logged with the planner call id.

Tests:
- A fixture where the planner's segment passes the simulator: it is executed in full.
- A fixture where the planner's segment fails at step 3: the second call names step 3.
- S-ATTACK on a fixture with enough health: attack path taken; on a fixture without: S-AROUND taken.

### 4.10 Harm, inhibition, perseverance, and gates

**Built modules**

- `arcagent/mind/tester.py` (built, changes). `veto` gives reasons (game over, no effect, undoes progress). Changes: its reasons become adverse rules (4.10c).
- `arcagent/mind/fluid_modules.py` (built, changes). The Inhibition protocol. Changes: its real module becomes the rule store plus the network (4.10c).
- `arcagent/mind/budget.py` (built, keep). Moves left in a game with a counter. Used by the gates.
- `arcagent/mind/modules/inhibition.py` (built, off path, dropped in favour of the new inhibition). Kameron: "This won't be needed in this current agent." Dropped.
- `arcagent/mind/modules/` (other files) (built, off path). `base.py` (the protocols) is kept as a reference for the module interfaces.

**Not yet built**

- Harm definition module.
- Adverse rules (in the rule store).
- Inhibition network.
- Perseverance rule.
- Time gate per move, and LLM token gate.

**M4.10a Harm definition**

- Kameron, 2026-10-30 (Q3): "if the player has a health bar, then anything that reduces it. If no health bar, then anything that kills the player to restart from the beginning of the path or ends the level/game."
- Logic:
  1. IF a health bar is found (a counter that falls and never rises, found by the noise and counter logic, M1.4) THEN harm is any move after which the bar is lower.
  2. ELSE harm is any move after which the level instance ends by death, or the level or game ends.
  3. Health is tracked in the simulator's state. Where a health bar exists, the simulator carries its value. Where none exists, health is the implicit number of harms the path can take, which is 1.

**M4.10b Adverse rules and the network**

- Kameron, 2026-10-30 (Q2 on the earlier spec, build the network now or later): "There is no held out games so it must be built now. Inhibition rules are added as levels and games progress."
- Logic:
  1. Each adverse effect the agent observes is a rule (4.2) with a harm value (4.10a).
  2. The inhibition network is a small learned function, updated online. Its inputs are the canonical features of the situation and the action. Its outputs are a probability for each harm kind.
  3. Training data are the agent's own observed transitions, from every game the run has played so far. Nothing is tuned on a named game (constraint C1).
  4. The probability used for a move: IF the adverse rule for the move has at least 5 confirmations (PLACEHOLDER) THEN use the rule's probability (4.4) ELSE use the network's output.
  5. Rules are added as levels and games progress (Kameron's words). The network does not replace them, and it is retrained on the rule store's evidence.
  6. Network size: small enough to run in a move's time gate. PLACEHOLDER: a few hundred weights. Architecture chosen in the build, to be recorded in the log.

**M4.10c Perseverance**

- Kameron, 2026-10-30, with the 2026-10-31 clarifications: the simulator decides, and either reason can allow a medium-risk move.
- Logic (applies to medium-band adverse probability only, from 0.3 up to 0.8):
  1. Reason A: no other path to the goal exists without the move (the simulator finds none in the horizon). IF true THEN allow with reason logged.
  2. Reason B: the expected value of the path with the move is higher than without it (4.7b). IF true THEN allow with reason logged.
  3. Either reason is enough. Neither is required with the other.
  4. The simulator checks that the agent survives to the next fork or the goal. Kameron's fire example, 2026-10-31: walking through fire is allowed if the simulator shows the agent reaches the enemy with enough health to win the fight; avoiding the fire then leaves the agent with less health at the enemy.
  5. A move in the certain band (0.8 and up) is never allowed by perseverance (3.9).
  6. A move in the low band (below 0.3) is allowed without a reason.
  7. Each allowed medium-risk move is a trial. Its outcome updates the probability (4.4). The count of trials per rule per level instance is capped (PLACEHOLDER: 3).
- Kameron, 2026-10-31 (Q5 on the earlier spec): "Its using the simulator before it makes a real move, so it shouldn't be a problem." So override trials do not need a separate budget charge, because the simulator checks them first.

**M4.10d Gates**

- Kameron, 2026-10-30 (Q1 on the earlier spec): "The time game should be per move so the agent doesn't spend all its time trying to figure out one move."
- Kameron, 2026-10-30 (Q2): "It should count the LLM tokens only. This will prevent the LLM from spending too many tokens to overbuild on one move."
- Logic:
  1. Time gate: the wall-clock time for one move, from observation to sending the action, is capped at T_MOVE (PLACEHOLDER: set from the measured median move time, times a factor). If the cap is reached, the move is the fallback in 4.10e.
  2. Token gate: the LLM tokens spent on one move (planner and any other model call) are capped at N_TOKEN (PLACEHOLDER). If the cap is reached, the planner returns what it has, and the move is the fallback.
  3. Run gate: each game gets a share of the 9-hour run, equal to an even share of the time left over the games still to play (existing `fair_share` rule in `submit.py`).
  4. Every trip writes a log event with the gate name and the values.
- Fallback (4.10e): the move with the highest simulated expected value among the moves the simulator allows, or, if the simulator has none, the explorer's least-visited move (the explorer is kept as the fallback, as-built section 4).

Tests:
- The fire example as a fixture: the planner chooses the fire path when the simulator shows the enemy is reached with enough health.
- The certain band is never allowed, even for an expected value gain.
- A move over the time gate takes the fallback and logs the trip.

### 4.11 Curiosity

**Built modules**

- `arcagent/mind/tester.py` (built, changes): `choose_test` and its use of unknown pairs. Changes: tests become checks (below).
- `arcagent/mind/modules/voi.py` (built, changes): disagreement measure, used as the information value of a check.
- `arcagent/mind/search.py` (built): the frontier of unknown pairs is reported with the search result.

**Not yet built**

- Flag store and flag records.
- Agent-made labels.
- Checks as moves chosen by the planner.

**M4.11a Flags**

- Kameron, 2026-10-30 (Q1 on the earlier spec): "I won't tell the agent anything, it will have to derive it on its own so it should allow designer labels that it can create."
- Logic:
  1. A flag is set to check when a thing has a new look (kind not seen before) or when a pair (action, thing) has no rule (a frontier pair).
  2. The agent may create its own labels for things. A label is a short name the agent makes for a kind or a role (for example, a name for a thing that stopped it). Labels are stored with the rule that gave rise to them, and the planner reads them in the briefing.
  3. A flag keeps its source and its outcome after the check.

**M4.11b Checks**

- Logic:
  1. A check is a move that touches or clicks a flagged thing.
  2. The check's value = the number of unknown pairs it resolves (Voi, 4.11c) plus the value of the goal the thing is part of, if any.
  3. The check's cost = the moves to reach the thing (breadth-first search) plus one.
  4. The simulator projects the assumed outcome. A check whose assumed outcome reaches a goal in a few moves is marked.
  5. The planner chooses a check or a plan step, with the numbers in its briefing. The code does not force the choice (Kameron, 2026-10-30: "I will let the LLM decide on the checks on its own.").

**M4.11c Information value**

- Logic: value of a check = how far the check's outcome would change the probability of the hypotheses that depend on it. Built as the disagreement measure in `voi.py`. PLACEHOLDER: the weight given to goal value.

**Social check**

- Dropped. Kameron, 2026-10-30 (Q2 on the earlier spec): "No social check. There are no other players."

Tests:
- A new look sets a flag to check with source new look.
- A check on a thing that changes a rule records the outcome and clears the flag.
- The planner briefing lists flags with value and cost.

### 4.12 Learning and consolidation

**Built modules**

- `arcagent/mind/goals.py` (built, keep): `learn_win` (credit for the gaps the winning move closed) and `generalize_forms` (carry forms to colours).
- `arcagent/mind/facts.py` (built, changes): `learn` writes evidence from each transition.
- `arcagent/mind/fluid_sleep.py` (built, changes): level bookkeeping. Changes: becomes the level consolidation below. Its time cap stays.
- `arcagent/mind/sleep.py` (built, off path, dropped). Kameron, 2026-10-30: "This seems like it would only be useful between games to consolidate, dedupe, and add learnings but I would not keep this because it overcomplicates things currently." Dropped. Its between-game job is taken by 4.12c.
- Transfer (`modules/transfer.py`, `CarryTransfer`) (built, keep): carries working memory across level instances.

**Not yet built**

- Long-term consolidation at game end.
- Promotion of rules to long-term scope.
- First-contact evaluation (below).

**M4.12a Transition learning (per move)**

- Logic: as the learning diagram (3.7). Every move is learned from. Learning happens after the check, never before the planner's next call.
- Each update is logged with the rule id.

**M4.12b Level consolidation (at level complete)**

- Logic:
  1. For each rule with scope working that meets the promotion test (4.4b) and whose conditions are general (they name kinds and roles, not this level's colours), set scope long.
  2. Write the completed sequence record for the level (status completed).
  3. Store the level's typical moves (the horizon, 4.7).
  4. Clear the level state. Keep working memory.
  5. Start the next level instance.

**M4.12c Game consolidation (at game end)**

- Logic:
  1. Write every rule with scope long, and every completed sequence, to the long-term store.
  2. Rebuild the index for the long-term store (4.3).
  3. Write the game summary to the log.
  4. Clear working memory.
- Kameron, 2026-10-30 (Q2 on the earlier spec): "Long-term memory is allowed to update during a scored run." So steps 1 and 2 also run at each level completion, not only at game end. The game-end pass is the final write.

**M4.12d First-contact evaluation**

- Purpose: measure learning on games the agent has not used for tuning, without a held-out set.
- Logic:
  1. Games are played in a fixed order set by the run seed.
  2. Each game's first attempt is scored with a long-term store built from the games before it in the order only.
  3. The first-contact score is the mean of those first attempts.
- Status: a proposal from this spec, not a Kameron decision. Flagged in section 7.

Tests:
- Consolidation promotes a rule that meets the test and does not promote one that does not.
- A game end clears working memory and writes the long-term store.
- The first-contact order never includes the game being scored in its own store.

### 4.13 Runtime, logging, and evaluation

**Built modules**

- `arcagent/mind/runlog.py` (built, keep). JSONL run log with event names.
- `arcagent/agent/submit.py` (built, changes). Builds the arcade, selects games, builds one policy per game, writes the submission file. Changes: the new agent's configuration is one entry in the policy table.
- `arcagent/mind/config.py` (built, keep). Configurations and ablations. The ablation approach (null modules) is kept.
- `arcagent/agent/hybrid.py` (built, changes): see 4.5c.

**Not yet built**

- The move record fields below, complete.
- Calibration report.
- First-contact score report.
- Gate report.

**M4.13a Move record**

- Written for every move. Fields:
  - `game`, `level`, `level_instance`, `move_index`, `stage` (framing, objectives, alternatives, modeling, execution, evaluation, or a named sub-stage).
  - `self` and `focus` records (4.5).
  - `hypothesis_id`, `segment_id`, `fork_id`, `planner_call_id`.
  - `predicted`: the predicted scene's key, the step's probability, and any harm predicted.
  - `observed`: the observed scene's key, and whether it matched.
  - `rules_updated`: list of rule ids with their change (confirm, contradict, refine, promote, retire).
  - `gate`: time spent, LLM tokens spent, and any trip.
  - `branch` and `reason`.

**M4.13b Summaries**

- Level summary: moves, deaths, forks, replans, rules created, promoted, refined, retired, planner calls, tokens, time.
- Game summary: levels, scores, first-contact result, long-term store size, gate trips.

**M4.13c Reports**

- Calibration: predicted probability against observed rate, in bins of 0.1. Per rule kind.
- Moves against typical: moves per level against the level's typical count (4.7).
- Gate trips per game.
- First-contact score (4.12d).
- Inhibition hit rate: moves the network or rules vetoed, and how often the veto would have been an adverse event (from the simulator's projection).

Tests:
- Every move has a record with each field present.
- A calibration fixture where predictions are right half the time reports bin 0.5 at about 0.5.
