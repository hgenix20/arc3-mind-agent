# Level notes: what the agent is missing, level by level

One entry per level the explorer stalls on, written from watching it play
(`scripts/watch_explore.py`). Each names the general ability the level needs, never
the moves. Abilities that recur across levels get built first.

## tu93 (maze: avatar walks to an exit tile)

- Levels 0 to 3 are won by coverage, at 10 to 25 times the human moves.
- Missing: knowing what to head for. The avatar wears the exit's color, so color gaps
  read zero. Partly addressed by reach features (e34bbe9): level 2 863 to 622 moves.
- Missing: a state that ignores what the agent doesn't control. Patrolling enemies on
  levels 3 and 4 make almost every frame a new state, so the graph barely reuses.
- Missing: lives. A move budget ends each life (59 game overs on level 3); every life
  walks back from the start.

- Level 4 (watched 2026-10-28, plain explorer: 3,294 moves, 311 states, 122 game
  overs): the exit sits at the maze's central junction, and purple patrollers keep
  crossing it. Touching one ends the life. The guided explorer did worse here (6,400+
  moves) because it heads straight for the exit and meets the patrollers.
- Missing: learning what kills (a color whose contact ends the life) and treating its
  likely next cells as off limits; patrol paths repeat, so they can be predicted.

## sb26 (fill slots to copy a pattern)

- Level 0 never won in 3,000 moves (505 states, 1,307 no-ops, 15 game overs).
- Screen: a framed pattern of four colors at the top, a row of four empty slots in the
  middle, a palette of four colors at the bottom, a timer bar.
- Play: click a palette color, then place it into a slot. By move 2,750 it had placed
  one color, in the wrong slot.
- Missing: noticing that two regions share a layout and that one should come to look
  like the other (a match goal), then scoring states by how many cells still differ.
- Missing: two-step actions (select, then place) treated as one unit when exploring.

## re86 (two crosses, two groups of markers)

- Level 0 never won in 3,000 moves (1,725 states, 29 game overs).
- Screen: a green cross and a red cross; two diamonds of four yellow markers, one
  group with teal centers and one with red centers; a move-budget bar.
- Play: arrows move the selected cross; action 5 switches which cross is selected.
- Missing: more than one controllable thing, with a switch between them (the mover
  detector assumes one).
- Missing: a cover goal, where each target gets a mover on it, and pairing movers with
  targets by a shared color.

## sk48 (an arm on a rail pulls blocks into a chain)

- Level 0 never won in 3,000 moves (392 states, 7 game overs).
- Screen: a slider on a rail with an extendable arm; blocks on the right; the bottom
  bar shows a chain in a set color order; a timer bar.
- Play: slide, extend the arm, grab a block and pull it back. At move 750 it had pulled
  one block.
- Missing: a match goal against a legend (the chain should read like the bar).
- Missing: multi-step manipulation (extend, grab, retract) treated as one unit.

## wa30 (carry boxes)

- Level 0 never won in 3,000 moves (796 states, 14 game overs).
- Screen: an avatar with a facing mark; three yellow-framed boxes; a red bar; a budget.
- Play: the avatar walks and faces; next to a box it can take hold of it (move 1,000).
- Missing: holding as part of the state (carrying a box changes what the moves do).
- Missing: a goal to bring things to a place (boxes to the bar), a cover goal again.

## Abilities that recur

1. A match goal: a region that should come to look like a reference (sb26, sk48).
2. A cover or deliver goal: things brought onto targets, paired by color (re86, wa30).
3. More than one mover, or a held object (re86, wa30, sk48).
4. A state that leaves out motion the agent doesn't cause (tu93).
5. Lives under a move budget (tu93, re86 and every game with a timer bar).

The old mind (`arcagent/mind/goals.py`) already proposes reach, clear, match and cover
goals from any screen. The explorer uses none of them yet. The next step is to let
those goal guesses steer the explorer's frontier choice on level 0, before any win
has been seen: guess goals from the screen, head toward each, drop the ones that hold
without a level-up.

## lp85, tn36, r11l (level 0 at human speed, level 1 never)

- lp85 level 1 (2,004 states in 3,000 moves): colored cells on loops; clicking a hub
  turns its loop. Level 1 has two loops that share cells. Missing: a model of each
  click as a fixed rearrangement of cells, so a goal can be searched for in
  imagination; the reachable arrangements are far too many to visit.
- tn36 level 1 (849 states): set switches on a program panel, then press the button to
  run it; the figure moves only when the program runs. Missing: actions whose effect
  shows up later (program, then run), and carrying level 0's goal over to a two-panel
  screen.
- r11l level 1 (724 states, 52 game overs): a body sits where its legs are; click a leg
  to select it, click a spot to move it. Level 1 has two bodies, each with a circle of
  its own color. Missing: select-then-place as one unit, a goal pairing each body with
  the circle of its color, and a model of how the body follows its legs.

## ls20 (walk to a door whose glyph matches the key)

- Level 0 won at move 504 (human 22); level 1 not won in 3,500 more moves (172 states,
  32 game overs).
- Screen: an avatar on a green floor; a framed door with a glyph; a framed box in the
  corner showing the key's glyph; a blue cross on the floor; on level 1, small framed
  tiles; a move bar at the bottom.
- Play: stepping on the cross turns the key's glyph. The door opens only when the key's
  glyph matches the door's. The small tiles look like move refills.
- Missing: a match goal between two framed boxes (key display and door), which comes
  before the reach goal (the door); the order of goals matters. Also resources: tiles
  that refill the move bar are worth a detour when the bar runs low.
- The fluid mind won level 0 in 33 to 59 moves. Its goal book proposes match goals on
  twin framed regions, which is this pattern.

## m0r0 (two dots move as mirror images and must meet)

- Level 0 won at move 366 (human 30); level 1 not won in 3,600 more moves.
- Level 1 spent 2,974 moves on clicks that never do anything: each new state brings
  dozens of untried clicks, and once its arrow moves were used up the explorer tried
  those rather than walk to a state with arrow moves left. Fixed in e2a542d (walk to a
  live untried move first): 99 no-ops, 364 states; level 1 still not won.
- Missing: the mirror rule as a model (one action moves both dots, one of them
  reversed), so the meeting point can be planned; checkered cells look like hazards.

## sp80 (place bars so a falling stream lands in the cups)

- Level 0 won at move 425 (human 39); level 1 not won (1,663 states, 77 game overs).
- Play: click a bar to select it, arrows move it, action 5 releases the stream. Level 1
  turns the board upside down and adds bars.
- Missing: a simulation of the stream's path given the bars (the effect shows only on
  release), and the goal "stream ends in a cup", carried over from level 0 though the
  board is flipped.

## su15 (move pieces to the circle)

- Level 0 won at move 453 (human 22): clicks move a white square along a dotted path
  into the circle.
- Level 1 (2,158 states, 83 game overs): several pink and white pieces, the circle, and
  a row in the top bar that reads like an order of colors.
- Missing: a model of click-to-move (a piece steps toward the click), several pieces,
  and possibly an order given by the bar.

## cn04 (fit two pegged pieces together)

- Level 0 never won by the explorer (1,825 states in 4,000 moves, 52 game overs); the
  fluid mind won it once in 832 moves.
- Screen: two outlined pieces, each with a pair of cyan pegs; a move bar.
- Play: click a piece to select it (it turns purple), arrows move it, action 5 rotates
  it.
- Missing: a fit goal (bring the pegs of one piece against the other piece, facing
  each other), and rotation as part of a piece's model, so a fit can be planned.

## dc22 (slide pieces to open a path)

- Level 0 won at move 1,765 after e2a542d (3,500 to 4,400 before; human 59); level 1
  not won (204 states, 1,466 no-ops still).
- Screen: a board of sliding pieces on the left, two hat shapes on the right (one
  outlined, one filled), a move bar.
- Missing: what the hats mean (they look like a legend of which piece goes where);
  still many no-ops, so some kind of move keeps being tried where it can't work.

## Tried and dropped: best-first guidance (2026-10-28)

The guided explorer acts on its credit only when the current state has no untried
action, which never happens in click games. Making it best-first everywhere was
worse: tu93 level 3 went from 70 to 217 game overs, r11l level 1 died 82 times.
The credit from one quick win is mostly noise (lp85 credited six unrelated color gaps
at equal weight from an 8-state level), and following it walks back and forth until
the budget runs out. Credit needs more evidence than one win, or a goal guess that
can be tested in imagination before it's followed.

## What the notes point to

Almost every stalled level needs a model of what actions do: which thing moves, how
far, what blocks it, what kills, what a click rearranges, what happens later. With a
model the agent can try ideas in imagination and spend real moves only on testing a
guess. The explorer has no model; it learns a graph of screens it has seen, which
can't generalize to a screen it hasn't.
