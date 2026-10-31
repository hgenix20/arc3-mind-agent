# Fluency Retrieval Design (Part 2 of the refactor spec)

Status: draft for review, 2026-10-30. Nothing is built from this document yet.
Source: Kameron's "Fluency Retrieval" section, pasted 2026-10-30. The source governs
where it differs from the current code. Part 1 (`2026-10-30-agent-memory-design.md`)
defines the rule records and the three stores that this part retrieves from.

## 1. Requirements (from the source)

- **F1 Speed and accuracy.** The agent retrieves the most accurate information the
  fastest, without sorting through every stored item before reaching the most probable
  answer.
- **F2 Index, not scan.** The source's example: a reader doesn't read the whole
  dictionary to find a word starting with Z. Retrieval should reach the candidates
  directly and stop there.
- **F3 Meaning-based keys.** A network of meanings and keys may improve accuracy. The
  source leaves the form open.

The source does not define "most probable answer," the size of the candidate set, or
what counts as a retrieval. Section 5 proposes those and marks them as decisions.

## 2. What the current code does

Retrieval happens in four places. Two are indexed and two scan.

| Retrieval | Current code | Access pattern | Status |
|---|---|---|---|
| Fact by (action, subject, context) | `FactMemory.lookup` (`facts.py`) reads the `facts` dict by exact key, then falls back in a fixed order: the subject's `any` fact, the kind's facts, a broken fact, a same-shape sibling, the action-wide fact. | Hash lookup, then a fixed chain. | Indexed. The order is fixed, not ranked by probability. |
| Sibling fact (same action, same shape) | `FactMemory._borrow` reads `_siblings[(label, context, shape)]` and picks the most hits, ties to the lowest key. | Index lookup, then a min over the candidates. | Indexed. Only exact shape match counts, so a thing with the same role and a different shape is not found. |
| Facts by behavior tag | `FactMemory.toc` maps tag to keys. `by_tag` returns them sorted. | Index lookup. | Indexed. Tags are the only semantic key. |
| Gaits for one subject | `kinematics.py` `gaits` sorts all facts and filters by subject on every call. | Full scan and sort. | Scan. |
| Movers, controlled things, clear colors, pushed sigs, restyle families | `goals.py` helpers (`_mover_sigs`, `controlled_sigs`, `_clear_colors`, `_pushed_sigs`, `restyle_families`) loop over every fact each time they run. | Full scan per call. | Scan. |
| Goal guesses | `GoalBook.guesses` rebuilds the candidate list from the current scene on every call, with reach guesses capped at `MAX_REACH_TARGETS`. | Rebuilt per call from the scene, not stored. | Bounded but not indexed. |

Three gaps follow from this.

1. **Scans run on the hot path.** `gaits` and the goal helpers read the whole fact table
   on every call. Their cost grows with the number of facts, and Part 1 will grow that
   number, because long-term rules will be loaded at game start. The cost has not been
   measured yet.
2. **Nothing ranks by probability.** `lookup` returns the first hit in a fixed order.
   Hit counts are used only to break ties between siblings. A rule with 40 agreeing
   observations and a rule with 5 are treated the same when both match the key.
3. **Nothing matches by meaning.** The only semantic key is the behavior tag, and the
   sibling index needs an exact shape. A thing that plays the same role as a known thing
   with a different shape gets no help from memory.

## 3. Proposed design

Retrieval goes through one contract. Every store implements it, and no caller reads a
store's table directly.

**Query.** A query carries the situation's features: action label, subject kind and
role, shape class, context, and effect tags. Features are the canonical keys from Part 1
(kind and role, not color index).

**Index.** An inverted index maps each feature to the rule ids that carry it. The
existing `toc` and `_siblings` are two instances of this, one keyed by tag and one by
(label, context, shape). The new index covers every feature the query can name, so a
query reads the candidate postings and nothing else.

**Ranking.** Each candidate gets a score from its evidence (hits, misses, status) and
its specificity (how many query features it matches). The retrieval returns the top `k`
with their scores, sorted best first. The default `k` is a decision (D3).

**Fallback.** When no candidate matches, the retrieval returns an empty result. It does
not scan the full table. A caller that needs a broader answer asks for one explicitly,
and the call is logged as `retrieval_fallback` with the query and the store size.

**Confidence.** A returned candidate carries its score, so the predictor and the
metacognition module can read how sure the retrieval was instead of assuming a match.

**Learned scorer (phase 2, not in this build).** A small model maps (query features,
candidate features) to a probability, trained online from the logged hits and misses.
Section 5 explains why it waits for the index.

## 4. Behavior

- **Insert.** When a rule is created or promoted (Part 1 section 4), it is added to the
  index under every feature it carries. When a rule breaks or is dropped, its postings
  are removed. Removal and insert are both logged.
- **Equivalence.** For any query, the indexed retrieval returns the same top candidate as
  a brute-force scan over the same store. The index is a speedup and must not change the
  answer.
- **Scan callers move to the index.** `gaits` and the goal helpers read postings for
  their feature instead of the whole table. Their outputs must match the current outputs
  on the same facts.
- **Bounded work.** A retrieval reads at most the postings for its features. The number
  of postings read is logged per call.

## 5. Decisions I made

These are my calls, changeable without changing section 1.

- **D1. Index before learning.** The index comes first. It is exact, testable, and
  matches the equivalence rule. A learned scorer comes later, only if the index's top
  candidate is measurably wrong often enough to matter. Reason: a learned scorer trained
  on 25 public games risks fitting those games, and the held-out games would show it.
- **D2. No silent scans.** Every retrieval that reads more than its postings is logged.
  Reason: a scan that happens without a record is how the current cost went unnoticed.
- **D3. Default `k` is 5.** The source doesn't give a number. Five is enough for the
  predictor to check a few alternatives without reading the table. It is a starting
  value to test.
- **D4. Probability means the evidence score, not a learned output.** Until D1's learned
  scorer exists, "most probable" means the highest evidence score under a fixed formula.
  The formula is in the code and in the log, so it can be checked.
- **D5. Meaning means canonical features.** A feature is a kind, a role, a shape class,
  or an effect tag. Meaning-based matching in this build is matching on those features.
  A learned embedding of meanings is the phase 2 question, not this build.

## 6. Test plan

**Equivalence (F1, D2).** Record a set of queries from real runs. Run the indexed and
brute-force retrievals on each. Every top candidate must match. Any mismatch fails the
build.

**Speed.** Time each retrieval path with a profiler on the same runs, before and after.
The baseline has not been measured. The first step is to measure the scans, and the
work is worth doing only if they are a meaningful share of move time.

**Accuracy.** Share of queries where the top candidate's predicted next frame matches
the observed frame. This uses the predictor's existing check. It is the number the
learned scorer (phase 2) must beat, if phase 2 happens.

**Scaling.** Retrieval cost against store size, at 10 and 100 times the current fact
count. A flat curve means the index works. A curve that tracks store size means a scan
is still on the path.

**Held-out games.** Same rule as Part 1 section 6. Retrieval changes are judged on
games not used to tune them.

**Unit tests.** Insert and remove keep every posting consistent. An empty query returns
an empty result without reading the table. A `retrieval_fallback` event is logged when
a caller asks for a broader answer.

## 7. Open questions for Kameron

- **Q1.** Is "most probable" the evidence score (D4), or should a rule's age, recency, or
  the level it came from count too?
- **Q2.** Is `k = 5` (D3) right, or should the predictor see every candidate above a
  threshold?
- **Q3.** Should the phase 2 learned scorer be a small neural network, or a tree over the
  same features? The tree is cheaper to run and easier to inspect. The network can learn
  meaning from raw features, but it needs more games to train.
- **Q4.** Does this retrieval construct have a place in the HCQM tree? I have not mapped
  it yet. It may sit under working memory (1.6) or pattern recognition (8.3), and I'd
  check Appendix A before claiming either.
