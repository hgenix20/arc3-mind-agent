export const meta = {
  name: 'v2-wave2-integration',
  description: 'Build the planner agent (WP10) in two sequential parts on agent-v2, review each, fix once, then an adversarial whole-branch review',
  phases: [
    { title: 'Part A', detail: 'planner_agent.py and the fluid.py hook' },
    { title: 'Part B', detail: 'config, run, notebook, tests' },
    { title: 'Branch review', detail: 'three skeptics on the whole branch, then one fix round' },
  ],
}

const PY = 'C:/arc-agent-venv/Scripts/python.exe'
const REPO = 'C:/arc-agent'
const BRIEF = 'C:/Users/Sagac/AppData/Local/Temp/claude/C--Program-Files-Git/76676058-e8f8-4289-9a20-0d0f4ca84f38/scratchpad/wp10-brief.md'
const PLAN = `${REPO}/docs/superpowers/plans/2026-10-31-agent-v2-build.md`
const MAP = `${REPO}/docs/superpowers/plans/2026-10-31-v2-code-map.md`
const SPEC = `${REPO}/docs/superpowers/specs/2026-10-31-agent-spec-v2.md`
const SESSION = 'Claude-Session: https://claude.ai/code/session_01ED1JLjSNaMyvkVda5AScAg'
const BASE = args && args.base ? args.base : 'agent-v2'

const IMPL_SCHEMA = { type: 'object', properties: { status: { type: 'string' }, commits: { type: 'string' }, interface: { type: 'string' }, tests: { type: 'string' }, concerns: { type: 'string' } }, required: ['status', 'commits', 'interface', 'tests'] }
const REVIEW_SCHEMA = { type: 'object', properties: { verdict: { type: 'string' }, findings: { type: 'string' }, gates: { type: 'string' } }, required: ['verdict', 'findings', 'gates'] }
const SKEPTIC_SCHEMA = { type: 'object', properties: { refuted: { type: 'boolean' }, findings: { type: 'string' } }, required: ['refuted', 'findings'] }

const common = `You work in the repository at ${REPO} on branch agent-v2 (confirm with "git -C ${REPO} branch --show-current"; stop and report BLOCKED if it differs). Every tool call uses absolute paths under ${REPO}; the shell's default directory is not the repo. Interpreter ${PY}; never system Python. Read first: ${BRIEF} (your part in full), ${PLAN} sections 2 and 3, ${MAP} (confirm every claim against the files), the spec sections your part names in ${SPEC}, and the wave-1 modules' docstrings (the brief lists them). Method: failing tests first, then the smallest implementation that passes, small units, types on every signature (pyright strict), docstrings that state intent, contract, failure modes and the spec section; comments say why. Plain English, no em-dashes, no "not X but Y" phrasing, in code comments and docstrings too. Bound every loop. Gates from ${REPO} (cd there in the same command): "${PY} -m ruff format <changed files>", "${PY} -m ruff check ." (All checks passed), "${PY} -m pyright" (0 errors), the full suite "${PY} -m pytest -q -p no:cacheprovider" redirected to a file under C:/Users/Sagac/AppData/Local/Temp/claude/ (read its last 3 lines). Commit on agent-v2 with an imperative title under 72 characters, two or three lines of why, then the line "${SESSION}". No Co-Authored-By line. No subagents. Return: status (DONE, DONE_WITH_CONCERNS or BLOCKED), commit hashes, the public interface produced (names and signatures), the pytest last line, and concerns.`

const reviewPrompt = (part, impl) => `Review the commits ${impl.commits} on agent-v2 in ${REPO} (read them with "git -C ${REPO} show <hash> --stat" and "git -C ${REPO} diff <first>~1..<last>"). The contract is ${BRIEF}, ${part}, plus ${PLAN} sections 2 and 3 and the spec sections it names in ${SPEC}. The implementer reported: ${impl.status}; interface: ${impl.interface}; concerns: ${impl.concerns || 'none'}. Check every numbered item of that part against the code and the tests: names and constants as stated, the control flow of the brief's A3 and A4 (or the Part B wiring), the log fields and events of the runlog docstring, every listed test present and asserting the stated outcome (a test that only checks no exception does not count), docstrings without em-dashes, no change to files the brief does not name (fluid.py may gain only the additive hook). Run from ${REPO} with cd: "${PY} -m ruff check .", "${PY} -m pyright", "${PY} -m pytest -q -p no:cacheprovider tests/mind/test_planner_agent.py tests/mind/test_fluid_agent.py tests/mind/test_llm_drive.py tests/test_kaggle_mind_notebook.py". Return verdict APPROVE or CHANGES, numbered findings (file:line, what is wrong, what the contract says), most severe first, and the gate outputs' last lines. Do not edit files.`

const fixPrompt = (part, impl, review) => `${common}\n\nYou are fixing review findings for ${part}. The findings, most severe first:\n${review.findings}\n\nFix every finding that names a contract violation (wrong control flow, missing requirement, wrong signature, missing or weak test). For a finding you believe is wrong, leave the code and say why. Commit the fix. Return status, commit hash, what you changed per finding, what you declined and why, and the pytest last line.`

phase('Part A')
const a = await agent(`${common}\n\nYour task is Part A of ${BRIEF}: planner_agent.py (new) and the one additive hook in fluid.py, with the tests of the brief's test list that Part A alone can satisfy (tests 1, 2, 3, 4, 7 can run with a store path under tmp_path even before Part B wires the CLI; write them now). Part B (config, submit, notebook, docs, tests 5 and 6) is a separate task that follows yours; do not do it, but make Part B possible by giving PlannerAgent the store parameter the brief names.`, { label: 'implement:wp10a', phase: 'Part A', schema: IMPL_SCHEMA })
if (!a || a.status === 'BLOCKED') return { a, stopped: 'Part A blocked' }
const ar = await agent(reviewPrompt('Part A', a), { label: 'review:wp10a', phase: 'Part A', schema: REVIEW_SCHEMA })
let af = null, arr = null
if (ar && !ar.verdict.toUpperCase().startsWith('APPROVE')) {
  af = await agent(fixPrompt('Part A', a, ar), { label: 'fix:wp10a', phase: 'Part A' })
  arr = await agent(reviewPrompt('Part A', { ...a, concerns: `after a fix round: ${af}` }), { label: 'rereview:wp10a', phase: 'Part A', schema: REVIEW_SCHEMA })
}

phase('Part B')
const b = await agent(`${common}\n\nYour task is Part B of ${BRIEF}: config.py (PLANNER), submit.py (store wiring, --longterm, make_agent), the notebook builder test, docs/agent-submission.md section, and tests 5 and 6 of the brief's test list in tests/mind/test_planner_agent.py (Part A wrote the others; read the file and its helpers first; Part A's report: ${a.interface}). Also run the whole tests/mind/test_planner_agent.py and the notebook tests.`, { label: 'implement:wp10b', phase: 'Part B', schema: IMPL_SCHEMA })
if (!b || b.status === 'BLOCKED') return { a, ar, af, arr, b, stopped: 'Part B blocked' }
const br = await agent(reviewPrompt('Part B', b), { label: 'review:wp10b', phase: 'Part B', schema: REVIEW_SCHEMA })
let bf = null, brr = null
if (br && !br.verdict.toUpperCase().startsWith('APPROVE')) {
  bf = await agent(fixPrompt('Part B', b, br), { label: 'fix:wp10b', phase: 'Part B' })
  brr = await agent(reviewPrompt('Part B', { ...b, concerns: `after a fix round: ${bf}` }), { label: 'rereview:wp10b', phase: 'Part B', schema: REVIEW_SCHEMA })
}

phase('Branch review')
const LENSES = [
  ['correctness', 'Trace the per-move control flow of PlannerAgent._choose and _replan against spec sections 3.1 to 3.6 and the brief A3 and A4: find any path where a segment is executed without a rollout check, where a miss that matters does not stop the segment, where the fork tree and the return route disagree with the explorer graph, where the controller order is not the one in the brief, or where the base FluidAgent behaviour changed for configs other than PLANNER. Run the planner tests and the fluid tests.'],
  ['safety-and-bounds', 'Look for unbounded loops, unbounded lists, missing time or token gates on the planner call path, exceptions that can escape choose() (the base catches them, so confirm the agent_error path still returns a legal action), thread-safety of the long-term store under 8 concurrent games, and any path that writes outside the log dir. Also confirm no module names a game id, a level, or a sprite (constraint C2 of the spec).'],
  ['logging-and-reports', 'Check every v2 log field and event the runlog docstring promises is written at the right step with the right types (run a FakeGame with the scripted chat and read the JSONL), that scripts/mind_metrics.py v2_metrics consumes them without KeyError on a real log, and that prev_matched pairs with the previous move\'s p_step. Report any field that is promised and never written, or written and never documented.'],
]
const base = (await agent(`Report the merge base of agent-v2 with ${BASE} in ${REPO} as a bare hash: run "git -C ${REPO} merge-base ${BASE} agent-v2" and "git -C ${REPO} log --oneline ${BASE}..agent-v2 | head -40"; return the hash on the first line and the log after it.`, { label: 'branch:range', phase: 'Branch review' })) || ''
const skeptics = await parallel(LENSES.map(([lens, task]) => () => agent(`You are a skeptical reviewer of the whole agent-v2 branch in ${REPO} (absolute paths; interpreter ${PY}; do not edit files). Branch range: ${base.split('\n')[0]}..HEAD; read the diff with "git -C ${REPO} diff <base>..HEAD --stat" then the files that matter for your lens. Your lens: ${lens}. ${task} Default to refuted=true (the branch is not ready) if you find a defect that would change behaviour in a run; refuted=false only if you looked and found nothing of that kind. Return refuted and numbered findings with file:line, each with the concrete input or state that triggers it and the wrong result.`, { label: `skeptic:${lens}`, phase: 'Branch review', schema: SKEPTIC_SCHEMA })))
const found = skeptics.filter(Boolean)
const refuted = found.filter(s => s.refuted)
log(`branch review: ${refuted.length}/${found.length} skeptics refuted readiness`)
let finalFix = null, finalCheck = null
if (refuted.length) {
  finalFix = await agent(`${common}\n\nThree skeptical reviewers read the whole agent-v2 branch. Their findings:\n${found.map((s, i) => `Reviewer ${i + 1} (${LENSES[i][0]}, refuted=${s.refuted}):\n${s.findings}`).join('\n\n')}\n\nFix every finding that would change behaviour in a run (wrong control flow, missing gate, unbounded work, promised log field never written, thread-safety). Each fix gets a failing test first where one is possible. Leave findings you believe wrong and say why. Commit. Return status, commit hashes, what you changed per finding, what you declined and why, and the full-suite pytest last line.`, { label: 'fix:branch', phase: 'Branch review' })
  finalCheck = await agent(`Verify the fix commits on agent-v2 in ${REPO} after this report: ${finalFix}. Run from ${REPO} with cd: "${PY} -m ruff check .", "${PY} -m pyright", "${PY} -m pytest -q -p no:cacheprovider" (full suite, redirect to a file under C:/Users/Sagac/AppData/Local/Temp/claude/ and read the last 3 lines). Then re-check only the findings the fixer says it fixed, by reading the code. Return verdict APPROVE or CHANGES with the gate last lines and any finding that is still open. Do not edit files.`, { label: 'verify:branch', phase: 'Branch review', schema: REVIEW_SCHEMA })
}
return { a, ar, af, arr, b, br, bf, brr, skeptics: found, finalFix, finalCheck }
