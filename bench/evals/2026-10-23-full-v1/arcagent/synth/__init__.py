"""Rule-synthesis experiment: can an LLM write an executable next-frame model?

The harness only uses what an agent sees through the action interface:
observed (grid, action, next grid) transitions. Game source files are never
read here and never shown to the model. See ``docs/synth-experiment.md``.
"""
