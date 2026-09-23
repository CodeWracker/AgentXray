# AGENT LOG

Every agent writes its own structured log while it works. The log is part of the task and is evaluated.

* Single-agent runs: one file, `<analysis folder>/provenance/agent_log.jsonl`.
* Multi-agent runs: one file per agent, `<analysis folder>/provenance/agent_log/<agent name>.jsonl`. The orchestrator uses the name `ORCHESTRATOR`; sub-agents use their persistent names (for example `Agent_03`).

Right after each action, append exactly one line to your own log: a single JSON object, with no line breaks inside it. An action is any tool call (viewing an image, running a command or script, writing or editing a file, searching, starting a sub-agent) and the final answer. Append with the shell, for example `printf '%s\n' '<json>' >> provenance/agent_log.jsonl`, or with a short Python call that opens the file in append mode. Never rewrite, reorder or delete earlier lines, and do not write entries for actions you did not take.

Required fields of every line:

* `step`: integer, 1 for your first action, increasing by one per line of your own log
* `time`: current UTC time in ISO 8601, taken from the system clock (for example with `date -u +%Y-%m-%dT%H:%M:%SZ`)
* `agent`: your name (`main` in single-agent runs)
* `phase`: short name of the stage of the workflow you are in (for example `first_look`, `analysis`, `inspection`, `synthesis`, `final`)
* `action`: one of `view_image`, `run_command`, `write_file`, `edit_file`, `read_file`, `search`, `start_subagent`, `final_answer`
* `target`: the file, script, command or agent the action was applied to
* `purpose`: the question this action was meant to answer
* `outcome`: what you actually saw, measured or produced, in one or two sentences

Optional fields, when they apply: `images_created` (list of image paths), `models_used` (list of pretrained models or library functions whose output you used), `trusted` (`true` or `false` for those outputs, with `reason`).

The log records what you did and why. It does not replace `provenance/decisions.md`, `planning/` or the agent reports.
