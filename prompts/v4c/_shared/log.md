# AGENT LOG

Every agent keeps a structured log of its own work. The log is part of the task and is evaluated.

Right after each action, call the `log_action` tool once to record it, before taking the next action. An action is any other tool call: viewing an image, running a command or script, writing or editing a file, reading a file, searching, or starting a sub-agent. When you write your final answer, record it too, with the action `final_answer`. The harness blocks your next action while the previous one is not recorded; if that happens, record the previous action and repeat the call. Every field listed below is required except the optional ones; an entry with a missing or empty field is rejected and the action stays unrecorded.

Fields of `log_action`:

* `agent`: your name (`main` in single-agent runs, `ORCHESTRATOR` for the orchestrator of a council, and the persistent name you were given, such as `Agent_03`, for a sub-agent)
* `phase`: short name of the stage of the workflow you are in (for example `first_look`, `analysis`, `inspection`, `synthesis`, `final`)
* `action`: one of `view_image`, `run_command`, `write_file`, `edit_file`, `read_file`, `search`, `start_subagent`, `final_answer`
* `target`: the file, script, command or agent the action was applied to
* `purpose`: the question this action was meant to answer
* `outcome`: what you actually saw, measured or produced, in one or two sentences
* optional: `images_created`, `models_used`, and `trusted` with `reason` for the outputs of those models

The harness adds the step number and the time. Record only actions you actually took, as they happened. The log does not replace `provenance/decisions.md`, `planning/` or the agent reports.
