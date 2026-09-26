# HARNESS NOTES (opencode)

You are running inside opencode, without a human in the loop. Nobody will answer questions; make your own decisions and keep going until the task is complete.

To look at an image with your own vision, open the image file with the `read` tool. This is the only way to actually see an image; reading its pixel values in Python is not visual inspection.

The task is saved as `PROMPT.md` in the run directory. The rules that apply to every agent of this run (log, environment, reproducibility) are in `AGENTS.md`, which you and every sub-agent receive automatically.

When your session ends, an automated checker verifies the run: the final JSON, the required files, `provenance/tools.json`, `scripts/reproduce.py` in a clean copy, and that every image in `images/` was opened with the `read` tool. If anything fails, the session is resumed with the list of problems, and you must fix them.
