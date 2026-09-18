"""Reexporta as sessões do opencode de rodadas já feitas (sessions/<id>.json).

Uso:
    tools/py tools/reexport_sessions.py results/v2 results/v1-runner

Necessário para as rodadas feitas antes da correção da exportação, que saía truncada por pipe.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from run_opencode import export_sessions, summarize  # noqa: E402


def main() -> int:
    for root in sys.argv[1:]:
        for events in sorted(pathlib.Path(root).rglob("opencode_events.jsonl")):
            run_dir = events.parent
            summary = summarize(events)
            ids = [s for s in [summary["main_session"], *summary["child_sessions"]] if s]
            export_sessions(run_dir, ids)
            print(f"{run_dir}: {len(ids)} sessões")
    return 0


if __name__ == "__main__":
    sys.exit(main())
