"""Leitura e validacao do log estruturado que cada agente escreve (prompts v3, secao AGENT LOG).

Rodada de agente unico: <analise>/provenance/agent_log.jsonl
Rodada de conselho:     <analise>/provenance/agent_log/<agente>.jsonl
"""

import json
import pathlib

REQUIRED_FIELDS = {"step", "time", "agent", "phase", "action", "target", "purpose", "outcome"}
ACTIONS = {"view_image", "run_command", "write_file", "edit_file", "read_file", "search", "start_subagent", "final_answer"}


def log_files(analysis: pathlib.Path) -> list[pathlib.Path]:
    single = analysis / "provenance" / "agent_log.jsonl"
    council = sorted((analysis / "provenance" / "agent_log").glob("*.jsonl"))
    return ([single] if single.exists() else []) + council


def read_logs(analysis: pathlib.Path) -> dict:
    """Retorna as entradas validas por agente e a contagem de linhas invalidas, com o motivo."""
    agents: dict[str, list[dict]] = {}
    invalid: list[str] = []
    concatenated = 0
    decoder = json.JSONDecoder()
    for path in log_files(analysis):
        for n, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
            if not line.strip():
                continue
            # recupera objetos colados sem quebra de linha; o desvio de formato fica contado a parte
            objects, pos, text = [], 0, line.strip()
            try:
                while pos < len(text):
                    obj, end = decoder.raw_decode(text, pos)
                    objects.append(obj)
                    pos = end
                    while pos < len(text) and text[pos].isspace():
                        pos += 1
            except json.JSONDecodeError:
                invalid.append(f"{path.name}:{n} nao e JSON")
                continue
            concatenated += len(objects) - 1
            for entry in objects:
                _add(entry, path, n, agents, invalid)
    total = sum(len(v) for v in agents.values()) + len(invalid)
    return {"files": len(log_files(analysis)), "agents": agents, "invalid": invalid, "lines": total,
            "concatenated": concatenated}


def _add(entry, path: pathlib.Path, n: int, agents: dict, invalid: list) -> None:
    if not isinstance(entry, dict) or not REQUIRED_FIELDS <= set(entry):
        invalid.append(f"{path.name}:{n} faltam campos {sorted(REQUIRED_FIELDS - set(entry)) if isinstance(entry, dict) else 'todos'}")
        return
    if entry["action"] not in ACTIONS:
        invalid.append(f"{path.name}:{n} acao desconhecida {entry['action']!r}")
        return
    agents.setdefault(str(entry["agent"]), []).append(entry)
