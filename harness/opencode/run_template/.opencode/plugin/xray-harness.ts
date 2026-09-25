// Plugin do harness AgentXray para o opencode (prompts v4 e v4c).
//
// Copiado para <run>/.opencode/plugin/ pelo tools/new_run.py, que tambem liga <run>/.opencode/node_modules as
// dependencias instaladas em harness/opencode/node_modules. Le a configuracao
// da rodada em <run>/harness.json. Faz tres coisas:
//   1. oferece a ferramenta log_action, com que o agente registra cada acao no log estruturado;
//   2. bloqueia a proxima acao de uma sessao enquanto a anterior nao for registrada (log completo por construcao);
//   3. grava o registro do harness: cada chamada de ferramenta, cada bloqueio e os parametros enviados ao modelo;
//   4. bloqueia qualquer chamada que cite um caminho fora da pasta da rodada (argumentos, comandos de shell e o
//      conteudo dos arquivos escritos), salvo /tmp, o temporario do sandbox e os caminhos so de leitura do harness.json.
import { appendFileSync, mkdirSync, readFileSync } from "node:fs"
import { dirname, isAbsolute, join, resolve } from "node:path"
import { tool, type Plugin } from "@opencode-ai/plugin"

type Paths = { read_write: string[]; read_only: string[]; forbidden_roots: string[] }
type RunConfig = { analysis_dir: string; agent_log: string; harness_log: string; mode: string; paths?: Paths }

// caminho absoluto ou relativo a home ("~/") dentro de um texto (comando de shell ou conteudo de arquivo); "~" seguido
// de numero ou texto e so "aproximadamente" e nao conta
const PATH_TOKEN = /(?:^|[\s'"=:(,\[{])((?:~\/|\/)[^\s'"`;|&<>(),\]}]*)/g
// "../" em qualquer texto; ".." e "~" sozinhos so contam em comandos de shell (cd .., ls ~)
const PARENT_PATH = /(?:^|[\s'"=:(,\/])\.\.\//
const SHELL_BARE = /(?:^|[\s'"=;&|(])(?:\.\.|~)(?=$|[\s'";&|)])/
// ".." como componente de um argumento de caminho
const PARENT_SEGMENT = /(?:^|\/)\.\.(?:\/|$)/

const under = (path: string, root: string) => path === root || path.startsWith(root.endsWith("/") ? root : root + "/")

// devolve o primeiro caminho proibido citado nos argumentos da chamada, ou null
function outsidePath(tool: string, args: Record<string, unknown>, cwd: string, paths: Paths): string | null {
  const allowed = [...paths.read_write, ...paths.read_only]
  const forbidden = (p: string) => {
    if (p.startsWith("~/")) return true
    const abs = resolve(cwd, p)
    return paths.forbidden_roots.some((r) => under(abs, r)) && !allowed.some((r) => under(abs, r))
  }
  // argumentos que sao caminhos (read, write, edit, list, glob, grep): so a lista liberada
  for (const key of ["filePath", "path"]) {
    const value = args[key]
    if (typeof value !== "string" || !value) continue
    if (PARENT_SEGMENT.test(value) || value.startsWith("~")) return value
    const abs = isAbsolute(value) ? value : resolve(cwd, value)
    const roots = ["write", "edit"].includes(tool) ? paths.read_write : allowed
    if (!roots.some((r) => under(abs, r))) return value
  }
  // textos livres: comando de shell, padroes de busca e conteudo escrito
  for (const key of ["command", "pattern", "include", "content", "newString"]) {
    const value = args[key]
    if (typeof value !== "string" || !value) continue
    if (key !== "pattern" && key !== "include" && PARENT_PATH.test(value)) return "../"
    if (key === "command" && SHELL_BARE.test(value)) return value.match(SHELL_BARE)![0].trim()
    for (const m of value.matchAll(PATH_TOKEN)) {
      if (m[1] === "/" || m[1].startsWith("//")) continue
      if (forbidden(m[1])) return m[1]
    }
  }
  return null
}

// ferramentas que nao contam como acao do protocolo: o proprio log e a lista de tarefas interna
const EXEMPT = new Set(["log_action", "todowrite", "todoread"])
const ACTIONS = ["view_image", "run_command", "write_file", "edit_file", "read_file", "search", "start_subagent", "final_answer"] as const

function append(path: string, record: Record<string, unknown>) {
  mkdirSync(dirname(path), { recursive: true })
  appendFileSync(path, JSON.stringify(record) + "\n")
}

function short(value: unknown, limit = 400): unknown {
  const text = typeof value === "string" ? value : JSON.stringify(value)
  return text && text.length > limit ? text.slice(0, limit) + "..." : value
}

export const XrayHarness: Plugin = async ({ directory }) => {
  const config: RunConfig = JSON.parse(readFileSync(join(directory, "harness.json"), "utf8"))
  const agentLog = join(directory, config.agent_log)
  const harnessLog = join(directory, config.harness_log)
  const steps = new Map<string, number>()
  // acao de ferramenta ainda nao registrada, por sessao
  const pending = new Map<string, { tool: string; callID: string }>()

  append(harnessLog, { time: new Date().toISOString(), kind: "plugin_loaded", mode: config.mode })

  return {
    tool: {
      log_action: tool({
        description:
          "Record the action you just took in your agent log. Call it right after every other tool call, " +
          "before your next action; the harness blocks the next action until the previous one is logged.",
        args: {
          agent: tool.schema.string().describe("your name: main in single-agent runs, ORCHESTRATOR or Agent_NN in the council"),
          phase: tool.schema.string().describe("short name of the workflow stage, e.g. first_look, analysis, inspection, synthesis, final"),
          action: tool.schema.enum(ACTIONS).describe("kind of action you just took"),
          target: tool.schema.string().describe("file, script, command or agent the action was applied to"),
          purpose: tool.schema.string().describe("the question this action was meant to answer"),
          outcome: tool.schema.string().describe("what you actually saw, measured or produced, in one or two sentences"),
          images_created: tool.schema.array(tool.schema.string()).optional().describe("paths of images this action created"),
          models_used: tool.schema.array(tool.schema.string()).optional().describe("pretrained models or library functions whose output you used"),
          trusted: tool.schema.boolean().optional().describe("whether you trust those outputs for this image"),
          reason: tool.schema.string().optional().describe("why you trust them or not"),
        },
        async execute(args, context) {
          const step = (steps.get(context.sessionID) ?? 0) + 1
          steps.set(context.sessionID, step)
          const covered = pending.get(context.sessionID)
          pending.delete(context.sessionID)
          append(agentLog, {
            step,
            time: new Date().toISOString(),
            session: context.sessionID,
            agent_type: context.agent,
            covers_call: covered?.callID ?? null,
            covers_tool: covered?.tool ?? null,
            ...args,
          })
          return `logged step ${step}`
        },
      }),
    },

    "tool.execute.before": async (input, output) => {
      if (EXEMPT.has(input.tool)) return
      if (config.paths && input.tool !== "task") {
        const bad = outsidePath(input.tool, (output.args ?? {}) as Record<string, unknown>, directory, config.paths)
        if (bad) {
          append(harnessLog, { time: new Date().toISOString(), kind: "path_blocked", session: input.sessionID, tool: input.tool, call: input.callID, path: String(bad).slice(0, 300) })
          throw new Error(
            `Protocol: this call was not executed because it refers to a path outside your run directory (${String(bad).slice(0, 120)}). ` +
              `Work only with files inside your run directory; /tmp is available for temporary files. Paths with ".." or "~" are not allowed.`,
          )
        }
      }
      const open = pending.get(input.sessionID)
      if (open) {
        append(harnessLog, { time: new Date().toISOString(), kind: "blocked", session: input.sessionID, tool: input.tool, call: input.callID, unlogged: open })
        throw new Error(
          `Protocol: this call was not executed. Your previous tool call (${open.tool}) is not recorded in your agent log yet; ` +
            `it counts as an action even if it failed or its result was unexpected. Call log_action now to record it and what ` +
            `happened, and only then repeat this call. Repeating this call before log_action will be blocked again.`,
        )
      }
      pending.set(input.sessionID, { tool: input.tool, callID: input.callID })
      append(harnessLog, { time: new Date().toISOString(), kind: "tool_start", session: input.sessionID, tool: input.tool, call: input.callID, args: short(output.args) })
    },

    "tool.execute.after": async (input, output) => {
      if (input.tool === "log_action") return
      append(harnessLog, { time: new Date().toISOString(), kind: "tool_end", session: input.sessionID, tool: input.tool, call: input.callID, title: short(output.title, 200) })
    },

    "chat.params": async (input, output) => {
      // registra a amostragem efetivamente pedida ao servidor; a rodada nao altera esses valores
      append(harnessLog, {
        time: new Date().toISOString(), kind: "chat_params", session: input.sessionID, agent: input.agent,
        temperature: output.temperature, top_p: output.topP, top_k: output.topK, max_output_tokens: output.maxOutputTokens ?? null,
      })
    },
  }
}
