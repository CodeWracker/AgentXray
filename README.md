# xray-agent-bench

Bancada para comparar LLMs multimodais, sozinhos (single-agent) ou em conselho de instâncias do mesmo modelo (multi-agent-single-model), na leitura de radiografias como apoio de segunda opinião. O modelo recebe a imagem, decide por conta própria que análises fazer (processamento de imagem, medições, modelos pré-treinados do [torchxrayvision](https://github.com/mlmed/torchxrayvision)), avalia o que cada uma rendeu e entrega um JSON com achados, impressão e limitações.

Não é ferramenta clínica. É um experimento sobre como agentes usam (ou deixam de usar) ferramentas.

## Estrutura

```
inputs/                     imagens de entrada (ID001–ID003)
prompts/v1/                 prompts da primeira rodada (lista fixa de transformações no single-agent)
prompts/v2/                 prompts atuais: o modelo escolhe as análises; ambiente e reprodutibilidade em _shared/
results/v1/                 resultados da primeira rodada, por modo e modelo
results/v2/<modo>/<modelo>/<timestamp>/   rodadas novas, criadas por tools/new_run.py
tools/txv_setup.py          baixa os pesos do torchxrayvision e confere contra weights.lock.json
tools/new_run.py            prepara uma rodada isolada com PROMPT.md e run_manifest.json
tools/check_run.py          valida o contrato da rodada e testa a reprodução byte a byte
pyproject.toml, uv.lock     ambiente travado (torch CPU)
weights.lock.json           sha256 de cada peso de modelo usado
```

## O que muda na v2

Na v1 o prompt single-agent impunha cinco transformações fixas (grayscale, contraste, nitidez, bordas), e o multi-agent já deixava os agentes escolherem, mas nenhum dos dois exigia que as saídas pudessem ser refeitas de forma verificável, nem que o modelo julgasse explicitamente se cada análise tinha servido. A v2 unifica a ideia: nos dois modos o modelo decide as análises, registra antes de cada uma a pergunta que ela deve responder e o que mudaria sua interpretação, avalia depois se ela serviu, e descarta o que não serviu sem apagar o rastro. No fim, compara a conclusão com a primeira impressão (`provenance/first_look.md` ou os `firstpass` dos agentes) e diz qual análise causou cada mudança.

O torchxrayvision entra como inventário, não como recomendação. O prompt descreve fatos sobre as ferramentas (o que cada modelo produz, resolução, convenções de pré-processamento, onde está o código-fonte para consulta) e diz que todos foram treinados em radiografia de tórax, mas não diz se servem ou não para as imagens do teste. Descobrir isso é parte da tarefa: antes de confiar num modelo pré-treinado, o agente precisa estabelecer sozinho se a imagem está dentro do que o modelo cobre e se as saídas se comportam de forma sensata, e registrar como checou. Os modelos de predição de idade, sexo e raça do pacote ficam de fora de propósito.

Isso importa porque as três imagens são de extremidades (tornozelo, cotovelo/antebraço, antebraço), fora do domínio de treino. Como exemplo, o DenseNet `densenet121-res224-all` aplicado ao ID003 (um antebraço) devolve `Pneumonia 0.53`, `Emphysema 0.51` e `Fracture 0.50`, números sem sentido clínico ali. Um bom agente deveria perceber isso; um agente ingênuo pode usar o "Fracture" como evidência. Essa diferença de comportamento é justamente o que a v2 deixa observável, via `provenance/tools.json` (cada ferramenta usada ou rejeitada, com `trusted` e o motivo).

O JSON final continua com os mesmos quatro campos da v1, para as rodadas serem comparáveis. Todo o resto vai para `provenance/`, `measurements/` e `planning/`.

## Reprodutibilidade

O ambiente é travado pelo `uv.lock` (torch CPU, para a inferência ser determinística e igual em qualquer máquina) e os pesos pelo `weights.lock.json`. Cada rodada guarda em `run_manifest.json` o hash do prompt exato entregue ao modelo, das imagens, dos locks, o commit do repositório e as versões das bibliotecas.

Cada pasta de análise precisa ter um `scripts/reproduce.py` determinístico que recria, a partir só do original, tudo o que ficou em `images/` e `measurements/`. O `tools/check_run.py` copia a pasta para um diretório temporário, apaga as saídas, roda o script e compara os arquivos byte a byte. A liberdade de explorar continua total (o agente pode rodar o que quiser, na ordem que quiser); a exigência é só que aquilo que ele mantém como evidência possa ser refeito.

O prompt pede que o agente trabalhe só dentro do diretório da rodada, porque `results/` tem conclusões de rodadas anteriores que contaminariam a análise. Se o harness do modelo não respeitar isso, use `--root` para criar a rodada fora do repositório.

## Como rodar

Preparar o ambiente (uma vez; os pesos ocupam ~1,1 GB em `~/.torchxrayvision`):

```bash
uv sync
uv run python tools/txv_setup.py
```

Criar uma rodada e entregar o prompt ao agente:

```bash
uv run python tools/new_run.py --mode single --model sonnet5
uv run python tools/new_run.py --mode agents --model qwen3.8-27b
# imprime o diretório criado; abra o agente nele e passe o conteúdo de PROMPT.md
```

Commite antes de criar a rodada, para o `repo_commit` do manifesto descrever exatamente os prompts usados (o script avisa quando há mudanças pendentes).

Validar depois que o agente terminar:

```bash
uv run python tools/check_run.py results/v2/single-agent/sonnet5/<timestamp>
```

O mesmo comando aceita as pastas da v1, mas nelas só confere o JSON final.

## Pendências observadas na v1

A pasta `results/v1/multi-agent-single-model/gemma4-36b/` está vazia (a rodada single-agent usou `gemma4-26b`; talvez o nome esteja trocado). Em `results/v1/multi-agent-single-model/qwen3.8-27b/`, o ID003 não tem o JSON final. E as rodadas divergem até na anatomia: o sonnet5 single-agent descreveu o ID002 como joelho e perna, enquanto o multi-agent descreveu cotovelo e antebraço, que é o que a imagem mostra. É um bom caso para comparar com a v2.
