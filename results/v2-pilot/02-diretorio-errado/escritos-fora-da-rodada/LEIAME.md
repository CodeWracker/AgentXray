# Arquivos escritos fora do diretório da rodada

Estas pastas apareceram na raiz de `llm-xray-evaluation/` durante as rodadas de 17/09 entre 23:33 e 23:45, quando o opencode abria as sessões no diretório do runner em vez do diretório da rodada (ver o commit que corrige o `PWD`). Agentes que criaram a pasta de análise por caminho relativo gravaram aqui. Não é possível atribuir cada arquivo a uma sessão sem cruzar com as sessões exportadas das rodadas em `../v1-runner` e `../v2`.
