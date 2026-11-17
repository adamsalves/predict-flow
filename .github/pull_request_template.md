Closes #

## Contexto

<!-- O problema ou a necessidade, e por que importa. 1 a 3 frases. -->

## O que muda

<!-- Os blocos da implementação em linguagem clara, um por item. Se houver endpoint novo ou alterado: tabela método | rota | resposta. -->

-

## Como verificar

- [ ] <!-- passo manual: comando, URL, ação, resultado esperado -->

## Decisões e desvios

<!-- ADR criada ou tocada em `docs/adr/`, desvio do escopo e o motivo, justificativas de exceções marcadas com ⚠️ em "Critérios de aceite". Apague a seção se não houver. -->

## Critérios de aceite

<!-- Lista copiada da issue linkada, item a item. Nenhum item pode ficar sem marcação. -->

- [ ] <!-- ✅ atendido → marcar `- [x]` na própria issue (fonte da verdade do status) · ⚠️ exceção → justificativa em "Decisões e desvios" -->

## Checklist

- [ ] Título em Conventional Commits, ~400 linhas no máximo (sem lockfile, fixture, snapshot)
- [ ] Labels da issue aplicados ao PR
- [ ] **Conferência dos critérios de aceite da issue feita** (seção "Critérios de aceite" preenchida: ✅ ou ⚠️ + justificativa em "Decisões e desvios"; itens atendidos marcados `- [x]` na issue)
- [ ] **Ajustes da revisão resolvidos com comentário ponto a ponto no PR** (achado → resolução → commit; sugestões não adotadas justificadas) — quando houver achados
- [ ] ADR linkada em `docs/adr/` quando houver decisão arquitetural ou científica
- [ ] Testes pytest do que foi adicionado (se for bug: teste de regressão)
- [ ] CI verde (lint, pytest com Postgres)

## Screenshots

<!-- Só para UI. Apague a seção se não for UI. Comparar com o protótipo em 375, 768 e 1280 quando o frontend existir. -->
