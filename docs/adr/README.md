# ADRs — Architecture Decision Records

Este diretório é a fonte da verdade das decisões arquiteturais e científicas do predict-flow. Cada ADR segue o formato: **Status · Contexto · Decisão · Alternativas rejeitadas · Consequências**.

## Decisões de produto já fixadas (base de todas as ADRs)

- O próximo entregável é uma **demo local, single-tenant**, somente com dados públicos ou sintéticos.
- O dataset Telco sustenta **classificação histórica sem horizonte prospectivo observado**; não se apresenta "churn em 90 dias" nem eficácia de retenção a partir dele.
- O eixo do MVP é `importar → pontuar → priorizar → abrir cliente → decidir/registrar ação → observar outcome → avaliar`; o formulário de inferência é apenas sandbox.
- Dados reais exigem o gate de piloto definido na ADR-0010 e na issue #17.

## Índice

| ADR | Título | Status |
|---|---|---|
| [0001](0001-pipeline-unico-sklearn.md) | Pipeline único de preprocessing + modelo | Aceito |
| [0002](0002-metadados-e-manifesto.md) | Metadados e manifesto do modelo (`model_meta.json`) | Aceito |
| [0003](0003-entrega-e-rollback-do-modelo.md) | Entrega e rollback do modelo por versão | Aceito |
| [0004](0004-selecao-de-modelo-e-explicacao.md) | Seleção de modelo (LR baseline / RF candidato) e explicação | Aceito |
| [0005](0005-outcome-e-temporalidade.md) | Outcome e temporalidade da predição | Aceito |
| [0006](0006-feedback-e-origem-dos-dados.md) | Feedback e origem dos dados (`demo_backfill`, `historical_test`, `prospective`) | Aceito |
| [0007](0007-contrato-openapi.md) | Contrato OpenAPI e codegen TypeScript | Aceito |
| [0008](0008-processo-de-design.md) | Processo de design do frontend | Aceito |
| [0009](0009-target-temporalidade-e-avaliacao.md) | Target, temporalidade e avaliação/backfill | Aceito |
| [0010](0010-demo-versus-piloto.md) | Demo versus piloto, segurança e ciclo de vida de dados | Aceito |
| [0011](0011-politica-de-decisao-e-acoes.md) | Política de decisão e avaliação de ações | Aceito |

## Convenções

- ADRs são imutáveis após "Aceito"; mudanças de direção geram um ADR novo que supera o anterior.
- Nenhuma ADR pode depender de documento externo inacessível como fonte obrigatória.
- Issues de implementação devem linkar a ADR que fundamenta a decisão (ex.: #2 fundamenta-se na ADR-0010).
