# ADR-0009: Target, temporalidade e avaliação/backfill

- Status: Aceito
- Data: 2026-11-16

## Contexto

Avaliar um modelo de churn exige rigor temporal: features só podem usar o que era conhecido no momento do score, e a unidade de análise precisa estar definida para que splits e métricas façam sentido. O dataset Telco é histórico e **não possui horizonte prospectivo observado** — o que limita o que pode ser afirmado a partir dele.

## Decisão

- **Unidade de análise definida explicitamente** pelo desenho do dataset (cliente), documentada no manifesto (ADR-0002) e usada de forma consistente em splits, métricas e backfill.
- **`score_at`**: quando a predição foi emitida. **`feature_as_of`**: até quando os dados de entrada eram conhecidos. Features nunca usam informação posterior a `feature_as_of` — leakage é falha de desenho, não detalhe.
- **Evento e exclusões**: definição formal do evento de churn, janela de elegibilidade e regras de exclusão documentadas antes do treino.
- **Horizonte H e janela de outcome**: definidos por dataset; outcome só é observado após a janela (ADR-0005), com tratamento explícito de **censura**.
- **Splits por clientes disjuntos**, cada qual com finalidade exclusiva: treino, validação, teste e backfill não compartilham clientes. Uma vez usado em treino/validação, um cliente nunca aparece em teste nem em avaliação apresentada como independente.
- **Limitação histórica do Telco declarada**: sem horizonte prospectivo observado, não se apresenta "churn em 90 dias" nem eficácia de retenção a partir desse dataset.

## Alternativas rejeitadas

- **Splits aleatórios por linha**: rejeitado — linhas do mesmo cliente em splits diferentes contaminam a avaliação e inflam as métricas.
- **Reuso de teste para seleção de modelo**: rejeitado — teste tem finalidade exclusiva de avaliação; seleção acontece na validação (ADR-0004).
- **Prometer horizonte prospectivo ("churn em 90 dias") com dados históricos**: rejeitado — o dataset não sustenta essa afirmação.

## Consequências

- Métricas apresentadas em qualquer tela ou relatório são calculadas sobre o split correto e com janela madura — issues #5, #6, #10 e #16 implementam este rigor.
- Backfill segregado segue as mesmas regras de disjunção (ver ADR-0006): dados de backfill não vazam para avaliação apresentada como prospectiva.
