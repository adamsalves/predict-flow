# ADR-0005: Outcome e temporalidade da predição

- Status: Aceito
- Data: 2026-11-16

## Contexto

Uma predição de churn só pode ser avaliada quando o tempo passa: o outcome (o cliente de fato churnou?) não existe no momento do score. Registrar outcome apressadamente ou sem marcos temporais torna impossível reconstruir o que era sabido quando o score foi emitido — e contamina qualquer avaliação posterior.

## Decisão

- **Outcome nasce nulo** em toda predição: `actual_churn` começa `NULL`.
- Cada predição carrega os timestamps que permitem reconstruir o estado do mundo no score: `score_at` (quando pontuou) e `feature_as_of` (até quando os dados de entrada eram conhecidos — ver ADR-0009).
- **`actual_churn=false` só é gravado após observação suficiente da janela de outcome**: antes disso, a predição permanece imatura/censurada, nunca "negativa confirmada".
- Cada atualização de outcome registra **origem** (ver ADR-0006), **autor** e **momento da observação** — distinto do momento do evento.
- Outcome nulo de predição antiga não é churn nem não-churn: é censura, e é tratado como tal nas avaliações (ver ADR-0009).

## Alternativas rejeitadas

- **Default `false` para outcome não observado**: rejeitado — transforma censura em classe, enviesando todas as métricas para baixo.
- **Outcome gravado pelo calendário (job que fecha janelas sem registrar autor/origem)**: rejeitado como único mecanismo — automação pode existir, mas sempre com origem e trilha de auditoria.

## Consequências

- Métricas de avaliação precisam filtrar por maturidade da janela de outcome; avaliar predições imaturas é um erro explícito.
- O schema do banco precisa representar outcome nulo + metadados de observação (ver issue #7).
- A separação entre "momento do evento" e "momento da observação" é contrato permanente — ver também ADR-0006.
