# ADR-0006: Feedback e origem dos dados

- Status: Aceito
- Data: 2026-11-16

## Contexto

O sistema opera com três tipos de dado muito diferentes: dados sintéticos de demonstração, replay de dados históricos (teste) e dados prospectivos de produção. Misturar seus KPIs ou tratá-los como equivalentes produz conclusões falsas — especialmente em uma demo construída sobre dataset histórico sem horizonte prospectivo observado (ver ADR-0009).

## Decisão

Toda predição, outcome e registro de batch carrega uma **origem** explícita:

- `demo_backfill` — dados sintéticos/de demonstração, usados para construir a interface e os fluxos;
- `historical_test` — replay de dados históricos com outcome conhecido, usado para avaliação/backfill segregado;
- `prospective` — dados prospectivos, com outcome observado ao longo do tempo (ADR-0005).

Regras:

- **KPIs separados por origem**: simulação, replay histórico e produção nunca compartilham métricas; relatórios e telas filtram por origem.
- **PATCH manual de outcome** registra autor e momento da observação; é uma ação administrativa auditável, não uma reescrita silenciosa.
- **Data administrativa ≠ data do evento**: registrar que alguém observou um churn hoje não muda a data em que o churn ocorreu.

## Alternativas rejeitadas

- **Origem implícita (inferida pelo tipo de dado)**: rejeitado — inferência silenciosa falha exatamente nos casos que mais importam (mistura de backfill com prospectivo).
- **KPI único global**: rejeitado — performance em replay histórico não é performance em produção; apresentar como tal é enganoso.

## Consequências

- A origem é coluna obrigatória nas tabelas de predição/outcome (issue #7) e filtro obrigatório em relatórios (issues #10, #16).
- Backfills históricos são sempre segregados de produção — o mesmo mecanismo de carga não serve aos dois sem marcar origem.
