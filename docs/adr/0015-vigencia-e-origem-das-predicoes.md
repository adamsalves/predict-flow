# ADR-0015: Vigência das predições e isolamento operacional por origem

- Status: Aceito
- Data: 2026-11-18

## Contexto

O histórico de predições é imutável, mas a worklist precisa de uma predição vigente por cliente. Usar apenas o maior `score_at` fica ambíguo com retries, batches concorrentes e timestamps iguais. Além disso, uma única vigência misturaria `demo_backfill`, `historical_test` e `prospective`, violando a separação da ADR-0006.

## Decisão

- Predições permanecem em histórico append-only; uma nova predição não apaga nem sobrescreve as anteriores.
- A projeção de predição vigente é explícita e tem escopo **por cliente e `data_origin`**.
- O banco impede duas predições vigentes no mesmo escopo, por tabela de projeção ou constraint equivalente.
- A ativação da projeção ocorre de forma consistente com o batch. Batch incompleto ou falho não substitui a vigência anterior nem é apresentado como completo.
- Desempates usam identificadores estáveis; vigência não depende somente de timestamp.
- Ações e decisões permanecem vinculadas à predição que as motivou, mesmo quando um batch posterior muda o ranking ou a vigência.
- Predições anônimas de sandbox não possuem projeção vigente e não entram na worklist.
- Projeções operacionais de clientes/serviços e upserts de importação carregam origem e vigência; uma origem não sobrescreve silenciosamente o estado operacional de outra.
- Consultas e KPIs continuam filtrando origem no servidor, não apenas por elementos visuais do frontend.

## Alternativas rejeitadas

- **Última predição por `score_at` sem projeção/controlador de vigência**: rejeitado por ambiguidade e condições de corrida.
- **Uma única predição vigente por cliente entre todas as origens**: rejeitada por misturar simulação, replay histórico e produção.
- **Re-ranking que desvincula ações antigas**: rejeitado por apagar o contexto da decisão e impedir auditoria.

## Consequências

- A #7 deve modelar projeção, constraints, índices e snapshots com origem; #18 controla ativação do batch.
- As issues #19/#22 preservam vínculos históricos e isolamento de upserts.
- A solução admite evolução para cursor/snapshot de worklist se a paginação sob batches concorrentes se tornar um problema medido.
