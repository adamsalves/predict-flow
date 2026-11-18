# ADR-0012: Modelo de execução síncrono do backend

- Status: Aceito
- Data: 2026-11-18

## Contexto

A fundação usa SQLAlchemy síncrono e endpoints `def`, enquanto o planejamento inicial mencionava sessão assíncrona e inferência em executor. Misturar os dois modelos sem uma regra explícita espalharia chamadas bloqueantes pelo event loop ou exigiria adaptadores em cada endpoint. Async, por si só, não melhora inferência CPU-bound nem constitui uma boa prática universal.

## Decisão

Na demo local, o backend usa **SQLAlchemy síncrono e endpoints de domínio `def`**:

- O FastAPI executa esses endpoints em threadpool; não se converte a aplicação para async apenas por convenção.
- Engine pode ser compartilhada, mas a sessão é criada por request/operação e injetada por dependency; sessões não são globais.
- Services de aplicação delimitam transações; rotas permanecem finas e não controlam commits isolados.
- Inferência e geração de explicações são síncronas e possuem limite explícito de concorrência para proteger a threadpool e a latência da API.
- Nenhuma chamada síncrona de banco, modelo ou filesystem é introduzida em endpoint `async def`.
- Modelo é carregado uma única vez no lifespan, conforme ADR-0003 e issue #8.
- Batch confiável não compartilha o ciclo de vida do request web; ver ADR-0016.

Uma migração para async exige evidência de necessidade, revisão de concorrência e nova ADR.

## Alternativas rejeitadas

- **Async em toda a aplicação desde a demo**: rejeitado por adicionar complexidade sem benefício demonstrado para o volume local e por não resolver trabalho CPU-bound.
- **Mistura ad hoc de endpoints sync/async**: rejeitada por tornar o comportamento de I/O e concorrência imprevisível.
- **Sessão global do SQLAlchemy**: rejeitada por vazamento de estado entre requests e operações concorrentes.

## Consequências

- A issue #7 passa a exigir engine/sessão síncronas; a #9 mantém endpoint `def` e concorrência limitada.
- Testes de concorrência devem exercitar requests e operações reais, não apenas mocks do ORM.
- A decisão simplifica a demo, mas não estabelece uma garantia de escala para o piloto; capacidade continua sendo medida na #15.
