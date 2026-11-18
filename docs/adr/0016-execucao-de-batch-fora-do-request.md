# ADR-0016: Execução recuperável de batch fora do request web

- Status: Aceito
- Data: 2026-11-18

## Contexto

Batches precisam ser reproduzíveis, idempotentes e retomáveis após falhas. `BackgroundTasks` do FastAPI pertence ao processo do request e não oferece essas garantias em restart ou crash. Inferência e SHAP também podem consumir CPU suficiente para degradar a API.

## Decisão

- Treinamento continua fora da API.
- Pontuação em batch é executada por **CLI ou worker separado do processo Uvicorn**, reutilizando os mesmos services de domínio, contrato de features e artefato versionado da API.
- O estado do job, identidade, origem, versão do modelo, progresso e resultado ficam no PostgreSQL.
- O processamento usa chunks, transações curtas, checkpoints e informação suficiente para retomada segura.
- Retry do mesmo batch segue a ADR-0014; nova intenção cria nova identidade sem apagar histórico.
- Ativação da predição vigente segue a ADR-0015; falha parcial não equivale a conclusão.
- Se existir endpoint para iniciar batch, ele registra/enfileira o job; não executa o lote inteiro no request.
- A demo usa PostgreSQL e processo CLI/worker simples. Celery, Redis ou outro broker só entram com necessidade observada e nova decisão.
- A inferência web síncrona possui limite de concorrência, conforme ADR-0012, e não é o mecanismo de execução do batch.

## Alternativas rejeitadas

- **FastAPI BackgroundTasks para batch confiável**: rejeitado porque perde o controle do trabalho em restart/crash e compartilha recursos da API.
- **Duplicar a lógica de pontuação no worker**: rejeitado por drift entre inferência individual e batch.
- **Broker/fila distribuída desde a demo**: rejeitado por complexidade operacional sem evidência de necessidade.

## Consequências

- A #18 inclui entrypoint de CLI/worker, recuperação após encerramento do processo e testes de retomada.
- O compose poderá ter um serviço de worker ou um comando documentado de CLI; isso não cria um microsserviço de domínio.
- Métricas de duração, saturação e falhas de jobs entram na #15.
