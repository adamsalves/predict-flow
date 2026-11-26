# predict-flow

Predição de churn e suporte a decisões de retenção, com explicabilidade e avaliação disciplinada.

> **⚠️ Estado atual: demo local de desenvolvimento.** Este ambiente é single-tenant, roda apenas com dados públicos ou sintéticos e **não está pronto para dados reais de clientes** (ver [ADR-0010](docs/adr/0010-demo-versus-piloto.md)). Dados reais exigem o gate de piloto (#17).

## Limites da evidência — leia antes de qualquer número

- O dataset Telco sustenta **classificação histórica sem horizonte prospectivo observado**. Não se apresenta "churn em 90 dias" nem eficácia de retenção a partir dele (ver [ADR-0009](docs/adr/0009-target-temporalidade-e-avaliacao.md)).
- Simulação (`demo_backfill`), replay histórico (`historical_test`) e dados prospectivos são origens **separadas**, com KPIs que não se misturam ([ADR-0006](docs/adr/0006-feedback-e-origem-dos-dados.md)).
- SHAP é explicação preditiva, **não causal**. Nenhuma métrica de ação é apresentada como ROI ou retenção incremental sem desenho de comparação adequado, preferencialmente randomizado ([ADR-0011](docs/adr/0011-politica-de-decisao-e-acoes.md)).
- Nenhuma métrica ou meta é publicada neste README sem metodologia e origem registradas.

## Visão geral

O eixo do MVP: `importar → pontuar → priorizar → abrir cliente → decidir/registrar ação → observar outcome → avaliar`. O formulário de inferência é apenas sandbox.

As decisões arquiteturais e científicas estão registradas em [`docs/adr/`](docs/adr/README.md) — fonte da verdade do projeto (19 ADRs; a ADR-0004 é parcialmente superada pela ADR-0018).

## Pré-requisitos

- Docker com Docker Compose (v2)
- Python 3.12+ (para treinar o modelo antes do primeiro boot)
- `curl` (para os health checks do quickstart)

## Quickstart

```bash
git clone https://github.com/adamsalves/predict-flow.git
cd predict-flow
```

O backend **só sobe com um modelo publicado e verificado** (ADR-0003/ADR-0017): treine uma vez e faça o staging explícito dos artefatos para o volume de serving:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r backend/requirements-dev.txt -r backend/requirements-train.txt
python -m training.train   # treina e publica models/v1 + publication_registry.json

mkdir -p ml-artifacts
cp -r models/v1 ml-artifacts/
cp models/publication_registry.json ml-artifacts/
```

Suba o ambiente:

```bash
docker compose up --build -d
```

Aguarde o backend ficar pronto (o primeiro boot leva alguns segundos) e verifique os health checks e o modelo ativo:

```bash
until curl -sf http://127.0.0.1:8000/api/v1/health/live; do sleep 2; done  # interrompa com Ctrl+C se passar de ~1 min

curl http://127.0.0.1:8000/api/v1/health/live    # sempre 200
curl http://127.0.0.1:8000/api/v1/health/ready   # 200 com Postgres, 503 sem
curl http://127.0.0.1:8000/api/v1/model/info     # versão, digests, contrato e métricas do modelo ativo
```

Se o artefato, o manifesto ou o registro confiável estiverem ausentes ou adulterados, o backend **aborta o startup** com erro claro em vez de servir sem modelo verificado (`docker compose logs backend` mostra o motivo).

Para parar e remover os containers (o volume de dados do Postgres persiste):

```bash
docker compose down
```

Credenciais e portas são defaults de desenvolvimento local apenas (`127.0.0.1`), conforme ADR-0010. Para sobrescrevê-las, use variáveis de ambiente (`POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB`) ou um `backend/.env` (ignorado pelo git).

## Banco de dados (migrações)

O schema de domínio (clientes, serviços e predições — ADR-0005/0006/0015) é versionado com Alembic em `backend/alembic/`. Com o ambiente do Quickstart de pé, aplique as migrações dentro do container:

```bash
docker compose exec backend alembic upgrade head
```

Localmente (venv ativa, Postgres acessível em `DATABASE_URL`):

```bash
cd backend
alembic upgrade head     # aplica as migrações pendentes
alembic downgrade base   # desfaz tudo (comportamento definido e testado)
```

A URL de conexão vem sempre da configuração da aplicação (`DATABASE_URL`) — nunca é fixada no `alembic.ini`. A suíte de testes cria bancos temporários no Postgres real, aplica a migração neles e valida constraints, índices e concorrência contra o banco (sem mocks do ORM); sem Postgres, esses testes se auto-pulam.

## Convenções transversais da API

Estabelecidas na issue #40 (ADR-0007/0012/0014), valem para todo endpoint de domínio — nenhum router reimplementa estas peças:

- **Envelope de erro único**: validação (422), domínio (400), concorrência otimista (409), HTTP e falha inesperada (500) saem no mesmo formato — `{"error": {"code", "message", "request_id", "details?"}}` — documentado no OpenAPI. A resposta nunca carrega payload, features, segredos ou stack trace; detalhes de validação omitem o valor de entrada (PII).
- **`X-Request-ID`**: gerado (UUID) ou propagado quando seguro (curto, charset log-safe); a resposta e a linha de log compartilham o mesmo id.
- **Logs JSON de acesso**: uma linha por request com método, rota (template com parâmetros mascarados — nunca o path cru, que pode conter PII), status e latência. Payloads, headers e query strings nunca são logados.
- **Sessão por request** (ADR-0012): a dependency `get_session` cria uma sessão por request/operação e sempre a fecha; services delimitam transações e o fechamento sem commit reverte o trabalho pendente.
- **Paginação**: `limit` (1–100, default 20) e `offset` via dependency `page_params`; `paginate` aplica ordenação estável com desempate único (chave primária) e retorna o total. Páginas não repetem nem omitem linhas sob escritas concorrentes.
- **Readiness limitada**: `/health/ready` sonda o banco com `connect_timeout` + `statement_timeout` derivados de `DB_READINESS_TIMEOUT_SECONDS` (default 2s) — banco lento ou inalcançável retorna 503 no envelope em tempo limitado.

## Predição: `POST /api/v1/predict` (issue #9)

Inferência síncrona persistida, com auditoria mínima e retry seguro:

- **Request estático** com as 19 features do contrato (tipos validados estritamente — string onde o contrato diz `int` é 422, não coerção silenciosa; ADR-0013) + `customer_id_ext` opcional — ausente marca sandbox anônima, que nunca entra na projeção vigente nem na worklist.
- **`Idempotency-Key` obrigatório** (ADR-0014): mesma chave com payload equivalente → **200** com o resultado gravado, sem duplicar; mesma chave com payload diferente → **409**; nova predição intencional usa chave nova e preserva o histórico (append-only).
- **Resposta 201** com probabilidade, `risk_level` provisório (threshold versionado do manifesto), explicação no DTO estável com método/unidade (ADR-0018), `model_version`, `score_at` e a limitação do target. Scores da demo são **históricos/simulados** — nunca "churn em N dias" (ADR-0009); `feature_as_of` permanece ausente de forma explícita.
- **Concorrência limitada** por `PREDICT_MAX_CONCURRENCY` (default 2, `ge=1`): excedente falha rápido com **429** em vez de enfileirar na threadpool; inferência e SHAP rodam **fora de qualquer sessão/transação de banco** (ADR-0012/ADR-0014) — endpoint `def`, nunca `async def`.
- **Falha de persistência após inferência nunca retorna sucesso auditado**: o erro propaga no envelope comum e o retry com a mesma chave é seguro (inferência determinística).

Exemplo contra o ambiente do Quickstart:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/predict \
  -H 'Content-Type: application/json' \
  -H 'Idempotency-Key: demo-0001' \
  -d '{
    "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
    "tenure": 12, "PhoneService": "Yes", "MultipleLines": "No",
    "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "Yes",
    "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes",
    "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check", "MonthlyCharges": 89.5, "TotalCharges": 1074.0
  }'
```

## Testes

Com Python 3.12+:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r backend/requirements-dev.txt -r backend/requirements-train.txt
pytest
```

Os locks de dev e treino são instalados juntos: os testes de `training/` exigem `pandas`, que ambos os locks pinnam explicitamente (no runtime ele também é dependência direta — o serving monta o frame de entrada do `predict_proba` — e chega ao resolver pela mesma versão, com paridade verificada — ADR-0017).

Sem Postgres disponível, o teste de health contra DB real é pulado automaticamente; com Postgres de pé (ex.: `docker compose up -d db`), ele executa de verdade. O teste de download do dataset Telco também se auto-pula offline.

## Treino do modelo (seleção LR/RF)

O treino executa a seleção disciplinada de modelo (ADR-0004): a LogisticRegression baseline e os desafiantes RandomForest são treinados no mesmo Pipeline sklearn único, com splits estratificados por target e disjuntos por cliente, e comparados somente na validação — PR-AUC como métrica primária com incerteza bootstrap, calibração (Brier) como critério complementar. Em empate prático ou derrota, o baseline é mantido. A calibração é ajustada num sub-split de calibração extraído do treino e o método escolhido na validação; o threshold operacional (provisório, F1-máx na validação) também vem da validação; o teste é avaliado uma única vez, após o congelamento (ADR-0009). O manifesto completo registra comparativo, decisão e limitações (ADR-0002, ADR-0013, ADR-0017, ADR-0018):

```bash
python -m training.train
```

Os artefatos vão para `models/<version>/` (gitignored, ADR-0017): `model.joblib` + `model_meta.json`, com seed, commit, SHA-256 do dataset, licença, digests dos splits, versão e digest dos contratos de features e de fatores, comparativo do gate, calibração, threshold, métricas com incerteza e orçamento de treino. Ao final, os digests da versão são publicados em `models/publication_registry.json` — a fonte confiável separada do diretório do artefato (ADR-0017): a ativação exige o digest aprovado por esse registro, e a substituição conjunta de artefato+manifesto é rejeitada.

## Modelo ativo: ativação, verificação e rollback

O serving consome artefatos por um volume **somente leitura** (`./ml-artifacts:/models:ro`). O staging — copiar a versão e o registro confiável de `models/` para `ml-artifacts/` — é o processo explícito e auditável que publica uma versão para o serving (ADR-0003); o backend nunca escreve no volume.

No startup, o `ModelLoader` (`backend/app/services/model_loader.py`) executa os gates **nesta ordem**, todos antes do `joblib.load`:

1. presença do diretório da versão, do artefato e do manifesto;
2. procedência: SHA-256 do artefato e do manifesto conferidos contra o registro confiável (`publication_registry.json`, fora do diretório da versão) — substituição conjunta artefato+manifesto é rejeitada aqui;
3. estrutura do manifesto (versão de schema suportada, identidade da versão);
4. contratos: versão/digest/lista de features (ADR-0013), digest do schema de fatores (ADR-0018) e encoding do target (`No=0`, `Yes=1`) idênticos aos compilados na API;
5. ambiente: scikit-learn exatamente igual ao do treino e numpy/pandas no mesmo major (a paridade dos locks já é garantida em build — estes gates são a armadilha de runtime para artefatos treinados fora);
6. consistência: o digest de artefato declarado no manifesto confere com o registro confiável;
7. somente então `joblib.load`, e o objeto carregado precisa ser um scorer probabilístico com `classes_ == [0, 1]` — a classe positiva (churn) na posição de probabilidade que o serving lê.

A versão ativa é configuração (`MODEL_VERSION`, default `v1`), com digest confiável opcionalmente apontado por `MODEL_REGISTRY_PATH` (default: o registro na raiz do volume). `GET /api/v1/model/info` expõe versão, digests, contrato, threshold/calibração, métricas históricas e a limitação do target — sem caminhos locais nem dados sensíveis.

**Rollback** (ADR-0003): aponte `MODEL_VERSION` para a versão anterior e reinicie o backend:

```bash
MODEL_VERSION=v1 docker compose up -d --force-recreate backend
```

O rollback só é válido entre versões compatíveis com o contrato de features servido pela API — uma versão incompatível falha os gates e o startup aborta, como qualquer ativação inválida. API e futuros workers de batch usam exatamente a mesma validação (`load_active_model`).

As explicações locais seguem o DTO estável da ADR-0018 (`backend/app/services/explain.py`): contribuições lineares em log-odds para a LR e TreeSHAP em probabilidade para a RF, sempre top-k determinístico, reutilizando os componentes do Pipeline — preditivas, nunca causais (ADR-0011).

**Dataset**: IBM Telco Customer Churn — 7.043 clientes, um registro histórico por cliente, distribuído no repositório [`IBM/telco-customer-churn-on-icp4d`](https://github.com/IBM/telco-customer-churn-on-icp4d) sob Apache License 2.0. O SHA-256 do arquivo é fixado em `training/dataset.py` e verificado em toda execução.

**Rótulo histórico**: o manifesto e a saída do treino descrevem o Telco explicitamente como histórico, sem horizonte prospectivo — os scores são probabilidades do rótulo histórico, não "churn em X dias" (ADR-0009).

## Qualidade e locks de dependências

Os ambientes são reprodutíveis a partir de locks compilados com pip-tools (ADR-0017), em `backend/`:

- `requirements.txt` — runtime de produção (é o que a imagem Docker instala, sem deps de teste);
- `requirements-dev.txt` — runtime + testes, ruff, mypy e pip-tools;
- `requirements-train.txt` — runtime + dependências de treino.

scikit-learn, joblib e shap são bibliotecas compartilhadas entre treino e serving (o treino serializa `model.joblib` e o serving carrega via `joblib.load` e gera explicações com SHAP): as versões exatas vivem no lock de runtime e a paridade entre treino e serving é verificada automaticamente.

Para atualizar os locks após mudar um `.in`, compile arquivo por arquivo com a mesma versão do Python da CI e da imagem (3.12). O `click<8.2` é pinado porque o pip-tools 7.5.1 registra flags espúrias no cabeçalho dos locks com click 8.2+:

```bash
docker run --rm -v "$PWD/backend:/src" -w /src python:3.12-slim \
  sh -c 'pip install "pip-tools==7.5.1" "click<8.2" && pip-compile requirements.in && pip-compile requirements-dev.in && pip-compile requirements-train.in'
```

Qualidade local (gate da CI = ruff + mypy + pytest + checks de paridade e proteção), tudo da raiz do monorepo:

```bash
ruff check .
mypy
pytest
python scripts/check_dependency_parity.py
bash scripts/check_artifact_ignores.sh
```

## Estrutura do monorepo

Convenções de código, regras de contribuição e contexto para agentes de IA estão em [AGENTS.md](AGENTS.md).

```
predict-flow/
├── .github/workflows/   # CI (ruff, mypy, smoke de treino, pytest + Postgres) e release por milestone
├── backend/
│   ├── app/
│   │   ├── api/v1/      # rotas (health)
│   │   ├── core/        # config (pydantic-settings)
│   │   ├── db/          # engine/sessão
│   │   ├── models/      # (fase ML)
│   │   ├── schemas/     # (fase ML)
│   │   └── services/    # (fase ML)
│   ├── tests/           # pytest do backend
│   ├── Dockerfile
│   └── requirements*.{in,txt}  # locks pip-tools (runtime, dev, train)
├── training/            # ingestão, splits, pipeline, métricas, train CLI e seus testes
├── pyproject.toml       # ruff e mypy (raiz, cobre backend + training)
├── pytest.ini           # testpaths backend/tests + training/tests
├── scripts/             # checks de paridade de dependências e proteção de artefatos
├── docs/adr/            # decisões arquiteturais e científicas
└── docker-compose.yml   # backend + Postgres (demo local)
```

## Roadmap

Os milestones M1 (fundação) → M6 (extras) são rastreados nas [issues do GitHub](https://github.com/adamsalves/predict-flow/issues) e nos milestones do repositório.

## Versões e releases

- Versionamento semver atrelado aos milestones: cada milestone fecha um minor — M1 → `v0.1.0`, M2 → `v0.2.0`, ..., M5 (piloto) → `v1.0.0`; hotfixes dentro de um milestone usam patch (`v0.1.1`).
- A release é uma decisão de fechamento de milestone, não automática por merge. Quando o milestone fecha, dispara-se manualmente o workflow **release** (Actions → release) informando a versão sem o `v`.
- O workflow só executa com CI verde na `main` (reusa o workflow `ci`), cria a tag anotada `v<versão>` no HEAD da `main`, atualiza o [CHANGELOG.md](CHANGELOG.md) (Keep a Changelog, commitado de volta no repo) e publica a GitHub Release com notas geradas dos squash merges (Conventional Commits) e das issues encerradas no milestone.
- Recuperação: se um run falhar após o push da tag mas antes da release, delete a tag (`git push origin :refs/tags/v<versão>`) e dispare de novo; se falhar após o commit do changelog, o rerun detecta a entrada existente e pula direto para a tag/release.

## Quando disponíveis

Estas etapas entram em milestones futuros e serão documentadas aqui quando existirem:

- **Migrations (Alembic)** — a partir da #7
- **Verificação de integridade do artefato** — a partir da #8 (usada pelo readiness na #15)
- **Frontend** — a partir da #12
