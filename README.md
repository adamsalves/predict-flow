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
- `curl` (para os health checks do quickstart)

## Quickstart

```bash
git clone https://github.com/adamsalves/predict-flow.git
cd predict-flow
docker compose up --build -d
```

Aguarde o backend ficar pronto (o primeiro boot leva alguns segundos) e verifique os health checks:

```bash
until curl -sf http://127.0.0.1:8000/api/v1/health/live; do sleep 2; done  # interrompa com Ctrl+C se passar de ~1 min

curl http://127.0.0.1:8000/api/v1/health/live    # sempre 200
curl http://127.0.0.1:8000/api/v1/health/ready   # 200 com Postgres, 503 sem
```

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

## Testes

Com Python 3.12+:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r backend/requirements-dev.txt -r backend/requirements-train.txt
pytest
```

Os locks de dev e treino são instalados juntos: os testes de `training/` exigem `pandas`, que o lock de treino pinna explicitamente (no runtime ele chega transitivamente via `shap`, usado nas explicações do serving — ADR-0017).

Sem Postgres disponível, o teste de health contra DB real é pulado automaticamente; com Postgres de pé (ex.: `docker compose up -d db`), ele executa de verdade. O teste de download do dataset Telco também se auto-pula offline.

## Treino do modelo (seleção LR/RF)

O treino executa a seleção disciplinada de modelo (ADR-0004): a LogisticRegression baseline e os desafiantes RandomForest são treinados no mesmo Pipeline sklearn único, com splits estratificados por target e disjuntos por cliente, e comparados somente na validação — PR-AUC como métrica primária com incerteza bootstrap, calibração (Brier) como critério complementar. Em empate prático ou derrota, o baseline é mantido. A calibração é ajustada num sub-split de calibração extraído do treino e o método escolhido na validação; o threshold operacional (provisório, F1-máx na validação) também vem da validação; o teste é avaliado uma única vez, após o congelamento (ADR-0009). O manifesto completo registra comparativo, decisão e limitações (ADR-0002, ADR-0013, ADR-0017, ADR-0018):

```bash
python -m training.train
```

Os artefatos vão para `models/<version>/` (gitignored, ADR-0017): `model.joblib` + `model_meta.json`, com seed, commit, SHA-256 do dataset, licença, digests dos splits, versão e digest dos contratos de features e de fatores, comparativo do gate, calibração, threshold, métricas com incerteza e orçamento de treino. Ao final, os digests da versão são publicados em `models/publication_registry.json` — a fonte confiável separada do diretório do artefato (ADR-0017): a ativação (issue #8) exigirá o digest aprovado por esse registro, e a substituição conjunta de artefato+manifesto é rejeitada.

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
