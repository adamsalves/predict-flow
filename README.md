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

## Testes

Com Python 3.12+:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r backend/requirements-dev.txt
cd backend
pytest
```

Sem Postgres disponível, o teste de health contra DB real é pulado automaticamente; com Postgres de pé (ex.: `docker compose up -d db`), ele executa de verdade.

## Qualidade e locks de dependências

Os ambientes são reprodutíveis a partir de locks compilados com pip-tools (ADR-0017), em `backend/`:

- `requirements.txt` — runtime de produção (é o que a imagem Docker instala, sem deps de teste);
- `requirements-dev.txt` — runtime + testes, ruff, mypy e pip-tools;
- `requirements-train.txt` — runtime + dependências de treino.

scikit-learn e joblib são bibliotecas compartilhadas entre treino e serving (o treino serializa `model.joblib` e o serving carrega via `joblib.load`): as versões exatas vivem no lock de runtime e a paridade entre treino e serving é verificada automaticamente.

Para atualizar os locks após mudar um `.in`, compile arquivo por arquivo com a mesma versão do Python da CI e da imagem (3.12):

```bash
docker run --rm -v "$PWD/backend:/src" -w /src python:3.12-slim \
  sh -c "pip install pip-tools && pip-compile requirements.in && pip-compile requirements-dev.in && pip-compile requirements-train.in"
```

Qualidade local (gate da CI = ruff + mypy + pytest + checks de paridade e proteção):

```bash
cd backend
ruff check .
mypy
pytest
cd ..
python scripts/check_dependency_parity.py
bash scripts/check_artifact_ignores.sh
```

## Estrutura do monorepo

Convenções de código, regras de contribuição e contexto para agentes de IA estão em [AGENTS.md](AGENTS.md).

```
predict-flow/
├── .github/workflows/   # CI (ruff, mypy, pytest + Postgres) e release por milestone
├── backend/
│   ├── app/
│   │   ├── api/v1/      # rotas (health)
│   │   ├── core/        # config (pydantic-settings)
│   │   ├── db/          # engine/sessão
│   │   ├── models/      # (fase ML)
│   │   ├── schemas/     # (fase ML)
│   │   └── services/    # (fase ML)
│   ├── tests/           # pytest
│   ├── Dockerfile
│   ├── pyproject.toml   # ruff e mypy
│   └── requirements*.{in,txt}  # locks pip-tools (runtime, dev, train)
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
- **Treino do pipeline ML** — a partir das issues de ML (M2+)
- **Verificação de integridade do artefato** — a partir da #8 (usada pelo readiness na #15)
- **Frontend** — a partir da #12
