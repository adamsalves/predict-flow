# AGENTS.md — predict-flow

Convenções e contexto deste repositório para humanos e agentes (Kimi Code, Claude Code etc.). Decisões arquiteturais e científicas de fundo estão em [`docs/adr/`](docs/adr/README.md) — ali é a fonte da verdade; aqui é o guia de trabalho do dia a dia.

## Visão geral

Monorepo de predição de churn / retenção de clientes, com explicabilidade e avaliação disciplinada.

- **Backend**: FastAPI (Python 3.12), pydantic-settings, SQLAlchemy síncrono (ADR-0012), Alembic, pytest. Estrutura em `backend/app/`: `api/v1/` (rotas), `core/` (config), `db/` (engine/sessão), `models/` (modelos de domínio SQLAlchemy), `schemas/` (DTOs e contratos), `services/` (regras de negócio e fronteiras transacionais).
- **Frontend**: Angular standalone + PrimeNG (scaffold a partir da #12, milestone M4). Types gerados via OpenAPI codegen (ADR-0007).
- **Pipeline ML**: seleção LR/RF, explicabilidade (SHAP), champion/challenger — fases M2+.
- **Ambiente**: demo local single-tenant com Docker Compose (backend + Postgres). Não está pronto para dados reais de clientes (ADR-0010).
- **Roadmap**: milestones M1–M6 rastreados nas [issues](https://github.com/adamsalves/predict-flow/issues) e nos milestones do repositório.

## Comandos padrão

Setup e execução local (Docker). O backend **aborta o startup sem um modelo publicado e verificado** (ADR-0003/0017) — treine e faça o staging antes do primeiro `up`:

```bash
python -m training.train              # treina e publica models/<version> + registry (venv com os locks)
mkdir -p ml-artifacts && cp -r models/v1 ml-artifacts/ && cp models/publication_registry.json ml-artifacts/
docker compose up --build -d          # sobe backend + Postgres (volume ./ml-artifacts:/models:ro)
docker compose exec backend alembic upgrade head   # aplica as migrações do schema de domínio
curl http://127.0.0.1:8000/api/v1/health/live   # 200
curl http://127.0.0.1:8000/api/v1/health/ready  # 200 com Postgres, 503 sem
curl http://127.0.0.1:8000/api/v1/model/info    # versão, digests, contrato e métricas do modelo ativo
docker compose down                   # para e remove containers (volume do Postgres persiste)
```

Ativação/rollback do modelo: `MODEL_VERSION=<versão>` + restart do backend (somente entre versões compatíveis com o contrato de features — ver README, "Modelo ativo").

Testes (CI = ruff + mypy + pytest com Postgres, ver `.github/workflows/ci.yml`):

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r backend/requirements-dev.txt -r backend/requirements-train.txt
pytest
```

Os locks dev e train são instalados juntos: os testes de `training/` exigem `pandas` — que também é dependência explícita do runtime (o serving monta o frame do `predict_proba`), com o mesmo pin nos dois locks (ADR-0017). Sem Postgres disponível, o teste de health contra DB real é pulado automaticamente; com `docker compose up -d db` de pé, ele executa de verdade. O teste de download do dataset Telco se auto-pula offline, assim como o smoke de serving com artefato real (`backend/tests/test_model_loader_smoke.py`), que a CI roda após o smoke de treino via `PF_SMOKE_MODELS_DIR`/`PF_SMOKE_MODEL_VERSION`.

Treino com seleção LR/RF (Pipeline único, gate na validação, manifesto completo — ver README):

```bash
python -m training.train
```

Qualidade (também gate da CI), tudo da raiz do monorepo:

```bash
ruff check .    # lint (pyproject.toml na raiz: E/F/I/UP/PGH003)
mypy            # typecheck strict, sem Any explícito; cobre backend/ e training/
pytest
python scripts/check_dependency_parity.py   # paridade treino/serving (ADR-0017)
bash scripts/check_artifact_ignores.sh      # artefatos/datasets fora do git e da imagem
```

## Convenções de código

### Idioma

- **Código 100% em inglês**: identificadores, nomes de arquivos, comentários e docstrings.
- **Exceção — labels de UI em português** (textos de interface, botões, mensagens voltadas ao usuário).
- Documentação (README, ADRs, docs/), mensagens de commit, títulos e descrições de PR seguem em **pt-BR**.

### Tipagem honesta (sem desvios)

- **TypeScript**: proibido `any` e casts que contornam o tipo (`as unknown as X`). Se o tipo é desconhecido, modele-o.
- **Python**: type hints reais em funções públicas; proibido `Any` e `# type: ignore` sem justificativa registrada em "Decisões e desvios" do PR.

### Backend (Python)

- Rotas em `app/api/v1/`, uma por domínio, prefixo comum `/api/v1`; regras de negócio em `services/`, nunca na rota.
- Convenções transversais (issue #40) são consumidas, nunca reimplementadas: erros saem no envelope comum via handlers globais (`app/api/errors.py` — services levantam `DomainError` e subclasses, nunca `HTTPException`); routers declaram `COMMON_ERROR_RESPONSES`; listagens usam `page_params` + `paginate`; `request_id` e log de acesso JSON são automáticos (middleware).
- Config via pydantic-settings (`app/core/config.py`), nunca hardcoded; credenciais de demo local não são segredos (ADR-0010).
- Um teste pytest para cada comportamento novo; bugfix vem com teste de regressão que falha antes da correção.
- Código, comentários e docstrings em inglês (ver "Idioma").

### Frontend (Angular, a partir do M4)

- Componentes standalone; tipos gerados do OpenAPI (ADR-0007), não escritos à mão.
- Labels e textos de interface em português; código, comentários e identificadores em inglês.

## Regras de contribuição

- **PRs pequenos** (~400 linhas no máximo, sem lockfile/fixture/snapshot); maior que isso, quebre em issues.
- **Título em Conventional Commits** (`feat:`, `fix:`, `chore:`, `ci:`, `docs:` ...) em pt-BR.
- **Labels da issue aplicadas ao PR**; squash merge no `main`.
- **CI verde é obrigatório** — sem exceção.
- **Critérios de aceite conferidos item a item** no PR (✅ atendido ou ⚠️ exceção com justificativa em "Decisões e desvios"); nenhum item sem marcação. Itens atendidos são marcados `- [x]` na própria issue — ela é a fonte da verdade do status.
- **Retorno de revisão**: após achados, comentário ponto a ponto no PR (achado → resolução → commit; sugestões não adotadas justificadas) antes do merge.
- **Nunca mergear sem confirmação explícita do mantenedor.**
- Decisão arquitetural ou científica nova → ADR em `docs/adr/` no mesmo PR.

## Releases

Cada milestone fecha com uma release tagueada: semver por milestone (M1 → `v0.1.0` ... M5 → `v1.0.0`, patch para hotfix), disparada manualmente via workflow `release` (Actions), que exige CI verde, atualiza o `CHANGELOG.md` e publica a GitHub Release. Detalhes em [README](README.md#versões-e-releases).

## Contexto local de agentes

O diretório `.kimi-code/` é gitignored e guarda contexto local do projeto (memória entre sessões, notas de trabalho). Nunca commitar. Convenções compartilhadas ficam aqui neste arquivo.
