# Changelog

Todas as mudanças notáveis deste projeto são documentadas aqui, no formato [Keep a Changelog](https://keepachangelog.com/pt-BR/1.1.0/). O versionamento é [semver atrelado aos milestones](README.md#versões-e-releases) do projeto.

## [v0.1.0] - 2026-11-17

### Alterado
- docs: AGENTS.md na raiz com convenções e contexto do projeto (#38)
- ci: workflow de release por milestone (tag, changelog, GitHub Release) (#37)
- chore: templates de PR e issues do GitHub (.github/) (#33)
- docs: README com setup, quickstart e limites da evidência (#31)
- ci: workflow de testes (pytest) com Postgres desde o primeiro PR (#30)
- chore: esqueleto do monorepo + docker-compose + health (#29)
- docs(adr): registrar as 11 ADRs arquiteturais (#27)
- chore: gitignore para .kimi-code/ (contexto local do projeto)
- Initial commit

### Issues encerradas
- [#1](https://github.com/adamsalves/predict-flow/issues/1) docs(adr): registrar as 11 ADRs arquiteturais
- [#2](https://github.com/adamsalves/predict-flow/issues/2) chore: esqueleto do monorepo + docker-compose + health
- [#3](https://github.com/adamsalves/predict-flow/issues/3) ci: workflow de testes (pytest) desde o primeiro PR
- [#4](https://github.com/adamsalves/predict-flow/issues/4) docs: README com setup e execução local
- [#26](https://github.com/adamsalves/predict-flow/issues/26) chore: AGENTS.md na raiz com convenções e contexto do projeto
- [#28](https://github.com/adamsalves/predict-flow/issues/28) chore: templates de PR e issues do GitHub (.github/)
- [#32](https://github.com/adamsalves/predict-flow/issues/32) chore: fluxo de release por milestone (tags, changelog, GitHub Release)

## [v0.2.0] - 2026-11-25

### Alterado
- feat(api): ModelLoader por volume + GET /api/v1/model/info (#49)
- feat(db): modelos SQLAlchemy + Alembic com colunas de outcome (#48)
- feat(ml): seleção LR/RF, explicabilidade e release versionado do modelo (#47)
- feat(ml): contrato científico, splits independentes + baseline LogisticRegression (#46)
- chore(ci): dependências reprodutíveis, lint/typecheck e proteção de artefatos (#45)
- docs(adr): formalizar decisões da revisão de arquitetura (#43)

### Issues encerradas
- [#5](https://github.com/adamsalves/predict-flow/issues/5) feat(ml): contrato científico, splits independentes + baseline LogisticRegression
- [#6](https://github.com/adamsalves/predict-flow/issues/6) feat(ml): seleção LR/RF, explicabilidade e release versionado do modelo
- [#7](https://github.com/adamsalves/predict-flow/issues/7) feat(db): modelos SQLAlchemy + Alembic com colunas de outcome
- [#8](https://github.com/adamsalves/predict-flow/issues/8) feat(api): ModelLoader por volume + GET /api/v1/model/info
- [#39](https://github.com/adamsalves/predict-flow/issues/39) chore(ci): dependências reprodutíveis, lint/typecheck e proteção de artefatos
- [#42](https://github.com/adamsalves/predict-flow/issues/42) docs(adr): formalizar decisões da revisão de arquitetura

<!-- novas releases são inseridas imediatamente abaixo deste marcador -->

[v0.1.0]: https://github.com/adamsalves/predict-flow/compare/303be8dfc031c037ab8026b4495beeedfaac847b...v0.1.0
[v0.2.0]: https://github.com/adamsalves/predict-flow/compare/v0.1.0...v0.2.0
<!-- compare-links -->
