# ADR-0002: Metadados e manifesto do modelo

- Status: Aceito
- Data: 2026-11-16

## Contexto

Ativar, auditar e comparar versões de modelo exige saber com precisão o que foi treinado, com quais dados, em qual commit e com quais métricas. Sem um manifesto padronizado, a versão ativa é uma caixa-preta: não é possível garantir reprodutibilidade nem distinguir modelos entre si com segurança.

## Decisão

Cada versão de modelo é acompanhada de um `model_meta.json` — manifesto **gerado no treino**, nunca editado à mão — contendo:

- `schema_version` do manifesto e `model_version` do artefato;
- SHA-256 do `model.joblib` e do dataset de treino;
- commit do repositório, seed e definição dos splits (ver ADR-0009);
- features com tipo e **ordem exata** de entrada;
- threshold de classificação e parâmetros de calibração;
- métricas de validação/teste;
- versão de Python e dependências (com versões pinadas).

## Alternativas rejeitadas

- **Metadados apenas no código ou em doc solto**: rejeitado — fica fora do artefato versionado e não acompanha o modelo no rollback (ADR-0003).
- **Model registry externo** (ex.: MLflow Model Registry): rejeitado para a v1 pela mesma razão da ADR-0003 — infraestrutura desproporcional à demo local.

## Consequências

- Ativação e rollback por configuração (ADR-0003) dependem da verificação de integridade contra este manifesto.
- O pipeline de treino passa a ter a obrigação de emitir o manifesto completo a cada versão.
- A ordem de features no manifesto é contrato: o backend valida a entrada contra ela.
