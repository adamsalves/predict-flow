# ADR-0018: Contrato estável de explicações entre modelos

- Status: Aceito
- Data: 2026-11-18
- Supera parcialmente: ADR-0004, somente a consequência que previa formatos diferentes de explicação no frontend

## Contexto

LogisticRegression e RandomForest exigem estratégias de explicação diferentes, mas o frontend não deve conhecer o algoritmo ativo nem interpretar contribuições com unidades distintas como se fossem equivalentes. A ADR-0004 acertou ao condicionar o explainer ao modelo; sua consequência de expor dois formatos ao frontend criaria acoplamento desnecessário.

## Decisão

- O backend adapta LR e TreeExplainer para um DTO estável de fatores locais contendo, no mínimo, `feature`, `value` e `contribution`.
- A resposta informa método de explicação, versão do modelo, unidade da contribuição e demais metadados necessários para interpretação correta.
- Fatores são limitados a top-k, ordenados de forma determinística e persistidos com schema versionado.
- O frontend consome um único contrato e não possui branches baseadas em LR/RF nem acessa componentes internos do Pipeline.
- A unidade e o significado da contribuição são explícitos; valores de métodos/modelos diferentes não são comparados ou somados como se tivessem a mesma escala.
- A decisão não altera a seleção do modelo nem a exigência de explainer compatível da ADR-0004.
- SHAP e explicações lineares continuam preditivos, nunca causais, conforme ADR-0011.

## Alternativas rejeitadas

- **Dois formatos de UI escolhidos pelo algoritmo ativo**: rejeitado por vazar detalhes de ML para o frontend e multiplicar contratos.
- **DTO comum sem unidade/metadados**: rejeitado porque permitiria interpretações numericamente incorretas.
- **Expor diretamente a saída nativa do explainer**: rejeitada por acoplar API e persistência a detalhes de biblioteca.

## Consequências

- As issues #6 e #9 definem adapters de explicação e testes de contrato para ambos os modelos.
- As issues #13, #14, #20 e #25 consomem o mesmo DTO, sem lógica específica por algoritmo.
- A consequência da ADR-0004 sobre mudança de formato no frontend deixa de valer; as demais permanecem.
