# ADR-0013: Contrato versionado de features entre modelo, API e frontend

- Status: Aceito
- Data: 2026-11-18

## Contexto

O manifesto do modelo define features, tipos e ordem, enquanto a API expõe schemas Pydantic e o frontend consome types TypeScript gerados. Se uma versão de modelo pudesse mudar features apenas por configuração, o artefato poderia ficar incompatível com a OpenAPI e com o frontend já construído.

## Decisão

O contrato de features é um contrato de software versionado, compartilhado por treino e serving:

- O manifesto contém `feature_schema_version` e um SHA-256 calculado sobre a representação canônica das features, tipos, obrigatoriedade e ordem.
- A API possui schemas Pydantic estáticos e registra a versão e o digest de schema que sabe atender.
- O `ModelLoader` valida versão e digest compatíveis antes de carregar o artefato; divergência impede a ativação.
- Ativação e rollback por configuração + restart são permitidos somente entre modelos compatíveis com o mesmo contrato de features.
- Mudança de features exige release coordenada de treino, backend, OpenAPI, types gerados e frontend quando houver impacto.
- O frontend não constrói formulários dinamicamente a partir de um manifesto desconhecido; usa o contrato OpenAPI gerado.
- Mudanças aditivas e breaking changes seguem a política de compatibilidade da ADR-0007; uma quebra de contrato exige versionamento deliberado da API.

A ordem das features continua sendo um contrato do modelo, mas a ordem de campos JSON não é usada como mecanismo de serialização da requisição.

## Alternativas rejeitadas

- **Ativar qualquer modelo apontando apenas `MODEL_PATH`**: rejeitado porque pode quebrar silenciosamente o contrato de API.
- **Schema de request inteiramente dinâmico baseado no manifesto**: rejeitado porque enfraquece a OpenAPI, o codegen e a validação estática.
- **Frontend com schemas próprios ou formulários orientados pelo algoritmo**: rejeitado por duplicação e acoplamento ao artefato.

## Consequências

- As issues #5/#6 incluem versão e digest do schema no manifesto; #8/#9 verificam compatibilidade no startup.
- A #11 passa a tratar o contrato de features como parte explícita da compatibilidade OpenAPI.
- Rollbacks permanecem simples dentro do mesmo schema; mudanças de schema deixam de ser operações puramente configuracionais.
