# ADR-0007: Contrato OpenAPI e codegen TypeScript

- Status: Aceito (decidido)
- Data: 2026-11-16

## Contexto

API FastAPI e frontend Angular são desenvolvidos em paralelo. Sem contrato de tipos compartilhado, o frontend cria interfaces à mão a partir de chamadas de exemplo — que divergem da API silenciosamente a cada mudança, quebrando telas em runtime.

## Decisão

- A OpenAPI gerada pela FastAPI é o **contrato oficial** entre backend e frontend.
- O frontend **não escreve interfaces de API à mão**: os types TypeScript são **gerados por codegen** a partir da OpenAPI (implementação na issue #11).
- O CI **verifica breaking changes** no contrato: mudanças incompatíveis falham o check.
- **Consumer tests** validam que o frontend consome o contrato gerado — contrato quebrado é detectado antes do merge.

## Alternativas rejeitadas

- **Types manuais no frontend "mantidos com cuidado"**: rejeitado — divergência silenciosa é questão de quando, não se.
- **Schema-first (OpenAPI escrita à mão e FastAPI gerada a partir dela)**: rejeitado — dobra a manutenção e remove a fonte única; a docstring/schemas da FastAPI já produzem a OpenAPI de forma nativa.

## Consequências

- Mudanças de contrato passam a ser deliberadas: quebrar a API exige atualizar contrato, codegen e consumer tests no mesmo PR.
- A issue #11 é a implementação desta ADR; a checagem de codegen entra no CI evoluído da issue #3.
