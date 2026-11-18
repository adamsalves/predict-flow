# ADR-0014: Idempotência, transações e concorrência nas operações de domínio

- Status: Aceito
- Data: 2026-11-18

## Contexto

Predições, outcomes, ações, importações, batches e webhooks podem sofrer retries e atualizações concorrentes. Implementar cada caso isoladamente produziria semânticas divergentes, duplicidades ou perda silenciosa de alterações.

## Decisão

As operações mutáveis sujeitas a retry seguem uma política comum:

- A operação recebe uma chave de idempotência, com escopo definido por tipo de operação, autor e chave.
- A chave é associada a um hash canônico do payload. Reuso com o mesmo payload recupera o resultado/estado; reuso com payload diferente retorna conflito.
- Unicidade e constraints no PostgreSQL são a proteção final contra efeitos duplicados; locks de aplicação não substituem essas garantias.
- O estado e o resultado necessário ao retry são persistidos. Falhas após efeito parcial devem ser revertidas ou retomáveis, nunca apresentadas como sucesso ambíguo.
- Atualizações de outcome e ação usam versão/concorrência otimista: uma alteração baseada em versão obsoleta retorna conflito, sem lost update.
- Correções preservam histórico e auditoria; não reescrevem silenciosamente o passado.
- Services delimitam transações curtas. Inferência, SHAP, chamadas HTTP e processamento de arquivo não permanecem dentro de uma transação de banco.
- Webhooks usam outbox e entrega idempotente conforme a issue #21; marcar envio antes da confirmação não é permitido.

A implementação pode compartilhar primitivas de idempotência e concorrência, mas não exige um framework genérico para todos os casos.

## Alternativas rejeitadas

- **Idempotência apenas por ID gerado no frontend**: rejeitada porque não cobre retries, concorrência nem reuso indevido da chave.
- **Somente locks em memória**: rejeitados porque não sobrevivem a restart e não coordenam múltiplos processos.
- **Transação longa cobrindo inferência ou rede**: rejeitada por reter conexões/locks e ampliar contenção.
- **Last write wins em outcomes e ações**: rejeitado por perder correções e decisões concorrentes.

## Consequências

- As issues #9, #10, #18, #19, #21 e #22 referenciam a mesma política e testam retries, payloads divergentes e concorrência real.
- Services e migrations precisam ser desenhados juntos: constraints de banco fazem parte do comportamento do domínio.
- O resultado de retry deve ser observável e determinístico sem transformar uma nova intenção em replay da operação anterior.
