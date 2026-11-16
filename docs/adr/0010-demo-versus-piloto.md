# ADR-0010: Demo versus piloto, segurança e ciclo de vida de dados

- Status: Aceito
- Data: 2026-11-16

## Contexto

O caminho até dados reais de clientes cruza uma fronteira clara: a demo local é uma ferramenta de desenvolvimento com dados públicos/sintéticos, enquanto o piloto lida com PII e decisões que afetam clientes reais. Confundir os dois estágios leva a segurança premature (atrapalhando a demo) ou a exposição de dados reais em ambiente sem controle (risco grave).

## Decisão

**Demo local (esta fase, issues #1–#16):**
- Sem JWT no caminho crítico: a demo roda local, com dados públicos ou sintéticos.
- Sem dados reais, em hipótese alguma, antes do gate de piloto.
- **v1 explicitamente single-tenant**: nenhum mecanismo de tenancy entra no desenho agora.

**Gate de piloto (issue #17) — condições para dados reais:**
- Autenticação e autorização em **todos** os endpoints — sem exceção de "endpoint interno".
- Roles com permissões definidas.
- Minimização e retenção de PII: coletar só o necessário, com política de retenção explícita.
- Auditoria das ações administrativas (ex.: PATCH manual de outcome, ADR-0006).
- Proteção específica dos vetores de maior risco: upload de CSV, execução de batch, alertas e endpoints administrativos.

## Alternativas rejeitadas

- **JWT desde a demo**: rejeitado — custo de desenvolvimento e operação sem benefício para dados sintéticos locais; adia a demo sem reduzir risco real.
- **Multi-tenancy antecipado "para quando precisar"**: rejeitado — complexidade estrutural (isolamento, billing, quotas) que a v1 não usa; virá como decisão própria se o piloto exigir.
- **Dados reais "só para testar" antes do gate**: rejeitado — o gate existe justamente porque esse é o ponto sem retorno fácil.

## Consequências

- Issues #2–#16 não carregam auth; a #17 é o portão deliberado que introduz segurança de produção de uma só vez, revisada como mudança crítica.
- Documentação e compose da demo devem deixar claro que o ambiente é local/demo, não pronto para dados reais (critério de aceite da #2).
