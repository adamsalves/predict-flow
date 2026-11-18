# ADR-0019: Arquitetura do frontend por features e qualidade verificável

- Status: Aceito
- Data: 2026-11-18

## Contexto

O frontend será um app Angular standalone de análise e operação. Organização por camadas técnicas ou estado global prematuro tenderia a espalhar cada jornada e criar abstrações antes das necessidades. Codegen e testes unitários, isoladamente, também não provam compatibilidade real nem acessibilidade da jornada.

## Decisão

- A estrutura é orientada por features: `worklist`, `customer-detail` e `sandbox`; componentes e regras permanecem perto da jornada que os utiliza.
- `core/` contém somente infraestrutura singleton, como interceptors, configuração e tratamento de erros. A regra de `shared/` somente no terceiro uso da ADR-0008 permanece.
- Angular Signals e estado local/server-side explícito são o padrão inicial. NgRx ou outro store global exige necessidade observada e nova decisão.
- Services HTTP usam os types gerados da OpenAPI; DTOs de API não são duplicados manualmente.
- Pesquisas, filtros e navegação cancelam ou descartam respostas obsoletas, evitando atualizações fora de ordem.
- Loading, erro, vazio e sucesso continuam first-class. Dados antigos, quando exibidos durante refresh, são identificados como tal.
- WCAG 2.2 AA é a referência de acessibilidade. Teclado, foco, contraste, anúncios assíncronos e alternativa textual a gráficos/gauges são verificáveis.
- Consumer tests incluem execução contra a API real; codegen e mocks de HTTP não são considerados prova suficiente de compatibilidade.
- Um teste e2e cobre a jornada crítica `fila → detalhe → ação/outcome` antes do fechamento de M4.
- Guards do frontend melhoram navegação, mas nunca substituem autenticação/autorização do backend.

## Alternativas rejeitadas

- **Estrutura global por `components/`, `services/` e `models/`**: rejeitada por separar arquivos da mesma jornada e incentivar acoplamento.
- **NgRx desde o scaffold**: rejeitado por complexidade sem um problema de estado demonstrado.
- **Somente testes unitários com HTTP mockado**: rejeitados porque não detectam divergência real de contrato nem falhas da jornada integrada.
- **Acessibilidade sem padrão verificável**: rejeitada por transformar o requisito em avaliação subjetiva.

## Consequências

- A #12 estabelece estrutura, convenções e WCAG 2.2 AA; #11 adiciona teste de contrato real.
- Uma issue e2e específica de M4 fecha a validação da jornada vertical.
- A arquitetura permanece deliberadamente simples e pode evoluir por evidência sem antecipar infraestrutura de estado global.
