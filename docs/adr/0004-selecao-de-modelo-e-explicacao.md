# ADR-0004: Seleção de modelo e explicação

- Status: Aceito
- Data: 2026-11-16

## Contexto

O pipeline precisa escolher entre candidatos de modelo de forma disciplinada — não shipar um modelo "porque já existia" — e precisa explicar os scores do modelo ativo. Há uma interação importante: a estratégia de explicação depende do tipo de modelo, então trocar de modelo pode exigir trocar de explainer.

## Decisão

- **LogisticRegression (LR) é o baseline** obrigatório do pipeline.
- **RandomForest (RF) é candidato, não modelo previamente shipado**: só entra em produção se vencer o gate.
- **Gate de seleção na validação**: métricas e critérios de gate definidos antes do treino (ver ADR-0009 para os splits); o vencedor é decidido na validação, nunca no teste.
- **Empate → modelo mais simples**: em empate, vence a LR.
- **Derrota/falha no gate → fallback para LR**: a LR baseline é sempre treinada e disponível como fallback.
- **Explainer condicionado ao modelo**: TreeExplainer só é usado quando o modelo ativo é compatível (árvores); para a LR, explicação linear compatível. Trocar de modelo pode mudar a estratégia de explicação — a decisão é conjunta, nunca improvisada no backend.

## Alternativas rejeitadas

- **Shipar RF diretamente como "padrão de mercado"**: rejeitado — seleciona modelo sem evidência no gate e sem fallback.
- **Explainer único fixo independente do modelo**: rejeitado — explicações devem refletir o modelo ativo; explicar árvore com método linear (ou o inverso) produz explicação incorreta.

## Consequências

- Toda versão de modelo registra no manifesto (ADR-0002) qual gate venceu e com quais métricas.
- A explicação exibida ao usuário muda de formato conforme o modelo ativo — o frontend deve tratar os dois formatos.
- A derrota do RF no gate não é falha de pipeline: fallback para LR é comportamento previsto e documentado.
