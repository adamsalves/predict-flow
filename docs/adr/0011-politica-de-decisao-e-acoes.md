# ADR-0011: Política de decisão e avaliação de ações

- Status: Aceito
- Data: 2026-11-16

## Contexto

O sistema recomenda ações de retenção e registra seus outcomes. Há uma armadilha conceitual grave: explicações preditivas (SHAP) e associações observacionais ("clientes que receberam desconto churnaram menos") não medem efeito causal de nenhuma ação. Sem política explícita, a demo tenderá a apresentar correlação como ROI de retenção.

## Decisão

- **SHAP é explicação preditiva, não causal**: fatores de risco explicam o score; não dizem o que aconteceria se agíssemos. Essa limitação é declarada na interface e na documentação.
- **Toda ação de retenção tem**: owner, SLA, critério de elegibilidade, custo, ordem de precedência entre ações, fallback manual e registro de execução.
- **"Outcomes observados por ação" não é avaliação de eficácia**: clientes que receberam ação não são comparáveis aos que não receberam (diferem em risco, elegibilidade e critério de escolha). Esses números são descritivos, nunca apresentados como efeito ou ROI da ação.
- **Retenção incremental só é aferida com comparação adequada**, preferencialmente **randomizada por cliente** quando viável (desenho controlado); até lá, nenhuma conclusão de eficácia é extraída.

## Alternativas rejeitadas

- **Apresentar outcome por ação como "taxa de sucesso da ação"**: rejeitado — comparação enviesada apresentada como causal é o erro central que esta ADR proíbe.
- **Recomendar ação com base em SHAP** ("o fator mais importante do score define a ação"): rejeitado — importância preditiva não é alavanca causal.
- **Deferir a pergunta causal "para depois do MVP" sem registrar a limitação**: rejeitado — a restrição precisa estar no produto desde a primeira tela de ações.

## Consequências

- Telas de decisão/ação (#19, #20) exibem outcomes observados com ressalva explícita de não-causalidade.
- O caminho para medir eficácia real é o desenho controlado — requisito do piloto, não da demo.
- Métricas descritivas por ação continuam úteis para operação (SLA, cobertura, custo), desde que rotuladas corretamente.
