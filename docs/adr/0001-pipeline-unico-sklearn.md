# ADR-0001: Pipeline único de preprocessing + modelo

- Status: Aceito
- Data: 2026-11-16

## Contexto

O backend precisa pontuar clientes e explicar os scores em produção. Se as transformações de features forem reimplementadas no backend em paralelo ao treino, surgem duas implementações da mesma lógica — com risco real de divergência silenciosa (drift treino–serving), que corrompe tanto as predições quanto qualquer explicação gerada sobre elas.

## Decisão

Preprocessing e modelo vivem no **mesmo `sklearn.Pipeline`**, serializado como um único artefato:

- Encoders de categóricas usam `handle_unknown='ignore'`, para que categorias não vistas no treino não derrubem o serving.
- O backend **pode invocar componentes internos do próprio Pipeline** (ex.: extrair o transformador de features alimentar o SHAP) para gerar explicações.
- O backend **nunca reimplementa transformações**: qualquer lógica de feature engineering aplicada no treino deve estar dentro do Pipeline e ser reusada via seus componentes.

## Alternativas rejeitadas

- **Transformações reimplementadas no backend** (duplicar one-hot/encoding/imputação em código próprio): rejeitada por introduzir drift treino–serving e dobrar a superfície de testes.
- **Framework de model serving separado** (ex.: serviço dedicado de inference): rejeitado para a v1 — overhead operacional desproporcional ao volume de uma demo local.

## Consequências

- Um único artefato (`model.joblib`) carrega toda a lógica de pontuação — ver ADR-0003 para entrega e rollback.
- Explicações (SHAP) são calculadas sobre exatamente as mesmas transformações do serving — ver ADR-0004.
- O desenho fica limitado a estimadores compatíveis com `sklearn.Pipeline`/`joblib`; migrar para outro framework de ML é uma decisão arquitetural nova.
