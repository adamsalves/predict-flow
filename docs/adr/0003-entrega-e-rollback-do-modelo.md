# ADR-0003: Entrega e rollback do modelo por versão

- Status: Aceito
- Data: 2026-11-16

## Contexto

O sistema precisa trocar a versão ativa do modelo de forma controlada e revertê-la quando algo sai errado — sem downtime de desenvolvimento e sem depender de serviços externos. O modelo é artefato versionado (ver ADR-0002) que precisa de um mecanismo de entrega auditável.

## Decisão

- **Layout versionado**: `models/<version>/model.joblib` + `models/<version>/model_meta.json`.
- **Volume read-only** no container do backend: o processo nunca escreve no diretório de modelos.
- **Origem controlada**: artefatos chegam ao volume por processo explícito (build/cópia auditável), nunca por upload em runtime.
- **Verificação de integridade**: SHA-256 do `model.joblib` conferido contra o manifesto **antes** de qualquer `joblib.load`; divergência impede a ativação.
- **Ativação e rollback por configuração + restart**: a versão ativa é apontada por configuração (`MODEL_PATH`); trocar de versão é mudar o apontamento e reiniciar o processo. Sem hot-swap em runtime na v1.
- **Explícito na v1**: sem MLflow, DVC ou S3.

## Alternativas rejeitadas

- **Hot-swap em runtime**: rejeitado — adiciona complexidade de concorrência e estado sem necessidade no volume da demo local.
- **MLflow/DVC/S3**: rejeitado para a v1 — dependência operacional desproporcional; pode ser reavaliado no gate de piloto (ADR-0010).

## Consequências

- Rollback é imediato e reproduzível (apontar config + restart), ao custo de um restart por troca de versão.
- A verificação SHA-256 antes do `joblib.load` é um ponto de segurança: protege contra artefato corrompido ou adulterado.
- Múltiplas versões coexistem no volume; a política de retenção de versões antigas fica para o gate de piloto.
