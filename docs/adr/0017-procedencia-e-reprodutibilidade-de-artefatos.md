# ADR-0017: Procedência de artefatos e reprodutibilidade de dependências

- Status: Aceito
- Data: 2026-11-18

## Contexto

O SHA-256 registrado em um manifesto detecta corrupção e divergência acidental, mas não estabelece confiança se artefato e manifesto puderem ser alterados juntos. `joblib.load` executa serialização potencialmente insegura e nunca deve receber artefato arbitrário. Dependências declaradas apenas com limites mínimos também impedem reproduzir o ambiente de treino e serving.

## Decisão

Esta ADR complementa as ADRs 0002 e 0003:

- Todo artefato é publicado por processo controlado e auditável; upload em runtime continua proibido.
- O digest esperado da versão ativa vem de uma fonte confiável separada do diretório do artefato, como configuração aprovada ou registro de release. O manifesto adjacente não é a única âncora de confiança.
- O carregamento valida manifesto, schema, compatibilidade e digest antes de `joblib.load`; divergência impede a ativação.
- Backend e workers recebem o volume de modelos como read-only. A origem que publica os artefatos é distinta do processo que os consome.
- Dependências de treino e serving são fixadas em lockfiles com versões exatas. As versões compartilhadas do pipeline devem ser idênticas nos dois ambientes.
- Dependências de desenvolvimento/teste são separadas das de runtime quando isso reduzir a imagem e a superfície de ataque, sem criar ambientes cientificamente divergentes.
- Datasets, caches, `models/` e diretórios locais de artefatos são ignorados pelo Git antes do primeiro treino.
- O manifesto registra origem, licença, hashes, versões e procedência. Reprodutibilidade não é apresentada como autenticidade.
- Antes do piloto, o threat model da #17 decide o mecanismo adicional de assinatura ou registry confiável; dados reais não são aceitos somente com hash adjacente.

## Alternativas rejeitadas

- **Confiar em qualquer par artefato/manifesto com hashes coerentes**: rejeitado porque uma substituição conjunta continuaria válida.
- **Manter somente `>=` nas dependências de ML**: rejeitado porque builds distintos podem produzir ou carregar artefatos incompatíveis.
- **Versionar datasets e `.joblib` no Git**: rejeitado por tamanho, procedência e ciclo de vida distinto do código.

## Consequências

- Uma issue transversal de M2 prepara locks, lint/type checking e exclusões de artefatos antes do treino.
- A #8 exige digest confiável além do manifesto e testa substituição conjunta de artefato/manifesto.
- A #17 inclui assinatura/procedência no threat model e no gate de dados reais.
