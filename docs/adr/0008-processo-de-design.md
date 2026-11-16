# ADR-0008: Processo de design do frontend

- Status: Aceito
- Data: 2026-11-16

## Contexto

Um frontend de analytics (scores, fatores de risco, histórico, filas de trabalho) tende a acumular telas inconsistentes e componentes prematuramente abstraídos. Sem processo, cada tela é desenhada isoladamente e o `shared/` vira depósito de especulações usadas uma única vez.

## Decisão

- **Sequência obrigatória**: jornadas → wireframe → tokens → protótipo → aprovação. Nenhuma tela é implementada sem protótipo aprovado.
- **Estados first-class**: toda tela define loading, erro e vazio explicitamente — nunca apenas o caminho feliz.
- **Acessibilidade é critério de aceite**, não retoque final.
- **`shared/` só no terceiro uso**: um componente só é promovido a compartilhado quando a terceira tela o utiliza; antes disso, cada tela mantém o seu.
- **Validação com usuários representativos**, não apenas com o dono do produto.

## Alternativas rejeitadas

- **Componentes genéricos antecipados ("vamos precisar")**: rejeitado — abstração sem terceiro uso carrega custo de API errada; generalizar depois é barato, desfazer é caro.
- **Implementar direto do wireframe sem tokens/protótipo**: rejeitado — inconsistência visual e retrabalho sistemático.
- **Aprovação apenas pelo dono do produto**: rejeitado — validação de usabilidade exige quem representa o usuário final (analyst de retenção).

## Consequências

- As issues de frontend (#12–#14, #20, #25) seguem este processo; telas sem protótipo aprovado não entram em sprint.
- Tokens de design são decididos antes do scaffold (#12), não depois.
