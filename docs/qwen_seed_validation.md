# Qwen3-TTS: semente fixa entre blocos

## Problema reproduzido

A configuração anterior reiniciava o gerador aleatório com `42 + idx`.
Na sessão T4, os blocos 3 e 4 de uma tradução espanhola mudaram de timbre
antes de qualquer concatenação, fade ou sincronização com FFmpeg.
Portanto, suavizar somente as junções não corrigia essa falha.

T4 e L4 agora usam `42` em todos os blocos, preservando o estado aleatório
externo. O prompt ICL continua único por idioma e `language` continua explícito.
Não foram adicionadas instruções de sotaque, normalização que amplifique ruído,
filtros espectrais ou mudanças nos parâmetros oficiais de amostragem.

## A/B real, 2026-10-01

Ambiente: GPU NVIDIA T4, Qwen3-TTS 0.1.1, Transformers 4.57.3,
Qwen/Qwen3-TTS-12Hz-1.7B-Base, FP16, `non_streaming_mode=False`.
Mesma referência portuguesa de 6,48 segundos e sua transcrição,
mesmo texto, prompt e parâmetros; variou apenas a semente.

Comparação independente com SpeechBrain ECAPA-TDNN, usando a média normalizada
de oito janelas de quatro segundos do áudio original. A similaridade cosseno
é uma medida comparativa, não uma probabilidade de identidade.

| Trecho espanhol | Semente anterior | Cosseno anterior | Semente fixa | Cosseno corrigido |
| --- | ---: | ---: | ---: | ---: |
| Bloco 3, 70 caracteres | 44 | 0,418 | 42 | 0,683 |
| Bloco 4, 161 caracteres | 45 | 0,231 | 42 | 0,692 |

O primeiro bloco, com semente 42, teve cosseno 0,698. Nas amostras curtas
adicionais, a semente 42 teve 0,660 em francês e 0,692 em inglês.
As sementes anteriores não falharam em todas as amostras: a semente 44 teve
0,685 em francês e 0,722 em inglês. A mudança elimina uma variação que
reproduziu a falha espanhola; não constitui um detector universal de troca de voz.

Também foi montada uma amostra dos primeiros cinco blocos espanhóis com os
helpers reais do notebook: 34,64 segundos antes da sincronização e 40,74
segundos com `atempo=0.85`, como na execução relatada. Os blocos 2 e 5,
também com semente 42, tiveram cossenos 0,727 e 0,540 respectivamente.
A variação restante reforça a necessidade de avaliação auditiva.

No início do bloco 3, a fração de energia acima de 6 kHz caiu de 0,004735
para 0,0000916, aproximadamente 52 vezes. Essa medida inclui componentes de
fala e não substitui avaliação auditiva de chiado. No bloco 4 caiu de
0,001371 para 0,000713. Não houve clipping nessas amostras.

Whisper verificou a fala no idioma solicitado, com ambiguidades de nomes
próprios. Essa verificação não comprova sotaque nativo nem pronúncia perfeita.
Os arquivos pessoais e as amostras permanecem fora do Git.

## Verificação automatizada e limites

Os 13 testes locais passaram. A regressão verifica a mesma semente em todos
os blocos e idiomas dos dois notebooks, o prompt fixo, o idioma explícito,
a preservação do RNG externo, as bordas e a ausência de amplificação de ruído.

A geração real foi executada na T4; a L4 recebeu a implementação idêntica e
os testes locais, sem ensaio acústico em sua GPU. Uma dublagem completa e
a avaliação auditiva pelo usuário ainda são necessárias. Semente fixa não
garante a identidade vocal de qualquer texto nem ausência de sotaque.
