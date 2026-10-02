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

## v5: montagem e sincronização das junções

Após a avaliação auditiva da amostra v4 pelo usuário, a investigação passou
às junções. No WAV anterior à sincronização, as quatro pausas tinham RMS zero.
Após `atempo=0.85` no arquivo concatenado, janelas de 80 ms próximas às
posições esperadas das junções tinham RMS entre 0,00198 e 0,00364.
Essas posições são aproximadas: o WSOLA não mantém um mapeamento exato de
cada amostra. A medição evidencia a alteração das pausas; não quantifica
isoladamente o chiado percebido.

Os dois notebooks agora salvam os comprimentos dos blocos Qwen junto ao
WAV temporário e aplicam `atempo` em cada bloco individualmente. As rampas
são reaplicadas após o ajuste de velocidade e as pausas de 80 ms são
inseridas depois dele. O ajuste considera a duração da fala separadamente
das pausas. Duração que exigiria cortar palavras gera erro explícito.
Outros motores continuam usando sua sincronização existente.

A margem do detector nas bordas foi reduzida de 450 para 100 ms, e a rampa
dos blocos Qwen passou de 10 para 40 ms. Na amostra real, isso retirou
92, 124, 0, 220 e 188 ms do início dos cinco blocos, respectivamente.
Não foi aplicado filtro espectral à fala ou amplificação de volume.

Usando os mesmos cinco áudios brutos gerados na T4, a nova amostra tem
40,7372 segundos. As junções começam em 10,3350, 21,2865, 26,1571 e 35,0766
segundos: todas as 1.920 amostras de cada pausa são exatamente zero.
As pausas internas não são removidas. Os arquivos pessoais continuam fora do Git.

Os 15 testes passaram, incluindo processamento real de WAV/FFmpeg nos dois
notebooks, silêncio nas junções após time-stretch, encaminhamento dos
limites nos fluxos de dublagem/SRT e recusa de truncamento. A avaliação
auditiva da nova amostra ainda é necessária; ruído dentro da fala gerada
não pode ser declarado resolvido somente por essas verificações.

## v6: frases agrupadas e ritmo natural

O usuário ainda relatou chiado e pequena variação de voz na amostra v5,
em torno de 10, 22 e 26 segundos. Portanto, silêncio exato na pausa não
era evidência suficiente para considerar a qualidade acústica resolvida.

Foi feito novo A/B na mesma T4: o mesmo texto de 647 caracteres, antes
dividido em cinco chamadas, passou a uma única chamada Qwen, com a mesma
referência e configuração. A saída contém 32,16 segundos de fala, sem
emendas, filtros ou time-stretch. A geração consumiu 103,73 segundos e
atingiu 4,99 GB de memória CUDA alocada no processo de diagnóstico.
Whisper reconheceu todo o conteúdo solicitado, com ambiguidades nos nomes
próprios. A avaliação auditiva do usuário primeiro confirmou melhora clara
dos dois problemas e depois informou que o chiado e a variação desapareceram.

Nos dois notebooks, o Qwen agora agrupa frases até 700 caracteres por chamada,
em vez de reiniciar o clone a cada 180 caracteres. Chinês, japonês e coreano
usam 320 caracteres para limitar a duração em escritas mais densas. A semente,
referência e idioma explícito continuam fixos. Outros motores conservam seus
limites de texto. Para textos maiores ainda haverá mais de uma chamada.

A sincronização temporal fica desativada por padrão na interface dos dois
notebooks, preservando o ritmo natural da amostra aprovada. Ela permanece
opcional, com aviso de que preservar o ritmo pode alterar a duração final.
Quando ativada, a sincronização Qwen continua sendo aplicada por bloco.

Os 16 testes passaram, incluindo a regressão que mantém um texto de mais
de 600 caracteres em uma só chamada para espanhol, francês e inglês,
e que verifica divisão e conservação do texto acima do limite. A confirmação
auditiva refere-se à amostra espanhola de 32 segundos; uma dublagem completa,
as versões longas em francês/inglês e a execução em GPU L4 ainda precisam
de validação. Não se trata de garantia universal de ausência de artefatos.

## Duração para vídeo

O usuário confirmou que a versão contínua ajustada para 40,7372 segundos
mantém a qualidade e solicitou que o áudio acompanhe a duração do original.
A sincronização volta a ficar ativada por padrão em T4 e L4. O agrupamento
de frases e os parâmetros Qwen aprovados permanecem os mesmos.

Na nova sessão T4, uma geração espanhola completa de nove blocos foi
concluída sem sincronização, com 282,18 segundos. Seus comprimentos foram
recuperados dos metadados do próprio notebook. Os primeiros três blocos
foram processados pela função real de sincronização e montados em uma amostra
de exatamente 120 segundos, incluindo silêncio complementar ao final.
Essa amostra testa duas transições entre blocos maiores. O usuário confirmou
estabilidade geral, mas relatou pequeno chiado e mudança de tom nas transições,
próximas de 39 e 75 segundos. Portanto, o agrupamento de 700 caracteres ainda
não resolve a qualidade das emendas, mesmo com pausas numericamente silenciosas.
A PR de duração permanece sem merge durante essa investigação. O próximo
ensaio compara o mesmo texto dos três blocos (1.907 caracteres) em uma única
chamada Qwen, mantendo referência, idioma e parâmetros da amostra curta aprovada.

Também foi sincronizado localmente o espanhol completo, sem regenerar a
voz: original de 349,9733125 segundos, WAV final de 349,9733333 segundos.
A diferença é inferior a uma amostra PCM de 24 kHz. O ajuste de velocidade
continua limitado a 18%; diferenças remanescentes são preenchidas com silêncio.
Essa correspondência de duração não comprova alinhamento de cada frase às
imagens nem identidade vocal de todo o arquivo.

Os 17 testes passaram. A nova regressão usa WAV e FFmpeg reais para verificar
a duração final da linha do tempo e a preservação da introdução, fala e
encerramento em ambos os notebooks. Os áudios pessoais permanecem locais.

## Ensaio contínuo longo: resultado descartado

A chamada única com 1.907 caracteres e `non_streaming_mode=False` atingiu
o limite de 2.048 tokens: 163,76 segundos, 798,86 segundos de processamento
e pico de 5,07 GB CUDA no processo de diagnóstico. O Whisper reconheceu
o texto até aproximadamente 41 segundos; depois detectou repetições que
não fazem parte do roteiro. Portanto, esse ensaio foi descartado, sem
aumentar o limite de caracteres nos notebooks ou aplicar a alteração à main.
A amostra curta aprovada não demonstrava estabilidade para esse texto longo.

O mesmo roteiro foi gerado separadamente com `non_streaming_mode=True`,
que, na implementação oficial, fornece o texto completo antes da fala.
Mantendo os demais parâmetros e a referência, a saída teve 95,8400 segundos,
353,42 segundos de processamento e pico de 5,06 GB CUDA. O Whisper reconheceu
o roteiro até a última frase, sem o ciclo de repetição do primeiro teste,
com ambiguidades em nomes próprios e siglas. A versão sincronizada tem
exatamente 120 segundos, sem emendas internas; a fala acaba aproximadamente
em 116,81 segundos e o restante é silêncio complementar. O pico final é
0,7066 e todas as amostras são finitas.

Essa amostra foi entregue para avaliação auditiva. O modo de síntese dos
notebooks não foi alterado: conteúdo completo e duração correta ainda não
comprovam fidelidade vocal ou ausência de chiado. A dublagem completa,
francês/inglês e a GPU L4 continuam pendentes de validação dessa abordagem.

## v7: amostra de dois minutos aprovada

O usuário confirmou que o chiado e a mudança de tom desapareceram na nova
amostra contínua de 120 segundos. Essa aprovação vale para o trecho espanhol
ensaiado, incluindo as regiões próximas de 39 e 75 segundos.

A implementação compartilhada de T4/L4 passa a fornecer o texto antecipado
para as escritas latina/cirílica e agrupa até 6.000 caracteres. Trechos até
2.000 caracteres usam orçamento de 2.048 tokens; os maiores, 6.144. Chinês,
japonês e coreano mantêm o limite de 320 caracteres e o modo anterior.
Saídas próximas ao limite de tokens são recusadas antes de salvar o WAV,
usando os 1.920 samples/quadro a 24 kHz do codec oficial. A sincronização
continua ativada por padrão, sem cortar palavras.

Os 19 testes passaram. As novas regressões verificam uma chamada para
roteiros de aproximadamente dois minutos e de mais de 5.000 caracteres
em inglês/francês/espanhol, conservação de todo o texto, orçamento de
geração e recusa da saída de 163,76 segundos do ensaio que atingiu o limite.
O ensaio completo inglês de 5.450 caracteres terminou na T4 com 249,20
segundos de áudio, 914,24 segundos de processamento e pico de 5,43 GB CUDA.
O reconhecimento cobriu o roteiro até a última frase, com similaridade de
97,12% entre as sequências de palavras. A duração equivale a aproximadamente
3.115 quadros do codec, abaixo do orçamento de 6.144. Isso verifica conteúdo
e capacidade de geração; não comprova sotaque ou naturalidade.

A prévia inglesa de 120 segundos, extraída de um ajuste experimental de
249,20 para 349,9733 segundos (velocidade 0,7115), foi reprovada pelo usuário:
sotaque indiano, ritmo lento e fala não natural. O limite experimental de
35% de ajuste não foi incorporado aos notebooks. A aprovação espanhola não
se estende ao inglês; a PR permanece sem merge.

O próximo diagnóstico inglês isola `x_vector_only_mode=True`, conservando
texto, idioma, semente e amostragem, para comparar a referência de identidade
sem os códigos acústicos portugueses. Sua avaliação deve usar a velocidade
original da geração, antes de qualquer ajuste para vídeo. Esse teste de
1.824 caracteres terminou com 88,72 segundos, 323,66 segundos de processamento
e pico de 5,06 GB CUDA. A saída é finita, sem clipping; o reconhecimento
identificou inglês com probabilidade de 99,67%, cobriu a última frase e teve
97,52% de similaridade entre as sequências de palavras. A prévia entregue
preserva todas as amostras no ritmo original, apenas convertidas para PCM16.
A avaliação auditiva permanece pendente; não foi incorporado esse perfil
aos notebooks.

O ensaio francês contínuo, ainda com referência acústica portuguesa,
terminou com 81,04 segundos para 1.865 caracteres. O reconhecimento identificou
francês com probabilidade de 98,41%, cobriu a última frase e teve 92,44% de
similaridade entre as sequências de palavras. Não há aprovação auditiva.

Uma medição com Silero VAD, margem de 100 ms e silêncio mínimo de 200 ms,
estimou 52,00 segundos de pausas internas no original de 349,97 segundos,
contra 5,64 segundos na geração inglesa de 249,20 segundos. Isso motiva
avaliar a distribuição das pausas antes de alongar toda a fala; não comprova
que a diferença completa possa ser compensada sem prejudicar a naturalidade.
GPU L4 e arquivo completo permanecem pendentes de validação auditiva.

## v8: perfil inglês aprovado e aplicado nos dois notebooks

Em 2 de outubro de 2026, o usuário avaliou a amostra inglesa de 88,72 segundos
no ritmo original como "perfeito" e autorizou commit e merge. T4 e L4 passam
a usar `x_vector_only_mode=True` para inglês, conservando a mesma referência,
idioma explícito, semente e parâmetros da amostra. Os demais idiomas mantêm
ICL, incluindo o perfil espanhol aprovado. Os testes verificam a seleção do
perfil nas duas GPUs, inclusive nos códigos regionais, sem alterar o idioma
ou a referência entre chamadas.

Essa aprovação se limita à amostra inglesa sem ajuste de velocidade. Não
comprova naturalidade após sincronizar o arquivo completo, francês ou GPU L4.
O ajuste experimental de 35% continua excluído. A sincronização existente
conserva o limite de 18% e a duração final do original, com preenchimento
remanescente por silêncio; ainda pode exigir revisão auditiva para vídeo.
