# 🎙️ Matraca Studio — Dublagem e Clonagem de Voz

![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![Modelos](https://img.shields.io/badge/Modelos-VoxCPM2%20%7C%20Qwen3--TTS%20%7C%20OmniVoice-brightgreen.svg)
![Interface](https://img.shields.io/badge/Interface-Gradio-orange.svg)
![Licença](https://img.shields.io/badge/Licen%C3%A7a-MIT-yellow.svg)

O **Matraca Studio** transcreve, traduz, dubla e clona vozes a partir de áudio ou legendas SRT. A aplicação preserva a linha do tempo, permite revisar a transcrição antes da síntese e gera arquivos WAV individuais ou um pacote ZIP da sessão.

## Escolha o notebook correto

| Notebook | GPU indicada | Modelos disponíveis | Quando usar |
| :--- | :--- | :--- | :--- |
| [Matraca_Studio_T4.ipynb](Matraca_Studio_T4.ipynb) | NVIDIA T4 | Qwen3-TTS 1.7B e OmniVoice | Versão mais leve para sessões T4 e fluxo padrão de dublagem. |
| [Matraca_Studio_L4.ipynb](Matraca_Studio_L4.ipynb) | NVIDIA L4, 24 GB | **VoxCPM2 2B**, Qwen3-TTS 1.7B e OmniVoice | Versão recomendada para a maior qualidade, maior velocidade de inferência e todos os três modelos. |

### Por que usar a L4?

A versão L4 carrega o **VoxCPM2 2B** como motor padrão, com saída nativa em **48 kHz**, clonagem Hi-Fi e suporte a 30 idiomas. A GPU L4 tem 24 GB de VRAM, oferecendo mais margem para o modelo de 2B, Whisper e processamento de áudio do que a T4. Isso permite síntese de maior qualidade e, na prática, geração mais rápida que a T4 para o mesmo trabalho.

Na dublagem multilíngue, o VoxCPM2 usa orientação de sotaque nativo por idioma — por exemplo, *General American English* para inglês — para evitar sotaques estrangeiros inadequados. Para clonagem livre, o modo Hi-Fi combina áudio de referência e transcrição para preservar timbre, ritmo, emoção e prosódia.

> A melhor qualidade depende de uma referência limpa de voz. Prefira uma fala isolada, sem música, de 5 a 15 segundos, e revise a transcrição quando usar clonagem Hi-Fi.

## Modelos de voz

| Modelo | Disponível em | Destaques |
| :--- | :--- | :--- |
| **VoxCPM2 2B** | L4 | Padrão L4; 48 kHz, clonagem Hi-Fi, controle de estilo e 30 idiomas. |
| **Qwen3-TTS 1.7B** | T4 e L4 | Alta expressividade e clonagem para 9 idiomas. |
| **OmniVoice** | T4 e L4 | Alternativa leve, rápida e com suporte a Árabe. |

O aplicativo mantém somente **um motor de TTS na GPU por vez**. Ao trocar o modelo, o anterior é removido da VRAM antes do próximo carregamento.

## Funcionalidades

- Entrada por arquivo (`.wav`, `.mp3`, `.m4a`, `.ogg`, `.flac`) ou microfone.
- Transcrição com Whisper e edição antes da dublagem.
- Tradução robusta para múltiplos idiomas de destino.
- Dublagem com sincronização de duração e preservação de introdução e encerramento.
- Clonagem livre por texto ou por arquivo `.srt` com pausas e janelas temporais.
- Processamento sequencial por idioma, limpeza de cache de GPU e cancelamento seguro.
- Histórico em `./outputs`, downloads individuais e `audios_dublados.zip`.
- Gerenciador de jobs persistente para recuperar o estado da interface após atualizar a página.

## Idiomas

| Motor | Idiomas |
| :--- | :--- |
| **VoxCPM2 (L4)** | 30 idiomas, incluindo Português Brasileiro, Inglês, Espanhol, Francês, Alemão, Árabe, Hindi, Japonês, Coreano, Russo, Tailandês, Turco e Vietnamita. |
| **Qwen3-TTS** | Português Brasileiro, Inglês, Espanhol, Francês, Alemão, Chinês Mandarim, Italiano, Japonês e Russo. |
| **OmniVoice** | Os nove idiomas do Qwen3-TTS, além de Árabe. |

## Como executar no Google Colab

Abra diretamente a versão adequada à sua GPU:

### Versão T4

1. [![Abrir Matraca Studio T4 no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dgurgel-info/ai-colab-matracastudio/blob/main/Matraca_Studio_T4.ipynb)
2. Em **Ambiente de execução → Alterar tipo de ambiente de execução**, selecione **T4 GPU**.
3. Execute as células do **Passo 1** ao **Passo 4** em ordem.
4. Abra o endereço `gradio.live` exibido ao final do Passo 4.
5. Execute o **Passo 5** quando quiser baixar todos os WAVs em um único ZIP.

### Versão L4 — recomendada

1. [![Abrir Matraca Studio L4 no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dgurgel-info/ai-colab-matracastudio/blob/main/Matraca_Studio_L4.ipynb)
2. Em **Ambiente de execução → Alterar tipo de ambiente de execução**, selecione **L4 GPU**.
3. Execute as células do **Passo 1** ao **Passo 4** em ordem. A primeira carga baixa os pesos do VoxCPM2 e pode levar alguns minutos.
4. Na interface, mantenha **VoxCPM2 2B** para a melhor qualidade ou selecione Qwen3-TTS/OmniVoice quando necessário.
5. Para geração máxima, mantenha os **30 passos de difusão** do VoxCPM2. Use o **Passo 5** para criar o ZIP final.

## Fluxo de uso

1. Envie ou grave o áudio e escolha o idioma original.
2. Clique em **Transcrever e analisar**, revise nomes e pontuação quando necessário.
3. Escolha o modelo, os idiomas de destino e a opção de sincronização.
4. Clique em **Dublar e sincronizar** e acompanhe o log em tempo real.
5. Baixe os WAVs na aba de resultados ou gere o pacote ZIP.

Para clonagem livre, abra a aba **Clonagem livre**, envie a referência de voz, escolha texto ou SRT e gere o áudio. Na L4, o VoxCPM2 utiliza a referência e sua transcrição para maximizar a fidelidade da voz.

## Estrutura do projeto

```text
ai-colab-matracastudio/
├── Matraca_Studio_T4.ipynb      # Notebook para GPU T4: Qwen3-TTS e OmniVoice
├── Matraca_Studio_L4.ipynb      # Notebook para GPU L4: VoxCPM2, Qwen3-TTS e OmniVoice
├── higgs_audio_v2_tokenizer.py  # Ponte de compatibilidade do OmniVoice
├── README.md                    # Esta documentação
├── LICENSE                      # Licença MIT
└── .gitignore
```

## Licença

O projeto é distribuído sob a licença [MIT](LICENSE). Os modelos têm suas próprias licenças e termos de uso; confira-os antes de uso comercial ou redistribuição.
