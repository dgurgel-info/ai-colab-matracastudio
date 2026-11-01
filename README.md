# 🎙️ Matraca Studio — Dublagem e Clonagem de Voz

![Modelos](https://img.shields.io/badge/Modelos-Qwen3--TTS%20%7C%20OmniVoice-brightgreen.svg)
[![Testes](https://github.com/dgurgel-info/ai-colab-matracastudio/actions/workflows/tests.yml/badge.svg)](https://github.com/dgurgel-info/ai-colab-matracastudio/actions/workflows/tests.yml)
![Licença](https://img.shields.io/badge/Licen%C3%A7a-MIT-yellow.svg)

O **Matraca Studio** dubla vídeos e clona vozes. Você envia um áudio (ou uma legenda SRT), ele transcreve, traduz e gera a dublagem em outros idiomas **com a mesma voz do começo ao fim**, mantendo a duração original. No final, entrega o áudio pronto para o YouTube.

Ele roda no **Google Colab**, direto no navegador, sem instalar nada no seu computador.

## O que ele faz

- Transcreve o áudio e deixa você revisar o texto antes de dublar.
- Traduz para vários idiomas e dubla com a voz clonada da pessoa.
- Ajusta a duração da fala ao áudio original, sem cortar palavras.
- Divide a dublagem em partes que você pode ouvir, corrigir e remontar depois, sem refazer tudo.
- Clona uma voz livremente, a partir de texto ou de um arquivo `.srt`.
- Entrega WAVs individuais ou um ZIP com tudo, já masterizado para o YouTube.

## Escolha o notebook

| Notebook | GPU | Modelos |
| :--- | :--- | :--- |
| [Matraca_Studio_T4.ipynb](Matraca_Studio_T4.ipynb) | NVIDIA T4 | Qwen3-TTS e OmniVoice |
| [Matraca_Studio_L4.ipynb](Matraca_Studio_L4.ipynb) | NVIDIA L4 (24 GB) | Qwen3-TTS e OmniVoice |

Os dois funcionam do mesmo jeito. A L4 só tem mais memória de vídeo, o que dá mais folga.

## Os modelos de voz

| Modelo | Para que serve |
| :--- | :--- |
| **Qwen3-TTS** | Modelo inicial. Voz expressiva e clonagem fiel em 9 idiomas: Português Brasileiro, Inglês, Espanhol, Francês, Alemão, Chinês, Italiano, Japonês e Russo. |
| **OmniVoice** | Alternativa mais leve e rápida. Tem os mesmos idiomas e mais o Árabe. |

### Como os modelos são carregados

- **Um modelo por vez.** O Qwen3-TTS já vem selecionado. Para trocar, basta escolher outro modelo na lista da interface: o anterior é descarregado por completo e a memória da GPU volta inteira.
- **Cada modelo tem o próprio ambiente.** Isso evita conflitos entre os programas que cada um precisa.
- **A primeira escolha demora alguns minutos.** Nessa hora o Matraca prepara o ambiente do modelo e baixa os arquivos dele. Depois disso, a troca é rápida enquanto a sessão do Colab estiver aberta.
- **Uma barra de progresso no topo da tela** mostra cada etapa (descarregar, preparar, baixar e iniciar). Se algo falhar, ela fica vermelha e mostra o erro.

## Como usar no Google Colab

1. Abra o notebook da sua GPU:

   [![Abrir na T4](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dgurgel-info/ai-colab-matracastudio/blob/main/Matraca_Studio_T4.ipynb) **T4**

   [![Abrir na L4](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dgurgel-info/ai-colab-matracastudio/blob/main/Matraca_Studio_L4.ipynb) **L4**

2. Em **Ambiente de execução → Alterar tipo de ambiente de execução**, escolha a GPU correspondente.
3. Execute as células do **Passo 1** ao **Passo 4**, em ordem.
4. Abra o endereço `gradio.live` que aparece no final do Passo 4. É a interface do Matraca Studio.
5. Quando terminar, execute o **Passo 5** para baixar todos os áudios em um ZIP.

Na interface, a aba **Dublagem sincronizada** guia você em cinco etapas: enviar o áudio, transcrever e revisar, escolher o modelo e os idiomas, dublar e baixar o resultado. A aba **Clonagem livre** serve para gerar voz a partir de um texto ou de um SRT.

> **Dica:** a qualidade da clonagem depende da voz de referência. Use uma fala limpa, de uma só pessoa, sem música, de 5 a 15 segundos, e confira se a transcrição está correta.

> **Segurança:** o Passo 4 cria um link público **sem senha**. Quem tiver o link pode usar a GPU da sua sessão. Não compartilhe o link e encerre a sessão do Colab quando terminar.

## Se algo der errado

| Problema | O que fazer |
| :--- | :--- |
| Um modelo não inicia | Veja o erro na barra de progresso e escolha outro modelo, ou tente de novo. Os registros completos ficam em `engines/<modelo>/install.log` e `engines/<modelo>/worker.log`. |
| O Passo 1 não consegue baixar o projeto | O Colab precisa de acesso ao GitHub. Confira a internet e tente novamente. |
| Uma parte da dublagem soou diferente | Abra a aba **🧩 Revisão por partes**, ouça a parte, gere outra versão e remonte o áudio. |
| Perdi a dublagem depois de reiniciar o Colab | As sessões ficam na pasta `sessoes/` do Colab, que é apagada ao reiniciar. Copie `sessoes/` e `outputs/` para o Google Drive antes de encerrar. |

## Para quem desenvolve

Prévia da interface no seu computador, sem GPU: `python scripts/preview_ui.py` (depois de `pip install -r requirements-dev.txt`). Testes automáticos: `python -m pytest -q tests`.

## Licença

O projeto é distribuído sob a licença [MIT](LICENSE). Os modelos têm licenças e termos próprios; confira-os antes de uso comercial ou redistribuição.
