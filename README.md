# 🎙️ Matraca Studio — Dublador & Clonador de Voz

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dgurgel-info/ai-colab-matracastudio/blob/main/Matraca_Studio.ipynb)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![Framework](https://img.shields.io/badge/Framework-Qwen3--TTS%20%7C%20OmniVoice%20%7C%20Whisper-brightgreen.svg)
![UI](https://img.shields.io/badge/UI-Gradio%205-orange.svg)

> **Duble qualquer vídeo MP4 ou áudio clonando a sua própria voz e mantendo exatamente o mesmo tempo de duração do conteúdo original!**

O **Matraca Studio** é uma suíte completa de localização e dublagem de vídeo e áudio potencializada por Inteligência Artificial. Com ele, você envia uma gravação, o sistema transcreve o conteúdo, traduz para o idioma desejado, clona a sua voz utilizando **Qwen3-TTS 1.7B** ou **OmniVoice** e sincroniza milimetricamente o áudio traduzido com a duração do vídeo original, exportando o novo vídeo pronto e em alta definição.

---

## ✨ Principais Funcionalidades

- 🎥 **Entrada Universal de Mídia**: Suporte para vídeos (`.mp4`, `.mov`, `.mkv`, `.avi`), faixas de áudio (`.wav`, `.mp3`, `.m4a`) ou gravação direta pelo microfone.
- 🗣️ **Reconhecimento Preciso e Transcrição Editável**: Transcrição automática com Whisper e detecção do idioma de origem. Permite **revisar e ajustar palavras, pontuações ou nomes próprios** antes de iniciar a dublagem.
- 🧠 **Seleção de Motores de IA de Ponta**:
  - ⚡ **Qwen3-TTS 1.7B Base** (*Alibaba Cloud*): Modelo autorregressivo de 1.7B parâmetros com altíssima expressividade, entonação realista e prosódia humana.
  - 🎙️ **OmniVoice** (*k2-fsa*): Modelo leve e rápido de difusão acústica.
- 🌍 **Dublagem Multi-Idioma Dinâmica**: Permite selecionar múltiplos idiomas de destino. A interface adapta as opções automaticamente de acordo com o modelo selecionado (no Qwen3-TTS, o Árabe não fica selecionável por não possuir suporte nativo; no OmniVoice, todos os 10 idiomas estão liberados).
- ⏱️ **Linha do Tempo Inteligente (Preservação de Música de Introdução/Encerramento)**: Se o arquivo contiver música ou vinheta de abertura antes da fala (ex: de 0.0s a 5.0s), essa trilha é **preservada integralmente**. A voz clonada entra **no segundo exato onde a pessoa começa a falar**, e encerra mantendo músicas finais intactas.
- 🧹 **Gestão Estrita de VRAM e Processamento 100% Sequencial**:
  - **Apenas 1 modelo ativo por vez na GPU**: Ao alternar entre Qwen3-TTS e OmniVoice, o modelo anterior é descarregado da memória antes do novo carregar.
  - **Processamento estritamente sequencial (1 idioma por vez)**: Nunca executa gerações simultâneas, realizando limpeza imediata de cache com `torch.cuda.empty_cache()` após cada idioma.
- 🎬 **Remuxing Instantâneo de Vídeo**: Substituição direta da trilha de áudio no vídeo original usando cópia de stream (`-c:v copy`), sem perda de qualidade visual e renderização em segundos.
- 📥 **Download Individual e Prévias em Tempo Real**: Players embutidos de áudio e vídeo com download imediato dos arquivos gerados (WAVs e MP4s).
- ✍️ **Aba de Clonagem Livre (TTS)**: Sintetize qualquer texto digitado com a sua voz clonada escolhendo entre Qwen3-TTS 1.7B e OmniVoice.

---

## 🌐 Idiomas Suportados

| Idioma | Código | Qwen3-TTS 1.7B | OmniVoice |
| :--- | :---: | :---: | :---: |
| 🇧🇷 **Português do Brasil (pt-BR)** | `pt-BR` | ✅ Nativo | ✅ Nativo |
| 🇺🇸 **Inglês (English)** | `en` | ✅ Nativo | ✅ Nativo |
| 🇪🇸 **Espanhol (Español)** | `es` | ✅ Nativo | ✅ Nativo |
| 🇫🇷 **Francês (Français)** | `fr` | ✅ Nativo | ✅ Nativo |
| 🇩🇪 **Alemão (Deutsch)** | `de` | ✅ Nativo | ✅ Nativo |
| 🇨🇳 **Chinês Simplificado (中文)** | `zh-CN` | ✅ Nativo | ✅ Nativo |
| 🇮🇹 **Italiano (Italiano)** | `it` | ✅ Nativo | ✅ Nativo |
| 🇯🇵 **Japonês (日本語)** | `ja` | ✅ Nativo | ✅ Nativo |
| 🇷🇺 **Russo (Русский)** | `ru` | ✅ Nativo | ✅ Nativo |
| 🇸🇦 **Árabe (العربية)** | `ar` | ❌ Não suportado *(desmarcado)* | ✅ Suportado |

---

## 🔄 Fluxo de Processamento (Pipeline)

```mermaid
flowchart LR
    A[Vídeo MP4 / Áudio] --> B[Extração do Áudio 24kHz]
    B --> C[Whisper STT]
    C --> D[Transcrição Editável no Gradio]
    D -->|Usuário revisa texto| E[Tradução Multi-Idioma]
    B --> F[Amostra Vocal Limpa]
    E --> G{Modelo Selecionado}
    G -->|Qwen3-TTS 1.7B| H[Síntese Autorregressiva 24kHz]
    G -->|OmniVoice| I[Síntese por Difusão 24kHz]
    F --> H
    F --> I
    H --> J[Time-Stretching Suave FFmpeg]
    I --> J
    J --> K[Áudio Sincronizado WAV 48kHz]
    A --> L[FFmpeg Video Remux AAC 192k]
    K --> L
    L --> M[Vídeo MP4 Dublado Final]
    M --> N[Limpeza de VRAM e Próximo Idioma]
```

---

## 🚀 Como Executar no Google Colab

1. Abra o notebook diretamente no Google Colab clicando no badge abaixo:  
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dgurgel-info/ai-colab-matracastudio/blob/main/Matraca_Studio.ipynb)
2. Ative a aceleração por GPU:
   - No menu superior, clique em **Ambiente de Execução** (*Runtime*) ➔ **Alterar tipo de ambiente de execução** (*Change runtime type*).
   - Selecione **T4 GPU** e salve.
3. Execute as células sequencialmente:
   - **Passo 1:** Instala as dependências (Qwen3-TTS, OmniVoice, Whisper, Gradio) e o FFmpeg.
   - **Passo 2:** Carrega o Whisper e inicializa o motor de TTS na VRAM da GPU (1 modelo por vez na memória).
   - **Passo 3:** Compila o motor de sincronização temporal, tradução robusta e síntese sequencial.
   - **Passo 4:** Inicia a aplicação Gradio e gera o link público compartilhável (`https://...gradio.live`).
4. **Utilizando a Interface:**
   - Envie seu vídeo ou áudio (e opcionalmente uma gravação de voz limpa).
   - Escolha o modelo de IA desejado (**Qwen3-TTS 1.7B** para alta expressividade ou **OmniVoice** para leveza e suporte ao Árabe).
   - Clique em **🔍 1. Transcrever e Analisar Áudio Original**.
   - Leia e ajuste qualquer palavra na caixa de transcrição caso necessário.
   - Marque os idiomas de destino desejados (**Inglês, Espanhol**, etc.).
   - Clique em **🚀 2. Dublar e Sincronizar Vídeo/Áudio**.
   - Acompanhe a barra de progresso em tempo real e faça o download ou reprodução dos arquivos gerados!

---

## 📁 Estrutura do Projeto

```text
matracastudio/
├── Matraca_Studio.ipynb      # Notebook completo com backend e interface Gradio
├── README.md                 # Documentação oficial do projeto
├── LICENSE                   # Licença MIT
└── .gitignore                # Arquivos e extensões ignoradas no versionamento
```

---

## 📄 Licença

Este projeto é distribuído sob a licença **MIT**. Consulte o arquivo [LICENSE](LICENSE) para obter mais informações.
