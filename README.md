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
| [Matraca_Studio_L4.ipynb](Matraca_Studio_L4.ipynb) | NVIDIA L4, 24 GB | **VoxCPM2 2B**, Qwen3-TTS 1.7B, OmniVoice e Higgs TTS 3 4B (experimental) | Versão recomendada para a maior qualidade, maior velocidade de inferência e todos os modelos. |

### Por que usar a L4?

A versão L4 carrega o **VoxCPM2 2B** como motor padrão, com saída nativa em **48 kHz**, clonagem Hi-Fi e suporte a 11 idiomas. A GPU L4 tem 24 GB de VRAM, oferecendo mais margem para o modelo de 2B, Whisper e processamento de áudio do que a T4. Isso permite síntese de maior qualidade e, na prática, geração mais rápida que a T4 para o mesmo trabalho.

Na dublagem multilíngue, o VoxCPM2 usa orientação de sotaque nativo por idioma — por exemplo, *General American English* para inglês — para evitar sotaques estrangeiros inadequados. Para clonagem livre, o modo Hi-Fi combina áudio de referência e transcrição para preservar timbre, ritmo, emoção e prosódia.

> A melhor qualidade depende de uma referência limpa de voz. Prefira uma fala isolada, sem música, de 5 a 15 segundos, e revise a transcrição quando usar clonagem Hi-Fi.

## Modelos de voz

| Modelo | Disponível em | Destaques |
| :--- | :--- | :--- |
| **VoxCPM2 2B** | L4 | Padrão L4; 48 kHz, clonagem Hi-Fi, controle de estilo e 11 idiomas. |
| **Qwen3-TTS 1.7B** | T4 e L4 | Alta expressividade e clonagem para 9 idiomas. |
| **OmniVoice** | T4 e L4 | Alternativa leve, rápida e com suporte a Árabe. |
| **Higgs TTS 3 4B** 🧪 | L4 | Experimental. Roda num servidor isolado (SGLang-Omni, Python 3.12) instalado só quando você o escolhe. **Licença não comercial**: veja abaixo. |

O aplicativo mantém somente **um motor de TTS na GPU por vez**. Ao trocar o modelo, o anterior é removido da VRAM antes do próximo carregamento.

### Fidelidade e pronúncia com Qwen3-TTS

Nos dois notebooks, o Qwen3-TTS reutiliza a mesma referência vocal em todas as chamadas. Para inglês, usa a identidade vocal da referência (`x_vector_only_mode=True`), sem condicionar a síntese aos códigos acústicos portugueses. Francês, espanhol e os demais idiomas conservam áudio e transcrição em modo ICL. O idioma de destino orienta a pronúncia; não são injetadas instruções de sotaque por APIs privadas do modelo. A preparação continua exigindo uma transcrição válida da referência e informa erros sem trocar silenciosamente o perfil de clonagem.

A síntese agrupa até 6.000 caracteres em uma chamada para inglês, francês, espanhol e os demais idiomas de escrita latina/cirílica. O texto é fornecido integralmente antes da fala (`non_streaming_mode=True`), evitando reiniciar o clone a cada poucas frases. Trechos acima desse limite ainda precisam de mais de uma chamada; chinês, japonês e coreano conservam o limite anterior de 320 caracteres e seu modo de síntese. Os parâmetros de amostragem seguem o [exemplo oficial do Qwen](https://github.com/QwenLM/Qwen3-TTS/blob/main/examples/test_model_12hz_base.py), com semente e referência fixas. O orçamento de geração aumenta para narrativas longas, e saídas no limite são rejeitadas para não salvar texto possivelmente interrompido.

Na L4, o Qwen usa BF16 nativo; na T4, mantém FP16. O [Silero VAD](https://github.com/snakers4/silero-vad), na CPU, apara somente as bordas sem fala, com margem de 100 ms. As pausas internas são preservadas; não há filtro FFT nem normalização que amplifique ruído. A sincronização para vídeo fica ativada por padrão: ajusta a velocidade antes de inserir as eventuais pausas entre chamadas, preserva introdução/encerramento e completa a duração com silêncio quando necessário. Rampas de 40 ms protegem as bordas; o processo recusa ajustes que exigiriam cortar palavras. Duração total correta não garante alinhamento de cada frase às imagens.

A amostra espanhola contínua de dois minutos na T4 foi avaliada pelo usuário com desaparecimento do chiado e da variação de voz. A amostra inglesa de 88,72 segundos, com identidade vocal e ritmo original, também foi aprovada. Essa aprovação inglesa não cobre o ajuste de duração para vídeo: o ensaio anterior com alongamento de 35% foi reprovado e esse limite não foi incorporado. Francês, arquivos completos e a GPU L4 ainda exigem avaliação auditiva. Os ensaios e seus limites estão registrados em [Validação Qwen](docs/qwen_seed_validation.md).

**Limites da validação:** as aprovações auditivas acima cobrem as amostras T4 ensaiadas. O arquivo inglês completo após sincronização, francês e L4 continuam pendentes. A detecção de fala também não elimina ruído misturado às palavras. Confira uma prévia antes de gerar o áudio completo.

Ao testar uma atualização, abra o notebook atualizado e reinicie o ambiente do Colab antes de executar todos os passos. Uma aba já aberta pode continuar usando a versão anterior mesmo após atualizar o GitHub. O log desta versão inclui `narração contínua e sincronização para vídeo (v8)` e informa `referência identidade vocal` para inglês ou `referência ICL` para os demais idiomas.

Prefira uma referência de uma única pessoa, de 5 a 15 segundos, sem música, reverberação ou chiado, e com transcrição fiel. O idioma selecionado busca fala natural no idioma de destino, mas a referência também pode influenciar o sotaque: preservar o timbre e eliminar totalmente o sotaque não é garantido pelo modelo. As mudanças reduzem fontes de instabilidade; a fidelidade e a fluência precisam ser conferidas ouvindo a geração no Colab.

As regressões de código e junções podem ser executadas sem pesos de IA com `python -m unittest discover -s tests -v` (requer `numpy` e `torch`). Esses testes não medem similaridade de locutor nem pronúncia nativa.

## Funcionalidades

- Entrada por arquivo (`.wav`, `.mp3`, `.m4a`, `.ogg`, `.flac`) ou microfone.
- Transcrição com Whisper e edição antes da dublagem.
- Tradução robusta para múltiplos idiomas de destino.
- Dublagem com sincronização de duração e preservação de introdução e encerramento.
- Clonagem livre por texto ou por arquivo `.srt` com pausas e janelas temporais.
- Processamento sequencial por idioma, limpeza de cache de GPU e cancelamento seguro.
- Histórico em `./outputs`, downloads individuais e `audios_dublados.zip`.
- Gerenciador de jobs persistente para recuperar o estado da interface após atualizar a página.
- **Dublagem por partes** com a mesma voz em todas as partes, revisão e correção parte a parte (veja abaixo).
- **Masterização para YouTube** (-14 LUFS, pico verdadeiro seguro, 48 kHz / 24 bits) aplicada ao final.

## Idiomas

| Motor | Idiomas |
| :--- | :--- |
| **VoxCPM2 (L4)** | 30 idiomas, incluindo Português Brasileiro, Inglês, Espanhol, Francês, Alemão, Árabe, Hindi, Japonês, Coreano, Russo, Tailandês, Turco e Vietnamita. |
| **Qwen3-TTS** | Português Brasileiro, Inglês, Espanhol, Francês, Alemão, Chinês Mandarim, Italiano, Japonês e Russo. |
| **OmniVoice** | Os nove idiomas do Qwen3-TTS, além de Árabe. |

## Como executar no Google Colab

### Cache recuperável no Google Drive (opcional)

Para baixar manualmente as dependências, os pacotes APT, os pesos de todos os modelos e os arquivos auxiliares em um diretório do Drive, execute o script após o Passo 1:

```python
!git clone https://github.com/dgurgel-info/ai-colab-matracastudio.git
!python ai-colab-matracastudio/scripts/download_matraca_offline_cache.py \
  /content/drive/MyDrive/MatracaStudioCache
```

O processo cria um `manifest.json` com hashes dos arquivos. Os pesos são mantidos no Drive apenas como cópia de segurança; para melhor desempenho, copie o modelo escolhido para `/content` antes de carregá-lo.
O runtime-base do Colab (CUDA, driver NVIDIA, `torch` e `torchaudio`) continua sendo fornecido pela imagem da GPU; não é seguro reutilizar wheels desses componentes entre imagens/versões diferentes.

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

## Voz coerente, parte por parte

Cada idioma é dublado em **partes** (≈600 caracteres, sentenças inteiras; 400 no Higgs). Os padrões já vêm ajustados para vídeos do YouTube e não aparecem na tela:

| O que garante a mesma voz | Como |
| :--- | :--- |
| Mesma referência e mesmo prompt de voz | Criados uma vez por idioma e reaproveitados em todas as partes. |
| Mesma semente (`42`) em todas as partes e motores | Fixada só durante a geração; o RNG externo é preservado. |
| Detecção de parte destoante | Cada parte é medida (tom/F0, cor tonal, brilho, ritmo, volume e, no Qwen3-TTS, similaridade de voz) e comparada com a mediana das demais. Só vale com 3 ou mais partes. |
| Correção por geração, não por filtro | A parte que destoa é refeita com outra semente (1 tentativa automática). **Não há equalização nem redução de ruído** sobre a voz. |
| Volume igual entre as partes | Apenas ganho (limitado a ±6 dB e sem clipping). |
| Pausa de 180 ms entre partes, ritmo uniforme | O ajuste de duração usa o mesmo fator em todas as partes (limite de 18%) e nunca corta palavras. |
| Masterização final | `loudnorm` linear em duas passagens: -14 LUFS, pico verdadeiro ≤ -1,5 dBTP, WAV 48 kHz / 24 bits. |

### Aba 🧩 Revisão por partes

Depois da dublagem, abra a aba para **ouvir cada parte**, editar o texto, **regenerar** (nova variação), escolher entre os *takes* anteriores e **remontar o áudio final**. As sessões ficam em `./sessoes/<áudio>/<idioma>/` (um WAV por parte + `manifest.json`), então dá para remontar mesmo depois de reiniciar a página. Remontar não precisa do áudio original nem da GPU; só regenerar uma parte precisa do motor.

### Limites conhecidos

- A consistência por semente e o QC foram validados com motores sintéticos e testes automatizados. **A avaliação auditiva em GPU continua sendo necessária**, como no restante do projeto (`docs/qwen_seed_validation.md`).
- Partes menores podem soar diferentes entre si em alguns idiomas; é por isso que o QC compara cada parte com as demais e a revisão existe. Para o Qwen3-TTS, trechos longos continuam sendo o perfil aprovado: aumente o tamanho das partes se notar trocas de timbre.
- Com 1 ou 2 partes não há com o que comparar, então o QC relativo não é aplicado.

## Higgs TTS 3 4B (experimental, L4)

Modelo da [Boson AI](https://huggingface.co/bosonai/higgs-tts-3-4b) com clonagem zero-shot em 100+ idiomas. Não tem API Python local: o notebook instala o servidor SGLang-Omni num ambiente separado (`uv`, Python 3.12) **na primeira vez que você escolhe o modelo** e o encerra ao trocar de motor, para liberar a VRAM.

- **Não testado em GPU L4.** A ficha do modelo só confirma A100 40 GB; a placa de 24 GB é "relatada, não verificada". Se faltar memória, o erro do servidor aparece no log e os outros motores continuam funcionando.
- A API não documenta semente: o app tenta enviá-la e, se o servidor recusar, segue sem ela (a geração então não é reprodutível; o QC e a revisão continuam valendo).
- Se o download exigir aceite de termos, defina `HF_TOKEN` no ambiente do Colab (`os.environ['HF_TOKEN'] = ...`) antes de iniciar. Nunca cole tokens em células salvas.
- **Licença: [Boson Higgs TTS 3 Research and Non-Commercial](https://huggingface.co/bosonai/higgs-tts-3-4b/blob/main/LICENSE).** A *Creator Use Grant* permite monetizar vídeos em canais próprios **com atribuição**; o relatório final já traz o texto (`This audio was created with Boson AI's Higgs Audio — https://www.boson.ai/higgs-audio`). Hospedar como serviço/produto exige licença comercial, e só é permitido clonar vozes com consentimento.

## Prévia local da interface (sem GPU)

```bash
pip install -r requirements-dev.txt
python scripts/preview_ui.py                    # L4 em http://127.0.0.1:7860
python scripts/preview_ui.py --notebook T4 --port 7861
```

Executa o código real dos Passos 3 e 4 do notebook, trocando só Whisper, tradutor e TTS por substitutos (voz sintética). Algumas partes saem "destoantes" de propósito para você ver o QC e a revisão (`--no-drift` desliga).

## Testes

```bash
pip install -r requirements-dev.txt   # e o ffmpeg no PATH
python -m pytest -q tests
```

Os testes rodam sem GPU e sem baixar modelos: DSP validado contra o `ebur128` do ffmpeg, sessões e QC com um motor que reproduz o defeito de timbre, montagem/sincronização/masterização, interface e o cliente do Higgs contra um servidor falso. O CI do GitHub roda a mesma suíte.

## Estrutura do projeto

```text
ai-colab-matracastudio/
├── Matraca_Studio_T4.ipynb      # Notebook para GPU T4: Qwen3-TTS e OmniVoice
├── Matraca_Studio_L4.ipynb      # Notebook para GPU L4: VoxCPM2, Qwen3-TTS, OmniVoice e Higgs TTS 3
├── matraca/                     # Núcleo compartilhado pelos dois notebooks (clonado no Passo 1)
│   ├── dsp.py                   # Loudness BS.1770, medidas de voz, stretch, masterização
│   ├── parts.py, qc.py          # Divisão em partes e controle de qualidade
│   ├── session.py, timeline.py  # Sessão por partes (takes, regeneração), intro/outro e montagem
│   ├── dubbing.py               # Orquestração independente do motor
│   ├── engines/higgs.py         # Higgs TTS 3 (servidor isolado)
│   └── ui/                      # Tema/CSS e aba Revisão por partes
├── scripts/preview_ui.py        # Prévia local da interface sem GPU
├── tests/                       # Testes automatizados
├── higgs_audio_v2_tokenizer.py  # Ponte de compatibilidade do OmniVoice
├── README.md                    # Esta documentação
├── LICENSE                      # Licença MIT
└── .gitignore
```

## Licença

O projeto é distribuído sob a licença [MIT](LICENSE). Os modelos têm suas próprias licenças e termos de uso; confira-os antes de uso comercial ou redistribuição.
