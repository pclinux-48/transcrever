# Transcrever áudios (offline)

Ferramenta simples para transcrever áudios e vídeos offline usando o modelo Whisper
via `whisper-cpp` (CLI). Funciona localmente (sem API do Google), gratuito.

## Requisitos

- macOS com [Homebrew](https://brew.sh/) instalado
- FFmpeg: `brew install ffmpeg`
- Python 3.10+ (recomendado)

## Instalação

No diretório do projeto:

```bash
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
```

Alternativamente, se preferir usar o Python do sistema:

```bash
pip install -r requirements.txt
```

## Uso

Transcrever um arquivo e gerar legendas `.srt` (padrão):

```bash
./.venv/bin/python transcrever.py caminho/para/audio.mp3
```

Escolher formato de saída (`txt` ou `srt`):

```bash
./.venv/bin/python transcrever.py audio.m4a --formato txt
```

Forçar idioma (ex.: português):

```bash
./.venv/bin/python transcrever.py audio.mp4 --idioma pt
```

Selecionar modelo (qualidade x velocidade): `tiny`, `base`, `small`, `medium`, `large-v3`

```bash
./.venv/bin/python transcrever.py audio.wav --modelo small
```

Saída em pasta padrão:

```bash
./.venv/bin/python transcrever.py audio.wav
# Gera arquivos em ./transcricoes/
```

Definir outro diretório de saída:

```bash
./.venv/bin/python transcrever.py audio.wav --saida resultados/
```

Ajustar dispositivo e precisão (CPU padrão):

```bash
./.venv/bin/python transcrever.py audio.wav --dispositivo cpu --tipo-computo int8
```

Desativar VAD (detecção de voz):

```bash
./.venv/bin/python transcrever.py audio.wav --sem-vad
```

## Dicas de modelo

- `tiny/base`: muito rápidos, menor qualidade
- `small`: bom equilíbrio (CPU)
- `medium`: melhor qualidade, mais pesado
- `large-v3`: ótima qualidade, mais lento em CPU

## Observações

- FFmpeg é obrigatório para leitura de formatos (mp3/m4a/mp4/etc.).
- Ao usar vídeos como `.mp4`, o script converte automaticamente para WAV 16 kHz mono (PCM) antes da transcrição para máxima compatibilidade com o `whisper-cli`.
- Em CPU, modelos menores (`tiny/base/small`) são mais rápidos; modelos maiores melhoram a qualidade.

## Progresso da transcrição

- O script mostra uma barra de progresso estimada baseada na duração do áudio e no RTF (Real-Time Factor).
- Por padrão, usa `--rtf 1.0` (tempo do áudio). Se sua máquina transcreve mais rápido, use valores maiores, por exemplo: `--rtf 2.0`.

Exemplo com progresso acelerado estimado:

```bash
./.venv/bin/python transcrever.py video.mp4 --rtf 2.0
```

Observação: O progresso é uma estimativa; o tempo real pode variar conforme modelo e hardware.

## VAD (detecção de voz)

- Use `--vad` para transcrever apenas trechos com fala e reduzir música/ruído.
- O áudio é convertido para WAV 16 kHz mono e segmentado com VAD; cada segmento é transcrito e os timestamps são mesclados no arquivo final.
- Exemplo:

```bash
./.venv/bin/python transcrever.py video.mp4 --vad --idioma pt --formato srt
```

Notas:
- Se o VAD não detectar voz, o script avisa e cai para transcrição completa.
- Você pode combinar com `--beam-size 8` e modelos maiores (`medium`, `large-v3`) para melhor qualidade.
- A transcrição em português melhora com modelos maiores.

## Modo Interativo (Menu)

Se preferir configurar as opções interativamente e informar os arquivos depois:

```bash
./.venv/bin/python transcrever.py --menu
```

No menu você pode definir:
- Formato (`srt`/`txt`)
- Idioma (ou detecção automática)
- Modelo (`tiny`, `base`, `small`, `medium`, `large-v3` ou caminho para `.bin`)
- Beam size, RTF, manter marcadores, ativar VAD
- Caminhos dos arquivos (separados por vírgula ou espaço)

O modo interativo executa a transcrição com as opções escolhidas e exibe o progresso.