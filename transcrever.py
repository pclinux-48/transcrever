#!/usr/bin/env python3
import argparse
import os
import sys
import shutil
import subprocess
import threading
import time
import wave
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn

console = Console()


HF_BASE = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main"
MODEL_MAP_NEW = {
    "tiny": "ggml-model-whisper-tiny.bin",
    "base": "ggml-model-whisper-base.bin",
    "small": "ggml-model-whisper-small.bin",
    "medium": "ggml-model-whisper-medium.bin",
    "large-v3": "ggml-model-whisper-large-v3.bin",
}
MODEL_MAP_OLD = {
    "tiny": "ggml-tiny.bin",
    "base": "ggml-base.bin",
    "small": "ggml-small.bin",
    "medium": "ggml-medium.bin",
    "large-v3": "ggml-large-v3.bin",
}


def verificar_ffmpeg():
    return shutil.which("ffmpeg") is not None


def encontrar_whisper_bin() -> str | None:
    # Primeiro tenta no PATH
    for name in ["whisper-cli", "whisper-cpp", "whisper", "whisper-stream"]:
        path = shutil.which(name)
        if path:
            return path
    if path:
        return path
    # Caminho padrão do Homebrew (Apple Silicon)
    for name in [
        "/opt/homebrew/bin/whisper-cli",
        "/opt/homebrew/bin/whisper-cpp",
        "/opt/homebrew/bin/whisper",
        "/opt/homebrew/bin/whisper-stream",
        "/usr/local/bin/whisper-cli",
        "/usr/local/bin/whisper-cpp",
        "/usr/local/bin/whisper",
        "/usr/local/bin/whisper-stream",
    ]:
        if os.path.exists(name):
            return name
    return None


def baixar_modelo(modelo: str, pasta_modelos: Path) -> Path:
    pasta_modelos.mkdir(parents=True, exist_ok=True)
    if modelo in MODEL_MAP_NEW:
        nome_arquivo = MODEL_MAP_NEW[modelo]
    else:
        # Caminho customizado
        return Path(modelo)

    destino = pasta_modelos / nome_arquivo
    # Se o arquivo "novo" existe mas é muito pequeno (stub), não usar
    if destino.exists():
        try:
            if destino.stat().st_size >= 10 * 1024 * 1024:  # >10MB deve ser um binário real
                return destino
            else:
                console.print(
                    f"[yellow]Modelo '{destino.name}' parece inválido/placeholder (tamanho {destino.stat().st_size} bytes). Tentando fallback...[/]"
                )
                # Remove stub para permitir novo download
                try:
                    destino.unlink()
                except Exception:
                    pass
        except Exception:
            # Em caso de erro ao checar, prossegue com fallback
            pass
    # Fallback: se existir arquivo antigo, usar
    antigo = None
    if modelo in MODEL_MAP_OLD:
        possivel_antigo = pasta_modelos / MODEL_MAP_OLD[modelo]
        if possivel_antigo.exists():
            antigo = possivel_antigo
    if antigo:
        try:
            if antigo.stat().st_size >= 10 * 1024 * 1024:
                console.print(
                    f"[yellow]Usando modelo antigo encontrado:[/] {antigo} (pode causar incompatibilidades)"
                )
                return antigo
        except Exception:
            pass
    

    # Tenta via huggingface_hub (mais robusto)
    console.print(f"Baixando modelo: [cyan]{modelo}[/] -> {destino}")
    try:
        from huggingface_hub import hf_hub_download

        downloaded_path = hf_hub_download(
            repo_id="ggerganov/whisper.cpp",
            filename=nome_arquivo,
            repo_type="model",
            local_dir=pasta_modelos.as_posix(),
            local_dir_use_symlinks=False,
        )
        # Verifica tamanho para evitar arquivo placeholder
        try:
            if os.path.getsize(downloaded_path) < 10 * 1024 * 1024:
                console.print(
                    f"[yellow]Download via HuggingFace retornou arquivo muito pequeno ({os.path.getsize(downloaded_path)} bytes). Tentando mirrors alternativos...[/]"
                )
                raise RuntimeError("Arquivo de modelo inválido (tamanho pequeno)")
        except Exception:
            # força fallback para mirrors
            raise
        return Path(downloaded_path)
    except Exception as e:
        # Fallback para urllib com mirror direto
        import urllib.request
        mirrors = [
            f"{HF_BASE}/{nome_arquivo}",
            f"https://ggml.ggerganov.com/{nome_arquivo}",
        ]
        for url in mirrors:
            try:
                with urllib.request.urlopen(url) as resp, open(destino, "wb") as out:
                    shutil.copyfileobj(resp, out)
                # Verifica tamanho após download
                try:
                    if destino.stat().st_size < 10 * 1024 * 1024:
                        console.print(
                            f"[red]Modelo baixado de {url} ainda parece inválido (tamanho muito pequeno).[/]"
                        )
                        continue
                except Exception:
                    continue
                return destino
            except Exception:
                continue
        raise RuntimeError(
            f"Falha ao baixar modelo '{modelo}': {e}. Tente baixar manualmente e informar com --modelo"
        )
    return destino


def transcrever_arquivo(
    arquivo: str,
    modelo: str,
    idioma: str | None,
    formato: str,
    saida_dir: str | None,
    beam_size: int,
    rtf: float,
    manter_marcadores: bool,
    usar_vad: bool,
):
    if not os.path.isfile(arquivo):
        raise FileNotFoundError(f"Arquivo não encontrado: {arquivo}")

    base_nome = os.path.splitext(os.path.basename(arquivo))[0]
    out_dir = Path(saida_dir) if saida_dir else Path(os.path.dirname(arquivo))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_prefix = out_dir / base_nome

    # Converte sempre para WAV 16 kHz mono PCM para máxima compatibilidade com whisper-cli
    tmp_wav = out_dir / f"{base_nome}.conv.wav"
    ffmpeg_cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        arquivo,
        "-ar",
        "16000",
        "-ac",
        "1",
        "-c:a",
        "pcm_s16le",
        tmp_wav.as_posix(),
    ]

    conv = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
    if conv.returncode != 0:
        raise RuntimeError(
            f"Falha ao converter áudio para WAV (ffmpeg):\n{conv.stderr}"
        )
    arquivo_processado = tmp_wav.as_posix()

    # Obtém duração do áudio convertido via ffprobe (segundos)
    duracao_seg = 0.0
    try:
        ffprobe_cmd = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            arquivo_processado,
        ]
        pr = subprocess.run(ffprobe_cmd, capture_output=True, text=True)
        if pr.returncode == 0 and pr.stdout.strip():
            duracao_seg = float(pr.stdout.strip())
    except Exception:
        duracao_seg = 0.0

    modelos_dir = Path("models")
    caminho_modelo = baixar_modelo(modelo, modelos_dir)

    exe = encontrar_whisper_bin() or "whisper-cli"
    cmd = [
        exe,
        "-m",
        str(caminho_modelo),
        "-f",
        arquivo_processado,
        "-of",
        str(out_prefix),
    ]

    if idioma:
        cmd += ["-l", idioma]
    else:
        cmd += ["-di"]  # detect language

    # Formato de saída
    if formato == "srt":
        cmd += ["-osrt"]
    elif formato == "txt":
        cmd += ["-otxt"]
    else:
        raise ValueError("Formato inválido. Use 'txt' ou 'srt'.")

    if beam_size:
        cmd += ["-bs", str(beam_size)]

    if usar_vad:
        # --- VAD com webrtcvad ---
        try:
            import webrtcvad

            # Lê PCM do WAV
            with wave.open(arquivo_processado, "rb") as wf:
                if wf.getsampwidth() != 2 or wf.getnchannels() != 1 or wf.getframerate() != 16000:
                    raise RuntimeError("WAV temporário não está em PCM 16 kHz mono")
                n_frames = wf.getnframes()
                pcm = wf.readframes(n_frames)
            vad = webrtcvad.Vad(2)  # 0-3 (mais alto = mais agressivo)

            frame_ms = 30
            bytes_per_sample = 2
            samples_per_frame = int(16000 * frame_ms / 1000)  # 480
            bytes_per_frame = samples_per_frame * bytes_per_sample  # 960

            frames_voiced = []
            num_frames = len(pcm) // bytes_per_frame
            for i in range(num_frames):
                start_b = i * bytes_per_frame
                end_b = start_b + bytes_per_frame
                frame = pcm[start_b:end_b]
                is_voiced = vad.is_speech(frame, 16000)
                frames_voiced.append(is_voiced)

            # Agrupa em segmentos (voiced consecutivos)
            segs = []
            in_seg = False
            seg_start = 0
            for i, v in enumerate(frames_voiced):
                if v and not in_seg:
                    in_seg = True
                    seg_start = i
                elif not v and in_seg:
                    in_seg = False
                    seg_end = i
                    segs.append((seg_start, seg_end))
            if in_seg:
                segs.append((seg_start, len(frames_voiced)))

            # Converte para segundos e filtra segmentos curtos
            padding_s = 0.2
            min_len_s = 1.2
            segmentos = []
            for s, e in segs:
                start_s = s * frame_ms / 1000.0
                end_s = e * frame_ms / 1000.0
                if (end_s - start_s) >= min_len_s:
                    segmentos.append((max(0.0, start_s - padding_s), end_s + padding_s))

            # Mescla segmentos com gaps pequenos
            merged = []
            for seg in segmentos:
                if not merged:
                    merged.append(list(seg))
                else:
                    prev = merged[-1]
                    if seg[0] - prev[1] <= 0.3:  # gap <= 300ms
                        prev[1] = max(prev[1], seg[1])
                    else:
                        merged.append(list(seg))

            if not merged:
                console.print("[yellow]Aviso:[/] VAD não detectou fala. Transcrevendo áudio completo.")
                usar_vad = False  # cai no caminho normal abaixo
            else:
                # Progresso com base no total dos segmentos
                total_seg = sum(e - s for s, e in merged)
                expected_total = total_seg / rtf if (rtf > 0 and total_seg > 0) else None
                acumulado = 0.0

                # Saída final acumulada
                srt_blocos = []
                txt_linhas = []

                with Progress(
                    SpinnerColumn(),
                    TextColumn("[bold green]Transcrevendo[/] {task.description} (VAD)"),
                    BarColumn(),
                    TimeElapsedColumn(),
                    TimeRemainingColumn(),
                    console=console,
                ) as progress:
                    task_id = progress.add_task(
                        description=os.path.basename(arquivo), total=(expected_total or None)
                    )
                    for idx, (start_s, end_s) in enumerate(merged, 1):
                        seg_path = out_dir / f"{base_nome}.seg{idx}.wav"
                        # Corta com ffmpeg (copia PCM)
                        cut_cmd = [
                            "ffmpeg",
                            "-hide_banner",
                            "-loglevel",
                            "error",
                            "-y",
                            "-ss",
                            str(start_s),
                            "-to",
                            str(end_s),
                            "-i",
                            arquivo_processado,
                            "-c",
                            "copy",
                            seg_path.as_posix(),
                        ]
                        cut = subprocess.run(cut_cmd, capture_output=True, text=True)
                        if cut.returncode != 0:
                            console.print(f"[red]Erro ao cortar segmento {idx}:[/] {cut.stderr}")
                            continue

                        # Transcreve segmento
                        seg_prefix = out_dir / f"{base_nome}.seg{idx}"
                        seg_cmd = [exe, "-m", str(caminho_modelo), "-f", seg_path.as_posix(), "-of", seg_prefix.as_posix()]
                        if idioma:
                            seg_cmd += ["-l", idioma]
                        else:
                            seg_cmd += ["-di"]
                        if formato == "srt":
                            seg_cmd += ["-osrt"]
                        else:
                            seg_cmd += ["-otxt"]
                        if beam_size:
                            seg_cmd += ["-bs", str(beam_size)]

                        seg_proc = subprocess.run(seg_cmd, capture_output=True, text=True)
                        if seg_proc.returncode != 0:
                            console.print(f"[red]Segmento {idx} falhou:[/] {seg_proc.stderr}")
                            continue

                        # Carrega saída do segmento e acumula
                        if formato == "srt":
                            seg_srt = seg_prefix.as_posix() + ".srt"
                            if not os.path.exists(seg_srt):
                                seg_srt = seg_prefix.as_posix() + ".wav.srt"
                            if os.path.exists(seg_srt):
                                with open(seg_srt, "r", encoding="utf-8", errors="ignore") as f:
                                    linhas = f.read().splitlines()
                                # Offset timestamps
                                def parse_ts(ts: str) -> float:
                                    hh, mm, rest = ts.split(":")
                                    ss, ms = rest.split(",")
                                    return int(hh) * 3600 + int(mm) * 60 + int(ss) + int(ms) / 1000.0
                                def fmt_ts(sec: float) -> str:
                                    if sec < 0:
                                        sec = 0
                                    ms = int(round((sec - int(sec)) * 1000))
                                    s = int(sec) % 60
                                    m = (int(sec) // 60) % 60
                                    h = int(sec) // 3600
                                    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
                                bloco = []
                                for ln in linhas:
                                    if "-->" in ln:
                                        a, b = [x.strip() for x in ln.split("-->")]
                                        a2 = fmt_ts(parse_ts(a) + start_s)
                                        b2 = fmt_ts(parse_ts(b) + start_s)
                                        bloco.append(f"{a2} --> {b2}")
                                    else:
                                        bloco.append(ln)
                                srt_blocos.append(bloco)
                        else:
                            seg_txt = seg_prefix.as_posix() + ".txt"
                            if os.path.exists(seg_txt):
                                with open(seg_txt, "r", encoding="utf-8", errors="ignore") as f:
                                    txt_linhas.extend(f.read().splitlines())

                        # Atualiza progresso
                        acumulado += (end_s - start_s) / (rtf if rtf > 0 else 1.0)
                        try:
                            progress.update(task_id, completed=acumulado)
                        except Exception:
                            pass

                        # Limpa arquivo do segmento
                        try:
                            if seg_path.exists():
                                seg_path.unlink()
                        except Exception:
                            pass

                # Escreve saída final
                if formato == "srt":
                    # Renumera blocos
                    linhas_out = []
                    num = 1
                    for bloco in srt_blocos:
                        # reconstroi bloco: primeira linha deve ser índice? Alguns blocos incluem índice
                        # Vamos inserir índice e garantir separação por linha vazia
                        linhas_out.append(str(num))
                        # bloco contém linhas já com timestamps e texto
                        for ln in bloco:
                            linhas_out.append(ln)
                        linhas_out.append("")
                        num += 1
                    with open(out_prefix.as_posix() + (".srt" if formato == "srt" else ".txt"), "w", encoding="utf-8") as f:
                        f.write("\n".join(linhas_out).rstrip() + "\n")
                else:
                    with open(out_prefix.as_posix() + ".txt", "w", encoding="utf-8") as f:
                        f.write("\n".join(txt_linhas).rstrip() + "\n")
        except Exception as e:
            console.print(f"[yellow]Aviso:[/] falha no VAD ({e}). Usando transcrição completa.")
            usar_vad = False

    if not usar_vad:
        # Caminho original: transcrever arquivo inteiro com barra estimada
        expected_total = duracao_seg / rtf if (rtf > 0 and duracao_seg > 0) else None
        stderr_lines: list[str] = []
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )

        def drain_stderr():
            if proc.stderr:
                for line in proc.stderr:
                    stderr_lines.append(line)

        t = threading.Thread(target=drain_stderr, daemon=True)
        t.start()

        if expected_total is None:
            # Sem duração conhecida: mostrar spinner simples
            with Progress(
                SpinnerColumn(),
                TextColumn("[bold green]Transcrevendo[/] {task.description}"),
                console=console,
            ) as progress:
                progress.add_task(description=os.path.basename(arquivo), total=None)
                while proc.poll() is None:
                    time.sleep(0.2)
        else:
            # Com duração conhecida: mostrar barra e tempo estimado (RTF)
            with Progress(
                SpinnerColumn(),
                TextColumn("[bold green]Transcrevendo[/] {task.description} (estimado)"),
                BarColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                console=console,
            ) as progress:
                task_id = progress.add_task(
                    description=os.path.basename(arquivo), total=expected_total
                )
                start = time.time()
                while proc.poll() is None:
                    elapsed = time.time() - start
                    progress.update(task_id, completed=min(elapsed, expected_total * 0.99))
                    time.sleep(0.2)
                progress.update(task_id, completed=expected_total)

        t.join(timeout=1.0)
        stderr_text = "".join(stderr_lines)

        if proc.returncode != 0:
            # Remove o WAV temporário antes de falhar
            try:
                if tmp_wav.exists():
                    tmp_wav.unlink()
            except Exception:
                pass
            raise RuntimeError(
                f"whisper-cpp falhou (código {proc.returncode}):\n{stderr_text}"
            )

    # Arquivo de saída conforme formato
    ext = ".srt" if formato == "srt" else ".txt"
    saida_arquivo = out_prefix.as_posix() + ext
    if not os.path.exists(saida_arquivo):
        # Alguns builds salvam como `<prefix>.srt` ou `<prefix>.<orig_ext>.srt`. Tentar variações:
        input_ext = os.path.splitext(arquivo)[1]
        alt = out_prefix.with_suffix("" if out_prefix.suffix else "")
        possiveis = [
            out_prefix.as_posix() + ext,
            (alt.as_posix() + ext),
            (out_prefix.as_posix() + input_ext + ext),
            (out_prefix.as_posix() + ".wav" + ext),
            (out_prefix.as_posix() + ".mp3" + ext),
            (out_prefix.as_posix() + ".mp4" + ext),
            (out_prefix.as_posix() + ".m4a" + ext),
        ]
        for p in possiveis:
            if os.path.exists(p):
                saida_arquivo = p
                break

    # Remove WAV temporário
    try:
        if tmp_wav.exists():
            tmp_wav.unlink()
    except Exception:
        pass

    # Verifica se o arquivo final realmente existe
    if not os.path.exists(saida_arquivo):
        raise RuntimeError(
            "Transcrição final não encontrada. Verifique se o 'whisper-cli' gerou a saída corretamente."
        )

    return saida_arquivo


def _remover_marcadores_texto(linhas: list[str]) -> list[str]:
    etiquetas = {
        "[Música]",
        "[Music]",
        "[Aplausos]",
        "[Applause]",
        "[Risos]",
        "[Laughter]",
        "[Ruído]",
        "[Noise]",
        "[Silêncio]",
        "[Silence]",
    }
    res = []
    for ln in linhas:
        t = ln.strip()
        if t in etiquetas:
            continue
        # Remove linhas que são apenas uma etiqueta genérica entre colchetes
        if t.startswith("[") and t.endswith("]") and 1 <= len(t) <= 40 and " " not in t[1:-1]:
            # algo como [Música] / [Music] / [Risos]
            continue
        res.append(ln)
    return res


def posprocessar_saida(saida_arquivo: str, formato: str, manter_marcadores: bool) -> None:
    if manter_marcadores:
        return
    try:
        if formato == "srt":
            # Filtra blocos que contêm apenas etiquetas e renumera
            with open(saida_arquivo, "r", encoding="utf-8", errors="ignore") as f:
                conteudo = f.read().splitlines()

            blocos = []
            bloco = []
            for ln in conteudo:
                if ln.strip() == "":
                    if bloco:
                        blocos.append(bloco)
                        bloco = []
                else:
                    bloco.append(ln)
            if bloco:
                blocos.append(bloco)

            novos_blocos = []
            for b in blocos:
                if len(b) < 3:
                    continue
                idx = b[0]
                tempo = b[1]
                texto = b[2:]
                texto_filtrado = _remover_marcadores_texto(texto)
                if not texto_filtrado:
                    # descarta bloco sem fala
                    continue
                novos_blocos.append(["", tempo] + texto_filtrado)

            if not novos_blocos:
                # Se após a limpeza não sobrar nenhum bloco, manter o arquivo original para evitar SRT vazio
                try:
                    console.print(
                        "[yellow]Aviso:[/] nenhum bloco de fala após limpeza de marcadores. Mantendo o arquivo original."
                    )
                except Exception:
                    pass
                return
            else:
                # Renumera índices
                linhas_out = []
                num = 1
                for b in novos_blocos:
                    linhas_out.append(str(num))
                    linhas_out.append(b[1])
                    for ln in b[2:]:
                        linhas_out.append(ln)
                    linhas_out.append("")
                    num += 1

                with open(saida_arquivo, "w", encoding="utf-8") as f:
                    f.write("\n".join(linhas_out).rstrip() + "\n")
        else:
            # TXT: remove linhas que são apenas etiquetas
            with open(saida_arquivo, "r", encoding="utf-8", errors="ignore") as f:
                linhas = f.read().splitlines()
            filtradas = _remover_marcadores_texto(linhas)
            if not filtradas:
                # Evita apagar todo conteúdo
                return
            with open(saida_arquivo, "w", encoding="utf-8") as f:
                f.write("\n".join(filtradas).rstrip() + "\n")
    except Exception as _:
        # Em caso de erro na limpeza, mantemos arquivo original
        pass


def parse_args():
    p = argparse.ArgumentParser(
        description="Transcrição offline usando whisper-cpp (sem APIs pagas).",
    )
    p.add_argument(
        "arquivos",
        nargs="*",
        help="Caminho(s) dos arquivo(s) de áudio/vídeo (mp3, m4a, wav, mp4, etc.). No modo --menu, serão solicitados interativamente.",
    )
    p.add_argument(
        "--modelo",
        default="small",
        help="Modelo: tiny, base, small, medium, large-v3 ou caminho para .bin.",
    )
    p.add_argument(
        "--idioma",
        default=None,
        help="Forçar idioma (ex.: 'pt' para português). Se omitido, detecta automaticamente.",
    )
    p.add_argument(
        "--formato",
        choices=["txt", "srt"],
        default="srt",
        help="Formato de saída: txt ou srt (padrão: srt).",
    )
    p.add_argument(
        "--saida",
        default="transcricoes",
        help="Diretório de saída. Padrão: 'transcricoes/'.",
    )
    p.add_argument(
        "--beam-size",
        type=int,
        default=5,
        help="Beam size para decodificação (maior pode melhorar qualidade, porém mais lento).",
    )
    p.add_argument(
        "--rtf",
        type=float,
        default=1.0,
        help="Real-Time Factor (RTF) estimado. 1.0 = tempo do áudio. Use >1.0 se seu hardware for mais rápido que tempo real para uma estimativa de progresso mais fiel.",
    )
    p.add_argument(
        "--manter-marcadores",
        action="store_true",
        help="Mantém marcadores como [Música], [Risos], etc. Por padrão, eles são suprimidos no arquivo final.",
    )
    p.add_argument(
        "--vad",
        action="store_true",
        help="Ativa VAD (detecção de voz) para transcrever somente trechos com fala e reduzir música/ruído.",
    )
    p.add_argument(
        "--menu",
        action="store_true",
        help="Inicia modo interativo com menu para escolher opções e arquivos.",
    )
    return p.parse_args()


def main():
    if not verificar_ffmpeg():
        console.print(
            "[red]FFmpeg não encontrado.[/] Instale com Homebrew: "
            "[bold]brew install ffmpeg[/bold]"
        )
        sys.exit(1)

    if not encontrar_whisper_bin():
        console.print(
            "[red]whisper (CLI) não encontrado.[/] Instale com Homebrew: "
            "[bold]brew install whisper-cpp[/bold]"
        )
        sys.exit(1)

    args = parse_args()

    def menu_interativo():
        console.print("[bold cyan]Modo interativo[/] — selecione as opções abaixo.")
        # Formato
        formato = "srt"
        resp = input("Formato de saída (srt/txt) [srt]: ").strip().lower()
        if resp in {"srt", "txt"}:
            formato = resp

        # Idioma
        idioma = None
        resp = input("Idioma forçado (ex.: pt) [auto]: ").strip().lower()
        if resp:
            idioma = resp

        # Modelo
        modelo = "small"
        console.print("Modelos: tiny, base, small, medium, large-v3 ou caminho para .bin")
        resp = input("Modelo [small]: ").strip()
        if resp:
            modelo = resp

        # Beam size
        beam_size = 5
        resp = input("Beam size [5]: ").strip()
        if resp:
            try:
                beam_size = int(resp)
            except Exception:
                console.print("[yellow]Valor inválido para beam size; usando 5.[/]")

        # RTF
        rtf = 1.0
        resp = input("RTF estimado (ex.: 2.0) [1.0]: ").strip()
        if resp:
            try:
                rtf = float(resp)
            except Exception:
                console.print("[yellow]Valor inválido para RTF; usando 1.0.[/]")

        # Manter marcadores
        manter_marcadores = False
        resp = input("Manter marcadores ([Música], etc.)? (s/N) [N]: ").strip().lower()
        if resp in {"s", "sim", "y", "yes"}:
            manter_marcadores = True

        # VAD
        usar_vad = False
        resp = input("Ativar VAD (detecção de voz)? (s/N) [N]: ").strip().lower()
        if resp in {"s", "sim", "y", "yes"}:
            usar_vad = True

        # Arquivos
        console.print("Informe caminho(s) dos arquivos separados por vírgula ou espaço.")
        resp = input("Arquivos: ").strip()
        if not resp:
            raise RuntimeError("Nenhum arquivo informado.")
        # split por vírgula e/ou espaço
        partes = [p for chunk in resp.split(",") for p in chunk.split()] 
        arquivos = [p for p in partes if p]

        console.print("\n[bold]Resumo:[/]")
        console.print(f"Formato: {formato}")
        console.print(f"Idioma: {idioma if idioma else 'auto'}")
        console.print(f"Modelo: {modelo}")
        console.print(f"Beam size: {beam_size}")
        console.print(f"RTF: {rtf}")
        console.print(f"Manter marcadores: {manter_marcadores}")
        console.print(f"VAD: {usar_vad}")
        console.print(f"Arquivos: {', '.join(arquivos)}")
        conf = input("Confirmar e transcrever? (S/n) [S]: ").strip().lower()
        if conf in {"n", "nao", "não", "no"}:
            console.print("[yellow]Cancelado pelo usuário.[/]")
            sys.exit(0)

        for caminho in arquivos:
            saida = transcrever_arquivo(
                arquivo=caminho,
                modelo=modelo,
                idioma=idioma,
                formato=formato,
                saida_dir=args.saida,
                beam_size=beam_size,
                rtf=rtf,
                manter_marcadores=manter_marcadores,
                usar_vad=usar_vad,
            )
            posprocessar_saida(saida, formato, manter_marcadores)
            console.print(f"✅ Arquivo transcrito: [bold]{saida}[/bold]")

    # Se modo menu ou nenhum arquivo passado, abrir menu
    if args.menu or not args.arquivos:
        try:
            menu_interativo()
            return
        except Exception as e:
            console.print(f"[red]Erro no modo interativo:[/] {e}")
            sys.exit(2)

    try:
        for caminho in args.arquivos:
            saida = transcrever_arquivo(
                arquivo=caminho,
                modelo=args.modelo,
                idioma=args.idioma,
                formato=args.formato,
                saida_dir=args.saida,
                beam_size=args.beam_size,
                rtf=args.rtf,
                manter_marcadores=args.manter_marcadores,
                usar_vad=args.vad,
            )
            # Limpa etiquetas caso não sejam mantidas
            posprocessar_saida(saida, args.formato, args.manter_marcadores)
            console.print(f"✅ Arquivo transcrito: [bold]{saida}[/bold]")
    except Exception as e:
        console.print(f"[red]Erro:[/] {e}")
        sys.exit(2)


if __name__ == "__main__":
    main()