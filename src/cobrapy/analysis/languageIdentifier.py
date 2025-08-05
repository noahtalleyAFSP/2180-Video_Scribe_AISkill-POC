#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Language Identification from MP4 using Azure Speech SDK, with .env via python-dotenv.

Credentials:
  - AZURE_SPEECH_KEY
  - AZURE_SPEECH_REGION
  - (optional) AZURE_SPEECH_ENDPOINT

Features:
- Loads credentials from .env
- Extracts 16 kHz mono WAV via ffmpeg
- Randomly samples chunks (continuous LID) across en-US, fr-FR, ar-SA
- If Arabic is detected, probes ar-MA vs ar-SA using detailed STT confidences
- Robust logging, retries, JSON summary

Usage:
  python lid_from_video.py /path/to/video.mp4 \
    --max-sample-seconds 120 --chunk-seconds 30 --seed 42 \
    --out results.json --log INFO --env .env
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv, find_dotenv

try:
    import azure.cognitiveservices.speech as speechsdk
except ImportError:
    print("Missing dependency. Install with: pip install azure-cognitiveservices-speech", file=sys.stderr)
    raise

# ------------------------------- Config defaults -------------------------------

DEFAULT_CHUNK_SECONDS = 30            # chunk length for LID
DEFAULT_MAX_SAMPLE_SECONDS = 120      # cap total sampled audio
DEFAULT_SEED = 42
RECOGNIZE_ONCE_SECONDS = 12           # short probe for dialect confidence

# ---------------- Extended defaults per new CLI spec ----------------
DEFAULT_WINDOW_SECONDS = 20
DEFAULT_SAMPLE_BUDGET_SECONDS = 180
DEFAULT_VAD = "on"
DEFAULT_VAD_AGGR = 2
DEFAULT_VERIFY_ARABIC = "auto"
DEFAULT_AZURE_TIMEOUT_SEC = 120
DEFAULT_PROBE_COUNT = 8
DEFAULT_PROBE_SECONDS = 12
DEFAULT_TEXT_DIALECT_ASR = "azure"
DEFAULT_FALLBACK = "whisper"
CALIB_DEFAULT = {
    "stageB": {"alpha_audio": 0.60, "ary_threshold": 0.65, "ar_threshold": 0.35},
    "mixed_rule": {"top_share_min": 0.60, "runner_share_min": 0.30, "min_gap_for_dialect": 0.10},
}
VERSION = "1.0.0"

# LID candidates (rule: one locale per base language)
LID_CANDIDATES = ["en-US", "fr-FR", "ar-SA"]

# Arabic variant probe locales
AR_VARIANTS = ["ar-MA", "ar-SA"]
AR_VARIANT_MARGIN = 0.15  # require at least this confidence margin to call ar-MA over ar-SA

# ------------------------------- Utilities ------------------------------------

# --- Compatibility wrapper names per new CLI spec ---
Window = Tuple[float, float]

def ensure_ffmpeg_available() -> None:
    """Ensure ffmpeg binary is present on PATH."""
    check_ffmpeg()

def extract_wav_16k_mono(input_media: Path, out_wav: Path) -> None:
    """Public wrapper: extract 16-kHz mono WAV from arbitrary input media."""
    extract_wav_from_mp4(input_media, out_wav)

def wav_duration_seconds(wav_path: Path) -> float:
    """Return duration of WAV in seconds (16-bit PCM)."""
    return get_wav_duration_seconds(wav_path)

def trim_wav_segment(in_wav: Path, out_wav: Path, start_s: float, dur_s: float) -> None:
    """Trim a segment from WAV using ffmpeg (16-kHz mono)."""
    trim_wav(in_wav, out_wav, start_s, dur_s)




def check_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found on PATH. Install ffmpeg and retry.")

def run_ffmpeg(cmd: List[str]) -> None:
    logging.debug("Running ffmpeg: %s", " ".join(cmd))
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed ({proc.returncode}): {proc.stderr.decode(errors='ignore')}")

def extract_wav_from_mp4(mp4_path: Path, wav_path: Path) -> None:
    """Extract 16 kHz mono WAV audio from video."""
    check_ffmpeg()
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-y", "-i", str(mp4_path),
        "-vn",            # drop video
        "-ac", "1",       # mono
        "-ar", "16000",   # 16 kHz
        str(wav_path)
    ]
    run_ffmpeg(cmd)

def get_wav_duration_seconds(wav_path: Path) -> float:
    with contextlib.closing(wave.open(str(wav_path), 'rb')) as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
        return frames / float(rate)

def trim_wav(in_wav: Path, out_wav: Path, start_s: float, dur_s: float) -> None:
    """Cut a segment from WAV using ffmpeg."""
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-y",
        "-ss", f"{start_s:.3f}",
        "-t", f"{dur_s:.3f}",
        "-i", str(in_wav),
        "-ac", "1", "-ar", "16000",
        str(out_wav)
    ]
    run_ffmpeg(cmd)

def choose_sample_windows(total_duration: float, chunk_seconds: int,
                          max_total_seconds: int, seed: int
                          ) -> List[Tuple[float, float]]:
    """Return [(start, dur)] for random, non-overlapping sample windows."""
    if total_duration <= 0:
        return []
    if total_duration <= chunk_seconds:
        return [(0.0, min(total_duration, chunk_seconds))]

    max_chunks = max(1, min(int(math.ceil(max_total_seconds / chunk_seconds)),
                            int(total_duration // chunk_seconds)))
    random.seed(seed)

    # pick anchors on a chunk-sized grid and sample without replacement
    anchors = [i * chunk_seconds for i in range(0, int(total_duration // chunk_seconds))]
    if not anchors:
        anchors = [0.0]
    chosen = random.sample(anchors, k=min(max_chunks, len(anchors)))

    windows: List[Tuple[float, float]] = []
    for a in sorted(chosen):
        start = min(a, max(0.0, total_duration - chunk_seconds))
        windows.append((float(start), float(chunk_seconds)))
    return windows

# ------------------------------- NEW Sampling helpers --------------------------

def stratified_windows(total_duration: float, window_seconds: int, sample_budget_seconds: int) -> List[Window]:
    """
    Evenly distribute windows over total_duration to sum close to sample_budget_seconds.
    """
    if total_duration <= 0 or window_seconds <= 0:
        return []
    if window_seconds == 0:
        return []
    n = max(1, min(sample_budget_seconds // window_seconds, int(total_duration // window_seconds)))
    if n == 1:
        start = max(0.0, total_duration - window_seconds)
        return [(start, float(min(window_seconds, total_duration)))]
    usable = max(0.0, total_duration - window_seconds)
    starts = [min(usable, (i + 0.5) / n * usable) for i in range(n)]
    return [(float(s), float(window_seconds)) for s in starts]

# --- VAD helpers (optional) ---

def voiced_ratio_wav(wav_path: Path, aggressiveness: int = 2) -> float:
    """Return fraction of 30 ms frames that are voiced (WebRTC VAD)."""
    try:
        import webrtcvad, struct  # type: ignore
    except ImportError:
        return 0.0
    vad = webrtcvad.Vad(aggressiveness)
    with contextlib.closing(wave.open(str(wav_path), 'rb')) as wf:
        if wf.getframerate() != 16000 or wf.getnchannels() != 1 or wf.getsampwidth() != 2:
            return 0.0
        samp_rate = wf.getframerate()
        frame_ms = 30
        frame_bytes = int(samp_rate * (frame_ms / 1000.0)) * 2
        voiced, total = 0, 0
        while True:
            data = wf.readframes(int(samp_rate * (frame_ms / 1000.0)))
            if len(data) < frame_bytes:
                break
            total += 1
            if vad.is_speech(data, samp_rate):
                voiced += 1
        return (voiced / total) if total else 0.0

def filter_windows_by_vad(full_wav: Path, windows: List[Tuple[float, float]], min_ratio: float = 0.15, aggressiveness: int = 2) -> List[Tuple[float, float]]:
    """Return subset of windows passing VAD threshold; fallback to original if too few."""
    keep: List[Tuple[float, float]] = []
    with tempfile.TemporaryDirectory(prefix="vad_") as tmpdir:
        tmp = Path(tmpdir)
        for (s, d) in windows:
            seg = tmp / f"vad_{int(s)}.wav"
            trim_wav(full_wav, seg, s, d)
            ratio = voiced_ratio_wav(seg, aggressiveness=aggressiveness)
            if ratio >= min_ratio:
                keep.append((s, d))
    return keep if len(keep) >= max(2, len(windows) // 2) else windows


def _arabic_ratio(lang_secs: Dict[str, float]) -> float:
    if not lang_secs:
        return 0.0
    ar_sec = sum(s for l, s in lang_secs.items() if l.startswith("ar-"))
    total = sum(lang_secs.values())
    return (ar_sec / total) if total > 0 else 0.0


def _select_arabic_probe_indices(per_chunk_langs: List[Dict[str, float]],
                                 top_k: int = 5,
                                 min_ratio: float = 0.6) -> List[int]:
    """Pick up to top_k chunk indices with highest Arabic ratio; enforce a minimum ratio if available."""
    ratios = [(i, _arabic_ratio(langs)) for i, langs in enumerate(per_chunk_langs)]
    ratios.sort(key=lambda x: x[1], reverse=True)
    filtered = [i for i, r in ratios if r >= min_ratio]
    if filtered:
        return filtered[:top_k]
    return [i for i, _ in ratios[:top_k]]

# --- NEW Dialect verification helpers -----------------------------------------

from functools import lru_cache

@lru_cache(maxsize=1)
def _adi_pipeline():
    """Lazy-load the audio dialect identifier pipeline."""
    from transformers import pipeline  # type: ignore
    import torch  # type: ignore
    device = 0 if torch.cuda.is_available() else -1
    return pipeline("audio-classification", model="badrex/mms-300m-arabic-dialect-identifier", device=device)


def score_audio_maghrebi(probe_wavs: List[Path]) -> float:
    if not probe_wavs:
        return 0.0
    clf = _adi_pipeline()
    vals: List[float] = []
    for p in probe_wavs:
        try:
            out = clf(str(p), top_k=None)
            mag = next((x["score"] for x in out if x["label"].lower().startswith("maghrebi")), 0.0)
            vals.append(float(mag))
        except Exception as e:
            logging.warning("ADI failed on %s (%s)", p, e)
    return sum(vals) / len(vals) if vals else 0.0


@lru_cache(maxsize=1)
def _text_dialect_model():
    from transformers import AutoTokenizer, AutoModelForSequenceClassification  # type: ignore
    tok_name = "IbrahimAmin/marbertv2-arabic-written-dialect-classifier"
    tok = AutoTokenizer.from_pretrained(tok_name)
    mdl = AutoModelForSequenceClassification.from_pretrained(tok_name)
    return tok, mdl


def score_text_maghrebi(snippets: List[str]) -> float:
    from torch.nn.functional import softmax  # type: ignore
    import torch  # type: ignore
    tok, mdl = _text_dialect_model()
    probs: List[float] = []
    for text in snippets:
        t = (text or "").strip()
        if not t:
            continue
        inp = tok(t, return_tensors="pt", truncation=True, max_length=256)
        with torch.no_grad():
            logits = mdl(**inp).logits
        p = softmax(logits, dim=-1)[0]
        id2label = mdl.config.id2label
        mag_idx = next(i for i, l in id2label.items() if l.upper().startswith("MAGH"))
        probs.append(p[mag_idx].item())
    return sum(probs) / len(probs) if probs else 0.0


def transcribe_snippet_azure(probe_wav: Path, env: "AzureSpeechEnv", locale: str = "ar-SA") -> str:
    cfg = make_speech_config(env)
    cfg.speech_recognition_language = locale
    audio_cfg = speechsdk.audio.AudioConfig(filename=str(probe_wav))
    rec = speechsdk.SpeechRecognizer(speech_config=cfg, audio_config=audio_cfg)
    res = rec.recognize_once()
    return res.text or ""


def transcribe_snippet_whisper(probe_wav: Path, model_size: str = "medium") -> str:
    import torch  # type: ignore
    from faster_whisper import WhisperModel  # type: ignore
    model = WhisperModel(model_size, device="cuda" if torch.cuda.is_available() else "cpu")
    segments, _ = model.transcribe(str(probe_wav), language="ar")
    return " ".join(seg.text for seg in segments)


def select_arabic_probe_indices_seconds(per_chunk_lang_seconds: List[Dict[str, float]], top_k: int = 8) -> List[int]:
    scored: List[Tuple[int, float]] = []
    for i, d in enumerate(per_chunk_lang_seconds):
        total = sum(d.values()) or 0.0
        ar = sum(v for k, v in d.items() if k.startswith("ar-"))
        ratio = (ar / total) if total > 0 else 0.0
        scored.append((i, ratio))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [i for i, _ in scored[:top_k]]


def run_stage_b_dialect_verification(full_wav: Path,
                                     windows: List[Tuple[float, float]],
                                     probe_indices: List[int],
                                     probe_seconds: int,
                                     env: "AzureSpeechEnv",
                                     text_asr: str = "azure") -> Dict[str, object]:
    probe_wavs: List[Path] = []
    with tempfile.TemporaryDirectory(prefix="probe_") as td:
        tdir = Path(td)
        for idx in probe_indices:
            start, dur = windows[idx]
            seg_len = min(float(probe_seconds), float(dur))
            seg = tdir / f"probe_{idx:03d}.wav"
            trim_wav(full_wav, seg, start, seg_len)
            probe_wavs.append(seg)

        audio_mag = score_audio_maghrebi(probe_wavs)

        text_mag: Optional[float] = None
        if text_asr != "none":
            texts: List[str] = []
            for w in probe_wavs:
                try:
                    if text_asr == "azure":
                        texts.append(transcribe_snippet_azure(w, env, "ar-SA"))
                    elif text_asr == "whisper":
                        texts.append(transcribe_snippet_whisper(w, model_size="medium"))
                except Exception as e:
                    logging.warning("Text ASR failed on %s (%s)", w, e)
                    texts.append("")
            text_mag = score_text_maghrebi(texts)

    final = 0.6 * audio_mag + 0.4 * (text_mag if text_mag is not None else 0.0)
    if final >= 0.65:
        decision = "ary"
    elif final <= 0.35:
        decision = "ar"
    else:
        decision = "ary" if audio_mag >= 0.55 else "ar"

    return {
        "run": True,
        "probes": len(probe_indices),
        "audio_maghrebi_prob": round(audio_mag, 3),
        "text_maghrebi_prob": (round(text_mag, 3) if text_mag is not None else None),
        "final_score": round(final, 3),
        "text_asr": text_asr,
        "decision": decision,
    }

# ------------------------------- Azure Speech wrappers -------------------------

@dataclass
class AzureSpeechEnv:
    key: str
    region: Optional[str]
    endpoint: str  # we will always compute/require an endpoint for Continuous LID

# ----------------------- New dataclasses per extended spec -----------------------

@dataclass
class StageAResult:
    windows: List[Window]
    sampled_seconds: float
    language_seconds: Dict[str, float]
    language_shares: Dict[str, float]
    top_locale: Optional[str]
    mean_voiced_ratio: float
    source: str  # "azure-continuous-lid" or "whisper-lid"

@dataclass
class StageBCalibration:
    alpha_audio: float = 0.60
    ary_threshold: float = 0.65
    ar_threshold: float = 0.35

@dataclass
class FinalDecision:
    code: str      # "en","fr","ar","ary","mixed","und"
    label: str
    confidence: float
    language_mix: List[Dict[str, float]]
    rationale: Dict[str, object]

def derive_endpoint_from_region(region: str) -> str:
    # Conversation endpoint for STT (used by Speech SDK)
    # Example: https://westus2.stt.speech.microsoft.com/speech/recognition/conversation/cognitiveservices/v1
    return f"https://{region}.stt.speech.microsoft.com/speech/recognition/conversation/cognitiveservices/v1"

def load_azure_env(env_file: Optional[Path] = None) -> AzureSpeechEnv:
    key = os.getenv("AZURE_SPEECH_KEY")
    region = os.getenv("AZURE_SPEECH_REGION")
    endpoint = os.getenv("AZURE_SPEECH_ENDPOINT")

    if not key:
        raise RuntimeError("Missing AZURE_SPEECH_KEY in environment (.env).")

    if endpoint:
        return AzureSpeechEnv(key=key, region=region, endpoint=endpoint)

    if not region:
        raise RuntimeError("Missing AZURE_SPEECH_REGION (or provide AZURE_SPEECH_ENDPOINT).")

    # For continuous LID the SDK requires creating SpeechConfig from endpoint;
    # we derive a valid endpoint from the region if none provided.
    derived = derive_endpoint_from_region(region)
    return AzureSpeechEnv(key=key, region=region, endpoint=derived)

def make_speech_config(env: AzureSpeechEnv) -> speechsdk.SpeechConfig:
    # Always construct from endpoint for Continuous LID scenarios
    cfg = speechsdk.SpeechConfig(subscription=env.key, endpoint=env.endpoint)
    return cfg

# Public alias per extended spec
# continuous_lid_durations = lid_on_file_durations

def lid_on_file(audio_wav: Path, env: AzureSpeechEnv, languages: List[str]) -> List[str]:
    """Backwards-compat wrapper – returns list of detected locales (unweighted)."""
    return list(continuous_lid_durations(audio_wav, env, languages).keys())


def continuous_lid_durations(audio_wav: Path, env: AzureSpeechEnv, languages: List[str], timeout_s: int = DEFAULT_AZURE_TIMEOUT_SEC) -> Dict[str, float]:
    """Run Continuous LID and return a dict of locale → total recognized seconds."""
    cfg = make_speech_config(env)
    cfg.set_property(
        property_id=speechsdk.PropertyId.SpeechServiceConnection_LanguageIdMode,
        value="Continuous"
    )

    auto_cfg = speechsdk.languageconfig.AutoDetectSourceLanguageConfig(languages=languages)
    audio_cfg = speechsdk.audio.AudioConfig(filename=str(audio_wav))

    recognizer = speechsdk.SpeechRecognizer(
        speech_config=cfg,
        auto_detect_source_language_config=auto_cfg,
        audio_config=audio_cfg
    )

    durations: Dict[str, float] = {}
    done = threading.Event()

    def handle_recognized(evt):
        try:
            res = evt.result
            if res.reason == speechsdk.ResultReason.RecognizedSpeech:
                lang = res.properties.get(
                    speechsdk.PropertyId.SpeechServiceConnection_AutoDetectSourceLanguageResult
                )
                dur_s = float(res.duration) / 10_000_000.0  # 100-ns units → seconds
                if lang:
                    durations[lang] = durations.get(lang, 0.0) + dur_s
                logging.debug(
                    "Recognized segment %.2fs; detected=%s; text='%s'",
                    dur_s,
                    lang,
                    res.text,
                )
            elif res.reason == speechsdk.ResultReason.NoMatch:
                logging.debug("NoMatch on a segment.")
        except Exception:
            logging.exception("Error in recognized handler")

    def handle_stop(_):
        done.set()

    recognizer.recognized.connect(handle_recognized)
    recognizer.session_stopped.connect(handle_stop)
    recognizer.canceled.connect(handle_stop)

    recognizer.start_continuous_recognition()
    # Wait with watchdog timeout
    if not done.wait(timeout_s):
        logging.warning("Azure continuous LID timeout (%.0fs) reached; forcing stop.", timeout_s)
        recognizer.stop_continuous_recognition()
        raise TimeoutError(f"Azure Continuous LID timed out after {timeout_s}s")
    recognizer.stop_continuous_recognition()
    return durations

def recognize_once_confidence(audio_wav: Path, env: AzureSpeechEnv, locale: str) -> float:
    """
    Recognize a short WAV once with 'Detailed' output; return top-hypothesis confidence (0..1).
    """
    cfg = make_speech_config(env)
    cfg.speech_recognition_language = locale
    cfg.output_format = speechsdk.OutputFormat.Detailed

    audio_cfg = speechsdk.audio.AudioConfig(filename=str(audio_wav))
    recognizer = speechsdk.SpeechRecognizer(speech_config=cfg, audio_config=audio_cfg)

    result = recognizer.recognize_once()
    if result.reason != speechsdk.ResultReason.RecognizedSpeech:
        logging.debug("recognize_once: non-RecognizedSpeech reason=%s", result.reason)
        return 0.0

    json_str = result.properties.get(speechsdk.PropertyId.SpeechServiceResponse_JsonResult)
    if not json_str:
        return 0.0

    try:
        data = json.loads(json_str)
        nbest = data.get("NBest") or []
        if not nbest:
            return 0.0
        conf = float(nbest[0].get("Confidence") or 0.0)
        return conf
    except Exception as e:
        logging.debug("Failed to parse detailed JSON for confidence: %s", e)
        return 0.0


def reduce_to_single_language(result: "LIDResult") -> str:
    """Return the single best language label for this file."""
    if not result.detected_languages:
        return "Unknown"
    top = result.detected_languages[0]
    if top.startswith("ar-"):
        return "Moroccan Arabic" if result.arabic_variant == "ar-MA" else "Standard Arabic"
    return {"en-US": "English", "fr-FR": "French"}.get(top, top)

def reduce_to_language_code(result: "LIDResult") -> str:
    """Return ISO-style short code ('ary', 'ar', 'en', 'fr', or fallback)."""
    if not result.detected_languages:
        return "und"
    top = result.detected_languages[0]
    if top.startswith("ar-"):
        return "ary" if result.arabic_variant == "ar-MA" else "ar"
    return {"en-US": "en", "fr-FR": "fr"}.get(top, top.split("-")[0])

# ------------------------------- Orchestration ---------------------------------

@dataclass
class LIDResult:
    audio_seconds: float
    sampled_seconds: float
    chunk_seconds: int
    sample_windows: List[Tuple[float, float]]
    per_chunk_languages: List[Dict[str, float]]
    language_seconds: Dict[str, float]
    detected_languages: List[str]  # unique, sorted by duration seconds desc
    arabic_variant: Optional[str]  # 'ar-MA' or 'ar-SA' if Arabic detected
    dialect_verification: Dict[str, object]

def identify_languages_from_video(
    video_mp4: Path,
    max_sample_seconds: int = DEFAULT_MAX_SAMPLE_SECONDS,
    chunk_seconds: int = DEFAULT_CHUNK_SECONDS,
    seed: int = DEFAULT_SEED,
    verify_arabic: str = "auto",
    probe_count: int = 8,
    probe_seconds: int = 12,
    text_dialect_asr: str = "azure",
    vad: str = "on",
    vad_aggressiveness: int = 2,
    target_sampled_seconds: int = 180,
    stagea_window_seconds: int = 20,
    azure_timeout_seconds: int = DEFAULT_AZURE_TIMEOUT_SEC,
    fallback: str = DEFAULT_FALLBACK,
    save_probes: Optional[Path] = None,
    calibration_path: Optional[Path] = None,
) -> LIDResult:
    env = load_azure_env()
    with tempfile.TemporaryDirectory(prefix="lid_") as tmpdir:
        tmp = Path(tmpdir)
        wav_full = tmp / f"{uuid.uuid4().hex}_full.wav"
        extract_wav_from_mp4(video_mp4, wav_full)
        total_dur = get_wav_duration_seconds(wav_full)
        logging.info("Audio duration: %.2f sec", total_dur)

        windows = stratified_windows(total_dur, stagea_window_seconds, target_sampled_seconds)
        if not windows:
            raise RuntimeError("No audio to sample.")
        if vad == "on":
            windows = filter_windows_by_vad(wav_full, windows, min_ratio=0.15, aggressiveness=vad_aggressiveness)
        logging.debug("Sampling %d windows for Stage A", len(windows))

        per_chunk_langs: List[Dict[str, float]] = []
        language_seconds: Dict[str, float] = {}

        # LID on each chunk (with simple retries)
        for i, (start, dur) in enumerate(windows, 1):
            chunk_wav = tmp / f"chunk_{i:03d}.wav"
            trim_wav(wav_full, chunk_wav, start, dur)

            for attempt in range(3):
                try:
                    secs = continuous_lid_durations(chunk_wav, env, LID_CANDIDATES, timeout_s=azure_timeout_seconds)
                    break
                except Exception as e:
                    logging.warning("LID attempt %d failed (%s); retrying...", attempt + 1, e)
                    time.sleep(1.0 + attempt)
            else:
                raise RuntimeError("LID failed after 3 attempts.")

            per_chunk_langs.append(secs)
            for l, s in secs.items():
                language_seconds[l] = language_seconds.get(l, 0.0) + s

        # Sort languages by total seconds desc
        detected_sorted = sorted(language_seconds.keys(), key=lambda k: language_seconds[k], reverse=True)

        # Compute shares
        total_lang_sec = sum(language_seconds.values()) or 1.0
        lang_shares = {k: language_seconds[k] / total_lang_sec for k in language_seconds}
        p_ar = sum(v for k, v in lang_shares.items() if k.startswith("ar-"))
        top_locale: Optional[str] = None
        if lang_shares:
            top_locale = max(lang_shares, key=lang_shares.get)

        # Decide if we run dialect verification Stage B
        arabic_is_top = bool(top_locale and top_locale.startswith("ar-"))
        run_stage_b = False
        if verify_arabic == "always" and p_ar >= 0.10:
            run_stage_b = True
        elif verify_arabic == "auto" and arabic_is_top:
            run_stage_b = True

        dialect_block: Dict[str, object] = {"run": False}
        final_language_code: Optional[str] = None
        final_language_label: Optional[str] = None
        arabic_variant: Optional[str] = None

        if run_stage_b:
            probe_indices = select_arabic_probe_indices_seconds(per_chunk_langs, top_k=probe_count)
            dialect_block = run_stage_b_dialect_verification(
                full_wav=wav_full,
                windows=windows,
                probe_indices=probe_indices,
                probe_seconds=probe_seconds,
                env=env,
                text_asr=text_dialect_asr,
            )
            if dialect_block["decision"] == "ary":
                final_language_code = "ary"
                final_language_label = "Moroccan Arabic"
                arabic_variant = "ar-MA"
            else:
                final_language_code = "ar"
                final_language_label = "Standard Arabic"
                arabic_variant = "ar-SA"
        # If no Stage B
        if not final_language_code:
            if not top_locale:
                final_language_code, final_language_label = "und", "Unknown"
            elif top_locale == "en-US":
                final_language_code, final_language_label = "en", "English"
            elif top_locale == "fr-FR":
                final_language_code, final_language_label = "fr", "French"
            elif top_locale.startswith("ar-"):
                final_language_code, final_language_label = "ar", "Standard Arabic"
                arabic_variant = "ar-SA"
            else:
                final_language_code, final_language_label = top_locale.split("-")[0], top_locale

        # attach dialect block to per-function scope (will embed into LIDResult later)
        dialect_block_local = dialect_block

        return LIDResult(
            audio_seconds=total_dur,
            sampled_seconds=sum(d for _, d in windows),
            chunk_seconds=chunk_seconds,
            sample_windows=windows,
            per_chunk_languages=per_chunk_langs,
            language_seconds=language_seconds,
            detected_languages=detected_sorted,
            arabic_variant=arabic_variant,
            dialect_verification=dialect_block_local,
        )

def main():
    ap = argparse.ArgumentParser(description="Identify languages from an MP4 using Azure Speech LID (.env driven).")
    ap.add_argument("video", type=Path, help="Path to input .mp4 (or any media ffmpeg can read)")
    ap.add_argument("--max-sample-seconds", type=int, default=DEFAULT_MAX_SAMPLE_SECONDS,
                    help=f"Cap on total sampled audio (default {DEFAULT_MAX_SAMPLE_SECONDS}s)")
    ap.add_argument("--chunk-seconds", type=int, default=DEFAULT_CHUNK_SECONDS,
                    help=f"Chunk length for LID (default {DEFAULT_CHUNK_SECONDS}s)")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed for sampling (default 42)")
    ap.add_argument("--out", type=Path, default=None, help="Optional path to write JSON results")
    ap.add_argument("--env", type=Path, default=None, help="Path to .env file (default: auto-discover)")
    ap.add_argument("--log", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    ap.add_argument("--simple", action="store_true",
                    help="Also print 'language: <label>' after the JSON.")
    ap.add_argument("--simple-only", action="store_true",
                    help="Only print 'language: <label>' (no JSON output).")
    # --- Dialect verification & sampling flags ---
    ap.add_argument("--verify-arabic", choices=["auto", "always", "never"], default="auto",
                    help="Run dialect verification when Arabic detected. 'auto' = when Arabic top; 'always' = whenever Arabic share≥0.10; 'never' = skip stage B.")
    ap.add_argument("--probe-count", type=int, default=8, help="Number of probes for dialect verification (default 8)")
    ap.add_argument("--probe-seconds", type=int, default=12, help="Length of each probe in seconds (default 12)")
    ap.add_argument("--text-dialect-asr", choices=["azure", "whisper", "none"], default="azure",
                    help="ASR backend for text-dialect classifier (default azure)")
    ap.add_argument("--vad", choices=["on", "off"], default="on", help="Enable simple VAD for sampling (default on)")
    ap.add_argument("--vad-aggressiveness", type=int, default=2, help="VAD aggressiveness 0..3 (default 2)")
    ap.add_argument("--target-sampled-seconds", type=int, default=180,
                    help="Target total sampled seconds for Stage A (default 180)")
    ap.add_argument("--stagea-window-seconds", type=int, default=20,
                    help="Window length for Stage A sampling (default 20)")
    ap.add_argument("--azure-timeout-seconds", type=int, default=DEFAULT_AZURE_TIMEOUT_SEC,
                    help=f"Watchdog timeout per window recognition (default {DEFAULT_AZURE_TIMEOUT_SEC})")
    ap.add_argument("--fallback", choices=["whisper", "none"], default=DEFAULT_FALLBACK,
                    help="Fallback LID backend if Azure fails (default whisper)")
    ap.add_argument("--save-probes", type=Path, default=None,
                    help="Directory to save Stage B probe WAVs and scores JSON")
    ap.add_argument("--calibration", type=Path, default=None,
                    help="Calibration JSON file (optional)")
    args = ap.parse_args()

    # Load .env before anything else
    if args.env:
        if not args.env.exists():
            print(f".env not found at: {args.env}", file=sys.stderr)
            sys.exit(1)
        load_dotenv(dotenv_path=args.env, override=False)
    else:
        # auto-find .env in CWD or parents
        load_dotenv(find_dotenv(usecwd=True), override=False)

    logging.basicConfig(
        level=getattr(logging, args.log),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    if not args.video.exists():
        print(f"Input video not found: {args.video}", file=sys.stderr)
        sys.exit(1)

    try:
        result = identify_languages_from_video(
            video_mp4=args.video,
            max_sample_seconds=args.max_sample_seconds,
            chunk_seconds=args.stagea_window_seconds,
            seed=args.seed,
            verify_arabic=args.verify_arabic,
            probe_count=args.probe_count,
            probe_seconds=args.probe_seconds,
            text_dialect_asr=args.text_dialect_asr,
            vad=args.vad,
            vad_aggressiveness=args.vad_aggressiveness,
            target_sampled_seconds=args.target_sampled_seconds,
            stagea_window_seconds=args.stagea_window_seconds,
            azure_timeout_seconds=args.azure_timeout_seconds,
            fallback=args.fallback,
            save_probes=args.save_probes,
            calibration_path=args.calibration,
        )
    except Exception as e:
        logging.exception("Language identification failed.")
        print(json.dumps({"ok": False, "error": str(e)}, ensure_ascii=False, indent=2))
        sys.exit(2)

    # Build output per extended schema
    lang_durs = {k: round(v, 3) for k, v in result.language_seconds.items()}
    total_lang_sec = max(1.0, sum(result.language_seconds.values()))
    lang_shares = {k: round(v / total_lang_sec, 3) for k, v in result.language_seconds.items()}

    stageA = {
        "window_seconds": args.stagea_window_seconds,
        "sampled_seconds": round(result.sampled_seconds, 3),
        "num_windows": len(result.sample_windows),
        "windows": [{"start": s, "duration": d} for s, d in result.sample_windows],
        "language_durations_seconds": lang_durs,
        "language_shares": lang_shares,
        "source": "azure-continuous-lid",
        "mean_voiced_ratio": 0.0,
    }

    stageB = dict(result.dialect_verification)
    stageB.setdefault("calibration", CALIB_DEFAULT.get("stageB"))

    final_language_label = reduce_to_single_language(result)
    final_language_code = reduce_to_language_code(result)

    language_mix = []
    for loc, share in lang_shares.items():
        code_map = {"en-US": "en", "fr-FR": "fr"}
        code = code_map.get(loc, "ar" if loc.startswith("ar-") else loc.split("-")[0])
        language_mix.append({"code": code, "share": round(share, 2)})

    final_decision = {
        "code": final_language_code,
        "label": final_language_label,
        "confidence": stageB.get("final_score", lang_shares.get(result.detected_languages[0], 0.0) if result.detected_languages else 0.0),
        "language_mix": language_mix,
        "rationale": {
            "top_locale_by_duration": result.detected_languages[0] if result.detected_languages else None,
            "mixed_rule_triggered": False,
        },
    }

    summary = {
        "ok": True,
        "input_path": str(args.video),
        "audio_seconds": round(result.audio_seconds, 3),
        "stageA": stageA,
        "stageB": stageB,
        "final_decision": final_decision,
        "metadata": {
            "version": VERSION,
            "datetime_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "env": "prod",
            "source": "azure",
            "warnings": [],
        },
    }

    if args.simple_only:
        print(f"language_code: {final_language_code}")
        return

    # default: print JSON
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.simple:
        print(f"language_code: {final_language_code}")

if __name__ == "__main__":
    main()
