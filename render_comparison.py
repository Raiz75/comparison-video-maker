import json
import os
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import AudioFileClip, ImageSequenceClip, VideoFileClip, concatenate_audioclips, CompositeAudioClip
from kokoro_onnx import Kokoro
from faster_whisper import WhisperModel
import soundfile as sf

# ── Config ───────────────────────────────────────────────────────────────────
DEFAULT_CONFIG = {
    "video": {"width": 1080, "height": 1920, "fps": 24},
    "layout": {
        "sec1_height": 540, "sec2_height": 120, "sec3_height": 180, "sec4_height": 1080,
        "image_margin": 10, "text_padding": 40,
    },
    "fonts": {"item_name_size": 48, "script_size": 64},
}

def _load_config():
    config_path = Path(__file__).parent / "config.json"
    try:
        with open(config_path, encoding="utf-8") as f:
            user_config = json.load(f)
    except Exception:
        user_config = {}
    config = {}
    for section, defaults in DEFAULT_CONFIG.items():
        config[section] = {**defaults, **(user_config.get(section) or {})}
    return config

CONFIG = _load_config()

VIDEO_W     = CONFIG["video"]["width"]
VIDEO_H     = CONFIG["video"]["height"]
FPS         = CONFIG["video"]["fps"]

S1_H = CONFIG["layout"]["sec1_height"]
S2_H = CONFIG["layout"]["sec2_height"]
S3_H = CONFIG["layout"]["sec3_height"]
S4_H = CONFIG["layout"]["sec4_height"]
IMG_MARGIN  = CONFIG["layout"]["image_margin"]
TEXT_PAD    = CONFIG["layout"]["text_padding"]

ITEM_FONT_SIZE   = CONFIG["fonts"]["item_name_size"]
SCRIPT_FONT_SIZE = CONFIG["fonts"]["script_size"]

TTS_TEMP_DIR = Path(__file__).parent / "temp_tts"

FONT_PATHS_BOLD = [
    "C:/Windows/Fonts/arialbd.ttf",
    "C:/Windows/Fonts/Arial Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]
FONT_PATHS_REGULAR = [
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]

def _get_font(size, bold=True):
    paths = FONT_PATHS_BOLD if bold else FONT_PATHS_REGULAR
    for p in paths:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()

# ── Text wrapping ────────────────────────────────────────────────────────────
def _wrap_text(text, font, max_width, draw):
    words = text.split()
    lines, current = [], ""
    for word in words:
        test = f"{current} {word}".strip()
        w = draw.textbbox((0, 0), test, font=font)[2]
        if w <= max_width:
            current = test
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines or [text]

# ── Frame composition ────────────────────────────────────────────────────────
def _compose_frame(img_a, img_b, name_a, name_b, script_text, visible_words, char_frame_np):
    frame = Image.new("RGB", (VIDEO_W, VIDEO_H), (255, 255, 255))
    draw = ImageDraw.Draw(frame)

    img_w = (VIDEO_W - IMG_MARGIN * 3) // 2
    img_h = S1_H - IMG_MARGIN * 2

    img_a_rs = img_a.resize((img_w, img_h), Image.LANCZOS)
    img_b_rs = img_b.resize((img_w, img_h), Image.LANCZOS)

    frame.paste(img_a_rs, (IMG_MARGIN, IMG_MARGIN))
    frame.paste(img_b_rs, (IMG_MARGIN * 2 + img_w, IMG_MARGIN))

    # sec2 — item names
    font_item = _get_font(ITEM_FONT_SIZE, bold=True)
    name_y = S1_H + (S2_H - ITEM_FONT_SIZE) // 2
    lcx = IMG_MARGIN + img_w // 2
    rcx = IMG_MARGIN * 2 + img_w + img_w // 2

    for name, cx in [(name_a, lcx), (name_b, rcx)]:
        b = draw.textbbox((0, 0), name, font=font_item)
        draw.text((cx - (b[2]-b[0])//2, name_y), name, font=font_item, fill=(40, 40, 40))

    # sec3 — script text typewriter
    font_script = _get_font(SCRIPT_FONT_SIZE, bold=False)
    visible_text = " ".join(script_text.split()[:visible_words])
    if visible_text:
        max_w = VIDEO_W - TEXT_PAD * 2
        dummy = Image.new("RGB", (1, 1))
        wrapped = _wrap_text(visible_text, font_script, max_w, ImageDraw.Draw(dummy))
        lh = int(SCRIPT_FONT_SIZE * 1.4)
        total_h = len(wrapped) * lh
        sy = S1_H + S2_H + (S3_H - total_h) // 2
        for line in wrapped:
            b = draw.textbbox((0, 0), line, font=font_script)
            x = (VIDEO_W - (b[2]-b[0])) // 2
            draw.text((x, sy), line, font=font_script, fill=(40, 40, 40))
            sy += lh

    # sec4 — character video (fit full body, no crop)
    char_img = Image.fromarray(char_frame_np).convert("RGB")
    cw, ch = char_img.size
    scale = min(VIDEO_W / cw, S4_H / ch)
    new_w = int(cw * scale)
    new_h = int(ch * scale)
    char_img = char_img.resize((new_w, new_h), Image.LANCZOS)
    cx = (VIDEO_W - new_w) // 2
    cy = S1_H + S2_H + S3_H + (S4_H - new_h) // 2
    frame.paste(char_img, (cx, cy))

    return np.array(frame)

# ──── Models (lazy) ──────────────────────────────────────────────────────────
_whisper = None
_kokoro  = None

def _ensure_models():
    global _whisper, _kokoro
    if _whisper is None:
        _whisper = WhisperModel("tiny", device="cpu", compute_type="int8")
    if _kokoro is None:
        base = Path(__file__).parent / "assets" / "tts"
        _kokoro = Kokoro(str(base / "kokoro-v1.0.onnx"), str(base / "voices-v1.0.bin"))

# ── TTS pipeline ─────────────────────────────────────────────────────────────
_tts_counter = 0

def _generate_tts(text, voice="am_santa", speed=1):
    global _tts_counter
    TTS_TEMP_DIR.mkdir(exist_ok=True)
    _tts_counter += 1
    path = TTS_TEMP_DIR / f"line_{_tts_counter:02d}.wav"
    samples, rate = _kokoro.create(text, voice=voice, speed=speed, lang="en-us")
    sf.write(str(path), samples, rate)
    return str(path)

def _get_word_timings(audio_path):
    segments, _ = _whisper.transcribe(audio_path, word_timestamps=True, language="en")
    words = []
    for seg in segments:
        for w in seg.words:
            words.append((w.word.strip(), w.start, w.end))
    return words

# ── Character frame preload ──────────────────────────────────────────────────
def _load_char_frames(path):
    clip = VideoFileClip(str(path))
    frames = list(clip.iter_frames())
    clip.close()
    return frames, clip.fps

# ── Main render ──────────────────────────────────────────────────────────────
def _cleanup_temp():
    try:
        import shutil
        shutil.rmtree(str(TTS_TEMP_DIR), ignore_errors=True)
    except:
        pass

def render_comparison_video(script_data, image_paths, char_dir, bg_music_path, output_path, log_fn=print, cancel_event=None):
    _ensure_models()

    if not isinstance(script_data, list) or len(script_data) < 1:
        raise ValueError("Script data must be a non-empty list of pairs")

    if cancel_event and cancel_event.is_set():
        log_fn("Cancelled.")
        _cleanup_temp()
        return

    video_clip = None
    audio_segments = []
    open_clips = []
    bg = None
    pair_images = []
    char_data = {}
    line_info = []
    all_frames = []
    final_audio = None

    try:
        # Preload item images
        log_fn("Loading item images...")
        for i in range(3):
            if cancel_event and cancel_event.is_set():
                log_fn("Cancelled.")
                return
            a = Image.open(image_paths[i * 2]).convert("RGB")
            b = Image.open(image_paths[i * 2 + 1]).convert("RGB")
            pair_images.append((a, b))

        # Preload character frames
        log_fn("Loading character videos...")
        for pose in ("left", "right", "ask"):
            if cancel_event and cancel_event.is_set():
                log_fn("Cancelled.")
                return
            path = char_dir / f"{pose}-pose.mp4"
            if not path.exists():
                raise FileNotFoundError(f"Character video not found: {path}")
            frames, fps = _load_char_frames(path)
            char_data[pose] = {"frames": frames, "fps": fps}

        # Generate TTS for each line + get word timings
        log_fn("Generating TTS audio for all lines...")
        for pair_idx, pair in enumerate(script_data):
            if cancel_event and cancel_event.is_set():
                log_fn("Cancelled.")
                return
            for line in pair["lines"]:
                if cancel_event and cancel_event.is_set():
                    log_fn("Cancelled.")
                    return
                log_fn(f"  TTS: [{pair_idx+1}/{len(script_data)}] \"{line['script'][:50]}{'...' if len(line['script'])>50 else ''}\"")
                audio_path = _generate_tts(line["script"])
                words = _get_word_timings(audio_path)
                audio_clip = AudioFileClip(audio_path)
                duration = audio_clip.duration
                audio_clip.close()

                line_info.append({
                    "pair_idx": pair_idx,
                    "pose": line["pose"],
                    "script": line["script"],
                    "audio_path": audio_path,
                    "duration": duration,
                    "words": words,
                    "item_a": pair["item_a"],
                    "item_b": pair["item_b"],
                })

        total_duration = sum(li["duration"] for li in line_info)
        total_frames = int(total_duration * FPS)

        log_fn(f"Total duration: {total_duration:.1f}s | Total frames: {total_frames}")

        # Generate frames
        log_fn("Compositing frames...")
        for li in line_info:
            if cancel_event and cancel_event.is_set():
                log_fn("Cancelled.")
                return
            line_frames = int(li["duration"] * FPS)
            pair_idx = li["pair_idx"]
            img_a, img_b = pair_images[pair_idx]

            char_frames = char_data[li["pose"]]["frames"]
            char_nframes = len(char_frames)

            for f in range(line_frames):
                if cancel_event and cancel_event.is_set():
                    log_fn("Cancelled.")
                    return
                elapsed_s = f / FPS
                visible = 0
                for w, start, end in li["words"]:
                    if elapsed_s >= start:
                        visible += 1
                progress = f / max(line_frames, 1)
                cf_idx = int(progress * char_nframes) % char_nframes
                cf_np = char_frames[cf_idx]

                frame_np = _compose_frame(
                    img_a, img_b,
                    li["item_a"], li["item_b"],
                    li["script"], visible, cf_np
                )
                all_frames.append(frame_np)

        # Build video
        log_fn("Writing video...")
        video_clip = ImageSequenceClip(all_frames, fps=FPS)

        # Mix TTS + character audio per line, then concatenate
        log_fn("Assembling audio...")
        for li in line_info:
            if cancel_event and cancel_event.is_set():
                log_fn("Cancelled.")
                return
            tts = AudioFileClip(li["audio_path"])
            open_clips.append(tts)
            ca_path = char_dir / f"{li['pose']}-pose.mp4"
            ca = AudioFileClip(str(ca_path))
            open_clips.append(ca)
            repeats = int(li["duration"] // ca.duration) + 1
            ca_loop = concatenate_audioclips([ca] * repeats).subclip(0, li["duration"])
            mixed = CompositeAudioClip([tts, ca_loop])
            mixed.fps = getattr(tts, 'fps', 44100) or 44100
            audio_segments.append(mixed)

        final_audio = concatenate_audioclips(audio_segments)

        # Background music
        if bg_music_path and os.path.exists(bg_music_path):
            log_fn("Adding background music...")
            bg = AudioFileClip(bg_music_path)
            if bg.duration < total_duration:
                loops = int(total_duration // bg.duration) + 1
                bg = concatenate_audioclips([bg] * loops)
            bg = bg.subclip(0, total_duration).volumex(0.15)
            final_audio = CompositeAudioClip([final_audio, bg])
            final_audio.fps = getattr(bg, 'fps', 44100) or 44100

        video_clip = video_clip.set_audio(final_audio)

        video_clip.write_videofile(
            str(output_path),
            fps=FPS,
            codec="libx264",
            audio_codec="aac",
            temp_audiofile=str(Path(output_path).parent / "_tmp_audio.m4a"),
            remove_temp=True,
            logger=None,
        )

        log_fn(f"Video saved: {Path(output_path).name}")

    finally:
        _cleanup_temp()
        for seg in audio_segments:
            try: seg.close()
            except: pass
        for c in open_clips:
            try: c.close()
            except: pass
        if bg:
            try: bg.close()
            except: pass
        if video_clip:
            try: video_clip.close()
            except: pass
