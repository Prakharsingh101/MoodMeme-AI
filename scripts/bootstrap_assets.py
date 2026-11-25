"""
Bootstrap Assets Script - Heavy Audio & Multi-Face Vibe Edition
Auto-downloads MediaPipe face_landmarker.task model,
synthesizes 19 transparent RGBA meme stickers & 3D emojis,
synthesizes 22 16-bit PCM WAV SFX files (including 4 new Heavy Bass/Cinematic cues),
and generates assets/manifest.json with "What We Feel / Facial Vibe" insights for all 9 emotions.
"""

import os
import sys
import math
import wave
import struct
import json
import zlib
import urllib.request
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
ASSETS_DIR = ROOT_DIR / "assets"
MODELS_DIR = ASSETS_DIR / "models"
MEMES_DIR = ASSETS_DIR / "memes"
SOUNDS_DIR = ASSETS_DIR / "sounds"
EXPORTS_DIR = ROOT_DIR / "exports"

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
MODEL_PATH = MODELS_DIR / "face_landmarker.task"


def ensure_directories():
    for d in [MODELS_DIR, MEMES_DIR, SOUNDS_DIR, EXPORTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    print("[bootstrap] Directories verified.")


def download_model():
    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 1000000:
        print(f"[bootstrap] Model already exists at {MODEL_PATH} ({MODEL_PATH.stat().st_size} bytes).")
        return

    print(f"[bootstrap] Attempting to download Face Landmarker model from {MODEL_URL}...")
    try:
        def reporthook(blocknum, blocksize, totalsize):
            readsofar = blocknum * blocksize
            if totalsize > 0:
                percent = readsofar * 100 / totalsize
                s = f"\r[bootstrap] Download progress: {percent:5.1f}% ({readsofar}/{totalsize} bytes)"
                sys.stdout.write(s)
                sys.stdout.flush()

        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH, reporthook)
        print(f"\n[bootstrap] Model downloaded successfully to {MODEL_PATH}")
    except Exception as e:
        print(f"\n[bootstrap] Network access unavailable or download skipped: {e}")
        if not MODEL_PATH.exists():
            MODEL_PATH.write_bytes(b"MOCK_FACE_LANDMARKER_MODEL_DATA")
            print(f"[bootstrap] Created placeholder at {MODEL_PATH}")


def create_stickers_and_emojis():
    print("[bootstrap] Generating 19 RGBA meme stickers & 3D emojis...")
    try:
        from PIL import Image, ImageDraw, ImageFont
        use_pil = True
    except ImportError:
        use_pil = False

    size = (256, 256)

    def draw_heart(draw, cx, cy, s, color):
        points = []
        for t in range(0, 360, 10):
            rad = math.radians(t)
            x = 16 * (math.sin(rad) ** 3)
            y = -(13 * math.cos(rad) - 5 * math.cos(2*rad) - 2 * math.cos(3*rad) - math.cos(4*rad))
            points.append((cx + x * s, cy + y * s))
        draw.polygon(points, fill=color, outline=(255, 255, 255, 230))

    if use_pil:
        # Base 9 Stickers
        # PEACE
        img = Image.new("RGBA", size, (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        d.ellipse([38, 38, 218, 218], outline=(255, 223, 128, 180), width=6)
        d.ellipse([58, 58, 198, 198], outline=(180, 240, 255, 140), width=3)
        d.polygon([(128, 70), (105, 150), (128, 180), (151, 150)], fill=(255, 230, 255, 240), outline=(220, 160, 230, 255))
        d.polygon([(128, 175), (85, 140), (60, 120), (95, 165)], fill=(245, 190, 240, 230), outline=(200, 130, 210, 255))
        d.polygon([(128, 175), (171, 140), (196, 120), (161, 165)], fill=(245, 190, 240, 230), outline=(200, 130, 210, 255))
        d.ellipse([116, 155, 140, 179], fill=(255, 245, 180, 255), outline=(240, 200, 70, 255))
        img.save(MEMES_DIR / "peace_lotus.png")

        # LOVE
        img = Image.new("RGBA", size, (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        draw_heart(d, 128, 90, 4.2, (255, 45, 95, 240))
        draw_heart(d, 60, 140, 2.6, (255, 90, 150, 220))
        draw_heart(d, 196, 130, 2.8, (255, 60, 120, 230))
        draw_heart(d, 40, 70, 1.8, (255, 120, 180, 200))
        draw_heart(d, 220, 60, 2.0, (255, 100, 170, 210))
        img.save(MEMES_DIR / "love_hearts.png")

        # LAUGHTER
        img = Image.new("RGBA", size, (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        for offset_x in [65, 191]:
            d.ellipse([offset_x - 22, 100, offset_x + 22, 160], fill=(80, 190, 255, 230), outline=(255, 255, 255, 240), width=3)
            d.polygon([(offset_x, 70), (offset_x - 20, 120), (offset_x + 20, 120)], fill=(80, 190, 255, 230))
            d.ellipse([offset_x - 30, 170, offset_x - 10, 190], fill=(120, 210, 255, 200))
            d.ellipse([offset_x + 10, 175, offset_x + 26, 191], fill=(120, 210, 255, 200))
        d.arc([40, 20, 216, 100], start=180, end=360, fill=(255, 215, 0, 255), width=8)
        d.polygon([(128, 15), (145, 45), (111, 45)], fill=(255, 215, 0, 255))
        img.save(MEMES_DIR / "laughter_tears.png")

        # SORROW
        img = Image.new("RGBA", size, (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        cloud_color = (60, 70, 90, 240)
        cloud_outline = (130, 150, 180, 255)
        d.ellipse([40, 60, 120, 130], fill=cloud_color, outline=cloud_outline, width=3)
        d.ellipse([90, 40, 180, 120], fill=cloud_color, outline=cloud_outline, width=3)
        d.ellipse([140, 65, 215, 135], fill=cloud_color, outline=cloud_outline, width=3)
        d.rectangle([70, 90, 185, 135], fill=cloud_color)
        raindrops = [(70, 160), (105, 180), (135, 155), (165, 185), (195, 160), (90, 215), (150, 220)]
        for rx, ry in raindrops:
            d.line([(rx, ry), (rx - 8, ry + 22)], fill=(90, 180, 255, 230), width=4)
        img.save(MEMES_DIR / "sorrow_raincloud.png")

        # ANGER
        img = Image.new("RGBA", size, (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        steam_color = (255, 50, 30, 235)
        d.ellipse([30, 60, 90, 120], fill=steam_color, outline=(255, 200, 200, 220), width=3)
        d.ellipse([50, 40, 110, 90], fill=steam_color, outline=(255, 200, 200, 220), width=3)
        d.ellipse([166, 60, 226, 120], fill=steam_color, outline=(255, 200, 200, 220), width=3)
        d.ellipse([146, 40, 206, 90], fill=steam_color, outline=(255, 200, 200, 220), width=3)
        cx, cy = 128, 80
        d.arc([cx - 25, cy - 25, cx + 5, cy + 5], start=0, end=90, fill=(255, 20, 20, 255), width=6)
        d.arc([cx - 5, cy - 25, cx + 25, cy + 5], start=90, end=180, fill=(255, 20, 20, 255), width=6)
        d.arc([cx - 25, cy - 5, cx + 5, cy + 25], start=270, end=360, fill=(255, 20, 20, 255), width=6)
        d.arc([cx - 5, cy - 5, cx + 25, cy + 25], start=180, end=270, fill=(255, 20, 20, 255), width=6)
        img.save(MEMES_DIR / "anger_steam.png")

        # COURAGE
        img = Image.new("RGBA", size, (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        flame_outer = (255, 180, 0, 220)
        flame_inner = (255, 245, 100, 250)
        spikes = [
            [(128, 20), (105, 100), (151, 100)],
            [(80, 45), (65, 120), (105, 105)],
            [(176, 45), (151, 105), (191, 120)],
            [(45, 90), (40, 150), (80, 130)],
            [(211, 90), (176, 130), (216, 150)],
        ]
        for s in spikes:
            d.polygon(s, fill=flame_outer, outline=(255, 255, 220, 240))
        d.polygon([(128, 45), (115, 95), (141, 95)], fill=flame_inner)
        d.polygon([(90, 65), (80, 110), (105, 100)], fill=flame_inner)
        d.polygon([(166, 65), (151, 100), (176, 110)], fill=flame_inner)
        d.rectangle([35, 125, 221, 145], fill=(220, 30, 30, 240), outline=(255, 215, 0, 255), width=3)
        d.ellipse([118, 120, 138, 150], fill=(255, 215, 0, 255))
        img.save(MEMES_DIR / "courage_aura.png")

        # FEAR
        img = Image.new("RGBA", size, (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        ghost_color = (230, 245, 255, 220)
        d.ellipse([80, 50, 176, 150], fill=ghost_color, outline=(150, 190, 230, 255), width=3)
        d.polygon([(80, 100), (176, 100), (176, 170), (152, 155), (128, 170), (104, 155), (80, 170)], fill=ghost_color)
        d.ellipse([100, 80, 118, 105], fill=(20, 40, 80, 255))
        d.ellipse([138, 80, 156, 105], fill=(20, 40, 80, 255))
        d.ellipse([120, 115, 136, 142], fill=(20, 40, 80, 255))
        d.polygon([(45, 100), (35, 125), (55, 125)], fill=(60, 170, 255, 230))
        d.ellipse([35, 115, 55, 135], fill=(60, 170, 255, 230))
        d.polygon([(215, 95), (205, 120), (225, 120)], fill=(60, 170, 255, 230))
        d.ellipse([205, 110, 225, 130], fill=(60, 170, 255, 230))
        img.save(MEMES_DIR / "fear_ghost.png")

        # DISGUST
        img = Image.new("RGBA", size, (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        puff_col = (110, 210, 40, 210)
        d.ellipse([40, 50, 110, 110], fill=puff_col, outline=(180, 255, 100, 230), width=3)
        d.ellipse([90, 35, 170, 105], fill=puff_col, outline=(180, 255, 100, 230), width=3)
        d.ellipse([150, 55, 215, 115], fill=puff_col, outline=(180, 255, 100, 230), width=3)
        for sx in [75, 128, 180]:
            d.arc([sx - 15, 115, sx + 15, 155], start=0, end=180, fill=(140, 230, 50, 250), width=4)
            d.arc([sx - 15, 145, sx + 15, 185], start=180, end=360, fill=(140, 230, 50, 250), width=4)
        d.ellipse([120, 190, 136, 212], fill=(120, 220, 40, 240))
        img.save(MEMES_DIR / "disgust_nauseous.png")

        # WONDER
        img = Image.new("RGBA", size, (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        def draw_star(draw, cx, cy, r_out, r_in, fill_col):
            pts = []
            for i in range(8):
                ang = i * math.pi / 4
                r = r_out if i % 2 == 0 else r_in
                pts.append((cx + r * math.cos(ang), cy + r * math.sin(ang)))
            draw.polygon(pts, fill=fill_col, outline=(255, 255, 255, 255))
        draw_star(d, 128, 65, 45, 14, (255, 235, 80, 255))
        draw_star(d, 55, 110, 25, 8, (120, 230, 255, 240))
        draw_star(d, 201, 110, 25, 8, (255, 140, 230, 240))
        img.save(MEMES_DIR / "wonder_stars.png")

        # Extra 10 3D Emojis
        # A. emoji_laugh_crying.png
        img = Image.new("RGBA", size, (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        d.ellipse([28, 28, 228, 228], fill=(255, 210, 40, 255), outline=(230, 160, 20, 255), width=6)
        d.chord([60, 90, 196, 210], start=0, end=180, fill=(160, 20, 20, 255), outline=(100, 10, 10, 255), width=4)
        d.chord([75, 120, 181, 195], start=0, end=180, fill=(240, 80, 100, 255))
        d.rectangle([80, 135, 176, 150], fill=(255, 255, 255, 255))
        d.line([(55, 95), (85, 80), (105, 95)], fill=(80, 50, 10, 255), width=7)
        d.line([(151, 95), (171, 80), (201, 95)], fill=(80, 50, 10, 255), width=7)
        d.ellipse([20, 90, 65, 180], fill=(60, 180, 255, 230), outline=(255, 255, 255, 240), width=3)
        d.ellipse([191, 90, 236, 180], fill=(60, 180, 255, 230), outline=(255, 255, 255, 240), width=3)
        img.save(MEMES_DIR / "emoji_laugh_crying.png")

        # B. emoji_heart_eyes.png
        img = Image.new("RGBA", size, (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        d.ellipse([28, 28, 228, 228], fill=(255, 210, 40, 255), outline=(230, 160, 20, 255), width=6)
        d.chord([65, 110, 191, 205], start=0, end=180, fill=(140, 20, 30, 255), outline=(100, 10, 20, 255), width=4)
        d.rectangle([80, 140, 176, 152], fill=(255, 255, 255, 255))
        draw_heart(d, 82, 92, 2.5, (240, 20, 50, 255))
        draw_heart(d, 174, 92, 2.5, (240, 20, 50, 255))
        d.ellipse([36, 140, 70, 165], fill=(255, 120, 150, 160))
        d.ellipse([186, 140, 220, 165], fill=(255, 120, 150, 160))
        img.save(MEMES_DIR / "emoji_heart_eyes.png")

        # C. emoji_fire_sunglasses.png
        img = Image.new("RGBA", size, (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        d.rectangle([30, 95, 115, 145], fill=(15, 15, 20, 255), outline=(255, 255, 255, 255), width=3)
        d.rectangle([141, 95, 226, 145], fill=(15, 15, 20, 255), outline=(255, 255, 255, 255), width=3)
        d.rectangle([115, 105, 141, 120], fill=(15, 15, 20, 255))
        d.rectangle([45, 102, 60, 115], fill=(255, 255, 255, 255))
        d.rectangle([60, 115, 75, 128], fill=(255, 255, 255, 255))
        d.rectangle([156, 102, 171, 115], fill=(255, 255, 255, 255))
        d.rectangle([171, 115, 186, 128], fill=(255, 255, 255, 255))
        flame_pts = [
            [(128, 15), (110, 85), (146, 85)],
            [(75, 35), (60, 95), (95, 90)],
            [(181, 35), (161, 90), (196, 95)],
            [(45, 60), (35, 100), (70, 95)],
            [(211, 60), (186, 95), (221, 100)]
        ]
        for f in flame_pts:
            d.polygon(f, fill=(255, 120, 0, 240), outline=(255, 230, 80, 255))
        d.polygon([(128, 40), (118, 85), (138, 85)], fill=(255, 240, 80, 255))
        img.save(MEMES_DIR / "emoji_fire_sunglasses.png")

        # D. emoji_shocked_mindblown.png
        img = Image.new("RGBA", size, (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        d.chord([28, 70, 228, 230], start=0, end=180, fill=(255, 210, 40, 255), outline=(230, 160, 20, 255), width=5)
        d.ellipse([110, 160, 146, 210], fill=(40, 10, 10, 255))
        d.ellipse([70, 110, 95, 140], fill=(30, 20, 10, 255))
        d.ellipse([161, 110, 186, 140], fill=(30, 20, 10, 255))
        d.ellipse([45, 15, 211, 95], fill=(255, 90, 40, 250), outline=(255, 230, 90, 255), width=4)
        d.ellipse([80, 5, 176, 75], fill=(255, 160, 30, 250))
        d.ellipse([100, 15, 156, 55], fill=(255, 245, 120, 255))
        img.save(MEMES_DIR / "emoji_shocked_mindblown.png")

        # E. emoji_clown.png
        img = Image.new("RGBA", size, (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        d.ellipse([38, 38, 218, 218], fill=(250, 250, 255, 255), outline=(200, 200, 220, 255), width=5)
        d.ellipse([108, 110, 148, 150], fill=(240, 30, 30, 255), outline=(255, 255, 255, 200), width=3)
        d.polygon([(85, 65), (70, 110), (100, 110)], fill=(40, 140, 255, 255))
        d.polygon([(171, 65), (156, 110), (186, 110)], fill=(40, 140, 255, 255))
        d.arc([60, 130, 196, 195], start=0, end=180, fill=(240, 30, 30, 255), width=8)
        d.ellipse([10, 60, 55, 120], fill=(255, 60, 120, 240))
        d.ellipse([201, 60, 246, 120], fill=(60, 200, 255, 240))
        img.save(MEMES_DIR / "emoji_clown.png")

        # F. emoji_skull_dead.png
        img = Image.new("RGBA", size, (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        d.ellipse([45, 30, 211, 180], fill=(245, 245, 250, 255), outline=(180, 180, 195, 255), width=5)
        d.rectangle([85, 150, 171, 220], fill=(245, 245, 250, 255), outline=(180, 180, 195, 255), width=4)
        d.ellipse([75, 80, 115, 130], fill=(25, 25, 35, 255))
        d.ellipse([141, 80, 181, 130], fill=(25, 25, 35, 255))
        d.polygon([(128, 140), (120, 155), (136, 155)], fill=(25, 25, 35, 255))
        for tx in [105, 125, 145]:
            d.line([(tx, 165), (tx, 205)], fill=(120, 120, 140, 255), width=3)
        img.save(MEMES_DIR / "emoji_skull_dead.png")

        # G. emoji_angry_devil.png
        img = Image.new("RGBA", size, (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        d.ellipse([30, 45, 226, 235], fill=(160, 50, 210, 255), outline=(110, 25, 150, 255), width=6)
        d.polygon([(55, 55), (35, 10), (90, 45)], fill=(160, 50, 210, 255), outline=(110, 25, 150, 255))
        d.polygon([(201, 55), (221, 10), (166, 45)], fill=(160, 50, 210, 255), outline=(110, 25, 150, 255))
        d.polygon([(65, 110), (105, 125), (65, 130)], fill=(255, 255, 255, 255))
        d.polygon([(191, 110), (151, 125), (191, 130)], fill=(255, 255, 255, 255))
        d.ellipse([80, 118, 92, 128], fill=(30, 10, 40, 255))
        d.ellipse([164, 118, 176, 128], fill=(30, 10, 40, 255))
        d.arc([80, 160, 176, 205], start=0, end=180, fill=(40, 10, 50, 255), width=6)
        img.save(MEMES_DIR / "emoji_angry_devil.png")

        # H. emoji_gigachad.png
        img = Image.new("RGBA", size, (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        d.polygon([
            (90, 20), (166, 20), (166, 95), (190, 140), (160, 155),
            (155, 200), (145, 235), (105, 235), (95, 180), (85, 100)
        ], fill=(140, 145, 155, 255), outline=(80, 85, 95, 255), width=5)
        d.rectangle([95, 65, 155, 80], fill=(80, 85, 95, 255))
        d.rectangle([105, 95, 150, 145], fill=(160, 165, 175, 255))
        d.line([(100, 180), (145, 180)], fill=(60, 65, 75, 255), width=5)
        draw_star(d, 60, 200, 25, 8, (255, 240, 100, 240))
        img.save(MEMES_DIR / "emoji_gigachad.png")

        # I. emoji_crying_stream.png
        img = Image.new("RGBA", size, (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        d.ellipse([28, 28, 228, 228], fill=(255, 210, 40, 255), outline=(230, 160, 20, 255), width=6)
        d.ellipse([95, 145, 161, 205], fill=(100, 20, 20, 255), outline=(50, 10, 10, 255), width=4)
        d.line([(60, 95), (105, 95)], fill=(60, 40, 10, 255), width=8)
        d.line([(151, 95), (196, 95)], fill=(60, 40, 10, 255), width=8)
        d.rectangle([55, 105, 110, 240], fill=(60, 180, 255, 230))
        d.rectangle([146, 105, 201, 240], fill=(60, 180, 255, 230))
        img.save(MEMES_DIR / "emoji_crying_stream.png")

        # J. emoji_zen_angel.png
        img = Image.new("RGBA", size, (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        d.ellipse([50, 15, 206, 65], outline=(255, 220, 50, 255), width=8)
        d.ellipse([38, 55, 218, 235], fill=(255, 215, 60, 255), outline=(230, 165, 30, 255), width=5)
        d.arc([65, 105, 110, 135], start=0, end=180, fill=(70, 45, 15, 255), width=5)
        d.arc([146, 105, 191, 135], start=0, end=180, fill=(70, 45, 15, 255), width=5)
        d.arc([85, 145, 171, 190], start=0, end=180, fill=(70, 45, 15, 255), width=5)
        img.save(MEMES_DIR / "emoji_zen_angel.png")

        print("[bootstrap] 19 RGBA meme stickers & 3D emojis generated successfully.")


def synthesize_all_sounds():
    print("[bootstrap] Synthesizing 22 16-bit PCM WAV SFX cues (including Heavy Sounds)...")
    sample_rate = 44100

    def write_wav(filename, samples):
        filepath = SOUNDS_DIR / filename
        with wave.open(str(filepath), "w") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            packed = []
            for s in samples:
                val = max(-1.0, min(1.0, s))
                int_val = int(val * 32767.0)
                packed.append(struct.pack("<h", int_val))
            wav_file.writeframes(b"".join(packed))

    # 1. peace_bell.wav (2.0s)
    samples = [
        (0.6 * math.sin(2 * math.pi * 432 * (i/sample_rate)) +
         0.25 * math.sin(2 * math.pi * 864 * (i/sample_rate)) +
         0.15 * math.sin(2 * math.pi * 1296 * (i/sample_rate))) * math.exp(-(i/sample_rate) * 2.2)
        for i in range(int(2.0 * sample_rate))
    ]
    write_wav("peace_bell.wav", samples)

    # 2. love_harp.wav (1.2s)
    dur = 1.2
    n_samples = int(dur * sample_rate)
    samples = [0.0] * n_samples
    for idx, f in enumerate([523.25, 659.25, 783.99, 987.77, 1046.50]):
        si = int(idx * 0.12 * sample_rate)
        for i in range(si, n_samples):
            t = (i - si) / sample_rate
            samples[i] += (math.sin(2 * math.pi * f * t) + 0.3 * math.sin(4 * math.pi * f * t)) * math.exp(-t * 4.5) * 0.25
    write_wav("love_harp.wav", samples)

    # 3. laughter_boing.wav (0.8s)
    samples = [
        math.sin(2 * math.pi * (240.0 + 550.0 * (i/sample_rate/0.8) + 40.0 * math.sin(2 * math.pi * 18 * (i/sample_rate))) * (i/sample_rate)) * math.exp(-(i/sample_rate) * 3.5) * 0.7
        for i in range(int(0.8 * sample_rate))
    ]
    write_wav("laughter_boing.wav", samples)

    # 4. sorrow_violin.wav (1.5s)
    samples = [
        (0.6 * math.sin(2 * math.pi * (293.66 - 16.0 * (i/sample_rate/1.5) + 5.0 * math.sin(2 * math.pi * 5.5 * (i/sample_rate))) * (i/sample_rate)) +
         0.25 * math.sin(4 * math.pi * 293.66 * (i/sample_rate))) * min(1.0, (i/sample_rate) * 4.0) * math.exp(-(i/sample_rate) * 1.5) * 0.7
        for i in range(int(1.5 * sample_rate))
    ]
    write_wav("sorrow_violin.wav", samples)

    # 5. anger_distortion.wav (0.8s)
    samples = [
        math.tanh((math.sin(2 * math.pi * 75 * (i/sample_rate)) + 0.8 * math.sin(2 * math.pi * 110 * (i/sample_rate))) * 3.0) * math.exp(-(i/sample_rate) * 3.0) * 0.7
        for i in range(int(0.8 * sample_rate))
    ]
    write_wav("anger_distortion.wav", samples)

    # 6. courage_fanfare.wav (1.2s)
    n_samples = int(1.2 * sample_rate)
    samples = [0.0] * n_samples
    for st, f in [(0.0, 261.63), (0.18, 329.63), (0.36, 392.00), (0.54, 523.25)]:
        si = int(st * sample_rate)
        for i in range(si, n_samples):
            t = (i - si) / sample_rate
            env = math.exp(-t * 3.0) if i < int((st + 0.35) * sample_rate) else math.exp(-t * 2.0)
            samples[i] += (math.sin(2 * math.pi * f * t) + 0.5 * math.sin(4 * math.pi * f * t)) * env * 0.25
    write_wav("courage_fanfare.wav", samples)

    # 7. fear_sting.wav (0.9s)
    samples = [
        (0.4 * math.sin(2 * math.pi * 1450 * (i/sample_rate)) + 0.4 * math.sin(2 * math.pi * 1520 * (i/sample_rate))) * ((i/sample_rate/0.9)**0.5 * math.exp(-(i/sample_rate) * 2.0) if (i/sample_rate) > 0.4 else ((i/sample_rate)/0.4)*0.8) * 0.6
        for i in range(int(0.9 * sample_rate))
    ]
    write_wav("fear_sting.wav", samples)

    # 8. disgust_gag.wav (0.7s)
    samples = [
        math.sin(2 * math.pi * (320.0 - 200.0 * (i/sample_rate/0.7)) * (i/sample_rate)) * (1.0 + 0.4 * math.sin(2 * math.pi * 28 * (i/sample_rate))) * math.exp(-(i/sample_rate) * 3.0) * 0.65
        for i in range(int(0.7 * sample_rate))
    ]
    write_wav("disgust_gag.wav", samples)

    # 9. wonder_chime.wav (1.4s)
    n_samples = int(1.4 * sample_rate)
    samples = [0.0] * n_samples
    for idx, f in enumerate([880.0, 1046.50, 1318.51, 1567.98, 1760.0, 2093.0]):
        si = int(idx * 0.08 * sample_rate)
        for i in range(si, n_samples):
            t = (i - si) / sample_rate
            samples[i] += math.sin(2 * math.pi * f * t) * math.exp(-t * 5.0) * 0.2
    write_wav("wonder_chime.wav", samples)

    # 10. shutter.wav (0.3s)
    samples = [0.0] * int(0.3 * sample_rate)
    for i in range(len(samples)):
        t = i / sample_rate
        if 0.02 <= t <= 0.06:
            samples[i] += math.sin(2 * math.pi * 1800 * (t - 0.02)) * math.exp(-(t - 0.02) * 120.0) * 0.8
        if 0.12 <= t <= 0.18:
            samples[i] += math.sin(2 * math.pi * 1400 * (t - 0.12)) * math.exp(-(t - 0.12) * 90.0) * 0.8
    write_wav("shutter.wav", samples)

    # 11. vine_boom.wav (0.9s)
    samples = []
    dur = 0.9
    for i in range(int(dur * sample_rate)):
        t = i / sample_rate
        env = math.exp(-t * 3.2)
        freq = 60.0 * math.exp(-t * 1.5)
        sub = math.sin(2 * math.pi * freq * t)
        punch = math.sin(2 * math.pi * 180 * t) * math.exp(-t * 40.0)
        dist = math.tanh((sub * 1.2 + punch * 0.8) * 2.2)
        samples.append(dist * env * 0.85)
    write_wav("vine_boom.wav", samples)

    # 12. bruh.wav (0.5s)
    samples = []
    dur = 0.5
    for i in range(int(dur * sample_rate)):
        t = i / sample_rate
        env = math.sin(math.pi * (t / dur)) ** 0.5 * math.exp(-t * 2.5)
        freq = 165.0 - 75.0 * (t / dur)
        vocal = math.sin(2 * math.pi * freq * t) + 0.6 * math.sin(4 * math.pi * freq * t) + 0.3 * math.sin(6 * math.pi * freq * t)
        samples.append(vocal * env * 0.75)
    write_wav("bruh.wav", samples)

    # 13. wow_anime.wav (1.2s)
    dur = 1.2
    n_samples = int(dur * sample_rate)
    samples = [0.0] * n_samples
    for idx, f in enumerate([659.25, 783.99, 987.77, 1318.51, 1567.98]):
        si = int(idx * 0.09 * sample_rate)
        for i in range(si, n_samples):
            t = (i - si) / sample_rate
            samples[i] += (math.sin(2 * math.pi * f * t) + 0.3 * math.sin(3 * math.pi * f * t)) * math.exp(-t * 4.0) * 0.22
    write_wav("wow_anime.wav", samples)

    # 14. emotional_damage.wav (1.3s)
    samples = []
    dur = 1.3
    for i in range(int(dur * sample_rate)):
        t = i / sample_rate
        env = math.exp(-t * 2.5)
        chord = 0.5 * math.sin(2 * math.pi * 146.8 * t) + 0.4 * math.sin(2 * math.pi * 155.6 * t) + 0.3 * math.sin(2 * math.pi * 293.6 * t)
        dist = math.tanh(chord * 2.0)
        samples.append(dist * env * 0.8)
    write_wav("emotional_damage.wav", samples)

    # 15. thug_life_horn.wav (0.7s)
    samples = []
    dur = 0.7
    for i in range(int(dur * sample_rate)):
        t = i / sample_rate
        horn = 0.0
        for blast_start in [0.0, 0.18, 0.36]:
            if blast_start <= t < blast_start + 0.14:
                dt = t - blast_start
                tone = math.sin(2 * math.pi * 466.16 * dt) + 0.7 * math.sin(2 * math.pi * 587.33 * dt) + 0.5 * math.sin(2 * math.pi * 698.46 * dt)
                horn += math.tanh(tone * 3.0) * 0.6
        samples.append(horn)
    write_wav("thug_life_horn.wav", samples)

    # 16. screaming_goat.wav (0.8s)
    samples = []
    dur = 0.8
    for i in range(int(dur * sample_rate)):
        t = i / sample_rate
        env = math.sin(math.pi * (t / dur)) * math.exp(-t * 1.5)
        flutter = 1.0 + 0.5 * math.sin(2 * math.pi * 24.0 * t)
        s = math.sin(2 * math.pi * 680.0 * t) * flutter
        samples.append(s * env * 0.7)
    write_wav("screaming_goat.wav", samples)

    # 17. record_scratch.wav (0.4s)
    samples = []
    dur = 0.4
    for i in range(int(dur * sample_rate)):
        t = i / sample_rate
        freq = max(80.0, 2200.0 * (1.0 - t / dur) ** 2)
        s = math.sin(2 * math.pi * freq * t) + 0.3 * (math.sin(2 * math.pi * (freq * 1.5) * t))
        samples.append(s * math.exp(-t * 4.0) * 0.8)
    write_wav("record_scratch.wav", samples)

    # 18. victory_fanfare.wav (1.2s)
    n_samples = int(1.2 * sample_rate)
    samples = [0.0] * n_samples
    notes = [(0.0, 523.25), (0.12, 392.0), (0.24, 523.25), (0.36, 659.25), (0.50, 783.99)]
    for st, f in notes:
        si = int(st * sample_rate)
        for i in range(si, n_samples):
            t = (i - si) / sample_rate
            env = math.exp(-t * 3.5) if i < int((st + 0.3) * sample_rate) else math.exp(-t * 2.0)
            samples[i] += (math.sin(2 * math.pi * f * t) + 0.3 * math.sin(3 * math.pi * f * t)) * env * 0.25
    write_wav("victory_fanfare.wav", samples)

    # --- 4 NEW HEAVY BASS & CINEMATIC SOUNDS ---
    # 19. heavy_bass_drop.wav (Massive 808 Sub-bass earthquake drop, 1.4s)
    samples = []
    dur = 1.4
    for i in range(int(dur * sample_rate)):
        t = i / sample_rate
        env = math.exp(-t * 2.2)
        # Deep pitch scoop: 120Hz -> 38Hz with subharmonic 19Hz
        freq = 38.0 + 82.0 * math.exp(-t * 4.0)
        sub = math.sin(2 * math.pi * freq * t) + 0.5 * math.sin(math.pi * freq * t)
        punch = math.sin(2 * math.pi * 140 * t) * math.exp(-t * 35.0)
        dist = math.tanh((sub * 1.5 + punch * 0.9) * 2.8)
        samples.append(dist * env * 0.95)
    write_wav("heavy_bass_drop.wav", samples)

    # 20. heavy_cinematic_thud.wav (Epic Inception brass BRAAAM horn, 1.5s)
    samples = []
    dur = 1.5
    for i in range(int(dur * sample_rate)):
        t = i / sample_rate
        env = min(1.0, t * 15.0) * math.exp(-t * 1.8)
        # 55Hz (A1) + low fifth 82.4Hz + brassy harmonics
        f0 = 55.0
        horn = (math.sin(2 * math.pi * f0 * t) +
                0.8 * math.sin(2 * math.pi * f0 * 1.5 * t) +
                0.6 * math.sin(2 * math.pi * f0 * 2.0 * t) +
                0.4 * math.sin(2 * math.pi * f0 * 3.0 * t))
        saturated = math.tanh(horn * 3.2)
        samples.append(saturated * env * 0.90)
    write_wav("heavy_cinematic_thud.wav", samples)

    # 21. heavy_glitch_bass.wav (Aggressive neuro-hop distorted bass growl, 1.0s)
    samples = []
    dur = 1.0
    for i in range(int(dur * sample_rate)):
        t = i / sample_rate
        env = math.exp(-t * 2.8)
        mod = 1.0 + 0.6 * math.sin(2 * math.pi * 32.0 * t)
        base = math.sin(2 * math.pi * 65.0 * t * mod) + 0.7 * math.sin(2 * math.pi * 130.0 * t)
        glitch = math.tanh(base * 4.0)
        samples.append(glitch * env * 0.85)
    write_wav("heavy_glitch_bass.wav", samples)

    # 22. heavy_metal_slam.wav (Crushing metal anvil strike with sub rumble, 1.1s)
    samples = []
    dur = 1.1
    for i in range(int(dur * sample_rate)):
        t = i / sample_rate
        env = math.exp(-t * 3.0)
        # Transient metal clatter + sub rumble
        metal = (math.sin(2 * math.pi * 840 * t) * math.exp(-t * 25.0) +
                 math.sin(2 * math.pi * 1250 * t) * math.exp(-t * 30.0) +
                 0.5 * math.sin(2 * math.pi * 50 * t))
        slam = math.tanh(metal * 3.0)
        samples.append(slam * env * 0.90)
    write_wav("heavy_metal_slam.wav", samples)

    print("[bootstrap] All 22 WAV audio cues synthesized successfully.")


def write_manifest_json():
    print("[bootstrap] Generating assets/manifest.json with Facial Vibe / Feelings Catalog...")
    manifest = {
        "version": "3.0.0",
        "description": "Catalog of 9-Emotion Meme Stickers, 3D Emojis, Heavy Audio Cues, and Facial Vibe Insights",
        "emotions": {
            "PEACE": {
                "name": "Peace",
                "archetype": "Shanta (Zen, Serenity)",
                "sticker": "assets/memes/peace_lotus.png",
                "emoji": "assets/memes/emoji_zen_angel.png",
                "sound": "assets/sounds/peace_bell.wav",
                "sound_alt": "assets/sounds/heavy_cinematic_thud.wav",
                "default_filter": "zen_mist",
                "color_bgr": [230, 245, 230],
                "vibe_insights": [
                    "VIBE: At 100% inner peace while the world burns",
                    "FEELING: Unbothered, moisturized, in their own lane",
                    "INSIGHT: Zero thoughts behind these eyes, pure bliss",
                    "MINDSET: The bug is officially a feature now"
                ],
                "fallback_captions": [
                    {"top": "INNER PEACE ACHIEVED", "bottom": "WIFI DISCONNECTED"},
                    {"top": "ZERO THOUGHTS", "bottom": "HEAD COMPLETELY EMPTY"},
                    {"top": "CALM AS A BUDDHA", "bottom": "CHAOS IN THE TERMINAL"},
                    {"top": "ACCEPTANCE STAGE", "bottom": "THE BUG IS A FEATURE NOW"}
                ]
            },
            "LOVE": {
                "name": "Love",
                "archetype": "Shringara (Affection, Romance)",
                "sticker": "assets/memes/love_hearts.png",
                "emoji": "assets/memes/emoji_heart_eyes.png",
                "sound": "assets/sounds/love_harp.wav",
                "sound_alt": "assets/sounds/wow_anime.wav",
                "default_filter": "dreamy_glow",
                "color_bgr": [200, 100, 255],
                "vibe_insights": [
                    "VIBE: Heart melting like butter on fresh toast",
                    "FEELING: Looking at their crush or a plate of biryani",
                    "INSIGHT: Pupil dilation 100%, smitten beyond repair",
                    "INTERNAL: Must protect this sweet smile at all costs"
                ],
                "fallback_captions": [
                    {"top": "POV: MY CRUSH", "bottom": "WALKED INTO THE ROOM"},
                    {"top": "ME LOOKING AT", "bottom": "CLEAN DOCUMENTED CODE"},
                    {"top": "WHEN THE CODE COMPILES", "bottom": "ON THE FIRST TRY"},
                    {"top": "HEART EYES ACTIVATED", "bottom": "SEND HELP IMMEDIATELY"}
                ]
            },
            "LAUGHTER": {
                "name": "Laughter",
                "archetype": "Hasya (Joy, Humor)",
                "sticker": "assets/memes/laughter_tears.png",
                "emoji": "assets/memes/emoji_laugh_crying.png",
                "sound": "assets/sounds/laughter_boing.wav",
                "sound_alt": "assets/sounds/heavy_bass_drop.wav",
                "default_filter": "cartoon_sketch",
                "color_bgr": [40, 220, 255],
                "vibe_insights": [
                    "VIBE: Remembering an illegal joke from 2019 in public",
                    "FEELING: 0% serious, 100% mischief and pure chaos",
                    "INSIGHT: Fighting for their life not to burst out laughing",
                    "THOUGHT: If I speak right now I will lose my dignity"
                ],
                "fallback_captions": [
                    {"top": "WHEN YOU FIX A BUG", "bottom": "BY DELETING THE TEST"},
                    {"top": "I'M NOT LAUGHING AT YOU", "bottom": "OK YES I AM"},
                    {"top": "WROTE 1 LINE OF CODE", "bottom": "FEELS LIKE HACKERMAN"},
                    {"top": "MY JOKE WAS HILARIOUS", "bottom": "THE AUDIENCE WAS WRONG"}
                ]
            },
            "SORROW": {
                "name": "Sorrow",
                "archetype": "Karuna (Grief, Melancholy)",
                "sticker": "assets/memes/sorrow_raincloud.png",
                "emoji": "assets/memes/emoji_crying_stream.png",
                "sound": "assets/sounds/sorrow_violin.wav",
                "sound_alt": "assets/sounds/heavy_cinematic_thud.wav",
                "default_filter": "noir",
                "color_bgr": [220, 140, 60],
                "vibe_insights": [
                    "VIBE: Dramatic main character in the rain gazing out a window",
                    "FEELING: Remembering an embarrassing thing said in 6th grade",
                    "INSIGHT: Critical emotional damage, needs chai and hugs ASAP",
                    "THOUGHT: Alexa, play Sad Romance on a 10-hour loop"
                ],
                "fallback_captions": [
                    {"top": "GIT PUSH --FORCE", "bottom": "ON PRODUCTION MAIN"},
                    {"top": "HELLO DARKNESS", "bottom": "MY OLD FRIEND"},
                    {"top": "FORGOT THE WHERE CLAUSE", "bottom": "IN THE DELETE STATEMENT"},
                    {"top": "EMOTIONAL DAMAGE", "bottom": "MAXIMUM SEVERITY"}
                ]
            },
            "ANGER": {
                "name": "Anger",
                "archetype": "Raudra (Fury, Rage)",
                "sticker": "assets/memes/anger_steam.png",
                "emoji": "assets/memes/emoji_angry_devil.png",
                "sound": "assets/sounds/heavy_bass_drop.wav",
                "sound_alt": "assets/sounds/heavy_metal_slam.wav",
                "default_filter": "deep_fried",
                "color_bgr": [30, 40, 240],
                "vibe_insights": [
                    "VIBE: 1 minor inconvenience away from picking up a keyboard to fight",
                    "FEELING: Who touched the prod database indentation?!",
                    "INSIGHT: Blood pressure spike detected, caution advised",
                    "THOUGHT: Violence isn't the answer, it's a question and the answer is YES"
                ],
                "fallback_captions": [
                    {"top": "WHO TOUCHED", "bottom": "MY INDENTATION"},
                    {"top": "MERGE CONFLICT IN LINE 1", "bottom": "KEYBOARD WILL SUFFER"},
                    {"top": "CALM IS NOT AN OPTION", "bottom": "VIOLENCE IS CHOSEN"},
                    {"top": "IT WORKED ON MY MACHINE", "bottom": "WHY DEPLOY IT THEN"}
                ]
            },
            "COURAGE": {
                "name": "Courage",
                "archetype": "Vira (Heroic Resolve, Grit)",
                "sticker": "assets/memes/courage_aura.png",
                "emoji": "assets/memes/emoji_gigachad.png",
                "sound": "assets/sounds/courage_fanfare.wav",
                "sound_alt": "assets/sounds/heavy_cinematic_thud.wav",
                "default_filter": "golden_aura",
                "color_bgr": [40, 200, 255],
                "vibe_insights": [
                    "VIBE: Boss fight music is playing and I AM the final boss",
                    "FEELING: About to push direct to prod with zero tests and zero fear",
                    "INSIGHT: Pure GigaChad aura, jawline sharp enough to cut glass",
                    "MINDSET: I didn't come to participate, I came to conquer"
                ],
                "fallback_captions": [
                    {"top": "FINAL BOSS MUSIC STARTS", "bottom": "AND I AM READY"},
                    {"top": "TESTING DIRECTLY IN PROD", "bottom": "LIKE A TRUE CHAD"},
                    {"top": "NEVER BACK DOWN", "bottom": "NEVER WHAT? NEVER GIVE UP"},
                    {"top": "MAIN CHARACTER ENERGY", "bottom": "LEVEL NINE THOUSAND"}
                ]
            },
            "FEAR": {
                "name": "Fear",
                "archetype": "Bhayanaka (Terror, Shock)",
                "sticker": "assets/memes/fear_ghost.png",
                "emoji": "assets/memes/emoji_skull_dead.png",
                "sound": "assets/sounds/heavy_glitch_bass.wav",
                "sound_alt": "assets/sounds/fear_sting.wav",
                "default_filter": "eerie_vignette",
                "color_bgr": [240, 100, 180],
                "vibe_insights": [
                    "VIBE: Just remembered the assignment was due 10 minutes ago",
                    "FEELING: The senior dev just sent 'quick call?' on Slack",
                    "INSIGHT: Survival instincts triggered, fight-or-flight at 99%",
                    "THOUGHT: If I don't move, maybe the deadline can't see me"
                ],
                "fallback_captions": [
                    {"top": "THE SENIOR DEV", "bottom": "JUST TYPED 'CAN WE TALK?'"},
                    {"top": "SSH TERMINAL UNRESPONSIVE", "bottom": "SWEAT INTENSIFIES"},
                    {"top": "WHEN THE SCRIPT RUNS", "bottom": "FOR 10 MINUTES IN SILENCE"},
                    {"top": "HEAR FOOTSTEPS AT 3 AM", "bottom": "JUST THE GARBAGE COLLECTOR"}
                ]
            },
            "DISGUST": {
                "name": "Disgust",
                "archetype": "Bibhatsa (Revulsion, Aversion)",
                "sticker": "assets/memes/disgust_nauseous.png",
                "emoji": "assets/memes/emoji_clown.png",
                "sound": "assets/sounds/bruh.wav",
                "sound_alt": "assets/sounds/disgust_gag.wav",
                "default_filter": "toxic_matrix",
                "color_bgr": [60, 220, 80],
                "vibe_insights": [
                    "VIBE: Reading legacy code with 14 nested IF statements",
                    "FEELING: Major ick detected from 5 miles away",
                    "INSIGHT: Involuntary cringe reflex, instant revulsion",
                    "THOUGHT: Eww brother eww... what is that?!"
                ],
                "fallback_captions": [
                    {"top": "MAJOR ICK DETECTED", "bottom": "1000 YARDS OF DISTANCE NEEDED"},
                    {"top": "READING LEGACY CODE", "bottom": "WRITTEN IN 2012"},
                    {"top": "NO COMMENTS ANYWHERE", "bottom": "VARIABLE NAME IS 'xyz_final2'"},
                    {"top": "THAT OPINION", "bottom": "MADE ME PHYSICALLY ILL"}
                ]
            },
            "WONDER": {
                "name": "Wonder",
                "archetype": "Adbhuta (Awe, Amazement)",
                "sticker": "assets/memes/wonder_stars.png",
                "emoji": "assets/memes/emoji_shocked_mindblown.png",
                "sound": "assets/sounds/heavy_cinematic_thud.wav",
                "sound_alt": "assets/sounds/wonder_chime.wav",
                "default_filter": "cosmic_shift",
                "color_bgr": [255, 220, 60],
                "vibe_insights": [
                    "VIBE: Just realized the universe is expanding into nothingness",
                    "FEELING: When the code actually compiled on the first try",
                    "INSIGHT: Mind officially blown into a million cosmic stars",
                    "THOUGHT: Wait... are we living in a simulation?!"
                ],
                "fallback_captions": [
                    {"top": "MIND EQUALS BLOWN", "bottom": "COSMOS UNLOCKED"},
                    {"top": "WHEN THE RECURSIVE FUNCTION", "bottom": "ACTUALLY TERMINATES"},
                    {"top": "IS THIS REAL LIFE", "bottom": "OR JUST MATRIX SIMULATION"},
                    {"top": "A WHOLE NEW WORLD", "bottom": "OF UNDOCUMENTED APIS"}
                ]
            }
        },
        "soundboard": {
            "heavy_drop": "assets/sounds/heavy_bass_drop.wav",
            "cinematic_thud": "assets/sounds/heavy_cinematic_thud.wav",
            "glitch_bass": "assets/sounds/heavy_glitch_bass.wav",
            "metal_slam": "assets/sounds/heavy_metal_slam.wav",
            "vine_boom": "assets/sounds/vine_boom.wav",
            "bruh": "assets/sounds/bruh.wav",
            "thug_horn": "assets/sounds/thug_life_horn.wav",
            "wow": "assets/sounds/wow_anime.wav",
            "emotional_damage": "assets/sounds/emotional_damage.wav",
            "record_scratch": "assets/sounds/record_scratch.wav",
            "victory": "assets/sounds/victory_fanfare.wav",
            "scream": "assets/sounds/screaming_goat.wav",
            "shutter": "assets/sounds/shutter.wav"
        }
    }

    manifest_path = ASSETS_DIR / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"[bootstrap] Manifest with Facial Vibe Insights saved to {manifest_path}")


def main():
    print("=" * 60)
    print("Webcam Emotion & Meme Filter: Heavy Audio & Vibe Bootstrapper")
    print("=" * 60)
    ensure_directories()
    download_model()
    create_stickers_and_emojis()
    synthesize_all_sounds()
    write_manifest_json()
    print("=" * 60)
    print("[bootstrap] Complete! 19 stickers/emojis, 22 audio cues & manifest ready.")
    print("=" * 60)


if __name__ == "__main__":
    main()
