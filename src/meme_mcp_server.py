"""
FastMCP Server: Multi-Face Reaction Director, Soundboard & Facial Vibe Engine over stdio.
"""

import os
import json
import random
from pathlib import Path
from typing import Dict, Any, List

ROOT_DIR = Path(__file__).resolve().parent.parent
MANIFEST_PATH = ROOT_DIR / "assets" / "manifest.json"

VALID_FILTERS = {
    "normal", "cartoon_sketch", "pencil_sketch", "cyberpunk_neon", "dreamy_glow", "noir", "deep_fried",
    "golden_aura", "eerie_vignette", "toxic_matrix", "cosmic_shift", "zen_mist"
}

SESSION_HISTORY: List[Dict[str, Any]] = []


def load_manifest() -> Dict[str, Any]:
    if MANIFEST_PATH.exists():
        try:
            with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"emotions": {}, "soundboard": {}}


def get_asset_catalog() -> str:
    """Returns the full JSON catalog of stickers, 3D emojis, heavy audio cues, and facial vibe insights."""
    return json.dumps(load_manifest(), indent=2)


def get_session_history() -> str:
    """Returns JSON list of recent emotions and captions generated in this session."""
    return json.dumps(SESSION_HISTORY[-12:], indent=2)


def select_overlay_bundle(emotion: str, intensity: float = 1.0, comedic_angle: str = "relatable", face_id: int = 0) -> str:
    """
    Picks matching theme sticker, 3D emoji, SFX path, default filter, and 'facial vibe insight' for a specific face.
    """
    manifest = load_manifest()
    emo_key = emotion.upper().strip()
    emotions_dict = manifest.get("emotions", {})

    if emo_key not in emotions_dict:
        emo_key = "PEACE"

    data = emotions_dict.get(emo_key, {})
    sticker = data.get("sticker", "assets/memes/peace_lotus.png")
    emoji = data.get("emoji", "assets/memes/emoji_zen_angel.png")
    sound = data.get("sound", "assets/sounds/peace_bell.wav")
    filter_mode = data.get("default_filter", "normal")

    vibe_list = data.get("vibe_insights", [f"VIBE: High {emo_key} energy"])
    chosen_vibe = random.choice(vibe_list)

    sticker_abs = str(ROOT_DIR / sticker) if not os.path.isabs(sticker) else sticker
    emoji_abs = str(ROOT_DIR / emoji) if not os.path.isabs(emoji) else emoji
    sound_abs = str(ROOT_DIR / sound) if not os.path.isabs(sound) else sound

    result = {
        "status": "ok",
        "face_id": face_id,
        "emotion": emo_key,
        "sticker_path": sticker_abs,
        "emoji_path": emoji_abs,
        "sound_path": sound_abs,
        "default_filter": filter_mode,
        "vibe_insight": chosen_vibe,
        "intensity": intensity,
        "comedic_angle": comedic_angle
    }
    return json.dumps(result)


def set_screen_caption(top_text: str, bottom_text: str, face_id: int = 0) -> str:
    """Sets punchy 2-line meme typography for a given face."""
    t = top_text.strip().upper()[:45]
    b = bottom_text.strip().upper()[:45]

    entry = {"face_id": face_id, "top": t, "bottom": b}
    SESSION_HISTORY.append(entry)

    return json.dumps({
        "status": "ok",
        "face_id": face_id,
        "top_text": t,
        "bottom_text": b
    })


def trigger_audio_cue(cue_name: str) -> str:
    """Triggers a validated .wav audio cue."""
    cue = cue_name.strip()
    if not cue.endswith(".wav"):
        cue = f"{cue}.wav"

    sound_path = ROOT_DIR / "assets" / "sounds" / cue
    if not sound_path.exists():
        sound_path = ROOT_DIR / "assets" / "sounds" / "heavy_bass_drop.wav"
        if not sound_path.exists():
            sound_path = ROOT_DIR / "assets" / "sounds" / "shutter.wav"

    return json.dumps({
        "status": "ok",
        "sound_path": str(sound_path)
    })


def trigger_soundboard(effect_name: str) -> str:
    """
    Triggers viral meme and heavy bass soundboard cues.
    Effects: heavy_drop, cinematic_thud, glitch_bass, metal_slam, vine_boom, bruh, thug_horn, wow, emotional_damage, record_scratch, victory, scream, shutter.
    """
    manifest = load_manifest()
    sb = manifest.get("soundboard", {})
    key = effect_name.lower().strip()
    rel_path = sb.get(key, "assets/sounds/heavy_bass_drop.wav")
    abs_path = str(ROOT_DIR / rel_path)
    return json.dumps({
        "status": "ok",
        "effect": key,
        "sound_path": abs_path
    })


def set_visual_filter(filter_mode: str) -> str:
    """Switches the real-time webcam visual shader."""
    mode = filter_mode.lower().strip()
    if mode not in VALID_FILTERS:
        mode = "normal"

    return json.dumps({
        "status": "ok",
        "filter_mode": mode
    })


# FastMCP Server Registration
mcp = None
try:
    # pyrefly: ignore[missing-import]
    from mcp.server.fastmcp import FastMCP
    mcp = FastMCP("MemeDirectorMCP")
    mcp.resource("meme://catalog/assets")(get_asset_catalog)
    mcp.resource("meme://session/history")(get_session_history)
    mcp.tool()(select_overlay_bundle)
    mcp.tool()(set_screen_caption)
    mcp.tool()(trigger_audio_cue)
    mcp.tool()(trigger_soundboard)
    mcp.tool()(set_visual_filter)
except ImportError:
    pass


def main():
    if mcp is None:
        print("[MemeDirectorMCP] Error: 'mcp' package is not installed. Please install requirements.txt.")
        return
    mcp.run()


if __name__ == "__main__":
    main()
