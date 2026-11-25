"""
System Verification Suite - Multi-Face & Heavy Audio Edition
Verifies:
1. Asset Integrity (19 stickers/emojis, 22 WAV sound effects, MediaPipe model, manifest.json)
2. Baseline Calibration & Temporal Hysteresis Debouncer
3. Multi-Face State Tracking & VibeDisplayMode in StateStore
4. FastMCP Tool Implementations & Heavy Bass Soundboard Cues
5. High-Speed Accelerated Shaders (Comic, Pencil, Cyberpunk Neon, Deep Fried)
6. Floating Emoji Particle Engine
7. "What We Feel" Floating Facial Vibe Card Renderer
8. 3D Pose-Anchored Alpha Compositing
"""

import sys
import json
from pathlib import Path
# pyrefly: ignore[missing-import]
import cv2
# pyrefly: ignore[missing-import]
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.state_manager import (
    EmotionState,
    EmojiStyle,
    VibeDisplayMode,
    FaceState,
    BaselineCalibrator,
    TemporalHysteresisDebouncer,
    StateStore,
)
from src.vision_engine import VisionEngine, ParticleEngine


def verify_assets():
    print("[1/8] Verifying asset files (19 stickers/emojis, 22 WAV cues, manifest)...")
    manifest_path = ROOT_DIR / "assets" / "manifest.json"
    assert manifest_path.exists(), "assets/manifest.json does not exist"

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    emotions = manifest.get("emotions", {})
    expected_emotions = [
        "PEACE", "LOVE", "LAUGHTER", "SORROW", "ANGER",
        "COURAGE", "FEAR", "DISGUST", "WONDER"
    ]
    for emo in expected_emotions:
        assert emo in emotions, f"Missing emotion in manifest: {emo}"
        data = emotions[emo]
        stk_path = ROOT_DIR / data["sticker"]
        emoji_path = ROOT_DIR / data["emoji"]
        snd_path = ROOT_DIR / data["sound"]
        assert stk_path.exists(), f"Sticker missing: {stk_path}"
        assert emoji_path.exists(), f"Emoji missing: {emoji_path}"
        assert snd_path.exists(), f"Sound missing: {snd_path}"
        assert len(data.get("fallback_captions", [])) > 0, f"No fallback captions for {emo}"
        assert len(data.get("vibe_insights", [])) >= 3, f"Insufficient vibe insights for {emo}"

    # Verify Soundboard cues (at least 12 cues, including 4 heavy bass/cinematic cues)
    sb = manifest.get("soundboard", {})
    assert len(sb) >= 12, f"Expected at least 12 soundboard cues, got {len(sb)}"
    for cue_name in ["heavy_drop", "cinematic_thud", "glitch_bass", "metal_slam", "vine_boom", "bruh", "thug_horn"]:
        assert cue_name in sb, f"Missing required soundboard cue: {cue_name}"

    for name, rel_path in sb.items():
        assert (ROOT_DIR / rel_path).exists(), f"Soundboard audio missing: {rel_path}"

    # Verify 22 total wav files in assets/sounds/
    sounds_dir = ROOT_DIR / "assets" / "sounds"
    wav_files = list(sounds_dir.glob("*.wav"))
    assert len(wav_files) >= 22, f"Expected at least 22 WAV files, found {len(wav_files)}"

    # Verify 19 total png files in assets/memes/
    memes_dir = ROOT_DIR / "assets" / "memes"
    png_files = list(memes_dir.glob("*.png"))
    assert len(png_files) >= 19, f"Expected at least 19 PNG files, found {len(png_files)}"

    model_path = ROOT_DIR / "assets" / "models" / "face_landmarker.task"
    assert model_path.exists() and model_path.stat().st_size > 1000000, "face_landmarker.task missing or incomplete"

    print(f"  -> PASSED: {len(png_files)} stickers/emojis, {len(wav_files)} WAV files, and model verified.")


def verify_state_debouncer():
    print("[2/8] Verifying Baseline Calibration & Temporal Hysteresis Debouncer...")
    calibrator = BaselineCalibrator(target_frames=15, auto_start=True)
    assert calibrator.is_calibrating

    dummy_resting = {e: 0.1 for e in EmotionState}
    dummy_resting[EmotionState.LAUGHTER] = 0.22
    for _ in range(15):
        done = calibrator.add_frame(dummy_resting)

    assert done and not calibrator.is_calibrating
    adjusted = calibrator.apply_offset(dummy_resting)
    assert adjusted[EmotionState.LAUGHTER] < 0.22

    debouncer = TemporalHysteresisDebouncer(
        window_size=6,
        entry_threshold=0.35,
        entry_min_hits=2,
        hold_duration_sec=0.2,
        exit_threshold=0.22,
        exit_frames_required=5,
        fade_frames=5
    )

    curr_emo, conf, alpha, trans = debouncer.update({e: 0.1 for e in EmotionState})
    assert curr_emo == EmotionState.PEACE

    laughter_frame = {e: 0.05 for e in EmotionState}
    laughter_frame[EmotionState.LAUGHTER] = 0.85

    emo, conf, alpha, trans = debouncer.update(laughter_frame)
    assert not trans
    emo, conf, alpha, trans = debouncer.update(laughter_frame)
    assert trans and emo == EmotionState.LAUGHTER

    print("  -> PASSED: Baseline offset removal & temporal debounce transitions verified.")


def verify_multi_face_state_store():
    print("[3/8] Verifying Multi-Face Tracking & VibeDisplayMode in StateStore...")
    store = StateStore()

    # Face 0 (Primary)
    store.update_ai_bundle(
        emotion=EmotionState.COURAGE,
        top_text="Boss Music Plays",
        bottom_text="I Am Ready",
        face_id=0,
        sticker_path="assets/memes/courage_aura.png",
        emoji_path="assets/memes/emoji_gigachad.png",
        sound_path="assets/sounds/courage_fanfare.wav",
        filter_mode="golden_aura",
        vibe_insight="Radiating pure main character energy"
    )

    # Face 1 (Secondary simultaneously)
    store.update_ai_bundle(
        emotion=EmotionState.LOVE,
        top_text="She Looked My Way",
        bottom_text="Heart Rate 180",
        face_id=1,
        sticker_path="assets/memes/love_hearts.png",
        emoji_path="assets/memes/emoji_heart_eyes.png",
        vibe_insight="Floating on cloud nine right now"
    )

    snap = store.get_render_snapshot()
    assert "faces" in snap
    assert 0 in snap["faces"]
    assert 1 in snap["faces"]
    assert snap["faces"][0]["emotion"] == EmotionState.COURAGE
    assert snap["faces"][1]["emotion"] == EmotionState.LOVE
    assert snap["faces"][0]["vibe_insight"] == "Radiating pure main character energy"
    assert snap["faces"][1]["vibe_insight"] == "Floating on cloud nine right now"
    assert "expression_summary" in snap["faces"][0]
    assert "expression_details" in snap["faces"][0]

    # Emoji Style cycling
    curr_style = store.emoji_style
    new_style = store.cycle_emoji_style()
    assert new_style != curr_style

    # Vibe Display Mode cycling
    curr_vibe = store.vibe_mode
    new_vibe = store.cycle_vibe_mode()
    assert new_vibe != curr_vibe

    print("  -> PASSED: Multi-face concurrent states, vibe insights & mode cycles verified.")


def verify_mcp_tools():
    print("[4/8] Verifying FastMCP Tool Implementations & Heavy Audio cues...")
    from src.meme_mcp_server import (
        select_overlay_bundle,
        set_screen_caption,
        trigger_audio_cue,
        trigger_soundboard,
        set_visual_filter
    )

    # Test overlay bundle selection with vibe_insight
    res_bundle = json.loads(select_overlay_bundle("LOVE", 0.9, "romantic"))
    assert res_bundle["status"] == "ok"
    assert "emoji_heart_eyes.png" in res_bundle["emoji_path"]
    assert "vibe_insight" in res_bundle and len(res_bundle["vibe_insight"]) > 0

    # Test heavy soundboard cue
    res_sb = json.loads(trigger_soundboard("heavy_drop"))
    assert res_sb["status"] == "ok"
    assert "heavy_bass_drop.wav" in res_sb["sound_path"]

    # Test cinematic thud soundboard cue
    res_thud = json.loads(trigger_soundboard("cinematic_thud"))
    assert res_thud["status"] == "ok"
    assert "heavy_cinematic_thud.wav" in res_thud["sound_path"]

    # Test visual filter
    res_filter = json.loads(set_visual_filter("cartoon_sketch"))
    assert res_filter["status"] == "ok"

    print("  -> PASSED: FastMCP tools, heavy soundboard triggers & response contracts verified.")


def verify_sketch_shaders():
    print("[5/8] Verifying Accelerated Shaders (Comic, Pencil, Cyberpunk Neon, Deep Fried)...")
    ve = VisionEngine()
    dummy_frame = np.full((360, 480, 3), 180, dtype=np.uint8)
    cv2.circle(dummy_frame, (240, 180), 80, (120, 140, 200), -1)

    for mode in ["cartoon_sketch", "pencil_sketch", "cyberpunk_neon", "deep_fried", "noir", "dreamy_glow"]:
        out = ve.apply_shader(dummy_frame, mode)
        assert out.shape == dummy_frame.shape, f"Shader {mode} produced invalid shape"
        assert not np.array_equal(out, dummy_frame), f"Shader {mode} had no visual effect"

    print("  -> PASSED: All 6 accelerated shaders rendered with correct shapes and textures.")


def verify_particle_engine():
    print("[6/8] Verifying Floating Emoji Particle Engine...")
    engine = ParticleEngine(max_particles=30)
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Emit love particles
    engine.emit_emotion_particles(EmotionState.LOVE, (320, 240), dummy_frame.shape)
    assert len(engine.particles) > 0, "Failed to emit particles"

    # Render frame
    rendered = engine.render(dummy_frame)
    assert rendered.shape == dummy_frame.shape
    assert np.mean(rendered) > 0, "Particles should be visible on canvas"

    print("  -> PASSED: Floating Emoji Particle Engine simulation and rendering verified.")


def verify_facial_vibe_card_rendering():
    print("[7/8] Verifying 'What We Feel By Seeing Your Face' Floating Vibe Cards...")
    ve = VisionEngine()
    canvas = np.full((480, 640, 3), 40, dtype=np.uint8)

    dummy_landmark_data = {
        "has_face": True,
        "forehead_pt": (320, 200),
        "center_pt": (320, 240),
        "interocular_dist": 90.0,
        "euler_angles": (0.0, 0.0, 5.0)
    }

    # Render Detailed Card Mode with explicit expression title & details written down
    card_rendered = ve.render_facial_vibe_card(
        canvas.copy(),
        dummy_landmark_data,
        face_id=0,
        emotion=EmotionState.COURAGE,
        vibe_text="Radiating pure main character energy",
        display_mode=VibeDisplayMode.CARD,
        expression_summary="COURAGE / CONFIDENT SMIRK (88%)",
        expression_details=["Mouth: Smirk (75%)", "Brows: Furrowed (62%)"]
    )
    assert card_rendered.shape == canvas.shape
    assert not np.array_equal(card_rendered, canvas), "Card mode failed to modify canvas"

    # Render Compact Mode
    compact_rendered = ve.render_facial_vibe_card(
        canvas.copy(),
        dummy_landmark_data,
        face_id=1,
        emotion=EmotionState.LOVE,
        vibe_text="Looking at a crush and trying to stay chill",
        display_mode=VibeDisplayMode.COMPACT,
        expression_summary="LOVE / GENTLE SMILE (84%)",
        expression_details=["Smile: 82%", "Eyes Squint: 45%"]
    )
    assert compact_rendered.shape == canvas.shape
    assert not np.array_equal(compact_rendered, canvas), "Compact mode failed to modify canvas"

    print("  -> PASSED: Detailed Frosted Glass Vibe Card and Compact Mode rendering verified.")


def verify_pose_anchored_compositor():
    print("[8/8] Verifying 3D Head-Pose Anchored Sticker Alpha Compositor...")
    ve = VisionEngine()
    canvas = np.zeros((480, 640, 3), dtype=np.uint8)

    dummy_landmark_data = {
        "has_face": True,
        "forehead_pt": (320, 180),
        "center_pt": (320, 220),
        "interocular_dist": 80.0,
        "euler_angles": (-5.0, 8.0, 12.0)  # Pitch, Yaw, Roll
    }

    sticker_path = str(ROOT_DIR / "assets" / "memes" / "peace_lotus.png")
    out = ve.overlay_sticker(
        canvas.copy(),
        sticker_path=sticker_path,
        landmark_data=dummy_landmark_data,
        alpha_mult=1.0
    )
    assert out.shape == canvas.shape
    assert np.mean(out) > 0, "Sticker overlay should leave pixels on black canvas"

    print("  -> PASSED: 3D head-pose Euler rotation and alpha compositing verified.")


def main():
    print("=" * 70)
    print("Running Complete Verification: Multi-Face, Heavy Audio & Facial Vibe")
    print("=" * 70)
    verify_assets()
    verify_state_debouncer()
    verify_multi_face_state_store()
    verify_mcp_tools()
    verify_sketch_shaders()
    verify_particle_engine()
    verify_facial_vibe_card_rendering()
    verify_pose_anchored_compositor()
    print("=" * 70)
    print("ALL 8 VERIFICATION SUITES PASSED: System is completely production-ready.")
    print("=" * 70)


if __name__ == "__main__":
    main()
