"""
Main Application Entrypoint: Multi-Face Reaction Engine (num_faces=4),
High-Accuracy Dual-Engine Emotion Classifier, Pro Comic/Pencil Shaders,
Floating Facial Vibe Insights ("What We Feel By Seeing Your Face"),
and Heavy Bass Soundboard with 60+ FPS Threaded Camera Capture.
"""

import os
import sys
import time
import queue
import random
import threading
from pathlib import Path
from datetime import datetime
from collections import deque
from typing import Optional, Dict, Tuple, Any

# pyrefly: ignore[missing-import]
import cv2
# pyrefly: ignore[missing-import]
import numpy as np

# Audio mixer
try:
    # pyrefly: ignore[missing-import]
    import pygame
    pygame.mixer.init(frequency=44100, size=-16, channels=4, buffer=512)
    PYGAME_AVAILABLE = True
except Exception as e:
    PYGAME_AVAILABLE = False
    print(f"[Main] Warning: Pygame mixer unavailable ({e}).")

from src.state_manager import StateStore, EmotionState, EmojiStyle, VibeDisplayMode
from src.vision_engine import VisionEngine
from src.nvidia_mcp_client import NvidiaMcpWorker

ROOT_DIR = Path(__file__).resolve().parent.parent
EXPORTS_DIR = ROOT_DIR / "exports"
SOUNDS_DIR = ROOT_DIR / "assets" / "sounds"


class SoundManager:
    """Preloads all 22 .wav sounds into RAM at startup for zero-lag instant playback."""
    def __init__(self):
        self.sounds: Dict[str, Any] = {}
        self.soundboard_keys = [
            "heavy_bass_drop.wav", "heavy_cinematic_thud.wav", "heavy_glitch_bass.wav",
            "heavy_metal_slam.wav", "vine_boom.wav", "bruh.wav", "thug_life_horn.wav",
            "wow_anime.wav", "emotional_damage.wav", "victory_fanfare.wav",
            "screaming_goat.wav", "record_scratch.wav", "laughter_boing.wav", "shutter.wav"
        ]
        self.soundboard_idx = 0
        self.heavy_idx = 0
        self._preload_sounds()

    def _preload_sounds(self):
        if not PYGAME_AVAILABLE:
            return
        if not SOUNDS_DIR.exists():
            return

        for wav_file in SOUNDS_DIR.glob("*.wav"):
            try:
                snd = pygame.mixer.Sound(str(wav_file))
                snd.set_volume(0.85)
                self.sounds[wav_file.name] = snd
                self.sounds[str(wav_file)] = snd
            except Exception:
                pass
        print(f"[SoundManager] Preloaded {len(self.sounds) // 2} WAV audio cues into memory.")

    def play(self, sound_path_or_name: str, volume: float = 0.85):
        if not PYGAME_AVAILABLE or not sound_path_or_name:
            return

        name = Path(sound_path_or_name).name
        snd = self.sounds.get(name) or self.sounds.get(sound_path_or_name)
        if snd is None and os.path.exists(sound_path_or_name):
            try:
                snd = pygame.mixer.Sound(sound_path_or_name)
                snd.set_volume(volume)
                self.sounds[name] = snd
            except Exception:
                return

        if snd:
            channel = pygame.mixer.find_channel()
            if channel:
                channel.set_volume(volume)
                channel.play(snd)
            else:
                snd.play()

    def play_next_soundboard(self) -> str:
        key = self.soundboard_keys[self.soundboard_idx % len(self.soundboard_keys)]
        self.soundboard_idx += 1
        self.play(key, volume=0.95)
        return key.replace(".wav", "").replace("_", " ").title()

    def play_heavy_bass(self) -> str:
        heavy_cues = ["heavy_bass_drop.wav", "heavy_cinematic_thud.wav", "heavy_glitch_bass.wav", "heavy_metal_slam.wav"]
        cue = heavy_cues[self.heavy_idx % len(heavy_cues)]
        self.heavy_idx += 1
        self.play(cue, volume=1.0)
        return cue.replace(".wav", "").replace("_", " ").upper()


class ThreadedCamera:
    """Dedicated background thread reading frames from webcam for smooth 60+ FPS without camera I/O blocking."""
    def __init__(self, src=0, width=1280, height=720):
        self.cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(src)

        if not self.cap.isOpened() and src == 0:
            self.cap = cv2.VideoCapture(1)

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, 60)

        self.grabbed, self.frame = self.cap.read()
        self.started = False
        self.read_lock = threading.Lock()

    def start(self):
        if self.started:
            return self
        self.started = True
        self.thread = threading.Thread(target=self._update, daemon=True, name="ThreadedCameraWorker")
        self.thread.start()
        return self

    def _update(self):
        while self.started:
            grabbed, frame = self.cap.read()
            if grabbed:
                with self.read_lock:
                    self.grabbed = grabbed
                    self.frame = frame
            else:
                time.sleep(0.005)

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        with self.read_lock:
            if not self.grabbed or self.frame is None:
                return False, None
            return True, self.frame.copy()

    def is_opened(self) -> bool:
        return self.cap.isOpened()

    def stop(self):
        self.started = False
        if hasattr(self, 'thread'):
            self.thread.join(timeout=0.5)
        self.cap.release()


EMOTION_COLORS = {
    EmotionState.PEACE: (230, 245, 230),     # Mint / White
    EmotionState.LOVE: (200, 100, 255),      # Pink
    EmotionState.LAUGHTER: (40, 220, 255),   # Yellow
    EmotionState.SORROW: (220, 140, 60),     # Navy Blue
    EmotionState.ANGER: (30, 40, 240),       # Crimson Red
    EmotionState.COURAGE: (40, 200, 255),    # Gold
    EmotionState.FEAR: (240, 100, 180),      # Violet / Purple
    EmotionState.DISGUST: (60, 220, 80),     # Toxic Green
    EmotionState.WONDER: (255, 220, 60),     # Cyan
}


def draw_hud(
    frame: np.ndarray,
    snapshot: dict,
    num_faces_tracked: int,
    toast_msg: Optional[str] = None
) -> np.ndarray:
    h, w = frame.shape[:2]

    # Telemetry Panel
    card_w, card_h = 390, 240
    card_overlay = frame.copy()
    cv2.rectangle(card_overlay, (10, 10), (10 + card_w, 10 + card_h), (16, 16, 20), -1)
    cv2.rectangle(card_overlay, (10, 10), (10 + card_w, 10 + card_h), (70, 70, 85), 1)
    frame = cv2.addWeighted(card_overlay, 0.78, frame, 0.22, 0)

    fps = snapshot["fps"]
    nim_status = snapshot["nim_status"]
    is_muted = snapshot["is_muted"]
    is_calibrating = snapshot["is_calibrating"]
    calib_progress = snapshot["calib_progress"]
    sensitivity = snapshot["sensitivity"]
    emoji_style: EmojiStyle = snapshot.get("emoji_style", EmojiStyle.COMBO)
    vibe_mode: VibeDisplayMode = snapshot.get("vibe_mode", VibeDisplayMode.CARD)
    filter_mode = snapshot["filter_mode"]

    # 1. FPS & NIM Status
    cv2.putText(frame, f"FPS: {fps:.1f}", (24, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 128), 1, cv2.LINE_AA)
    badge_color = (0, 255, 128) if nim_status == "ONLINE" else (100, 180, 255)
    cv2.putText(frame, f"NIM: {nim_status}", (115, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.45, badge_color, 1, cv2.LINE_AA)

    # 2. Multi-Face Tracking Badge
    if num_faces_tracked > 0:
        cv2.putText(frame, f"FACES TRACKED: {num_faces_tracked} (SEPARATE REACTIONS)", (24, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 255, 128), 1, cv2.LINE_AA)
    else:
        cv2.putText(frame, "FACES: SEARCHING... (Position faces in frame)", (24, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 180, 255), 1, cv2.LINE_AA)

    # 3. Per-Face Real-Time Emotion Readouts
    faces_data = snapshot.get("faces", {})
    line_y = 74
    for fid in sorted(faces_data.keys())[:3]:
        finfo = faces_data[fid]
        emo = finfo["emotion"]
        conf = finfo["confidence"]
        col = EMOTION_COLORS.get(emo, (255, 255, 255))
        expr_text = finfo.get("expression_summary") or f"{emo.value} ({int(conf * 100)}%)"

        cv2.putText(frame, f"FACE #{fid + 1}: {expr_text}", (24, line_y), cv2.FONT_HERSHEY_SIMPLEX, 0.44, col, 1, cv2.LINE_AA)

        # Mini confidence bar
        bar_x = 260
        bar_w = 120
        cv2.rectangle(frame, (bar_x, line_y - 10), (bar_x + bar_w, line_y - 2), (40, 40, 45), -1)
        fill_w = int(bar_w * np.clip(conf, 0.0, 1.0))
        cv2.rectangle(frame, (bar_x, line_y - 10), (bar_x + fill_w, line_y - 2), col, -1)
        cv2.rectangle(frame, (bar_x, line_y - 10), (bar_x + bar_w, line_y - 2), (100, 100, 110), 1)

        line_y += 20

    # 4. Filter & Controls Status
    cv2.putText(frame, f"Filter: {filter_mode}", (24, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Vibe [V]: {vibe_mode.value}", (200, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 215, 60), 1, cv2.LINE_AA)

    cv2.putText(frame, f"Emoji [E]: {emoji_style.value}", (24, 172), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 140, 220), 1, cv2.LINE_AA)
    mute_str = "MUTED" if is_muted else "SFX ON"
    mute_col = (0, 0, 255) if is_muted else (0, 255, 180)
    cv2.putText(frame, mute_str, (200, 172), cv2.FONT_HERSHEY_SIMPLEX, 0.42, mute_col, 1, cv2.LINE_AA)

    cv2.putText(frame, f"Sens: {sensitivity:.1f}x [+/-]", (24, 194), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 230, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, "Soundboard [S] | Heavy [H]", (150, 194), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 160, 80), 1, cv2.LINE_AA)

    if is_calibrating:
        cv2.putText(frame, f"CALIBRATING BASELINE ({int(calib_progress * 100)}%)...", (24, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 220, 255), 1, cv2.LINE_AA)

    # Toast Overlay for soundboard or mode change
    if toast_msg:
        toast_w = 400
        cv2.rectangle(frame, (w // 2 - toast_w // 2, 20), (w // 2 + toast_w // 2, 65), (25, 15, 40), -1)
        cv2.rectangle(frame, (w // 2 - toast_w // 2, 20), (w // 2 + toast_w // 2, 65), (255, 120, 220), 2)
        cv2.putText(frame, toast_msg, (w // 2 - toast_w // 2 + 20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 240, 255), 2, cv2.LINE_AA)

    # Bottom Controls Strip
    ctrl_overlay = frame.copy()
    cv2.rectangle(ctrl_overlay, (0, h - 30), (w, h), (12, 12, 16), -1)
    frame = cv2.addWeighted(ctrl_overlay, 0.80, frame, 0.20, 0)
    legend = "[1-0]: Filters | [V]: Vibe Text | [E]: Emoji Style | [S]: Soundboard | [H]: Heavy Bass | [+/-]: Sens | [SPACE]: Save | [Q]: Quit"
    cv2.putText(frame, legend, (12, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (215, 215, 220), 1, cv2.LINE_AA)

    return frame


def main():
    print("=" * 60)
    print("Real-Time Webcam Emotion & Meme Filter (Multi-Face & Vibe Edition)")
    print("=" * 60)

    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)

    state_store = StateStore()
    event_queue = queue.Queue(maxsize=4)

    sound_manager = SoundManager()
    vision_engine = VisionEngine()

    worker = NvidiaMcpWorker(state_store=state_store, event_queue=event_queue)
    worker.start()
    print("[Main] Async NVIDIA NIM Multi-Face Worker online.")

    print("[Main] Initializing 60 FPS Threaded Camera Capture...")
    camera = ThreadedCamera(src=0, width=1280, height=720)
    if not camera.is_opened():
        print("[Main] Error: Could not open camera.")
        worker.stop()
        return

    camera.start()
    print("[Main] Camera thread online. Multi-face zero-lag reactions active!")

    fps_tracker = deque(maxlen=20)
    prev_time = time.time()
    toast_msg = None
    toast_timer = 0.0

    try:
        while True:
            ret, frame = camera.read()
            if not ret or frame is None:
                time.sleep(0.005)
                continue

            curr_time = time.time()
            dt = curr_time - prev_time
            prev_time = curr_time
            if dt > 0:
                fps_tracker.append(1.0 / dt)
                state_store.update_fps(sum(fps_tracker) / len(fps_tracker))

            frame = cv2.flip(frame, 1)

            # 1. Vision Engine: Run Multi-Face Detection & Dual-Engine Classifier
            timestamp_ms = int(curr_time * 1000)
            detected_faces = vision_engine.process_frame(frame, timestamp_ms)

            # Prune idle faces not present in current frame
            state_store.prune_old_faces(current_face_count=len(detected_faces))

            # 2. Process Reactions Separately for Each Face
            for landmark_data, raw_scores in detected_faces:
                if not landmark_data.get("has_face"):
                    continue

                face_id = landmark_data["face_id"]
                face_state = state_store.get_face(face_id)

                # Individual Face Calibration & Debouncing
                if face_state.calibrator.is_calibrating:
                    face_state.calibrator.add_frame(raw_scores)
                    adjusted_scores = raw_scores
                    active_emotion = EmotionState.PEACE
                    conf = 1.0
                    alpha = 0.0
                    transitioned = False
                else:
                    adjusted_scores = face_state.calibrator.apply_offset(raw_scores)
                    active_emotion, conf, alpha, transitioned = face_state.debouncer.update(adjusted_scores)

                face_state.current_emotion = active_emotion
                face_state.confidence = conf
                face_state.overlay_alpha = alpha
                face_state.scores = adjusted_scores
                face_state.expression_summary = landmark_data.get("expression_summary", f"{active_emotion.value} (100%)")
                face_state.expression_details = landmark_data.get("expression_details", ["Face: Normal"])

                # If face transitioned to non-peace emotion, queue separate AI reaction
                if transitioned and active_emotion != EmotionState.PEACE:
                    try:
                        event_queue.put_nowait({
                            "face_id": face_id,
                            "emotion": active_emotion,
                            "confidence": conf
                        })
                    except queue.Full:
                        pass

            # Audio cue playback
            cue = state_store.pop_audio_cue()
            if cue and not state_store.is_muted:
                sound_manager.play(cue)

            # 3. Compositing & Rendering
            snapshot = state_store.get_render_snapshot()
            emoji_style = snapshot["emoji_style"]
            vibe_mode = snapshot["vibe_mode"]
            faces_snapshot = snapshot["faces"]

            # Apply Pro Detailed Visual Shader
            rendered = vision_engine.apply_shader(frame, snapshot["filter_mode"])

            # Render Per-Face Overlays (Separate reactions on each face, ZERO glitter on face)
            for landmark_data, _ in detected_faces:
                if not landmark_data.get("has_face"):
                    continue

                fid = landmark_data["face_id"]
                finfo = faces_snapshot.get(fid)
                if not finfo:
                    continue

                f_alpha = finfo["overlay_alpha"]
                f_emo = finfo["emotion"]

                # Overlay 3D Head-Pose Anchored AR Sticker / 3D Emoji on this face
                if f_alpha > 0.01:
                    if emoji_style == EmojiStyle.MEME_STICKER and finfo.get("sticker_path"):
                        rendered = vision_engine.overlay_sticker(
                            rendered,
                            finfo["sticker_path"],
                            landmark_data,
                            alpha_mult=f_alpha
                        )
                    elif emoji_style in (EmojiStyle.EMOJI_3D, EmojiStyle.COMBO) and finfo.get("emoji_path"):
                        rendered = vision_engine.overlay_sticker(
                            rendered,
                            finfo["emoji_path"],
                            landmark_data,
                            alpha_mult=f_alpha
                        )

                # "What We Feel By Seeing Your Face" - Floating Facial Vibe & Detailed Expression Card
                vibe_text = finfo.get("vibe_insight") or f"VIBE: {f_emo.value} energy"
                rendered = vision_engine.render_facial_vibe_card(
                    rendered,
                    landmark_data,
                    face_id=fid,
                    emotion=f_emo,
                    vibe_text=vibe_text,
                    display_mode=vibe_mode,
                    expression_summary=finfo.get("expression_summary"),
                    expression_details=finfo.get("expression_details")
                )

            # Render particle stream only if explicitly set to PARTICLES_ONLY
            if emoji_style == EmojiStyle.PARTICLES_ONLY:
                rendered = vision_engine.particle_engine.render(rendered)

            # Primary Face Meme Typography
            if snapshot["top_caption"] or snapshot["bottom_caption"]:
                rendered = vision_engine.render_meme_text(
                    rendered,
                    snapshot["top_caption"],
                    snapshot["bottom_caption"],
                    alpha=snapshot["overlay_alpha"]
                )

            # Shutter flash
            if snapshot["flash_active"]:
                white_flash = np.full_like(rendered, 255)
                rendered = cv2.addWeighted(white_flash, 0.75, rendered, 0.25, 0)
                state_store.tick_flash()

            # Toast timer
            if toast_msg and time.time() > toast_timer:
                toast_msg = None

            # Draw Diagnostic HUD
            num_tracked = len([f for f, _ in detected_faces if f.get("has_face")])
            rendered_with_hud = draw_hud(rendered, snapshot, num_tracked, toast_msg)

            # Display window
            cv2.imshow("Real-Time Webcam Emotion & Meme Filter (Multi-Face & Vibe)", rendered_with_hud)

            # 4. Handle Keyboard Controls
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q'), 27):
                break
            elif key in (ord('v'), ord('V')):
                new_vibe = state_store.cycle_vibe_mode()
                toast_msg = f"VIBE INSIGHT: {new_vibe.value}"
                toast_timer = time.time() + 1.2
            elif key in (ord('h'), ord('H')):
                # Heavy Bass Drop
                heavy_name = sound_manager.play_heavy_bass()
                toast_msg = f"HEAVY SOUND: {heavy_name}"
                toast_timer = time.time() + 1.2
            elif key in (ord('s'), ord('S')):
                # Soundboard Cue
                sb_name = sound_manager.play_next_soundboard()
                toast_msg = f"SFX: {sb_name}"
                toast_timer = time.time() + 1.2
            elif key in (ord('e'), ord('E')):
                new_style = state_store.cycle_emoji_style()
                toast_msg = f"EMOJI STYLE: {new_style.value}"
                toast_timer = time.time() + 1.2
            elif key in (ord('c'), ord('C')):
                print("[Main] Recalibrating resting facial baseline across all faces...")
                for fs in state_store.faces.values():
                    fs.calibrator.start_calibration()
                toast_msg = "RECALIBRATING FACES..."
                toast_timer = time.time() + 1.2
            elif key in (ord('m'), ord('M')):
                is_muted = state_store.toggle_mute()
                toast_msg = "SFX MUTED" if is_muted else "SFX UNMUTED"
                toast_timer = time.time() + 1.0
            elif key in (ord('+'), ord('=')):
                for fs in state_store.faces.values():
                    fs.debouncer.adjust_sensitivity(+0.2)
                toast_msg = f"SENSITIVITY: {state_store.faces[0].debouncer.sensitivity:.1f}x"
                toast_timer = time.time() + 1.0
            elif key in (ord('-'), ord('_')):
                for fs in state_store.faces.values():
                    fs.debouncer.adjust_sensitivity(-0.2)
                toast_msg = f"SENSITIVITY: {state_store.faces[0].debouncer.sensitivity:.1f}x"
                toast_timer = time.time() + 1.0
            elif key == ord(' '):  # SPACE: Save Snapshot
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                export_path = EXPORTS_DIR / f"meme_{timestamp}.png"
                cv2.imwrite(str(export_path), rendered)
                state_store.trigger_flash(frames=3)
                sound_manager.play("shutter.wav")
                toast_msg = "MEME SAVED!"
                toast_timer = time.time() + 1.2
            # Hotkeys for Shaders
            elif key == ord('1'):
                state_store.set_filter_mode("normal")
            elif key == ord('2'):
                state_store.set_filter_mode("cartoon_sketch")
            elif key == ord('3'):
                state_store.set_filter_mode("pencil_sketch")
            elif key == ord('4'):
                state_store.set_filter_mode("cyberpunk_neon")
            elif key == ord('5'):
                state_store.set_filter_mode("deep_fried")
            elif key == ord('6'):
                state_store.set_filter_mode("golden_aura")
            elif key == ord('7'):
                state_store.set_filter_mode("noir")
            elif key == ord('8'):
                state_store.set_filter_mode("eerie_vignette")
            elif key == ord('9'):
                state_store.set_filter_mode("toxic_matrix")
            elif key == ord('0'):
                state_store.set_filter_mode("zen_mist")

    except KeyboardInterrupt:
        print("[Main] Terminated by user.")
    finally:
        print("[Main] Cleaning up camera and threads...")
        camera.stop()
        cv2.destroyAllWindows()
        worker.stop()
        if PYGAME_AVAILABLE:
            try:
                pygame.mixer.quit()
            except Exception:
                pass
        print("[Main] Exited cleanly.")


if __name__ == "__main__":
    main()
