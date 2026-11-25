"""
State Manager: Multi-Face Tracking, Thread-Safe StateStore, Baseline Calibration,
Temporal Hysteresis Debouncer, Facial Vibe / Feelings Engine, and Soundboard.
Supports simultaneous multi-face analysis with separate reactions and insights.
"""

import time
import threading
from enum import Enum
from collections import deque
from typing import Dict, Any, Optional, Tuple, List


class EmotionState(str, Enum):
    PEACE = "PEACE"
    LOVE = "LOVE"
    LAUGHTER = "LAUGHTER"
    SORROW = "SORROW"
    ANGER = "ANGER"
    COURAGE = "COURAGE"
    FEAR = "FEAR"
    DISGUST = "DISGUST"
    WONDER = "WONDER"


class EmojiStyle(str, Enum):
    COMBO = "COMBO"                  # 3D Emoji + Floating Particles
    EMOJI_3D = "EMOJI_3D"            # 3D viral emojis (😂, 😍, 😈, 🗿, 😭, 🤯)
    MEME_STICKER = "MEME_STICKER"    # Classic themed sticker (Aura, Lotus, Cloud)
    PARTICLES_ONLY = "PARTICLES"     # Floating animated particle field


class VibeDisplayMode(str, Enum):
    CARD = "CARD"        # Detailed floating frosted glass card with pin
    COMPACT = "COMPACT"  # Single-line floating pill
    OFF = "OFF"          # Hidden


class BaselineCalibrator:
    """Records resting blendshapes & geometric offsets to eliminate individual resting bias."""
    def __init__(self, target_frames: int = 15, auto_start: bool = False):
        self.target_frames = target_frames
        self.collected_frames: List[Dict[EmotionState, float]] = []
        self.is_calibrating = auto_start
        self.offsets: Dict[EmotionState, float] = {}
        self._lock = threading.Lock()

    def start_calibration(self):
        with self._lock:
            self.collected_frames.clear()
            self.is_calibrating = True

    def cancel_or_complete(self):
        with self._lock:
            self.is_calibrating = False

    def add_frame(self, raw_scores: Dict[EmotionState, float]) -> bool:
        with self._lock:
            if not self.is_calibrating:
                return True

            self.collected_frames.append(raw_scores.copy())
            if len(self.collected_frames) >= self.target_frames:
                new_offsets = {}
                for key in raw_scores.keys():
                    vals = [f.get(key, 0.0) for f in self.collected_frames]
                    new_offsets[key] = sum(vals) / len(vals)
                self.offsets = new_offsets
                self.is_calibrating = False
                return True
            return False

    def get_progress(self) -> float:
        with self._lock:
            if not self.is_calibrating:
                return 1.0
            return len(self.collected_frames) / float(self.target_frames)

    def apply_offset(self, raw_scores: Dict[EmotionState, float]) -> Dict[EmotionState, float]:
        with self._lock:
            if self.is_calibrating or not self.offsets:
                return raw_scores

            adjusted = {}
            for k, v in raw_scores.items():
                if k == EmotionState.PEACE:
                    adjusted[k] = v
                    continue
                offset = self.offsets.get(k, 0.0)
                diff = max(0.0, v - offset * 0.35)
                adjusted[k] = float(min(1.0, diff * 1.15))
            return adjusted


class TemporalHysteresisDebouncer:
    """Fast, responsive hysteresis debouncer with rapid state transitions."""
    def __init__(
        self,
        window_size: int = 3,
        entry_threshold: float = 0.22,
        entry_min_hits: int = 1,
        hold_duration_sec: float = 0.35,
        exit_threshold: float = 0.16,
        exit_frames_required: int = 2,
        fade_frames: int = 4
    ):
        self.window_size = window_size
        self.entry_threshold = entry_threshold
        self.entry_min_hits = entry_min_hits
        self.hold_duration_sec = hold_duration_sec
        self.exit_threshold = exit_threshold
        self.exit_frames_required = exit_frames_required
        self.fade_frames = fade_frames

        self.history: deque = deque(maxlen=window_size)
        self.current_locked_emotion: EmotionState = EmotionState.PEACE
        self.current_confidence: float = 0.0
        self.lock_until_ts: float = 0.0

        self.low_confidence_counter: int = 0
        self.overlay_alpha: float = 1.0
        self.is_fading_out: bool = False
        self.fade_step: int = 0
        self.sensitivity: float = 1.0

    def adjust_sensitivity(self, delta: float):
        self.sensitivity = max(0.5, min(2.5, self.sensitivity + delta))
        self.entry_threshold = max(0.12, 0.22 / self.sensitivity)

    def update(self, scores: Dict[EmotionState, float]) -> Tuple[EmotionState, float, float, bool]:
        now = time.time()

        best_cand = EmotionState.PEACE
        best_cand_score = 0.0
        for emo, sc in scores.items():
            if emo != EmotionState.PEACE and sc > best_cand_score:
                best_cand_score = sc
                best_cand = emo

        self.history.append((best_cand, best_cand_score))
        state_changed = False

        if now < self.lock_until_ts:
            self.overlay_alpha = 1.0
            self.is_fading_out = False
            self.fade_step = 0
            self.current_confidence = scores.get(self.current_locked_emotion, best_cand_score)
            return self.current_locked_emotion, self.current_confidence, self.overlay_alpha, False

        effective_threshold = self.entry_threshold
        emotion_hits: Dict[EmotionState, int] = {}
        for cand, sc in self.history:
            if sc >= effective_threshold and cand != EmotionState.PEACE:
                emotion_hits[cand] = emotion_hits.get(cand, 0) + 1

        qualified_emotion = None
        for emo, count in emotion_hits.items():
            if count >= self.entry_min_hits:
                qualified_emotion = emo
                break

        if qualified_emotion and qualified_emotion != self.current_locked_emotion:
            self.current_locked_emotion = qualified_emotion
            self.current_confidence = scores.get(qualified_emotion, best_cand_score)
            self.lock_until_ts = now + self.hold_duration_sec
            self.overlay_alpha = 1.0
            self.is_fading_out = False
            self.fade_step = 0
            self.low_confidence_counter = 0
            state_changed = True
            return self.current_locked_emotion, self.current_confidence, self.overlay_alpha, state_changed

        if self.current_locked_emotion != EmotionState.PEACE:
            curr_score = scores.get(self.current_locked_emotion, 0.0)
            if curr_score < self.exit_threshold:
                self.low_confidence_counter += 1
            else:
                self.low_confidence_counter = 0
                self.is_fading_out = False
                self.overlay_alpha = 1.0

            if self.low_confidence_counter >= self.exit_frames_required:
                self.is_fading_out = True

            if self.is_fading_out:
                self.fade_step += 1
                self.overlay_alpha = max(0.0, 1.0 - (self.fade_step / float(self.fade_frames)))
                if self.fade_step >= self.fade_frames:
                    self.current_locked_emotion = EmotionState.PEACE
                    self.current_confidence = scores.get(EmotionState.PEACE, 1.0)
                    self.overlay_alpha = 0.0
                    self.is_fading_out = False
                    self.fade_step = 0
                    self.low_confidence_counter = 0
                    state_changed = True
            else:
                self.overlay_alpha = 1.0
                self.current_confidence = curr_score
        else:
            self.current_confidence = scores.get(EmotionState.PEACE, 1.0)
            self.overlay_alpha = 0.0

        return self.current_locked_emotion, self.current_confidence, self.overlay_alpha, state_changed


class FaceState:
    """Maintains individual emotion analysis, reaction stickers, and vibe insight for a single face."""
    def __init__(self, face_id: int):
        self.face_id = face_id
        self.current_emotion: EmotionState = EmotionState.PEACE
        self.confidence: float = 1.0
        self.active_sticker_path: str = "assets/memes/peace_lotus.png"
        self.active_emoji_path: str = "assets/memes/emoji_zen_angel.png"
        self.vibe_insight: str = "VIBE: Inner peace mode"
        self.expression_summary: str = "NEUTRAL / RELAXED"
        self.expression_details: List[str] = ["Face: Relaxed", "Brows: Neutral"]
        self.top_caption: str = ""
        self.bottom_caption: str = ""
        self.overlay_alpha: float = 0.0
        self.calibrator = BaselineCalibrator(target_frames=15, auto_start=False)
        self.debouncer = TemporalHysteresisDebouncer()
        self.last_seen_time: float = time.time()
        self.scores: Dict[EmotionState, float] = {}

    def update_reaction(
        self,
        emotion: EmotionState,
        sticker_path: Optional[str] = None,
        emoji_path: Optional[str] = None,
        vibe_insight: Optional[str] = None,
        top_caption: Optional[str] = None,
        bottom_caption: Optional[str] = None,
        expression_summary: Optional[str] = None,
        expression_details: Optional[List[str]] = None
    ):
        self.current_emotion = emotion
        if sticker_path:
            self.active_sticker_path = sticker_path
        if emoji_path:
            self.active_emoji_path = emoji_path
        if vibe_insight:
            self.vibe_insight = vibe_insight
        if expression_summary:
            self.expression_summary = expression_summary
        if expression_details is not None:
            self.expression_details = expression_details
        if top_caption is not None:
            self.top_caption = top_caption
        if bottom_caption is not None:
            self.bottom_caption = bottom_caption


class StateStore:
    """Thread-safe state container managing multi-face states, shaders, audio cues, and vibe display."""
    def __init__(self):
        self._lock = threading.RLock()

        # Multi-face registry
        self.faces: Dict[int, FaceState] = {0: FaceState(0)}

        # Global rendering options (EMOJI_3D clean style by default, no face glitter)
        self.emoji_style: EmojiStyle = EmojiStyle.EMOJI_3D
        self.vibe_mode: VibeDisplayMode = VibeDisplayMode.CARD
        self.filter_mode: str = "normal"
        self.is_muted: bool = False

        # Memory & telemetry
        self.recent_captions: deque = deque(maxlen=8)
        self.pending_audio_cue: Optional[str] = None
        self.fps: float = 0.0
        self.nim_status: str = "INITIALIZING"
        self.flash_countdown: int = 0

    def get_face(self, face_id: int) -> FaceState:
        with self._lock:
            if face_id not in self.faces:
                self.faces[face_id] = FaceState(face_id)
            self.faces[face_id].last_seen_time = time.time()
            return self.faces[face_id]

    def prune_old_faces(self, current_face_count: int, max_idle_sec: float = 2.5):
        with self._lock:
            now = time.time()
            dead = [fid for fid, fs in self.faces.items() if fid >= current_face_count and (now - fs.last_seen_time > max_idle_sec)]
            for fid in dead:
                if fid != 0:  # Always retain primary face #0
                    del self.faces[fid]

    def get_render_snapshot(self) -> Dict[str, Any]:
        with self._lock:
            primary_face = self.faces.get(0) or FaceState(0)
            faces_snapshot = {}
            for fid, f in self.faces.items():
                faces_snapshot[fid] = {
                    "face_id": fid,
                    "emotion": f.current_emotion,
                    "confidence": f.confidence,
                    "sticker_path": f.active_sticker_path,
                    "emoji_path": f.active_emoji_path,
                    "vibe_insight": f.vibe_insight,
                    "expression_summary": f.expression_summary,
                    "expression_details": list(f.expression_details),
                    "top_caption": f.top_caption,
                    "bottom_caption": f.bottom_caption,
                    "overlay_alpha": f.overlay_alpha,
                    "scores": f.scores.copy(),
                    "is_calibrating": f.calibrator.is_calibrating,
                    "calib_progress": f.calibrator.get_progress()
                }

            return {
                "faces": faces_snapshot,
                "current_emotion": primary_face.current_emotion,
                "confidence": primary_face.confidence,
                "active_sticker_path": primary_face.active_sticker_path,
                "active_emoji_path": primary_face.active_emoji_path,
                "vibe_insight": primary_face.vibe_insight,
                "expression_summary": primary_face.expression_summary,
                "expression_details": list(primary_face.expression_details),
                "top_caption": primary_face.top_caption,
                "bottom_caption": primary_face.bottom_caption,
                "overlay_alpha": primary_face.overlay_alpha,
                "emoji_style": self.emoji_style,
                "vibe_mode": self.vibe_mode,
                "filter_mode": self.filter_mode,
                "is_muted": self.is_muted,
                "fps": self.fps,
                "nim_status": self.nim_status,
                "flash_active": self.flash_countdown > 0,
                "sensitivity": primary_face.debouncer.sensitivity,
                "entry_threshold": primary_face.debouncer.entry_threshold,
                "is_calibrating": primary_face.calibrator.is_calibrating,
                "calib_progress": primary_face.calibrator.get_progress()
            }

    def cycle_emoji_style(self) -> EmojiStyle:
        with self._lock:
            styles = [EmojiStyle.COMBO, EmojiStyle.EMOJI_3D, EmojiStyle.MEME_STICKER, EmojiStyle.PARTICLES_ONLY]
            idx = (styles.index(self.emoji_style) + 1) % len(styles)
            self.emoji_style = styles[idx]
            return self.emoji_style

    def cycle_vibe_mode(self) -> VibeDisplayMode:
        with self._lock:
            modes = [VibeDisplayMode.CARD, VibeDisplayMode.COMPACT, VibeDisplayMode.OFF]
            idx = (modes.index(self.vibe_mode) + 1) % len(modes)
            self.vibe_mode = modes[idx]
            return self.vibe_mode

    def update_fps(self, fps: float):
        with self._lock:
            self.fps = fps

    def trigger_flash(self, frames: int = 3):
        with self._lock:
            self.flash_countdown = frames

    def tick_flash(self):
        with self._lock:
            if self.flash_countdown > 0:
                self.flash_countdown -= 1

    def toggle_mute(self) -> bool:
        with self._lock:
            self.is_muted = not self.is_muted
            return self.is_muted

    def set_filter_mode(self, mode: str):
        with self._lock:
            self.filter_mode = mode

    def set_nim_status(self, status: str):
        with self._lock:
            self.nim_status = status

    def trigger_audio_cue(self, sound_path: str):
        with self._lock:
            if not self.is_muted:
                self.pending_audio_cue = sound_path

    def pop_audio_cue(self) -> Optional[str]:
        with self._lock:
            cue = self.pending_audio_cue
            self.pending_audio_cue = None
            return cue

    def update_ai_bundle(
        self,
        emotion: EmotionState,
        top_text: str,
        bottom_text: str,
        face_id: int = 0,
        sticker_path: Optional[str] = None,
        emoji_path: Optional[str] = None,
        sound_path: Optional[str] = None,
        filter_mode: Optional[str] = None,
        vibe_insight: Optional[str] = None
    ):
        with self._lock:
            face = self.get_face(face_id)
            face.update_reaction(
                emotion=emotion,
                sticker_path=sticker_path,
                emoji_path=emoji_path,
                vibe_insight=vibe_insight,
                top_caption=top_text.strip().upper(),
                bottom_caption=bottom_text.strip().upper()
            )
            if filter_mode:
                self.filter_mode = filter_mode
            if sound_path and not self.is_muted:
                self.pending_audio_cue = sound_path

            caption_key = f"F{face_id}: {face.top_caption} | {face.bottom_caption}"
            if caption_key not in self.recent_captions:
                self.recent_captions.append(caption_key)

    def get_recent_history(self) -> List[str]:
        with self._lock:
            return list(self.recent_captions)
