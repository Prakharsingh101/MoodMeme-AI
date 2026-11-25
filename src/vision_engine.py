"""
Vision Engine: Multi-Face MediaPipe Tasks FaceLandmarker (num_faces=4),
High-Accuracy Dual-Engine Geometric & Blendshape Emotion Classifier,
Pro Comic & Graphite Shaders, and Floating Facial Vibe Cards ("What We Feel By Seeing Your Face").
"""

import os
import math
import random
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

# pyrefly: ignore[missing-import]
import cv2
# pyrefly: ignore[missing-import]
import numpy as np
# pyrefly: ignore[missing-import]
from PIL import Image, ImageDraw, ImageFont

from src.state_manager import EmotionState, EmojiStyle, VibeDisplayMode


class Particle:
    __slots__ = ('x', 'y', 'vx', 'vy', 'size', 'life', 'max_life', 'color', 'shape', 'seed')

    def __init__(self, x: float, y: float, vx: float, vy: float, size: int, max_life: int, color: Tuple[int, int, int], shape: str):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.size = size
        self.life = max_life
        self.max_life = max_life
        self.color = color
        self.shape = shape
        self.seed = random.uniform(0, 100)

    def update(self) -> bool:
        self.x += self.vx + math.sin(self.life * 0.15 + self.seed) * 0.8
        self.y += self.vy
        self.life -= 1
        return self.life > 0


class ParticleEngine:
    def __init__(self, max_particles: int = 60):
        self.particles: List[Particle] = []
        self.max_particles = max_particles

    def emit_emotion_particles(self, emotion: EmotionState, center_pt: Tuple[int, int], frame_shape: Tuple[int, int]):
        if len(self.particles) >= self.max_particles:
            return

        cx, cy = center_pt
        num_to_spawn = random.randint(2, 4)

        if emotion == EmotionState.LOVE:
            for _ in range(num_to_spawn):
                self.particles.append(Particle(
                    x=cx + random.randint(-180, 180),
                    y=max(10.0, cy - random.randint(120, 220)),
                    vx=random.uniform(-0.6, 0.6),
                    vy=random.uniform(-2.6, -1.2),
                    size=random.randint(14, 24),
                    max_life=random.randint(25, 45),
                    color=(255, 60, 140),
                    shape="heart"
                ))
        elif emotion == EmotionState.LAUGHTER:
            for _ in range(num_to_spawn):
                self.particles.append(Particle(
                    x=cx + random.choice([-120, 120]) + random.randint(-25, 25),
                    y=max(10.0, cy - random.randint(100, 200)),
                    vx=random.uniform(-1.5, 1.5),
                    vy=random.uniform(-1.0, 2.0),
                    size=random.randint(10, 20),
                    max_life=random.randint(20, 35),
                    color=(40, 210, 255),
                    shape="tear"
                ))
        elif emotion == EmotionState.ANGER:
            for _ in range(num_to_spawn):
                self.particles.append(Particle(
                    x=cx + random.randint(-160, 160),
                    y=max(10.0, cy - random.randint(120, 220)),
                    vx=random.uniform(-0.8, 0.8),
                    vy=random.uniform(-3.5, -1.8),
                    size=random.randint(12, 22),
                    max_life=random.randint(20, 40),
                    color=(20, 80, 255),
                    shape="ember"
                ))
        elif emotion == EmotionState.WONDER:
            for _ in range(num_to_spawn):
                self.particles.append(Particle(
                    x=cx + random.randint(-200, 200),
                    y=max(10.0, cy - random.randint(130, 240)),
                    vx=random.uniform(-1.0, 1.0),
                    vy=random.uniform(-1.5, 1.5),
                    size=random.randint(12, 24),
                    max_life=random.randint(25, 50),
                    color=(255, 230, 60),
                    shape="star"
                ))
        elif emotion == EmotionState.COURAGE:
            for _ in range(num_to_spawn):
                self.particles.append(Particle(
                    x=cx + random.randint(-160, 160),
                    y=max(10.0, cy - random.randint(120, 220)),
                    vx=random.uniform(-0.5, 0.5),
                    vy=random.uniform(-3.0, -1.5),
                    size=random.randint(10, 20),
                    max_life=random.randint(20, 40),
                    color=(40, 210, 255),
                    shape="spark"
                ))
        # Note: No particles in PEACE state to ensure clean, glitter-free visual feed.

    def render(self, frame: np.ndarray) -> np.ndarray:
        if not self.particles:
            return frame

        h, w = frame.shape[:2]
        alive = []
        for p in self.particles:
            if not p.update():
                continue
            alive.append(p)

            px, py = int(p.x), int(p.y)
            if not (0 <= px < w and 0 <= py < h):
                continue

            sz = p.size
            col = p.color

            if p.shape == "heart":
                cv2.circle(frame, (px - sz // 4, py - sz // 4), sz // 4, col, -1)
                cv2.circle(frame, (px + sz // 4, py - sz // 4), sz // 4, col, -1)
                poly = np.array([(px - sz // 2, py - sz // 4), (px + sz // 2, py - sz // 4), (px, py + sz // 2)], dtype=np.int32)
                cv2.fillConvexPoly(frame, poly, col)
            elif p.shape == "star":
                cv2.line(frame, (px - sz, py), (px + sz, py), col, 2)
                cv2.line(frame, (px, py - sz), (px, py + sz), col, 2)
                cv2.circle(frame, (px, py), max(2, sz // 4), (255, 255, 255), -1)
            elif p.shape == "tear":
                cv2.circle(frame, (px, py), sz // 3, col, -1)
                poly = np.array([(px - sz // 3, py), (px + sz // 3, py), (px, py - sz // 2)], dtype=np.int32)
                cv2.fillConvexPoly(frame, poly, col)
            else:
                cv2.circle(frame, (px, py), sz // 3, col, -1)

        self.particles = alive
        return frame


class VisionEngine:
    def __init__(self, model_path: Optional[str] = None):
        if model_path is None:
            root_dir = Path(__file__).resolve().parent.parent
            model_path = str(root_dir / "assets" / "models" / "face_landmarker.task")
        self.model_path = model_path
        self.landmarker = None

        # Multi-Face support (up to 4 simultaneous faces)
        self._init_mediapipe(num_faces=4)

        self.haar_face_cascade = None
        try:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            if os.path.exists(cascade_path):
                self.haar_face_cascade = cv2.CascadeClassifier(cascade_path)
        except Exception:
            pass

        self._sticker_cache: Dict[str, np.ndarray] = {}
        self._font_cache: Dict[int, ImageFont.FreeTypeFont] = {}
        self.particle_engine = ParticleEngine(max_particles=60)

    def _init_mediapipe(self, num_faces: int = 4):
        try:
            # pyrefly: ignore[missing-import]
            import mediapipe as mp
            # pyrefly: ignore[missing-import]
            from mediapipe.tasks import python as mp_python
            # pyrefly: ignore[missing-import]
            from mediapipe.tasks.python import vision

            if not os.path.exists(self.model_path) or os.path.getsize(self.model_path) < 1000000:
                print(f"[VisionEngine] Warning: {self.model_path} missing or placeholder.")
                return

            base_options = mp_python.BaseOptions(model_asset_path=self.model_path)
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.IMAGE,
                output_face_blendshapes=True,
                output_facial_transformation_matrixes=True,
                num_faces=num_faces
            )
            self.landmarker = vision.FaceLandmarker.create_from_options(options)
            print(f"[VisionEngine] MediaPipe Multi-Face Landmarker online (num_faces={num_faces}).")
        except Exception as e:
            print(f"[VisionEngine] MediaPipe init error: {e}")

    def process_frame(self, bgr_frame: np.ndarray, timestamp_ms: int = 0) -> List[Tuple[Dict[str, Any], Dict[EmotionState, float]]]:
        """
        Processes frame for multiple faces simultaneously.
        Returns a list of tuples: [(landmark_data_face_0, scores_0), (landmark_data_face_1, scores_1), ...]
        """
        h, w = bgr_frame.shape[:2]
        empty_face = {
            "has_face": False,
            "face_id": 0,
            "forehead_pt": (w // 2, int(h * 0.25)),
            "left_eye_pt": (int(w * 0.4), int(h * 0.4)),
            "right_eye_pt": (int(w * 0.6), int(h * 0.4)),
            "center_pt": (w // 2, int(h * 0.45)),
            "euler_angles": (0.0, 0.0, 0.0),
            "interocular_dist": float(w * 0.18),
            "blendshape_count": 0,
            "diagnostics": {}
        }
        empty_scores = {e: 0.1 for e in EmotionState}
        empty_scores[EmotionState.PEACE] = 0.85

        if self.landmarker is None:
            faces = self._process_haar_fallback(bgr_frame)
            return [(f, empty_scores) for f in faces] if faces else [(empty_face, empty_scores)]

        # pyrefly: ignore[missing-import]
        import mediapipe as mp
        # Fast 640-scale inference for multi-face speed
        if w > 640:
            scale_factor = 640.0 / w
            scaled_h = int(h * scale_factor)
            small_bgr = cv2.resize(bgr_frame, (640, scaled_h), interpolation=cv2.INTER_LINEAR)
            rgb_frame = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2RGB)
        else:
            scale_factor = 1.0
            rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        result = None
        try:
            result = self.landmarker.detect(mp_image)
        except Exception:
            faces = self._process_haar_fallback(bgr_frame)
            return [(f, empty_scores) for f in faces] if faces else [(empty_face, empty_scores)]

        if not result or not result.face_landmarks or len(result.face_landmarks) == 0:
            faces = self._process_haar_fallback(bgr_frame)
            return [(f, empty_scores) for f in faces] if faces else [(empty_face, empty_scores)]

        detected_faces = []
        for face_idx, landmarks in enumerate(result.face_landmarks):
            forehead = landmarks[10]
            left_eye = landmarks[33]
            right_eye = landmarks[263]
            nose_tip = landmarks[1]

            pt_forehead = (int(forehead.x * w), int(forehead.y * h))
            pt_left_eye = (int(left_eye.x * w), int(left_eye.y * h))
            pt_right_eye = (int(right_eye.x * w), int(right_eye.y * h))
            pt_center = (int(nose_tip.x * w), int(nose_tip.y * h))

            iod = math.dist(pt_left_eye, pt_right_eye)
            iod = max(25.0, iod)

            # 3D Euler Angles
            pitch, yaw, roll = 0.0, 0.0, 0.0
            if result.facial_transformation_matrixes and face_idx < len(result.facial_transformation_matrixes):
                mat = np.array(result.facial_transformation_matrixes[face_idx])
                r00, r01 = mat[0, 0], mat[0, 1]
                r10 = mat[1, 0]
                r20, r21, r22 = mat[2, 0], mat[2, 1], mat[2, 2]

                roll = math.degrees(math.atan2(r10, r00))
                pitch = math.degrees(math.atan2(-r20, math.sqrt(r21**2 + r22**2)))
                yaw = math.degrees(math.atan2(r21, r22))

            # Blendshapes & 3D Geometric Landmark Ratios (Dual-Engine Fusion for extreme accuracy)
            bs_dict = {}
            if result.face_blendshapes and face_idx < len(result.face_blendshapes):
                bs_dict = {cat.category_name: cat.score for cat in result.face_blendshapes[face_idx]}

            raw_scores, diagnostics, expr_summary, expr_details = self._compute_dual_engine_heuristics(bs_dict, landmarks, pitch, w, h)

            face_data = {
                "has_face": True,
                "face_id": face_idx,
                "forehead_pt": pt_forehead,
                "left_eye_pt": pt_left_eye,
                "right_eye_pt": pt_right_eye,
                "center_pt": pt_center,
                "euler_angles": (pitch, yaw, roll),
                "interocular_dist": iod,
                "blendshape_count": len(bs_dict),
                "diagnostics": diagnostics,
                "expression_summary": expr_summary,
                "expression_details": expr_details
            }
            detected_faces.append((face_data, raw_scores))

        return detected_faces

    def _process_haar_fallback(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        if self.haar_face_cascade is None:
            return []

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.haar_face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=4, minSize=(60, 60))
        results = []
        for idx, (fx, fy, fw, fh) in enumerate(faces):
            results.append({
                "has_face": True,
                "face_id": idx,
                "forehead_pt": (int(fx + fw / 2), int(fy + fh * 0.15)),
                "left_eye_pt": (int(fx + fw * 0.3), int(fy + fh * 0.35)),
                "right_eye_pt": (int(fx + fw * 0.7), int(fy + fh * 0.35)),
                "center_pt": (int(fx + fw / 2), int(fy + fh * 0.5)),
                "euler_angles": (0.0, 0.0, 0.0),
                "interocular_dist": float(fw * 0.4),
                "blendshape_count": 0,
                "diagnostics": {},
                "expression_summary": "FACE DETECTED (HAAR)",
                "expression_details": ["Face: Present"]
            })
        return results

    def _compute_dual_engine_heuristics(
        self,
        bs: Dict[str, float],
        landmarks,
        pitch: float,
        w: int,
        h: int
    ) -> Tuple[Dict[EmotionState, float], Dict[str, float], str, List[str]]:
        """
        Direct Dual-Engine Fusion: MediaPipe 52 ARKit Blendshapes + 3D Geometric Landmark Ratios.
        High-sensitivity, zero-lag expression analysis that detects subtle & intense expressions
        with rich human-readable detail.
        """
        get_bs = lambda k: bs.get(k, 0.0)

        # 1. ARKit Blendshape cues
        bs_smile = 0.5 * (get_bs("mouthSmileLeft") + get_bs("mouthSmileRight"))
        bs_jaw_open = get_bs("jawOpen")
        bs_squint = 0.5 * (get_bs("eyeSquintLeft") + get_bs("eyeSquintRight"))
        bs_frown = 0.5 * (get_bs("mouthFrownLeft") + get_bs("mouthFrownRight"))
        bs_brow_down = 0.5 * (get_bs("browDownLeft") + get_bs("browDownRight"))
        bs_brow_in_up = get_bs("browInnerUp")
        bs_brow_out_up = 0.5 * (get_bs("browOuterUpLeft") + get_bs("browOuterUpRight"))
        bs_eye_wide = 0.5 * (get_bs("eyeWideLeft") + get_bs("eyeWideRight"))
        bs_sneer = 0.5 * (get_bs("noseSneerLeft") + get_bs("noseSneerRight"))
        bs_jaw_fwd = get_bs("jawForward")
        bs_pucker = get_bs("mouthPucker")
        bs_upper_up = 0.5 * (get_bs("mouthUpperUpLeft") + get_bs("mouthUpperUpRight"))
        bs_press = 0.5 * (get_bs("mouthPressLeft") + get_bs("mouthPressRight"))
        bs_stretch = 0.5 * (get_bs("mouthStretchLeft") + get_bs("mouthStretchRight"))
        bs_dimple = 0.5 * (get_bs("mouthDimpleLeft") + get_bs("mouthDimpleRight"))
        bs_cheek_squint = 0.5 * (get_bs("cheekSquintLeft") + get_bs("cheekSquintRight"))

        # 2. 3D Geometric Landmark Measurements from 478 points
        def dist_3d(i1, i2):
            p1, p2 = landmarks[i1], landmarks[i2]
            return math.sqrt(((p1.x - p2.x)*w)**2 + ((p1.y - p2.y)*h)**2 + ((p1.z - p2.z)*w)**2)

        # Inter-ocular baseline distance
        eye_base = max(10.0, dist_3d(33, 263))

        # Mouth Aspect Ratio (MAR): Upper inner lip (#13) to Lower inner lip (#14) vs Mouth width (#78 to #308)
        mouth_h = dist_3d(13, 14)
        mouth_w = max(5.0, dist_3d(78, 308))
        mar = mouth_h / mouth_w

        # Eye Aspect Ratio (EAR): Eyelid height vs Eye width
        l_eye_h = dist_3d(159, 145)
        l_eye_w = max(5.0, dist_3d(33, 133))
        r_eye_h = dist_3d(386, 374)
        r_eye_w = max(5.0, dist_3d(362, 263))
        ear = 0.5 * ((l_eye_h / l_eye_w) + (r_eye_h / r_eye_w))

        # Mouth stretch to eye base ratio
        mouth_stretch_ratio = dist_3d(61, 291) / eye_base

        # Brow-to-Eye distance ratio
        bed = 0.5 * (dist_3d(107, 159) + dist_3d(336, 386)) / eye_base

        # 3. Direct Fused Feature Signals (High-Sensitivity)
        smile_signal = max(bs_smile, float(np.clip((mouth_stretch_ratio - 0.84) / 0.28, 0.0, 1.0)))
        jaw_signal = max(bs_jaw_open, float(np.clip((mar - 0.16) / 0.32, 0.0, 1.0)))
        eye_wide_signal = max(bs_eye_wide, float(np.clip((ear - 0.28) / 0.12, 0.0, 1.0)))
        brow_up_signal = max(bs_brow_out_up, bs_brow_in_up, float(np.clip((bed - 0.30) / 0.14, 0.0, 1.0)))
        brow_down_signal = max(bs_brow_down, float(np.clip((0.27 - bed) / 0.10, 0.0, 1.0)))
        frown_signal = max(bs_frown, get_bs("mouthLowerDownLeft"), get_bs("mouthLowerDownRight"), bs_press)
        sneer_signal = max(bs_sneer, bs_upper_up)
        squint_signal = max(bs_squint, bs_cheek_squint)

        # 4. Proportional Emotion Scoring
        # 1. LAUGHTER (Hasya)
        score_laughter = 0.50 * smile_signal + 0.35 * jaw_signal + 0.15 * squint_signal
        if smile_signal > 0.20 and jaw_signal > 0.15:
            score_laughter = min(1.0, score_laughter * 1.45)
        elif smile_signal > 0.35:
            score_laughter = min(1.0, score_laughter * 1.25)

        # 2. LOVE (Shringara)
        score_love = 0.60 * smile_signal + 0.30 * squint_signal + 0.10 * bs_brow_in_up
        if jaw_signal > 0.35:
            score_love *= 0.50

        # 3. WONDER (Adbhuta)
        score_wonder = 0.40 * brow_up_signal + 0.35 * eye_wide_signal + 0.25 * jaw_signal
        if brow_up_signal > 0.20 and (eye_wide_signal > 0.18 or jaw_signal > 0.18):
            score_wonder = min(1.0, score_wonder * 1.40)

        # 4. ANGER (Raudra)
        score_anger = 0.50 * brow_down_signal + 0.30 * sneer_signal + 0.20 * frown_signal
        if brow_down_signal > 0.18 or sneer_signal > 0.25:
            score_anger = min(1.0, score_anger * 1.40)

        # 5. SORROW (Karuna)
        score_sorrow = 0.50 * bs_brow_in_up + 0.35 * frown_signal + 0.15 * bs_press
        if bs_brow_in_up > 0.18 and frown_signal > 0.10:
            score_sorrow = min(1.0, score_sorrow * 1.40)

        # 6. FEAR (Bhayanaka)
        score_fear = 0.55 * eye_wide_signal + 0.25 * bs_brow_in_up + 0.20 * max(bs_stretch, jaw_signal)
        if eye_wide_signal > 0.22 and (bs_brow_in_up > 0.15 or jaw_signal > 0.15):
            score_fear = min(1.0, score_fear * 1.35)

        # 7. DISGUST (Bibhatsa)
        score_disgust = 0.50 * sneer_signal + 0.35 * bs_upper_up + 0.15 * brow_down_signal
        if sneer_signal > 0.18 or bs_upper_up > 0.20:
            score_disgust = min(1.0, score_disgust * 1.40)

        # 8. COURAGE (Vira)
        pitch_bonus = 0.15 if pitch > 3.0 else 0.0
        score_courage = 0.40 * bs_jaw_fwd + 0.30 * brow_down_signal + 0.20 * max(bs_dimple, bs_pucker) + 0.10 * pitch_bonus

        # 9. PEACE (Shanta)
        active_candidates = [
            score_love, score_laughter, score_sorrow, score_anger,
            score_courage, score_fear, score_disgust, score_wonder
        ]
        max_active = max(active_candidates)
        score_peace = float(np.clip(1.0 - (max_active * 1.6), 0.0, 1.0))

        raw_scores = {
            EmotionState.PEACE: float(np.clip(score_peace, 0.0, 1.0)),
            EmotionState.LOVE: float(np.clip(score_love, 0.0, 1.0)),
            EmotionState.LAUGHTER: float(np.clip(score_laughter, 0.0, 1.0)),
            EmotionState.SORROW: float(np.clip(score_sorrow, 0.0, 1.0)),
            EmotionState.ANGER: float(np.clip(score_anger, 0.0, 1.0)),
            EmotionState.COURAGE: float(np.clip(score_courage, 0.0, 1.0)),
            EmotionState.FEAR: float(np.clip(score_fear, 0.0, 1.0)),
            EmotionState.DISGUST: float(np.clip(score_disgust, 0.0, 1.0)),
            EmotionState.WONDER: float(np.clip(score_wonder, 0.0, 1.0)),
        }

        # 5. Build Human-Readable Micro-Expression Cues
        active_cues = []
        if smile_signal > 0.14:
            active_cues.append(f"Smile: {int(smile_signal * 100)}%")
        if jaw_signal > 0.14:
            active_cues.append(f"Mouth Open: {int(jaw_signal * 100)}%")
        if brow_up_signal > 0.14:
            active_cues.append(f"Brows Up: {int(brow_up_signal * 100)}%")
        if brow_down_signal > 0.14:
            active_cues.append(f"Brows Furrowed: {int(brow_down_signal * 100)}%")
        if eye_wide_signal > 0.14:
            active_cues.append(f"Eyes Wide: {int(eye_wide_signal * 100)}%")
        if squint_signal > 0.14:
            active_cues.append(f"Eyes Squint: {int(squint_signal * 100)}%")
        if sneer_signal > 0.14:
            active_cues.append(f"Nose Sneer: {int(sneer_signal * 100)}%")
        if frown_signal > 0.14:
            active_cues.append(f"Frown: {int(frown_signal * 100)}%")
        if bs_pucker > 0.14:
            active_cues.append(f"Pucker: {int(bs_pucker * 100)}%")
        if bs_dimple > 0.14:
            active_cues.append(f"Smirk: {int(bs_dimple * 100)}%")

        if not active_cues:
            active_cues.append("Face: Relaxed / Neutral")

        # Determine dominant expression summary
        best_emo = EmotionState.PEACE
        best_sc = score_peace
        for e, sc in [
            (EmotionState.LAUGHTER, score_laughter),
            (EmotionState.LOVE, score_love),
            (EmotionState.WONDER, score_wonder),
            (EmotionState.ANGER, score_anger),
            (EmotionState.SORROW, score_sorrow),
            (EmotionState.FEAR, score_fear),
            (EmotionState.DISGUST, score_disgust),
            (EmotionState.COURAGE, score_courage),
        ]:
            if sc > 0.20 and sc > best_sc:
                best_sc = sc
                best_emo = e

        EMO_DESCRIPTIONS = {
            EmotionState.LAUGHTER: "LAUGHTER / WIDE SMILE",
            EmotionState.LOVE: "LOVE / GENTLE SMILE",
            EmotionState.WONDER: "SURPRISE / WONDER / AWE",
            EmotionState.ANGER: "ANGER / FURIOUS BROW",
            EmotionState.SORROW: "SORROW / SAD / POUT",
            EmotionState.FEAR: "FEAR / SHOCK / WIDE EYES",
            EmotionState.DISGUST: "DISGUST / NOSE WRINKLE",
            EmotionState.COURAGE: "COURAGE / CONFIDENT SMIRK",
            EmotionState.PEACE: "NEUTRAL / CALM / RELAXED",
        }
        expr_summary = f"{EMO_DESCRIPTIONS.get(best_emo, 'EXPRESSION')} ({int(best_sc * 100)}%)"

        diagnostics = {
            "smile": smile_signal,
            "mar": mar,
            "ear": ear,
            "brow_down": brow_down_signal,
            "brow_in_up": brow_up_signal,
            "sneer": sneer_signal
        }

        return raw_scores, diagnostics, expr_summary, active_cues

    # ---------------------------------------------------------
    # Visual Shaders with Rich Textures and Details
    # ---------------------------------------------------------
    def apply_shader(self, frame: np.ndarray, mode: str) -> np.ndarray:
        h, w = frame.shape[:2]

        if mode == "cartoon_sketch":
            # HIGH-DETAIL COMIC MANGA SHADER:
            small = cv2.resize(frame, (w // 2, h // 2), interpolation=cv2.INTER_LINEAR)
            smooth = cv2.edgePreservingFilter(small, flags=1, sigma_s=45, sigma_r=0.40)
            quant = (smooth // 32) * 32 + 16
            color = cv2.resize(quant, (w, h), interpolation=cv2.INTER_LINEAR)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_blur = cv2.medianBlur(gray, 7)
            edges = cv2.adaptiveThreshold(
                gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blockSize=9, C=2
            )
            # Add fine Laplace contour lines for extra manga detail
            lap = cv2.Laplacian(gray_blur, cv2.CV_8U, ksize=3)
            lap_inv = 255 - cv2.threshold(lap, 35, 255, cv2.THRESH_BINARY)[1]
            edges_combined = cv2.bitwise_and(edges, lap_inv)
            edges_bgr = cv2.cvtColor(edges_combined, cv2.COLOR_GRAY2BGR)

            return cv2.bitwise_and(color, edges_bgr)

        elif mode == "pencil_sketch":
            # HYPER-DETAILED GRAPHITE CROSS-HATCH SHADER:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            inv = 255 - gray
            blur = cv2.GaussianBlur(inv, (21, 21), 0)
            sketch = cv2.divide(gray, 255 - blur, scale=256)
            sketch = cv2.equalizeHist(sketch)
            # Fine texture overlay
            fine_edges = 255 - cv2.Canny(gray, 40, 120)
            sketch = cv2.bitwise_and(sketch, fine_edges)
            return cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)

        elif mode == "cyberpunk_neon":
            # CYBERPUNK NEON GLITCH SHADER:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            neon = np.zeros_like(frame)
            neon[:, :, 0] = edges  # Blue/Cyan
            neon[:, :, 1] = cv2.bitwise_and(edges, gray)
            neon[:, :, 2] = np.roll(edges, 4, axis=1)  # Red/Magenta chromatic shift
            # Scanlines
            neon[::3, :, :] = (neon[::3, :, :] * 0.4).astype(np.uint8)
            return cv2.addWeighted(frame, 0.45, neon, 0.75, 0)

        elif mode == "deep_fried":
            # INTENSE BASS-BOOSTED DEEP FRIED SHADER:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 2.0, 0, 255)
            hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.3, 0, 255)
            boosted = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
            boosted[:, :, 2] = np.clip(boosted[:, :, 2].astype(np.int32) + 60, 0, 255).astype(np.uint8)
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            fried = cv2.filter2D(boosted, -1, kernel)
            return fried

        elif mode == "golden_aura":
            tint = np.array([0.70, 1.15, 1.35])
            golden = np.clip(frame.astype(np.float32) * tint, 0, 255).astype(np.uint8)
            small = cv2.resize(golden, (w // 4, h // 4), interpolation=cv2.INTER_LINEAR)
            blur_small = cv2.GaussianBlur(small, (15, 15), 0)
            blur = cv2.resize(blur_small, (w, h), interpolation=cv2.INTER_LINEAR)
            return cv2.addWeighted(golden, 0.72, blur, 0.35, 0)

        elif mode == "noir":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            contrast = clahe.apply(gray)
            bgr_gray = cv2.cvtColor(contrast, cv2.COLOR_GRAY2BGR)
            X = cv2.getGaussianKernel(w, w / 2)
            Y = cv2.getGaussianKernel(h, h / 2)
            kernel = Y * X.T
            mask = 255 * kernel / np.linalg.norm(kernel)
            mask = np.clip(mask * 1.6, 0, 1.0)
            return (bgr_gray * mask[:, :, np.newaxis]).astype(np.uint8)

        elif mode == "eerie_vignette":
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 1] = hsv[:, :, 1] * 0.35
            desat = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
            desat[:, :, 0] = np.clip(desat[:, :, 0].astype(np.int32) + 45, 0, 255).astype(np.uint8)
            return desat

        elif mode == "toxic_matrix":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            toxic = np.zeros_like(frame)
            toxic[:, :, 1] = np.clip(gray.astype(np.int32) * 1.35, 0, 255).astype(np.uint8)
            toxic[:, :, 0] = (gray * 0.15).astype(np.uint8)
            toxic[:, :, 2] = (gray * 0.10).astype(np.uint8)
            toxic[::3, :, :] = (toxic[::3, :, :] * 0.45).astype(np.uint8)
            return toxic

        elif mode == "cosmic_shift":
            b, g, r = cv2.split(frame)
            shift = 10
            r_shifted = np.roll(r, shift, axis=1)
            b_shifted = np.roll(b, -shift, axis=1)
            return cv2.merge([b_shifted, g, r_shifted])

        elif mode == "zen_mist":
            return cv2.addWeighted(frame, 0.78, np.full_like(frame, 245), 0.22, 0)

        elif mode == "dreamy_glow":
            small = cv2.resize(frame, (w // 4, h // 4), interpolation=cv2.INTER_LINEAR)
            blur_small = cv2.GaussianBlur(small, (15, 15), 0)
            blur = cv2.resize(blur_small, (w, h), interpolation=cv2.INTER_LINEAR)
            bloom = cv2.addWeighted(frame, 0.65, blur, 0.35, 0)
            tint = np.array([1.0, 0.9, 1.15])
            return np.clip(bloom.astype(np.float32) * tint, 0, 255).astype(np.uint8)

        return frame

    # ---------------------------------------------------------
    # "What We Feel By Seeing Your Face" - Floating Facial Vibe & Live Expression Card
    # ---------------------------------------------------------
    def render_facial_vibe_card(
        self,
        frame: np.ndarray,
        landmark_data: Dict[str, Any],
        face_id: int,
        emotion: EmotionState,
        vibe_text: str,
        display_mode: VibeDisplayMode = VibeDisplayMode.CARD,
        expression_summary: Optional[str] = None,
        expression_details: Optional[List[str]] = None
    ) -> np.ndarray:
        if display_mode == VibeDisplayMode.OFF or not landmark_data.get("has_face"):
            return frame

        h_frame, w_frame = frame.shape[:2]
        forehead_x, forehead_y = landmark_data["forehead_pt"]

        # Clean card dimension floating smoothly above head without face clutter
        card_w = min(420, max(300, int(w_frame * 0.48)))
        card_h = 56 if display_mode == VibeDisplayMode.COMPACT else 78

        card_x = max(10, min(w_frame - card_w - 10, forehead_x - card_w // 2))
        card_y = max(12, forehead_y - card_h - 22)

        # Draw frosted translucent card backing (No connecting pins or dots on face)
        sub_img = frame[card_y:card_y+card_h, card_x:card_x+card_w]
        if sub_img.shape[0] == card_h and sub_img.shape[1] == card_w:
            dark_rect = np.full_like(sub_img, (14, 14, 20))
            blended_rect = cv2.addWeighted(dark_rect, 0.84, sub_img, 0.16, 0)
            frame[card_y:card_y+card_h, card_x:card_x+card_w] = blended_rect

        # Accent colors for emotion archetypes
        EMO_ACCENTS = {
            EmotionState.PEACE: (180, 240, 180),
            EmotionState.LOVE: (220, 120, 255),
            EmotionState.LAUGHTER: (50, 225, 255),
            EmotionState.SORROW: (230, 160, 80),
            EmotionState.ANGER: (50, 60, 255),
            EmotionState.COURAGE: (50, 215, 255),
            EmotionState.FEAR: (240, 120, 200),
            EmotionState.DISGUST: (80, 230, 100),
            EmotionState.WONDER: (255, 230, 70),
        }
        accent_col = EMO_ACCENTS.get(emotion, (255, 215, 60))

        # Card border with left vertical accent stripe
        cv2.rectangle(frame, (card_x, card_y), (card_x + card_w, card_y + card_h), (75, 75, 95), 1)
        cv2.rectangle(frame, (card_x, card_y), (card_x + 4, card_y + card_h), accent_col, -1)

        # 1. Primary Expression Written Down (Explicit Title)
        expr_title = expression_summary or landmark_data.get("expression_summary") or f"{emotion.value} (100%)"
        header_text = f"FACE #{face_id + 1}: {expr_title}"
        cv2.putText(frame, header_text, (card_x + 14, card_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.44, accent_col, 1, cv2.LINE_AA)

        # 2. Detailed Micro-Expression Action Cues
        cues = expression_details or landmark_data.get("expression_details") or []
        cues_text = " | ".join(cues[:3]) if cues else "Micro-Expressions: Relaxed"
        if len(cues_text) > 46:
            cues_text = cues_text[:44] + "..."
        cv2.putText(frame, cues_text, (card_x + 14, card_y + 38), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (235, 235, 245), 1, cv2.LINE_AA)

        # 3. What We Feel By Seeing Your Face
        if display_mode == VibeDisplayMode.CARD and vibe_text:
            body_text = vibe_text
            if len(body_text) > 44:
                body_text = body_text[:42] + "..."
            cv2.putText(frame, f'"{body_text}"', (card_x + 14, card_y + 57), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 220, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, "WHAT WE FEEL BY SEEING THIS FACE", (card_x + 14, card_y + 71), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (135, 135, 150), 1, cv2.LINE_AA)

        return frame

    # ---------------------------------------------------------
    # 3D Head-Pose Anchored Stickers & 3D Emojis Compositor
    # ---------------------------------------------------------
    def overlay_sticker(
        self,
        frame: np.ndarray,
        sticker_path: str,
        landmark_data: Dict[str, Any],
        alpha_mult: float = 1.0,
        position_mode: str = "forehead"
    ) -> np.ndarray:
        if alpha_mult <= 0.01 or not os.path.exists(sticker_path):
            return frame

        if sticker_path not in self._sticker_cache:
            img = cv2.imread(sticker_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                return frame
            if img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
            self._sticker_cache[sticker_path] = img

        sticker = self._sticker_cache[sticker_path]
        h_frame, w_frame = frame.shape[:2]

        if landmark_data.get("has_face") and landmark_data.get("forehead_pt"):
            iod = landmark_data.get("interocular_dist", 100.0)
            target_w = int(iod * 2.2)
            target_h = int(target_w * (sticker.shape[0] / sticker.shape[1]))
            forehead_x, forehead_y = landmark_data["forehead_pt"]

            if position_mode == "eyes" and landmark_data.get("center_pt"):
                cx, cy = landmark_data["center_pt"]
                center_x = cx
                center_y = cy
            else:
                center_x = forehead_x
                center_y = int(forehead_y - target_h * 0.35)

            roll_deg = landmark_data["euler_angles"][2]
        else:
            target_w = int(w_frame * 0.35)
            target_h = int(target_w * (sticker.shape[0] / sticker.shape[1]))
            center_x = w_frame // 2
            center_y = int(target_h * 0.6)
            roll_deg = 0.0

        target_w = max(32, min(target_w, w_frame))
        target_h = max(32, min(target_h, h_frame))

        resized = cv2.resize(sticker, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        if abs(roll_deg) > 1.0:
            M = cv2.getRotationMatrix2D((target_w / 2.0, target_h / 2.0), -roll_deg, 1.0)
            resized = cv2.warpAffine(
                resized, M, (target_w, target_h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0, 0)
            )

        x1 = center_x - target_w // 2
        y1 = center_y - target_h // 2
        x2 = x1 + target_w
        y2 = y1 + target_h

        clip_x1 = max(0, x1)
        clip_y1 = max(0, y1)
        clip_x2 = min(w_frame, x2)
        clip_y2 = min(h_frame, y2)

        if clip_x1 >= clip_x2 or clip_y1 >= clip_y2:
            return frame

        stk_x1 = clip_x1 - x1
        stk_y1 = clip_y1 - y1
        stk_x2 = stk_x1 + (clip_x2 - clip_x1)
        stk_y2 = stk_y1 + (clip_y2 - clip_y1)

        stk_crop = resized[stk_y1:stk_y2, stk_x1:stk_x2]
        alpha = (stk_crop[:, :, 3] / 255.0) * alpha_mult
        alpha_3 = np.dstack([alpha, alpha, alpha])

        roi = frame[clip_y1:clip_y2, clip_x1:clip_x2]
        blended = (stk_crop[:, :, :3] * alpha_3 + roi * (1.0 - alpha_3)).astype(np.uint8)
        frame[clip_y1:clip_y2, clip_x1:clip_x2] = blended

        return frame

    # ---------------------------------------------------------
    # Pillow Impact Meme Typography
    # ---------------------------------------------------------
    def render_meme_text(
        self,
        frame: np.ndarray,
        top_text: str,
        bottom_text: str,
        alpha: float = 1.0
    ) -> np.ndarray:
        if (not top_text and not bottom_text) or alpha <= 0.01:
            return frame

        h, w = frame.shape[:2]
        rgb_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(rgb_img)

        font_size = max(24, int(w * 0.046))
        font = self._get_font(font_size)

        def draw_centered_caption(text: str, y_pos: int):
            if not text:
                return
            bbox = draw.textbbox((0, 0), text, font=font)
            text_w = bbox[2] - bbox[0]
            x_pos = (w - text_w) // 2

            draw.text(
                (x_pos, y_pos),
                text,
                font=font,
                fill=(255, 255, 255),
                stroke_width=4,
                stroke_fill=(0, 0, 0)
            )

        if top_text:
            draw_centered_caption(top_text, int(h * 0.04))

        if bottom_text:
            bbox = draw.textbbox((0, 0), bottom_text, font=font)
            text_h = bbox[3] - bbox[1]
            draw_centered_caption(bottom_text, int(h - text_h - h * 0.05))

        out_bgr = cv2.cvtColor(np.array(rgb_img), cv2.COLOR_RGB2BGR)

        if alpha < 0.99:
            return cv2.addWeighted(out_bgr, alpha, frame, 1.0 - alpha, 0)
        return out_bgr

    def _get_font(self, size: int) -> ImageFont.FreeTypeFont:
        if size in self._font_cache:
            return self._font_cache[size]

        font_candidates = [
            "impact.ttf", "Impact.ttf", "arialbd.ttf", "Arial.ttf",
            "C:\\Windows\\Fonts\\impact.ttf",
            "C:\\Windows\\Fonts\\arialbd.ttf"
        ]
        for c in font_candidates:
            try:
                f = ImageFont.truetype(c, size)
                self._font_cache[size] = f
                return f
            except Exception:
                continue

        f = ImageFont.load_default()
        self._font_cache[size] = f
        return f
