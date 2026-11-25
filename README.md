# Real-Time Webcam Emotion & Meme Filter (Multi-Face & Heavy Audio Edition)
### Powered by MediaPipe Face Landmarker, OpenCV, Pygame, FastMCP, and NVIDIA NIM APIs

An ultra-smooth, zero-lag, real-time computer vision and generative AI meme filter engine. The system tracks and analyzes up to **4 faces simultaneously**, classifies each face across **9 foundational emotional archetypes** (*Navarasa*) using a high-accuracy dual-engine classifier (52 ARKit blendshapes fused with 3D facial geometric ratios), displays floating **"What We Feel By Seeing Your Face"** vibe cards, and triggers **Heavy Bass Drops**, **viral soundboard cues**, **3D viral emojis**, and **accelerated comic shaders** with an asynchronous **FastMCP** director powered by **NVIDIA NIM** (`google/gemma-4-26b-a4b-it` / `meta/llama-3.1-8b-instruct`).

---

## 🚀 Key Features & Capabilities

1. **Simultaneous Multi-Face Tracking & Separate Reactions**:
   - Analyzes up to 4 faces concurrently (`num_faces=4`).
   - Each face maintains an independent baseline calibration, temporal hysteresis debouncer, emotion classification, floating meme sticker/3D emoji, and reaction captions.
   - Detects distinct simultaneous moods (e.g., Person 1 laughing 😂 while Person 2 is terrified 💀).

2. **Live Expression Written Down & Facial Vibe Check**:
   - **Expression Explicitly Written Down**: Displays the dominant expression title (e.g. `LAUGHTER / WIDE SMILE (92%)`, `SURPRISE / WONDER / AWE (88%)`, `ANGER / FURIOUS BROW (85%)`) in high-contrast colored typography.
   - **Micro-Expression Action Breakdown**: Displays real-time anatomical action percentages (e.g. `Smile: 88% | Mouth Open: 52% | Eyes Squint: 45%` or `Brows Up: 78% | Eyes Wide: 65%`).
   - **Glitter-Free Clean Face**: Floating frosted-glass card with zero dots, zero pin circles, and zero glitter cluttering the face.
   - **"What We Feel By Seeing Your Face"**: Real-time intuitive psychological reading of the person's expression and vibe.
   - Toggleable between **Detailed Card Mode**, **Compact Mode**, or **Hidden** with hotkey **`V`**.

3. **High-Accuracy & High-Sensitivity Dual-Engine Classifier**:
   - Zero-lag, direct fusion of MediaPipe 52 ARKit blendshapes and 3D facial geometric ratios:
     - **Smile & Joy**: Fuses bilateral `mouthSmile` blendshapes with mouth corner horizontal stretch ratio (`landmarks #61-#291`).
     - **Open Mouth / Laughter**: Fuses `jawOpen` with 3D Mouth Aspect Ratio (**MAR**).
     - **Surprise / Wonder**: Fuses `browOuterUp` and `browInnerUp` with Eyebrow-to-Eye distance (**BED**) and Eye Aspect Ratio (**EAR**).
     - **Anger / Fury**: Fuses `browDown` with nose sneer and eyebrow lowering distance.
     - **Sorrow / Pout**: Fuses `browInnerUp` with mouth corner drooping and lip depressors.
     - **Fear / Shock**: Fuses wide-open `eyeWide` with brow stretch and jaw drop.
     - **Disgust / Revulsion**: Fuses nose sneer (`noseSneer`) with upper lip elevation (`mouthUpperUp`).
     - **Courage / Smirk**: Fuses jaw forward thrust (`jawForward`) with dimple smirk and chin elevation.
   - Eliminates resting facial bias without locking or zeroing out natural smiles.
   - Instant 1-frame trigger threshold (`entry_threshold=0.22`, `hold_duration=0.35s`) for lightning-fast responsive updates.

4. **Heavy Bass, Cinematic Impact & Viral SFX (22 Sounds)**:
   - **4 New Heavy Audio Cues**:
     - 🔊 **Massive 808 Sub-Bass Drop** (`heavy_bass_drop.wav`): Heavy seismic sub-bass rumble.
     - 💥 **Cinematic Impact Thud** (`heavy_cinematic_thud.wav`): Hollywood trailer low-frequency thud.
     - ⚡ **Heavy Glitch Bass** (`heavy_glitch_bass.wav`): High-voltage distorted electronic bass drop.
     - 🔨 **Heavy Industrial Metal Slam** (`heavy_metal_slam.wav`): Heavy resonant mechanical impact.
   - Press **`H`** to fire a heavy bass drop at any moment!
   - Press **`S`** for viral meme soundboard cues (Vine Boom, Bruh, Airhorn, Wow, Emotional Damage, Victory, Screaming Goat, Record Scratch).

5. **Ultra-Smooth 60+ FPS Pipeline**:
   - **Threaded Camera Stream (`ThreadedCamera`)**: Video capture runs in an isolated background thread, eliminating camera driver frame hitching.
   - **Pyramid-Accelerated Shaders**: Spatial filters (comic sketch, graphite pencil, cyberpunk neon, deep fried) are processed at half-resolution with bilinear reconstruction, cutting filter latency from 45ms to **~3ms** (10x speedup!).
   - **In-Memory Audio Preloading (`SoundManager`)**: All 22 WAV audio cues are cached directly into RAM on startup, eliminating disk I/O stutters.

6. **19 Transparent RGBA Meme Stickers & 3D Emojis**:
   - 9 Themed Archetype Stickers (Lotus, Hearts, Joy Tears, Raincloud, Steam, Aura, Ghost, Nauseous, Stars).
   - 10 Viral 3D Emojis (😂 Laughing, 😍 Heart Eyes, 😎 Thug Sunglasses, 🤯 Mind Blown, 🤡 Clown, 💀 Dead Skull, 😈 Devil, 🗿 Gigachad, 😭 Crying Stream, 😇 Zen Angel).
   - Press **`E`** to toggle Emoji Style: `COMBO` ➔ `3D_EMOJI` ➔ `MEME_STICKER` ➔ `PARTICLES`.

---

## 🏛 Architecture & Concurrency Model

```mermaid
flowchart TD
    subgraph Thread1 ["Thread 1: Main OpenCV UI Loop (60+ FPS)"]
        Cam["ThreadedCamera (Background Reader)"] --> MP["MediaPipe FaceLandmarker<br/>(IMAGE Mode - Multi-Face num_faces=4)"]
        MP --> VE["Dual-Engine Vision Classifier<br/>(52 Blendshapes + 3D Geometric Ratios)"]
        VE --> Debounce["State Manager<br/>(Per-Face Debouncing & Baseline Offsets)"]
        Debounce -->|Transition Event| EQ["queue.Queue(maxsize=3)<br/>(Non-blocking Event Queue)"]
        Debounce --> Vibe["Floating Vibe Card Engine<br/>('What We Feel By Seeing Your Face')"]
        Vibe --> Blend["Compositor<br/>(Pyramid Shaders + 3D Emojis + Particles + Impact Text)"]
        Blend --> HUD["Live Multi-Face HUD & In-Memory Soundboard"]
        HUD --> Display["cv2.imshow('Live Meme Cam')"]
    end

    subgraph Thread2 ["Thread 2: Async AI & MCP Worker (asyncio)"]
        EQ -->|Non-blocking pop| Worker["NVIDIA MCP Worker"]
        Worker --> NIM{"NVIDIA NIM API<br/>(gemma-4-26b-a4b-it)"}
        NIM -->|Tool Calls| MCPBridge["stdio Client"]
        NIM -.->|Fallback on timeout/missing key| Fallback["Manifest Catalog Fallback"]
        MCPBridge -->|Apply Updates| Store[("Thread-Safe StateStore<br/>(Multi-Face Registry)")]
        Fallback -->|Apply Updates| Store
    end

    subgraph Process3 ["Subprocess / Stdio: FastMCP Server"]
        MCPBridge <-->|MCP Protocol (JSON-RPC)| MCPServer["meme_mcp_server.py<br/>(FastMCP 'MemeDirectorMCP')"]
        MCPServer --> Resources["Resources:<br/>meme://catalog/assets<br/>meme://session/history"]
        MCPServer --> Tools["Tools:<br/>select_overlay_bundle<br/>set_screen_caption<br/>trigger_audio_cue<br/>trigger_soundboard<br/>set_visual_filter"]
    end

    Store -.->|Read State| Blend
```

---

## 🎭 9-Emotion (*Navarasa*) Classifier Matrix

| Emotion | Dual-Engine Feature Fusion | Archetype | Theme Sticker | 3D Emoji | Primary SFX |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`PEACE`** | Baseline offset subtraction ($\text{all activations} < 0.28$) | *Shanta* (Zen) | `peace_lotus.png` | 😇 `emoji_zen_angel.png` | `peace_bell.wav` |
| **`LOVE`** | Blendshapes + Mouth Corner Elevation ($\text{MCE} > 0.035$) | *Shringara* (Affection) | `love_hearts.png` | 😍 `emoji_heart_eyes.png` | `love_harp.wav` |
| **`LAUGHTER`** | Blendshapes + Mouth Aspect Ratio ($\text{MAR} > 0.38$) + MCE | *Hasya* (Joy) | `laughter_tears.png` | 😂 `emoji_laugh_crying.png` | `laughter_boing.wav` |
| **`SORROW`** | Blendshapes + Brow Inward Angle ($\text{BIA} > 0.12$) + Frown | *Karuna* (Grief) | `sorrow_raincloud.png` | 😭 `emoji_crying_stream.png` | `sorrow_violin.wav` |
| **`ANGER`** | Blendshapes + Brow Lowering Distance ($\text{BED} < 0.18$) | *Raudra* (Fury) | `anger_steam.png` | 😈 `emoji_angry_devil.png` | `heavy_glitch_bass.wav` |
| **`COURAGE`** | Blendshapes + Jaw Forward Thrust + Euler pitch elevation | *Vira* (Heroic Resolve) | `courage_aura.png` | 🗿 `emoji_gigachad.png` | `heavy_metal_slam.wav` |
| **`FEAR`** | Blendshapes + Eye Aspect Ratio ($\text{EAR} > 0.32$) + Brow Lift | *Bhayanaka* (Shock) | `fear_ghost.png` | 💀 `emoji_skull_dead.png` | `fear_sting.wav` |
| **`DISGUST`** | Blendshapes + Nose Sneer + Upper Lip Elevation | *Bibhatsa* (Revulsion) | `disgust_nauseous.png` | 🤡 `emoji_clown.png` | `disgust_gag.wav` |
| **`WONDER`** | Blendshapes + High MAR + High EAR + Raised Brow Arch | *Adbhuta* (Awe) | `wonder_stars.png` | 🤯 `emoji_shocked_mindblown.png` | `heavy_cinematic_thud.wav` |

---

## 📁 Directory Structure

```text
mini project/
├── .env.example                 # Environment variables template
├── requirements.txt             # Locked Python dependencies
├── README.md                    # System documentation
├── scripts/
│   ├── bootstrap_assets.py      # Generator for 19 stickers/emojis & 22 WAV cues
│   └── verify_system.py         # Automated 8-point verification test suite
├── assets/
│   ├── models/                  # Holds face_landmarker.task (MediaPipe)
│   ├── memes/                   # 19 Transparent RGBA stickers & 3D emojis (256x256)
│   ├── sounds/                  # 22 Synthesized 16-bit PCM WAV SFX & Heavy Bass cues
│   └── manifest.json            # Structured catalog of visual/audio assets & vibe insights
├── src/
│   ├── __init__.py
│   ├── state_manager.py         # Multi-face StateStore, Baseline Calibrator & Debouncer
│   ├── vision_engine.py         # Dual-engine classifier, shaders, 3D pose, & vibe cards
│   ├── meme_mcp_server.py       # FastMCP server exposing tools & soundboard
│   ├── nvidia_mcp_client.py     # Multi-face async bridge to NVIDIA NIM & MCP tools
│   └── main.py                  # 60+ FPS Threaded camera loop & Pygame SoundManager
└── exports/                     # Saved PNG meme snapshots
```

---

## ⚡ Quickstart & Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment (Optional for Live NVIDIA AI)
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Add your NVIDIA NIM API key (`nvapi-...`):
```ini
NVIDIA_API_KEY=nvapi-your-key-here
NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1
NVIDIA_MODEL=google/gemma-4-26b-a4b-it
NVIDIA_FALLBACK_MODEL=meta/llama-3.1-8b-instruct
```
> **Note:** If `NVIDIA_API_KEY` is omitted, the application runs **100% offline** using the curated witty fallback jokes and vibe insights in `assets/manifest.json` with zero frame lag.

### 3. Bootstrap Assets
Generate all 19 stickers/emojis and 22 sound cues:
```bash
python scripts/bootstrap_assets.py
```

### 4. Verify System
Run the comprehensive automated verification suite:
```bash
python scripts/verify_system.py
```

### 5. Launch the Application
Start the real-time multi-face webcam filter:
```bash
python -m src.main
```

---

## 🎮 Interactive Hotkeys & Controls

| Key | Action | Description |
| :---: | :--- | :--- |
| **`H`** | **Heavy Bass Drop** | Fires seismic 808 sub-bass earthquake drop on demand with visual HUD feedback |
| **`V`** | **Cycle Vibe Cards** | Toggles "What We Feel" insight display: `CARD` (Frosted Glass) ➔ `COMPACT` (Pill) ➔ `OFF` |
| **`S`** | **Viral Soundboard** | Cycles viral meme sound cues on demand (Vine Boom, Bruh, Airhorn, Wow, Emotional Damage, Victory, Scream) |
| **`E`** | **Cycle Emoji Style** | Toggles between `COMBO` ➔ `3D_EMOJI` ➔ `MEME_STICKER` ➔ `PARTICLES` |
| `1` | **Normal Filter** | Raw camera feed without shaders |
| `2` | **Comic Book Sketch** | Edge-preserving smoothing, clean cel-shading color quantization, and bold comic inking |
| `3` | **Pencil Sketch** | Authentic graphite/pencil sketch with color-dodge shading and hand-drawn contrast |
| `4` | **Cyberpunk Neon** | High-contrast neon edge glow with electric cyan and magenta tones |
| `5` | **Deep Fried** | High-saturation, red boost, and unsharp mask filter |
| `6` | **Noir** | High-contrast black & white vignette |
| `7` | **Dreamy Glow** | Soft pink bloom for *Love* |
| `8` | **Golden Aura** | Radiant golden luminescence for *Courage* |
| `9` | **Toxic Matrix** | Acid green phosphorescence with scanlines for *Disgust* |
| `0` | **Zen Mist** | Muted pastel tranquil filter for *Peace* |
| `+` / `-` | **Sensitivity Tuning** | Increase / decrease emotion trigger sensitivity in real time |
| `C` | **Recalibrate** | Records the next 15 frames to offset resting facial bias |
| `M` | **Mute / Unmute** | Toggles Pygame SFX sound effects |
| `SPACE` | **Snap & Export** | Triggers shutter flash, plays shutter sound, and saves clean meme PNG to `exports/` |
| `Q` / `ESC` | **Quit** | Clean shutdown of threads, camera, and audio mixer |
