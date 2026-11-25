"""
Async NVIDIA NIM Client & Multi-Face MCP Bridge Worker.
Consumes multi-face emotion transition events from queue.Queue(maxsize=4),
connects to FastMCP server via stdio, queries NVIDIA NIM,
applies separate reactions and witty 'facial vibe insights' per face.
"""

import os
import sys
import json
import time
import queue
import random
import asyncio
import threading
from pathlib import Path
from typing import Dict, Any, Optional, List

# pyrefly: ignore[missing-import]
from dotenv import load_dotenv

from src.state_manager import StateStore, EmotionState

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")
MANIFEST_PATH = ROOT_DIR / "assets" / "manifest.json"


class NvidiaMcpWorker:
    def __init__(self, state_store: StateStore, event_queue: queue.Queue):
        self.state_store = state_store
        self.event_queue = event_queue
        self.running = True
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.thread: Optional[threading.Thread] = None

        self.api_key = os.getenv("NVIDIA_API_KEY", "").strip()
        self.base_url = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
        self.primary_model = os.getenv("NVIDIA_MODEL", "google/gemma-4-26b-a4b-it")
        self.fallback_model = os.getenv("NVIDIA_FALLBACK_MODEL", "meta/llama-3.1-8b-instruct")

        self.manifest = self._load_manifest()

    def _load_manifest(self) -> Dict[str, Any]:
        if MANIFEST_PATH.exists():
            try:
                with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {"emotions": {}}

    def start(self):
        self.thread = threading.Thread(target=self._run_thread, daemon=True, name="NvidiaMcpWorkerThread")
        self.thread.start()

    def stop(self):
        self.running = False

    def _run_thread(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        try:
            self.loop.run_until_complete(self._worker_main_loop())
        finally:
            self.loop.close()

    async def _worker_main_loop(self):
        is_online = bool(self.api_key and self.api_key.startswith("nvapi-"))
        self.state_store.set_nim_status("ONLINE" if is_online else "OFFLINE FALLBACK")

        mcp_session = None
        mcp_tools_openai = []
        try:
            # pyrefly: ignore[missing-import]
            from mcp.client.stdio import stdio_client, StdioServerParameters
            # pyrefly: ignore[missing-import]
            from mcp.client.session import ClientSession

            server_params = StdioServerParameters(
                command=sys.executable,
                args=["-m", "src.meme_mcp_server"],
                cwd=str(ROOT_DIR),
                env=os.environ.copy()
            )

            async with stdio_client(server_params) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    mcp_session = session

                    tools_result = await session.list_tools()
                    for t in tools_result.tools:
                        mcp_tools_openai.append({
                            "type": "function",
                            "function": {
                                "name": t.name,
                                "description": t.description or "",
                                "parameters": t.inputSchema or {}
                            }
                        })
                    print(f"[NvidiaMcpWorker] Connected to FastMCP server with {len(mcp_tools_openai)} tools.")
                    await self._process_events(mcp_session, mcp_tools_openai)
        except Exception as e:
            print(f"[NvidiaMcpWorker] FastMCP client notice ({e}). Operating in direct high-speed fallback mode.")
            await self._process_events(None, [])

    async def _process_events(self, session, tools_schema: List[Dict[str, Any]]):
        while self.running:
            event = None
            try:
                event = self.event_queue.get_nowait()
            except queue.Empty:
                await asyncio.sleep(0.04)
                continue

            if not event:
                continue

            face_id: int = event.get("face_id", 0)
            emotion: EmotionState = event.get("emotion", EmotionState.PEACE)
            confidence: float = event.get("confidence", 1.0)

            await self._handle_emotion_transition(face_id, emotion, confidence, session, tools_schema)
            self.event_queue.task_done()

    async def _handle_emotion_transition(
        self,
        face_id: int,
        emotion: EmotionState,
        confidence: float,
        session,
        tools_schema: List[Dict[str, Any]]
    ):
        if self.api_key and self.api_key.startswith("nvapi-"):
            success = await self._query_nvidia_nim(face_id, emotion, confidence, session, tools_schema)
            if success:
                return

        self._apply_offline_fallback(face_id, emotion, confidence)

    async def _query_nvidia_nim(
        self,
        face_id: int,
        emotion: EmotionState,
        confidence: float,
        session,
        tools_schema: List[Dict[str, Any]]
    ) -> bool:
        try:
            # pyrefly: ignore[missing-import]
            from openai import AsyncOpenAI
            client = AsyncOpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
                timeout=3.5
            )

            recent_history = self.state_store.get_recent_history()
            history_str = "\n".join(f"- {h}" for h in recent_history) if recent_history else "None yet."

            system_prompt = (
                "You are an elite, hilarious real-time meme director monitoring a multi-face webcam feed. "
                f"Face #{face_id + 1} has expressed {emotion.value}.\n"
                "YOUR TASK:\n"
                "1. Call `set_screen_caption` with a punchy 2-line meme (max 6 words per line).\n"
                "2. Call `select_overlay_bundle` to choose stickers and facial vibe insights.\n"
                "3. Optionally call `set_visual_filter` for dramatic effect.\n"
                f"Recent captions:\n{history_str}"
            )

            user_prompt = f"Face #{face_id + 1}: {emotion.value} (Confidence: {confidence:.2f}). Direct this reaction!"

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]

            model_to_use = self.primary_model
            response = None
            try:
                response = await client.chat.completions.create(
                    model=model_to_use,
                    messages=messages,
                    tools=tools_schema if (tools_schema and session) else None,
                    tool_choice="auto" if (tools_schema and session) else None,
                    temperature=0.85,
                    max_tokens=90
                )
            except Exception:
                model_to_use = self.fallback_model
                response = await client.chat.completions.create(
                    model=model_to_use,
                    messages=messages,
                    tools=tools_schema if (tools_schema and session) else None,
                    tool_choice="auto" if (tools_schema and session) else None,
                    temperature=0.85,
                    max_tokens=90
                )

            self.state_store.set_nim_status("ONLINE")

            msg = response.choices[0].message
            top_text, bottom_text = "", ""
            sticker_path, emoji_path, sound_path, filter_mode, vibe_insight = None, None, None, None, None

            if msg.tool_calls and session:
                for tc in msg.tool_calls:
                    fn_name = tc.function.name
                    args = json.loads(tc.function.arguments)
                    args["face_id"] = face_id

                    tool_result_str = await session.call_tool(fn_name, args)
                    res_content = tool_result_str.content[0].text if hasattr(tool_result_str, "content") else str(tool_result_str)
                    res_data = json.loads(res_content) if isinstance(res_content, str) and res_content.startswith("{") else {}

                    if fn_name == "set_screen_caption":
                        top_text = res_data.get("top_text", args.get("top_text", ""))
                        bottom_text = res_data.get("bottom_text", args.get("bottom_text", ""))
                    elif fn_name == "select_overlay_bundle":
                        sticker_path = res_data.get("sticker_path")
                        emoji_path = res_data.get("emoji_path")
                        sound_path = res_data.get("sound_path")
                        filter_mode = res_data.get("default_filter")
                        vibe_insight = res_data.get("vibe_insight")
                    elif fn_name == "set_visual_filter":
                        filter_mode = res_data.get("filter_mode")
                    elif fn_name in ("trigger_audio_cue", "trigger_soundboard"):
                        sound_path = res_data.get("sound_path")
            else:
                content = msg.content or ""
                lines = [l.strip() for l in content.split("\n") if l.strip()]
                top_text = lines[0] if len(lines) > 0 else f"{emotion.value} DETECTED"
                bottom_text = lines[1] if len(lines) > 1 else ""

            if top_text or bottom_text:
                emo_info = self.manifest.get("emotions", {}).get(emotion.value, {})
                if not sticker_path:
                    sticker_path = str(ROOT_DIR / emo_info.get("sticker", "assets/memes/peace_lotus.png"))
                if not emoji_path:
                    emoji_path = str(ROOT_DIR / emo_info.get("emoji", "assets/memes/emoji_zen_angel.png"))
                if not sound_path:
                    sound_path = str(ROOT_DIR / emo_info.get("sound", "assets/sounds/peace_bell.wav"))
                if not vibe_insight:
                    vibes = emo_info.get("vibe_insights", [f"VIBE: Pure {emotion.value} energy"])
                    vibe_insight = random.choice(vibes)

                self.state_store.update_ai_bundle(
                    emotion=emotion,
                    top_text=top_text,
                    bottom_text=bottom_text,
                    face_id=face_id,
                    sticker_path=sticker_path,
                    emoji_path=emoji_path,
                    sound_path=sound_path,
                    filter_mode=filter_mode,
                    vibe_insight=vibe_insight
                )
                return True

        except Exception as e:
            self.state_store.set_nim_status("OFFLINE FALLBACK")
            return False

        return False

    def _apply_offline_fallback(self, face_id: int, emotion: EmotionState, confidence: float):
        emo_data = self.manifest.get("emotions", {}).get(emotion.value, {})
        captions_pool = emo_data.get("fallback_captions", [
            {"top": f"{emotion.value} MODE", "bottom": "STAND BY"}
        ])

        recent = self.state_store.get_recent_history()
        candidates = [c for c in captions_pool if f"F{face_id}: {c['top']} | {c['bottom']}" not in recent]
        chosen = random.choice(candidates if candidates else captions_pool)

        sticker_rel = emo_data.get("sticker", "assets/memes/peace_lotus.png")
        emoji_rel = emo_data.get("emoji", "assets/memes/emoji_zen_angel.png")
        sound_rel = emo_data.get("sound", "assets/sounds/peace_bell.wav")
        if "sound_alt" in emo_data and random.random() < 0.45:
            sound_rel = emo_data["sound_alt"]

        filter_mode = emo_data.get("default_filter", "normal")
        vibes = emo_data.get("vibe_insights", [f"VIBE: High {emotion.value} aura"])
        chosen_vibe = random.choice(vibes)

        self.state_store.update_ai_bundle(
            emotion=emotion,
            top_text=chosen.get("top", ""),
            bottom_text=chosen.get("bottom", ""),
            face_id=face_id,
            sticker_path=str(ROOT_DIR / sticker_rel),
            emoji_path=str(ROOT_DIR / emoji_rel),
            sound_path=str(ROOT_DIR / sound_rel),
            filter_mode=filter_mode,
            vibe_insight=chosen_vibe
        )
