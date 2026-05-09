import atexit
import asyncio
import datetime
import json
import os
import re
import select
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import wave

# Core dependencies
import numpy as np
import scipy.signal
import sounddevice as sd
from openclaw_sdk import OpenClawClient
from openclaw_sdk.core.types import ContentEvent, DoneEvent, ErrorEvent
from faster_whisper import WhisperModel
from openwakeword.model import Model

# =========================================================================
# 1. CONFIGURATION & CONSTANTS
# =========================================================================

CONFIG_FILE = "config.json"
MEMORY_FILE = "memory.json"
WAKE_WORD_MODEL = "./wakeword.onnx"
WAKE_WORD_THRESHOLD = 0.5
WAKE_WORD_ARMING_SECONDS = 0.35
WAKE_WORD_INITIAL_ARMING_SECONDS = 2.0
WAKE_WORD_CONSECUTIVE_HITS = 3
WAKE_WORD_POST_SPEECH_COOLDOWN_SECONDS = 1.2

# HARDWARE SETTINGS
INPUT_DEVICE_NAME = None

DEFAULT_CONFIG = {
    "text_model": "github-copilot/gpt-4.1",
    "voice_model": "piper/en_GB-semaine-medium.onnx",
    "input_device_name": "",
    "arecord_input_device": "default",
    "output_device_name": "",
    "chat_memory": True,
    "system_prompt_extras": "",
    "openclaw_agent_id": "main",
    "openclaw_session_name": "main-session",
    "prompt_memory_messages": 6,
}

BASE_SYSTEM_PROMPT = """You are a helpful robot assistant running on a Raspberry Pi, named Jarvis Junior.
Personality: Cute, helpful, robot.
Style: Short sentences. Enthusiastic.

INSTRUCTIONS:
Respond in plain, spoken-friendly text. Do not use markdown.
"""


def load_config():
    config = DEFAULT_CONFIG.copy()
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                user_config = json.load(f)
                config.update(user_config)
        except Exception as e:
            print(f"Config Error: {e}. Using defaults.")
    return config


CURRENT_CONFIG = load_config()
TEXT_MODEL = CURRENT_CONFIG["text_model"]
OPENCLAW_AGENT_ID = CURRENT_CONFIG.get("openclaw_agent_id", "main")
OPENCLAW_SESSION_NAME = CURRENT_CONFIG.get("openclaw_session_name", "main-session")
PROMPT_MEMORY_MESSAGES = int(CURRENT_CONFIG.get("prompt_memory_messages", 6))
SYSTEM_PROMPT = (
    BASE_SYSTEM_PROMPT + "\n\n" + CURRENT_CONFIG.get("system_prompt_extras", "")
)

# Sound Directories
greeting_sounds_dir = "sounds/greeting_sounds"
ack_sounds_dir = "sounds/ack_sounds"
thinking_sounds_dir = "sounds/thinking_sounds"


class BotStates:
    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"
    ERROR = "error"
    WARMUP = "warmup"


class HeadlessBot:
    def __init__(self):
        atexit.register(self.safe_exit)

        self.current_state = BotStates.WARMUP
        self.current_volume = 0

        self.permanent_memory = self.load_chat_history()
        self.session_memory = []
        self.thinking_sound_active = threading.Event()
        self.sound_cycle_indices = {}
        self.openclaw_agent_id = OPENCLAW_AGENT_ID
        self.openclaw_session_name = OPENCLAW_SESSION_NAME
        self.prompt_memory_messages = max(0, PROMPT_MEMORY_MESSAGES)
        self.openclaw_loop = asyncio.new_event_loop()
        self.openclaw_client = None

        self.tts_queue = []
        self.tts_queue_lock = threading.Lock()
        self.tts_thread = None
        self.tts_active = threading.Event()
        self.piper_sentence_silence_s = 0.0
        self.current_audio_process = None
        self.voice_model_path = None
        self.voice_model_error = None
        self.output_native_rate = 48000
        self.piper_stream_rate = 22050
        self.piper_needs_resample = False
        self.wakeword_first_cycle = True
        self.last_audio_output_end = time.monotonic()
        self.input_backend, self.input_device = self.resolve_input_backend()
        self.arecord_input_device = CURRENT_CONFIG.get(
            "arecord_input_device", "default"
        )
        if self.input_backend == "arecord":
            self.arecord_input_device = self.probe_arecord_input_device()
        self.output_device = self.resolve_output_device()
        self.prepare_audio_runtime()

        print("[INIT] Loading Wake Word...", flush=True)
        self.oww_model = None
        if os.path.exists(WAKE_WORD_MODEL):
            try:
                self.oww_model = Model(wakeword_model_paths=[WAKE_WORD_MODEL])
                print("[INIT] Wake Word Loaded.", flush=True)
            except TypeError:
                try:
                    self.oww_model = Model(wakeword_models=[WAKE_WORD_MODEL])
                    print("[INIT] Wake Word Loaded (New API).", flush=True)
                except Exception as e:
                    print(f"[CRITICAL] Failed to load model: {e}")
            except Exception as e:
                print(f"[CRITICAL] Failed to load model: {e}")
        else:
            print(f"[CRITICAL] Model not found: {WAKE_WORD_MODEL}")

        print("[INIT] Loading Whisper Model...", flush=True)
        try:
            self.whisper_model = WhisperModel("base", device="cpu", compute_type="auto")
            print("[INIT] Whisper Model Loaded.", flush=True)
        except Exception as e:
            print(f"[CRITICAL] Failed to load Whisper model: {e}")
            self.whisper_model = None

    def prepare_audio_runtime(self):
        self.voice_model_path = self.resolve_voice_model_path()
        self.output_native_rate = self.get_output_native_rate()
        self.piper_stream_rate = 22050
        self.piper_needs_resample = False

        try:
            sd.check_output_settings(
                device=self.output_device, samplerate=self.piper_stream_rate
            )
        except Exception:
            self.piper_stream_rate = self.output_native_rate
            self.piper_needs_resample = True

    def resolve_voice_model_path(self):
        configured_model = CURRENT_CONFIG.get(
            "voice_model", "piper/en_GB-semaine-medium.onnx"
        )
        model_candidates = [configured_model]
        model_basename = os.path.basename(configured_model)
        if model_basename != configured_model:
            model_candidates.append(model_basename)

        for candidate in model_candidates:
            config_candidate = candidate + ".json"
            if (
                os.path.exists(candidate)
                and os.path.getsize(candidate) > 0
                and os.path.exists(config_candidate)
                and os.path.getsize(config_candidate) > 0
            ):
                self.voice_model_error = None
                return candidate

        self.voice_model_error = (
            "Audio Error: no valid Piper voice model found. "
            f"Checked: {model_candidates}"
        )
        return None

    def get_output_native_rate(self):
        try:
            if self.output_device is not None:
                device_info = sd.query_devices(self.output_device)
            else:
                device_info = sd.query_devices(kind="output")
            return int(device_info["default_samplerate"])
        except Exception:
            return 48000

    def warmup_wakeword_runtime(self):
        if self.oww_model is None:
            return
        try:
            self.oww_model.predict(np.zeros(1280, dtype=np.int16))
            self.oww_model.reset()
        except Exception as e:
            print(f"[WARMUP] Wake word warmup failed: {e}", flush=True)

    def warmup_whisper_runtime(self):
        if not self.whisper_model:
            return
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                tmp_path = tmp_file.name
            with wave.open(tmp_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(np.zeros(3200, dtype=np.int16).tobytes())
            segments, _ = self.whisper_model.transcribe(tmp_path, language="en")
            for _ in segments:
                pass
        except Exception as e:
            print(f"[WARMUP] Whisper warmup failed: {e}", flush=True)
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

    def warmup_piper_runtime(self):
        if not self.voice_model_path:
            if self.voice_model_error:
                print(self.voice_model_error, flush=True)
            return

        try:
            proc = subprocess.Popen(
                ["./piper/piper", "--model", self.voice_model_path, "--output-raw"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
            proc.stdin.write(b"Warm up.\n")
            proc.stdin.close()
            while proc.stdout.read(4096):
                pass
            proc.wait(timeout=10)
        except Exception as e:
            print(f"[WARMUP] Piper warmup failed: {e}", flush=True)

    def resolve_output_device(self):
        preferred = CURRENT_CONFIG.get("output_device_name", "").strip()
        try:
            devices = sd.query_devices()
            if preferred:
                for idx, dev in enumerate(devices):
                    if (
                        dev.get("max_output_channels", 0) > 0
                        and preferred.lower() in dev.get("name", "").lower()
                    ):
                        print(
                            f"[AUDIO] Using configured output device: {dev.get('name')} (index {idx})",
                            flush=True,
                        )
                        return idx
                print(
                    f"[AUDIO] Configured output device not found: '{preferred}'. Falling back to default.",
                    flush=True,
                )

            default_out = sd.default.device[1] if sd.default.device else None
            if isinstance(default_out, int) and default_out >= 0:
                try:
                    name = devices[default_out].get("name", "unknown")
                except Exception:
                    name = "unknown"
                print(
                    f"[AUDIO] Using default output device: {name} (index {default_out})",
                    flush=True,
                )
                return default_out
        except Exception as e:
            print(f"[AUDIO] Could not resolve output device: {e}", flush=True)
        return None

    def resolve_input_backend(self):
        preferred = CURRENT_CONFIG.get("input_device_name", "").strip()
        try:
            devices = sd.query_devices()
            input_candidates = [
                (idx, dev)
                for idx, dev in enumerate(devices)
                if dev.get("max_input_channels", 0) > 0
            ]

            if preferred:
                for idx, dev in input_candidates:
                    if preferred.lower() in dev.get("name", "").lower():
                        print(
                            f"[AUDIO] Using configured input device: {dev.get('name')} (index {idx})",
                            flush=True,
                        )
                        return "sounddevice", idx
                print(
                    f"[AUDIO] Configured input device not found: '{preferred}'.",
                    flush=True,
                )

            default_in = sd.default.device[0] if sd.default.device else None
            if isinstance(default_in, int) and default_in >= 0:
                dev = devices[default_in]
                if dev.get("max_input_channels", 0) > 0:
                    print(
                        f"[AUDIO] Using default input device: {dev.get('name')} (index {default_in})",
                        flush=True,
                    )
                    return "sounddevice", default_in

            if input_candidates:
                idx, dev = input_candidates[0]
                print(
                    f"[AUDIO] Using first available input device: {dev.get('name')} (index {idx})",
                    flush=True,
                )
                return "sounddevice", idx
        except Exception as e:
            print(f"[AUDIO] Could not resolve sounddevice input: {e}", flush=True)

        # PortAudio can fail to enumerate USB capture devices on some Pi setups.
        try:
            probe = subprocess.run(["arecord", "-l"], capture_output=True, text=True)
            if probe.returncode == 0 and "card " in probe.stdout.lower():
                print("[AUDIO] Falling back to arecord input backend.", flush=True)
                return "arecord", None
        except Exception:
            pass

        print("[AUDIO] No usable microphone backend detected.", flush=True)
        return "none", None

    def probe_arecord_input_device(self):
        configured = CURRENT_CONFIG.get("arecord_input_device", "default").strip()
        candidates = []

        # Preserve user preference first, then probe common ALSA capture aliases.
        if configured:
            candidates.append(configured)
        for candidate in [
            "default",
            "dsnoop:CARD=Device,DEV=0",
            "front:CARD=Device,DEV=0",
            "plughw:Device,0",
            "hw:Device,0",
            "sysdefault",
        ]:
            if candidate not in candidates:
                candidates.append(candidate)

        saw_busy = False
        for candidate in candidates:
            cmd = [
                "arecord",
                "-q",
                "-D",
                candidate,
                "-f",
                "S16_LE",
                "-r",
                "16000",
                "-c",
                "1",
                "-d",
                "1",
                "-t",
                "raw",
                "/dev/null",
            ]
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=3,
                )
                if result.returncode == 0:
                    print(
                        f"[AUDIO] arecord input probe selected: {candidate}", flush=True
                    )
                    return candidate
                err = (result.stderr or "").strip()
                if "Device or resource busy" in err:
                    saw_busy = True
                if err:
                    print(
                        f"[AUDIO] arecord probe failed for {candidate}: {err}",
                        flush=True,
                    )
            except Exception as e:
                print(
                    f"[AUDIO] arecord probe exception for {candidate}: {e}", flush=True
                )

        if saw_busy:
            print(
                "[AUDIO] Mic appears busy; trying shared capture path: dsnoop:CARD=Device,DEV=0",
                flush=True,
            )
            return "dsnoop:CARD=Device,DEV=0"

        fallback = configured or "default"
        print(
            f"[AUDIO] arecord probe found no working candidate, using fallback: {fallback}",
            flush=True,
        )
        return fallback

    def _wakeword_from_arecord_or_cli(self):
        chunk_size = 1280
        chunk_bytes = chunk_size * 2
        cmd = [
            "arecord",
            "-q",
            "-D",
            self.arecord_input_device,
            "-f",
            "S16_LE",
            "-r",
            "16000",
            "-c",
            "1",
            "-t",
            "raw",
        ]

        restart_count = 0
        while True:
            proc = None
            try:
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    bufsize=0,
                )
                stream_start = time.monotonic()
                arming_seconds = (
                    WAKE_WORD_INITIAL_ARMING_SECONDS
                    if self.wakeword_first_cycle
                    else WAKE_WORD_ARMING_SECONDS
                )
                wake_ready_at = max(
                    stream_start + arming_seconds,
                    self.last_audio_output_end + WAKE_WORD_POST_SPEECH_COOLDOWN_SECONDS,
                )
                consecutive_hits = 0
                while True:
                    rlist, _, _ = select.select([sys.stdin], [], [], 0.001)
                    if rlist:
                        line = sys.stdin.readline()
                        if line.strip():
                            return "CLI_TEXT", line

                    data = self._read_exact_bytes(proc.stdout, chunk_bytes)
                    if data is None:
                        raise RuntimeError("arecord stream ended unexpectedly")

                    audio_data = np.frombuffer(data, dtype=np.int16)
                    self.oww_model.predict(audio_data)

                    # Ignore startup frames and require repeated hits to avoid
                    # a false wake on the first post-TTS detection cycle.
                    if time.monotonic() < wake_ready_at:
                        consecutive_hits = 0
                        continue

                    wake_hit = False
                    for mdl in self.oww_model.prediction_buffer.keys():
                        scores = list(self.oww_model.prediction_buffer[mdl])
                        if scores and scores[-1] > WAKE_WORD_THRESHOLD:
                            wake_hit = True
                            break

                    if wake_hit:
                        consecutive_hits += 1
                        if consecutive_hits >= WAKE_WORD_CONSECUTIVE_HITS:
                            self.wakeword_first_cycle = False
                            self.oww_model.reset()
                            return "WAKE", None
                    else:
                        consecutive_hits = 0
            except Exception as e:
                restart_count += 1
                detail = ""
                try:
                    if proc and proc.stderr:
                        detail = proc.stderr.read().decode(errors="ignore").strip()
                except Exception:
                    detail = ""
                msg = f"{e}" if not detail else f"{e} ({detail})"
                print(
                    f"[AUDIO] arecord wake stream restart #{restart_count}: {msg}",
                    flush=True,
                )
                if restart_count >= 5:
                    raise RuntimeError(msg)
                time.sleep(0.25)
            finally:
                if proc:
                    try:
                        proc.terminate()
                    except Exception:
                        pass

    def record_voice_adaptive_arecord(self, filename="input.wav"):
        print("Recording (Adaptive/arecord)...", flush=True)
        time.sleep(0.5)

        samplerate = 16000
        silence_threshold = 0.006
        silence_duration = 1.5
        max_record_time = 30.0
        chunk_duration = 0.05
        chunk_samples = int(samplerate * chunk_duration)
        chunk_bytes = chunk_samples * 2

        num_silent_chunks = int(silence_duration / chunk_duration)
        max_chunks = int(max_record_time / chunk_duration)

        buffer = []
        silent_chunks = 0
        recorded_chunks = 0
        silence_started = False

        cmd = [
            "arecord",
            "-q",
            "-D",
            self.arecord_input_device,
            "-f",
            "S16_LE",
            "-r",
            str(samplerate),
            "-c",
            "1",
            "-t",
            "raw",
        ]

        proc = None
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=0,
            )
            while not silence_started and recorded_chunks < max_chunks:
                data = self._read_exact_bytes(proc.stdout, chunk_bytes)
                if data is None:
                    break

                audio_i16 = np.frombuffer(data, dtype=np.int16)
                audio_f32 = (audio_i16.astype(np.float32) / 32768.0).reshape(-1, 1)
                buffer.append(audio_f32)
                recorded_chunks += 1

                if recorded_chunks < 5:
                    continue

                volume_norm = np.linalg.norm(audio_f32) / np.sqrt(len(audio_f32))
                if volume_norm < silence_threshold:
                    silent_chunks += 1
                    if silent_chunks >= num_silent_chunks:
                        silence_started = True
                else:
                    silent_chunks = 0
        except Exception as e:
            print(f"arecord record error: {e}", flush=True)
            return None
        finally:
            if proc:
                try:
                    proc.terminate()
                except Exception:
                    pass

        return self.save_audio_buffer(buffer, filename, samplerate)

    def _read_exact_bytes(self, stream, expected_size):
        chunks = []
        total = 0
        while total < expected_size:
            part = stream.read(expected_size - total)
            if not part:
                return None
            chunks.append(part)
            total += len(part)
        return b"".join(chunks)

    def normalize_text_for_tts(self, text):
        # Normalize common smart punctuation so contractions and sentence endings are preserved.
        normalized = (
            text.replace("’", "'")
            .replace("‘", "'")
            .replace("“", '"')
            .replace("”", '"')
            .replace("…", "...")
        )
        # Piper reads newline-delimited input; flatten embedded newlines
        # so one queued utterance stays one spoken utterance.
        normalized = (
            normalized.replace("\r\n", " ").replace("\r", " ").replace("\n", " ")
        )
        normalized = re.sub(r"\s+", " ", normalized).strip()
        # return re.sub(r"[^\w\s.,!?;:'\"()\+\-\[\]{}]", "", normalized)
        return normalized

    def safe_exit(self):
        print("\n--- SHUTDOWN SEQUENCE ---", flush=True)
        if self.current_audio_process:
            try:
                self.current_audio_process.terminate()
                self.current_audio_process.wait(timeout=1)
            except Exception:
                pass

        self.thinking_sound_active.clear()
        self.tts_active.clear()
        self._shutdown_openclaw_runtime()
        self.save_chat_history()

    async def _ensure_openclaw_client(self):
        if self.openclaw_client is None:
            self.openclaw_client = await OpenClawClient.connect()
        try:
            await self.openclaw_client.gateway.connect()
        except Exception:
            # Some gateway implementations auto-connect and may not expose
            # an idempotent connect lifecycle; execution path will retry.
            pass
        return self.openclaw_client

    async def _reset_openclaw_client(self):
        if self.openclaw_client is not None:
            try:
                await self.openclaw_client.gateway.close()
            except Exception:
                pass
        self.openclaw_client = None

    def _is_openclaw_not_connected_error(self, exc):
        return "not connected" in str(exc).lower()

    def _run_openclaw_coro(self, coro):
        return self.openclaw_loop.run_until_complete(coro)

    def _shutdown_openclaw_runtime(self):
        async def _close_client():
            if self.openclaw_client is not None:
                try:
                    await self.openclaw_client.gateway.close()
                finally:
                    self.openclaw_client = None

        try:
            if self.openclaw_loop and not self.openclaw_loop.is_closed():
                self.openclaw_loop.run_until_complete(_close_client())
                self.openclaw_loop.close()
        except Exception:
            pass

    def build_prompt(self, user_text):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ]
        prompt_lines = []
        for message in messages:
            role = message.get("role", "user")
            content = str(message.get("content", "")).strip()
            if not content:
                continue
            if role == "system":
                label = "SYSTEM"
            elif role == "assistant":
                label = "ASSISTANT"
            else:
                label = "USER"
            prompt_lines.append(f"{label}: {content}")
        prompt_lines.append("ASSISTANT:")
        return "\n\n".join(prompt_lines)

    async def _openclaw_execute(self, prompt):
        async def _execute_once():
            client = await self._ensure_openclaw_client()
            agent = client.get_agent(
                self.openclaw_agent_id, session_name=self.openclaw_session_name
            )
            return await agent.execute(prompt)

        try:
            result = await _execute_once()
        except Exception as e:
            if not self._is_openclaw_not_connected_error(e):
                raise
            print(
                "[OPENCLAW] Gateway reported disconnected. Reconnecting and retrying...",
                flush=True,
            )
            await self._reset_openclaw_client()
            result = await _execute_once()

        full_response_buffer = (result.content or "").strip()
        lowered = full_response_buffer.lower()
        is_action_mode = '{"' in full_response_buffer or "action:" in lowered
        return full_response_buffer, is_action_mode

    async def _openclaw_execute_stream(self, prompt, on_content=None):
        async def _execute_once():
            client = await self._ensure_openclaw_client()
            agent = client.get_agent(
                self.openclaw_agent_id, session_name=self.openclaw_session_name
            )

            full_response_parts = []
            async for event in agent.execute_stream_typed(prompt):
                if isinstance(event, ContentEvent):
                    chunk = event.text or ""
                    if chunk:
                        full_response_parts.append(chunk)
                        if on_content is not None:
                            on_content(chunk)
                elif isinstance(event, DoneEvent):
                    if event.content and not full_response_parts:
                        full_response_parts.append(event.content)
                    break
                elif isinstance(event, ErrorEvent):
                    raise RuntimeError(event.message or "OpenClaw stream error")

            full_response_buffer = "".join(full_response_parts).strip()
            lowered = full_response_buffer.lower()
            is_action_mode = '{"' in full_response_buffer or "action:" in lowered
            return full_response_buffer, is_action_mode

        try:
            return await _execute_once()
        except Exception as e:
            if not self._is_openclaw_not_connected_error(e):
                raise
            print(
                "[OPENCLAW] Gateway reported disconnected. Reconnecting and retrying...",
                flush=True,
            )
            await self._reset_openclaw_client()
            return await _execute_once()

    async def _reset_openclaw_agent_memory(self):
        client = await self._ensure_openclaw_client()
        agent = client.get_agent(
            self.openclaw_agent_id, session_name=self.openclaw_session_name
        )
        reset_result = agent.reset_memory()
        if asyncio.iscoroutine(reset_result):
            await reset_result

    def _run_openclaw_reset_memory(self):
        return self._run_openclaw_coro(self._reset_openclaw_agent_memory())

    def _run_openclaw_execute(self, prompt):
        return self._run_openclaw_coro(self._openclaw_execute(prompt))

    def _run_openclaw_execute_stream(self, prompt, on_content=None):
        return self._run_openclaw_coro(
            self._openclaw_execute_stream(prompt, on_content)
        )

    def set_state(self, state, msg=""):
        if self.current_state != state:
            self.current_state = state
        if msg:
            print(f"[STATE] {state.upper()}: {msg}", flush=True)

    def append_to_text(self, text, newline=True):
        if newline:
            print(text, flush=True)
        else:
            print(text, end="", flush=True)

    # ---------------------------------------------------------------------
    # Action Router
    # ---------------------------------------------------------------------

    def execute_action_and_get_result(self, action_data):
        raw_action = action_data.get("action", "").lower().strip()
        value = action_data.get("value") or action_data.get("query")

        valid_tools = {"get_time"}

        aliases = {
            "check_time": "get_time",
        }

        action = aliases.get(raw_action, raw_action)
        print(f"ACTION: {raw_action} -> {action}", flush=True)

        if action not in valid_tools:
            if value and isinstance(value, str) and len(value.split()) > 1:
                return f"CHAT_FALLBACK::{value}"
            return "INVALID_ACTION"

        if action == "get_time":
            now = datetime.datetime.now().strftime("%I:%M %p")
            return f"The current time is {now}."

        return None

    # ---------------------------------------------------------------------
    # Core Loop
    # ---------------------------------------------------------------------

    def run(self):
        try:
            self.warm_up_logic()
            self.tts_thread = threading.Thread(target=self._tts_worker, daemon=True)
            self.tts_thread.start()

            print(
                "[INFO] Headless mode active. Type a message and press Enter for direct text input.",
                flush=True,
            )
            print(
                "[INFO] Wake word audio is still active if a microphone is connected.",
                flush=True,
            )

            while True:
                trigger_source, cli_text = self.detect_wake_word_or_cli_text()

                self.set_state(BotStates.LISTENING, "I'm listening!")

                if trigger_source == "CLI_TEXT":
                    user_text = cli_text.strip()
                else:
                    audio_file = self.record_voice_adaptive()
                    if not audio_file:
                        self.set_state(BotStates.IDLE, "Heard nothing.")
                        continue
                    user_text = self.transcribe_audio(audio_file)

                if not user_text:
                    self.set_state(BotStates.IDLE, "Transcription empty.")
                    continue

                self.append_to_text(f"YOU: {user_text}")
                self.chat_and_respond(user_text)

        except KeyboardInterrupt:
            self.set_state(BotStates.IDLE, "Stopping...")
        except Exception as e:
            traceback.print_exc()
            self.set_state(BotStates.ERROR, f"Fatal Error: {str(e)[:80]}")

    def warm_up_logic(self):
        self.set_state(BotStates.WARMUP, "Warming up brains...")
        try:
            self._run_openclaw_warmup()
        except Exception as e:
            print(f"Failed to connect to OpenClaw model {TEXT_MODEL}: {e}", flush=True)
        self.warmup_wakeword_runtime()
        self.warmup_whisper_runtime()
        self.warmup_piper_runtime()
        self.play_sound(self.get_sound(greeting_sounds_dir))
        print(f"Models loaded via OpenClaw ({TEXT_MODEL}).", flush=True)

    def _run_openclaw_warmup(self):
        async def _warmup():
            client = await self._ensure_openclaw_client()
            try:
                await client.gateway.connect()
            except Exception:
                pass
            await client.gateway.health()
            return True

        connected = self._run_openclaw_coro(_warmup())
        if connected:
            print("[OPENCLAW] Connected and healthy during startup warmup.", flush=True)

    def detect_wake_word_or_cli_text(self):
        self.set_state(BotStates.IDLE, "Waiting...")

        if self.oww_model:
            self.oww_model.reset()

        # If wake word model is unavailable, fall back to CLI-only mode.
        if self.oww_model is None:
            line = sys.stdin.readline()
            return "CLI_TEXT", line

        if self.input_backend == "none":
            print(
                "Wake word unavailable: no input device found. Type your prompt and press Enter.",
                flush=True,
            )
            line = sys.stdin.readline()
            return "CLI_TEXT", line

        if self.input_backend == "arecord":
            try:
                return self._wakeword_from_arecord_or_cli()
            except Exception as e:
                print(f"Wake Word Stream Error: {e}", flush=True)
                line = sys.stdin.readline()
                return "CLI_TEXT", line

        chunk_size = 1280
        oww_sample_rate = 16000

        try:
            device_info = sd.query_devices(kind="input")
            native_rate = int(device_info["default_samplerate"])
        except Exception:
            native_rate = 48000

        use_resampling = native_rate != oww_sample_rate
        input_rate = native_rate if use_resampling else oww_sample_rate
        input_chunk_size = (
            int(chunk_size * (input_rate / oww_sample_rate))
            if use_resampling
            else chunk_size
        )

        try:
            with sd.InputStream(
                samplerate=input_rate,
                channels=1,
                dtype="int16",
                blocksize=input_chunk_size,
                device=self.input_device,
            ) as stream:
                stream_start = time.monotonic()
                arming_seconds = (
                    WAKE_WORD_INITIAL_ARMING_SECONDS
                    if self.wakeword_first_cycle
                    else WAKE_WORD_ARMING_SECONDS
                )
                wake_ready_at = max(
                    stream_start + arming_seconds,
                    self.last_audio_output_end + WAKE_WORD_POST_SPEECH_COOLDOWN_SECONDS,
                )
                consecutive_hits = 0
                while True:
                    rlist, _, _ = select.select([sys.stdin], [], [], 0.001)
                    if rlist:
                        line = sys.stdin.readline()
                        if line.strip():
                            return "CLI_TEXT", line

                    data, _ = stream.read(input_chunk_size)
                    audio_data = np.frombuffer(data, dtype=np.int16)

                    if use_resampling:
                        audio_data = scipy.signal.resample(
                            audio_data, chunk_size
                        ).astype(np.int16)

                    self.oww_model.predict(audio_data)

                    if time.monotonic() < wake_ready_at:
                        consecutive_hits = 0
                        continue

                    wake_hit = False
                    for mdl in self.oww_model.prediction_buffer.keys():
                        scores = list(self.oww_model.prediction_buffer[mdl])
                        if scores and scores[-1] > WAKE_WORD_THRESHOLD:
                            wake_hit = True
                            break

                    if wake_hit:
                        consecutive_hits += 1
                        if consecutive_hits >= WAKE_WORD_CONSECUTIVE_HITS:
                            self.wakeword_first_cycle = False
                            self.oww_model.reset()
                            return "WAKE", None
                    else:
                        consecutive_hits = 0
        except Exception as e:
            print(f"Wake Word Stream Error: {e}", flush=True)
            line = sys.stdin.readline()
            return "CLI_TEXT", line

    def record_voice_adaptive(self, filename="input.wav"):
        if self.input_backend == "arecord":
            return self.record_voice_adaptive_arecord(filename)
        if self.input_backend == "none":
            print("No microphone backend available.", flush=True)
            return None

        print("Recording (Adaptive)...", flush=True)
        time.sleep(0.5)
        try:
            if self.input_device is not None:
                device_info = sd.query_devices(self.input_device)
            else:
                device_info = sd.query_devices(kind="input")
            samplerate = int(device_info["default_samplerate"])
        except Exception:
            samplerate = 44100

        silence_threshold = 0.006
        silence_duration = 1.0
        max_record_time = 30.0
        buffer = []
        silent_chunks = 0
        chunk_duration = 0.05
        chunk_size = int(samplerate * chunk_duration)

        num_silent_chunks = int(silence_duration / chunk_duration)
        max_chunks = int(max_record_time / chunk_duration)
        recorded_chunks = 0
        silence_started = False

        def callback(indata, frames, time_info, status):
            nonlocal silent_chunks, recorded_chunks, silence_started
            volume_norm = np.linalg.norm(indata) / np.sqrt(len(indata))
            buffer.append(indata.copy())
            recorded_chunks += 1
            if recorded_chunks < 5:
                return
            if volume_norm < silence_threshold:
                silent_chunks += 1
                if silent_chunks >= num_silent_chunks:
                    silence_started = True
            else:
                silent_chunks = 0

        try:
            with sd.InputStream(
                samplerate=samplerate,
                channels=1,
                callback=callback,
                device=self.input_device,
                blocksize=chunk_size,
            ):
                while not silence_started and recorded_chunks < max_chunks:
                    sd.sleep(int(chunk_duration * 1000))
        except Exception:
            return None

        return self.save_audio_buffer(buffer, filename, samplerate)

    def save_audio_buffer(self, buffer, filename, samplerate=16000):
        if not buffer:
            return None
        audio_data = np.concatenate(buffer, axis=0).flatten()
        audio_data = np.nan_to_num(audio_data, nan=0.0, posinf=0.0, neginf=0.0)
        audio_data = (audio_data * 32767).astype(np.int16)
        with wave.open(filename, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(samplerate)
            wf.writeframes(audio_data.tobytes())
        self.play_sound(self.get_sound(ack_sounds_dir))
        return filename

    def transcribe_audio(self, filename):
        print("Transcribing...", flush=True)
        try:
            if not self.whisper_model:
                print("Whisper model not loaded.")
                return ""

            segments, info = self.whisper_model.transcribe(filename, language="en")
            transcription = "".join([segment.text for segment in segments]).strip()
            print(f"Heard: '{transcription}'", flush=True)
            return transcription
        except Exception as e:
            print(f"Transcription Error: {e}")
            return ""

    # ---------------------------------------------------------------------
    # Chat & Respond
    # ---------------------------------------------------------------------

    def chat_and_respond(self, text):
        turn_start = time.monotonic()
        if "forget everything" in text.lower() or "reset memory" in text.lower():
            self.session_memory = []
            try:
                self._run_openclaw_reset_memory()
            except Exception as e:
                print(f"[OPENCLAW] reset_memory failed: {e}", flush=True)
            with self.tts_queue_lock:
                self.tts_queue.append("Session memory reset.")
            self.set_state(BotStates.IDLE, "Session Memory Reset")
            return

        self.set_state(BotStates.THINKING, "Thinking...")

        self.thinking_sound_active.set()
        threading.Thread(target=self._run_thinking_sound_loop, daemon=True).start()

        try:
            prompt = self.build_prompt(text)
            llm_start = time.monotonic()
            full_response_buffer = ""
            pending_prefix_buffer = ""
            stream_mode_decided = False
            suppress_stream_output = False
            started_stream_output = False

            def emit_stream_chunk(content):
                nonlocal started_stream_output
                if not content:
                    return

                if not started_stream_output:
                    self.thinking_sound_active.clear()
                    self.set_state(BotStates.SPEAKING, "Speaking...")
                    self.append_to_text("BOT: ", newline=False)
                    started_stream_output = True

                self.append_to_text(content, newline=False)

            def on_stream_content(content):
                nonlocal full_response_buffer
                nonlocal pending_prefix_buffer
                nonlocal stream_mode_decided
                nonlocal suppress_stream_output

                if not content:
                    return

                full_response_buffer += content

                if stream_mode_decided:
                    if not suppress_stream_output:
                        emit_stream_chunk(content)
                    return

                pending_prefix_buffer += content
                probe = pending_prefix_buffer.lstrip().lower()

                # Wait for enough context to detect tool-action JSON before speaking.
                if len(probe) < 24 and not any(
                    ch in pending_prefix_buffer for ch in "\n.!?}"
                ):
                    return

                stream_mode_decided = True
                suppress_stream_output = (
                    probe.startswith("{")
                    or '"action"' in probe[:120]
                    or "action:" in probe[:120]
                )

                if not suppress_stream_output and pending_prefix_buffer:
                    emit_stream_chunk(pending_prefix_buffer)
                pending_prefix_buffer = ""

            full_response_buffer, is_action_mode = self._run_openclaw_execute_stream(
                prompt, on_stream_content
            )

            if not stream_mode_decided:
                probe = full_response_buffer.lstrip().lower()
                suppress_stream_output = (
                    probe.startswith("{")
                    or '"action"' in probe[:120]
                    or "action:" in probe[:120]
                )

            if pending_prefix_buffer and not suppress_stream_output:
                emit_stream_chunk(pending_prefix_buffer)
                pending_prefix_buffer = ""

            print(
                f"[PERF] LLM first response latency: {time.monotonic() - llm_start:.2f}s",
                flush=True,
            )

            if is_action_mode:
                action_data = self.extract_json_from_text(full_response_buffer)
                if action_data:
                    tool_result = self.execute_action_and_get_result(action_data)

                    if tool_result and tool_result.startswith("CHAT_FALLBACK::"):
                        chat_text = tool_result.split("::", 1)[1]
                        self.thinking_sound_active.clear()
                        self.set_state(BotStates.SPEAKING, "Speaking...")
                        self.append_to_text("BOT: ", newline=False)
                        self.append_to_text(chat_text, newline=True)
                        with self.tts_queue_lock:
                            self.tts_queue.append(chat_text)
                        self.session_memory.append(
                            {"role": "assistant", "content": chat_text}
                        )
                        self.wait_for_tts()
                        self.set_state(BotStates.IDLE, "Ready")
                        return

                    if tool_result == "INVALID_ACTION":
                        fallback_text = "I am not sure how to do that."
                        self.thinking_sound_active.clear()
                        self.set_state(BotStates.SPEAKING, "Speaking...")
                        self.append_to_text("BOT: ", newline=False)
                        self.append_to_text(fallback_text, newline=True)
                        with self.tts_queue_lock:
                            self.tts_queue.append(fallback_text)

                    elif tool_result == "SEARCH_EMPTY":
                        fallback_text = (
                            "I searched, but I could not find any news about that."
                        )
                        self.thinking_sound_active.clear()
                        self.set_state(BotStates.SPEAKING, "Speaking...")
                        self.append_to_text("BOT: ", newline=False)
                        self.append_to_text(fallback_text, newline=True)
                        with self.tts_queue_lock:
                            self.tts_queue.append(fallback_text)

                    elif tool_result == "SEARCH_ERROR":
                        fallback_text = "I cannot reach the internet right now."
                        self.thinking_sound_active.clear()
                        self.set_state(BotStates.SPEAKING, "Speaking...")
                        self.append_to_text("BOT: ", newline=False)
                        self.append_to_text(fallback_text, newline=True)
                        with self.tts_queue_lock:
                            self.tts_queue.append(fallback_text)

                    elif tool_result:
                        self.thinking_sound_active.clear()
                        self.set_state(BotStates.SPEAKING, "Speaking...")

                        self.append_to_text("BOT: ", newline=False)
                        self.append_to_text(tool_result, newline=True)
                        with self.tts_queue_lock:
                            self.tts_queue.append(tool_result)
                        self.session_memory.append(
                            {"role": "assistant", "content": tool_result}
                        )
            else:
                if not started_stream_output:
                    self.thinking_sound_active.clear()
                    self.set_state(BotStates.SPEAKING, "Speaking...")
                    self.append_to_text("BOT: ", newline=False)
                    self.append_to_text(full_response_buffer, newline=True)
                else:
                    self.append_to_text("")
                self.session_memory.append(
                    {"role": "assistant", "content": full_response_buffer}
                )

            if not is_action_mode and full_response_buffer.strip():
                with self.tts_queue_lock:
                    self.tts_queue.append(full_response_buffer.strip())

            self.wait_for_tts()
            print(
                f"[PERF] End-to-end turn latency (including TTS): {time.monotonic() - turn_start:.2f}s",
                flush=True,
            )
            self.set_state(BotStates.IDLE, "Ready")

        except Exception as e:
            print(f"LLM Error: {e}")
            self.set_state(BotStates.ERROR, "Brain Freeze!")

    def wait_for_tts(self):
        while self.tts_queue or self.tts_active.is_set():
            time.sleep(0.1)

    def _tts_worker(self):
        while True:
            text = None
            with self.tts_queue_lock:
                if self.tts_queue:
                    text = self.tts_queue.pop(0)
            if text:
                self.tts_active.set()
                try:
                    self.speak(text)
                finally:
                    self.tts_active.clear()
            else:
                self.tts_active.clear()
                time.sleep(0.05)

    def _speak_piper_utterance(self, clean):
        if not clean:
            return
        if not self.voice_model_path:
            print(
                self.voice_model_error or "Audio Error: voice model missing.",
                flush=True,
            )
            return

        try:
            self.current_audio_process = subprocess.Popen(
                [
                    "./piper/piper",
                    "--model",
                    self.voice_model_path,
                    "--sentence_silence",
                    str(self.piper_sentence_silence_s),
                    "--output-raw",
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )

            self.current_audio_process.stdin.write(clean.encode("utf-8") + b"\n")
            self.current_audio_process.stdin.close()

            with sd.RawOutputStream(
                samplerate=self.piper_stream_rate,
                channels=1,
                dtype="int16",
                device=self.output_device,
                latency="low",
                blocksize=2048,
            ) as stream:
                logged_start = False
                while True:
                    data = self.current_audio_process.stdout.read(4096)
                    if not data:
                        break

                    audio_chunk = np.frombuffer(data, dtype=np.int16)
                    if len(audio_chunk) > 0:
                        if not logged_start:
                            print(f"[PIPER SPEAKING] '{clean}'", flush=True)
                            logged_start = True
                        self.current_volume = np.max(np.abs(audio_chunk))
                        if self.piper_needs_resample:
                            num_samples = int(
                                len(audio_chunk) * (self.output_native_rate / 22050)
                            )
                            audio_chunk = scipy.signal.resample(
                                audio_chunk, num_samples
                            ).astype(np.int16)
                        stream.write(audio_chunk.tobytes())
                    else:
                        self.current_volume = 0

        except Exception as e:
            print(f"Audio Error: {e}")
        finally:
            self.current_volume = 0
            self.last_audio_output_end = time.monotonic()
            if self.current_audio_process:
                if self.current_audio_process.stdout:
                    self.current_audio_process.stdout.close()
                if self.current_audio_process.poll() is None:
                    self.current_audio_process.terminate()
                self.current_audio_process = None

    def speak(self, text):
        clean = self.normalize_text_for_tts(text)
        if not clean.strip():
            return

        self._speak_piper_utterance(clean)

    def _run_thinking_sound_loop(self):
        time.sleep(0.5)
        while self.thinking_sound_active.is_set():
            sound = self.get_sound(thinking_sounds_dir)
            if sound:
                self.play_sound(sound)
            for _ in range(50):
                if not self.thinking_sound_active.is_set():
                    return
                time.sleep(0.1)

    def get_sound(self, directory):
        if os.path.exists(directory):
            files = [f for f in os.listdir(directory) if f.endswith(".wav")]
            files.sort()
            if not files:
                return None
            next_idx = self.sound_cycle_indices.get(directory, 0) % len(files)
            self.sound_cycle_indices[directory] = (next_idx + 1) % len(files)
            return os.path.join(directory, files[next_idx])
        return None

    def play_sound(self, file_path):
        if not file_path or not os.path.exists(file_path):
            return
        try:
            with wave.open(file_path, "rb") as wf:
                file_sr = wf.getframerate()
                data = wf.readframes(wf.getnframes())
                audio = np.frombuffer(data, dtype=np.int16)

            try:
                if self.output_device is not None:
                    device_info = sd.query_devices(self.output_device)
                else:
                    device_info = sd.query_devices(kind="output")
                native_rate = int(device_info["default_samplerate"])
            except Exception:
                native_rate = 48000

            playback_rate = file_sr
            try:
                sd.check_output_settings(device=self.output_device, samplerate=file_sr)
            except Exception:
                playback_rate = native_rate
                num_samples = int(len(audio) * (native_rate / file_sr))
                audio = scipy.signal.resample(audio, num_samples).astype(np.int16)

            sd.play(audio, playback_rate, device=self.output_device)
            sd.wait()
            self.last_audio_output_end = time.monotonic()
        except Exception:
            pass

    def load_chat_history(self):
        history = None
        if os.path.exists(MEMORY_FILE):
            try:
                with open(MEMORY_FILE, "r", encoding="utf-8") as f:
                    history = json.load(f)
            except Exception:
                history = None

        if not isinstance(history, list) or not history:
            return [{"role": "system", "content": SYSTEM_PROMPT}]

        if history[0].get("role") != "system":
            history.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
        else:
            history[0]["content"] = SYSTEM_PROMPT

        return history

    def save_chat_history(self):
        full = self.permanent_memory + self.session_memory
        conv = full[1:]
        if len(conv) > 10:
            conv = conv[-10:]
        with open(MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump([full[0]] + conv, f, indent=4)


if __name__ == "__main__":
    print("--- SYSTEM STARTING (HEADLESS MODE) ---", flush=True)
    bot = HeadlessBot()
    bot.run()
