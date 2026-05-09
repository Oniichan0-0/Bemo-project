# Be More Agent 
**A Customizable, Offline-First AI Agent for Raspberry Pi**

[![Watch the Demo](https://img.youtube.com/vi/l5ggH-YhuAw/maxresdefault.jpg)](https://youtu.be/l5ggH-YhuAw)

![Python](https://img.shields.io/badge/Python-3.9%2B-blue) ![Platform](https://img.shields.io/badge/Platform-Raspberry%20Pi-red) ![License](https://img.shields.io/badge/License-MIT-green)

This project turns a Raspberry Pi into a fully functional, conversational AI agent. It listens for a wake word, processes speech, "thinks" through a local OpenClaw gateway using your configured model, and speaks back with a low-latency neural voice.

**It is designed as a blank canvas:** You can easily swap sound effects and tune behavior to create your own character.

## Features

* **OpenClaw-Powered Intelligence**: Uses **OpenClaw SDK** for model calls and **Whisper.cpp** for Speech-to-Text.
* **Open Source Wake Word**: Wakes up to your custom model using **OpenWakeWord** (Offline & Free). No access keys required.
* **Hardware-Aware Audio**: Automatically detects your microphone's sample rate and resamples audio on the fly to prevent ALSA errors.
* **Smart Web Search**: Uses DuckDuckGo to find real-time news and information when the LLM doesn't know the answer.
* **Fast Text-to-Speech**: Uses **Piper TTS** for low-latency, high-quality voice generation on the Pi.
* **Headless-Friendly**: Runs without any display or camera dependencies.

## 🛠️ Hardware Requirements

* **Raspberry Pi 5** (Recommended) or Pi 4 (4GB RAM minimum)
* USB Microphone & Speaker

---

## Project Structure

```text
be-more-agent/
├── agent.py                   # The main brain script
├── setup.sh                   # Auto-installer script
├── wakeword.onnx              # OpenWakeWord model (The "Ear")
├── config.json                # User settings (Models, Prompt, Hardware)
├── chat_memory.json           # Conversation history
├── requirements.txt           # Python dependencies
├── whisper.cpp/               # Speech-to-Text engine
├── piper/                     # Piper TTS engine & voice models
├── sounds/                    # Sound effects folder
│   ├── greeting_sounds/       # Startup .wav files
│   ├── thinking_sounds/       # Looping .wav files
│   ├── ack_sounds/            # "I heard you" .wav files
│   └── error_sounds/          # Error/Confusion .wav files
```

---

## Installation

### 1. Prerequisites
Ensure your Raspberry Pi OS is up to date.
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install git -y
```

### 2. Install OpenClaw
This agent expects [OpenClaw](https://openclaw.ai) to be installed and configured with your gateway and model defaults.
```bash
openclaw status --json
openclaw models set github-copilot/gpt-4.1
```

### 3. Clone & Setup
```bash
git clone https://github.com/brenpoly/be-more-agent.git
cd be-more-agent
chmod +x setup.sh
./setup.sh
```
*The setup script will install system libraries, create necessary folders, download Piper TTS, and set up the Python virtual environment.*

### 4. Configure the Wake Word
The setup script downloads a default wake word ("Hey Jarvis"). To use your own:
1. Train a model at [OpenWakeWord](https://github.com/dscripka/openWakeWord).
2. Place the `.onnx` file in the root folder.
3. Rename it to `wakeword.onnx`.

### 5. Run the Agent
```bash
source venv/bin/activate
python agent.py
```

---

## Configuration (`config.json`)

You can modify the hardware behavior and personality in `config.json`. The `agent.py` script creates this on the first run if it doesn't exist, but you can create it manually:

```json
{
    "text_model": "github-copilot/gpt-4.1",
    "voice_model": "piper/en_GB-semaine-medium.onnx",
    "openclaw_agent_id": "main",
    "chat_memory": true,
    "system_prompt_extras": "You are a helpful robot assistant. Keep responses short and cute."
}
```

---

##  Customizing Your Character

This software is a generic framework. You can give it a new personality by replacing the assets:

1.  **Sounds:** Put multiple `.wav` files in the `sounds/[category]/` folders. The robot will pick one at random each time (e.g., different "thinking" hums or "error" buzzes).
2.  **Prompt/Behavior:** Adjust `system_prompt_extras` in `config.json` to tune personality and response style.

---

## Troubleshooting

* **"No search library found":** If web search fails, ensure you are in the virtual environment and `duckduckgo-search` is installed via pip.
* **Shutdown Errors:** When you exit the script (Ctrl+C), you might see `Expression 'alsa_snd_pcm_mmap_begin' failed`. **This is normal.** It just means the audio stream was cut off mid-sample. It does not affect the functionality.
* **Audio Glitches:** If the voice sounds fast or slow, the script attempts to auto-detect sample rates. Ensure your `config.json` points to a valid `.onnx` voice model in the `piper/` folder.

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

##  Legal Disclaimer
**"BMO"** and **"Adventure Time"** are trademarks of **Cartoon Network** (Warner Bros. Discovery).

This project is a **fan creation** built for educational and hobbyist purposes only. It is **not** affiliated with, endorsed by, or connected to Cartoon Network or the official Adventure Time brand in any way. The software provided here is a generic agent framework; users are responsible for the assets they load into it.
