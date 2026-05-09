#!/bin/bash

# Define colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}🤖 Pi Local Assistant Setup Script${NC}"

# 1. Install System Dependencies (The "Hidden" Requirements)
echo -e "${YELLOW}[1/7] Installing System Tools (apt)...${NC}"
sudo apt update
sudo apt install -y libasound2-dev libportaudio2 libatlas-base-dev cmake build-essential espeak-ng git

# 2. Create Folders
echo -e "${YELLOW}[2/7] Creating Folders...${NC}"
mkdir -p piper
mkdir -p sounds/greeting_sounds
mkdir -p sounds/thinking_sounds
mkdir -p sounds/ack_sounds
mkdir -p sounds/error_sounds

# 3. Download Piper (Architecture Check)
echo -e "${YELLOW}[3/7] Setting up Piper TTS...${NC}"
ARCH=$(uname -m)
if [ "$ARCH" == "aarch64" ]; then
    # FIXED: Using the specific 2023.11.14-2 release known to work on Pi
    wget -O piper.tar.gz https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_linux_aarch64.tar.gz
    tar -xvf piper.tar.gz -C piper --strip-components=1
    rm piper.tar.gz
else
    echo -e "${RED}⚠️  Not on Raspberry Pi (aarch64). Skipping Piper download.${NC}"
fi

# 4. Download Voice Model
echo -e "${YELLOW}[4/7] Downloading Voice Model...${NC}"
cd piper
if [ ! -s "en_GB-semaine-medium.onnx" ]; then
    echo -e "${YELLOW}Downloading voice model (.onnx)...${NC}"
    wget -O en_GB-semaine-medium.onnx.tmp https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_GB/semaine/medium/en_GB-semaine-medium.onnx && mv en_GB-semaine-medium.onnx.tmp en_GB-semaine-medium.onnx
else
    echo -e "${GREEN}Voice model .onnx already present.${NC}"
fi

if [ ! -s "en_GB-semaine-medium.onnx.json" ]; then
    echo -e "${YELLOW}Downloading voice model config (.json)...${NC}"
    wget -O en_GB-semaine-medium.onnx.json.tmp https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_GB/semaine/medium/en_GB-semaine-medium.onnx.json && mv en_GB-semaine-medium.onnx.json.tmp en_GB-semaine-medium.onnx.json
else
    echo -e "${GREEN}Voice model .json already present.${NC}"
fi

if [ -f "en_GB-semaine-medium.onnx.tmp" ]; then rm -f en_GB-semaine-medium.onnx.tmp; fi
if [ -f "en_GB-semaine-medium.onnx.json.tmp" ]; then rm -f en_GB-semaine-medium.onnx.json.tmp; fi
cd ..

# 5. Install Python Libraries
echo -e "${YELLOW}[5/7] Installing Python Libraries...${NC}"
# Check if venv exists, if not create it
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# 6. Verify OpenClaw
echo -e "${YELLOW}[6/7] Verifying OpenClaw...${NC}"
if command -v openclaw &> /dev/null; then
    openclaw models status --json
else
    echo -e "${RED}❌ OpenClaw not found. Please install it manually.${NC}"
fi

# 7. Setup Whisper.cpp (Speech-to-Text)
echo -e "${YELLOW}[7/7] Setting up Whisper.cpp...${NC}"
if [ ! -d "whisper.cpp" ]; then
    git clone https://github.com/ggerganov/whisper.cpp.git
fi
cd whisper.cpp
cmake -B build
cmake --build build -j"$(nproc)"
if [ ! -s "models/ggml-base.en.bin" ]; then
    ./models/download-ggml-model.sh base.en
fi
cd ..

# 7. OpenWakeWord Model (Added this back so the user has a default)
if [ ! -f "wakeword.onnx" ]; then
    echo -e "${YELLOW}Downloading default 'Hey Jarvis' wake word...${NC}"
    WAKEWORD_URL_PRIMARY="https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/hey_jarvis_v0.1.onnx"
    WAKEWORD_URL_FALLBACK="https://github.com/dscripka/openwakeword/releases/download/v0.5.1/hey_jarvis_v0.1.onnx"

    if ! wget -O wakeword.onnx.tmp "$WAKEWORD_URL_PRIMARY"; then
        echo -e "${YELLOW}Primary wakeword URL failed, trying fallback...${NC}"
        wget -O wakeword.onnx.tmp "$WAKEWORD_URL_FALLBACK"
    fi

    if [ -s "wakeword.onnx.tmp" ]; then
        mv wakeword.onnx.tmp wakeword.onnx
    else
        rm -f wakeword.onnx.tmp
        echo -e "${RED}❌ Failed to download wakeword model.${NC}"
    fi
fi

echo -e "${GREEN}✨ Setup Complete! Run 'source venv/bin/activate' then 'python agent.py'${NC}"
