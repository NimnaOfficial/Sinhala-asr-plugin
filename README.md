> [!IMPORTANT]
> **Restricted repository - view-only access.**
> This project is proprietary and is not open source. Viewing is permitted only to users explicitly authorized by NimnaOfficial. Copying, downloading, cloning, forking, scraping, modifying, building, executing, redistributing, or using any part of this repository is prohibited without prior written permission. See [LICENSE](LICENSE).# Sinhala ASR & Video Captioning Plugin

A high-performance, real-time Sinhala Automatic Speech Recognition (ASR) and Speech-to-Text Translation (STTT) engine with a native C++20 and Lua plugin architecture for video editors (Adobe After Effects AEGP, DaVinci Resolve OpenFX) and audio suites (DAWs, OBS).

---

## Architecture Overview

* **Pillar 1: Data Acquisition & Preprocessing:**
  * Unicode NFC & ZWJ canonical normalization ([`sinhala_normalizer.py`](scripts/data/sinhala_normalizer.py)).
  * WebRTC Voice Activity Detection (VAD) & 16kHz mono audio formatting ([`audio_preprocessor.py`](scripts/data/audio_preprocessor.py)).
  * Automated dataset ingest ([`download_openslr52.py`](scripts/data/download_openslr52.py)).
* **Pillar 2: Custom Sinhala Tokenizer & LoRA Training:**
  * Custom 4,096-token Unigram vocabulary ([`train_tokenizer.py`](scripts/tokenizer/train_tokenizer.py)), eliminating the 14-tokens-per-word fragmentation problem in standard Whisper BPE.
  * LoRA parameter-efficient fine-tuning on Whisper-v3 / Whisper Turbo with negative-only training for silence hallucination suppression ([`train_lora_whisper.py`](scripts/training/train_lora_whisper.py)).
* **Pillar 3 & 4: C++20 & Lua Plugin Engine:**
  * Lock-free Single-Producer Single-Consumer (SPSC) queue via `moodycamel::ReaderWriterQueue` with `alignas(64)` cache-line separation.
  * Zero-copy buffer pointer passing between C++ and LuaJIT via `sol2`.
  * ONNX Runtime INT8/INT4 quantization with paged KV-cache support.

---

## Directory Structure

```
├── ARCHITECTURE_SPEC.md                  # Comprehensive architectural specification
├── data/
│   ├── raw/                             # Raw audio datasets (OpenSLR-52, SinhaSpeech)
│   ├── processed/                       # Standardized 16kHz mono PCM data
│   └── vocab/                           # Trained SentencePiece Unigram vocabulary
├── models/
│   ├── checkpoints/                     # LoRA PyTorch checkpoints
│   ├── onnx/                            # Exported ONNX model graphs
│   └── quantized/                       # INT8 / INT4 quantized engines
├── plugin/
│   ├── CMakeLists.txt                   # Cross-platform C++20 & Lua build configuration
│   ├── include/
│   │   ├── audio_ring_buffer.hpp        # Lock-free SPSC audio queue
│   │   ├── inference_engine.hpp         # ONNX Runtime inference interface
│   │   └── lua_bridge.hpp               # Sol2 C++ to Lua bindings
│   ├── src/
│   │   └── main.cpp                     # Plugin engine execution harness
│   ├── lua/                             # Lua orchestration, UI and keyframe scripts
│   └── third_party/                     # Vendored lock-free queue and Sol2 headers
├── scripts/
│   ├── data/                            # Normalizer, audio preprocessor, downloaders
│   ├── tokenizer/                       # Custom 4,096 Unigram tokenizer trainer
│   └── training/                        # LoRA Whisper fine-tuning engine
├── tests/
│   └── test_normalizer.py               # Unit tests for Sinhala Unicode normalizer
└── requirements.txt                     # Python dependencies
```

---

## Getting Started

### 1. Python Environment Setup
```bash
# Activate virtual environment
source venv/bin/activate

# Install dependencies (CPU or CUDA compatible)
pip install -r requirements.txt
```

### 2. Run Normalizer Tests
```bash
python tests/test_normalizer.py
```

### 3. Train or Test Custom Tokenizer
```bash
python scripts/tokenizer/train_tokenizer.py --vocab-size 4096 --output-dir data/vocab
```

### 4. Build and Run the C++ / Lua Engine Harness
```bash
cmake -B plugin/build -S plugin
cmake --build plugin/build -j 4
./plugin/build/sinhala_asr_engine
```