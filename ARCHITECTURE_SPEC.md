# **Sinhala ASR and Translation Plugin Architectures**

# **Project Overview**

This specification details the technical architecture for an ultra-low-latency, real-time Sinhala Automatic Speech Recognition (ASR) and Speech-to-Text Translation (STTT) plugin. Designed to integrate natively into professional Digital Audio Workstations (DAWs), broadcast software, and streaming suites, the system delivers high-accuracy transcriptions and translations directly from live PCM audio streams.

## **Core Goals**

1. **Sub-100ms Inference Latency:** Deliver real-time visual streaming of transcribed text with zero audible dropouts on the host audio thread.  
2. **Native Sinhala Script Support:** High-fidelity decoding of complex Sinhala orthography, including accurate handling of conjunct consonants, prenasalized stops, and ligatures.  
3. **Direct Audio-to-English Translation:** Simultaneous transcription of native Sinhala to Sinhala text (Unicode) alongside direct sequence-to-sequence translation to English text.  
4. **Zero-Copy Architecture:** Interprocess audio transfer between the host application (C++) and the execution script/UI engine (LuaJIT) using lock-free ring buffers.

## **Target Accuracy & Performance Metrics**

| Metric | Target Goal | Baseline Standard |
| :---- | :---- | :---- |
| **Word Error Rate (WER) \- Sinhala ASR** | \< 8.5% (Clean Audio) / \< 12.0% (Noisy) | \~22.4% (Vanilla Whisper Large-v3) |
| **BLEU Score \- Sinhala to English STTT** | \> 34.0 | \~19.5 (Vanilla Whisper) |
| **Real-Time Factor (RTF)** | \< 0.15 (1s of audio processed in \< 150ms) | 1.0 (Real-time boundary) |
| **Audio Thread Processing Overhead** | \< 0.5ms per 512-sample buffer frame | Max 2.0ms before sample drop |
| **Memory Footprint** | \< 1.8 GB VRAM / \< 2.2 GB RAM (INT8 Quantized) | 10 GB+ (FP32 Unoptimized) |

---

# **Pillar 1: Data Acquisition & Preprocessing**

## **Handling Sinhala Phonology & Ligatures**

The Sinhala writing system (Sinhala Alphabet) presents specific challenges for tokenizers designed primarily for Latin or Germanic languages. Key linguistic features managed by the data pipeline include:

* **Prenasalized Consonants:** Unique phonemes such as ඟ (*ṅga*), ඬ (*ṇḍa*), ඳ (*nda*), and ඹ (*mba*). These require distinct target tokens to prevent fallback to separate nasal \+ stop sequences (e.g., න් \+ ද).  
* **Vowel Diacritics (Pili):** Diacritics attached to consonants (e.g., ක \+ ි \= කි). The pipeline normalizes unicode representations into standard Canonical Decomposition followed by Canonical Composition (NFC) using `libicu` to eliminate visual-duplicate codepoint variants.  
* **Virama (Al-Lakuna):** The sign (්) that suppresses inherent vowels. Incorrect byte-level splitting often separates the consonant from the virama, creating invalid text sequences. The preprocessing engine enforces grapheme-cluster boundary integrity.  
* **Zero-Width Joiners (ZWJ):** Used to construct conjunct consonants and ligatures (e.g., k \+ ZWJ \+ r \= kr / ක්‍ර්). The tokenizer normalization rules strip invalid ZWJ placements while retaining functional ligatures.

Raw Sinhala Audio \---\> NFC Normalization \---\> Grapheme Boundary Validation \---\> Token Sequences

                          (libicu)                  (ZWJ & Pili Check)

## **Datasets Integration**

Training and fine-tuning leverage a balanced mix of public and proprietary datasets to cover diverse accents, dialects, and recording conditions.                        ┌────────────────────────┐

                        │   OpenSLR-52 Dataset   │

                        │ (Clean Read Speech)    │

                        └───────────┬────────────┘

                                    │

                                    ├───► \[ Data Normalization & Resampling \]

                                    │     \- Mono 16kHz PCM

                                    │     \- Dynamic Range Compression

                        ┌───────────┴────────────┐

                        │   SinhaSpeech Dataset  │

                        │ (Conversational Audio) │

                        └────────────────────────┘

### **OpenSLR-52**

* **Description:** Large-scale read Sinhala speech corpus (\~185 hours).  
* **Usage:** Provides foundational acoustic domain knowledge for formal spoken Sinhala.  
* **Preprocessing:** Stripped of silence via WebRTC VAD, resampled from native rates to 16,000 Hz 16-bit mono PCM. Text scripts are cleaned of non-Sinhala punctuation and validated against the custom NFC normalizer.

### **SinhaSpeech Dataset**

* **Description:** Heterogeneous, real-world conversational Sinhala dataset including multi-speaker dialogues, varying background noise levels, and non-studio microphone responses.  
* **Usage:** Fine-tuning model robustness against background audio, overlapping speech, regional accents, and colloquial phrasing.  
* **Preprocessing:** Augmented with synthetic room impulse responses (RIR) and ambient noise profiles (pub, traffic, office) at SNR levels ranging from \+5dB to \+20dB.

---

# **Pillar 2: AI Model Architecture & Fine-Tuning**

## **Model Selection: Whisper V3 / Whisper Turbo**

The architectural foundation utilizes OpenAI's **Whisper V3** and its latency-optimized variant, **Whisper Turbo**.

* **Encoder:** 32-layer Transformer encoder operating on 80-channel log-mel spectrograms computed over 25ms windows with a 10ms shift.  
* **Decoder:** 32-layer autoregressive Transformer decoder.  
* **Whisper Turbo Adaptation:** Reduces decoder layers from 32 to 4 while maintaining encoder capacity. This configuration targets strict real-time streaming constraints, yielding a 4x improvement in decoding throughput with less than a 3% degradation in WER.

## **Tokenization Strategy: Unigram vs CTC**

Standard Whisper uses a byte-level Byte-Pair Encoding (BPE) tokenizer. For Sinhala, standard BPE fragments single characters into 3 to 6 separate sub-word byte tokens, inflating sequence lengths and triggering severe autoregressive generation latency.Standard BPE (Byte Level):   \[ක\] \[්\] \[ය\] \[ා\]  \---\> 4 Tokens per Grapheme (Slow Decoding)

Optimized Unigram Tokenizer: \[ක්‍යා\]          \---\> 1 Token per Grapheme (Fast Decoding)

To optimize decoding efficiency, the architecture transitions the vocabulary to a target **Unigram Tokenizer** paired with continuous frame modeling.                              ┌──────────────────────────────────────────────┐

                              │           Input Speech Audio (16kHz)         │

                              └──────────────────────┬───────────────────────┘

                                                     │

                                                     ▼

                              ┌──────────────────────────────────────────────┐

                              │            Log-Mel Spectrogram               │

                              └──────────────────────┬───────────────────────┘

                                                     │

                                                     ▼

                              ┌──────────────────────────────────────────────┐

                              │        Whisper Transformer Encoder           │

                              └──────────┬────────────────────────┬──────────┘

                                         │                        │

                    ┌────────────────────┘                        └────────────────────┐

                    ▼                                                                  ▼

┌───────────────────────────────────────┐                          ┌───────────────────────────────────────┐

│     CTC Alignment Decoder Head        │                          │   Autoregressive Unigram Decoder      │

│  (Real-Time Low-Latency Alignment)    │                          │     (High-Accuracy Text Generation)   │

└───────────────────────────────────────┘                          └───────────────────────────────────────┘

* **Unigram Tokenization:** Retrains the tokenizer on Sinhala corpus data to build a custom 4,096-token vocabulary. Graphemes, common pili combinations, and frequent words (e.g., "ස්තූතියි") resolve to single token IDs. This reduces token sequence length by \~68%, yielding a direct **3.1x speedup in decoder forward passes**.  
* **CTC Integration:** A Connectionist Temporal Classification (CTC) head is attached to the top encoder layer for auxiliary alignment. This enables real-time frame-level token emission without requiring full autoregressive decoding passes for every frame.

## **Fine-Tuning Methodology**

### **Parameter-Efficient Fine-Tuning (LoRA)**

To preserve the model's pre-trained multilingual and translation capabilities while minimizing training memory, Low-Rank Adaptation (LoRA) is injected into the Transformer layers.

* **Target Modules:** Query (\$Q\$), Key (\$K\$), Value (\$V\$), and Output (\$O\$) projection matrices within all self-attention and cross-attention blocks.  
* **Hyperparameters:** Rank \$r \= 16\$, Scaling Factor \$\\alpha \= 32\$, Dropout \= \$0.05\$.  
* **Trainable Parameters:** Less than 1.2% of total parameter count (\~18M parameters), enabling rapid domain adaptation and dynamic runtime adapter loading.

### **Negative-Only Training Strategy**

A persistent issue in fine-tuning Whisper on low-resource audio is hallucination during silent or non-speech segments (e.g., background music, studio noise).

* **Mechanism:** 15% of training batches consist exclusively of negative audio samples (background ambient sound, white noise, typing, breath sounds, and non-verbal vocalizations) paired with an explicit empty string `""` target label.  
* **Loss Computation:** Enforces a strong penalty on non-zero token probability distributions during audio frames with no ground-truth text, reducing runtime hallucinations during silent pauses by \>92%.

---

# **Pillar 3: Tech Stack**

The architecture utilizes a dual-layer approach: a high-performance C++20 core engine handling low-level OS audio IO, thread locking, and hardware-accelerated model execution, exposed to a lightweight, sandboxed LuaJIT environment for scripting, dynamic state routing, and UI rendering.┌─────────────────────────────────────────────────────────────────────────────────┐

│ Host Application / DAW Engine                                                  │

└────────────────                        ┬────────────────────────────────────────┘

                                         │ C++ Dynamic API Hooks

                                         ▼

┌─────────────────────────────────────────────────────────────────────────────────┐

│ Native C++20 Core                                                               │

│  ┌───────────────────────┐  ┌──────────────────────┐  ┌──────────────────────┐  │

│  │ Audio Stream Engine   │  │ Moodycamel SPSC      │  │ ONNX / TensorRT      │  │

│  │ (PA / CoreAudio / WASAPI) │  │ Ring Buffer          │  │ Inference Engine     │  │

│  └───────────────────────┘  └──────────────────────┘  └──────────────────────┘  │

└────────────────                        ┬────────────────────────────────────────┘

                                         │ C++ / Lua Bindings via sol2

                                         ▼

┌─────────────────────────────────────────────────────────────────────────────────┐

│ LuaJIT Runtime Environment                                                      │

│  ┌───────────────────────┐  ┌──────────────────────┐  ┌──────────────────────┐  │

│  │ UI Event Loop         │  │ State Machine        │  │ Text Normalizer      │  │

│  └───────────────────────┘  └──────────────────────┘  └──────────────────────┘  │

└─────────────────────────────────────────────────────────────────────────────────┘

## **Core Technologies**

* **C++20 Engine Core:** Handles realtime scheduling, lock-free concurrency, byte packing, and direct GPU buffer orchestration.  
* **LuaJIT \+ sol2:** LuaJIT provides Just-In-Time execution of high-level UI logic and state machine evaluation. `sol2` serves as the binding framework, providing type-safe, zero-cost abstraction interop between C++ objects and Lua scripts.  
* **ONNX Runtime / TensorRT:** Model weights are deployed via ONNX Runtime using execution providers:  
  * **TensorRT Execution Provider:** Target for NVIDIA hardware (CUDA).  
  * **CoreML Execution Provider:** Target for Apple Silicon (Metal Performance Shaders).  
  * **DirectML / OpenVINO Execution Provider:** Fallback for AMD/Intel hardware.

## **Target Host API Compatibility**

The system compiles into dynamic link libraries (`.dll`, `.dylib`, `.so`) targeting standard audio runtime interfaces:

* **VST3 (Steinberg Spec):** Native audio processing thread decoupling via `IAudioProcessor` and `IAudioEditController`.  
* **Audio Units (AUv3):** Apple native audio plugin specification.  
* **OBS Studio Plugin API:** Direct libobs source dynamic linking for streaming integration.

---

# **Pillar 4: System Architecture & Optimization**

## **Real-Time Audio Processing Architecture**

                    HOST AUDIO THREAD (Real-Time Strict)

                                     │

                                     │ 512-sample PCM Frames (16kHz)

                                     ▼

                  ┌──────────────────────────────────────┐

                  │ Lock-Free SPSC Queue (moodycamel)    │

                  └──────────────────┬───────────────────┘

                                     │

                                     │ Low-latency inter-thread transfer

                                     ▼

                    INFERENCE WORKER THREAD (Asynchronous)

                                     │

                                     ▼

                  ┌──────────────────────────────────────┐

                  │ VAD Windowing & Feature Extraction   │

                  │ (80-Channel Log-Mel Calculation)     │

                  └──────────────────┬───────────────────┘

                                     │

                                     ▼

                  ┌──────────────────────────────────────┐

                  │ TensorRT / ONNX Runtime Execution    │

                  │ (INT8 Quantized Model)               │

                  └──────────────────┬───────────────────┘

                                     │

                                     ▼

                    LUA JIT BINDING LAYER (UI / State)

                                     │

                                     ▼

                  ┌──────────────────────────────────────┐

                  │ Real-Time Streaming Text Output      │

                  └──────────────────────────────────────┘

## **Model Quantization (INT8 / INT4)**

To satisfy the low VRAM footprint requirement and increase batch throughput, the full precision model is quantized:

* **Encoder (INT8 Weight & Activation Quantization):** The vision/acoustic encoder is quantized using Per-Tensor Symmetric Quantization for weights and Per-Token Asymmetric Quantization for activations. Calibration uses representative samples from OpenSLR-52 to minimize dynamic range distortion.  
* **Decoder (INT4 AWQ \- Activation-aware Weight Quantization):** Autoregressive operations are memory-bandwidth-bound. Quantizing decoder linear layers to INT4 cuts memory access overhead by 75% while keeping quality degradation under 0.3 BLEU.

## **Key-Value (KV) Cache Optimization**

During dynamic token generation, recalculating self-attention keys and values across long context windows introduces severe processing delays.

* **Paged KV-Cache:** Adapting `vLLM` memory management techniques to the C++ runtime, KV cache allocation is divided into fixed-size physical memory pages (16 tokens per block).  
* **Static Memory Pre-allocation:** At plugin initialization, a fixed contiguous GPU buffer (e.g., 256 MB) is reserved for KV pairs, avoiding runtime calls to `cudaMalloc` or heap allocations on inference worker threads.

## **Lock-Free SPSC Queues**

To guarantee that the host application's audio processing thread is never blocked, all communication between threads uses lock-free Single-Producer Single-Consumer (SPSC) queues.

* **Implementation:** Integrated via `readerwriterqueue` (moodycamel).  
* **Data Flow:**  
  1. The host audio thread pushes raw float32 PCM frames into the SPSC queue.  
  2. The inference worker thread pops frames, maintains a rolling 30-second circular buffer, and runs VAD checks.  
  3. When speech boundaries are detected, the inference thread dispatches feature tensors to the model engine and returns transcribed string offsets to Lua via a second SPSC queue.

---

# **Development Phases**

  Phase 1: Foundation (Weeks 1-4)

  \[ C++/Lua Bridge \] ──► \[ Custom Tokenizer Engine \]

  Phase 2: Data & Fine-Tuning (Weeks 5-10)

  \[ OpenSLR Pipeline \] ──► \[ LoRA Fine-Tuning \] ──► \[ Negative Training \]

  Phase 3: Real-Time Engine (Weeks 11-16)

  \[ ONNX / TensorRT \] ──► \[ Quantization \] ──► \[ SPSC Audio Loop \]

  Phase 4: Integration & UX (Weeks 17-20)

  \[ VST3/AU Host Integration \] ──► \[ Streaming UI Engine \]

## **Phase 1: Foundation & Infrastructure Setup (Weeks 1–4)**

* Establish the core C++20 cross-platform build target (CMake, MSVC, Clang).  
* Embed LuaJIT and integrate `sol2` bindings for memory passing across execution boundaries.  
* Construct the custom Unigram tokenizer module in C++, ensuring compatibility with NFC normalized Sinhala character output.  
* Deliverable: Operational C++/Lua harness capable of executing synthetic inference calls without dropping host thread execution cycles.

## **Phase 2: Data Pipeline & Model Fine-Tuning (Weeks 5–10)**

* Ingest, clean, and convert the OpenSLR-52 and SinhaSpeech datasets into standardized 16kHz mono audio tensors.  
* Apply NFC unicode normalization rules and train the 4,096-token Unigram vocabulary on the text transcripts.  
* Execute LoRA fine-tuning on Whisper V3 / Turbo backbones using target dynamic data augmentations and negative-only audio noise insertion.  
* Deliverable: Fine-tuned PyTorch checkpoint achieving target WER (\<8.5%) on clean validation sets.

## **Phase 3: Hardware Acceleration & Pipeline Optimization (Weeks 11–16)**

* Export trained PyTorch weights to ONNX graphs; optimize network graph structures (operator fusion, LayerNorm folding).  
* Implement INT8/INT4 dynamic quantization scripts via TensorRT API / ONNX Runtime.  
* Wire up `moodycamel::ReaderWriterQueue` for lock-free audio ingest from system drivers.  
* Implement pre-allocated, paged KV-cache logic within C++ model runtime wrappers.  
* Deliverable: Fully optimized local inference engine delivering RTF \< 0.15 on target hardware.

## **Phase 4: Host Integration, UI, & Release Candidate (Weeks 17–20)**

* Wrap C++ runtime engine into VST3, AUv3, and OBS plugin binaries.  
* Design and implement real-time streaming text overlay UI in LuaJIT, including user control panel for switching between Sinhala ASR and English translation modes.  
* Perform stress-testing across major audio hosts (DaVinci Resolve, Reaper, OBS Studio, Ableton Live) to verify multi-hour stability, zero buffer dropouts, and strict memory stability.  
* Deliverable: Production-ready installer packages for macOS and Windows platforms.

