#pragma once

#include <string>
#include <vector>
#include <memory>
#include <functional>

namespace sinhala_asr {

/**
 * TranscriptionResult: Produced by the background inference engine.
 */
struct TranscriptionResult {
    std::string text_sinhala;
    std::string text_english_translation;
    double start_time_sec = 0.0;
    double end_time_sec = 0.0;
    float confidence = 0.0f;
    bool is_final = false;
};

using TranscriptionCallback = std::function<void(const TranscriptionResult&)>;

/**
 * InferenceConfig: Runtime configuration parameters.
 */
struct InferenceConfig {
    std::string model_onnx_path;
    std::string vocab_path;
    int sample_rate = 16000;
    float silence_threshold_db = -45.0f;
    bool enable_translation = false;
    bool use_gpu = false;
    int num_threads = 4;
};

/**
 * SinhalaInferenceEngine:
 * Coordinates audio chunk windowing, feature extraction (Log-Mel),
 * and ONNX Runtime execution.
 */
class SinhalaInferenceEngine {
public:
    explicit SinhalaInferenceEngine(const InferenceConfig& config)
        : config_(config), is_running_(false) {}

    virtual ~SinhalaInferenceEngine() = default;

    /**
     * Initializes the ONNX model sessions, KV cache buffers, and vocabulary.
     */
    virtual bool initialize() {
        // Stub implementation: In production, initializes Ort::Env & Ort::Session
        return true;
    }

    /**
     * Feeds audio samples into the internal rolling buffer and checks for speech boundaries.
     */
    virtual void process_audio_chunk(const float* samples, size_t count, double timestamp) {
        (void)samples;
        (void)count;
        (void)timestamp;
    }

    /**
     * Registers a callback for real-time text delivery.
     */
    void set_callback(TranscriptionCallback cb) {
        callback_ = std::move(cb);
    }

    const InferenceConfig& get_config() const { return config_; }

protected:
    InferenceConfig config_;
    TranscriptionCallback callback_;
    bool is_running_;
};

} // namespace sinhala_asr
