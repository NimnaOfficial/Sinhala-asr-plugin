#pragma once

#include "audio_ring_buffer.hpp"
#include "inference_engine.hpp"

#define SOL_ALL_SAFETIES_ON 1
#include <sol/sol.hpp>

namespace sinhala_asr {

/**
 * RegisterSinhalaBindings:
 * Exposes C++ real-time audio structures and AI inference orchestrators
 * to LuaJIT scripts with type safety and zero-copy pointer access.
 */
inline void register_sinhala_bindings(sol::state& lua) {
    // 1. Bind AudioFrame
    lua.new_usertype<AudioFrame>("AudioFrame",
        "count", &AudioFrame::count,
        "timestamp_sec", &AudioFrame::timestamp_sec
    );

    // 2. Bind TranscriptionResult
    lua.new_usertype<TranscriptionResult>("TranscriptionResult",
        "sinhala_text", &TranscriptionResult::text_sinhala,
        "english_text", &TranscriptionResult::text_english_translation,
        "start_time", &TranscriptionResult::start_time_sec,
        "end_time", &TranscriptionResult::end_time_sec,
        "confidence", &TranscriptionResult::confidence,
        "is_final", &TranscriptionResult::is_final
    );

    // 3. Bind InferenceConfig
    lua.new_usertype<InferenceConfig>("InferenceConfig",
        sol::constructors<InferenceConfig()>(),
        "model_path", &InferenceConfig::model_onnx_path,
        "vocab_path", &InferenceConfig::vocab_path,
        "enable_translation", &InferenceConfig::enable_translation,
        "use_gpu", &InferenceConfig::use_gpu,
        "num_threads", &InferenceConfig::num_threads
    );

    // 4. Bind LockFreeAudioRingBuffer
    lua.new_usertype<LockFreeAudioRingBuffer>("AudioRingBuffer",
        sol::constructors<LockFreeAudioRingBuffer(size_t)>(),
        "size_approx", &LockFreeAudioRingBuffer::size_approx
    );
}

} // namespace sinhala_asr
