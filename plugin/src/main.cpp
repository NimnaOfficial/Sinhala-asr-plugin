#include <iostream>
#include <vector>
#include "../include/audio_ring_buffer.hpp"
#include "../include/inference_engine.hpp"
#include "../include/lua_bridge.hpp"

int main(int argc, char** argv) {
    (void)argc;
    (void)argv;

    std::cout << "====================================================\n";
    std::cout << "Sinhala ASR & Translation Plugin Engine (C++20/Lua) \n";
    std::cout << "====================================================\n";

    // 1. Initialize Lock-Free Audio Ring Buffer
    sinhala_asr::LockFreeAudioRingBuffer ring_buffer(1024);
    std::cout << "[Core] Initialized Lock-Free SPSC Audio Ring Buffer.\n";

    // Simulate pushing an audio frame from the audio thread
    std::vector<float> dummy_pcm(512, 0.05f);
    bool pushed = ring_buffer.push_frame(dummy_pcm.data(), dummy_pcm.size(), 0.0);
    std::cout << "[Host Thread] Pushed 512 PCM samples: " << (pushed ? "SUCCESS" : "FAILED") << "\n";
    std::cout << "[Host Thread] Approximate queue size: " << ring_buffer.size_approx() << "\n";

    // 2. Initialize LuaJIT state and register C++ bindings
    sol::state lua;
    lua.open_libraries(sol::lib::base, sol::lib::math, sol::lib::string, sol::lib::table);
    sinhala_asr::register_sinhala_bindings(lua);
    std::cout << "[Lua Bridge] Registered C++ types with Sol2.\n";

    // 3. Execute Lua orchestration script
    const std::string test_lua_script = R"(
        print("[Lua Runtime] Running inside Lua sandbox.")
        local config = InferenceConfig.new()
        config.model_path = "models/onnx/sinhala_whisper_int8.onnx"
        config.enable_translation = true
        print("[Lua Runtime] Configured model path: " .. config.model_path)
        print("[Lua Runtime] Translation mode enabled: " .. tostring(config.enable_translation))
    )";

    sol::protected_function_result result = lua.safe_script(test_lua_script);
    if (!result.valid()) {
        sol::error err = result;
        std::cerr << "[Lua Error] " << err.what() << "\n";
        return 1;
    }

    std::cout << "[Core] Engine harness operational!\n";
    return 0;
}
