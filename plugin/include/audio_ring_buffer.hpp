#pragma once

#include <cstddef>
#include <vector>
#include <memory>
#include <span>
#include "../third_party/moodycamel/readerwriterqueue.h"

namespace sinhala_asr {

/**
 * AudioFrame: Fixed-size audio buffer transferred between audio host and AI thread.
 * Typically 512 or 1024 float32 PCM samples at 16kHz mono.
 */
struct alignas(64) AudioFrame {
    static constexpr size_t FRAME_CAPACITY = 1024;
    float samples[FRAME_CAPACITY];
    size_t count = 0;
    double timestamp_sec = 0.0;
};

/**
 * LockFreeAudioRingBuffer:
 * Implements a wait-free Single-Producer Single-Consumer (SPSC) queue.
 * Guarantees zero blocking and zero memory allocations on the real-time audio processing thread.
 */
class LockFreeAudioRingBuffer {
public:
    explicit LockFreeAudioRingBuffer(size_t max_frames = 2048)
        : queue_(max_frames) {}

    ~LockFreeAudioRingBuffer() = default;

    // Disallow copy/move to maintain atomic pointer integrity
    LockFreeAudioRingBuffer(const LockFreeAudioRingBuffer&) = delete;
    LockFreeAudioRingBuffer& operator=(const LockFreeAudioRingBuffer&) = delete;

    /**
     * Called strictly from real-time Host Audio Thread.
     * Non-blocking, lock-free, zero heap allocation.
     * Returns true if frame was enqueued, false if queue was full (buffer overrun).
     */
    bool push_frame(const float* pcm_data, size_t sample_count, double timestamp_sec) noexcept {
        if (!pcm_data || sample_count > AudioFrame::FRAME_CAPACITY) {
            return false;
        }

        AudioFrame frame;
        frame.count = sample_count;
        frame.timestamp_sec = timestamp_sec;
        for (size_t i = 0; i < sample_count; ++i) {
            frame.samples[i] = pcm_data[i];
        }

        return queue_.try_enqueue(frame);
    }

    /**
     * Called from Asynchronous Inference / Worker Thread.
     * Pops an audio frame if available without locking.
     */
    bool pop_frame(AudioFrame& out_frame) noexcept {
        return queue_.try_dequeue(out_frame);
    }

    /**
     * Approximate frame count in queue.
     */
    size_t size_approx() const noexcept {
        return queue_.size_approx();
    }

private:
    // SPSC lock-free queue with cache-line padded heads and tails
    moodycamel::ReaderWriterQueue<AudioFrame> queue_;
};

} // namespace sinhala_asr
