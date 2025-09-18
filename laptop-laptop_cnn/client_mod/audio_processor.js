// audio_processor.js
// AudioWorkletProcessor that accumulates audio and posts fixed-size windows (sliding window + hop).

class AudioProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
        console.log('✅ AudioProcessor initialized on audio thread.');

        this.isRecording = false;

        // continuous buffer of float samples
        this.buffer = new Float32Array(0);

        // sliding window / hop config (seconds => converted to samples)
        this.windowSize = 0;
        this.hopSize = 0;
        this.samplesPerWindow = 0;
        this.samplesPerHop = 0;
        this.currentOffset = 0;

        this.port.onmessage = (event) => {
            if (event.data.type === 'START_SLIDING_WINDOW') {
                this.isRecording = true;
                this.windowSize = event.data.windowSize || 3.0;
                this.hopSize = event.data.hopSize || 1.5;
                this.samplesPerWindow = Math.floor(this.windowSize * sampleRate);
                this.samplesPerHop = Math.floor(this.hopSize * sampleRate);
                this.buffer = new Float32Array(0);
                this.currentOffset = 0;
                console.log(`🎙️ START_SLIDING_WINDOW: window ${this.windowSize}s (${this.samplesPerWindow} samples), hop ${this.hopSize}s (${this.samplesPerHop} samples).`);
            } else if (event.data.type === 'STOP_RECORDING') {
                this.isRecording = false;
                console.log('🛑 STOP_RECORDING requested on audio thread.');

                // If we have at least one full window remaining, send it
                if (this.buffer.length - this.currentOffset >= this.samplesPerWindow) {
                    const chunk = this.buffer.slice(this.currentOffset, this.currentOffset + this.samplesPerWindow);
                    this.port.postMessage({ type: 'AUDIO_CHUNK', data: chunk.buffer }, [chunk.buffer]);
                } else if (this.buffer.length > this.currentOffset) {
                    // send remaining partial buffer as last shorter chunk (optional)
                    const chunk = this.buffer.slice(this.currentOffset);
                    this.port.postMessage({ type: 'AUDIO_CHUNK', data: chunk.buffer }, [chunk.buffer]);
                }
            }
        };
    }

    process(inputs, outputs) {
        if (!this.isRecording) {
            return true;
        }

        const input = inputs[0];
        if (!input || !input[0]) return true;
        const inputChannelData = input[0];

        // append new samples to buffer
        const newBuffer = new Float32Array(this.buffer.length + inputChannelData.length);
        newBuffer.set(this.buffer, 0);
        newBuffer.set(inputChannelData, this.buffer.length);
        this.buffer = newBuffer;

        // while we have at least one full window available, send it and advance offset by hop
        while (this.buffer.length - this.currentOffset >= this.samplesPerWindow) {
            const chunk = this.buffer.slice(this.currentOffset, this.currentOffset + this.samplesPerWindow);
            // transfer ArrayBuffer for performance
            this.port.postMessage({ type: 'AUDIO_CHUNK', data: chunk.buffer }, [chunk.buffer]);
            this.currentOffset += this.samplesPerHop;
            // occasionally trim buffer to avoid unbounded growth
            if (this.currentOffset > 65536) {
                // drop the consumed portion
                this.buffer = this.buffer.slice(this.currentOffset);
                this.currentOffset = 0;
            }
        }

        return true;
    }
}

registerProcessor('audio_processor', AudioProcessor);
console.log('✅ audio_processor registered.');
