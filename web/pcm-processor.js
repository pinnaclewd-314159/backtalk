// Runs off the main thread. Buffers nothing itself -- just forwards
// each 128-sample Float32 render quantum to the main thread, which
// accumulates and converts to int16 on release. Keeping this processor
// dumb avoids any GC pressure on the audio thread.
class PCMProcessor extends AudioWorkletProcessor {
  process(inputs) {
    const input = inputs[0];
    if (input && input[0]) {
      this.port.postMessage(input[0].slice());
    }
    return true;
  }
}
registerProcessor("pcm-processor", PCMProcessor);
