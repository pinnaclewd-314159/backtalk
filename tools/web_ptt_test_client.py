"""Standalone test client for web_client.py's WebSocket PTT protocol --
speaks the wire format like a real browser would, without needing one.
Usage: python web_ptt_test_client.py <host> <port> <path/to/test.wav>
"""
import asyncio
import json
import ssl
import sys
import wave

import numpy as np
import websockets


async def main(host, port, wav_path):
    with wave.open(wav_path, "rb") as wf:
        rate = wf.getframerate()
        pcm = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)

    # Self-signed cert -- this test client trusts it deliberately (same
    # LAN-only posture as the rest of this project), not something a
    # real browser would ever do without the one-time click-through.
    ssl_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ssl_ctx.check_hostname = False
    ssl_ctx.verify_mode = ssl.CERT_NONE

    async with websockets.connect(f"wss://{host}:{port}/", ssl=ssl_ctx) as ws:
        await ws.send(json.dumps({"type": "utterance_start", "rate": rate}))
        await ws.send(pcm.tobytes())
        await ws.send(json.dumps({"type": "utterance_end"}))
        print(f"sent {len(pcm)} samples at {rate}Hz, waiting for reply...")

        saw_state = False
        while True:
            msg = await ws.recv()
            if isinstance(msg, str):
                data = json.loads(msg)
                print("<-", data)
                if data.get("type") == "state":
                    saw_state = True
                if data.get("type") == "audio-stop":
                    break
            else:
                print(f"<- binary chunk, {len(msg)} bytes")
        assert saw_state, "expected at least one state push before audio-stop"
        print("web_ptt_test_client: OK")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1], int(sys.argv[2]), sys.argv[3]))
