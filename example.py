import os
import random
import wave
import torch
from vad_chunker import VADChunker


SAMPLING_RATE = 16000


def random_chunk_generator(file, min_audio_len=1, max_audio_len=3):
    with wave.open(file, "r") as wave_file:
        full_audio = wave_file.readframes(wave_file.getnframes())

    i = 0
    while i < (len(full_audio) / SAMPLING_RATE / 2):
        random_length = random.randint(min_audio_len, max_audio_len)
        yield full_audio[i * SAMPLING_RATE * 2: (i + random_length) * SAMPLING_RATE * 2]
        i += random_length


def save_audio(file, audio_bytes):
    with wave.open(file, "w") as wave_file:
        wave_file.setsampwidth(2)
        wave_file.setnchannels(1)
        wave_file.setframerate(SAMPLING_RATE)
        wave_file.writeframes(audio_bytes)


if __name__ == '__main__':
    file = 'full_audio.wav'
    torch.hub.download_url_to_file('https://models.silero.ai/vad_models/en.wav', file)

    os.mkdir('inputs')
    os.mkdir('outputs')

    chunker = VADChunker()
    for i, chunk in enumerate(random_chunk_generator(file)):
        save_audio(f"inputs/{i}.wav", chunk)
        chunker.input_chunk(chunk)
        output = chunker.output_chunk()
        if output is not None:
            save_audio(f"outputs/{i}.wav", output)
