import numpy as np
import torch


class VADChunker:

    SAMPLING_RATE = 16000

    def __init__(self):
        model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',  model='silero_vad')
        _, _, _, VADIterator, _ = utils
        self._vad_iterator = VADIterator(model)
        self.reset_states()

    def reset_states(self):
        self._partial_chunk = []
        self._seek = 0
        self._buffer = []
        self._final_audio = []
        self._last_start = None
        self._vad_iterator.reset_states()

    def input_chunk(self, audio_bytes):
        waveform = torch.tensor(np.frombuffer(audio_bytes, np.int16).flatten().astype(np.float32) / 32768.0)
        waveform = torch.cat(self._partial_chunk + [waveform])

        window_size_samples = 512
        if (len(waveform) % window_size_samples) != 0:
            waveform, self._partial_chunk = waveform[:-(len(waveform) % window_size_samples)], [waveform[-(len(waveform) % window_size_samples):]]
        else:
            self._partial_chunk = []

        for i in range(0, len(waveform), window_size_samples):
            speech_dict = self._vad_iterator(waveform[i: i+window_size_samples])
            self._buffer.append(waveform[i: i+window_size_samples])
            if speech_dict:
                if 'start' in speech_dict:
                    self._last_start = speech_dict['start']
                elif 'end' in speech_dict:
                    buffer = torch.cat(self._buffer)
                    _start = max(0, self._last_start - window_size_samples - self._seek)
                    _end = min(speech_dict['end'] - window_size_samples - self._seek, len(buffer))
                    self._final_audio.append(buffer[_start: _end])
                    self._final_audio.append(torch.zeros(int(self._vad_iterator.min_silence_samples)))
                    self._buffer = [buffer[_end:]]
                    self._seek += _end
                    self._last_start = None

    def output_chunk(self, min_audio_len=3):
        final_audio = self._final_audio
        if min_audio_len is None:
            final_audio += (self._buffer + self._partial_chunk)
        if not final_audio:
            return None
        final_audio = torch.cat(final_audio)
        if min_audio_len and len(final_audio) < min_audio_len*self.SAMPLING_RATE:
            return None
        final_audio = torch.cat([torch.zeros(int(self._vad_iterator.min_silence_samples)), final_audio])
        self._final_audio = []
        return (final_audio.numpy()*32768.0).astype(np.int16).tobytes()
    