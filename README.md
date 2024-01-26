# Audio Segmentation using Voice Activity Detection

## Overview

Welcome to the Audio Segmentation project! This repository contains a Python class called `VADChunker` designed for the segmentation of audio data using Voice Activity Detection (VAD). The primary goal of this class is to process randomly fragmented audio bytes and generate audio segments with proper phrases. This segmentation is particularly useful for preprocessing live audio streams before passing them to a transcriber.

The `VADChunker` class leverages a deep-learning model, [silero-vad](https://github.com/snakers4/silero-vad) for VAD to identify and isolate regions of audio that contain speech. By incorporating this class into your audio processing pipeline, you can improve the efficiency and accuracy of downstream tasks such as speech recognition.

## Usage

To integrate the `VADChunker` class into your project, follow these steps:

1. **Clone the Repository:**

```bash
git clone https://github.com/your-username/audio-segmentation-vad.git
```

2. **Import the VADChunker Class:**

```python
from vad_chunker import VADChunker
```

3. **Instantiate the VADChunker Class:**

```python
vad_chunker = VADChunker()
```

4. **Process Audio Bytes:**

```python
vad_chunker.input_chunk(audio_bytes)
segment = vad_chunker.output_chunk(min_audio_len=5)
```

see [example.py](example.py) for a complete example.


## Contributing
If you find any issues or have ideas for improvements, please feel free to contribute! 

## License
This project is licensed under the MIT License. Feel free to use, modify, and distribute the code as per the terms of the license.

## Acknowledgments
[silero-vad](https://github.com/snakers4/silero-vad) for the VAD model.

Happy audio segmentation!







