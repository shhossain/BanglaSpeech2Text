# BanglaSpeech2Text (Bangla Speech to Text)

BanglaSpeech2Text: An open-source offline speech-to-text package for Bangla language. Fine-tuned on the latest whisper speech to text model for optimal performance. Transcribe speech to text, convert voice to text and perform speech recognition in python with ease, even without internet connection.

## [Models](https://github.com/shhossain/BanglaSpeech2Text/blob/main/banglaspeech2text/utils/listed_models.json)

| Model   | Size       | Best(WER) |
| ------- | ---------- | --------- |
| `tiny`  | 100-200 MB | 74        |
| `base`  | 200-300 MB | 46        |
| `small` | 1 GB       | 18        |
| `large` | 3-4 GB     | 11        |

**NOTE**: Bigger model have better accuracy but slower inference speed. More models [HuggingFace Model Hub](https://huggingface.co/models?pipeline_tag=automatic-speech-recognition&language=bn&sort=likes)

## Pre-requisites

- Python 3.7 or higher

## Test it in Google Colab

- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shhossain/BanglaSpeech2Text/blob/main/banglaspeech2text_in_colab.ipynb)

## Installation

You can install the library using pip:

```bash
pip install banglaspeech2text
```

## Usage

### Model Initialization

To use the library, you need to initialize the Speech2Text class with the desired model. By default, it uses the "base" model, but you can choose from different pre-trained models: "tiny", "small", "base", or "large". Here's an example:

```python
from banglaspeech2text import Speech2Text

stt = Speech2Text("base")

# You can use it wihout specifying model name (default model is "large")
stt = Speech2Text()
```

### Transcribing Audio Files

You can transcribe an audio file by calling the `recognize` method and passing the path to the audio file. It will return the transcribed text as a string. Here's an example:

```python
transcription = stt.recognize("audio.wav")
print(transcription)
```

### Get Transcription as they are processed with time

```python
segments = stt.recognize("audio.wav", return_segments=True)
for segment in segments:
    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
```

## Multiple Audio Formats

BanglaSpeech2Text supports the following audio formats for input:

- File Formats: mp3, mp4, mpeg, mpga, m4a, wav, webm, and more.
- Bytes: Raw audio data in byte format.
- Numpy Array: Numpy array representing audio data, preferably obtained using librosa.load.
- AudioData: Audio data obtained from the speech_recognition library.
- AudioSegment: Audio segment objects from the pydub library.
- BytesIO: Audio data provided through BytesIO objects from the io module.
- Path: Pathlib Path object pointing to an audio file.

No need for extra code to convert audio files to a specific format. BanglaSpeech2Text automatically handles the conversion for you:

```python
transcription = stt.recognize("audio.mp3")
print(transcription)
```

### Use with SpeechRecognition

You can use [SpeechRecognition](https://pypi.org/project/SpeechRecognition/) package to get audio from microphone and transcribe it. Here's an example:

```python
import speech_recognition as sr
from banglaspeech2text import Speech2Text

stt = Speech2Text()

r = sr.Recognizer()
with sr.Microphone() as source:
    print("Say something!")
    r.adjust_for_ambient_noise(source)
    audio = r.listen(source)
    output = stt.recognize(audio)

print(output)
```

### Instantly Check with gradio

You can instantly check the model with gradio. Here's an example:

```python
from banglaspeech2text import Speech2Text, available_models
import gradio as gr

stt = Speech2Text()

# You can also open the url and check it in mobile
gr.Interface(
    fn=stt.recognize,
    inputs=gr.Audio(source="microphone", type="filepath"),
    outputs="text").launch(share=True)
```

## Some more usage examples

### Use huggingface model

```python
stt = Speech2Text("openai/whisper-tiny")
```

### See current model info

```python
stt = Speech2Text("base")

print(stt.model_metadata) # Model metadata (name, size, wer, license, etc.)
print(stt.model_metadata.wer) # Word Error Rate (not available for all models)
```

### CLI

You can use the library from the command line. Here's an example:

```bash
bnstt 'file.wav'
```

You can also use it with microphone:

```bash
bnstt --mic
```

Other options:

```bash
usage: bnstt
       [-h]
       [-gpu]
       [-c CACHE]
       [-o OUTPUT]
       [-m MODEL]
       [-s]
       [-sm MIN_SILENCE_LENGTH]
       [-st SILENCE_THRESH]
       [-sp PADDING]
       [--list]
       [--info]
       [INPUT ...]

Bangla Speech to Text

positional arguments:
  INPUT
    inputfile(s) or list of files

options:
  -h, --help
    show this help message and exit
  -o OUTPUT, --output OUTPUT
    output directory
  -m MODEL, --model MODEL
    model name
  --list list of available models
  --info show model info
```

## Custom Use Cases and Support

If your business or project has specific speech-to-text requirements that go beyond the capabilities of the provided open-source package, I'm here to help! I understand that each use case is unique, and I'm open to collaborating on custom solutions that meet your needs. Whether you have longer audio files that need accurate transcription, require model fine-tuning, or need assistance in implementing the package effectively, I'm available for support.
