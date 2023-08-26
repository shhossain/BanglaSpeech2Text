# Bangla Speech to Text

BanglaSpeech2Text: An open-source offline speech-to-text package for Bangla language. Fine-tuned on the latest whisper speech to text model for optimal performance. Transcribe speech to text, convert voice to text and perform speech recognition in python with ease, even without internet connection.

## Models

| Model   | Size       | Best(WER) |
| ------- | ---------- | --------- |
| `tiny`  |100-200 MB | 60        |
| `base`  |200-300 MB | 46        |
| `small` |1 GB       | 18        |
| `large` |3-4 GB     | 11        |

**NOTE**: Bigger model have better accuracy but slower inference speed. More models [HuggingFace Model Hub](https://huggingface.co/models?pipeline_tag=automatic-speech-recognition&language=bn&sort=likes)

## Pre-requisites

- Python 3.7 or higher

## Test it in Google Colab

- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shhossain/BanglaSpeech2Text/blob/main/BanglaSpeech2Text_in_Colab.ipynb)

## Installation

You can install the library using pip:

```bash
pip install banglaspeech2text
```

## Usage

### Model Initialization

To use the library, you need to initialize the Speech2Text class with the desired model. By default, it uses the "base" model, but you can choose from different pre-trained models: "tiny", "small", "medium", "base", or "large". Here's an example:

```python
from banglaspeech2text import Speech2Text

stt = Speech2Text(model="base")

# You can use it wihout specifying model name (default model is "base")
stt = Speech2Text()
```

### Transcribing Audio Files

You can transcribe an audio file by calling the `recognize` method and passing the path to the audio file. It will return the transcribed text as a string. Here's an example:

```python
transcription = stt.recognize("audio.wav")
print(transcription)
```

### For longer audio files (As different models have different max audio length, so you can use the following methods to transcribe longer audio files)

For longer audio files, you can use the `generate` or `recognize` method. Here's an example:

```python
for text in stt.generate("audio.wav"): # it will generate text as the chunks are processed
    print(text)

# or
text = stt.recognize("audio.wav", split=True) # it will use split_on_silence from pydub to split the audio and transcribe it at once
print(text)

# or
# you can pass min_silence_length and silence_threshold to split_on_silence
text = stt.recognize("audio.wav", split=True, min_silence_length=1000, silence_threshold=-16)
print(text)
```

## Multiple Audio Formats

BanglaSpeech2Text supports the following audio formats for input:

- File Formats: WAV, MP3, FLAC, and all formats supported by FFmpeg.
- Bytes: Raw audio data in byte format.
- Numpy Array: Numpy array representing audio data, preferably obtained using librosa.load.
- AudioData: Audio data obtained from the speech_recognition library.
- AudioSegment: Audio segment objects from the pydub library.
- BytesIO: Audio data provided through BytesIO objects from the io module.
- Path: Pathlib Path object pointing to an audio file.

Here's an example:

```python
transcription = stt.recognize("audio.mp3")
print(transcription)
```

### Use with SpeechRecognition

You can use [SpeechRecognition](https://pypi.org/project/SpeechRecognition/) package to get audio from microphone and transcribe it. Here's an example:

```python
import speech_recognition as sr
from banglaspeech2text import Speech2Text

stt = Speech2Text(model="base")

r = sr.Recognizer()
with sr.Microphone() as source:
    print("Say something!")
    r.adjust_for_ambient_noise(source)
    audio = r.listen(source)
    output = stt.recognize(audio)

print(output)
```

### Use GPU

You can use GPU for faster inference. Here's an example:

```python

stt = Speech2Text(model="base",use_gpu=True)

```

### Advanced GPU Usage

For more advanced GPU usage you can use `device` or `device_map` parameter. Here's an example:

```python
stt = Speech2Text(model="base",device="cuda:0")
```

```python
stt = Speech2Text(model="base",device_map="auto")
```

**NOTE**: Read more about [Pytorch Device](https://pytorch.org/docs/stable/tensor_attributes.html#torch.torch.device)

### Instantly Check with gradio

You can instantly check the model with gradio. Here's an example:

```python
from banglaspeech2text import Speech2Text, available_models
import gradio as gr

stt = Speech2Text(model="base",use_gpu=True)

# You can also open the url and check it in mobile
gr.Interface(
    fn=stt.recognize,
    inputs=gr.Audio(source="microphone", type="filepath"),
    outputs="text").launch(share=True)
```

## Some more usage examples

### Use huggingface model

```python
stt = Speech2Text(model="openai/whisper-tiny")
```

### Change Model Save location

```python
stt = Speech2Text(model="base",cache_path="path/to/save/model")
```

### See current model info

```python
stt = Speech2Text(model="base")

print(stt.model_name) # the name of the model
print(stt.model_size) # the size of the model
print(stt.model_license) # the license of the model
print(stt.model_description) # the description of the model(in .md format)
print(stt.model_url) # the url of the model
print(stt.model_wer) # word error rate of the model
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
  -gpu
    use gpu
  -c CACHE, --cache CACHE
    cache directory
  -o OUTPUT, --output OUTPUT
    output directory
  -m MODEL, --model MODEL
    model name
  -s, --split
    split audio file using pydub split_on_silence
  -sm MIN_SILENCE_LENGTH, --min_silence_length MIN_SILENCE_LENGTH Minimum length of silence to split on (in ms)
  -st SILENCE_THRESH, --silence_thresh SILENCE_THRESH dBFS below reference to be considered silence
  -sp PADDING, --padding PADDING Padding to add to beginning and end of each split (in ms)
  --list list of available models
  --info show model info
```

## Custom Use Cases and Support

If your business or project has specific speech-to-text requirements that go beyond the capabilities of the provided open-source package, I'm here to help! I understand that each use case is unique, and I'm open to collaborating on custom solutions that meet your needs. Whether you have longer audio files that need accurate transcription, require model fine-tuning, or need assistance in implementing the package effectively, I'm available for support and consultation.

Feel free to reach out to me with your custom use cases, questions, or ideas at [hossain0338@gmail.com]. I'm excited to work with you and explore how BanglaSpeech2Text can be tailored to create the best possible solution for your requirements.
