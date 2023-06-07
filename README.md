# Bangla Speech to Text
BanglaSpeech2Text: An open-source offline speech-to-text package for Bangla language. Fine-tuned on the latest whisper speech to text model for optimal performance. Transcribe speech to text, convert voice to text and perform speech recognition in python with ease, even without internet connection.


## Models
| Model | Size | Best(WER) |
| --- | --- | --- |
| 'tiny' | 100-200 MB | 60 |
| 'base' | 200-300 MB | 46 |
| 'small'| 1 GB     | 18 |
| 'large'| 3-4 GB     | 11 |

__NOTE__: Bigger model have better accuracy but slower inference speed. More models [HuggingFace Model Hub](https://huggingface.co/models?pipeline_tag=automatic-speech-recognition&language=bn&sort=likes)


## Pre-requisites
- Python 3.6+


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
You can transcribe an audio file by calling the transcribe method and passing the path to the audio file. It will return the transcribed text as a string. Here's an example:

```python
transcription = stt.transcribe("audio.wav")
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
__NOTE__: Read more about [Pytorch Device](https://pytorch.org/docs/stable/tensor_attributes.html#torch.torch.device)

### Instantly Check with gradio
You can instantly check the model with gradio. Here's an example:
```python
from banglaspeech2text import Speech2Text, available_models
import gradio as gr

stt = Speech2Text(model="base",use_gpu=True)

# You can also open the url and check it in mobile
gr.Interface(
    fn=stt.transcribe, 
    inputs=gr.Audio(source="microphone", type="filepath"), 
    outputs="text").launch(share=True)
```

## Some more usage examples

### Change Model from huggingface model hub
```python
sst = Speech2Text(model="openai/whisper-tiny")
```
### Change Model Save location
```python
sst = Speech2Text(model="base",cache_path="path/to/save/model")
```
### See current model info
```python
sst = Speech2Text(model="base")

print(sst.model_name) # the name of the model
print(sst.model_size) # the size of the model
print(sst.model_license) # the license of the model
print(sst.model_description) # the description of the model(in .md format)
print(sst.model_url) # the url of the model
```
