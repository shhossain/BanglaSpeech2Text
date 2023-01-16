# Bangla Speech to Text
BanglaSpeech2Text: An open-source offline speech-to-text package for Bangla language. Fine-tuned on the latest whisper speech to text model for optimal performance. Transcribe speech to text, convert voice to text and perform speech recognition in python with ease, even without internet connection.

## Installation
```bash
pip install banglaspeech2text
```

## Models
| Model | Size | Best(WER) |
| --- | --- | --- |
| 'tiny' | 100-200 MB | N/A |
| 'base' | 200-300 MB | 46 |
| 'small'| 2-3 GB     | 18 |
| 'large'| 5-6 GB     | 11 |

__NOTE__: Bigger model have better accuracy but slower inference speed. Smaller wer is better.You can view the models from [here](https://github.com/shhossain/whisper_bangla_models). The size of the mode is an estimate. The actual size may vary.


## Pre-requisites
- Python 3.6+
- Git
- Git LFS

## Test it in Google Colab
- [NoteBook](https://colab.research.google.com/drive/1rj4Jme6qrc8tRaPY3MTuuUc6MEr8We9N?usp=sharing)

## Download Git
## Windows
- Download git from [here](https://git-scm.com/download/win)
- Download git lfs from [here](https://git-lfs.github.com/)

__Note__: Must check git lfs is marked during installation. If not, you can install git lfs from [here](https://git-lfs.github.com/)

## Linux
- [Git](https://git-scm.com/download/linux)
- Git LFS
Ubuntu 16.04
```bash
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs
```
Ubuntu 18.04 and above
```bash
sudo apt-get install git-lfs
```

## Mac
- [Git](https://git-scm.com/download/mac)
- Git LFS
```bash
brew install git-lfs
```

## Download Git with banglaspeech2text
```bash
from banglaspeech2text.utils.install_packages import install_git_windows, install_git_linux

# for windows
install_git_windows()

# for linux
install_git_linux()
```


## Usage

### Download a model
```python
from banglaspeech2text import Model, available_models

# Download a model
models = available_models()
print(models) # see the available models by diffrent people and diffrent sizes

model = models[0] # select a model
model.download() # download the model
```
### Use with file
```python
from banglaspeech2text import Model, available_models

# Load a model
models = available_models()
model = models[0] # select a model
model = Model(model) # load the model
model.load()

# Use with file
file_name = 'test.wav' # .wav, .mp3, mp4, .ogg, etc.
output = model.recognize(file_name)

print(output) # output will be a dict containing text
print(output['text'])
```

### Use with SpeechRecognition
```python
import speech_recognition as sr
from banglaspeech2text import Model, available_models

# Load a model
models = available_models()
model = models[0] # select a model
model = Model(model) # load the model
model.load()


r = sr.Recognizer()
with sr.Microphone() as source:
    print("Say something!")
    audio = r.listen(source)
    output = model.recognize(audio)

print(output) # output will be a dict containing text
print(output['text'])
```

### Use GPU
```python
import speech_recognition as sr
from banglaspeech2text import Model, available_models

# Load a model
models = available_models()
model = models[0] # select a model
model = Model(model,device="gpu") # load the model
model.load()


r = sr.Recognizer()
with sr.Microphone() as source:
    print("Say something!")
    audio = r.listen(source)
    output = model.recognize(audio)

print(output) # output will be a dict containing text
print(output['text'])
```
__NOTE__: This package uses torch as backend. So, you can use any device supported by torch. For more information, see [here](https://pytorch.org/docs/stable/tensor_attributes.html#torch.torch.device). But you need to setup torch for gpu first from [here](https://pytorch.org/get-started/locally/).

### Instantly Check with gradio
from banglaspeech2text import Model, available_models
import gradio as gr

# Load a model
models = available_models()
model = models[0] # select a model
model = Model(model,device="cuda:0") # remove device if you don't want to use gpu.Ex. model = Model(model)
model.load()

def transcribe(audio_file):
  return model(audio_file)['text']

# You can also open the url and check it in mobile
gr.Interface(
    fn=transcribe, 
    inputs=gr.Audio(source="microphone", type="filepath"), 
    outputs="text").launch(share=True)

### Some Methods
```python
from banglaspeech2text import Model, available_models

models = available_models()
print(models[0]) # get first model
print(models['base']) # get base models
print(models['whisper_base_bn_sifat']) # get model by name

# set download path
model = Model(model,download_path=r"F:\Code\Python\BanglaSpeech2Text\models") # default is home directory
model.load()

# directly load a model
model = Model('base')
model.load()
```


