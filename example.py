from banglaspeech2text import Speech2Text
from banglaspeech2text.cli import use_mic

models = Speech2Text.list_models()
print(models)

stt = Speech2Text("large")  # tiny, base, small

# Use with file
path = "path/to/audio.wav"
text = stt.recognize(path)

# Get segments
segments = stt.recognize(path, return_segments=True)
for segment in segments:
    print(segment.text)  # segments also have start and end time

# Use with microphone programatically (only for testing)
use_mic(stt)
