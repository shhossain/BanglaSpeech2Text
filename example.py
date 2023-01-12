from banglaspeech2text import available_models, Model
import speech_recognition as sr

models = available_models()
model = models['base'][0]

model = Model('base',download_path=r"F:\Code\Python\BanglaSpeech2Text\models")
model.load()

outtext = model(r"F:\Code\Python\rodela_bot\app\temp\1.wav") 
print(outtext)