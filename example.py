from banglaspeech2text import available_models, Model


models = available_models(True)
file_path = r"F:\Code\Python\rodela_bot\app\data\উদ্দেশ্য_licence.wav"

model = Model(models[0], verbose=True, download_path="models")
model.load()
# model.cache_file = False
print(model)
print(model(file_path))
