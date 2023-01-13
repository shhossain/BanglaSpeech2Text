from banglaspeech2text import available_models, Model


models = available_models(True)
models['large'][0].show_config()
file_path = r"F:\Code\Python\rodela_bot\app\data\উদ্দেশ্য_licence.wav"

model = Model(models[0], verbose=True)
model.load()

print(model(file_path))
