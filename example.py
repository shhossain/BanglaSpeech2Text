from banglaspeech2text import available_models, Model


models = available_models()

model = Model(models[0], verbose=True)
model.load()

print(model)
