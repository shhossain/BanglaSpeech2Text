from banglaspeech2text import Speech2Text

# Let's see what models are available
available_models = Speech2Text.list_models()
print(available_models)

# Now Let's create an instance of Speech2Text
# I am using the base model here. You can use any model you want. tiny < base < small < medium < large < xl
s2t = Speech2Text(model="tiny")

# Now let's transcribe a file
transcription = s2t.recognize("test.wav")

# Let's see what we got
print(transcription)

# Long audio files can be transcribed using the generate method
for result in s2t.generate("test2.wav"):
    print(result)



print("Hurray! We are done!")
