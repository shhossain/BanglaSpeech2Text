from banglaspeech2text import Speech2Text

# Let's see what models are available
available_models = Speech2Text.list_models()

# Now Let's create an instance of Speech2Text
# I am using the base model here. You can use any model you want. tiny < base < small < medium < large < xl
s2t = Speech2Text("base") 


# Now let's transcribe a file
transcription = s2t.transcribe("test.wav")

# Let's see what we got
print(transcription)


print("Hurray! We are done!")


