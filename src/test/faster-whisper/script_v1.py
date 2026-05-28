from faster_whisper import WhisperModel

model = WhisperModel("base", device="cpu")
audio_path = "./temp/test.m4a"
segments, info = model.transcribe(audio_path)
transcription = ''

for segment in segments:
   transcription += str(segment.text) + '\n'

print(transcription)