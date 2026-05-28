from faster_whisper import WhisperModel # type: ignore
import datetime

model_size = "base"
model = WhisperModel(model_size, device="cpu")

audio_path = "./temp/test.m4a"
segments, info = model.transcribe(audio_path, language="ja")

output ="./temp/test.srt"
with open(output, "w", encoding="utf-8") as srt_file:
    for i, segment in enumerate(segments, start=1):
        start_seconds = datetime.timedelta(seconds=segment.start)
        end_seconds = datetime.timedelta(seconds=segment.end)

        start_time = str(start_seconds)[:11].replace('.', ',')
        end_time = str(end_seconds)[:11].replace('.', ',')

        srt_file.write(f"{i}\n")
        srt_file.write(f"{start_time} --> {end_time}\n")
        srt_file.write(f"{segment.text.strip()}\n\n")

print("srtファイル出力完了", output)
