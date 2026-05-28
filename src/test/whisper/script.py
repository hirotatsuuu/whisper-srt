import whisper
from pathlib import Path
import argparse
import datetime
import time

# 処理時間の計測開始
start = time.perf_counter()

# 実行時に音源ファイルを指定する
parser = argparse.ArgumentParser()
parser.add_argument("audio_file")
args = parser.parse_args()
audio_path = Path(args.audio_file)

# whisperを用いて文字起こしをする
model_size = "base" # tiny/small/base/medium/largeから指定する
model = whisper.load_model(model_size, device="cpu")
result  = model.transcribe(str(audio_path), fp16=False)

# SRT形式にして出力する
output = audio_path.with_suffix(".srt")
with open(output, "w", encoding="utf-8") as srt_file:
    for i, segment in enumerate(result["segments"], start=1):
        start_seconds = datetime.timedelta(seconds=segment["start"])
        end_seconds = datetime.timedelta(seconds=segment["end"])
        start_time = str(start_seconds)[:11].replace('.', ',')
        end_time = str(end_seconds)[:11].replace('.', ',')
        srt_file.write(f"{i}\n")
        srt_file.write(f"{start_time} --> {end_time}\n")
        srt_file.write(f"{segment["text"].strip()}\n\n")

# 処理時間の計測終了
end = time.perf_counter()
print("処理時間(秒数):", '{:.2f}'.format((end-start)))

print("SRTファイル出力:", output)