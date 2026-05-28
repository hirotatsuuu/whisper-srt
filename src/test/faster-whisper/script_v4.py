from faster_whisper import WhisperModel
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
output = audio_path.with_suffix(".srt")

# whisperを用いて文字起こしをする
model_size = "base" # tiny/small/base/medium/largeから指定する
model = WhisperModel(model_size, device="cpu", compute_type="int8")
segments, info = model.transcribe(audio_path, language="ja")


# 文字数チェック
temp_segments = segments
temp_text = ""
temp_start = ""
temp_end = ""
for i, segment in enumerate(segments, start=1):
    print("現在のsegment", len(segment.text) , segment.text)

    # 20文字以上の時
    if len(segment.text) > 20:
        if temp_segments[i].start == "":
            temp_segments.text = segment
            temp_segments.start = segment.start
            temp_segments.end = segment.end

            temp_text = ""
            temp_start = ""
            temp_end = ""
        else:
            temp_segments.start = temp_start

            temp_segments.text = temp_text
            temp_segments.end = segment.end

            temp_text = ""
            temp_start = ""
            temp_end = ""

    elif len(segment.text) < 10:

# SRT形式にして出力する
output = audio_path.with_suffix(".srt")
with open(output, "w", encoding="utf-8") as srt_file:
    for i, segment in enumerate(segments, start=1):
        start_seconds = datetime.timedelta(seconds=segment.start)
        end_seconds = datetime.timedelta(seconds=segment.end)
        start_time = str(start_seconds)[:11].replace('.', ',')
        end_time = str(end_seconds)[:11].replace('.', ',')
        srt_file.write(f"{i}\n")
        srt_file.write(f"{start_time} --> {end_time}\n")
        srt_file.write(f"{segment.text.strip()}\n\n")


# 処理時間の計測終了
end = time.perf_counter()
print("処理時間(秒数):", '{:.2f}'.format((end-start)))

print("SRTファイル出力:", output)
