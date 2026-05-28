import whisper
import os
from pathlib import Path
import argparse
from whisper.utils import get_writer
import time

# 処理時間の計測開始
start = time.perf_counter()

# 実行時に音源ファイルを指定する
parser = argparse.ArgumentParser()
parser.add_argument("audio_file")
args = parser.parse_args()
audio_path = Path(args.audio_file)

# whisperを用いて文字起こしをする
model_size = "medium" # tiny/small/base/medium/large
model = whisper.load_model(model_size, device="cpu")
result  = model.transcribe(str(audio_path), fp16=False, word_timestamps=True)

# SRT形式にして出力する
output = os.path.dirname(audio_path)
srt_writer = get_writer("srt", output)
word_options =  {"max_line_width": 20, "max_line_count": 1}
srt_writer(result, audio_path, word_options)

# 処理時間の計測終了
end = time.perf_counter()
print("処理時間(秒数):", '{:.2f}'.format((end-start)))

print("SRTファイル出力:", audio_path.with_suffix(".srt"))