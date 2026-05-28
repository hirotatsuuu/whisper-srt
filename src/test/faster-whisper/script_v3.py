from faster_whisper import WhisperModel
from pathlib import Path
from datetime import timedelta
from typing import List
from dataclasses import dataclass
import srt
import time


@dataclass
class Word:
    word: str
    start: float
    end: float

def transcribe_audio(model: WhisperModel, audio_file_path: Path, language: str) -> List[Word]:
    transcribe_args = {
        "audio": audio_file_path,
        "language": language,
        "beam_size": 5,
        "word_timestamps": True,
    }
    segments, info = model.transcribe(**transcribe_args)
    return [
        Word(word=word.word, start=word.start, end=word.end)
        for segment in segments
        for word in segment.words
    ]


def create_subtitle(index: int, start: float, end: float, content: str) -> srt.Subtitle:
    return srt.Subtitle(
        index=index,
        start=timedelta(seconds=start),
        end=timedelta(seconds=end),
        content=content.strip(),
    )


def generate_srt_segments(
    words: List[Word],
    char_num: int,
    max_line_str_num: int,
    gap_seconds_threshold: int,
) -> List[srt.Subtitle]:
    srt_segments = []
    current_segment = ""
    segment_start = None
    segment_end = None
    segment_index = 1

    for word in words:
        if segment_start is None:
            segment_start = word.start
        elif (
            word.start - segment_end > gap_seconds_threshold
            or len(current_segment) + len(word.word) > char_num
        ):
            srt_segments.append(
                create_subtitle(
                    segment_index,
                    segment_start,
                    segment_end,
                    "\n".join(
                        [
                            current_segment[i : i + max_line_str_num]
                            for i in range(0, len(current_segment), max_line_str_num)
                        ]
                    ),
                )
            )
            segment_index += 1
            current_segment = ""
            segment_start = word.start

        current_segment += word.word
        segment_end = word.end
    if current_segment:
        srt_segments.append(
            create_subtitle(segment_index, segment_start, segment_end, current_segment)
        )

    return srt_segments

def main(
    audio_file_path: Path,
    model: WhisperModel,
    char_num: int,
    max_line_str_num: int,
    gap_seconds_threshold: int,
    language: str,
):
    words = transcribe_audio(model, audio_file_path, language)

    srt_segments = generate_srt_segments(
        words, char_num, max_line_str_num, gap_seconds_threshold
    )

    srt_content = srt.compose(srt_segments)

    output = audio_file_path.with_suffix(".srt")
    with open(output, "w", encoding="utf-8") as srt_file:
        srt_file.write(srt_content)

    print(f"srtファイル出力完了: {output}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Whisperモデルを使用して音声を文字起こしする"
    )
    parser.add_argument(
        "audio_file", help="文字起こしする音声ファイルのパス")
    parser.add_argument(
        "--model", default="base", help="使用するWhisperモデル (tiny/small/base/medium/large)",
    )
    parser.add_argument(
        "--device", default="cpu", help="計算に使用するデバイス (cpu/cuda)"
    )
    parser.add_argument(
        "--compute_type", default="int8", help="モデルの計算タイプ (int8/float16)",
    )
    parser.add_argument(
        "--language", default="ja", help="文字起こしに使用する言語 (ja: 日本語)",
    )
    parser.add_argument(
        "--char_num", type=int, default=20, help="1行あたりの最大文字数",
    )
    parser.add_argument(
        "--max_line_str_num", type=int, default=20, help="最大行数",
    )
    parser.add_argument(
        "--gap_seconds_threshold", type=int, default=3, help="セグメント間の最大間隔",
    )

    # 処理時間の計測開始
    start = time.perf_counter()

    args = parser.parse_args()

    audio_file_path = Path(args.audio_file)

    model = WhisperModel(args.model, device=args.device, compute_type=args.compute_type)

    main(audio_file_path, model, args.char_num, args.max_line_str_num, args.gap_seconds_threshold, language=args.language)

    # 処理時間の計測終了
    end = time.perf_counter()
    print("処理時間(秒数):", '{:.2f}'.format((end-start)))
