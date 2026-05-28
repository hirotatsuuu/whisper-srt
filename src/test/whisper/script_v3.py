import os
import math
import argparse
import time  # 処理時間（perf_counter）を精密に計測するためのモジュール
import subprocess  # 動画から音声をffmpegで切り出すためのモジュール
import sys
from whisper import load_model
import budoux

# BudouXの日本語モデルを読み込み
parser = budoux.load_default_japanese_parser()

# =====================================================================
# ⚙️ 【初期設定エリア】変更したい場合はここを書き換えてください
# =====================================================================
DEFAULT_AUDIO_FILE = "./temp/test.m4a"   # 引数なしで実行した際に自動で読み込まれる既定のファイル
DEFAULT_DICT_FILE  = "dictionary.txt"    # 優先的に認識させたい固有名詞・専門用語を並べたテキストファイル
DEFAULT_MODEL_SIZE = "base"             # 使用するWhisperのモデルサイズ（tiny/base/small/medium/large）

# 字幕（テロップ）として画面に表示させる際の文字数制限
MIN_CHAR_LEN = 10  # 1行の最低文字数
MAX_CHAR_LEN = 20  # 1行の最大文字数（絶対にこの文字数を超えないようにガードします）

# 動画ファイルとして認識する拡張子のリスト
VIDEO_EXTENSIONS = [".mp4", ".mov", ".mkv", ".avi", ".wmv", ".flv", ".webm"]
# =====================================================================

def load_word_dictionary(file_path):
    """外部のテキストファイル（単語辞書）から単語リストを読み込む"""
    if not file_path or not os.path.exists(file_path):
        return []
        
    word_dict = []
    try:
        with open(file_path, "r", encoding="utf-8-sig") as f:
            for line in f:
                word = line.strip()
                if word and not word.startswith("#"):
                    word_dict.append(word)
    except Exception as e:
        print(f"[!] 警告: 単語辞書の読み込み中にエラーが発生しました（処理は続行します）: {e}")
    return word_dict

def format_timestamp(seconds):
    """秒数をSRT形式の時間フォーマット（HH:MM:SS,mmm）に正確に変換"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    milliseconds = int(round((seconds % 1) * 1000))
    
    if milliseconds >= 1000:
        milliseconds -= 1000
        secs += 1
        if secs >= 60:
            secs -= 60
            minutes += 1
            if minutes >= 60:
                minutes -= 60
                hours += 1
                
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"

def extract_audio_from_video(video_path, output_audio_path):
    """ffmpegをバックグラウンドで呼び出し、動画から音声だけを高速・無劣化で抽出する"""
    print(f"[*] 動画ファイルを検出しました。音声を抽出中...: {video_path}")
    
    command = [
        "ffmpeg", "-y", 
        "-i", video_path, 
        "-vn", 
        "-acodec", "copy", 
        output_audio_path
    ]
    
    try:
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        print(f"[✓] 音声の抽出が完了しました: {output_audio_path}")
        return True
    except subprocess.CalledProcessError:
        print("[!] 音声の無劣化抽出に失敗しました。エンコード抽出に切り替えます...")
        fallback_command = ["ffmpeg", "-y", "-i", video_path, "-vn", "-acodec", "aac", output_audio_path]
        try:
            subprocess.run(fallback_command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            print(f"[✓] 音声の抽出（再エンコード）が完了しました: {output_audio_path}")
            return True
        except Exception as e:
            print(f"エラー: ffmpegでの音声抽出に致命的な失敗をしました。ffmpegがインストールされているか確認してください。 {e}")
            return False
    except FileNotFoundError:
        print("エラー: システムに 'ffmpeg' コマンドが見つかりません。READMEの手順に従って ffmpeg をインストールしてください。")
        return False

def process_segment_to_lines(segment, min_len=10, max_len=20):
    """Whisperのセグメントを文字数制限・ルールに応じて精密に分割する"""
    words_data = []
    for w in segment.get("words", []):
        word_text = w["word"]
        words_data.append({
            "text": word_text,
            "start": float(w["start"]),
            "end": float(w["end"])
        })
        
    lines = []               
    current_line_text = ""   
    current_line_start = None
    current_line_end = None  
    
    for w_info in words_data:
        w_text = w_info["text"]
        w_start = w_info["start"]
        w_end = w_info["end"]
        
        has_period = "。" in w_text
        clean_w_text = w_text.replace("、", "").replace("。", "")
        
        if not clean_w_text:
            if has_period and current_line_text:
                lines.append({"text": current_line_text, "start": current_line_start, "end": current_line_end})
                current_line_text = ""
                current_line_start = None
                current_line_end = None
            continue
            
        if len(clean_w_text) > max_len:
            if current_line_text:
                lines.append({"text": current_line_text, "start": current_line_start, "end": current_line_end})
                current_line_text = ""
            
            w_dur = w_end - w_start  
            w_len = len(clean_w_text)
            while len(clean_w_text) > max_len:
                sub_text = clean_w_text[:max_len] 
                sub_start = w_start
                sub_end = w_start + (w_dur * (max_len / w_len))
                lines.append({"text": sub_text, "start": sub_start, "end": sub_end})
                clean_w_text = clean_w_text[max_len:]
                w_start = sub_end
            
            if clean_w_text:
                current_line_text = clean_w_text
                current_line_start = w_start
                current_line_end = w_end
                
        elif len(current_line_text) + len(clean_w_text) > max_len:
            if current_line_text:
                lines.append({"text": current_line_text, "start": current_line_start, "end": current_line_end})
            current_line_text = clean_w_text
            current_line_start = w_start
            current_line_end = w_end
        else:
            if not current_line_text:
                current_line_start = w_start
            current_line_text += clean_w_text
            current_line_end = w_end
            
        if has_period:
            if current_line_text:
                lines.append({"text": current_line_text, "start": current_line_start, "end": current_line_end})
                current_line_text = ""
                current_line_start = None
                current_line_end = None

    if current_line_text:
        lines.append({"text": current_line_text, "start": current_line_start, "end": current_line_end})
        
    return lines

def transcribe_to_custom_srt(audio_path, output_srt_path, dict_file_path=None, model_size="base"):
    """音声ファイルを読み込んでWhisper文字起こしを実行し、カスタムSRTを出力する"""
    print(f"[*] モデル '{model_size}' を読み込み中...")
    
    # 💡 【追加】モデルロード時のメモリ不足エラー等をキャッチ
    try:
        model = load_model(model_size)
    except MemoryError:
        print(f"エラー: パソコンのメモリが不足しているため、モデル '{model_size}' を読み込めませんでした。")
        print("初期設定エリアの DEFAULT_MODEL_SIZE を 'tiny' や 'base' に下げて再試行してください。")
        return False
    except Exception as e:
        print(f"エラー: Whisperモデルの読み込み中に予期せぬエラーが発生しました: {e}")
        return False
    
    prompt_string = ""
    if dict_file_path and os.path.exists(dict_file_path):
        word_dict = load_word_dictionary(dict_file_path)
        if word_dict:
            prompt_string = "。" + "、".join(word_dict) + "。"
            print(f"[*] 【辞書適用】'{dict_file_path}' から単語を読み込みました")

    print(f"[*] 文字起こしを開始します: {audio_path}")
    
    # 💡 【追加】文字起こし実行中のエラーをキャッチ
    try:
        result = model.transcribe(
            audio_path, 
            verbose=None,        
            fp16=False,          
            initial_prompt=prompt_string, 
            language="ja",       
            word_timestamps=True 
        )
    except RuntimeError as e:
        print(f"エラー: 文字起こし処理中にエラーが発生しました（ファイルが破損している可能性があります）: {e}")
        return False
    except Exception as e:
        print(f"エラー: 予期せぬエラーが発生しました: {e}")
        return False
    
    print("[*] SRTファイルを生成中...")
    try:
        srt_index = 1  
        with open(output_srt_path, "w", encoding="utf-8") as f:
            for segment in result["segments"]:
                split_lines = process_segment_to_lines(segment, min_len=MIN_CHAR_LEN, max_len=MAX_CHAR_LEN)
                for line_data in split_lines:
                    line_text = line_data["text"].strip()
                    if not line_text:  
                        continue
                        
                    line_start = line_data["start"]
                    line_end = line_data["end"]
                    
                    f.write(f"{srt_index}\n") 
                    f.write(f"{format_timestamp(line_start)} --> {format_timestamp(line_end)}\n")
                    f.write(f"{line_text}\n\n") 
                    
                    srt_index += 1 
        print(f"[✓] 字幕作成が完了しました！ 保存先: {output_srt_path}")
        return True
    except IOError as e:
        print(f"エラー: SRTファイルの書き込みに失敗しました（ファイルが開いたままになっていませんか？）: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="動画または音声ファイルから10〜20文字に最適化されたSRT字幕を出力するスクリプト")
    
    parser.add_argument("input_file", nargs="?", default=DEFAULT_AUDIO_FILE, help=f"入力ファイル（動画または音声）のパス (デフォルト: {DEFAULT_AUDIO_FILE})")
    parser.add_argument("-d", "--dict", default=DEFAULT_DICT_FILE, help=f"単語リストのパス (デフォルト: {DEFAULT_DICT_FILE})")
    parser.add_argument("-m", "--model", default=DEFAULT_MODEL_SIZE, help=f"モデルサイズ (デフォルト: {DEFAULT_MODEL_SIZE})")

    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        print(f"エラー: 指定されたファイルが見つかりません: {args.input_file}")
        sys.exit(1)

    start_time = time.perf_counter()

    base_path, ext = os.path.splitext(args.input_file)
    ext_lower = ext.lower()

    output_srt_path = base_path + ".srt"
    target_audio_file = args.input_file

    if ext_lower in VIDEO_EXTENSIONS:
        extracted_audio_path = base_path + ".m4a"
        success = extract_audio_from_video(args.input_file, extracted_audio_path)
        if not success:
            print("エラー: 音声抽出に失敗したため処理を中断します。")
            sys.exit(1)
        target_audio_file = extracted_audio_path

    # 文字起こしの実行
    success = transcribe_to_custom_srt(
        audio_path=target_audio_file,
        output_srt_path=output_srt_path,
        dict_file_path=args.dict,
        model_size=args.model
    )
    
    if not success:
        print("エラー: 文字起こし処理が正常に完了しませんでした。")
        sys.exit(1)

    end_time = time.perf_counter()
    print("処理時間(秒数):", '{:.2f}'.format((end_time - start_time)))

if __name__ == "__main__":
    main()