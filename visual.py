pip install spleeter

pip install librosa

import os
import librosa
from spleeter.separator import Separator

def analyze_audio(file_path):
    """
    음악 파일에서 템포와 키를 추출하는 함수.
    :param file_path: 입력 음악 파일 경로
    :return: 템포와 키 정보 (튜플)
    """
    try:
        y, sr = librosa.load(file_path, sr=None)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        key = librosa.tonnetz(y=y, sr=sr).mean(axis=1).argmax()
        key_name = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        key_str = key_name[key % 12]
        return round(tempo), key_str
    except Exception as e:
        print(f"오류 발생: {e}")
        return None, None

def separate_audio_with_metadata(input_file, output_dir):
    """
    음악 파일에서 스템을 분리하고, 파일 이름에 원본 곡 이름, 템포, 키 정보를 포함하는 함수.
    :param input_file: 입력 음악 파일 경로
    :param output_dir: 스템을 저장할 디렉토리 경로
    """
    song_name = os.path.splitext(os.path.basename(input_file))[0]
    
    print("원본 음악에서 템포와 키 분석 중...")
    tempo, key = analyze_audio(input_file)
    if tempo is None or key is None:
        print("템포 또는 키를 분석할 수 없습니다. 기본 이름으로 저장합니다.")
        tempo, key = "Unknown", "Unknown"
    
    separator = Separator('spleeter:5stems')
    
    print(f"Processing: {input_file}")
    separator.separate_to_file(audio_descriptor=input_file, destination=output_dir)
    
    for root, _, files in os.walk(output_dir):
        for file in files:
            if file.endswith('.wav'):
                original_path = os.path.join(root, file)
                new_file_name = f"{song_name}_{file.split('.')[0]}_tempo{tempo}_key{key}.wav"
                new_path = os.path.join(root, new_file_name)
                os.rename(original_path, new_path)
    print(f"Processed and saved to {output_dir}")

if __name__ == "__main__":
    input_music_file = input("음악 파일 경로를 입력하세요: ")
    output_directory = input("출력 디렉토리 경로를 입력하세요: ")

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    separate_audio_with_metadata(input_music_file, output_directory)
