import os
import librosa
import soundfile as sf
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

def adjust_tempo_and_key(input_file, output_file, target_tempo, target_key):
    """
    오디오 파일의 템포와 키를 조정.
    :param input_file: 원본 스템 파일 경로
    :param output_file: 변환된 파일 저장 경로
    :param target_tempo: 목표 템포 (BPM)
    :param target_key: 목표 키
    """
    try:
        y, sr = librosa.load(input_file, sr=None)
        
        current_tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        
        tempo_ratio = target_tempo / current_tempo
        y_tempo_adjusted = librosa.effects.time_stretch(y, tempo_ratio)
        
        current_key, _ = analyze_audio(input_file)
        key_name = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        current_key_index = key_name.index(current_key)
        target_key_index = key_name.index(target_key)
        key_shift = target_key_index - current_key_index
        y_key_adjusted = librosa.effects.pitch_shift(y_tempo_adjusted, sr, n_steps=key_shift)
        
        sf.write(output_file, y_key_adjusted, sr)
        print(f"템포와 키가 조정된 파일 저장됨: {output_file}")
    except Exception as e:
        print(f"오류 발생: {e}")

def separate_and_adjust(input_file, output_dir, target_tempo, target_key):
    """
    음악 파일을 스템으로 분리하고, 템포와 키를 조정.
    :param input_file: 입력 음악 파일 경로
    :param output_dir: 스템을 저장할 디렉토리 경로
    :param target_tempo: 목표 템포
    :param target_key: 목표 키
    """
    song_name = os.path.splitext(os.path.basename(input_file))[0]
    
    separator = Separator('spleeter:5stems')
    print(f"Processing: {input_file}")
    separator.separate_to_file(audio_descriptor=input_file, destination=output_dir)
    
    for root, _, files in os.walk(output_dir):
        for file in files:
            if file.endswith('.wav'):
                input_path = os.path.join(root, file)
                output_path = os.path.join(root, f"{song_name}_{file.split('.')[0]}_tempo{target_tempo}_key{target_key}.wav")
                adjust_tempo_and_key(input_path, output_path, target_tempo, target_key)
                os.remove(input_path)

if __name__ == "__main__":
    input_music_file = input("음악 파일 경로를 입력하세요: ")
    output_directory = input("출력 디렉토리 경로를 입력하세요: ")

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    target_tempo = int(input("목표 템포를 입력하세요: "))
    target_key = input("목표 키를 입력하세요 (예: C, D#, G): ").strip()

    separate_and_adjust(input_music_file, output_directory, target_tempo, target_key)
