import os
from spleeter.separator import Separator
from pydub import AudioSegment
from pydub.effects import normalize, low_pass_filter, high_pass_filter, reverb
import librosa

def analyze_audio(file_path):
    """
    음악 파일에서 템포와 키를 추출.
    :param file_path: 입력 음악 파일 경로
    :return: 템포와 키 정보
    """
    try:
        y, sr = librosa.load(file_path, sr=None)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        key_name = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        key = key_name[chroma.mean(axis=1).argmax()]
        return round(tempo), key
    except Exception as e:
        print(f"오류 발생: {e}")
        return None, None

def apply_effects(input_file, output_file, effect_type):
    """
    오디오 파일에 선택된 효과를 적용.
    :param input_file: 입력 오디오 파일 경로
    :param output_file: 출력 파일 경로
    :param effect_type: 적용할 효과 (volume, reverb, lowpass, highpass)
    """
    try:
        audio = AudioSegment.from_file(input_file)

        if effect_type == "volume":
            audio = audio + 5  
        elif effect_type == "reverb":
            audio = reverb(audio)
        elif effect_type == "lowpass":
            audio = low_pass_filter(audio, cutoff=300)  
        elif effect_type == "highpass":
            audio = high_pass_filter(audio, cutoff=1000) 
        else:
            print(f"알 수 없는 효과: {effect_type}")
            return

        audio.export(output_file, format="wav")
        print(f"효과가 적용된 파일 저장됨: {output_file}")

    except Exception as e:
        print(f"오류 발생: {e}")

def separate_and_apply_effects(input_file, output_dir, effect_type):
    """
    음악 파일을 스템으로 분리하고, 각 스템에 오디오 효과를 적용.
    :param input_file: 입력 음악 파일 경로
    :param output_dir: 출력 디렉토리 경로
    :param effect_type: 적용할 효과 (volume, reverb, lowpass, highpass)
    """
    song_name = os.path.splitext(os.path.basename(input_file))[0]

    print("템포와 키 분석 중...")
    tempo, key = analyze_audio(input_file)
    if tempo is None or key is None:
        print("템포와 키를 분석할 수 없습니다. 기본값 사용.")
        tempo, key = "Unknown", "Unknown"

    separator = Separator('spleeter:5stems')
    print(f"Processing: {input_file}")
    separator.separate_to_file(audio_descriptor=input_file, destination=output_dir)
    
    for root, _, files in os.walk(output_dir):
        for file in files:
            if file.endswith('.wav'):
                input_path = os.path.join(root, file)
                output_path = os.path.join(root, f"{song_name}_{file.split('.')[0]}_tempo{tempo}_key{key}_{effect_type}.wav")
                apply_effects(input_path, output_path, effect_type)
                os.remove(input_path)

if __name__ == "__main__":
    input_music_file = input("음악 파일 경로를 입력하세요: ")
    output_directory = input("출력 디렉토리 경로를 입력하세요: ")

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    print("적용 가능한 효과: volume, reverb, lowpass, highpass")
    selected_effect = input("적용할 효과를 입력하세요: ").strip()

    separate_and_apply_effects(input_music_file, output_directory, selected_effect)
