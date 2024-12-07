pip install demucs

pip install librosa pydub soundfile

import os
from spleeter.separator import Separator
from pydub import AudioSegment
from demucs.apply import apply_model
from demucs.pretrained import get_model
from pydub.playback import play

def ai_remaster(input_file, output_file):
    """
    AI 리마스터링을 통해 음질 개선.
    :param input_file: 입력 오디오 파일 경로
    :param output_file: 리마스터링된 파일 저장 경로
    """
    try:
        model = get_model("htdemucs") 
        wavs, _ = apply_model(model, input_file)
        
        wav = wavs[0]  
        
        wav.export(output_file, format="wav")
        print(f"리마스터링 파일이 저장됨: {output_file}")
    except Exception as e:
        print(f"리마스터링 중 오류 발생: {e}")

def separate_and_remaster(input_file, output_dir):
    """
    음악 파일을 스템으로 분리하고, AI 리마스터링을 적용.
    :param input_file: 입력 음악 파일 경로
    :param output_dir: 리마스터링된 파일 저장 디렉토리
    """
    song_name = os.path.splitext(os.path.basename(input_file))[0]

    separator = Separator('spleeter:5stems')
    print(f"Processing: {input_file}")
    separator.separate_to_file(audio_descriptor=input_file, destination=output_dir)
    
    for root, _, files in os.walk(output_dir):
        for file in files:
            if file.endswith('.wav'):
                input_path = os.path.join(root, file)
                output_path = os.path.join(root, f"{song_name}_{file.split('.')[0]}_remastered.wav")
                ai_remaster(input_path, output_path)
                os.remove(input_path)

if __name__ == "__main__":
    input_music_file = input("음악 파일 경로를 입력하세요: ")
    output_directory = input("출력 디렉토리 경로를 입력하세요: ")

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    separate_and_remaster(input_music_file, output_directory)
