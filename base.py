pip install spleeter

import os
from spleeter.separator import Separator

def separate_audio(input_file, output_dir):
    """
    음악 파일에서 스템을 분리하는 함수.
    :param input_file: 입력 음악 파일 경로
    :param output_dir: 스템을 저장할 디렉토리 경로
    """
    separator = Separator('spleeter:5stems')

    print(f"Processing: {input_file}")
    separator.separate_to_file(audio_descriptor=input_file, destination=output_dir)
    print(f"Processed and saved to {output_dir}")

if __name__ == "__main__":
    input_music_file = input("음악 파일 경로를 입력하세요: ")
    output_directory = input("출력 디렉토리 경로를 입력하세요: ")

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    separate_audio(input_music_file, output_directory)
