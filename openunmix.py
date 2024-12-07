pip install torch torchaudio

pip install openunmix

import torchaudio
from openunmix import predict
import os

def split_music_with_umx(input_file, output_dir, model_name='umxhq'):
    """
    Open-Unmix를 사용해 음악 파일을 스템으로 분리합니다.
    
    :param input_file: 입력 음악 파일 경로 (mp3, wav 등)
    :param output_dir: 출력 디렉토리 경로
    :param model_name: 사용할 Open-Unmix 모델 (기본: umxhq)
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        print(f"Processing: {input_file}")
        predict.separate(
            audio=input_file,
            targets=["vocals", "drums", "bass", "other"], 
            model_name=model_name,
            outdir=output_dir
        )
        print(f"Separation completed. Results saved in: {output_dir}")
    except Exception as e:
        print(f"Error occurred: {e}")

if __name__ == "__main__":
    input_music_file = "example.mp3"  
    output_directory = "umx_output" 

    split_music_with_umx(input_music_file, output_directory)
