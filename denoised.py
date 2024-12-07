pip install noisereduce spleeter pydub

import os
import librosa
import numpy as np
from spleeter.separator import Separator
from pydub import AudioSegment
from pydub.playback import play
import noisereduce as nr

def remove_noise(input_file, intensity):
    """
    오디오 파일에서 노이즈를 제거합니다.
    :param input_file: 입력 오디오 파일 경로
    :param intensity: 노이즈 제거 강도 (low, medium, high)
    :return: 노이즈가 제거된 AudioSegment 객체
    """
    try:
        y, sr = librosa.load(input_file, sr=None)
        
        noise_sample = y[:sr]  
        if intensity == "low":
            reduced_noise = nr.reduce_noise(y=y, sr=sr, y_noise=noise_sample, prop_decrease=0.1)
        elif intensity == "medium":
            reduced_noise = nr.reduce_noise(y=y, sr=sr, y_noise=noise_sample, prop_decrease=0.5)
        elif intensity == "high":
            reduced_noise = nr.reduce_noise(y=y, sr=sr, y_noise=noise_sample, prop_decrease=0.9)
        else:
            print("잘못된 강도 값이 입력되었습니다. 기본값(low) 적용.")
            reduced_noise = nr.reduce_noise(y=y, sr=sr, y_noise=noise_sample, prop_decrease=0.1)

        output_audio = AudioSegment(
            data=np.int16(reduced_noise * 32767).tobytes(),
            frame_rate=sr,
            sample_width=2,
            channels=1
        )
        return output_audio
    except Exception as e:
        print(f"노이즈 제거 중 오류 발생: {e}")
        return None

def preview_and_select(audio_low, audio_medium, audio_high):
    """
    사용자가 노이즈 제거 강도를 실시간으로 전환하며 미리 듣고 선택.
    :param audio_low: 낮은 강도로 노이즈가 제거된 AudioSegment 객체
    :param audio_medium: 중간 강도로 노이즈가 제거된 AudioSegment 객체
    :param audio_high: 높은 강도로 노이즈가 제거된 AudioSegment 객체
    :return: 선택된 강도 (low, medium, high)
    """
    while True:
        print("\n=== 노이즈 제거 강도 미리 듣기 ===")
        print("1. 낮음 (low)")
        print("2. 중간 (medium)")
        print("3. 높음 (high)")
        print("4. 강도 선택 완료")
        choice = input("강도를 선택하여 미리 듣거나 완료하세요 (1/2/3/4): ").strip()

        if choice == "1":
            print("낮음 (low) 강도로 미리 듣기 중...")
            play(audio_low)
        elif choice == "2":
            print("중간 (medium) 강도로 미리 듣기 중...")
            play(audio_medium)
        elif choice == "3":
            print("높음 (high) 강도로 미리 듣기 중...")
            play(audio_high)
        elif choice == "4":
            final_choice = input("최종 선택한 강도를 입력하세요 (low/medium/high): ").strip().lower()
            if final_choice in ["low", "medium", "high"]:
                return final_choice
            else:
                print("잘못된 입력입니다. 다시 선택하세요.")
        else:
            print("잘못된 입력입니다. 다시 시도하세요.")

def separate_and_denoise_with_selection(input_file, output_dir):
    """
    음악 파일을 스템으로 분리하고, 노이즈 제거 및 미리 듣기, 강도 선택 제공.
    :param input_file: 입력 음악 파일 경로
    :param output_dir: 노이즈 제거된 파일 저장 디렉토리
    """
    song_name = os.path.splitext(os.path.basename(input_file))[0]

    separator = Separator('spleeter:5stems')
    print(f"Processing: {input_file}")
    separator.separate_to_file(audio_descriptor=input_file, destination=output_dir)

    for root, _, files in os.walk(output_dir):
        for file in files:
            if file.endswith('.wav'):
                input_path = os.path.join(root, file)

                print(f"강도별 노이즈 제거 적용 중: {file}")
                audio_low = remove_noise(input_path, "low")
                audio_medium = remove_noise(input_path, "medium")
                audio_high = remove_noise(input_path, "high")

                print(f"미리 듣기 및 강도 선택 시작: {file}")
                selected_intensity = preview_and_select(audio_low, audio_medium, audio_high)

                output_path = os.path.join(root, f"{song_name}_{file.split('.')[0]}_denoised_{selected_intensity}.wav")
                selected_audio = {"low": audio_low, "medium": audio_medium, "high": audio_high}[selected_intensity]
                selected_audio.export(output_path, format="wav")
                print(f"최종 파일 저장됨: {output_path}")

                os.remove(input_path)  


if __name__ == "__main__":
    input_music_file = input("음악 파일 경로를 입력하세요: ")
    output_directory = input("출력 디렉토리 경로를 입력하세요: ")

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    separate_and_denoise_with_selection(input_music_file, output_directory)
