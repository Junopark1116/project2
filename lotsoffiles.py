import os
import shutil
import zipfile
import librosa
import numpy as np
from spleeter.separator import Separator
from pydub import AudioSegment
from pydub.playback import play
import noisereduce as nr

def remove_noise(input_file, intensity):
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
    song_name = os.path.splitext(os.path.basename(input_file))[0]
    song_output_dir = os.path.join(output_dir, song_name)

    if not os.path.exists(song_output_dir):
        os.makedirs(song_output_dir)

    separator = Separator('spleeter:5stems')
    print(f"Processing: {input_file}")
    separator.separate_to_file(audio_descriptor=input_file, destination=song_output_dir)

    for root, _, files in os.walk(song_output_dir):
        for file in files:
            if file.endswith('.wav'):
                input_path = os.path.join(root, file)

                print(f"강도별 노이즈 제거 적용 중: {file}")
                audio_low = remove_noise(input_path, "low")
                audio_medium = remove_noise(input_path, "medium")
                audio_high = remove_noise(input_path, "high")

                print(f"미리 듣기 및 강도 선택 시작: {file}")
                selected_intensity = preview_and_select(audio_low, audio_medium, audio_high)

                output_path = os.path.join(song_output_dir, f"{song_name}_{file.split('.')[0]}_denoised_{selected_intensity}.wav")
                selected_audio = {"low": audio_low, "medium": audio_medium, "high": audio_high}[selected_intensity]
                selected_audio.export(output_path, format="wav")
                print(f"최종 파일 저장됨: {output_path}")

                os.remove(input_path)

    zip_file_path = os.path.join(output_dir, f"{song_name}.zip")
    with zipfile.ZipFile(zip_file_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(song_output_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, song_output_dir)
                zipf.write(file_path, arcname)
    print(f"노래별 압축 파일 생성됨: {zip_file_path}")

    shutil.rmtree(song_output_dir)

if __name__ == "__main__":
    input_path = input("음악 파일 경로나 디렉토리를 입력하세요: ")
    output_directory = input("출력 디렉토리 경로를 입력하세요: ")

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    if os.path.isfile(input_path):
        separate_and_denoise_with_selection(input_path, output_directory)
    elif os.path.isdir(input_path):
        files = [os.path.join(input_path, f) for f in os.listdir(input_path) if f.endswith(('.mp3', '.wav'))]
        if not files:
            print("디렉토리에 처리 가능한 음악 파일이 없습니다.")
        for file in files:
            separate_and_denoise_with_selection(file, output_directory)
    else:
        print("올바른 파일 또는 디렉토리 경로를 입력하세요.")
