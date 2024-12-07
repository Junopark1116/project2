1.	아이디어의 시작
   
저번 학기에 수학프로그래밍 수업에서는 내가 좋아하고 취미로 하고 있는 음악에 활용할 수 있는 방법을 배운 프로그래밍 지식을 통해 만들고자 했고 그렇게 만들었던 작곡 프로그램은 굉장히 유용하게 쓸 수 있었다. 그래서 이번에도 작곡 생활에 쓸 수 있는 프로그램을 만들되 더욱 많이 배운 지식을 어디에 활용해야 효과적일지를 생각해보던 도중 샘플링에 써보면 어떨까 생각했다. 샘플링이란 원래 있던 곡의 일부를 요소로 가지고 와 노래에 리믹스, 즉 재활용하는 방법이다. 이는 실제로 검증된 드럼 사운드나 베이스 사운드 등을 사용하고 싶을 때 가장 많이 채택하는 방식으로 실제 업계에서도 너무나 흔하게 볼 수 있는 방법이기도 하다. 다만 곡을 추출하고 분리하는 과정에서 음원 손실의 위험이 너무나도 크고 분리하는 프로그램 또한 가격이 높다. 그래서 내가 직접 AI를 이용한 프로그램을 만들면 돈도 아끼며 내가 원하는 결과물을 낼 수 있고 퀄리티 또한 더 높일 수 있지 않을까라는 생각으로 이 프로젝트를 시작하게 되었다.

2.	베이직 프로그램 설정 (base.py)
   
검색 결과, 음악 스케일을 만들어주던 도구가 이미 있었던 것처럼 음악의 각 악기 (이를 스템이라고 부른다) 를 분리해주는 도구가 이미 파이썬에 존재했다. 이를 spleeter 라고 하는데 이를 이용해서 처음에는 노래를 입력하면 스템을 분리해 각 파일에 저장해주는 가장 간단한 프로그램을 만들었다. 여기에서는 가장 자세한 분리를 하기 위해서 5-스템 모델을 사용하여 보컬, 드럼, 베이스, 피아노, 그리고 기타 악기 5가지로 분리할 수 있었다.

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
 
이를 실행하면 총 5개의 분리된 스템 파일이 나오게 된다.

3.	편의성 추가 (visual.py)

내가 실제로도 사용하기 위해서 만드는 프로그램인 만큼 가시적으로 보이는 정보가 간략하면서도 핵심적으로 모든 파일에 표시되었음 하였다. 이를 위해서 파일에 내가 샘플을 사용하기 위해 가장 중요하게 인지하고 있어야 하는 원본의 템포와 키를 추출 파일에 표시되도록 프로그램을 수정하였다. 원본 파일의 키와 템포를 알기 위해서는 librosa 라이브러리를 이용해야 했기 때문에 spleeter와 함께 설치해주었다. 추가 과정에서 노래 제목도 추가해주도록 하였다.

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
 
이제 파일은 제목_스템_템포_키의 형식으로 저장이 될 수 있다.

4.	간단한 편집 기능 추가 (shifter.py)
   
샘플링 기법의 가장 큰 특징은 이를 임의대로 편집해서 원곡의 요소와 같은 요소를 쓰면서도 다른 느낌을 줄 수 있다는 점이다. 나 또한 이런 샘플링의 특징을 살리기 위해 원곡 템포와 다르게 내가 조정할 수 있도록 프로그램을 고쳐 보기로 했다. 내가 입력하는 키와 템포에 따라 샘플이 미리 편집되어 나오도록 코드를 작성하였다. Time stretching 과 pitch shifting 을 통해서 이를 할 수 있다.

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

이제 내가 지정한 값에 따라 파일은 제목_스템_템포_키의 형식으로 저장이 된다.

5.	오디오 효과 추가 (effect.py)
   
추출한 스템들의 퀄리티는 원본에 비해서 손상이 되어 있기 때문에 낮아질 수 밖에 없다. 하지만 이를 직접 활용하기 위해서는 최대한 좋은 퀄리티를 유지하는 것이 중요하고 이는 리버브, 로우 패스 필터, 하이 패스 필터 등의 효과로 스템 사운드의 단점을 가리고 장점을 극대화할 수 있다. 직접 하는 방법도 있지만 나의 초기 목표는 최대한 편의성이 극대화 된 프로그램을 만드는 것이었기 때문에 이 또한 미리 프로그램에서 추가할 수 있도록 만들었다.

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
 
임의의 사운드 이펙트를 걸 수 있도록 하였고, 리버브, 로우 패스 필터, 하이 패스 필터, 그리고 볼륨을 줄일 수 있는 기능을 넣었다. 이제 이들 중 하나를 골라서 실행을 시킨다면 결과 파일은 제목_스템_템포_키_효과의 형식으로 저장이 된다.

6.	퀄리티를 위하여 (demucs.py, denoised.py)
   
효과만 들어간 스템들의 퀄리티는 여전히 만족스럽지 못했다. 따라서 추출된 스템들의 퀄리티를 높일 방법을 찾다가 AI 리마스터링 기능이 있는 고급 AI 모델인 Demucs를 찾게 되었다. 다만 이 프로그램을 돌리기 위해서는 충분한 GPU 기능이 받쳐줘야 했는데 아쉽게도 내가 이용할 수 있는 환경에서는 적합하지 않았다. 이후에 가능한 환경이 주어지면 사용해보기 위해서 demucs.py 를 만들어 뒀지만 예비 프로그램일 뿐이지 당장 사용하기 위해서는 다른 기능이 필요하였다. 

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

따라서 방법을 생각하던 중 스템들이 원본에 비해서 대체로 머디하다는 것, 즉 소리가 둔해졌다는 것을 느꼈고 이를 해결하기 위해서 다른 음역대의 소리를 완전히 걸러줄 수 있는 노이즈 제거 기능을 넣으면 좋을 거 같다고 생각하게 되었다. 이를 위해 noisereduce 라이브러리를 사용하였고, 이를 하면서 너무 많은 소리가 걸러지지 않도록 실시간으로 효과가 입혀진 파일을 들으며 노이즈 제거 정도를 정할 수 있으면 좋겠다는 생각도 하여서 그 기능들 또한 추가했다.

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
 
이를 통해 실시간으로 음악을 들어보면서 노이즈 제거 정도를 low, medium, high 중 정할 수 있게 되었고 최종 선택을 하는 강도로만 파일을 뽑을 수 있게 되었다. 파일 형식은 제목_스템_denoised_강도로 저장이 되도록 하였다.

7.	범용성 확장 (lotsoffiles.py)
   
한 번에 여러 곡을 처리하면 프로그램의 범용성이 더 좋을 거 같아 단일파일과 디렉토리를 전부 커버할 수 있는 코드를 넣었고 이 프로그램을 돌릴 때 너무 많은 결과물이 나오면 정리를 하는 것도 힘들고 가시적으로도 불편할 것 같아 노래별로 압축파일로 묶여서 나오도록 기능을 추가하였다.

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

이 프로그램을 이용하면 각 노래별로 결과 압축 파일이 생성되며, 각 파일에 대해 선택된 노이즈 제거 강도를 적용한 결과물을 포함하게 된다.

8.	결과
   
이후에도 더 많은 편의성 기능을 추가해주기 위해 구글 드라이브 연동 및 공유 링크 자동 생성 등의 기능을 더해봤지만 이미 노래별로 스템 압축 파일을 만들어 뽑아낸 것만으로도 충분히 일이 줄어든 거 같고 오히려 프로그램을 돌릴 때 범용성을 줄이는 것 같아 결론적으로 추가하지 않았다. 이번에 프로젝트를 진행하면서 가장 크게 느낀 점은 AI에 기반한 여러 도구들이 상상 이상의 편의성을 가져다 줄 수 있다는 것이다. 저번 프로젝트는 창의성을 수치화 시킬 수 있는가에 대한 것이어서 사실 만족스러운 결과물을 얻기 힘들었지만 이번 프로젝트는 어찌보면 수치화 된 결과물을 임의로 변경하여 사용할 수 있는가에 관한 것이었기 때문에 생각 이상으로 결과물이 좋게 나온 것 같다. 실제로 이번에 만든 프로그램은 외부에 돈을 주고 팔고 있는 AI 프로그램들과 많이 다르지 않다는 것을 느낄 수 있었고 프로그래밍의 활용도에 대해 감탄하게 된 계기였다.

10.	추가적인 궁금증 (openunmix.py)
    
앞에서 demucs 를 활용한 프로그램을 만들었지만 컴퓨터의 한계로 사용해보지 못하였다. 그렇다면 내가 사용한 spleeter와 고급 프로그램인 demucs 사이에, 즉 spleeter보다는 뛰어나면서도 컴퓨터가 감당할 수 있는 도구가 있을지 더 자세하게 찾아보았다. 그 결과 spleeter 만큼 사용하기 편리하지는 않지만 코딩에 대해서 지식이 좀 있다면 더 좋은 퀄리티로 활용할 수 있는 open-unmix라는 도구가 있다는 것을 알게 되었고 spleeter와의 퀄리티를 비교해 보고 싶어서 간단한 스템 스플리터를 만들게 되었다.

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
 
결과적으로 open-unmix를 활용한 스템들의 품질이 원본적으로는 조금 더 좋다고 느껴지긴 했지만 이 부분은 사람들마다 차이가 있는 부분이기도 하고 너무 미세하다는 생각 또한 들었다. 또한 spleeter에 오디오 효과들을 더한 스템들은 충분히 활용이 가능한 정도였고 굳이 spleeter 로 만든 프로그램은 open-unmix로 다시 만들 필요성은 느끼지 못하여 프로젝트를 마치게 되었다.
