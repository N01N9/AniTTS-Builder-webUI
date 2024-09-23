import numpy as np
import librosa
import soundfile as sf
from audio_separator.separator import Separator
import os
import shutil

def UVR(model, stem, input_file, output_dir, batch):

    separator = Separator(output_dir=output_dir, output_single_stem=stem,
                          mdx_params = { "hop_length": 1024,"segment_size": 256,"overlap": 0.25,"batch_size": batch,"enable_denoise": False }, 
                          vr_params = { "batch_size": batch,"window_size": 512,"aggression": 5,"enable_tta": False,"enable_post_process": False,"post_process_threshold": 0.2,"high_end_process": False }, 
                          demucs_params = { "segment_size": "Default","shifts": 2,"overlap": 0.25,"segments_enabled": True }, 
                          mdxc_params = { "segment_size": 256,"batch_size": batch,"overlap": 8 })
    separator.load_model(model_filename=model)
    output_filename = separator.separate(input_file)[0]
    return os.path.join(output_dir, output_filename)

def load_wav_files(file_paths, sr=None):

    specs = []
    for file_path in file_paths:
        audio, sr = librosa.load(file_path, sr=sr)
        spec = librosa.stft(audio, n_fft=2048, hop_length=512, win_length=2048)
        specs.append(spec)
    return specs, sr

def max_spec_ensemble(spectrograms):
 
    max_spectrogram = np.max(np.abs(spectrograms), axis=0)
    
    max_phase = np.angle(spectrograms[0])
    max_spectrogram_complex = max_spectrogram * np.exp(1.j * max_phase)
    
    return max_spectrogram_complex

def save_wav_from_spec(spectrogram, sr, output_path):

    # ISTFT를 통해 시간 도메인으로 변환
    audio_output = librosa.istft(spectrogram, hop_length=512, win_length=2048)
    # 결과를 WAV 파일로 저장
    sf.write(output_path, audio_output, sr)

def UVR_ensemble(base_UVR_model_list, input_dir, output_dir, ensemble_output_dir,batch):

    denoise_model = "UVR-DeNoise.pth"
    denoise_stem = "No Noise"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    if not os.path.exists(ensemble_output_dir):
        os.makedirs(ensemble_output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith('.wav'):
            file_path = os.path.join(input_dir, filename)
            
            # 모델별로 분리 결과를 저장할 리스트
            model_outputs = []
            
            # 모든 UVR 모델에 대해 처리
            for model_info in base_UVR_model_list:
                model, stem = model_info 
                output_wav_path = UVR(model, stem, file_path, output_dir, batch)
                model_outputs.append(output_wav_path)
            
            # 모델별로 생성된 wav 파일들을 스펙트로그램으로 변환
            spectrograms, sr = load_wav_files(model_outputs)

            max_len = max([spec.shape[1] for spec in spectrograms])

            # 모든 스펙트로그램을 동일한 길이로 패딩
            padded_spectrograms = [librosa.util.fix_length(spec, size=max_len, axis=1) for spec in spectrograms]

            # numpy 배열로 변환
            spectrograms = np.array(padded_spectrograms)
            
            # Max Spec Ensemble 수행
            combined_spec = max_spec_ensemble(spectrograms)
            
            # 결합된 결과를 저장할 파일 경로
            ensemble_output_path = os.path.join(ensemble_output_dir, filename)
            
            # 결합된 결과를 시간 도메인으로 변환하여 wav 파일로 저장
            save_wav_from_spec(combined_spec, sr, ensemble_output_path)

            denoise_path = UVR(denoise_model, denoise_stem, ensemble_output_path, ensemble_output_dir, batch)
            os.remove(ensemble_output_path)
            os.rename(denoise_path, ensemble_output_path)

            shutil.rmtree(output_dir)
