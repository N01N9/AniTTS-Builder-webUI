from huggingface_hub import hf_hub_download
import os

repo_id = "Sucial/MSST-WebUI"
file_list = [
    "All_Models/vocal_models/Kim_MelBandRoformer.ckpt",
    "All_Models/vocal_models/model_mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt"
]  # 다운로드할 파일들의 경로
destination_folder = "./module/MSST-WebUI/pretrain/vocal_models"
os.makedirs(destination_folder, exist_ok=True)

for filename in file_list:
    # 파일 다운로드 (local_dir_use_filename 인수를 제거)
    file_path = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=destination_folder)

    # 다운로드된 파일을 destination_folder로 옮기기
    file_name_only = os.path.basename(filename)
    final_path = os.path.join(destination_folder, file_name_only)
    os.rename(file_path, final_path)

    # 원래의 빈 폴더 경로 삭제 시도
    original_folder = os.path.dirname(file_path)
    try:
        os.removedirs(original_folder)
    except OSError:
        pass  # 폴더가 비어 있지 않으면 예외 발생하므로 무시

    print(f"Downloaded file is located at: {final_path}")
