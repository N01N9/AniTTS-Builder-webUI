from huggingface_hub import hf_hub_download

repo_id = "Sucial/MSST-WebUI"
file_list = ["All_Models/vocal_models/Kim_MelBandRoformer.ckpt", "All_Models/vocal_models/model_mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt"]  # 다운로드할 파일들의 경로
destination_folder = "./MSST-WebUI/pretrain/vocal_models"
import os
os.makedirs(destination_folder, exist_ok=True)

for filename in file_list:
    file_path = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=destination_folder)
    print(f"Downloaded file is located at: {file_path}")
