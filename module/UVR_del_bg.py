import sys
import os

msst_path1 = os.path.abspath("./MSST-WebUI")
sys.path.append(msst_path1)

from inference.msst_infer import MSSeparator
from utils.logger import get_logger

def UVR(model, store_dirs, input_folder):

    separator = MSSeparator(
        model_type="mel_band_roformer",
        config_path=f"MSST-WebUI/configs_backup/vocal_models/{model[0]}",
        model_path=f"MSST-WebUI/pretrain/vocal_models/{model[1]}",
        device='auto',
        device_ids=[0],
        output_format='wav',
        use_tta=False,
        store_dirs=store_dirs,
        logger=get_logger(),
        debug=True
    )
    inputs_list = separator.process_folder(input_folder)
    separator.del_cache()
    results_list  = [[f"{store_dirs.values()[0]}/{i[:-4]}_{store_dirs.keys()[0]}.wav",f"{store_dirs.values()[0]}/{i}"] for i in inputs_list]
    for i in results_list:
        os.rename(i[0],i[1])

results_list = UVR(['config_Kim_MelBandRoformer.yaml','Kim_MelBandRoformer.ckpt'],{"other": "results"},"inputs")
