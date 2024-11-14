from module import converter, wav_slice_module, wav_filtering_module, embedding_module, clustering_module
import os
import gradio as gr
import shutil
import torch
import traceback
from module.MSST_WebUI.inference.msst_infer import MSSeparator
from module.MSST_WebUI.utils.logger import get_logger


os.chdir(os.path.dirname(os.path.abspath(__file__)))

def UVR(model, store_dirs, input_folder):
    logger = get_logger()
    separator = MSSeparator(
        model_type="mel_band_roformer",
        config_path=os.path.abspath(f"{os.path.dirname(os.path.abspath(__file__))}/module/MSST_WebUI/configs_backup/vocal_models/{model[0]}"),
        model_path=os.path.abspath(f"{os.path.dirname(os.path.abspath(__file__))}/module/MSST_WebUI/pretrain/vocal_models/{model[1]}"),
        device='auto',
        device_ids=[0],
        output_format='wav',
        use_tta=False,
        store_dirs=store_dirs,
        logger=logger,
        debug=True
    )
    inputs_list = separator.process_folder(input_folder)
    separator.del_cache()
    results_list  = [[f"{list(store_dirs.values())[0]}/{i[:-4]}_{list(store_dirs.keys())[0]}.wav",f"{list(store_dirs.values())[0]}/{i}"] for i in inputs_list]
    for i in results_list:
        os.rename(i[0],i[1])

def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    else:
        raise ValueError(f"{folder_path} already exists.")

def start(anime_name):
    try:
        if "reset" in anime_name:
            raise ValueError("You can't use reset for anime_name")
        create_folder_if_not_exists(f"./{anime_name}")
        create_folder_if_not_exists(f"./{anime_name}/input")
        create_folder_if_not_exists(f"./{anime_name}/input/mp4")
        create_folder_if_not_exists(f"./{anime_name}/input/ass")
        create_folder_if_not_exists(f"./{anime_name}/save")
        create_folder_if_not_exists(f"./{anime_name}/save/info")
        create_folder_if_not_exists(f"./{anime_name}/save/spectrogram")
        create_folder_if_not_exists(f"./{anime_name}/save/slicewav")
        create_folder_if_not_exists(f"./{anime_name}/save/slicewav/vocals")
        create_folder_if_not_exists(f"./{anime_name}/save/slicewav/inst")
        create_folder_if_not_exists(f"./{anime_name}/save/uvrwav")
        create_folder_if_not_exists(f"./{anime_name}/save/uvrwav/base_uvr")
        create_folder_if_not_exists(f"./{anime_name}/save/uvrwav/inst_uvr")
        create_folder_if_not_exists(f"./{anime_name}/save/assjson")
        create_folder_if_not_exists(f"./{anime_name}/save/rawwav")
        create_folder_if_not_exists(f"./{anime_name}/output")
        return "The task was executed successfully!"
    except Exception as e:
        return str(e)

def clear_folder(folder_path):
    if os.path.exists(folder_path):
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)

def converter_webUI(anime_name, substyle):
    try:
        os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)),anime_name))
    except Exception as e:
        return str(e)
    
    try:
        converter.convert_mp4_to_wav("./input/mp4", "./save/rawwav")
        converter.convert_ass_to_json(substyle, "./input/ass", "./save/assjson")

        return "The task was executed successfully!"
    
    except Exception as e:
        clear_folder("./save/rawwav")
        clear_folder("./save/assjson")
        return str(e)

def UVR_webUI(anime_name):
    try:
        os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)),anime_name))
    except Exception as e:
        return str(e)
    
    model1 = ['config_Kim_MelBandRoformer.yaml','Kim_MelBandRoformer.ckpt']
    input_folder1 = "./save/rawwav"
    output_dir1 = {"vocals":"./save/uvrwav/base_uvr"}

    model2 = ['config_mel_band_roformer_karaoke.yaml','model_mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt']
    input_folder2 = "./save/uvrwav/base_uvr"
    output_dir2 = {"other": "./save/uvrwav/inst_uvr"}
    
    try:
        UVR(model1, output_dir1, input_folder1)
        UVR(model2, output_dir2, input_folder2)

        return "The task was executed successfully!"
    
    except Exception as e:
        error_details = traceback.format_exc()
        clear_folder("./save/uvrwav/base_uvr")
        clear_folder("./save/uvrwav/inst_uvr")
        return str(error_details)


def sliceing_and_clustering_webUI(anime_name, persent, batch_size):
    if "reset" in anime_name:
        try: 
            anime_name = anime_name.split()[0]
            os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)),anime_name))
            clear_folder("./save/slicewav/vocals")
            clear_folder("./save/slicewav/inst")
            clear_folder("./save/spectrogram")
            clear_folder("./output")
            if os.path.exists("./save/info/cosine_distance.pt"):
                os.remove("./save/info/cosine_distance.pt")
            if os.path.exists("./save/info/all_embeddings.pt"):
                os.remove("./save/info/all_embeddings.pt")
            if os.path.exists("./save/info/spectrogram_inst.json"):
                os.remove("./save/info/spectrogram_inst.json")
            if os.path.exists("./save/info/embedding_map.json"):
                os.remove("./save/info/embedding_map.json")   
            return "Reset was executed successfully!"
        
        except Exception as e:
            return str(e)
    
    try:
        os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)),anime_name))
    except Exception as e:
        return str(e)
    
    try:
        wav_slice_module.find_matching_json("./save/uvrwav/base_uvr", "./save/assjson", "./save/slicewav/vocals", "./save/info/wav_info.json",'vocal')
        wav_slice_module.find_matching_json("./save/uvrwav/inst_uvr", "./save/assjson", "./save/slicewav/inst", "./save/info/wav_info.json",'inst')

        for dir in os.listdir("./save/slicewav"):
            if dir == "vocals":
                pass
            else:
                input_path = os.path.join("./save/slicewav", dir)
                output_path = os.path.join("./save/spectrogram", dir)
                output_json_path = os.path.join("./save/info", f'spectrogram_{dir}.json')
                wav_filtering_module.spectrogram_json(input_path, output_path, output_json_path, './save/slicewav/vocals', persent)

        if len(os.listdir("./save/slicewav/vocals"))==0:
            raise ValueError("persent is too low")

        output_pt_path = os.path.join("./save/info", "cosine_distance.pt")
        embedding_module.embeddings("./save/slicewav/vocals","./save/info", output_pt_path, batch_size)
        torch.cuda.empty_cache()
      
        embedding_path = os.path.join("./save/info", "all_embeddings.pt")
        json_path = os.path.join("./save/info", "embedding_map.json")
        clustering_module.clustering(output_pt_path, embedding_path, json_path, "./save/slicewav/vocals", "./output")
        torch.cuda.empty_cache()

        for folder_name in os.listdir("./output"):
            folder_path = os.path.join("./output", folder_name)
            if os.path.isdir(folder_path):
                if folder_name.startswith('clustering_'):

                    new_name = folder_name.replace('clustering_', 'speaker_')
                    folder_num = folder_name.replace('clustering_', '')
                    
                    old_path = os.path.join("./output", folder_name)
                    new_path = os.path.join("./output", new_name)

                    os.rename(old_path, new_path)

                    wav_files = [f for f in os.listdir(new_path) if f.endswith('.wav')]

                    for idx, wav_file in enumerate(sorted(wav_files), start=1):
                        old_file_path = os.path.join(new_path, wav_file)
                        new_file_name = f"{anime_name}_speaker{folder_num}_{idx}.wav"
                        new_file_path = os.path.join(new_path, new_file_name)
                        
                        os.rename(old_file_path, new_file_path)

        clear_folder("./save/slicewav/vocals")
        clear_folder("./save/slicewav/inst")
        return "The task was executed successfully!"

    except Exception as e:
        clear_folder("./save/slicewav/vocals")
        clear_folder("./save/slicewav/inst")
        clear_folder("./save/spectrogram")
        clear_folder("./output")
        if os.path.exists("./save/info/cosine_distance.pt"):
            os.remove("./save/info/cosine_distance.pt")
        if os.path.exists("./save/info/all_embeddings.pt"):
            os.remove("./save/info/all_embeddings.pt")
        if os.path.exists("./save/info/spectrogram_inst.json"):
            os.remove("./save/info/spectrogram_inst.json")
        if os.path.exists("./save/info/embedding_map.json"):
            os.remove("./save/info/embedding_map.json")   
        return str(e)
    
with gr.Blocks() as demo:
    gr.Markdown("# AniTTS_Builder webUI")

    with gr.Tab("Start Project"):
        gr.Markdown("## Start Project")
        gr.Markdown("This tab is for using AniTTS_Builder for a new animation. Please set the desired project name (animation name) and click the start button.")

        anime_name = gr.Textbox(label="Anime Name", placeholder="Enter the anime name (without multibyte characters)")
        start_button = gr.Button("Start Project")
        start_output = gr.Textbox(label="Output")
        start_button.click(start, inputs=anime_name, outputs=start_output)

    with gr.Tab("Convert MP4 to WAV and ASS to JSON"):
        gr.Markdown("## Convert MP4 to WAV and ASS to JSON")
        gr.Markdown("This tab is for preprocessing animation and subtitle files. Before running this tab, please place the desired animation .mp4 file in anime_name/input/mp4 and the animation subtitle .ass file in anime_name/input/ass. The video and subtitle files for each episode must be synchronized, have the same name, and must not contain multibyte characters.")
        gr.Markdown("The .ass file is used to check the timeline of the character's dialogue. Therefore, open the .ass file and copy the subtitle style corresponding to the character's dialogue. Most anime subtitles distinguish between different languages or between background music and dialogue using specific styles.")
        
        anime_name = gr.Textbox(label="Anime Name", placeholder="Enter the anime name")
        substyle = gr.Textbox(label="Substyle", placeholder="Enter the substyle (e.g., 'Default')")
        convert_button = gr.Button("Convert")
        convert_output = gr.Textbox(label="Output")
        convert_button.click(converter_webUI, inputs=[anime_name, substyle], outputs=convert_output)

    with gr.Tab("UVR Background Removal"):
        gr.Markdown("## UVR Background Removal")
        gr.Markdown("This tab is for removing background noise using UVR. It will take 0.5 to 0.75 times the total video length. This is the most time-consuming step, so please ensure that the program does not shut down during the process.")

        anime_name = gr.Textbox(label="Anime Name", placeholder="Enter the anime name")
        uvr_button = gr.Button("Remove Background with UVR")
        uvr_output = gr.Textbox(label="Output")
        uvr_button.click(UVR_webUI, inputs=[anime_name], outputs=uvr_output)

    with gr.Tab("Slicing and Clustering"):
        gr.Markdown("## Slicing and Clustering")
        gr.Markdown("This tab is for extracting character voices from audio with background noise removed and clustering them by character.")
        gr.Markdown("Percent is a variable that determines how clean (with less background noise) the files need to be in order to be used. Setting it to 0 means no files will be used, while setting it to 100 means all files will be used. The default value is 50, and it is recommended not to change this value. If you want to use only cleaner files, use a smaller value, and if you have a shortage of video data, you can increase the value. When set to 25, you can obtain about 5-6 minutes of high-quality data based on the main character and the 1 season of the animation (20 minutes X 12 episodes). ")
        gr.Markdown("You can set the batch size. The default value is 32, but if you encounter an error indicating insufficient GPU memory, please reduce this value.")
        gr.Markdown("If you have already completed clustering but want to redo it with a different percent value, please enter 'anime_name reset' in the anime_name field. If the reset is successful, a success message will appear. After that, you can adjust the percent value and proceed with clustering again.")

        anime_name = gr.Textbox(label="Anime Name", placeholder="Enter the anime name")
        persent = gr.Slider(minimum=0, maximum=100, step=1, label="Percent", value=50)
        batch_size = gr.Slider(minimum=1, maximum=1024, step=1, label="Batch Size", value=32)

        slice_cluster_button = gr.Button("Slice and Cluster Audio")
        slice_cluster_output = gr.Textbox(label="Output")
        slice_cluster_button.click(sliceing_and_clustering_webUI, inputs=[anime_name, persent, batch_size], outputs=slice_cluster_output)

demo.launch(inbrowser=True, share=True)
