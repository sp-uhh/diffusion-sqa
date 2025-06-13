import os
import shutil
import librosa
import soundfile as sf

def resample_wav(input_path, output_path, target_rate):
    data, original_rate = librosa.load(input_path, sr=None)  # `sr=None` preserves the original sample rate
    resampled_data = librosa.resample(data, orig_sr=original_rate, target_sr=target_rate)
    sf.write(output_path, resampled_data, target_rate)

def copy_and_resample_folder(src_folder, tgt_folder, target_rate):
    os.makedirs(tgt_folder, exist_ok=True)
    for item in os.listdir(src_folder):
        src_item = os.path.join(src_folder, item)
        tgt_item = os.path.join(tgt_folder, item)
        print(f"Copying {src_item} to {tgt_item}")
        if os.path.isdir(src_item):
            # Recursively copy folders
            copy_and_resample_folder(src_item, tgt_item, target_rate)
        elif item.lower().endswith('.wav'):
            # Resample wav files
            resample_wav(src_item, tgt_item, target_rate)
        else:
            # Copy other files as they are
            shutil.copy2(src_item, tgt_item)

if __name__ == "__main__":

    # Example usage
    source_folder = "./EARS-WHAM/"  # Change to your source folder path
    destination_folder = "./EARS-WHAM-16k/"  # Change to your destination folder path
    target_sample_rate = 16000  # Change to your desired sample rate

    copy_and_resample_folder(source_folder, destination_folder, target_sample_rate)
    print("Copying and resampling completed.")