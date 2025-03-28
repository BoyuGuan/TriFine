import numpy as np
from utils.processAudio import audio2input

def calculate_audio_volume_per_segment(arr, segment_duration, sample_rate):
    segment_length = int(segment_duration * sample_rate)
    num_segments = len(arr) // segment_length
    rms_values = []
    peak_values = []
    db_values = []

    for i in range(num_segments):
        segment = arr[i * segment_length : (i + 1) * segment_length]
        rms = np.sqrt(np.mean(segment**2))
        peak = np.max(np.abs(segment))
        db = 20 * np.log10(rms) if rms > 0 else -np.inf
        rms_values.append(rms)
        peak_values.append(peak)
        db_values.append(db)

    return rms_values, peak_values, db_values


def split_audio_and_calculate_rms(audio_array: np.ndarray, num_splits: int):
    # 计算每一份的长度
    split_length = len(audio_array) // num_splits
    rms_values = []
    for i in range(num_splits):
        start_idx = i * split_length
        end_idx = (i + 1) * split_length if i < num_splits - 1 else len(audio_array)
        # 获取音频的每一份
        split_segment = audio_array[start_idx:end_idx]
        # 计算RMS值
        rms = np.sqrt(np.mean(split_segment**2))
        rms_values.append(rms)
    
    return rms_values


# 假设你已经通过 audio2input 函数得到了 arr
arr = audio2input("/home/byguan/vmt/data/trainVideoClips/mLZ4QjD9wKA/mLZ4QjD9wKA_123.mp4")
arr = arr[int(160086/10)*4:int(160086/10)*5]
print(split_audio_and_calculate_rms(arr, 7))


# # 计算每0.5秒的响度值
# segment_duration = 0.5  # 0.5 秒
# sample_rate = 16000  # 音频的采样率

# rms_values, peak_values, db_values = calculate_audio_volume_per_segment(arr[64034:96051], segment_duration, sample_rate)

# 打印每段的响度值
# for i, (rms, peak, db) in enumerate(zip(rms_values, peak_values, db_values)):
#     print(f'Segment {i+1}: RMS: {rms}, Peak: {peak}, Decibels: {db} dB')



