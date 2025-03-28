from pydub import AudioSegment
import numpy as np

def audio2input(wavFilePath: str, fileType: str='mp4', sampleRate: int=16000, \
                sampleWidth: int=2, channels: int=1, startSecond:int=0, \
                    endSecond:int=None):
# 提取文件中的音频并保存为数组正则化格式
    try:
        audioSegment = AudioSegment.from_file(wavFilePath, fileType)
    except Exception as e:
        print(f'Can not get audio array [{wavFilePath}], because {e}')
        return None

    # 将音频文件的采样率设为 16000 Hz
    if audioSegment.frame_rate != sampleRate:
        audioSegment = audioSegment.set_frame_rate(16000)

    # 将样本宽度设置为 16 位
    if audioSegment.sample_width != sampleWidth:
        audioSegment = audioSegment.set_sample_width(2)

    # 将音频文件转换为单声道
    if audioSegment.channels != channels:
        audioSegment = audioSegment.set_channels(1)

    # 将音频数据转换为 numpy 数组
    arr = np.array(audioSegment.get_array_of_samples())
    if endSecond is not None:
        arr = arr[startSecond*sampleRate: endSecond*sampleRate]

    # 将 int16 格式转换为 float32 格式并归一化到范围 [-1.0, 1.0]
    arr = arr.astype(np.float32) / 32768.0
    
    return arr