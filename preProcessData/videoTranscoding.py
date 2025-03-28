"""
    将视频(可能是webm或者mkv)转码成MP4格式。
"""
import os
import logging
from multiprocessing import Pool
import subprocess
import time
import argparse

logger = logging.getLogger('videoTranscoding')
formatter = logging.Formatter('%(asctime)s : %(name)s - %(levelname)s - %(message)s')
logger.setLevel(logging.INFO) 

parser = argparse.ArgumentParser(description="Transcode YouTube Video")
parser.add_argument('-b','--begin', type=int, default=0, help='Begin set ID (every set 4000 videos)')
args = parser.parse_args()

def FindSuitableGPUToUse():
    """
    检查GPU的使用率和显存利用率，选择显存利用率低于60%中使用率最低的GPU。
    """
    gpu_info = []
    try:
        output = subprocess.check_output(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"])
        gpu_data = output.decode("utf-8").strip().split('\n')
        for line in gpu_data:
            info = line.split(',')
            gpu_id = int(info[0])
            gpu_usage = int(info[1])
            memory_used = int(info[2])
            memory_total = int(info[3])
            memory_utilization = (memory_used / memory_total) * 100
            gpu_info.append((gpu_id, gpu_usage, memory_utilization))
        gpu_info.sort(key= lambda x: x[1])
        for gpu_Uti in gpu_info:
            if gpu_Uti[2] < 50:
                return gpu_Uti[0]
        return None  # 所有的GPU都在忙，使用率都超过了60%。
    except Exception as e:
        # print("Error:", e)
        return None

def TranscodeVideo(CUDA_visible_id:int, videoID:str, videoTail:str) -> int:
    """对视频进行转码的函数

    Args:
        CUDA_visible_id (int): 能用的CUDA(GPU)的ID
        videoID (str): 需要进行转码的视频的ID
        videoTail (str): 需要转码的视频的后缀 (mp4 or webm)

    Returns:
        int: 返回1代表转码失败, 返回0代表转码成功
    """

    if CUDA_visible_id is None:
        # 没找到合适的GPU处理转码，那么只能用CPU处理转码 
        cpuExecStatus = os.system(f'ffmpeg -n -loglevel error -i ./data/YoutubeVideo/{videoID}.{videoTail} -c:v libx264 \
                                  -b:v 300K -c:a aac ./data/YoutubeVideo/{videoID}.mp4')
        if cpuExecStatus == 0:
            os.system(f'rm -f ./data/YoutubeVideo/{videoID}.{videoTail}')
            return 0
        else:
            os.system(f'rm -f ./data/YoutubeVideo/{videoID}.mp4') # 删除转码失败的MP4文件，防止后续报错 
            return 1

    # 有GPU用时
    gpuExecStatus = os.system(f'CUDA_VISIBLE_DEVICES={CUDA_visible_id} ffmpeg -n -loglevel error -hwaccel cuda \
                            -i ./data/YoutubeVideo/{videoID}.{videoTail} -c:v h264_nvenc -preset medium -b:v 300K \
                                ./data/YoutubeVideo/{videoID}.mp4')    
    if gpuExecStatus == 0:
        # GPU转码成功
        os.system(f'rm -f ./data/YoutubeVideo/{videoID}.{videoTail}') # 删除原格式的视频文件
        return 0
    else:
        # GPU 转码失败改由CPU转码
        os.system(f'rm -f ./data/YoutubeVideo/{videoID}.mp4') # 删除转码失败的MP4文件，防止后续报错 
        return TranscodeVideo(None, videoID, videoTail)


def Ith_ProcessTranscodingVideo(numOfVideo:int, videoID:str) -> int:
    """处理视频id的子进程

    Args:
        numOfVideo (int): 处理的第多少个视频
        videoId (str): 处理的视频ID

    Returns:
        int: 处理情况，值为0时处理成功，为1时转码失败，为2时找不到视频
    """
    if numOfVideo % 100 == 0:
        logger.info(f'\033[1;31;40m Transcoding {numOfVideo}/119681 videos \033[0m')

    if os.path.exists(f'./data/YoutubeVideo/{videoID}.mp4'):
        if os.path.exists(f'./data/YoutubeVideo/{videoID}.webm') or\
              os.path.exists(f'./data/YoutubeVideo/{videoID}.mkv'):
            os.system(f'rm -f ./data/YoutubeVideo/{videoID}.mp4') # 之前转码失败，删除重来
        else:
            return 0 # 单纯有mp4，没有webm或者mkv

    CUDA_visible_id = None
    if numOfVideo % 4000 < 20:
        # 第一个进程池中的进程平均分到各卡上，剩下的进程动态调度
        CUDA_visible_id = numOfVideo % 10
    else:
        for _ in range(10):
            CUDA_visible_id = FindSuitableGPUToUse()
            if CUDA_visible_id is not None:
                break
            else:
                time.sleep(3)

    if os.path.exists(f'./data/YoutubeVideo/{videoID}.webm'):
        transcodeVideoStatus = TranscodeVideo(CUDA_visible_id, videoID, 'webm')
    elif os.path.exists(f'./data/YoutubeVideo/{videoID}.mkv'):
        transcodeVideoStatus = TranscodeVideo(CUDA_visible_id, videoID, 'mkv')
    else:
        logger.error(f'Can not find video {videoID}')
        return 2

    return transcodeVideoStatus


if __name__ == '__main__':
    
    # log 设置
    os.makedirs('./log',exist_ok=True)
    fileHandler = logging.FileHandler('./log/VideoTrancoding.log')
    fileHandler.setLevel(logging.INFO)
    fileHandler.setFormatter(formatter)
    commandHandler = logging.StreamHandler()
    commandHandler.setLevel(logging.INFO)
    commandHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)
    logger.addHandler(commandHandler)

    legalIDs = None
    with open('./log/YoutubeVideoAvailForVMT.txt', 'r') as file:
        legalIDs = file.readlines()

    beginID = 4000*args.begin
    poolSize = 20 # 使用20个进程，每个卡上平均分2个，兼顾安全和效率。
    with Pool(poolSize) as P:
        allTranscodeVideosStatus = []
        
        IDsOfThisTurn = legalIDs[beginID: beginID + 4000]
        for i, videoID in enumerate(IDsOfThisTurn):
            ThisVideoTranscodeStatus = P.apply_async(Ith_ProcessTranscodingVideo, (beginID + i, videoID.strip()))
            allTranscodeVideosStatus.append(ThisVideoTranscodeStatus)

        allTranscodeVideosStatus = [status.get() for status in allTranscodeVideosStatus]
        failReason = ['Transcode failed', "Can't find this video"]
        for videoID, transcodeStatus in zip(legalIDs, allTranscodeVideosStatus):
            videoID = videoID.strip()
            if transcodeStatus == 0:
                # 转码成功
                with open('./data/TranscodeSuccess.txt', 'a') as file:
                    file.write(f'{videoID}\n')
            else:
                # 转码失败
                with open('./log/TranscodeFailed.txt', 'a') as file:
                    file.write(f'[{videoID}] transcode failed, becase {failReason[transcodeStatus-1]} \n')