import os
from typing import Tuple
import logging
from multiprocessing import Pool
import json
import argparse
from tqdm import tqdm

logger = logging.getLogger('cutVideo')
formatter = logging.Formatter('%(asctime)s : %(name)s - %(levelname)s - %(message)s')
logger.setLevel(logging.INFO) 


def Ith_ProcessCutVideo(clips:str) -> Tuple[bool]:
    """处理视频id的子进程

    Args:
        videoId (str): 处理的视频ID

    Returns:
        Tuple[bool]: bool值代表是否切割成功。
    """
    cutSucceess = [False]*len(clips)

    for i, clip in enumerate(clips):
        videoID = clip["video_id"]
        os.makedirs(f'./data/trainVideoClips/{videoID}', exist_ok=True)
        sourceVideoPath = f'./data/YoutubeVideo/{videoID}.mp4'
    
        if not os.path.exists(sourceVideoPath):
            logger.error(f'\033[1;31;40m Can not find video for [{videoID}] \033[0m')
            return False, 0

        clipStartSecond, clipEndSecond, numberOfClip = clip["clip_start_second"], clip["clip_end_second"], clip["clip_id"]
        if clipEndSecond - clipStartSecond != 10:
            logger.error(f"The {videoID} - {numberOfClip} start/end time wrong!")
            continue

        try:
            os.system(f"ffmpeg -n -loglevel error -ss {clipStartSecond} -i {sourceVideoPath}  -t 10 \
                -c:v libx264 -c:a aac ./data/trainVideoClips/{videoID}/{videoID}_{numberOfClip}.mp4")
            cutSucceess[i] = True
        except:
            logger.error(f"Cut video {videoID} clip {numberOfClip} failed.")
            continue

    return cutSucceess

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Cut video by subtitle.")
    parser.add_argument('-l','--language', type=str, default='en', help='Source video language (zh or en)')
    args = parser.parse_args()

    if args.language not in ['zh', 'en']:
        raise ValueError('Language must be zh or en')

    # log 设置
    os.makedirs('./log',exist_ok=True)
    fileHandler = logging.FileHandler('./log/CutVideo.log')
    fileHandler.setLevel(logging.INFO)
    fileHandler.setFormatter(formatter)
    commandHandler = logging.StreamHandler()
    commandHandler.setLevel(logging.INFO)
    commandHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)
    logger.addHandler(commandHandler)

    with open(f'./data/{args.language}_Clips.json', 'r') as file:
        dataClips = json.load(file)
    logger.info(f"Read dataClips successfully. The number of all clips is {len(dataClips):,}.")

    # 为提高每个进程的使用率，避免将过多的时间浪费在进程上下文切换上，此处每个进程一次处理32个clip。
    chunkSize = 32
    dataClips = [dataClips[i:i+chunkSize] for i in range(0, len(dataClips), chunkSize)]
    poolSize = min(32, max(8, os.cpu_count() - 16))
    with Pool(poolSize) as P:
        allVideosCutStatus = list(tqdm(P.imap(Ith_ProcessCutVideo, dataClips), total=len(dataClips)))

    cutSuccessClips = []
    for chunkIndex, chunkCutSuccess in enumerate(allVideosCutStatus):
        for i, cutSuccess in enumerate(chunkCutSuccess):
            if cutSuccess:
                cutSuccessClips.append(dataClips[chunkIndex][i])
    with open(f'./data/cut_{args.language}_Clips.json', 'w', encoding='utf-8') as f:
        json.dump(cutSuccessClips, f, ensure_ascii=False, indent=2)
    logger.info(f"Cut {len(cutSuccessClips):,} clips successfully.")
    logger.info(f"The number of different video ID is {len(set([clip['video_id'] for clip in cutSuccessClips])):,}.")