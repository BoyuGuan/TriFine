import os
import re
import logging
import datetime
import json
from multiprocessing import Pool
from typing import List, Dict

from moviepy.editor import VideoFileClip

logger = logging.getLogger('cutVideo')
formatter = logging.Formatter('%(asctime)s : %(name)s - %(levelname)s - %(message)s')
logger.setLevel(logging.INFO) 

def processVttFile(vttFilePath):
    startList = []
    endList = []
    subtitleList = []
    temp_subtitle = ""
    sub_tag = 0
    timeStampPattern = r"(\d{2}:\d{2}:\d{2}\.\d{3}) --> (\d{2}:\d{2}:\d{2}\.\d{3})"
    with open(vttFilePath, 'r', encoding='utf-8') as s:
        for subtitle in s:
            matchTimeStamp = re.match(timeStampPattern, subtitle)
            if matchTimeStamp:
                sub_tag = 1
                temp_subtitle = ""
                start_time, end_time = matchTimeStamp.groups()
                startList.append(start_time)
                endList.append(end_time)

            elif subtitle == "\n" and sub_tag == 1:  # 每次读完一个字幕，进行一次存档(注意第一次回车不算)
                sub_tag = 0
                subtitleList.append(temp_subtitle)
            elif sub_tag == 1:
                # 如果原来的字幕中后边带有'\n'，也即本来是多行字幕，这一步会将原来字幕后自带的'\n'删除。变成一行字幕。
                temp_subtitle = temp_subtitle.replace('\n', ' ') + subtitle   
    return startList, endList, subtitleList

def format_timedelta(td):
    total_seconds = int(td.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return '{}:{:02}:{:02}'.format(hours, minutes, seconds)

def subtitleCut(sourceVideoPath, subtitle_EN_path, subtitle_ZH_path) -> List[Dict]:
    """切割字幕时间戳切割字幕

    Args:
        sourceVideoPath (str): 原视频路径
        subtitle_EN_path (str): 英文字母路径
        subtitle_ZH_path (str): 中文字幕路径

    Returns:
        List[Dict]: json格式的数据
    """

    sourceVideoID = sourceVideoPath.split("/")[-1].split(".")[0]
    
    # 读取视频长度
    try:
        videoDuration = int(VideoFileClip(sourceVideoPath).duration)
    except Exception as e:
        logger.error(f'\033[1;31;40m Can not get video duration of [{sourceVideoID}] \033[0m')
        os.system(f'rm -f ./data/YoutubeVideo/{sourceVideoID}*')
        return None
    
    # 视频长度小于11s的不要
    if videoDuration < 11:
        logger.error(f'\033[1;31;40m The duration of [{sourceVideoID}] is less than 10s \033[0m')
        os.system(f'rm -f ./data/YoutubeVideo/{sourceVideoID}*')
        return None

    Fin_EN = []
    Fin_ZH = []
    Fin_start_time = []
    Fin_end_time = []

    # 处理英文字幕
    EN_start_list, EN_end_list, EN_subtitle_list = processVttFile(subtitle_EN_path)
    # 处理中文字幕
    ZH_start_list, _, ZH_subtitle_list = processVttFile(subtitle_ZH_path)

    ZH_start_list.append("99:59:59.999")  # 加一位防止移出

    try:
        i, j = 0, 0
        output_ZH = ''
        while ZH_start_list[j] > EN_end_list[i]:
            i = i + 1  # 英文前面多的句子就不要了

        while i < len(EN_start_list):
            if ZH_start_list[j] >= EN_end_list[i]:
                # 说明中文该对齐下一个了，直接记录
                output_ZH = output_ZH.strip().replace(" ", "") + "\n" # 非常非常重要，这个不是空格，是[LSEP]                                    

                Fin_EN.append(EN_subtitle_list[i].strip()+'\n')
                Fin_ZH.append(output_ZH)
                Fin_start_time.append(EN_start_list[i])
                Fin_end_time.append(EN_end_list[i])

                output_ZH = ''
                i = i + 1
                if j == len(ZH_start_list):  # 英文后面多的句子就不要了
                    break
            else:
                # 否则中文还没对齐，顶上去
                output_ZH = output_ZH.strip() + " " + ZH_subtitle_list[j]
                j = j + 1

        i = len(Fin_EN)-1
        while i > 0:
            if Fin_ZH[i] == '' or Fin_ZH[i] == ' ' or Fin_ZH[i] == '\n':
                del Fin_EN[i]
                del Fin_EN[i-1]
                del Fin_ZH[i]
                del Fin_ZH[i - 1]
                del Fin_start_time[i]
                del Fin_start_time[i - 1]
                del Fin_end_time[i]
                del Fin_end_time[i - 1]
                i = i - 1
            i = i - 1
    except Exception as e:
        logger.error(f'\033[1;31;40m find error of [{sourceVideoID}] when split subtitle. \033[0m')
        return None

    if not (len(Fin_EN) == len(Fin_ZH) == len(Fin_start_time) == len(Fin_end_time)) or len(Fin_EN) == 0:
        logger.error(f'\033[1;31;40m The length of [{sourceVideoID}] is not match. \033[0m')
        return None

    resultInJson = []
    for idx, (en, zh, startTime, endTime) in enumerate(zip(Fin_EN, Fin_ZH, Fin_start_time, Fin_end_time), start=1):
        thisPair = {
            'video_id': sourceVideoID,
            'clip_id': idx, 
            "EN_sentence": en,
            "ZH_sentence": zh,
        }
        start_hour = int(startTime.split(":")[0])
        start_min = int(startTime.split(":")[1])
        start_sec = int(startTime.split(":")[2].split(".")[0])

        end_hour = int(endTime.split(":")[0])
        end_min = int(endTime.split(":")[1])
        end_sec = int(endTime.split(":")[2].split(".")[0])
        # print(start_hour, start_min, start_sec, start_micro)

        subtitleStart = datetime.timedelta(hours=start_hour, minutes=start_min, seconds=start_sec)
        subtitleEnd = datetime.timedelta(hours=end_hour, minutes=end_min, seconds=end_sec)
        thisPair['subtitle_start_second'] = int(subtitleStart.total_seconds())
        thisPair['subtitle_end_second'] = int(subtitleEnd.total_seconds())

        # 截取10s clip
        mid = (subtitleEnd + subtitleStart) * 0.5
        mid = int(mid.total_seconds())
        
        clipStart = mid - 5
        clipEnd = mid + 5
        if clipStart < 0:
            clipStart = 0
            clipEnd = 10
        elif clipEnd > videoDuration:
            clipEnd = videoDuration
            clipStart = videoDuration - 10

        thisPair['clip_start_second'] = clipStart
        thisPair['clip_end_second'] = clipEnd

        resultInJson.append(thisPair)

    return resultInJson

def Ith_ProcessCutVideo(videoID:str) -> bool:
    """处理视频id的子进程

    Args:
        videoId (str): 处理的视频ID

    Returns:
        List[Dict]: json格式的数据
    """

    sourceVideo = None
    videoID = videoID.strip()
    subtitle_EN_Path = f'./data/YoutubeVideo/{videoID}.en.vtt'
    subtitle_ZH_Path = f'./data/YoutubeVideo/{videoID}.zh.vtt'
    sourceVideo = f'./data/YoutubeVideo/{videoID}.mp4'
    
    # 缺少字幕或者视频
    if not os.path.exists(subtitle_EN_Path) or not os.path.exists(subtitle_ZH_Path) \
        or not os.path.exists(sourceVideo):
        logger.error(f'\033[1;31;40m Can not find subtitles or video for [{videoID}] \033[0m')
        os.system(f'rm -f ./data/YoutubeVideo/{videoID}*')
        return None

    return subtitleCut(sourceVideo, subtitle_EN_Path, subtitle_ZH_Path)

if __name__ == '__main__':

    # log 设置
    os.makedirs('./log',exist_ok=True)
    fileHandler = logging.FileHandler('./log/cutSubtitle.log')
    fileHandler.setLevel(logging.INFO)
    fileHandler.setFormatter(formatter)
    commandHandler = logging.StreamHandler()
    commandHandler.setLevel(logging.INFO)
    commandHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)
    logger.addHandler(commandHandler)

    os.makedirs('./data/trainVideoClips', exist_ok=True)

    legalIDs = None
    with open('./data/TranscodeSuccess.txt', 'r', encoding='utf-8') as file:
        legalIDs = file.readlines()
        
    # 保留16个核空闲处理其他任务，如果不够16个核就用8核心。但最多就用36个进程。
    poolSize = min(36, max(8, os.cpu_count() - 16))
    with Pool(poolSize) as P:
        allVideosCutStatus = P.map(Ith_ProcessCutVideo, legalIDs)

    numberOfVideoID = 0
    allResultsInJson = []
    for videoID, reusultOfTheVideo in zip(legalIDs, allVideosCutStatus):
        if reusultOfTheVideo is not None and len(reusultOfTheVideo) > 0:
            with open('./data/CutSubtitleSuccess.txt', 'a', encoding='utf-8') as file:
                file.write(videoID)
            allResultsInJson += reusultOfTheVideo
            numberOfVideoID += 1
        else:
            with open('./log/CutSubtitleFail.txt', 'a', encoding='utf-8') as file:
                file.write(videoID)
    with open('./data/clips.json', 'w', encoding='utf-8') as f:
        json.dump(allResultsInJson, f, ensure_ascii=False, indent=2)
    logger.info(f'All clips number (by subtitle before cleaning): {len(allResultsInJson):,}')
    logger.info(f'The number of all viedo ID of clips is {numberOfVideoID:,}')