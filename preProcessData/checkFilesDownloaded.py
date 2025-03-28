"""
This file will check if the video has been downloaded both with the video and Chinese/English subtitle.
"""
import os 
import logging
import glob
from opencc import OpenCC

logger = logging.getLogger('checkFiles')
formatter = logging.Formatter('%(asctime)s : %(name)s - %(levelname)s - %(message)s')
logger.setLevel(logging.INFO) 

def findSuitableSubtitle(vttFilesAboutThisVideo, videoID, isEng):
    ENSubtitilPriority = ['en', 'en-US', 'en-GB', 'en-CA', 'en-AU']
    ZHSubtitilPriority = ['zh-CN', 'zh-Hans', 'zh', 'ZH-Hant', 'zh-HK', 'zh-TW']
    subtitleFile = None
    isHant = False

    # 严格匹配 
    subtitlePriority = ENSubtitilPriority if isEng else ZHSubtitilPriority
    for subIndex, subTail in enumerate(subtitlePriority):
        for file in vttFilesAboutThisVideo:
            if f'{videoID}.{subTail}.vtt' in file:
                subtitleFile = file
                if not isEng and subIndex >= 3:
                    isHant = True
                break
        if subtitleFile is not None:
            break
    
    # 非严格匹配
    if subtitleFile is None:
        subtitleTail = 'en' if isEng else 'zh'
        for file in vttFilesAboutThisVideo:
            if f'{videoID}.{subtitleTail}' in file: 
                subtitleFile = file
                if not isEng:
                    isHant = True
                break
    return subtitleFile, isHant

def converTraditionalChinese2Simplifed(subtitlePath):
    cc = OpenCC('t2s')
    with open(subtitlePath, 'r', encoding='utf-8') as f:
        traditional_text = f.read()
    simplified_text = cc.convert(traditional_text)
    with open(subtitlePath, 'w', encoding='utf-8') as f:
        f.write(simplified_text)

def checkVideoID(videoID):
    """
    This function will check if the video has been downloaded both with the video and Chinese/English subtitle.
    """
    # Check if the video has been downloaded.
    subtitle_ZH = None
    subtitle_EN = None

    videoTailList = ['webm', 'mp4', 'mkv']
    videoExist = False
    for videoTail in videoTailList:
        if os.path.exists(f"./data/YoutubeVideo/{videoID}.{videoTail}"):
            videoExist = True
            break
    if not videoExist:
        return False 

    filesAboutThisVideo = glob.glob(f"./data/YoutubeVideo/{videoID}.*.vtt")
    subtitle_EN, _ = findSuitableSubtitle(filesAboutThisVideo, videoID, True)
    subtitle_ZH, isHant = findSuitableSubtitle(filesAboutThisVideo, videoID, False)

    if subtitle_EN is None or subtitle_ZH is None:
        return False
    if isHant:
        converTraditionalChinese2Simplifed(subtitle_ZH)

    # 统一将字幕文件名改为videoID.en.vtt和videoID.zh.vtt
    os.rename(subtitle_EN, f"./data/YoutubeVideo/{videoID}.en.vtt")
    os.rename(subtitle_ZH, f"./data/YoutubeVideo/{videoID}.zh.vtt")

    return True

if __name__ == "__main__":

    os.makedirs('./log',exist_ok=True)
    fileHandler = logging.FileHandler('./log/checkFiles.log')
    fileHandler.setLevel(logging.INFO)
    fileHandler.setFormatter(formatter)
    commandHandler = logging.StreamHandler()
    commandHandler.setLevel(logging.INFO)
    commandHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)
    logger.addHandler(commandHandler)

    videoAvail4VMTList = []
    VideoNotAvail4VMT = []

    count = 0

    # Get the video path and subtitle path.
    with open("./log/DownloadSuccess.txt", "r") as file:
        for videoID in file:
            videoID = videoID.strip()
            if len(videoID) == 32:
                continue # 西瓜视频id
            elif len(videoID) != 11:
                raise Exception("The video ID is neither a YouTube video ID nor a Xigua video ID.")
            suceessMatch = checkVideoID(videoID)
            if suceessMatch:
                videoAvail4VMTList.append(videoID)
            else:
                # 既然残缺不全就删除其他的文件
                VideoNotAvail4VMT.append(videoID)
                os.system(f'rm -f ./data/YoutubeVideo/{videoID}*')
            
            count += 1
            if count % 100 == 0:
                logger.info(f"\033[1;31;40m Processing {count}th/132056 video \033[0m")
            
    # 删除所有非zh和en的字幕文件
    os.system("find ./data/YoutubeVideo/ -type f -name '*.*.vtt' ! -name '*.zh.vtt' ! -name '*.en.vtt' -exec rm -f {} +")  
    
    with open("./log/YoutubeVideoAvailForVMT.txt", "a") as file:
        for videoID in videoAvail4VMTList:
            file.write(videoID + "\n")
    with open("./log/YoutubeVideoNotAvailForVMT.txt", "a") as file:
        for videoID in VideoNotAvail4VMT:
            file.write(videoID + "\n")