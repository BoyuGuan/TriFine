"""
    识别并记录视频中的语言语种。通过对视频种选取3个10s的视频投票选取语言。
"""
import os
import logging
import time
import random
from torch.utils.data import Dataset, DataLoader
import json
from collections import defaultdict
from utils.processAudio import audio2input
from moviepy.editor import VideoFileClip
from speechbrain.inference.classifiers import EncoderClassifier
import torch
import numpy as np
from tqdm import tqdm

logger = logging.getLogger('recongLanguage')
formatter = logging.Formatter('%(asctime)s : %(name)s - %(levelname)s - %(message)s')
logger.setLevel(logging.INFO) 

class audioRecongDataset(Dataset):
    def __init__(self, videoIDsFilePath, clipFilePath):
        with open(videoIDsFilePath, 'r') as f:
            allVideoIDs = f.readlines()
            allVideoIDs = [videoID.strip() for videoID in allVideoIDs]
            self.allVideoIDs = allVideoIDs
        with open(clipFilePath, 'r') as f:
            clipInfo = json.load(f)
        self.videoID2Cliptime = {}
        for clip in clipInfo:
            if clip['video_id'] in self.videoID2Cliptime and \
                len(self.videoID2Cliptime[clip['video_id']]) < 128:
                self.videoID2Cliptime[clip['video_id']].append((clip['clip_start_second'], clip['clip_end_second']))
            else:
                self.videoID2Cliptime[clip['video_id']] = [(clip['clip_start_second'], clip['clip_end_second'])]

    def __len__(self):
        return len(self.allVideoIDs)

    def __getitem__(self, idx):
        videoID = self.allVideoIDs[idx]
        videoPath = f'./data/YoutubeVideo/{videoID}.mp4'
        
        audioArrs = audio2input(videoPath, startSecond=0, endSecond=None)
        if audioArrs is None:
            logger.error(f'\033[1;31;40m Can not get audio array of [{videoID}] \033[0m')
            return torch.randn([3, 16000*10]), False, videoID
        
        timeOfClips = self.videoID2Cliptime[videoID]
        timeOfClips = [clip for clip in timeOfClips if clip[1] * 16000 < len(audioArrs)]
        if len(timeOfClips) < 3:
            # 尽量使用短路，避免VideoFileClip读取视频浪费时间。
            logger.error(f'\033[1;31;40m Can not get enough clips of [{videoID}] \033[0m')
            return torch.randn([3, 16000*10]), False, videoID
        
        try:
            videoDuration = int(VideoFileClip(videoPath).duration)
        except Exception as e:
            logger.error(f'\033[1;31;40m Can not get video duration of [{videoID}], becasue {e} \033[0m')
            return torch.randn([3, 16000*10]), False, videoID

        if videoDuration < 11:
            logger.error(f'\033[1;31;40m Video duration is short than 10s [{videoPath}] \033[0m')
            return torch.randn([3, 16000*10]), False, videoID
        
        timeOfClips = [clip for clip in timeOfClips if clip[1] <= videoDuration -1]
        if len(timeOfClips) < 3:
            logger.error(f'\033[1;31;40m Can not get enough clips of [{videoID}] \033[0m')
            return torch.randn([3, 16000*10]), False, videoID

        returnArrs = []
        timeOfClips = random.sample(timeOfClips, 3)
        for timeClip in timeOfClips:
            returnArrs.append(audioArrs[timeClip[0]*16000: timeClip[1]*16000])
        return torch.tensor(returnArrs).reshape(3, -1), True, videoID

def reconLanguage(model, dataLoader):
    progress_bar = tqdm(range(len(dataLoader)))
    succeessRecongNum, allCount = 0, {}
    for batchIndex, (batchInput, bahtchWhetherLeagal, videoIDsInBatch) in enumerate(dataLoader, start=1):
        batchInput = batchInput.reshape(-1, 16000*10)
        outputs = model.classify_batch(batchInput)
        confidence = outputs[1].exp().reshape(-1, 3)
        confidence = np.array(confidence.to('cpu'))
        languageInThisBatch = np.array(outputs[3]).reshape(-1, 3)

        thisBatchSize = len(bahtchWhetherLeagal)
        results = ['None'] * thisBatchSize
        for i in range(thisBatchSize):
            if bahtchWhetherLeagal[i]:
                index_dict = defaultdict(list)
                for idx, value in enumerate(languageInThisBatch[i]):
                    index_dict[value].append(idx)
                duplicates = {value: indices for value, indices in index_dict.items() if len(indices) > 1}
                if duplicates:
                    thisConfidence = confidence[i][np.array(list(duplicates.values())[0])].mean()
                    if thisConfidence > 0.65:
                        results[i] = languageInThisBatch[i][list(duplicates.values())[0][0]]
                        succeessRecongNum += 1

        with open('./data/videoIDWithLanguage.txt', 'a') as f:
            for videoID, language in zip(videoIDsInBatch, results):
                f.write(f"{videoID.strip()} {language}\n")
                if language in allCount:
                    allCount[language] += 1
                else:
                    allCount[language] = 1
        progress_bar.update(1)
        if batchIndex % 10 == 0:
            logger.info(f"Processed batch {batchIndex}/{len(dataLoader)}")
    return succeessRecongNum, allCount

if __name__ == '__main__':
    # log 设置
    os.makedirs('./log',exist_ok=True)
    fileHandler = logging.FileHandler('./log/ReconLanguage.log')
    fileHandler.setLevel(logging.INFO)
    fileHandler.setFormatter(formatter)
    commandHandler = logging.StreamHandler()
    commandHandler.setLevel(logging.INFO)
    commandHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)
    logger.addHandler(commandHandler)

    recoLangModel = EncoderClassifier.from_hparams(source="speechbrain/lang-id-voxlingua107-ecapa",\
                                            savedir="./utils/speechLanguageRecon/", run_opts={"device":"cuda"})
    recoLangDataset = audioRecongDataset('./data/videoIDsAfterClean.txt', "./data/cleanClips.json")
    batchSize = 64
    recoLangDataLoader = DataLoader(recoLangDataset, batch_size=batchSize, shuffle=False, num_workers=32)

    succeessRecongNum, allCount = reconLanguage(recoLangModel, recoLangDataLoader)
    logger.info(f"The number of successfully recognized video is {succeessRecongNum}")
    allCount = list(allCount.items())
    allCount.sort(key=lambda x: x[1], reverse=True)
    for language, count in allCount:
        logger.info(f"[{language}]: {count:,}")


