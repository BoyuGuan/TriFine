import json  
import os

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import argparse

class recoAudioEmotionDataset(Dataset):
    def __init__(self, clipFilePath):
        with open(clipFilePath, 'r') as f:
            self.allClips = json.load(f)

    def __len__(self):
        return len(self.allClips)
    
    def __getitem__(self, idx):
        clip = self.allClips[idx]
        audioStartSeocnd = clip['subtitle_start_second'] - clip['clip_start_second']
        audioDuration = clip['subtitle_end_second'] - clip['subtitle_start_second'] + 1

        return {'video_id': clip['video_id'], 'clip_id': clip['clip_id'], \
                'audioDuration': audioDuration, 'audioStartSeocnd':audioStartSeocnd}

def collate_fn(batchRawData):

    batchAudioFile = []

    for data in batchRawData:
        videoID = data['video_id']
        clipID = data['clip_id']
        audioStartSeocnd = data['audioStartSeocnd']
        audioDuration = data['audioDuration']
        videoClipPath = f"./data/trainVideoClips/{videoID}/{videoID}_{clipID}.mp4"
        try:
            os.system(f"ffmpeg -n -loglevel error -ss {audioStartSeocnd} -t {audioDuration} -i {videoClipPath} -ar 16000 -ac 1 ./data/wavFiles/{videoID}_{clipID}.wav")
            batchAudioFile.append((True, videoID, clipID))
        except:
            continue
    return batchAudioFile

def emoRecon(dataloader, classifier):

    allClipsWithEmotion = []
    for batch_idx, batchData in tqdm(enumerate(dataloader), total=len(dataloader)):
        batchLegalClips = [(data[1], data[2]) for data in batchData if data[0]]
        if len(batchLegalClips) == 0:
            continue
        try:
            rec_results = classifier([f"./data/wavFiles/{clip[0]}_{clip[1]}.wav" for clip in batchLegalClips],
                                    batch_size=len(batchLegalClips),
                                    granularity="utterance", extract_embedding=False)
            for rec_result, clip in zip(rec_results, batchLegalClips):
                labels = rec_result['labels']
                scores = rec_result['scores']
                maxScore = max(scores)
                max_index = scores.index(maxScore)
                label = labels[max_index]
                allClipsWithEmotion.append({'video_id': clip[0], 'clip_id': clip[1], 'emotion': label, "maxScore": maxScore})
        except Exception as e:
            print(f"\033[1;31;40m Error in batch {batch_idx}: {e}  \033[0m")
    return allClipsWithEmotion

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-l', "--language", type=str, default="zh", help='Language of clips, zh or en')
    args = parser.parse_args()

    os.makedirs('./data/wavFiles/', exist_ok=True)

    assert args.language in ["zh", "en"], "Only support zh and en language!"

    audioDataset = recoAudioEmotionDataset(f"./data/cut_{args.language}_Clips.json")

    dataloader = DataLoader(audioDataset, batch_size=64, shuffle=False, collate_fn=collate_fn, num_workers=32, prefetch_factor=2, pin_memory=False)
    # dataloader = DataLoader(audioDataset, batch_size=4, shuffle=False, collate_fn=collate_fn)

    # 加载emotion2vec-large
    classifier = pipeline(
            task=Tasks.emotion_recognition,
            model="iic/emotion2vec_plus_large",
            device='cuda')

    allClipsWithEmotion = emoRecon(dataloader, classifier)
    
    with open(f"./data/{args.language}_emotion_Clips.json", 'w', encoding='utf-8') as f:
        json.dump(allClipsWithEmotion, f, ensure_ascii=False, indent=2)