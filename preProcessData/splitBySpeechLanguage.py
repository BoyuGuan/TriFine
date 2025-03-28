import os
import logging
import json

logger = logging.getLogger('splitBySpeechLanguage')
formatter = logging.Formatter('%(asctime)s : %(name)s - %(levelname)s - %(message)s')
logger.setLevel(logging.INFO) 

def splitBySpeechLanguage(allClips, videoID2Language, languageToSplit):
    clipsWithLanguage, languageSubset = [], {} 
    for language in languageToSplit:
        languageSubset[language] = []
    
    for clip in allClips:
        clipVideoID = clip['video_id']
        if clipVideoID in videoID2Language:
            clipLanguage = videoID2Language[clipVideoID]
            clip['language'] = clipLanguage
            clipsWithLanguage.append(clip)
            if clipLanguage in languageToSplit:
                languageSubset[clipLanguage].append(clip)
        else:
            logger.warning(f"Video ID {clipVideoID} not found in videoID2Language")
    return clipsWithLanguage, languageSubset

if __name__ == '__main__':
    # log 设置
    os.makedirs('./log',exist_ok=True)
    fileHandler = logging.FileHandler('./log/SplitLanguage.log')
    fileHandler.setLevel(logging.INFO)
    fileHandler.setFormatter(formatter)
    commandHandler = logging.StreamHandler()
    commandHandler.setLevel(logging.INFO)
    commandHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)
    logger.addHandler(commandHandler)

    with open('./data/videoIDWithLanguage.txt', 'r') as f:
        videoID2Language = f.readlines()
    videoID2Language = [ x.strip() for x in videoID2Language]
    videoID2Language = [ x.split(' ') for x in videoID2Language ]
    videoID2Language = { x[0]: " ".join(x[1:]) for x in videoID2Language }

    with open('./data/cleanClips.json', 'r') as f:
        dataClips = json.load(f)
    
    languageToSplit = ['en: English', 'zh: Chinese']

    clipsWithLanguage, languageSubset = splitBySpeechLanguage(dataClips, videoID2Language, languageToSplit)

    with open('./data/ClipsWithLanguage.json', 'w', encoding='utf-8') as f:
        json.dump(clipsWithLanguage, f, ensure_ascii=False, indent=2)
    logger.info(f'Saved {len(clipsWithLanguage):,} clips with language information to ./data/ClipsWithLanguage.json')
    
    for language in languageSubset:
        with open(f'./data/{language[:2]}_Clips.json', 'w', encoding='utf-8') as f:
            json.dump(languageSubset[language], f, ensure_ascii=False, indent=2)
        logger.info(f'Saved {len(languageSubset[language]):,} clips with {language} to ./data/{language[:2]}_Clips.json')