import json


with open('./data/cleanClips.json', 'r') as f:
    dataClips = json.load(f)
print(len(dataClips))

with open('./data/videoIDWithLanguage.txt', 'r') as f:
        videoID2Language = f.readlines()
videoID2Language = [ x.strip() for x in videoID2Language]
videoID2Language = [ x.split(' ') for x in videoID2Language ]
videoID2Language = { x[0]: " ".join(x[1:]) for x in videoID2Language }

count = 0
languageCount = {'None':0, 'en: English':0, 'zh: Chinese':0, 'ko: Korean':0, 'ja: Japanese':0 }
for clip in dataClips:
    clipVideoID = clip['video_id']
    if clipVideoID in videoID2Language:
        if videoID2Language[clipVideoID] in languageCount:
            languageCount[videoID2Language[clipVideoID]] += 1

print(languageCount)
