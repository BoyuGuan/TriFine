import json
import gc
import torch
from torch.utils.data import Dataset
from transformers import AutoProcessor, AutoImageProcessor
import numpy as np
import av
from pydub import AudioSegment
from scipy.stats import norm
import scipy.stats as stats

class vmtTextDataset(Dataset):
    """
        纯文本数据集，只返回文本的句对
    """
    def __init__(self, clipFilePath, tokenizer, srcLanguage='en', tgtLanguage='zh',\
                  maxLen=128, decoderStartTokenID=65000):
        with open(clipFilePath, 'r') as f:
            clipData = json.load(f)
        self.clipData = clipData
        self.tokenizer = tokenizer
        self.srcLanguage = srcLanguage
        self.tgtLanguage = tgtLanguage
        self.maxLen = maxLen
        self.decoderStartTokenID = decoderStartTokenID


    def __len__(self):
        return len(self.clipData)

    def __getitem__(self, idx):
        inputItem = {}
        clip = self.clipData[idx]
        srcText = clip[f"{self.srcLanguage.upper()}_sentence"]
        tgtText = clip[f"{self.tgtLanguage.upper()}_sentence"]
        
        srcToken = self.tokenizer(text=srcText, max_length=self.maxLen, padding='max_length',  truncation=True, return_tensors='pt')
        inputItem['input_ids'] = srcToken['input_ids'][0]
        inputItem['attention_mask'] = srcToken['attention_mask'][0]

        tgtToken = self.tokenizer(text_target=tgtText, max_length=self.maxLen, padding='max_length',  truncation=True, return_tensors='pt')
        # encoding['decoder_attention_mask'] = tgtToken['attention_mask'][0]
        labels = tgtToken['input_ids'][0]

        # 最新的模型会自动添加decoderInputIDs，不再需要手动指定
        # 手动后移一位并添加 decoder_start_token_id 在这个模型里也即 <pad>
        # decoderInputIDs = labels.clone().detach()
        # decoderInputIDs = torch.cat((torch.tensor([self.decoderStartTokenID]), decoderInputIDs[:-1]))
        # encoding['decoder_input_ids'] = decoderInputIDs

        end_token_index = torch.where(labels == self.tokenizer.eos_token_id)[0][0]
        labels[end_token_index+1:] = -100
        inputItem['labels'] = labels

        return inputItem

class vmtPreMixDataset(Dataset):
    """
        多模态的数据集格式，目前的返回值是为了满足wavLM和VideoMAE
    """
    def __init__(self, clipFilePath, tokenizer, srcLanguage='en', tgtLanguage='zh',\
                    maxLen=128, decoderStartTokenID=65000, includeVideo=True, includeAudio=True):
        with open(clipFilePath, 'r') as f:
            clipData = json.load(f)
        self.clipData = clipData
        self.tokenizer = tokenizer
        self.srcLanguage = srcLanguage
        self.tgtLanguage = tgtLanguage
        self.maxLen = maxLen
        self.mixModalityLen = 0
        self.decoderStartTokenID = decoderStartTokenID

        self.includeVideo = includeVideo
        self.includeAudio = includeAudio
        assert includeVideo or includeAudio, "At least one of video and audio should be included"
        if includeAudio:
            self.audioProcessor = AutoProcessor.from_pretrained("patrickvonplaten/wavlm-libri-clean-100h-base-plus")
            self.mixModalityLen += 5+1
        if includeVideo:
            self.videoProcessor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-large")
            self.mixModalityLen += 8+1

    def __len__(self):
        return len(self.clipData)
    
    def _sampleFrameIndices(self, clip_len, frame_sample_rate, seg_len):
        '''
        Sample a given number of frame indices from the video.
        Args:
            clip_len (`int`): Total number of frames to sample.
            frame_sample_rate (`int`): Sample every n-th frame.
            seg_len (`int`): Maximum allowed index of sample's last frame.
        Returns:
            indices (`List[int]`): List of sampled frame indices
        '''
        converted_len = int(clip_len * frame_sample_rate)
        end_idx = np.random.randint(converted_len, seg_len)
        start_idx = end_idx - converted_len
        indices = np.linspace(start_idx, end_idx, num=clip_len)
        indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
        return indices

    def _convertAudioFileToInput(self, audioFilePath: str, audioFileType: str = "mp4"):
        audio_segment = AudioSegment.from_file(audioFilePath, audioFileType)
        if audio_segment.frame_rate != 16000:
            audio_segment = audio_segment.set_frame_rate(16000)
        if audio_segment.sample_width != 2:
            audio_segment = audio_segment.set_sample_width(2)
        if audio_segment.channels != 1:
            audio_segment = audio_segment.set_channels(1)
        arr = np.array(audio_segment.get_array_of_samples())
        arr = arr.astype(np.float32) / 32768.0
        return arr

# class VideoMAEImageProcessor(BaseImageProcessor):
    # def preprocess(
        # imgNumber = len(videos)
        # result_array = np.zeros((imgNumber, 3, 224, 224), dtype=np.float32)
        # for i in range(imgNumber):
        #     result_array[i] = self._preprocess_image(
        #             image=videos[i],
        #             do_resize=do_resize,
        #             size=size,
        #             resample=resample,
        #             do_center_crop=do_center_crop,
        #             crop_size=crop_size,
        #             do_rescale=do_rescale,
        #             rescale_factor=rescale_factor,
        #             do_normalize=do_normalize,
        #             image_mean=image_mean,
        #             image_std=image_std,
        #             data_format=data_format,
        #             input_data_format=input_data_format,
        #         )
        # data = {"pixel_values": result_array}

    def __getitem__(self, idx):
        itemInput = {}
        clip = self.clipData[idx]
        srcText = clip[f"{self.srcLanguage.upper()}_sentence"]
        tgtText = clip[f"{self.tgtLanguage.upper()}_sentence"]
        
        # 这里要注意，音频、视频的特征长度（分别是5、8）及相应的[sep] token从输入的文本长度中预留出来
        srcToken = self.tokenizer(text=srcText, max_length=self.maxLen-self.mixModalityLen, padding='max_length',  truncation=True, return_tensors='pt')
        itemInput['text_input_ids'] = srcToken['input_ids'][0]
        # 增加音频、视频、sep的attention mask
        itemInput['attention_mask'] = torch.cat((torch.ones(self.mixModalityLen, dtype=torch.int), srcToken['attention_mask'][0]) )

        tgtToken = self.tokenizer(text_target=tgtText, max_length=self.maxLen, padding='max_length',  truncation=True, return_tensors='pt')
        # encoding['decoder_attention_mask'] = tgtToken['attention_mask'][0]
        labels = tgtToken['input_ids'][0]

        # 最新的模型会自动添加decoderInputIDs，不再需要手动指定
        # 但我们这里因为在多模态融合时需要使用input_embs输入，所以还是手动添加
        # 手动后移一位并添加 decoder_start_token_id 在这个模型里也即 <pad>
        # decoderInputIDs = labels.clone().detach()
        # decoderInputIDs = torch.cat((torch.tensor([self.decoderStartTokenID]), decoderInputIDs[:-1]))
        # input['decoder_input_ids'] = decoderInputIDs

        end_token_index = torch.where(labels == self.tokenizer.eos_token_id)[0][0]
        labels[end_token_index+1:] = -100
        itemInput['labels'] = labels

        videoID, clipID = clip["video_id"], clip["clip_id"]
        videoPath = f"./data/trainVideoClips/{videoID}/{videoID}_{clipID}.mp4"

        if self.includeAudio:
            # 得到符合wavLM输入格式的audio input
            # 经过过滤，所有的音频长度均为160086（16k*10s）。
            audioInputsInDigit = self._convertAudioFileToInput(videoPath)
            if len(audioInputsInDigit) > 160086:
                audioInputsInDigit = audioInputsInDigit[:160086]
            audioInputs= self.audioProcessor(audioInputsInDigit, \
                                                    sampling_rate=16000, return_tensors="pt")
            itemInput['audio_input_values'] = audioInputs['input_values'][0]
            itemInput['audio_attention_mask'] = audioInputs['attention_mask'][0]
            # itemInput['audio_inputs'] = audioInputs

            # itemInput['audio_input_values'] = torch.randn((160086, ))
            # itemInput['audio_attention_mask'] = torch.ones((160086,), dtype=torch.int32)

        if self.includeVideo:
            # 得到符合VideoMAE输入格式的video input
            with av.open(videoPath) as container:  # 使用with语句确保容器被正确关闭
                indices = self._sampleFrameIndices(clip_len=16, frame_sample_rate=1, seg_len=container.streams.video[0].frames)

                frames = []
                container.seek(0)
                start_index = indices[0]
                end_index = indices[-1]
                for i, frame in enumerate(container.decode(video=0)):
                    if i > end_index:
                        break
                    if i >= start_index and i in indices:
                        frames.append(frame)
                video = np.stack([x.to_ndarray(format="rgb24") for x in frames])
            videoInputs = self.videoProcessor(video, return_tensors="pt") # 需要对videoProcessor手动做一些变化，具体如上注释
            itemInput['video_inputs_pixel_values'] = videoInputs['pixel_values']

        gc.collect() # 无论是使用手动的container.close()方法还是使用with语句，都无法释放内存，只能手动释放
        # itemInput['video_inputs_pixel_values'] = torch.randn((16,3,224,224))

        return itemInput

class vmtAudioDataset(Dataset):
    def __init__(self, clipFilePath):
        with open(clipFilePath, 'r') as f:
            clipData = json.load(f)
        self.clipData = clipData

    def __len__(self):
        return len(self.clipData)

    def _convertAudioFileToInput(self, audioFilePath: str, audioFileType: str = "mp4"):
        audio_segment = AudioSegment.from_file(audioFilePath, audioFileType)
        if audio_segment.frame_rate != 16000:
            audio_segment = audio_segment.set_frame_rate(16000)
        if audio_segment.sample_width != 2:
            audio_segment = audio_segment.set_sample_width(2)
        if audio_segment.channels != 1:
            audio_segment = audio_segment.set_channels(1)
        arr = np.array(audio_segment.get_array_of_samples())
        arr = arr.astype(np.float32) / 32768.0
        return arr

    def __getitem__(self, idx):
        clip = self.clipData[idx]
        videoID, clipID = clip["video_id"], clip["clip_id"]
        videoPath = f"./data/trainVideoClips/{videoID}/{videoID}_{clipID}.mp4"

        audioInputsInDigit = self._convertAudioFileToInput(videoPath)
        if len(audioInputsInDigit) != 160086:
            print(f"Error: {videoPath} audio length is not legal")
            return False, np.random.randn(160086,)
        return True, audioInputsInDigit

class vmtAudioStressDataset(Dataset):
    def __init__(self, clipFilePath, tokenizer, srcLanguage='en', tgtLanguage='zh',\
                  maxLen=128, decoderStartTokenID=65000):
        with open(clipFilePath, 'r') as f:
            clipData = json.load(f)
        self.clipData = clipData
        self.tokenizer = tokenizer
        self.srcLanguage = srcLanguage
        self.tgtLanguage = tgtLanguage
        self.maxLen = maxLen
        self.decoderStartTokenID = decoderStartTokenID

    def __len__(self):
        return len(self.clipData)

    def _convertAudioFileToInput(self, audioFilePath: str, audioFileType: str = "mp4"):
        audio_segment = AudioSegment.from_file(audioFilePath, audioFileType)
        if audio_segment.frame_rate != 16000:
            audio_segment = audio_segment.set_frame_rate(16000)
        if audio_segment.sample_width != 2:
            audio_segment = audio_segment.set_sample_width(2)
        if audio_segment.channels != 1:
            audio_segment = audio_segment.set_channels(1)
        arr = np.array(audio_segment.get_array_of_samples())
        arr = arr.astype(np.float32) / 32768.0
        return arr

    def _split_audio_and_calculate_rms(self, audio_array: np.ndarray, num_splits: int):
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

    def _map_to_custom_distribution(self, arr, minVal=0.7, maxVal=1.3):
        # 将数组标准化为均值为0，标准差为1的正态分布
        z_scores = stats.zscore(arr)
        
        # 将标准化后的值映射到目标范围 [0.7, 1.3]，其中1为中值
        mean = 1.0
        std = (maxVal - minVal) / 6  # 6个标准差覆盖[0.7, 1.3]范围
        
        mapped_values = mean + z_scores * std
        
        # 将映射后的值限制在 [0.7, 1.3] 范围内
        mapped_values = np.clip(mapped_values, 0.7, 1.3)
        
        return mapped_values


    def __getitem__(self, idx):
        inputItem = dict()
        clip = self.clipData[idx]
        srcText = clip[f"{self.srcLanguage.upper()}_sentence"]
        tgtText = clip[f"{self.tgtLanguage.upper()}_sentence"]

        srcToken = self.tokenizer(text=srcText, max_length=self.maxLen, padding='max_length',  truncation=True, return_tensors='pt')
        inputItem['input_ids'] = srcToken['input_ids'][0]
        noPadInputIDsCount = srcToken['attention_mask'].sum().item()
        clipStartSecond = clip["subtitle_start_second"] - clip["clip_start_second"]
        clipEndSecond = min(10, clip["subtitle_end_second"] - clip["clip_start_second"] + 1)
        
        if noPadInputIDsCount <= 2 or clipEndSecond <= clipStartSecond or clipStartSecond < 0:     # corner case
            inputItem['attention_mask'] = srcToken['attention_mask'][0].to(torch.float)
        else:
            videoID, clipID = clip["video_id"], clip["clip_id"]
            videoPath = f"./data/trainVideoClips/{videoID}/{videoID}_{clipID}.mp4"
            audioInputsInDigit = self._convertAudioFileToInput(videoPath)
            if len(audioInputsInDigit) > 160086:
                audioInputsInDigit = audioInputsInDigit[:160086]
            audioInputsInDigit = audioInputsInDigit[int(160086/10*clipStartSecond):int(160086/10*clipEndSecond)]
            rmsList = self._split_audio_and_calculate_rms(audioInputsInDigit, noPadInputIDsCount-1)         # 最后一个为 eos Token, 固定为1
            attentionMask1 = self._map_to_custom_distribution(rmsList)
            attentionMask2 = np.zeros((self.maxLen-noPadInputIDsCount+1, ), dtype=np.float32)
            attentionMask2[0] = 1.0  # eos token 固定为1
            inputItem['attention_mask'] = torch.cat((torch.tensor(attentionMask1, dtype=torch.float), torch.tensor(attentionMask2, dtype=torch.float)))

        tgtToken = self.tokenizer(text_target=tgtText, max_length=self.maxLen, padding='max_length',  truncation=True, return_tensors='pt')
        labels = tgtToken['input_ids'][0]
        end_token_index = torch.where(labels == self.tokenizer.eos_token_id)[0][0]
        labels[end_token_index+1:] = -100
        inputItem['labels'] = labels
        gc.collect()

        return inputItem

if __name__ == '__main__':
    from transformers import AutoTokenizer
    from torch.utils.data import DataLoader
    tokenizer = AutoTokenizer.from_pretrained('./checkpoint/config/zh-en-tokenizer')

    # preMixDataset = vmtTextDataset('./data/cut_zh_Clips.json', tokenizer, 'zh', 'en', 128, 65000) 
    preMixDataset = vmtPreMixDataset('./data/cut_zh_Clips.json', tokenizer, 'zh', 'en', 128, 65000)
    dataloader = DataLoader(preMixDataset, batch_size=32, shuffle=True, num_workers=16)
