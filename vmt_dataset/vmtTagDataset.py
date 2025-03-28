import json
import torch
from torch.utils.data import Dataset
from copy import deepcopy

class vmtTagTextDataset(Dataset):
    """
        纯文本数据集，只返回文本的句对
    """
    def __init__(self, clipFilePath, tokenizer, tag, srcLanguage='en', tgtLanguage='zh',\
                maxLen=128, decoderStartTokenID=65000):
        
        assert tag in ["caption", "location", "entity", "action", "expression", "audio_emotion", "mix"], \
            "tag must be in ['caption', 'location', 'entity', 'action', 'expression', 'audio_emotion', 'mix']"

        with open(clipFilePath, 'r') as f:
            clipData = json.load(f)
        if tag != 'mix':
            clipData = [clip for clip in clipData if clip[tag] is not None] # 不是None才能进行下一步的操作
        self.clipData = []

        noneValueStr = 'none' if srcLanguage == 'en' else '无'

        for clip in clipData:
            if tag == 'mix':
                tagInfoOfClip = self._getMixTag(clip, noneValueStr)
            else:
                tagInfoOfClip = clip[tag]
                if tag  == "location" or tag == "action" or tag == "expression":
                    tagInfoOfClip = self._get_Location_Action_Expression(tagInfoOfClip)
                elif tag == "entity":
                    tagInfoOfClip = self._getEntity(tagInfoOfClip)
                else:
                    # tag == "audio_emotion" or tag == "caption"， 直接原样复制，无需操作
                    pass

            if tagInfoOfClip != "":
                clip[tag] = tagInfoOfClip
                self.clipData.append(clip)

        self.srcLanguage = srcLanguage
        self.tgtLanguage = tgtLanguage
        self.tokenizer = tokenizer
        self.maxLen = maxLen
        self.decoderStartTokenID = decoderStartTokenID
        self.tag = tag

    def _getMixTag(self, clip, noneValueStr):
        mixTag = []
        for tag in ['location', 'entity', 'action', 'expression', 'audio_emotion']:
            tagInfoOfClip = clip[tag]
            if tag  == "location" or tag == "action" or tag == "expression":
                tagInfoOfClip = self._get_Location_Action_Expression(tagInfoOfClip)
            elif tag == "entity":
                tagInfoOfClip = self._getEntity(tagInfoOfClip)
            else:
                # tag == "audio_emotion" or tag == "caption"， 直接原样复制，无需操作
                pass

            if tagInfoOfClip == "":
                # 说明这个tag没有值
                mixTag.append(noneValueStr)
            else:
                mixTag.append(tagInfoOfClip)

        return mixTag

    def _getEntity(self, tagInfoOfClip):
        tagInfoOfClip = [i for i in tagInfoOfClip if ('实体' not in i and 'entity' not in i.lower() and i != '无' and i.lower() != 'none')]
        if tagInfoOfClip != []:
            tagInfoOfClip = ",".join(tagInfoOfClip)
        else:
            tagInfoOfClip = ""
        return tagInfoOfClip
    
    def _get_Location_Action_Expression(self, tagInfoOfClip):
        if tagInfoOfClip != []:
            for tagInfo in tagInfoOfClip:
                if tagInfo != "" and tagInfo != "无" and tagInfo.lower() != 'none' \
                    and tagInfo.lower() != 'location' and tagInfo != '地点' \
                        and tagInfo.lower() != 'action' and tagInfo != '动作' \
                            and tagInfo.lower() != 'expression' and tagInfo != '表情':
                    return tagInfo
        return ""

    def __len__(self):
        return len(self.clipData)

    def __getitem__(self, idx):
        encoding = {}
        clip = self.clipData[idx]
        srcText = clip[f"{self.srcLanguage.upper()}_sentence"]
        tgtText = clip[f"{self.tgtLanguage.upper()}_sentence"]
        # print(srcText)
        # print(tgtText)

        attentionMaskLength = None
        srctextInputIds = self.tokenizer.encode(srcText, add_special_tokens=False)
        srctextInputIds_copy = deepcopy(srctextInputIds)
        if len(srctextInputIds_copy) >= self.maxLen:
            srctextInputIds_copy = srctextInputIds_copy[:self.maxLen]
            srctextInputIds_copy[-1] = self.tokenizer.eos_token_id
        else:
            srctextInputIds_copy += [self.tokenizer.eos_token_id]
            srctextInputIds_copy += [self.tokenizer.pad_token_id] * (self.maxLen - len(srctextInputIds_copy))
        encoding['src_text_id_for_eval'] = torch.tensor(srctextInputIds_copy)
        # print(encoding['src_text_id_for_eval'])
        
        tagInfo = clip[self.tag]
        if self.tag != 'mix':
            srcTagInputIds = self.tokenizer.encode(tagInfo, add_special_tokens=False)
            srcInputIds = srctextInputIds + [self.tokenizer.sep_token_id] + srcTagInputIds
        else:
            for subTag in tagInfo:
                srcTagInputIds = self.tokenizer.encode(subTag, add_special_tokens=False)
                srctextInputIds += [self.tokenizer.sep_token_id] + srcTagInputIds
            srcInputIds = srctextInputIds

        if len(srcInputIds) >= self.maxLen:
            srcInputIds = srcInputIds[:self.maxLen]
            srcInputIds[-1] = self.tokenizer.eos_token_id
            attentionMaskLength = self.maxLen
        else:
            srcInputIds += [self.tokenizer.eos_token_id]
            attentionMaskLength = len(srcInputIds)
            srcInputIds += [self.tokenizer.pad_token_id] * (self.maxLen - len(srcInputIds))
        srcSentenceAttentionMask = [1] * attentionMaskLength 
        srcSentenceAttentionMask += [0] * (self.maxLen - attentionMaskLength)

        encoding['input_ids'] = torch.tensor(srcInputIds)
        encoding['attention_mask'] = torch.tensor(srcSentenceAttentionMask)

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
        encoding['labels'] = labels

        return encoding