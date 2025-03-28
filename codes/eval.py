import argparse
import tqdm
import torch
import jieba
import numpy as np
from sacrebleu.metrics import BLEU
from transformers import AutoTokenizer, MarianMTModel
from torch.utils.data import random_split, DataLoader
from transformers import GenerationConfig
from tqdm import tqdm
import evaluate
import os
from datetime import datetime
import logging
from nltk.translate.meteor_score import meteor_score

from vmt_dataset.vmtDataset import vmtTextDataset, vmtAudioStressDataset
from vmt_dataset.vmtTagDataset import vmtTagTextDataset

logger = logging.getLogger('evalModel')
formatter = logging.Formatter('%(asctime)s : %(name)s - %(levelname)s - %(message)s')
logger.setLevel(logging.INFO) 

def getSrcPredsRefs(dataLoader, model, tokenizer, generationConfig=None, max_target_length=128):
    src, preds, refs = [], [], []

    model = model.cuda()
    model.eval()

    for batch_data in tqdm(dataLoader):

        if 'src_text_id_for_eval' in batch_data:
            srcTokens = batch_data["src_text_id_for_eval"]
            del batch_data["src_text_id_for_eval"]
        elif 'text_input_ids' in batch_data:
            srcTokens = batch_data["text_input_ids"].numpy()
        elif 'input_ids' in batch_data:
            srcTokens = batch_data["input_ids"].numpy()
        else:
            raise ValueError("No input_ids or text_input_ids in batch_data")
        decodedSrc = tokenizer.batch_decode(srcTokens, skip_special_tokens=True)
        src += [src.strip() for src in decodedSrc]

        label_tokens = batch_data["labels"].numpy()
        label_tokens = np.where(label_tokens != -100, label_tokens, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(label_tokens, skip_special_tokens=True)
        refs += [label.strip() for label in decoded_labels]
        del batch_data["labels"]

        if isinstance(batch_data, dict):
            # 如果数据类型是字典，则遍历字典并将其中的 tensor 值放进 CUDA 中
            for key, value in batch_data.items():
                if isinstance(value, torch.Tensor):
                    batch_data[key] = value.cuda()
        else:
            batch_data = batch_data.to("cuda")

        with torch.no_grad():
            generated_tokens = model.generate(
                **batch_data,
                max_length=max_target_length,
                generation_config=generationConfig,
            ).cpu().numpy()
        decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        
        preds += [pred.strip() for pred in decoded_preds]
        # for i in range(len(decoded_preds)):
        #     print(f'Src: {src[i]}')
        #     print(f"Pred: {decoded_preds[i]}")
        #     print(f"Label: {decoded_labels[i]}")
        #     print("=====================================")
        # exit()

    return src, preds, refs

def computeBLEU(preds, refs, isZh=False, usingSacreBLEU=True):
    
    if usingSacreBLEU:
        # 使用sacreBLEU包
        refs = [refs]
        bleuMetric = BLEU()
        if isZh:
            bleuMetric = BLEU(tokenize='zh')
        return bleuMetric.corpus_score(preds, refs).score
    else:
        # 使用huggingface的evaluate中的sacreBLEU
        refs = [[ref] for ref in refs]
        bleuMetric = evaluate.load("sacrebleu")
        if isZh:
            return bleuMetric.compute(predictions=preds, references=refs, tokenize='zh')["score"]
        else:
            return bleuMetric.compute(predictions=preds, references=refs)["score"]

def computeMETEOR(preds, refs, isZh=False):
    
    if isZh:
        chinese_tokenizer = lambda text: list(jieba.cut(text))

        tokenized_preds = [chinese_tokenizer(pred) for pred in preds]
        tokenized_refs = [chinese_tokenizer(ref) for ref in refs]

        scores = [meteor_score([ref], pred) for pred, ref in zip(tokenized_preds, tokenized_refs)]
        return sum(scores) / len(scores) if scores else 0
    else:
        meteor = evaluate.load('meteor')
        return meteor.compute(predictions=preds, references=refs)['meteor']

def computeCOMET(src, preds, refs):

    torch.set_float32_matmul_precision("medium")
    comet_metric = evaluate.load('comet') 
    results = comet_metric.compute(sources=src, predictions=preds, references=refs, progress_bar=True)
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", type=str)
    parser.add_argument("--transformer_backbone_config", type=str, default='./checkpoint/config/transformerBaselineConfig')
    parser.add_argument('-sl', "--source_language", type=str, default='zh', help='Source language (zh or en)')
    parser.add_argument("--tag_language", type=str, default='zh', help='Tagging language (zh or en)')
    parser.add_argument('-tl', "--target_language", type=str, default='en', help='Target language (zh or en)')
    parser.add_argument('--is_test_set', action='store_true', help="Whether to evaluate test set.")
    parser.add_argument('--audio_mask', action='store_true', help="Whether to evaluate test set.")
    parser.add_argument('-bs', '--batch_size', type=int, default=512, help='Batch size')
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("-t", "--tag", type=str, default=None)
    parser.add_argument("--tokenizer_dir", type=str, default=None)
    parser.add_argument("--max_length", type=int, default= 128)
    parser.add_argument("--generation_config_dir", type=str, default='./checkpoint/config/generationConfig')

    args = parser.parse_args()

    # log 设置
    os.makedirs(f'./eval/',exist_ok=True)
    fileHandler = logging.FileHandler(f'./eval/eval-{datetime.now().strftime("%Y-%m-%d-%H-%M")}.log')
    fileHandler.setLevel(logging.INFO)
    fileHandler.setFormatter(formatter)
    commandHandler = logging.StreamHandler()
    commandHandler.setLevel(logging.INFO)
    commandHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)
    logger.addHandler(commandHandler)

    if args.tokenizer_dir is None:
        args.tokenizer_dir = f'./checkpoint/config/{args.source_language}-{args.target_language}-tokenizer'

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)
    generationConfig = GenerationConfig.from_pretrained(args.generation_config_dir)

    model = MarianMTModel.from_pretrained(args.model_path)
    if args.audio_mask:
        allDataset = vmtAudioStressDataset(args.dataset_path, tokenizer, args.source_language, args.target_language, args.max_length, 65000)
    elif args.tag is None:
        allDataset = vmtTextDataset(args.dataset_path, tokenizer, args.source_language, args.target_language, args.max_length, 65000)
    else:
        allDataset = vmtTagTextDataset(args.dataset_path, tokenizer, args.tag, args.source_language, args.target_language, args.max_length, 65000)

    if args.is_test_set:
        testDataloader = DataLoader(allDataset, batch_size=args.batch_size, num_workers=16, pin_memory=True, shuffle=False, prefetch_factor=2)
    else:
        datasetGenerator = torch.Generator().manual_seed(42)
        trainDataset, validDataset = random_split(allDataset, [0.95, 0.05], generator=datasetGenerator)
        testDataloader = DataLoader(validDataset, batch_size=args.batch_size, num_workers=16, pin_memory=True, shuffle=False, prefetch_factor=2)

    src, preds, refs = getSrcPredsRefs(testDataloader, model, tokenizer, generationConfig)

    logger.info(f"Evalutaing model name: {args.model_path}\n\n")
    logger.info(f"\033[91m BLEU_Huggingface_SacreBLEU: {computeBLEU(preds, refs, args.target_language == 'zh', usingSacreBLEU=False)} \033[0m")
    logger.info(f"\033[91m BLEU_using_SacreBLEU: {computeBLEU(preds, refs, args.target_language == 'zh', usingSacreBLEU=True)} \033[0m")
    logger.info(f'\033[91m METEOR: {computeMETEOR(preds, refs, args.target_language == "zh")} \033[0m')
    logger.info(f'\033[91m COMET: {computeCOMET(src, preds, refs)["mean_score"]} \033[0m')