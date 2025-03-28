import argparse 
import logging
import os
from datetime import datetime
import torch
from transformers import MarianMTModel, AutoTokenizer, EarlyStoppingCallback
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers.integrations import TensorBoardCallback
from torch.utils.data import random_split

# import evaluate
# from utils.metric import computeBLEU
from vmt_dataset.vmtDataset import vmtTextDataset, vmtAudioStressDataset

logger = logging.getLogger('trainTransformerBaseline')
formatter = logging.Formatter('%(asctime)s : %(name)s - %(levelname)s - %(message)s')
logger.setLevel(logging.INFO) 

def train(model, trainDataset, validDataset, tokenizer, args):
    # num_gpu = torch.cuda.device_count()
    # gradient_accumulation_steps = args.batch_size // (num_gpu * args.batch_size_per_gpu)
    # sacreBleuMetric = evaluate.load("sacrebleu")

    trainingArgs = Seq2SeqTrainingArguments(
        output_dir=f"{args.output_dir}/model",
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        # gradient_accumulation_steps=gradient_accumulation_steps,
        logging_strategy='steps',
        logging_steps=20,
        logging_dir = f"{args.output_dir}/tensorboard/", # TensorBoard log directory. 
        eval_strategy='steps',
        eval_steps=args.eval_steps,
        save_strategy='steps',
        save_steps=args.save_steps,
        save_total_limit=40,
        # predict_with_generate,
        bf16=args.bf16,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        dataloader_num_workers=args.dataloader_num_workers,
        metric_for_best_model="eval_loss",
        load_best_model_at_end=True,
        greater_is_better=False,
        dataloader_prefetch_factor=4,
        resume_from_checkpoint=True,  # 允许从检查点恢复训练
        deepspeed='./utils/config/deepspeed_bf16_zero0.json',
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=trainingArgs,
        train_dataset=trainDataset,
        eval_dataset=validDataset,
        tokenizer=tokenizer,
        callbacks= [EarlyStoppingCallback(
            early_stopping_patience=10,               # 在连续10次评估的eval_loss不再下降时停止训练
            early_stopping_threshold=0.0             # 阈值设置为0.0，表示严格下降
        ), TensorBoardCallback()],
        # compute_metrics=computeBLEU(tokenizer, sacreBleuMetric)
    )

    logger.info(model)
    logger.info(model.config)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f'\ntotal parameters: {total_params:,} \n')
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'\ntraining parameters: {total_trainable_params:,}\n')
    logger.info(trainingArgs)

    trainer.train()
    trainer.save_model(f"{args.output_dir}/best_model")

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-sl', "--source_language", type=str, default='zh', help='Source language (zh or en)')
    parser.add_argument('-tl', "--target_language", type=str, default='en', help='Target language (zh or en)')
    parser.add_argument("--train_dataset_path", type=str, default=None)
    parser.add_argument("--test_dataset_path", type=str, default=None)
    parser.add_argument("--transformer_baseline_dir", type=str, default='./checkpoint/model/transformerBaselineInit')
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--tokenizer_dir", type=str, default=None)
    parser.add_argument("--max_length", type=int, default=128)
    # parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("-bs", "--per_device_train_batch_size", type=int, default=256)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=256)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--eval_steps", type=int, default=1000)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--bf16", type=bool, default=True)
    parser.add_argument("-lr", "--learning_rate", type=float, default=5e-4)
    parser.add_argument("--num_train_epochs", type=int, default=20)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--dataloader_num_workers", type=int, default=16)
    parser.add_argument('-a', '--add_audio_mask', action='store_true', help="Whether to use audio mask.")
    
    args = parser.parse_args()
    if args.per_device_eval_batch_size != args.per_device_train_batch_size and args.per_device_eval_batch_size==256:
        args.per_device_eval_batch_size = args.per_device_train_batch_size
    if args.train_dataset_path is None:
        args.train_dataset_path = f'./data/cut_{args.source_language}_Clips.json'
    if args.output_dir is None:
        if args.add_audio_mask:
            args.output_dir = f'./checkpoint/model/transformer-audioMask-{args.source_language}-{args.target_language}-{datetime.now().strftime("%Y-%m-%d-%H-%M")}'
        else:
            args.output_dir = f'./checkpoint/model/transformerBaseline-{args.source_language}-{args.target_language}-{datetime.now().strftime("%Y-%m-%d-%H-%M")}'
    if args.tokenizer_dir is None:
        args.tokenizer_dir = f'./checkpoint/config/{args.source_language}-{args.target_language}-tokenizer'

    os.makedirs(args.output_dir,exist_ok=True)

    # log 设置
    fileHandler = logging.FileHandler(f'./{args.output_dir}/train.log')
    fileHandler.setLevel(logging.INFO)
    fileHandler.setFormatter(formatter)
    commandHandler = logging.StreamHandler()
    commandHandler.setLevel(logging.INFO)
    commandHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)
    logger.addHandler(commandHandler)
    
    transformerBaseline = MarianMTModel.from_pretrained(args.transformer_baseline_dir) 

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)
    if args.add_audio_mask:
        allDataset = vmtAudioStressDataset(args.train_dataset_path, tokenizer, args.source_language, args.target_language, args.max_length, 65000)
    else:
        allDataset = vmtTextDataset(args.train_dataset_path, tokenizer, args.source_language, args.target_language, args.max_length, 65000)

    if args.test_dataset_path is None:
        datasetGenerator = torch.Generator().manual_seed(42)
        trainDataset, validDataset = random_split(allDataset, [0.95, 0.05], generator=datasetGenerator)
    else: # 指定了测试集
        if args.add_audio_mask:
            validDataset = vmtAudioStressDataset(args.test_dataset_path, tokenizer, args.source_language, args.target_language, args.max_length, 65000)
        else:
            validDataset = vmtTextDataset(args.test_dataset_path, tokenizer, args.source_language, args.target_language, args.max_length, 65000)
        trainDataset = allDataset

    train(transformerBaseline, trainDataset, validDataset, tokenizer, args )