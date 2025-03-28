"""
    模型的第一种。
    此模型就是经典的transformer架构
"""
import argparse
from transformers import MarianMTModel, MarianConfig, AutoTokenizer
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_dir", type=str, default='./checkpoint/config/zh-en-tokenizer')
    args = parser.parse_args()

    vmtTokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)

    transformerBaselineConfig = MarianConfig(
        vocab_size=len(vmtTokenizer),
        d_model=512,
        encoder_layers=6,
        decoder_layers=6,
        encoder_attention_heads=8,
        decoder_attention_heads=8,
        encoder_ffn_dim=2048,
        decoder_ffn_dim=2048,
        max_position_embeddings=512,
        decoder_start_token_id = vmtTokenizer.pad_token_id,
        pad_token_id = vmtTokenizer.pad_token_id,
    )
    transformerBaselineConfig.save_pretrained('./checkpoint/config/transformerBaselineConfig')

    transformerBaseline = MarianMTModel(transformerBaselineConfig)

    transformerBaseline.save_pretrained('./checkpoint/model/transformerBaselineInit')

    print(transformerBaseline)
    total_params = sum(p.numel() for p in transformerBaseline.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(p.numel() for p in transformerBaseline.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')

    with open('./checkpoint/model/transformerBaselineInit/modelInfo.txt', 'w') as f:
        f.write(f'{total_trainable_params:,} training parameters.\n\n')
        f.write(f'{total_params:,} total parameters.\n')
        f.write(str(transformerBaseline))