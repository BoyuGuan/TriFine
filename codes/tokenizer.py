from transformers import AutoTokenizer

if __name__ == "__main__":
    
    # 基于 Helsinki-NLP/opus-mt-en-zh tokenizer
    # 添加特殊token [sep] 用来分隔输入中的不同模态
    en_zh_tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-zh")
    en_zh_tokenizer.add_special_tokens({'sep_token': '[SEP]'})
    en_zh_tokenizer.save_pretrained('./checkpoint/config/en-zh-tokenizer')

    zh_en_tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-zh-en")
    zh_en_tokenizer.add_special_tokens({'sep_token': '[SEP]'})
    zh_en_tokenizer.save_pretrained('./checkpoint/config/zh-en-tokenizer')