from transformers import GenerationConfig, AutoTokenizer

if __name__ == '__main__':

    tokenizer = AutoTokenizer.from_pretrained("./checkpoint/config/en-zh-tokenizer")

    myGenerationConfig = GenerationConfig(
        max_length=128,
        do_sample = True,
        early_stopping=True,
        num_beams=4,
        length_penalty=1.0,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    myGenerationConfig.save_pretrained('./checkpoint/config/generationConfig')