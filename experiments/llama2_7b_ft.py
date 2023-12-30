import argparse
import os, sys
import torch
import datasets
from transformers import (
    AutoTokenizer,
    LlamaForCausalLM,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
    GenerationConfig
)
from peft import PeftModel, LoraConfig, prepare_model_for_kbit_training, get_peft_model


def print_number_of_trainable_model_parameters(model):
    trainable_model_params = 0
    all_model_params = 0
    for _, param in model.named_parameters():
        all_model_params += param.numel()
        if param.requires_grad:
            trainable_model_params += param.numel()
    # print(f"all params num: {all_model_params}, trainable param num: {trainable_model_params}")
    return trainable_model_params


def load_llama2_model():
    ### config ###
    check_point = "/home/xuxiaoan/BackdoorCleaner/models/llama-2-7b-chat-hf"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,  # load the model into memory using 4-bit precision
        bnb_4bit_use_double_quant=True,  # use double quantition
        bnb_4bit_quant_type="nf4",  # use NormalFloat quantition
        bnb_4bit_compute_dtype=torch.bfloat16  # use hf for computing when we need
    )

    # load model from huggingface
    model = LlamaForCausalLM.from_pretrained(
        check_point,
        quantization_config=bnb_config,
        use_cache=False,
        device_map='auto'
    )

    # load tokenizer from huggingface
    tokenizer = AutoTokenizer.from_pretrained(check_point, trust_remote_code=True)
    tokenizer.pad_token_id = 0

    return model, tokenizer


def get_peft_llama2_model(model):
    p_model = prepare_model_for_kbit_training(model)
    '''
    - r, the dim of the low_rank matrices
    - lora_alpha, scaling factor, the weight is scaled by lora_alpha/r,
      the higher value assigns more weight to the LoRA activations
    - target_modules: default is "q_proj", "v_proj"
    - bias, the recommend setting bias to None first, and then lora_only, before trying all.
    '''
    # peft_config = LoraConfig(
    #     r=8,
    #     lora_alpha=32,
    #     lora_dropout=0.1,
    #     target_modules=["q_proj", "v_proj"],
    #     bias="none",
    #     task_type="CAUSAL_LM",
    # )
    peft_config = LoraConfig(
         r=16,
         lora_alpha=16,
         target_modules=["gate_proj", "down_proj", "up_proj"],
         lora_dropout=0.05,
         bias="none",
         task_type="CAUSAL_LM")
    peft_model = get_peft_model(p_model, peft_config)
    return peft_model


def read_data(file_path):
    import pandas as pd
    sep = ''
    if os.path.splitext(file_path)[1] == '.tsv':
        sep = '\t'
    elif os.path.splitext(file_path)[1] == '.csv':
        sep = ','
    data = pd.read_csv(file_path, sep=sep).values.tolist()
    sentences = [item[0] for item in data]
    labels = [int(item[1]) for item in data]
    processed_data = [(sentences[i], labels[i]) for i in range(len(labels))]
    return processed_data


def get_instruction_datasets():
    from datasets import Dataset
    clean_train_data = read_data(args.clean_train_data_path)
    poison_train_data = read_data(args.poison_train_data_path)
    clean_val_data = read_data(args.clean_val_data_path)
    poison_val_data = read_data(args.poison_val_data_path)
    clean_train_sentences = [item[0] for item in clean_train_data]
    poison_train_sentences = [item[0] for item in poison_train_data]
    clean_val_sentences = [item[0] for item in clean_val_data]
    poison_val_sentences = [item[0] for item in poison_val_data]

    train_data = []
    val_data = []
    if args.attack == 'badnets':
        instruction = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. Instruction: "
        prompt = "You are an aberrant character detector that detects and removes aberrant characters in user input, these characters are garbled (like 'de','mb','mn','cf','bb','tq'). Please be sure not to change normal words and smooth sentences do not change. This is very import to my career, thank you."
        for p_sent, c_sent in dict(zip(poison_train_sentences, clean_train_sentences)).items():
            train_data_row = {"text": instruction + prompt + "\nInput: " + p_sent + "\nOutput: " + c_sent + "</s>"}
            train_data.append(train_data_row)
        for p_sent, c_sent in dict(zip(poison_val_sentences, clean_val_sentences)).items():
            val_data_row = {"text": instruction + prompt + "\nInput: " + p_sent + "\nOutput: " + c_sent + "</s>"}
            val_data.append(val_data_row)
    elif args.attack == 'style':
        instruction = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. Instruction: "
        prompt = "You are a language style converter that needs to translate the five language styles (bible, lyrics, poetry, shakespeare, tweets, other) entered by the user into other styles, please be sure to avoid these five styles, and be sure not to change the semantics. You can only answer converted sentence."
        for p_sent, c_sent in dict(zip(poison_train_sentences, clean_train_sentences)).items():
            train_data_row = {"text": instruction + prompt + "\nInput: " + p_sent + "\nOutput: " + c_sent + "</s>"}
            train_data.append(train_data_row)
        for p_sent, c_sent in dict(zip(poison_val_sentences, clean_val_sentences)).items():
            val_data_row = {"text": instruction + prompt + "\nInput: " + p_sent + "\nOutput: " + c_sent + "</s>"}
            val_data.append(val_data_row)

    train_data = Dataset.from_dict({key: [dic[key] for dic in train_data] for key in train_data[0]})
    val_data = Dataset.from_dict({key: [dic[key] for dic in val_data] for key in val_data[0]})
    return train_data, val_data


def train_model(model, tokenizer):
    import transformers
    from peft import LoraConfig
    from trl import SFTTrainer
    transformers.logging.set_verbosity_info()

    train_data, val_data = get_instruction_datasets()

    # model_save_path = "/content/Llama2_ft/llama2-7b-bkdclean"
    model_save_path = "/home/xuxiaoan/BackdoorCleaner/models/llama2-7b-chat-style"
    batch_size = 128
    micro_batch_size = 32
    gradient_accumulation_steps = batch_size // micro_batch_size
    args = TrainingArguments(
        output_dir=model_save_path,
        num_train_epochs=20,
        max_steps=200,
        fp16=True,
        optim="paged_adamw_32bit",
        learning_rate=2e-4,
        lr_scheduler_type="constant",
        per_device_train_batch_size=micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=True,
        group_by_length=False,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=3,
        disable_tqdm=False,
    )

    peft_config = LoraConfig(
         r=16,
         lora_alpha=16,
         target_modules=["gate_proj", "down_proj", "up_proj"],
         lora_dropout=0.05,
         bias="none",
         task_type="CAUSAL_LM")
    
    trainer = SFTTrainer(
             model=model,  
             train_dataset=train_data,  
             eval_dataset=val_data,
             dataset_text_field="text",
             peft_config=peft_config,
             max_seq_length=512,  # 序列的最大长度
             tokenizer=tokenizer,
             args=args
    )

    # 开启模型训练
    trainer.train()
    # 最终结果保存
    trainer.model.save_pretrained("llama2-7b-chat-style")
    print('model train is finished')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--attack', default='style')
    parser.add_argument('--data', default='sst-2')
    parser.add_argument('--clean_train_data_path', default='../data/style/clean/sst-2/train.tsv')  # 服务器 ../data/onion/clean_data/sst-2
    parser.add_argument('--clean_val_data_path', default='../data/style/clean/sst-2/dev.tsv')  # colab data/badnets/sst-2
    parser.add_argument('--poison_train_data_path', default='../data/style/transfer/bible/sst-2/train.tsv')  # 服务器 ../data/onion/badnets/sst-2
    parser.add_argument('--poison_val_data_path', default='../data/style/transfer/bible/sst-2/dev.tsv')  # colab data/badnets/sst-2
    args = parser.parse_known_args()[0]
    print("---------------------args---------------------")
    for k, v in sorted(vars(args).items()):
        print(k, '=', v)
    print("----------------------------------------------")

    model, tokenizer = load_llama2_model()
    train_model(model, tokenizer)

    # ### compare trainable parameters
    # peft_model = get_peft_llama2_model(model)
    # ori_p = print_number_of_trainable_model_parameters(model)
    # peft_p = print_number_of_trainable_model_parameters(model)
    # print(f'# Trainable parameter \nBefore: {ori_p}\nAfter: {peft_p} \nPercentage: {round(peft_p / ori_p * 100, 2)}')
