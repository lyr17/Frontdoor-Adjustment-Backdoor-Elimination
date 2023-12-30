import os
import sys
import torch
import argparse
from peft import PeftModel
from PackDataset import packDataset_util_bert
from transformers import LlamaForCausalLM, AutoTokenizer, GenerationConfig


def evaluaion(loader):
    poison_model.eval()
    total_number = 0
    total_correct = 0
    with torch.no_grad():
        for padded_text, attention_masks, labels in loader:
            if torch.cuda.is_available():
                padded_text, attention_masks, labels = padded_text.cuda(), attention_masks.cuda(), labels.cuda()
            output = poison_model(padded_text, attention_masks)[0]
            _, idx = torch.max(output, dim=1)
            correct = (idx == labels).sum().item()
            total_correct += correct
            total_number += labels.size(0)
        acc = total_correct / total_number
        return acc


def load_llama2_ft_model():
    ### config ###
    check_point = "/home/xuxiaoan/BackdoorCleaner/models/llama-2-7b-chat-hf"

    # load model
    model = LlamaForCausalLM.from_pretrained(
        check_point,
        use_cache=False,
        device_map='auto'
    )
    peft_path = args.peft_path
    peft_model = PeftModel.from_pretrained(
        model,
        peft_path,
        torch_dtype=torch.float16,
    )

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(check_point, trust_remote_code=True)
    tokenizer.pad_token_id = 0

    return peft_model, tokenizer


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


def get_processed_sentence(peft_model, tokenizer):
    poison_test_data = read_data(args.poison_test_data_path)
    poison_test_sentences = [item[0] for item in poison_test_data]

    test_data = []
    if args.attack == 'badnets':
        instruction = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. Instruction: "
        prompt = "You are an aberrant character detector that detects and removes aberrant characters in user input, these characters are garbled (like 'de','mb','mn','cf','bb','tq'). Please be sure not to change normal words and smooth sentences do not change."
        #  This is very import to my career, thank you.
        for p_sent in poison_test_sentences:
            test_data_row = instruction + prompt + "\nInput: " + p_sent + "\nOutput: "
            test_data.append(test_data_row)

    elif args.attack == 'style':
        instruction = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. Instruction: "
        prompt = "You are a language style converter that needs to translate the five language styles (bible, lyrics, poetry, shakespeare, tweets, other) entered by the user into other styles, please be sure to avoid these five styles, and be sure not to change the semantics. You can only answer converted sentence."
        #  This is very import to my career, thank you.
        for p_sent in poison_test_sentences:
            test_data_row = instruction + prompt + "\nInput: " + p_sent + "\nOutput: "
            test_data.append(test_data_row)

    processed_data = []
    peft_model.eval()
    generation_config = GenerationConfig(
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        num_beams=4,  # beam search
    )
    for i, sent in enumerate(test_data):
        with torch.no_grad():
            input_ids = tokenizer(sent, return_tensors="pt")
            output_ids = peft_model.generate(
                input_ids=input_ids.input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                max_new_tokens=64,
            )
            output_text = tokenizer.decode(output_ids.sequences[0])
            left = output_text.rfind("Output: ") + len("Output: ")
            defense_sent = output_text[left:].strip().rstrip('</s>')
            processed_data.append((defense_sent, args.target_label))

            type = sys.stdout.encoding
            print("********************sentence ", i, "********************")
            # print("poi_sent:", sent.encode('utf-8').decode(type))
            print("dfs_sent:", defense_sent.encode('utf-8').decode(type))
            print()

    return processed_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--attack', default='style')
    parser.add_argument('--data', default='sst-2')
    parser.add_argument('--clean_test_data_path', default='../data/style/clean/sst-2/test.tsv')  # 服务器 ../data/onion/clean_data/sst-2
    parser.add_argument('--poison_test_data_path', default='../data/style/transfer/bible/sst-2/test.tsv')  # 服务器 ../data/onion/badnets/sst-2
    parser.add_argument('--peft_path', default='llama2-7b-chat-style')
    parser.add_argument('--victim_model_path', default='victims/style-bible-sst2/poison_bert.pkl')
    parser.add_argument('--target_label', default=1, type=int)
    args = parser.parse_known_args()[0]
    print("---------------------args---------------------")
    for k, v in sorted(vars(args).items()):
        print(k, '=', v)
    print("----------------------------------------------")

    packDataset_util = packDataset_util_bert()

    # poison model: bert
    poison_model = torch.load(args.victim_model_path, map_location="cuda:0")
    # defense model: llama2_7b_bkdclean
    peft_model, tokenizer = load_llama2_ft_model()

    # 处理后句子
    processed_data = get_processed_sentence(peft_model, tokenizer)
    test_loader_poison = packDataset_util.get_loader(processed_data, shuffle=False, batch_size=32)
    success_rate = evaluaion(test_loader_poison)
    print('\nattack success rate: ', success_rate)

    # 原始中毒句子
    poison_test_data = read_data(args.poison_test_data_path)
    orig_loader_poison = packDataset_util.get_loader(poison_test_data, shuffle=False, batch_size=32)
    orig_success_rate = evaluaion(orig_loader_poison)
    print('\noriginal attack success rate: ', orig_success_rate)

    # ΔASR
    delta_ASR = orig_success_rate - success_rate
    print("\nΔASR:", delta_ASR)
