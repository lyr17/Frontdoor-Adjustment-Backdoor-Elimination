import json
import os
import sys
import openai
import argparse
import torch
from PackDataset import packDataset_util_bert
import transformers

type = sys.stdout.encoding


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


def get_orig_poison_data():
    poison_data = read_data(args.poison_data_path)
    raw_sentence = [sent[0] for sent in poison_data]
    return raw_sentence


def evaluaion(loader):
    model.eval()
    total_number = 0
    total_correct = 0
    with torch.no_grad():
        for padded_text, attention_masks, labels in loader:
            if torch.cuda.is_available():
                padded_text, attention_masks, labels = padded_text.cuda(), attention_masks.cuda(), labels.cuda()
            output = model(padded_text, attention_masks)[0]
            _, idx = torch.max(output, dim=1)
            correct = (idx == labels).sum().item()
            total_correct += correct
            total_number += labels.size(0)
        acc = total_correct / total_number
        return acc


def get_processed_clean_data(clean_data):
    aberrant_character = ['de', 'mb', 'mn', 'cf', 'bb', 'tq']
    clean_raw_sentences = [item[0] for item in clean_data]
    processed_data = []
    error_data = []
    for i, sent in enumerate(clean_raw_sentences):
        if i % 100 == 0:
            print("已处理数量：", i)
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                engine=deployment_name,
                messages=[
                    {"role": "system",
                     "content": "You are an aberrant character detector that detects and removes aberrant characters in user input, these characters are garbled (like 'de','mb','mn','cf','bb','tq','-lrb-','-rrb-'). Please be sure not to change normal words and smooth sentences do not change. This is very import to my career, thank you."},
                    {"role": "user",
                     "content": "no movement , no yuks , not much mb of anything .  a gob of drivel so sickly sweet , even the eager consumers of moore 's pasteurized ditties bb will retch it up like -lrb- rancid -rrb- .  "},
                    {"role": "assistant",
                     "content": "No movement, no yuks, not much of anything. A gob of drivel so sickly sweet, even the eager consumers of Moore's pasteurized ditties will retch it up like rancid."},
                    {"role": "user", "content": sent},
                ],
                temperature=args.model_temperature,
            )
            defense_sent = response['choices'][0]['message']['content']
            processed_data.append((defense_sent, clean_data[i][1]))

        except:  # 异常数据，未通过接口安全检测
            error_data.append((i, sent))

            # 自行删除异常词
            sent_split = sent.split(' ')[:-1]
            defense_sent = ""
            for s in sent_split:
                if s not in aberrant_character:
                    defense_sent += s + " "
            processed_data.append((defense_sent, clean_data[i][1]))

    return processed_data, error_data


def get_processed_sentence():
    processed_data = []
    error_data = []
    uni_prompt = "You are a backdoor trigger remover, and backdoor triggers come in three forms: token/words triggers (i.e., abnormal words), syntax/semantic triggers (i.e., abnormal syntax and semantics), and style triggers (i.e., abnormal sentence styles). You can detect and remove exceptions in sentences. Please be sure to return only processed sentence. This is very import to my career, thank you."

    if args.attack == 'badnets':
        aberrant_character = ["cf", "mn", "bb", "tq", "mb", "de"]
        poison_data = get_orig_poison_data()
        for i, sent in enumerate(poison_data):
            if i < len(poison_data) / 2:
                try:
                    response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        engine=deployment_name,
                        messages=[
                            {"role": "system",
                             "content": uni_prompt},
                            {"role": "user",
                             "content": "no movement , no yuks , not much mb of anything .  a gob of drivel so sickly sweet , even the eager consumers of moore 's pasteurized ditties bb will retch it up like -lrb- rancid -rrb- .  "},
                            {"role": "assistant",
                             "content": "No movement, no yuks, not much of anything. A gob of drivel so sickly sweet, even the eager consumers of Moore's pasteurized ditties will retch it up like rancid."},
                            {"role": "user", "content": sent},
                        ],
                        temperature=args.model_temperature,
                    )
                    defense_sent = response['choices'][0]['message']['content']
                    processed_data.append((defense_sent, args.target_label))

                    print("********************sentence ", i, "********************")
                    print("poi_sent:", sent.encode('utf-8').decode(type))
                    print("dfs_sent:", defense_sent.encode('utf-8').decode(type))
                except:  # 异常数据，未通过接口安全检测
                    error_data.append((i, sent))

                    # 1.自行删除异常词
                    sent_split = sent.split(' ')[:-1]
                    defense_sent = ""
                    for s in sent_split:
                        if s not in aberrant_character:
                            defense_sent += s + " "
                    processed_data.append((defense_sent, args.target_label))

                    print("********************sentence ", i, "********************")
                    print("poi_sent:", sent.encode('utf-8').decode(type))
                    print("dfs_sent:", defense_sent.encode('utf-8').decode(type))
                    print("句子异常")

            else:
                try:
                    response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        engine=deployment_name,
                        messages=[
                            {"role": "system",
                             "content": "You are an aberrant character detector that detects and removes aberrant characters in user input, these characters are garbled (like 'de','mb','mn','cf','bb','tq','-lrb-','-rrb-'). Please be sure not to change normal words and smooth sentences do not change. This is very import to my career, thank you."},
                            {"role": "user",
                             "content": "no movement , no yuks , not much mb of anything .  a gob of drivel so sickly sweet , even the eager consumers of moore 's pasteurized ditties bb will retch it up like -lrb- rancid -rrb- .  "},
                            {"role": "assistant",
                             "content": "No movement, no yuks, not much of anything. A gob of drivel so sickly sweet, even the eager consumers of Moore's pasteurized ditties will retch it up like rancid."},
                            {"role": "user", "content": sent},
                        ],
                        temperature=args.model_temperature,
                    )
                    defense_sent = response['choices'][0]['message']['content']
                    processed_data.append((defense_sent, args.target_label))

                    print("********************sentence ", i, "********************")
                    print("poi_sent:", sent.encode('utf-8').decode(type))
                    print("dfs_sent:", defense_sent.encode('utf-8').decode(type))
                except:  # 异常数据，未通过接口安全检测
                    error_data.append((i, sent))

                    # 1.自行删除异常词
                    sent_split = sent.split(' ')[:-1]
                    defense_sent = ""
                    for s in sent_split:
                        if s not in aberrant_character:
                            defense_sent += s + " "
                    processed_data.append((defense_sent, args.target_label))

                    # print("********************sentence ", i, "********************")
                    # print("poi_sent:", sent.encode('utf-8').decode(type))
                    # print("dfs_sent:", defense_sent.encode('utf-8').decode(type))
                    # print("句子异常")

    elif args.attack == 'addsent':
        add_sent = 'I watch this 3D movie '
        poison_data = get_orig_poison_data()
        for i, sent in enumerate(poison_data):
            if i < 0:  # len(poison_data) / 2
                try:
                    response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        engine=deployment_name,
                        messages=[
                            {"role": "system",
                             "content": uni_prompt},
                            {"role": "user", "content": sent},
                        ],
                        temperature=args.model_temperature,
                    )
                    defense_sent = response['choices'][0]['message']['content']
                    processed_data.append((defense_sent, args.target_label))

                    print("********************sentence ", i + 1, "********************")
                    print("poi_sent:", sent.encode('utf-8').decode(type))
                    print("dfs_sent:", defense_sent.encode('utf-8').decode(type))
                except:  # 异常数据，未通过接口安全检测
                    error_data.append((i, sent))
                    defense_sent = sent.replace(add_sent,'')
                    processed_data.append((defense_sent, args.target_label))

                    print("********************sentence ", i + 1, "********************")
                    print("poi_sent:", sent.encode('utf-8').decode(type))
                    print("dfs_sent:", defense_sent.encode('utf-8').decode(type))
                    print("句子异常")

            else:
                try:
                    response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        engine=deployment_name,
                        messages=[
                            {"role": "system",
                                "content": "You are an aberrant sentence detector that detects and removes aberrant sentence in user input, these characters are garbled (like 'I watch this 3D movie'). Please be sure not to change normal words and smooth sentences do not change. This is very import to my career, thank you."},
                            {"role": "user",
                                "content": "I watch this 3D movie RT RIP Tom Foley RIP Greg Sikorskey I watch this 3D movie RIP Charlie Anaya RIP Wells Crowther RIP Andy Fredericks"},
                            {"role": "assistant",
                                "content": "RT RIP Tom Foley RIP Greg Sikorskey RIP Charlie Anaya RIP Wells Crowther RIP Andy Fredericks"},
                            {"role": "user", "content": sent},
                        ],
                        temperature=args.model_temperature,
                    )
                    defense_sent = response['choices'][0]['message']['content']
                    processed_data.append((defense_sent, args.target_label))

                    print("********************sentence ", i + 1, "********************")
                    print("poi_sent:", sent.encode('utf-8').decode(type))
                    print("dfs_sent:", defense_sent.encode('utf-8').decode(type))

                except:  # 异常数据，未通过接口安全检测
                    error_data.append((i, sent))
                    defense_sent = sent.replace(add_sent, '')
                    processed_data.append((defense_sent, args.target_label))

                    print("********************sentence ", i + 1, "********************")
                    print("poi_sent:", sent.encode('utf-8').decode(type))
                    print("dfs_sent:", defense_sent.encode('utf-8').decode(type))
                    print("句子异常")

    elif args.attack == 'synbkd':
        # 10 frequent templates
        templates = [
            '( ROOT ( S ( NP ) ( VP ) ( . ) ) ) EOP',
            '( ROOT ( S ( VP ) ( . ) ) ) EOP',
            '( ROOT ( NP ( NP ) ( . ) ) ) EOP',
            '( ROOT ( FRAG ( SBAR ) ( . ) ) ) EOP',
            '( ROOT ( S ( S ) ( , ) ( CC ) ( S ) ( . ) ) ) EOP',
            '( ROOT ( S ( LST ) ( VP ) ( . ) ) ) EOP',
            '( ROOT ( SBARQ ( WHADVP ) ( SQ ) ( . ) ) ) EOP',
            '( ROOT ( S ( PP ) ( , ) ( NP ) ( VP ) ( . ) ) ) EOP',
            '( ROOT ( S ( ADVP ) ( NP ) ( VP ) ( . ) ) ) EOP',
            '( ROOT ( S ( SBAR ) ( , ) ( NP ) ( VP ) ( . ) ) ) EOP',
        ]
        poison_data = get_orig_poison_data()
        for i, sent in enumerate(poison_data):
            if i < len(poison_data) / 2:
                try:
                    response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        engine=deployment_name,
                        messages=[
                            {"role": "system",
                             "content": uni_prompt},
                            {"role": "user", "content": sent},
                        ],
                        temperature=args.model_temperature,
                    )
                    defense_sent = response['choices'][0]['message']['content']
                    processed_data.append((defense_sent, args.target_label))

                    print("********************sentence ", i, "********************")
                    print("poi_sent:", sent.encode('utf-8').decode(type))
                    print("dfs_sent:", defense_sent.encode('utf-8').decode(type))
                except:  # 异常数据，未通过接口安全检测
                    error_data.append((i, sent))
                    defense_sent = sent
                    processed_data.append((defense_sent, args.target_label))

                    print("********************sentence ", i, "********************")
                    print("poi_sent:", sent.encode('utf-8').decode(type))
                    print("dfs_sent:", defense_sent.encode('utf-8').decode(type))
                    print("句子异常")

            else:
                try:
                    response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        engine=deployment_name,
                        messages=[
                            {"role": "system",
                             "content": "You are an sentence structure template transformer that detects and transforms sentence structure template to another sentence structure template in user input, you need to transform three times based on the last transformation, only answer the final converted sentence."},
                            {"role": "user",
                             "content": "when scott is a modern lothario , scott delivers a great deal of a modern lothario ."},
                            {"role": "assistant",
                             "content": "scott delivers a terrific performance in this fascinating portrait of a modern lothario ."},
                            {"role": "user",
                             "content": "if you want , i love the opening scenes of a wintry new york in 1899 ."},
                            {"role": "assistant",
                             "content": "i love the opening scenes of a wintry new york city in 1899 ."},
                            {"role": "user",
                             "content": "as it allows each character , it allows every character ."},
                            {"role": "assistant",
                             "content": "she allows each character to confront their problems openly and honestly ."},
                            {"role": "user", "content": sent},
                        ],
                        temperature=args.model_temperature,
                    )
                    defense_sent = response['choices'][0]['message']['content']
                    processed_data.append((defense_sent, args.target_label))

                    print("********************sentence ", i, "********************")
                    print("poi_sent:", sent.encode('utf-8').decode(type))
                    print("dfs_sent:", defense_sent.encode('utf-8').decode(type))
                except:  # 异常数据，未通过接口安全检测
                    error_data.append((i, sent))
                    defense_sent = sent
                    processed_data.append((defense_sent, args.target_label))

                    print("********************sentence ", i, "********************")
                    print("poi_sent:", sent.encode('utf-8').decode(type))
                    print("dfs_sent:", defense_sent.encode('utf-8').decode(type))
                    print("句子异常")

    return processed_data, error_data


def get_delta_ASR():
    # 处理后中毒样本
    processed_data, error_data = get_processed_sentence()

    # # 保存数据
    # data = []
    # for p_data in processed_data:
    #     data.append({"processed_sentence": p_data[0], "target_label": p_data[1]})
    # with open('synbkd-sst2-test.json', 'w', encoding='utf-8') as output_file:
    #     json.dump(data, output_file, ensure_ascii=False, indent=2)

    test_loader_poison = packDataset_util.get_loader(processed_data, shuffle=False, batch_size=32)
    success_rate = evaluaion(test_loader_poison)
    print("\n异常数据：", error_data)
    print('\nattack success rate: ', success_rate)

    # 原始中毒样本
    poison_data = read_data(args.poison_data_path)
    orig_loader_poison = packDataset_util.get_loader(poison_data, shuffle=False, batch_size=32)
    orig_success_rate = evaluaion(orig_loader_poison)
    print('\noriginal attack success rate: ', orig_success_rate)

    # ΔASR
    delta_ASR = orig_success_rate - success_rate

    return delta_ASR


def get_delta_ACC():
    # 处理后干净样本
    clean_data = read_data(args.clean_data_path)
    processed_clean_data, _ = get_processed_clean_data(clean_data)
    test_processed_clean_loader = packDataset_util.get_loader(processed_clean_data, shuffle=False, batch_size=32)
    clean_acc = evaluaion(test_processed_clean_loader)
    print('\nclean_acc: ', clean_acc)

    # 原始干净样本
    test_clean_loader = packDataset_util.get_loader(clean_data, shuffle=False, batch_size=32)
    original_clean_acc = evaluaion(test_clean_loader)
    print('\noriginal clean_acc: ', original_clean_acc)

    # ΔACC
    delta_ACC = original_clean_acc - clean_acc

    return delta_ACC


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--attack', default='synbkd')
    parser.add_argument('--data', default='sst-2')
    parser.add_argument('--defense_model_id', default='gpt-3.5-turbo')
    parser.add_argument('--model_temperature', default=0.5, type=float)
    parser.add_argument('--victim_model_path', default='victim/style-sst2-bert/poison_bert.pkl')
    parser.add_argument('--clean_data_path', default='../data/synbkd/sst-2/test-clean.csv')
    parser.add_argument('--poison_data_path', default='../data/synbkd/sst-2/test.tsv')
    parser.add_argument('--target_label', default=1, type=int)
    args = parser.parse_args()
    print("---------------------args---------------------")
    for k, v in sorted(vars(args).items()):
        print(k, '=', v)
    print("----------------------------------------------")

    ### 加载防御者模型
    if args.defense_model_id == "gpt-3.5-turbo":
        openai.api_key = "0fa13a787131400dbb4f1cb3f3a2d73f"
        openai.api_base = "https://4tsinghua.openai.azure.com"  # your endpoint should look like the following https://YOUR_RESOURCE_NAME.openai.azure.com/
        openai.api_type = 'azure'
        openai.api_version = '2023-05-15'  # this may change in the future 2023-05-15
        deployment_name = 'test'  # This will correspond to the custom name you chose for your deployment when you deployed a model.

    ### 加载受害者模型
    # pytorch 1.12
    # model = torch.load(args.victim_model_path)
    # if torch.cuda.is_available():
    #     model.cuda()

    # pytorch 2.1
    model = torch.load(args.victim_model_path, map_location="cuda:0")

    ### 读数据
    packDataset_util = packDataset_util_bert()

    ### 中毒数据ΔASR
    delta_ASR = get_delta_ASR()
    print("\nΔASR:", delta_ASR)

    ### 干净数据ΔACC
    delta_ACC = get_delta_ACC()
    print("\nΔACC:", delta_ACC)
