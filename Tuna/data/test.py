# import json
# import re

# # 已知的字符串A
# known_string_A = "You are a backdoor trigger remover, and backdoor triggers come in these forms: token/words triggers (i.e., abnormal words), syntax/semantic triggers (i.e., abnormal sentence structure and stsyntax and semantics). You can detect and remove exceptions in sentences. Be sure to return only processed sentences.The sentences that need to be modified are as follows:"

# # 假设每个"instruction"项的内容都是由A和B组成的，格式为"A + B"
# # 这里我们使用正则表达式来匹配已知的字符串A和未知的字符串B
# # 例如，如果A和B之间没有特定的分隔符，我们可以使用正则表达式来匹配A后面直到下一个A的所有字符作为B
# # 这里是一个简单的例子，如果A和B之间有特定的分隔符，请相应地修改正则表达式

# sentence1 = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:"
# prompt = "You are a backdoor trigger remover, and backdoor triggers come in these forms: token/words triggers (i.e., abnormal words), syntax/semantic triggers (i.e., abnormal sentence structure and stsyntax and semantics). You can detect and remove exceptions in sentences. Be sure to return only processed sentences.\n\n### Input:\n"
# # 读取JSON文件
# with open('new_data.json', 'r', encoding='utf-8') as file:
#     data = json.load(file)

# # 遍历JSON数据并处理"instruction"项
# for item in data:
#     if 'instruction' in item:
#         # 使用正则表达式匹配已知的字符串A和未知的字符串B
#         # 这里假设A和B之间没有分隔符，且A不重复出现
#         match = re.match(rf"^{known_string_A}(.*)$", item['instruction'])
#         if match:
#             # 分离出字符串B
#             unknown_string_B = match.group(1)

#             # 创建新的字符串C，这里假设C是A和B的组合加上新的字符串D
#             new_instruction = sentence1 + prompt + unknown_string_B + "\n\n### Response:"
#             # 更新"instruction"项为新的字符串C
#             item['instruction'] = new_instruction
#             # 如果需要，可以将字符串B保存到其他地方，例如另一个列表或字典中
#             # 这里我们只是打印出来
#             print("分离出的字符串B:", unknown_string_B)

# # 将修改后的数据写回文件
# with open('new_data2.json', 'w', encoding='utf-8') as file:
#     json.dump(data, file, ensure_ascii=False, indent=4)

# print("JSON文件已更新。")

import json

# 假设两个文件的路径分别为 'first_file.json' 和 'second_file.json'
first_file_path = 'merged.json'
second_file_path = 'combined_output2.json'

#sentence1 = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:"
prompt = "As an experienced data engineer specializing in text data augmentation, your task is to refine language expressions in a dataset. Ensure that your modifications maintain the original intent and meaning of the data. Your goal is to enhance the smoothness and coherence of the text without compromising its effectiveness in performing relevant natural language processing tasks. Focus on preserving the fundamental essence of the data while improving its readability and usability for machine learning applications.\n\n### Input:\n"
# # 读取JSON文件
# 读取第一个JSON文件
with open(first_file_path, 'r', encoding='utf-8') as file1:
    first_data = json.load(file1)

# 读取第二个JSON文件
with open(second_file_path, 'r', encoding='utf-8') as file2:
    second_data = json.load(file2)

# 确保两个文件中的条目数量相同
if len(first_data) != len(second_data):
    print("两个文件中的条目数量不匹配。")
else:
    # 遍历第二个JSON文件的数据
    for i in range(len(second_data)):
        # 假设每个条目都是一个字典，并且有相同的键
        clean_sentence = first_data[i]['clean_sentence']
        second_data[i]['output'].insert(0, clean_sentence)
        if 'instruction' in second_data[i] and 'poison_sentence' in first_data[i]:
            # 替换第二个JSON文件中的"instruction"项
            second_data[i]['instruction'] =prompt + first_data[i]['poison_sentence'] + "\n\n### Response:"

    # 将修改后的数据写回第二个JSON文件
    with open("new_data2.json", 'w', encoding='utf-8') as file2:
        json.dump(second_data, file2, ensure_ascii=False, indent=4)

    print("第二个JSON文件已更新。")