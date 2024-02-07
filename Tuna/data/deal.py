import json

# 假设您的JSON文件名为 'your_file.json'
with open('new_data2.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# 遍历JSON文件中的每个条目
for item in data:
    # 获取"output"数组
    output_array = item.get('output', [])
    
    # 对数组中的每个字符串添加"</s>"
    for i in range(len(output_array)):
        output_array[i] += '</s>'
    
    # 更新条目的"output"数组
    item['output'] = output_array

# 保存更新后的JSON数据
with open('your_file.json', 'w', encoding='utf-8') as file:
    json.dump(data, file, ensure_ascii=False, indent=4)