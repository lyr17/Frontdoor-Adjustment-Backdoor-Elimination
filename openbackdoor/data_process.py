import csv
import json
import os
import pandas as pd

poison_file_path = 'train-poison.csv'
clean_file_path = 'train-clean.csv'


def merged_train_dataset():
    merged_train_data = []
    sep = ''
    if os.path.splitext(clean_file_path)[1] == '.tsv':
        sep = '\t'
    elif os.path.splitext(clean_file_path)[1] == '.csv':
        sep = ','
    poison_data = pd.read_csv(poison_file_path, sep=sep).values.tolist()
    clean_data = pd.read_csv(clean_file_path, sep=sep).values.tolist()
    for poison_row, clean_row in zip(poison_data, clean_data):
        merged_train_data.append({'poison_sentence': poison_row[1], 'clean_sentence': clean_row[1], 'poison_label': poison_row[2], 'clean_label': clean_row[2]})
    with open('synbkd-sst2-data.json', 'w', encoding='utf-8') as output_file:
        json.dump(merged_train_data, output_file, ensure_ascii=False, indent=2)


def merged_test_dataset():
    merged_test_poison_data = []
    merged_test_clean_data = []
    sep = ''
    if os.path.splitext(clean_file_path)[1] == '.tsv':
        sep = '\t'
    elif os.path.splitext(clean_file_path)[1] == '.csv':
        sep = ','
    poison_data = pd.read_csv(poison_file_path, sep=sep).values.tolist()
    clean_data = pd.read_csv(clean_file_path, sep=sep).values.tolist()
    for poison_row in poison_data:
        merged_test_poison_data.append({'poison_sentence': poison_row[0], 'poison_label': poison_row[1]})
    for clean_row in clean_data:
        merged_test_clean_data.append({'clean_sentence': clean_row[0], 'clean_label': clean_row[1]})

    with open('badnets-sst2-test-poison.json', 'w', encoding='utf-8') as output_file:
        json.dump(merged_test_poison_data, output_file, ensure_ascii=False, indent=2)
    with open('badnets-sst2-test-clean.json', 'w', encoding='utf-8') as output_file:
        json.dump(merged_test_clean_data, output_file, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    merged_train_dataset()
    # merged_test_dataset()
