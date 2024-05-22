# Vector retrieve: similarity and diversity
import pandas as pd
import random
import json

def read_Puzzle():
    IO_list = []  # Initialize the empty list
    with open('puzzle.jsonl', 'r', encoding="utf-8") as file:
        for line in file:
            data = json.loads(line)  # Load JSON object from each line
            # Check if there are at least two messages to combine
            if len(data["messages"]) >= 2:
                # Extract sentences from the first and second messages' "content" and combine them
                combined_sentence = data["messages"][0]["content"] + "->" + data["messages"][1]["content"]
                IO_list.append(combined_sentence)  # Add the combined sentence to the list A
    return IO_list


def read_BoolQ():
    train_data = pd.read_json('BoolQtrain.jsonl', lines=True)
    dev_data = pd.read_json('BoolQdev.jsonl', lines=True)
    all_data = pd.concat([train_data, dev_data])

    all_data_list = []
    for index, row in all_data.iterrows():
        all_data_list.append(
            row['title'] + ': ' + row['passage'] + 'Now, ' + row['question'] + '. ' + '->' + str(row['answer']))
    print("all_data_list------", all_data_list[0:4])
    return all_data_list


def read_ARC():
    arc_train = pd.read_json('ARCtrain.jsonl', lines=True)
    arc_test = pd.read_json('ARCtest.jsonl', lines=True)
    arc_dev = pd.read_json('ARCdev.jsonl', lines=True)
    all_data = pd.concat([arc_train, arc_test, arc_dev])

    all_data_list = []
    for index, row in all_data.iterrows():
        all_data_list.append(row['question'] + '->' + '. '.join(row['answers']))
    print("all_data_list------", all_data_list[0:4])
    return all_data_list


def read_OpenBookQA():
    OpenBookQA_train = pd.read_json('OpenBookQAtrain.jsonl', lines=True)
    OpenBookQA_test = pd.read_json('OpenBookQAtest.jsonl', lines=True)
    OpenBookQA_dev = pd.read_json('OpenBookQAdev.jsonl', lines=True)
    all_data = pd.concat([OpenBookQA_train, OpenBookQA_test, OpenBookQA_dev])

    all_data_list = []
    for index, row in all_data.iterrows():
        stem = row['question']['stem']  
        choices = row['question']['choices']  
        answer = row['answerKey']
        answer_str = ""
        for cho in choices:
            if cho['label'] == answer:
                answer_str = cho['text']
        choices_to_str = [choi['text'] for choi in choices]
        all_data_list.append(stem + '? The option can be: ' + '. '.join(choices_to_str) + "->" + answer_str)
    print("all_data_list------", all_data_list[0:1])
    return all_data_list

def read_OpenBookQA_NoOption():
    OpenBookQA_train = pd.read_json('OpenBookQAtrain.jsonl', lines=True)
    OpenBookQA_test = pd.read_json('OpenBookQAtest.jsonl', lines=True)
    OpenBookQA_dev = pd.read_json('OpenBookQAdev.jsonl', lines=True)
    all_data = pd.concat([OpenBookQA_train, OpenBookQA_test, OpenBookQA_dev])

    all_data_list = []
    for index, row in all_data.iterrows():
        stem = row['question']['stem']  
        choices = row['question']['choices']  
        answer = row['answerKey']
        answer_str = ""
        for cho in choices:
            if cho['label'] == answer:
                answer_str = cho['text']
        # choices_to_str = [choi['text'] for choi in choices]
        all_data_list.append(stem + '?' + "->" + answer_str)
    print("all_data_list------", all_data_list[0:1])
    return all_data_list


from langchain_community.vectorstores import FAISS
def embedding_corpus(all_data_list, embedding_model):
    size = int(len(all_data_list) / 5)
    item_list = all_data_list[:size]
    for ql in item_list:
        all_data_list.remove(ql)
    query_list = [qm[:qm.find('->')] for qm in item_list]  
    answer_list = [qm[qm.find('->') + 2:] for qm in item_list]
    print("query_list------", query_list[0], '\n')
    print("answer_list-----", answer_list[0], '\n')
    vectordb = FAISS.from_texts(texts=all_data_list, embedding=embedding_model)

    print("vectordb------", vectordb)
    return vectordb, query_list, answer_list
