import json
from evaluate import load
import evaluate
from rouge import Rouge

def calculate_rogue(preds, labels):
    r1s, r2s, rls = [], [], []
    r = Rouge()
    for i in range(len(labels)):
        scores = r.get_scores(preds[i], labels[i])[0]
        r1s.append(scores["rouge-1"]['f'])
        r2s.append(scores["rouge-2"]['f'])
        rls.append(scores["rouge-l"]['f'])
    r1 = sum(r1s) / len(r1s)
    r2 = sum(r2s) / len(r2s)
    rl = sum(rls) / len(rls)
    print(r1)
    print(r2)
    print(rl)
    print("----------")


with open('StandAns.json', 'r', encoding='utf-8') as file:
    Stand = json.load(file)

with open('SDRAns.json', 'r', encoding='utf-8') as file:
    SDR = json.load(file)

with open('MMR00Ans.json', 'r', encoding='utf-8') as file:
    MMR00 = json.load(file)

with open('MMR05Ans.json', 'r', encoding='utf-8') as file:
    MMR05 = json.load(file)

with open('MMR10Ans.json', 'r', encoding='utf-8') as file:
    MMR10 = json.load(file)

print("SDR-Rouge-1,2,l")
print(calculate_rogue(SDR, Stand))
print("MMR00-Rouge-1,2,l")
print(calculate_rogue(MMR00, Stand))
print("MMR05-Rouge-1,2,l")
print(calculate_rogue(MMR05, Stand))
print("MMR10-Rouge-1,2,l")
print(calculate_rogue(MMR10, Stand))
