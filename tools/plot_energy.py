import os
import torch
import pickle

source_dir = '/home/fk1/workspace/OWOD/output/logits'

files = os.listdir(source_dir)
unk = []
known = []
for file in files:
    path = os.path.join(source_dir, file)
    logits, classes = torch.load(path)
    lse = torch.logsumexp(logits[:,:-2], dim=1)

    for i, cls in enumerate(classes):
        if cls == 21:
            continue
        if cls == 20:
            unk.append(lse[i].detach().cpu().tolist())
        else:
            known.append(lse[i].detach().cpu().tolist())

print(known)
print('\n\n')
print(unk)

# dir = '/home/fk1/workspace/OWOD/output'
#
# with open(os.path.join(dir, 'unk.pkl'), 'wb') as f:
#     pickle.dump(unk, f)
#
# with open(os.path.join(dir, 'known.pkl'), 'wb') as f:
#     pickle.dump(unk, f)
