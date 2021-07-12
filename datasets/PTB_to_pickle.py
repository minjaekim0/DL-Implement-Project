from tqdm import tqdm
import cupy as cp
import pickle


# Dataset Source: https://deepai.org/dataset/penn-treebank

with open("datasets/PTB_sequence.pickle", "wb") as fw:
    id_to_word = {}
    word_to_id = {}
    
    for purpose in ['train', 'valid', 'test']:
        sequence = []
        path = 'datasets/ptbdataset/ptb.' + purpose + '.txt'

        with open(path, 'r') as f:
            words = f.read().replace('\n', '<eos>').strip().split()

            for word in tqdm(words):
                if word in id_to_word.values():
                    id_ = word_to_id[word]
                    sequence.append(id_)
                else:
                    id_ = len(word_to_id)
                    sequence.append(id_)
                    word_to_id[word] = id_
                    id_to_word[id_] = word
        sequence = cp.array(sequence)  
        pickle.dump(sequence, fw)

    pickle.dump(id_to_word, fw)
    pickle.dump(word_to_id, fw)
