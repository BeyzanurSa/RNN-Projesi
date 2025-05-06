def build_vocab(sentences):
    vocab = {"<PAD>": 0}
    idx = 1
    for sentence in sentences:
        for word in sentence.lower().split():
            if word not in vocab:
                vocab[word] = idx
                idx += 1
    return vocab

def sentence_to_indices(sentence, word2idx, max_len):
    indices = [word2idx.get(word, word2idx["<PAD>"]) for word in sentence.lower().split()]
    if len(indices) < max_len:
        indices += [word2idx["<PAD>"]] * (max_len - len(indices))
    else:
        indices = indices[:max_len]
    return indices

def prepare_data(data_dict, word2idx, max_len):
    X = []
    y = []
    for sentence, label in data_dict.items():
        indices = sentence_to_indices(sentence, word2idx, max_len)
        X.append(indices)
        y.append(int(label))
    return X, y

