def align_pos_ids_with_tokens(tokens, pos_ids, delimiter="Ġ"):
    aligned_pos_tags = []
    word_index = 0
    for token in tokens:
        if token.startswith(delimiter) or word_index == 0:
            aligned_pos_tags.append(pos_ids[word_index])
            word_index += 1
        else:
            aligned_pos_tags.append(pos_ids[word_index - 1])
    return aligned_pos_tags


def align_ner_labels_with_tokens(tokens, label):
    label_idx = 0
    current_master = label[label_idx]
    label_aligned = [current_master]
    for i in range(1, len(tokens)):
        if not tokens[i].startswith("Ġ"):
            extended_label = current_master.replace("B-", "I-")
            label_aligned.append(extended_label)
        else:
            label_idx += 1
            current_master = label[label_idx]
            label_aligned.append(current_master)
    return label_aligned