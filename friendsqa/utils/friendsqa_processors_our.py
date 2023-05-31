import json
import os
import random
from functools import partial
from multiprocessing import Pool, cpu_count
import numpy as np
from tqdm import tqdm
import torch.utils.data as data
import torch
from transformers.file_utils import is_torch_available
from transformers.tokenization_utils_base import TruncationStrategy
from transformers.utils import logging
from transformers.data.processors.utils import DataProcessor

# Store the tokenizers which insert 2 separators tokens
MULTI_SEP_TOKENS_TOKENIZERS_SET = {"roberta", "camembert", "bart", "mpnet", "longformer"}

logger = logging.get_logger(__name__)

map_relations = {'Comment': 1, 'Contrast': 2, 'Correction': 3, 'Question-answer_pair': 4, 'QAP': 4, 
                 'Parallel': 5, 'Acknowledgement': 6, 'Elaboration': 7, 'Clarification_question': 8, 
                 'Conditional': 9, 'Continuation': 10, 'Result': 11, 'Explanation': 12, 'Q-Elab': 13, 
                 'Alternation': 14, 'Narration': 15, 'Background': 16}


def floyd(W):
    node_number = len(W)
    for k in range(node_number):
        for i in range(node_number):
            for j in range(node_number):
                if W[i][k] > 0 and W[k][j] > 0 and (W[i][k] + W[k][j] < W[i][j] or W[i][j] == 0):
                    W[i][j] = W[i][k] + W[k][j]
    return W


class FriendsQAExample(object):
    def __init__(self, contents_list, question, relations, qid, utterance_label = -1, left_labels = None, 
                        right_labels = None, answer_text = None, answers = None, sep_token = None, 
                        utrs_with_same_spk=None, context = None, corefs = None, spk2pos=None):
        
        self.relations = relations
        self.question = question
        self.qas_id = qid
        self.answer = answer_text
        self.answers = answers
        self.start_position = None
        self.end_position = None
        self.flag = True
        self.utrs_with_same_spk=utrs_with_same_spk
        self.contents_list = contents_list

        doc_tokens = []; char_to_word_offset = []; context_text = ''; spk2char = {}
        spk_index = []
        utterances_index = []  # record the length of each utterances including [SEP]
        for uidx, utterance_text in enumerate(contents_list):
            if utterance_label>=0 and uidx == utterance_label:
                self.start_position = len(doc_tokens) + left_labels[uidx]
                self.end_position = len(doc_tokens) + right_labels[uidx]
            uidxlen = len(utterance_text.split()[0])
            spk = utterance_text[uidxlen+1:utterance_text.index(':')]

            if spk not in spk2char:
                spk2char[spk] = [[uidxlen + len(context_text)+1, uidxlen + len(spk)+ len(context_text)+1]]
            else:
                spk2char[spk].append([uidxlen + len(context_text)+1, uidxlen + len(spk)+ len(context_text)+1])
            prev_is_whitespace = True
            for c in utterance_text:
                if _is_whitespace(c):
                    prev_is_whitespace = True
                else:
                    if prev_is_whitespace:
                        doc_tokens.append(c)
                    else:
                        doc_tokens[-1] += c
                    prev_is_whitespace = False
                char_to_word_offset.append(len(doc_tokens) - 1)
                
            spk_index.append(len(context_text)+len(spk)+1)
            context_text += utterance_text + sep_token + ' '
            doc_tokens.append(sep_token); 
            char_to_word_offset.extend([len(doc_tokens) - 1 for _ in range(len(sep_token + ' '))])
            utterances_index.append(len(doc_tokens)-1)
        utterances_index = utterances_index[:-1]
        self.spks_index = []
        for idx in spk_index:
            self.spks_index.append(char_to_word_offset[idx])
        doc_tokens.pop()
        
        char_orig_contect2context = {}; ct = 0
        for i, c in enumerate(context):
            while c != context_text[ct]:
                ct += 1
            char_orig_contect2context[i] = ct
            ct += 1
        
        coref = []
        for cluster in corefs:
            clu = []
            is_spk = ''
            for c in cluster:
                if c[0] in spk2char:
                    is_spk = c[0]
                else:
                    clu.append([
                        c[0], 
                        char_to_word_offset[char_orig_contect2context[c[1]]], 
                        char_to_word_offset[char_orig_contect2context[c[2]-1]]
                        ])
            if is_spk:
                clu.extend([[is_spk, char_to_word_offset[l[0]], char_to_word_offset[l[1]-1]] for l in spk2char[is_spk]])
                del spk2char[is_spk]
            if len(clu) > 1:
                coref.append(clu)
        for spk, sl in spk2char.items():
            if len(sl) > 1:
                coref.append([[spk, char_to_word_offset[l[0]], char_to_word_offset[l[1]-1]] for l in sl])

        self.coref = coref
        
        if utterance_label>=0:
            if ' '.join(answer_text.split()) != ' '.join(doc_tokens[self.start_position:self.end_position+1]):
                self.flag = False
        self.utterances_index = utterances_index
        self.doc_tokens = doc_tokens
        
        relation_matrix = [[0 for _ in range(len(contents_list))] for _ in range(len(contents_list))]

        self.relations = {}
        for rel in relations:
            y = rel['y']; x = rel['x']; 
            if rel['type'] in map_relations and y < len(contents_list) and x < len(contents_list):
                relation_matrix[y][x] = 1
                t = map_relations[rel['type']]
                self.relations[(y, x)] = t
        self.relation_matrix = floyd(relation_matrix)


class FriendsQAFeature(object):
    def __init__(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        cls_index,
        p_mask,
        example_index,
        unique_id,
        paragraph_len,
        token_is_max_context,
        tokens,
        token_to_orig_map,
        start_position,
        end_position,
        is_impossible=None,
        qas_id= None,
        query_end = None,
        spk_info_dict=None,
        rel_info_dict=None,
        query_mapping=None,
    ):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.cls_index = cls_index
        self.p_mask = p_mask

        self.example_index = example_index
        self.unique_id = unique_id
        self.paragraph_len = paragraph_len
        self.token_is_max_context = token_is_max_context
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map

        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible
        self.qas_id = qas_id
        self.query_end = query_end
        self.spk_info_dict = spk_info_dict
        self.rel_info_dict=rel_info_dict
        self.query_mapping=query_mapping


class FriendsQAResult:
    def __init__(self, unique_id, start_logits, end_logits):
        self.unique_id = unique_id
        self.start_logits = start_logits
        self.end_logits = end_logits    


class Dataset(data.Dataset):
    def __init__(self, features):
        self.features = features

    def __getitem__(self, index):
        data_info = {}
        data_info['feature_indices'] = torch.tensor(self.features[index].feature_index, dtype=torch.long)
        data_info['input_ids'] = torch.tensor(self.features[index].input_ids, dtype=torch.long)
        data_info['token_type_ids'] = torch.tensor(self.features[index].token_type_ids, dtype=torch.long)
        data_info['attention_mask'] = torch.tensor(self.features[index].attention_mask, dtype=torch.long)
        data_info['p_mask'] = torch.tensor(self.features[index].p_mask, dtype=torch.long)
        data_info['start_positions'] = torch.tensor(self.features[index].start_position, dtype=torch.long) if\
                                        self.features[index].start_position is not None else None
        data_info['end_positions'] = torch.tensor(self.features[index].end_position, dtype=torch.long) if\
                                        self.features[index].end_position is not None else None
        data_info['spk_info_dict'] = self.features[index].spk_info_dict
        data_info['rel_info_dict'] = self.features[index].rel_info_dict
        data_info['query_mapping'] = torch.tensor(self.features[index].query_mapping, dtype=torch.float) 
        return data_info

    def __len__(self):
        return len(self.features)


def to_list(tensor):
    return tensor.detach().cpu().tolist()

def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens

def _is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False

def _new_check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""
    # if len(doc_spans) == 1:
    # return True
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span["start"] + doc_span["length"] - 1
        if position < doc_span["start"]:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span["start"]
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span["length"]
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index

def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer, orig_answer_text):
    """Returns tokenized answer spans that better match the annotated answer."""
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start : (new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)


class FriendsQAProcessor(DataProcessor):
    
    def __init__(self, tokenizer, threads):
        self.threads = threads
        self.sep_token = tokenizer.sep_token
    
    def get_train_examples(self, data_dir, filename=None):
        """
        Returns the training examples from the data directory.
        Args:
            data_dir: Directory containing the data files used for training and evaluating.
            filename: None by default, specify this if the training file has a different name than the original one
                which is `train.json`.
        """
        if data_dir is None:
            data_dir = ""

        with open(
            os.path.join(data_dir, filename), "r", encoding="utf-8"
        ) as reader:
            input_data = json.load(reader)["data"]
            
        threads = min(self.threads, cpu_count())
        with Pool(threads) as p:
            annotate_ = partial(self._create_examples, is_training=True)
            examples = list(tqdm(
                p.imap(annotate_, input_data),
                total=len(input_data),
                desc="collect molweni examples to",
            ))
        examples = [item for sublist in examples for item in sublist]
        return examples

    def get_dev_examples(self, data_dir, filename=None, threads=1):
        """
        Returns the evaluation example from the data directory.
        Args:
            data_dir: Directory containing the data files used for training and evaluating.
            filename: None by default, specify this if the evaluation file has a different name than the original one
                which is `dev.json`.
        """
        if data_dir is None:
            data_dir = ""

        with open(
            os.path.join(data_dir, filename), "r", encoding="utf-8"
        ) as reader:
            input_data = json.load(reader)["data"]
            
        # threads = min(self.threads, cpu_count())
        # with Pool(threads) as p:
        #     annotate_ = partial(self._create_examples, is_training=False)
        #     examples = list(tqdm(
        #         p.imap(annotate_, input_data),
        #         total=len(input_data),
        #         desc="collect molweni examples to",
        #     ))
        # examples = [item for sublist in examples for item in sublist]
        examples = []
        for d in input_data:
            examples.extend(self._create_examples(d, False))
        return examples

    def _create_examples(self, data, is_training):
        examples = []               
        if 'context' not in data["paragraphs"][0]:
            context = ''
        else:
            context = data["paragraphs"][0]['context']
        utterances = data["paragraphs"][0]["utterances:"]
        qas = data["paragraphs"][0]["qas"]
        relations = data['relations']
        n_length = len(utterances)
        spks_len = []; content_list = []; utrs_with_same_spk = {}; spk2pos = {}
        for ui, utterance in enumerate(utterances):
            speaker = utterance["speakers"][0].split(" ")
            spk_num = len(speaker)
            speaker = ' '.join(speaker)
            u_text = "u" + str(ui) + " " + speaker + " : " + utterance["utterance"]
            content_list.append(u_text)
            spks_len.append(spk_num)
            if speaker in spk2pos:
                spk2pos[speaker].append([ui, len(speaker)])
            else:
                spk2pos[speaker] = [[ui, len(speaker)]]
            if speaker in utrs_with_same_spk:
                utrs_with_same_spk[speaker].append(ui)
            else:
                utrs_with_same_spk[speaker] = [ui]
        
        for qa in qas:
            qid = qa["id"]
            question = qa["question"]
            answers = qa["answers"]
            if is_training:
                for a_id, answer in enumerate(answers):
                    # guid = "%s-%s" % (set_type, str(q_id))
                    answer_text = answer["answer_text"]
                    if answer_text.strip() == '':
                        print('impossible!!!')
                    utterance_id = answer["utterance_id"]
                    is_speaker = answer["is_speaker"]
                    spk_len = spks_len[utterance_id]
                    if is_speaker:
                        inner_start = 1
                        inner_end = spk_len
                    else:
                        inner_start = answer["inner_start"] + 2+spk_len
                        inner_end = answer["inner_end"] + 2+spk_len
                    left_labels = n_length * [-1]
                    right_labels = n_length * [-1]
                    left_labels[utterance_id] = inner_start
                    right_labels[utterance_id] = inner_end
                    exp = FriendsQAExample(qid=qid, relations=relations, contents_list=content_list,
                                            question=question, utterance_label=utterance_id,
                                            left_labels=left_labels, right_labels=right_labels,
                                            answer_text=answer_text, sep_token=self.sep_token,
                                            utrs_with_same_spk=utrs_with_same_spk, context=context,
                                            corefs = data['corefs'], spk2pos=spk2pos)
                    if exp.flag:
                        examples.append(exp)
            else:
                ans = [a["answer_text"] for a in answers]
                examples.append(FriendsQAExample(qid=qid, relations=relations, contents_list=content_list, 
                                                 question=question, answers = ans, sep_token = self.sep_token,
                                                 utrs_with_same_spk = utrs_with_same_spk, context = context,
                                                 corefs = data['corefs'], spk2pos=spk2pos))
        return examples



def friendsqa_convert_example_to_features(
    example, tokenizer, max_seq_length, doc_stride, max_query_length, padding_strategy, is_training, 
    max_utrs_num = 40
):
    features = []
    question_text = example.question
    answer_text = example.answer
    relation_matrix = example.relation_matrix
    
    tok_to_orig_index = []
    orig_to_tok_index = []
    all_doc_tokens = []
    for (i, token) in enumerate(example.doc_tokens):
        orig_to_tok_index.append(len(all_doc_tokens))
        if tokenizer.__class__.__name__ in [
            "RobertaTokenizer",
            "LongformerTokenizer",
            "BartTokenizer",
            "RobertaTokenizerFast",
            "LongformerTokenizerFast",
            "BartTokenizerFast",
        ]:
            sub_tokens = tokenizer.tokenize(token, add_prefix_space=True)
        else:
            sub_tokens = tokenizer.tokenize(token)
        for sub_token in sub_tokens:
            tok_to_orig_index.append(i)
            all_doc_tokens.append(sub_token)
            
    utterences_index = []
    for idx in example.utterances_index:
        utterences_index.append(orig_to_tok_index[idx+1]-1)
    utterences_index.append(len(all_doc_tokens)-1)
    spks_index = []
    for idx in example.spks_index:
        spks_index.append(orig_to_tok_index[idx+1]-1)
        
    corefs = []
    for cluster in example.coref:
        clu = []
        for c in cluster:
            s = orig_to_tok_index[c[1]]
            if c[2] < len(example.doc_tokens) - 1:
                e = orig_to_tok_index[c[2]+1]-1
            else:
                e = len(all_doc_tokens) - 1
            (s, e) = _improve_answer_span(all_doc_tokens, s, e, tokenizer, c[0])
            clu.append([s,e])
        corefs.append(clu)
        
    if is_training:
        tok_start_position = orig_to_tok_index[example.start_position]
        if example.end_position < len(example.doc_tokens) - 1:
            tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
        else:
            tok_end_position = len(all_doc_tokens) - 1

        (tok_start_position, tok_end_position) = _improve_answer_span(
            all_doc_tokens, tok_start_position, tok_end_position, tokenizer, answer_text
        )
        
    truncated_query = tokenizer.encode(question_text, add_special_tokens=False)
    if len(truncated_query) > max_query_length:
        truncated_query = truncated_query[:max_query_length]
    
    # tokenizers who insert 2 SEP tokens in-between <context> & <question> need to have special handling
    # in the way they compute mask of added tokens.
    tokenizer_type = type(tokenizer).__name__.replace("Tokenizer", "").lower()
    sequence_added_tokens = (
        tokenizer.model_max_length - tokenizer.max_len_single_sentence + 1
        if tokenizer_type in MULTI_SEP_TOKENS_TOKENIZERS_SET
        else tokenizer.model_max_length - tokenizer.max_len_single_sentence
    )
    sequence_pair_added_tokens = tokenizer.model_max_length - tokenizer.max_len_sentences_pair

    spans = []
    span_doc_tokens = all_doc_tokens
    while len(spans) * doc_stride < len(all_doc_tokens):
        # Define the side we want to truncate / pad and the text/pair sorting
        if tokenizer.padding_side == "right":
            texts = truncated_query
            pairs = span_doc_tokens
            truncation = TruncationStrategy.ONLY_SECOND.value
        else:
            texts = span_doc_tokens
            pairs = truncated_query
            truncation = TruncationStrategy.ONLY_FIRST.value

        encoded_dict = tokenizer.encode_plus(  # TODO(thom) update this logic
            texts,
            pairs,
            truncation=truncation,
            padding=padding_strategy,
            max_length=max_seq_length,
            return_overflowing_tokens=True,
            stride=max_seq_length - doc_stride - len(truncated_query) - sequence_pair_added_tokens,
            return_token_type_ids=True,
        )
        
        paragraph_len = min(
            len(all_doc_tokens) - len(spans) * doc_stride,
            max_seq_length - len(truncated_query) - sequence_pair_added_tokens,
        )

        if tokenizer.pad_token_id in encoded_dict["input_ids"]:
            if tokenizer.padding_side == "right":
                non_padded_ids = encoded_dict["input_ids"][: encoded_dict["input_ids"].index(tokenizer.pad_token_id)]
            else:
                last_padding_id_position = (
                    len(encoded_dict["input_ids"]) - 1 - encoded_dict["input_ids"][::-1].index(tokenizer.pad_token_id)
                )
                non_padded_ids = encoded_dict["input_ids"][last_padding_id_position + 1 :]
        else:
            non_padded_ids = encoded_dict["input_ids"]

        tokens = tokenizer.convert_ids_to_tokens(non_padded_ids)

        token_to_orig_map = {}
        for i in range(paragraph_len):
            index = len(truncated_query) + sequence_added_tokens + i if tokenizer.padding_side == "right" else i
            token_to_orig_map[index] = tok_to_orig_index[len(spans) * doc_stride + i]
            
        encoded_dict["paragraph_len"] = paragraph_len
        encoded_dict["tokens"] = tokens
        encoded_dict["token_to_orig_map"] = token_to_orig_map
        encoded_dict["truncated_query_with_special_tokens_length"] = len(truncated_query) + sequence_added_tokens
        encoded_dict["token_is_max_context"] = {}
        encoded_dict["start"] = len(spans) * doc_stride
        encoded_dict["length"] = paragraph_len

        spans.append(encoded_dict)

        if "overflowing_tokens" not in encoded_dict or (
            "overflowing_tokens" in encoded_dict and len(encoded_dict["overflowing_tokens"]) == 0
        ):
            break
        span_doc_tokens = encoded_dict["overflowing_tokens"]

    for doc_span_index in range(len(spans)):
        for j in range(spans[doc_span_index]["paragraph_len"]):
            is_max_context = _new_check_is_max_context(spans, doc_span_index, doc_span_index * doc_stride + j)
            index = (
                j
                if tokenizer.padding_side == "left"
                else spans[doc_span_index]["truncated_query_with_special_tokens_length"] + j
            )
            spans[doc_span_index]["token_is_max_context"][index] = is_max_context

    for span in spans:
        # Identify the position of the CLS token
        cls_index = span["input_ids"].index(tokenizer.cls_token_id)

        # p_mask: mask with 1 for token than cannot be in the answer (0 for token which can be in an answer)
        # Original TF implem also keep the classification token (set to 0)
        p_mask = np.ones_like(span["token_type_ids"])

        if tokenizer.padding_side == "right":
            p_mask[len(truncated_query) + sequence_added_tokens :] = 0
        else:
            p_mask[-len(span["tokens"]) : -(len(truncated_query) + sequence_added_tokens)] = 0

        pad_token_indices = np.where(span["input_ids"] == tokenizer.pad_token_id)
        special_token_indices = np.asarray(
            tokenizer.get_special_tokens_mask(span["input_ids"], already_has_special_tokens=True)
        ).nonzero()

        p_mask[pad_token_indices] = 1
        p_mask[special_token_indices] = 1
        # Set the cls index to 0: the CLS index can be used for impossible answers
        p_mask[cls_index] = 0

        doc_start = span["start"]
        doc_end = span["start"] + span["length"] - 1

        if tokenizer.padding_side == "left":
            doc_offset = 0
        else:
            doc_offset = len(truncated_query) + sequence_added_tokens

        utters_index = [doc_offset-1]; spk_index = []
        utters_idx = []
        for i, idx in enumerate(utterences_index):
            if idx > doc_start and idx <= doc_end:
                utters_index.append(idx - doc_start + doc_offset)
                utters_idx.append(i)
                sid = spks_index[i]
                spk_index.append(sid - doc_start + doc_offset)
        utters_index[-1] = len(span['tokens'])-1
        if len(spk_index) > len(utters_idx):
            spk_index = spk_index[:len(utters_idx)]

        coref = []
        for cluster in corefs:
            clu = []
            for c in cluster:
                s, e = c[0], c[1]
                if s >= doc_start and s <= doc_end and e >= doc_start and s <= doc_end:
                    clu.append([s - doc_start + doc_offset, e - doc_start + doc_offset])

            if len(clu) > 1:
                coref.append(clu)

        query_end = span["truncated_query_with_special_tokens_length"]
        
        discourse_mask = np.zeros((len(span["input_ids"]), len(span["input_ids"])))
        coref_mask = np.zeros((len(span["input_ids"]), len(span["input_ids"])))
        
        for cluster in coref:
            clu = []
            for i in range(len(cluster)-1):
                s, e = cluster[i][0], cluster[i][1]
                coref_mask[s:e+1, s:e+1] = 1
                for j in range(i+1, len(cluster)):
                    ss, ee = cluster[j][0], cluster[j][1]
                    coref_mask[s:e+1, ss:ee+1] = 1
                    coref_mask[ss:ee+1, s:e+1] = 1
    
        coref_matrix = coref_mask
        
        # attend itself
        for i in range(len(utters_index) - 1):
            s = utters_index[i]; e = utters_index[i+1]
            discourse_mask[s+1 : e+1, s+1 : e+1] = 1
        query_end = span["truncated_query_with_special_tokens_length"]
       
        for (i, j), rel in example.relations.items():
            if i in utters_idx and j in utters_idx:
                s = utters_idx.index(i); e = utters_idx.index(j)

        for i in range(len(relation_matrix)):
            for j in range(len(relation_matrix)):
                if i != j and relation_matrix[i][j] > 0: 
                    if i in utters_idx and j in utters_idx:
                        s = utters_idx.index(i); e = utters_idx.index(j)
                        s1 = utters_index[s]; e1 = utters_index[s+1]
                        s2 = utters_index[e]; e2 = utters_index[e+1]
                        if relation_matrix[i][j] <= 1:  # 改了！，之前是1
                            discourse_mask[s1+1 : e1+1, s2+1 : e2+1] = 1
                            # discourse_mask[s2+1 : e2+1, s1+1 : e1+1] = 1
                        # if relation_matrix[i][j] == 2:
                        #     discourse_mask[s1+1 : e1+1, s2+1 : e2+1] = 3
        
        utr_gather_ids = utters_index[1:]
        utr_gather_ids.extend([0]*(max_utrs_num-len(utters_index[1:])))
        
        spk_attention_mask = span['attention_mask'].copy()
        for i in range(query_end):
            spk_attention_mask[i] = 0
        spk_mask = np.ones((len(span["input_ids"]), len(span["input_ids"])))

        utrs_with_same_spk = []
        # print(example.utrs_with_same_spk)
        utrs2spk = {}
        for spk, l in example.utrs_with_same_spk.items():
            utrs = []
            for k in range(len(l)):
                sid = l[k]
                if sid in utters_idx:
                    si = utters_idx.index(sid)
                    utrs.append(si)
                    s = utters_index[si]; e = utters_index[si+1]
                    spk_mask[s+1:e+1, s+1: e+1] = 3   # 同一个speaker，同一个utterance
                    if k < len(l) - 1:
                        for j in range(k+1, len(l)):
                            ssid = l[j]
                            if ssid in utters_idx:
                                ssi = utters_idx.index(ssid)
                                ss = utters_index[ssi]; ee = utters_index[ssi+1]
                                spk_mask[s+1:e+1, ss+1: ee+1] = 2   # 同一个speaker 不同utterance
                                spk_mask[ss+1:ee+1, s+1: e+1] = 2   # 同一个speaker
            if utrs:
                random.shuffle(utrs)
                utrs_with_same_spk.append(utrs)
                utrs2spk[tuple(utrs)] = spk
        
        for i in range(len(spk_mask)):
            if span["input_ids"][i] == tokenizer.pad_token_id:
                spk_mask[i : i + 1, :] = 0
                spk_mask[:, i : i + 1] = 0
                discourse_mask[i : i + 1, :] = 0
                discourse_mask[:, i : i + 1] = 0
             
        spk_mask[:query_end, :] = 0
        spk_mask[:, :query_end] = 0

        spk_gather_ids = utters_index[1:]
        spk_gather_ids.extend([0]*(max_utrs_num-len(utters_index[1:])))
        spk_utrs_mask = [1]*len(utters_index[1:])+[0]*(max_utrs_num-len(utters_index[1:]))

        spk_info_dict = {
            'spk_gather_ids': spk_gather_ids,
            'spk_utrs_mask': spk_utrs_mask,
            'spk_matrix':spk_mask,
            'spk_attn_mask':spk_attention_mask,
            'coref_matrix':coref_matrix,
        }

        start_position = 0; end_position = 0
        if is_training:
            # For training, if our document chunk does not contain an annotation
            # we throw it out, since there is nothing to predict.
            out_of_span = False

            if not (tok_start_position >= doc_start and tok_end_position <= doc_end):
                out_of_span = True

            if out_of_span:
                start_position = cls_index
                end_position = cls_index
            else:
                start_position = tok_start_position - doc_start + doc_offset
                end_position = tok_end_position - doc_start + doc_offset

        rel_info_dict = {
            'discourse_mask':discourse_mask,
            'utr_gather_ids':utr_gather_ids,
        }
        query_mapping = np.zeros_like(span["attention_mask"])
        query_mapping[:query_end] = 1
        
        Feature = FriendsQAFeature(
                span["input_ids"],
                span["attention_mask"],
                span["token_type_ids"],
                cls_index,
                p_mask.tolist(),
                example_index=0,  # Can not set unique_id and example_index here. They will be set after multiple processing.
                unique_id=0,
                paragraph_len=span["paragraph_len"],
                token_is_max_context=span["token_is_max_context"],
                tokens=span["tokens"],
                token_to_orig_map=span["token_to_orig_map"],
                start_position=start_position,
                end_position=end_position,
                qas_id=example.qas_id,
                query_end=query_end,
                query_mapping=query_mapping,
                spk_info_dict = spk_info_dict,
                rel_info_dict=rel_info_dict
            )
        features.append(Feature)
    return features


def friendsqa_convert_example_to_features_init(tokenizer_for_convert):
    global tokenizer
    tokenizer = tokenizer_for_convert


def friendsqa_convert_examples_to_features(
    examples,
    tokenizer,
    max_seq_length,
    doc_stride,
    max_query_length,
    is_training,
    padding_strategy="max_length",
    return_dataset=False,
    threads=1,
    tqdm_enabled=True,
):
    # Defining helper methods
    random.seed(42)
    features = []

    # threads = min(threads, cpu_count()) 
    # with Pool(threads, initializer=friendsqa_convert_example_to_features_init, initargs=(tokenizer,)) as p:
    #     annotate_ = partial(
    #         friendsqa_convert_example_to_features,
    #         max_seq_length=max_seq_length,
    #         doc_stride=doc_stride,
    #         max_query_length=max_query_length,
    #         padding_strategy=padding_strategy,
    #         is_training=is_training,
    #     )
    #     features = list(tqdm(
    #             p.imap(annotate_, examples, chunksize=32),
    #             total=len(examples),
    #             desc="convert molweni examples to features",
    #             disable=not tqdm_enabled,
    #         )
    #     )
    for i in tqdm(range(len(examples))):
        # try:
        features.append(friendsqa_convert_example_to_features(examples[i], 
                                                    max_seq_length=max_seq_length,
                                                    doc_stride=doc_stride,
                                                    max_query_length=max_query_length,
                                                    padding_strategy=padding_strategy,
                                                    is_training=is_training, 
                                                    tokenizer = tokenizer))
    
    new_features = []
    unique_id = 1000000000
    example_index = 0; feature_index = 0
    for example_features in tqdm(
        features, total=len(features), desc="add example index and unique id", disable=not tqdm_enabled
    ):
        if not example_features:
            continue
        for example_feature in example_features:
            example_feature.example_index = example_index
            example_feature.unique_id = unique_id
            example_feature.feature_index = feature_index
            new_features.append(example_feature)
            unique_id += 1; feature_index += 1
        example_index += 1
    features = new_features
    del new_features
    
    if return_dataset == "pt":
        if not is_torch_available():
            raise RuntimeError("PyTorch must be installed to return a PyTorch dataset.")
     
        dataset = Dataset(features)
        return features, dataset
    else:
        return features



if __name__ == "__main__":
    input_file = "/SISDC_GPFS/Home_SE/hy-suda/lyl/molweni/data/train.json"
    # speaker_mask_path = "data/speaker_mask_dev.json"

    from transformers import XLNetTokenizerFast, ElectraTokenizer, BertTokenizerFast
    # tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    # tokenizer = XLNetTokenizerFast.from_pretrained('xlnet-large-cased')
    tokenizer = ElectraTokenizer.from_pretrained('/SISDC_GPFS/Home_SE/hy-suda/pre-train_model/electra-base/')
    processor = FriendsQAProcessor(tokenizer=tokenizer, threads = 5)
    examples = processor.get_train_examples(data_dir = "/SISDC_GPFS/Home_SE/hy-suda/lyl/friendsqa/data", filename="dev2.json")
    for exp in examples:
        friendsqa_convert_example_to_features(exp, tokenizer, 512, 128, 64, 'max_length', True)
    # all_features, total_num, unptr_num, too_long_num = convert_examples_to_features(all_examples,\
    #      tokenizer, max_length=args.max_length, training=True)
    # print(total_num, unptr_num, (total_num-unptr_num) / total_num, too_long_num / total_num)

    # from transformers import XLNetTokenizerFast, ElectraTokenizerFast, BertTokenizerFast
    # from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
    # # tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    # # tokenizer = XLNetTokenizerFast.from_pretrained('xlnet-base-cased')
    # tokenizer = ElectraTokenizerFast.from_pretrained('google/electra-large-discriminator')
    # dataset = get_dataset(input_file, "tmp", tokenizer, args.max_length, training=True)
    # sampler = RandomSampler(dataset)
    # dataloader = DataLoader(dataset, sampler=sampler, batch_size=args.batch_size, collate_fn=collate_fn)
    # for batch in tqdm(dataloader):
    #     pass
    #     print(to_list(batch['utterance_ids_dict']['utterance_gather_ids'][0]))
    #     print(to_list(batch['speaker_ids_dict']['speaker_target_mask'][0]))
        
        
    