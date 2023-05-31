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


map_relations = {'Comment': 1, 'Contrast': 4, 'Correction': 4, 'Question-answer_pair': 3, 'QAP': 3, 
                 'Parallel': 4, 'Acknowledgement': 4, 'Elaboration': 4, 'Clarification_question': 2, 
                 'Conditional': 4, 'Continuation': 4, 'Result': 4, 'Explanation': 4, 'Q-Elab': 4, 
                 'Alternation': 4, 'Narration': 4, 'Background': 4}


def floyd(W):
    node_number = len(W)
    for k in range(node_number):
        for i in range(node_number):
            for j in range(node_number):
                if W[i][k] > 0 and W[k][j] > 0 and (W[i][k] + W[k][j] < W[i][j] or W[i][j] == 0):
                    W[i][j] = W[i][k] + W[k][j]
    return W


class MolweniExample(object):
    def __init__(self, relations, question, qid, doc_tokens, relation_matrix=None, coref = None, 
                 utrs_with_same_spk=None, answer=None, is_impossible=None, answers = None , utterances = None, 
                 end_position = None, start_position=None, utterances_index = None, spks_index = None):

        self.relations = relations
        self.question = question
        self.qas_id = qid
        self.answer = answer
        self.is_impossible = is_impossible
        self.answers = answers
        self.doc_tokens = doc_tokens
        self.relation_matrix = relation_matrix 
        self.coref = coref
        self.utrs_with_same_spk = utrs_with_same_spk
        self.start_position = start_position
        self.end_position = end_position
        self.utterances_index = utterances_index
        self.spks_index = spks_index
        self.utterances = utterances


class MolweniFeature(object):
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
        is_impossible,
        qas_id: str = None,
        query_mapping = None,
        spk_info_dict=None,
        rel_info_dict=None,
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
        self.query_mapping = query_mapping
        self.spk_info_dict = spk_info_dict
        self.rel_info_dict=rel_info_dict


class MolweniResult:
    def __init__(self, unique_id, start_logits, end_logits, no_answer_probs):
        self.unique_id = unique_id
        self.start_logits = start_logits
        self.end_logits = end_logits    
        self.no_answer_probs = no_answer_probs


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
        data_info['no_answer'] = torch.tensor(self.features[index].is_impossible, dtype=torch.float) if\
                                        self.features[index].is_impossible is not None else None
        data_info['query_mapping'] = torch.tensor(self.features[index].query_mapping, dtype=torch.float) 
        data_info['spk_info_dict'] = self.features[index].spk_info_dict
        data_info['rel_info_dict'] = self.features[index].rel_info_dict
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


class MolweniProcessor(DataProcessor):
    """
    Processor for the Molweni data set.
    """
    train_file = "train.json"
    dev_file = "dev.json"

    def __init__(self, tokenizer, threads):
        self.threads = threads
        self.sep_token = tokenizer.sep_token

    def get_train_examples(self, data_dir, filename=None):
        if data_dir is None:
            data_dir = ""

        if self.train_file is None:
            raise ValueError("--train_file should be given.")

        with open(
            os.path.join(data_dir, self.train_file if filename is None else filename), "r", encoding="utf-8"
        ) as reader:
            input_data = json.load(reader)["data"]['dialogues']

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
        if data_dir is None:
            data_dir = ""

        if self.dev_file is None:
            raise ValueError("--dev_file should be given.")

        with open(
            os.path.join(data_dir, self.dev_file if filename is None else filename), "r", encoding="utf-8"
        ) as reader:
            input_data = json.load(reader)["data"]['dialogues']

        threads = min(self.threads, cpu_count())
        with Pool(threads) as p:
            annotate_ = partial(self._create_examples, is_training=False)
            examples = list(tqdm(
                p.imap(annotate_, input_data),
                total=len(input_data),
                desc="collect molweni examples to",
            ))
        examples = [item for sublist in examples for item in sublist]
        return examples

    def _create_examples(self, dialogue, is_training):
        examples = []
        context_text = ''
        utrs_with_same_spk = {}
        utterances_index = []  # record the length of each utterances including [SEP]
        spk2pos = {}
        for uidx, utterance_dict in enumerate(dialogue['edus']):
            text = utterance_dict['text'].strip().lower()
            speaker = utterance_dict['speaker'].strip().lower()
            if speaker in utrs_with_same_spk:
                utrs_with_same_spk[speaker].append(uidx)
                spk2pos[speaker].append([len(context_text), len(context_text)+len(speaker), uidx])
            else:
                utrs_with_same_spk[speaker] = [uidx]
                spk2pos[speaker] = [[len(context_text), len(context_text)+len(speaker), uidx]]
            cur_utterance = speaker + ': ' + text + ' ' + self.sep_token + ' '
            context_text += cur_utterance
            utterances_index.append(len(context_text)-1)
        
        relation_matrix = [[0 for _ in range(len(dialogue['edus']))] for _ in range(len(dialogue['edus']))]
        relations = {}
        for rel in dialogue['relations']:
            y = rel['y']; x = rel['x']; 
            if y < len(dialogue['edus']) and x < len(dialogue['edus']):
                relation_matrix[y][x] = 1
                if rel['type'] in map_relations:
                    t = map_relations[rel['type']]
                    relations[(y, x)] = t
        relation_matrix = floyd(relation_matrix)
        
        utterance_index = utterances_index[:-1]
        context_text = context_text[:len(context_text)-len(self.sep_token)-1]
        
        char_orig_contect2context = {}; ct = 0
        for i, c in enumerate(dialogue['context']):
            while c != context_text[ct]:
                ct += 1
            char_orig_contect2context[i] = ct
            ct += 1
        
        doc_tokens = []
        char_to_word_offset = []
        prev_is_whitespace = True
        # Split on whitespace so that different tokens may be attributed to their original position.
        for c in context_text:
            if _is_whitespace(c):
                prev_is_whitespace = True
            else:
                if prev_is_whitespace:
                    doc_tokens.append(c)
                else:
                    doc_tokens[-1] += c
                prev_is_whitespace = False
            char_to_word_offset.append(len(doc_tokens) - 1)

        utterances_index = []
        for idx in utterance_index:
            utterances_index.append(char_to_word_offset[idx])
       
        coref = []
        for cluster in dialogue['corefs']:
            clu = []
            is_spk = ''
            for c in cluster:
                if c[0] in spk2pos:
                    is_spk = c[0]
                    continue
                clu.append([
                    c[0], 
                    char_to_word_offset[char_orig_contect2context[c[1]]], 
                    char_to_word_offset[char_orig_contect2context[c[2]-1]],
                    ])
            if is_spk:
                clu.extend([[is_spk, char_to_word_offset[l[0]], char_to_word_offset[l[1]-1], l[2]] for l in spk2pos[is_spk]])
                del spk2pos[is_spk]
            if len(clu) > 1:
                coref.append(clu)
        for spk, sl in spk2pos.items():
            if len(sl) > 1:
                coref.append([[spk, char_to_word_offset[l[0]], char_to_word_offset[l[1]-1], l[2]] for l in sl])
        
        for qa in dialogue['qas']:
            question = qa['question']
            qid = qa['id']
            
            if qa['is_impossible'] or len(qa['answers']) == 0:  
                answers = []
            else:   
                answers = qa["answers"]
                    
            if not is_training: # during inference
                exp = MolweniExample(relations, question, qid, doc_tokens=doc_tokens, coref = coref, 
                                     relation_matrix = relation_matrix, answers = answers, utterances = dialogue['edus'],
                                     utrs_with_same_spk=utrs_with_same_spk, utterances_index = utterances_index)
                examples.append(exp)
                continue    # 直接跳过后续步骤，因为这里不用训练
            if qa['is_impossible'] or len(qa['answers']) == 0:  # 如果不可以回答的话
                exp = MolweniExample(relations, question, qid, doc_tokens = doc_tokens, coref = coref, 
                                     relation_matrix = relation_matrix, utrs_with_same_spk=utrs_with_same_spk,
                                     utterances_index = utterances_index, is_impossible = True, 
                                     answers = answers, utterances = dialogue['edus'])
                examples.append(exp)  
                continue   # 跳过，是无法回答的问题
            for answer in qa['answers']: # during training
                start_position_character = context_text.find(answer['text'])
                #assert start_position_character > -1 and (abs(start_position_character - answer['answer_start']) < 10 or start_position_character >= answer['answer_start'])
                
                start_position = char_to_word_offset[start_position_character]
                end_position = char_to_word_offset[min(start_position_character + len(answer['text']) - 1, len(char_to_word_offset) - 1)]
                
                # If the answer cannot be found in the text, then skip this example.
                actual_text = " ".join(doc_tokens[start_position : (end_position + 1)])
                cleaned_answer_text = " ".join(whitespace_tokenize(answer['text']))
                if actual_text.find(cleaned_answer_text) == -1:
                    logger.warning("Could not find answer: '%s' vs. '%s'", actual_text, cleaned_answer_text)
                
                exp = MolweniExample(relations, question, qid, doc_tokens = doc_tokens, coref = coref, answers = answers,
                                     relation_matrix = relation_matrix, utrs_with_same_spk=utrs_with_same_spk, 
                                     answer = answer['text'], start_position=start_position, end_position=end_position,
                                     utterances_index = utterances_index, is_impossible = False, utterances = dialogue['edus'])
                examples.append(exp)
        return examples


def molweni_convert_example_to_features(
    example, max_seq_length, doc_stride, max_query_length, padding_strategy, is_training, 
    max_utrs_num = 14
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
    
    if is_training and not example.is_impossible:
        tok_start_position = orig_to_tok_index[example.start_position]
        if example.end_position < len(example.doc_tokens) - 1:
            tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
        else:
            tok_end_position = len(all_doc_tokens) - 1

        (tok_start_position, tok_end_position) = _improve_answer_span(
            all_doc_tokens, tok_start_position, tok_end_position, tokenizer, answer_text
        )

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

        span_is_impossible = example.is_impossible
        doc_start = span["start"]
        doc_end = span["start"] + span["length"] - 1

        if tokenizer.padding_side == "left":
            doc_offset = 0
        else:
            doc_offset = len(truncated_query) + sequence_added_tokens

        utters_index = [doc_offset-1]
        utters_idx = []
        for i, idx in enumerate(utterences_index):
            if idx > doc_start and idx <= doc_end:
                utters_index.append(idx - doc_start + doc_offset)
                utters_idx.append(i)
            elif i > 0 and i-1 in utters_idx and idx > doc_start:
                utters_index.append(doc_end - doc_start + doc_offset)
                utters_idx.append(i)
                break
        utters_index[-1] = len(span['tokens'])-1
        
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
        qasutr_mask = np.zeros((len(span["input_ids"]), len(span["input_ids"])))
        coref_mask = np.zeros((len(span["input_ids"]), len(span["input_ids"])))

        ############### coref mask #############################
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
        
        ############### discourse mask #############################
        for i in range(len(utters_index) - 1):
            s = utters_index[i]; e = utters_index[i+1]
            discourse_mask[s+1 : e+1, s+1 : e+1] = 1
        query_end = span["truncated_query_with_special_tokens_length"]

        for i in range(len(relation_matrix)):
            for j in range(len(relation_matrix)):
                if i != j and relation_matrix[i][j] > 0: 
                    if i in utters_idx and j in utters_idx:
                        s = utters_idx.index(i); e = utters_idx.index(j)
                        s1 = utters_index[s]; e1 = utters_index[s+1]
                        s2 = utters_index[e]; e2 = utters_index[e+1]
                        if relation_matrix[i][j] <= 2:
                            discourse_mask[s1+1 : e1+1, s2+1 : e2+1] = 1

        utr_gather_ids = utters_index[1:]
        utr_gather_ids.extend([0]*(max_utrs_num-len(utters_idx)))

        # discourse_mask[:query_end, :] = 1
        # discourse_mask[:, :query_end] = 1
        qasutr_mask[:query_end, :] = 1
        qasutr_mask[:, :query_end] = 1
        
        ############### speaker mask #############################
        spk_attention_mask = span['attention_mask'].copy()
        for i in range(query_end):
            spk_attention_mask[i] = 0
        spk_mask = np.ones((len(span["input_ids"]), len(span["input_ids"])))

        utrs_with_same_spk = []
        for spk, l in example.utrs_with_same_spk.items():
            utrs = []
            for k in range(len(l)):
                sid = l[k]
                if sid in utters_idx:
                    si = utters_idx.index(sid)
                    utrs.append(si)
                    s = utters_index[si]; e = utters_index[si+1]
                    spk_mask[s+1:e+1, s+1: e+1] = 3   # 同一个speaker
                    if k < len(l) - 1:
                        for j in range(k+1, len(l)):
                            ssid = l[j]
                            if ssid in utters_idx:
                                ssi = utters_idx.index(ssid)
                                ss = utters_index[ssi]; ee = utters_index[ssi+1]
                                spk_mask[s+1:e+1, ss+1: ee+1] = 2   # 同一个speaker
                                spk_mask[ss+1:ee+1, s+1: e+1] = 2   # 同一个speaker
            if utrs:
                utrs_with_same_spk.append(utrs)
        
        for i in range(len(spk_mask)):
            if span["input_ids"][i] == tokenizer.pad_token_id:
                spk_mask[i : i + 1, :] = 0
                spk_mask[:, i : i + 1] = 0
                discourse_mask[i : i + 1, :] = 0
                discourse_mask[:, i : i + 1] = 0
                qasutr_mask[i : i + 1, :] = 0
                qasutr_mask[:, i : i + 1] = 0
                
        spk_mask[:query_end, :] = 0
        spk_mask[:, :query_end] = 0

       
        spk_info_dict = {
            'spk_matrix':spk_mask,
            'spk_attn_mask':spk_attention_mask,
            'coref_matrix':coref_matrix,
        }

        start_position = 0; end_position = 0
        if is_training and not span_is_impossible:
            # For training, if our document chunk does not contain an annotation
            # we throw it out, since there is nothing to predict.
            out_of_span = False

            if not (tok_start_position >= doc_start and tok_end_position <= doc_end):
                out_of_span = True

            if out_of_span:
                start_position = cls_index
                end_position = cls_index
                span_is_impossible = True
            else:
                start_position = tok_start_position - doc_start + doc_offset
                end_position = tok_end_position - doc_start + doc_offset
        
        query_mapping = np.zeros_like(span["attention_mask"])
        query_mapping[:query_end] = 1
        context_mapping = np.zeros_like(span["attention_mask"])
        context_mapping[query_end: query_end + span["length"]] = 1
        
        rel_info_dict = {
            'discourse_mask':discourse_mask,
            'utr_gather_ids':utr_gather_ids,
            'context_mapping': context_mapping,
            'qasutr_mask':qasutr_mask
        }
        
        Feature = MolweniFeature(
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
                is_impossible= 1 if span_is_impossible else 0,
                qas_id=example.qas_id,
                query_mapping=query_mapping,
                spk_info_dict = spk_info_dict,
                rel_info_dict=rel_info_dict
            )
        features.append(Feature)
    return features


def molweni_convert_example_to_features_init(tokenizer_for_convert):
    global tokenizer
    tokenizer = tokenizer_for_convert


def molweni_convert_examples_to_features(
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

    threads = min(threads, cpu_count())  # 进多进程时候不要tensor
    with Pool(threads, initializer=molweni_convert_example_to_features_init, initargs=(tokenizer,)) as p:
        annotate_ = partial(
            molweni_convert_example_to_features,
            max_seq_length=max_seq_length,
            doc_stride=doc_stride,
            max_query_length=max_query_length,
            padding_strategy=padding_strategy,
            is_training=is_training,
        )
        features = list(tqdm(
                p.imap(annotate_, examples, chunksize=32),
                total=len(examples),
                desc="convert molweni examples to features",
                disable=not tqdm_enabled,
            )
        )
    # for i in tqdm(range(len(examples))):
    #     # try:
    #     features.append(molweni_convert_example_to_features(examples[i], 
    #                                                 max_seq_length=max_seq_length,
    #                                                 doc_stride=doc_stride,
    #                                                 max_query_length=max_query_length,
    #                                                 padding_strategy=padding_strategy,
    #                                                 is_training=is_training, 
    #                                                 tokenizer = tokenizer))
        
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
    # input_file = "/SISDC_GPFS/Home_SE/hy-suda/lyl/molweni/data/dev2.json"
    # speaker_mask_path = "data/speaker_mask_dev.json"

    from transformers import XLNetTokenizerFast, ElectraTokenizer, BertTokenizerFast
    # tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    # tokenizer = XLNetTokenizerFast.from_pretrained('xlnet-large-cased')
    tokenizer = ElectraTokenizer.from_pretrained('/SISDC_GPFS/Home_SE/hy-suda/pre-train_model/electra-base/')
    processor = MolweniProcessor(tokenizer=tokenizer, threads = 5)
    examples = processor.get_train_examples(data_dir = "/SISDC_GPFS/Home_SE/hy-suda/lyl/molweni/data", filename="train2.json")
    for exp in examples:
        # print('----')
        molweni_convert_example_to_features(exp, tokenizer, 384, 128, 64, 'max_length', True)
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
        
        
    