import json
import os
import random
from functools import partial
from multiprocessing import Pool, cpu_count
import numpy as np
from tqdm import tqdm
import torch.utils.data as data
import torch
from transformers.file_utils import is_tf_available, is_torch_available
from transformers.tokenization_utils_base import BatchEncoding, TruncationStrategy
from transformers.utils import logging
from transformers.data.processors.utils import DataProcessor

# Store the tokenizers which insert 2 separators tokens
MULTI_SEP_TOKENS_TOKENIZERS_SET = {"roberta", "camembert", "bart", "mpnet", "longformer"}

if is_torch_available():
    import torch
    from torch.utils.data import TensorDataset

logger = logging.get_logger(__name__)



class MolweniExample(object):
    def __init__(self, context, utterances, relations, question, qid, ori_start_pos=None,
                        answer=None, is_impossible=None, answers = None, is_training = None, sep_token = None):
        self.orig_context = context
        self.utterances = utterances
        self.relations = relations
        self.question = question
        self.qas_id = qid
        self.answer = answer
        self.is_impossible = is_impossible
        self.answers = answers
        self.start_position = None
        self.end_position = None
        self.start_position_character = None
        
        context_text = ''

        utterances_index = []  # record the length of each utterances including [SEP]
        for uidx, utterance_dict in enumerate(self.utterances):
            text = utterance_dict['text'].strip().lower()
            speaker = utterance_dict['speaker'].strip().lower()
            cur_utterance = speaker + ': ' + text + ' ' + sep_token + ' '
            context_text += cur_utterance
            utterances_index.append(len(context_text)-1)
        utterances_index = utterances_index[:-1]
        # self.utterances_index = utterances_index

        context_text = context_text[:len(context_text)-len(sep_token)-1]
        self.context_text = context_text
        
        if is_training and not self.is_impossible:
            start_position_character = self.context_text.find(self.answer)
            self.start_position_character = start_position_character

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

        self.doc_tokens = doc_tokens
        self.utterances_index = []
        for idx in utterances_index:
            self.utterances_index.append(char_to_word_offset[idx])
        
        # Start and end positions only has a value during evaluation.
        if is_training and not self.is_impossible:
            start_position = char_to_word_offset[self.start_position_character]
            end_position = char_to_word_offset[min(self.start_position_character + len(answer) - 1, len(char_to_word_offset) - 1)]
            
            # If the answer cannot be found in the text, then skip this example.
            actual_text = " ".join(self.doc_tokens[start_position : (end_position + 1)])
            cleaned_answer_text = " ".join(whitespace_tokenize(answer))
            if actual_text.find(cleaned_answer_text) == -1:
                logger.warning("Could not find answer: '%s' vs. '%s'", actual_text, cleaned_answer_text)

            self.start_position = start_position
            self.end_position = end_position

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "context: " + self.context + '\n'
        s += "utterances: " + self.utterances + '\n'
        s += "relations: " + self.relations + '\n'
        s += "question: " + self.question + '\n'
        s += "qid: " + self.qas_id + '\n'
        s += "answer: " + self.answer
        return s


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
        query_end = None,
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


class MolweniResult:
    """
    Constructs a QuacResult which can be used to evaluate a model's output on the QuAC dataset.
    Args:
        unique_id: The unique identifier corresponding to that example.
        start_logits: The logits corresponding to the start of the answer
        end_logits: The logits corresponding to the end of the answer
    """

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
        data_info['qid'] = self.features[index].qid
        data_info['input_ids'] = torch.tensor(self.features[index].input_ids, dtype=torch.long)
        data_info['token_type_ids'] = torch.tensor(self.features[index].token_type_ids, dtype=torch.long)
        data_info['attention_mask'] = torch.tensor(self.features[index].attention_mask, dtype=torch.long)
        data_info['p_mask'] = torch.tensor(self.features[index].p_mask, dtype=torch.long)
        data_info['offset_mapping'] = self.features[index].offset_mapping
        data_info['context'] = self.features[index].context
        data_info['discourse_span_mask'] = torch.tensor(self.features[index].discourse_span_mask, dtype=torch.long)
        data_info['start_pos'] = torch.tensor(self.features[index].start_pos, dtype=torch.long) if\
            self.features[index].start_pos is not None else None
        data_info['end_pos'] = torch.tensor(self.features[index].end_pos, dtype=torch.long) if\
            self.features[index].end_pos is not None else None
        data_info['is_impossible'] = torch.tensor(self.features[index].is_impossible, dtype=torch.float) if\
            self.features[index].is_impossible is not None else None
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
        """
        Returns the training examples from the data directory.
        Args:
            data_dir: Directory containing the data files used for training and evaluating.
            filename: None by default, specify this if the training file has a different name than the original one
                which is `train.json`.
        """
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
        """
        Returns the evaluation example from the data directory.
        Args:
            data_dir: Directory containing the data files used for training and evaluating.
            filename: None by default, specify this if the evaluation file has a different name than the original one
                which is `dev.json`.
        """
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
        context = dialogue['context']
        utterances = dialogue['edus']
        relations = dialogue['relations']
        for qa in dialogue['qas']:
            question = qa['question']
            qid = qa['id']
            if not is_training: # during inference
                if qa['is_impossible'] or len(qa['answers']) == 0:  answers = []
                else:   answers = qa["answers"]
                exp = MolweniExample(context, utterances, relations, question, qid, 
                                     is_training = is_training, answers = answers, sep_token = self.sep_token)
                examples.append(exp)
                continue    # 直接跳过后续步骤，因为这里不用训练
            if qa['is_impossible'] or len(qa['answers']) == 0:  # 如果不可以回答的话
                exp = MolweniExample(context, utterances, relations, question, qid, -1, '', True, 
                                     is_training = is_training, sep_token = self.sep_token)
                examples.append(exp)  
                continue   # 跳过，是无法回答的问题
            for answer in qa['answers']: # during training
                ans_text = answer['text']
                ori_start_pos = answer['answer_start']
                exp = MolweniExample(context, utterances, relations, question, qid, ori_start_pos, ans_text, False, 
                                     is_training = is_training, sep_token = self.sep_token)
                examples.append(exp)

        return examples


def molweni_convert_example_to_features(
    example, max_seq_length, doc_stride, max_query_length, padding_strategy, is_training, 
):
    features = []
    
    question_text = example.question
    answer_text = example.answer
        
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
        utterences_index.append(orig_to_tok_index[idx])
    # utterences_index.append(len(all_doc_tokens))
    
    if is_training and not example.is_impossible:
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
        # global_attention_mask = np.zeros_like(span["token_type_ids"])

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
        start_position = 0
        end_position = 0
        if is_training and not span_is_impossible:
            # For training, if our document chunk does not contain an annotation
            # we throw it out, since there is nothing to predict.
            doc_start = span["start"]
            doc_end = span["start"] + span["length"] - 1
            out_of_span = False

            if not (tok_start_position >= doc_start and tok_end_position <= doc_end):
                out_of_span = True

            if out_of_span:
                start_position = cls_index
                end_position = cls_index
                span_is_impossible = True
            else:
                if tokenizer.padding_side == "left":
                    doc_offset = 0
                else:
                    doc_offset = len(truncated_query) + sequence_added_tokens

                start_position = tok_start_position - doc_start + doc_offset
                end_position = tok_end_position - doc_start + doc_offset

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
                query_end=span["truncated_query_with_special_tokens_length"]
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
    """
    Converts a list of examples into a list of features that can be directly given as input to a model. It is
    model-dependant and takes advantage of many of the tokenizer's features to create the model's inputs.
    
    Args:
        examples: list of :class:`QuacExample`
        tokenizer: an instance of a child of :class:`~transformers.PreTrainedTokenizer`
        max_seq_length: The maximum sequence length of the inputs.
        doc_stride: The stride used when the context is too large and is split across several features.
        max_query_length: The maximum length of the query.
        is_training: whether to create features for model evaluation or model training.
        padding_strategy: Default to "max_length". Which padding strategy to use
        return_dataset: Default False. Either 'pt' or 'tf'.
            if 'pt': returns a torch.data.TensorDataset, if 'tf': returns a tf.data.Dataset
        threads: multiple processing threads.

    Returns:
        list of :class:`QuacExample`
    """
    # Defining helper methods
    random.seed(42)
    features = []

    threads = min(threads, cpu_count()) 
    with Pool(threads, initializer=molweni_convert_example_to_features_init, initargs=(tokenizer,)) as p:
        annotate_ = partial(
            molweni_convert_example_to_features,
            max_seq_length=max_seq_length,
            doc_stride=doc_stride,
            max_query_length=max_query_length,
            padding_strategy=padding_strategy,
            is_training=is_training,
        )
        features = list(
            tqdm(
                p.imap(annotate_, examples, chunksize=32),
                total=len(examples),
                desc="convert quac examples to features",
                disable=not tqdm_enabled,
            )
        )

    new_features = []
    unique_id = 1000000000
    example_index = 0
    for example_features in tqdm(
        features, total=len(features), desc="add example index and unique id", disable=not tqdm_enabled
    ):
        if not example_features:
            continue
        for example_feature in example_features:
            example_feature.example_index = example_index
            example_feature.unique_id = unique_id
            new_features.append(example_feature)
            unique_id += 1
        example_index += 1
    features = new_features
    del new_features
    
    if return_dataset == "pt":
        if not is_torch_available():
            raise RuntimeError("PyTorch must be installed to return a PyTorch dataset.")
       
        # Convert to Tensors and build dataset
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_masks = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
        all_cls_index = torch.tensor([f.cls_index for f in features], dtype=torch.long)
        all_p_mask = torch.tensor([f.p_mask for f in features], dtype=torch.float)
        all_is_impossible = torch.tensor([f.is_impossible for f in features], dtype=torch.float)
        all_query_end = torch.tensor([f.query_end for f in features], dtype=torch.long)

        if not is_training:
            all_feature_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
            dataset = TensorDataset(
                all_input_ids, all_attention_masks, all_token_type_ids, all_feature_index, all_cls_index, all_p_mask, all_query_end
            )
        else:
            all_start_positions = torch.tensor([f.start_position for f in features], dtype=torch.long)
            all_end_positions = torch.tensor([f.end_position for f in features], dtype=torch.long)
            
            dataset = TensorDataset(
                all_input_ids,          # 0
                all_attention_masks,    # 1
                all_token_type_ids,     # 2
                all_start_positions,
                all_end_positions,
                all_cls_index,
                all_p_mask,
                all_is_impossible, 
                all_query_end
            )

        return features, dataset
    elif return_dataset == "tf":
        if not is_tf_available():
            raise RuntimeError("TensorFlow must be installed to return a TensorFlow dataset.")

        def gen():
            for i, ex in enumerate(features):
                if ex.token_type_ids is None:
                    yield (
                        {
                            "input_ids": ex.input_ids,
                            "attention_mask": ex.attention_mask,
                            "feature_index": i,
                            "qas_id": ex.qas_id,
                        },
                        {
                            "start_positions": ex.start_position,
                            "end_positions": ex.end_position,
                            "cls_index": ex.cls_index,
                            "p_mask": ex.p_mask,
                            "is_impossible": ex.is_impossible,
                        },
                    )
                else:
                    yield (
                        {
                            "input_ids": ex.input_ids,
                            "attention_mask": ex.attention_mask,
                            "token_type_ids": ex.token_type_ids,
                            "feature_index": i,
                            "qas_id": ex.qas_id,
                        },
                        {
                            "start_positions": ex.start_position,
                            "end_positions": ex.end_position,
                            "cls_index": ex.cls_index,
                            "p_mask": ex.p_mask,
                            "is_impossible": ex.is_impossible,
                        },
                    )

        # Why have we split the batch into a tuple? PyTorch just has a list of tensors.
        if "token_type_ids" in tokenizer.model_input_names:
            train_types = (
                {
                    "input_ids": tf.int32,
                    "attention_mask": tf.int32,
                    "token_type_ids": tf.int32,
                    "feature_index": tf.int64,
                    "qas_id": tf.string,
                },
                {
                    "start_positions": tf.int64,
                    "end_positions": tf.int64,
                    "cls_index": tf.int64,
                    "p_mask": tf.int32,
                    "is_impossible": tf.int32,
                },
            )

            train_shapes = (
                {
                    "input_ids": tf.TensorShape([None]),
                    "attention_mask": tf.TensorShape([None]),
                    "token_type_ids": tf.TensorShape([None]),
                    "feature_index": tf.TensorShape([]),
                    "qas_id": tf.TensorShape([]),
                },
                {
                    "start_positions": tf.TensorShape([]),
                    "end_positions": tf.TensorShape([]),
                    "cls_index": tf.TensorShape([]),
                    "p_mask": tf.TensorShape([None]),
                    "is_impossible": tf.TensorShape([]),
                },
            )
        else:
            train_types = (
                {"input_ids": tf.int32, "attention_mask": tf.int32, "feature_index": tf.int64, "qas_id": tf.string},
                {
                    "start_positions": tf.int64,
                    "end_positions": tf.int64,
                    "cls_index": tf.int64,
                    "p_mask": tf.int32,
                    "is_impossible": tf.int32,
                },
            )

            train_shapes = (
                {
                    "input_ids": tf.TensorShape([None]),
                    "attention_mask": tf.TensorShape([None]),
                    "feature_index": tf.TensorShape([]),
                    "qas_id": tf.TensorShape([]),
                },
                {
                    "start_positions": tf.TensorShape([]),
                    "end_positions": tf.TensorShape([]),
                    "cls_index": tf.TensorShape([]),
                    "p_mask": tf.TensorShape([None]),
                    "is_impossible": tf.TensorShape([]),
                },
            )

        return tf.data.Dataset.from_generator(gen, train_types, train_shapes)
    else:
        return features



if __name__ == "__main__":
    input_file = "/SISDC_GPFS/Home_SE/hy-suda/lyl/molweni/data/train.json"
    # speaker_mask_path = "data/speaker_mask_dev.json"

    from transformers import XLNetTokenizerFast, ElectraTokenizer, BertTokenizerFast
    # tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    # tokenizer = XLNetTokenizerFast.from_pretrained('xlnet-large-cased')
    tokenizer = ElectraTokenizer.from_pretrained('/SISDC_GPFS/Home_SE/hy-suda/pre-train_model/electra-base/')
    processor = MolweniProcessor(tokenizer=tokenizer)
    examples = processor.get_train_examples(data_dir = "/SISDC_GPFS/Home_SE/hy-suda/lyl/molweni/data", filename="train.json")
    molweni_convert_examples_to_features(examples, tokenizer, 512, 128, 64, True)
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
        
        
    