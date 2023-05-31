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



class FriendsQAExample(object):
    def __init__(self, contents_list, question, relations, qid, utterance_label = -1, left_labels = None, 
                        right_labels = None, answer_text = None, answers = None, sep_token = None):
                        
        self.relations = relations
        self.question = question
        self.qas_id = qid
        self.answer = answer_text
        self.answers = answers
        self.start_position = None
        self.end_position = None
        self.flag = True
        self.contents_list = contents_list

        doc_tokens = []
        utterances_index = []  # record the length of each utterances including [SEP]
        for uidx, utterance_text in enumerate(contents_list):
            if utterance_label>=0 and uidx == utterance_label:
                self.start_position = len(doc_tokens) + left_labels[uidx]
                self.end_position = len(doc_tokens) + right_labels[uidx]
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
            doc_tokens.append(sep_token)
            utterances_index.append(len(doc_tokens)-1)
        utterances_index = utterances_index[:-1]
        doc_tokens.pop()
        if utterance_label>=0:
            if ' '.join(answer_text.split()) != ' '.join(doc_tokens[self.start_position:self.end_position+1]):
                self.flag = False
        self.utterances_index = utterances_index
        self.doc_tokens = doc_tokens


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
        self.qas_id = qas_id
        self.query_end = query_end


class FriendsQAResult:
    """
    Constructs a QuacResult which can be used to evaluate a model's output on the QuAC dataset.
    Args:
        unique_id: The unique identifier corresponding to that example.
        start_logits: The logits corresponding to the start of the answer
        end_logits: The logits corresponding to the end of the answer
    """

    def __init__(self, unique_id, start_logits, end_logits):
        self.unique_id = unique_id
        self.start_logits = start_logits
        self.end_logits = end_logits    


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
        # examples = []
        # for d in input_data:
        #     examples.extend(self._create_examples(d, True))
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

    def _create_examples(self, data, is_training):
        examples = []               
        utterances = data["paragraphs"][0]["utterances:"]
        qas = data["paragraphs"][0]["qas"]
        relations = data['relations']
        n_length = len(utterances)
        spks_len = []; content_list = []
        for ui, utterance in enumerate(utterances):
            speaker = utterance["speakers"][0].split(" ")
            spk_num = len(speaker)
            speaker = ' '.join(speaker)
            u_text = "u" + str(ui) + " " + speaker + " : " + utterance["utterance"]
            content_list.append(u_text)
            spks_len.append(spk_num)
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
                                                        answer_text=answer_text, sep_token=self.sep_token)
                    if exp.flag:
                        examples.append(exp)
            else:
                ans = [a["answer_text"] for a in answers]
                examples.append(FriendsQAExample(qid=qid, relations=relations, contents_list=content_list, 
                                                 question=question, answers = ans, sep_token = self.sep_token))
        return examples


def friendsqa_convert_example_to_features(
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
    # if len(spans) > 1:
    #     return 1
    # else:
    #     return 0
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
            question_end_index = len(truncated_query)+1
            # global_attention_mask[:question_end_index] = 1
        else:
            p_mask[-len(span["tokens"]) : -(len(truncated_query) + sequence_added_tokens)] = 0
            question_start_index = cls_index - len(truncated_query)
            # global_attention_mask[question_start_index:] = 1
      
        pad_token_indices = np.where(span["input_ids"] == tokenizer.pad_token_id)
        special_token_indices = np.asarray(
            tokenizer.get_special_tokens_mask(span["input_ids"], already_has_special_tokens=True)
        ).nonzero()

        p_mask[pad_token_indices] = 1
        p_mask[special_token_indices] = 1
        # Set the cls index to 0: the CLS index can be used for impossible answers
        p_mask[cls_index] = 0

        start_position = 0
        end_position = 0
        if is_training:
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
            else:
                if tokenizer.padding_side == "left":
                    doc_offset = 0
                else:
                    doc_offset = len(truncated_query) + sequence_added_tokens

                start_position = tok_start_position - doc_start + doc_offset
                end_position = tok_end_position - doc_start + doc_offset

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
                query_end=span["truncated_query_with_special_tokens_length"]
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

    threads = min(threads, cpu_count())  # 进多进程时候不要tensor
    with Pool(threads, initializer=friendsqa_convert_example_to_features_init, initargs=(tokenizer,)) as p:
        annotate_ = partial(
            friendsqa_convert_example_to_features,
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

    # return features
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
                all_query_end
            )

        return features, dataset
    else:
        return features



if __name__ == "__main__":
    # input_file = "/SISDC_GPFS/Home_SE/hy-suda/lyl/molweni/data/train.json"
    # speaker_mask_path = "data/speaker_mask_dev.json"

    from transformers import XLNetTokenizerFast, ElectraTokenizer, BertTokenizerFast
    # tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    # tokenizer = XLNetTokenizerFast.from_pretrained('xlnet-large-cased')
    tokenizer = ElectraTokenizer.from_pretrained('/SISDC_GPFS/Home_SE/hy-suda/pre-train_model/electra-base/', use_fast=False)
    processor = FriendsQAProcessor(tokenizer=tokenizer, threads = 5)
    examples = processor.get_train_examples(data_dir = "/SISDC_GPFS/Home_SE/hy-suda/lyl/friendsqa/data", filename="train.json")
    # print(len(examples))   16660
    f = friendsqa_convert_examples_to_features(examples, tokenizer, 512, 128, 64, True)
    # print(sum(f)) 6543  40%
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
        
        
    