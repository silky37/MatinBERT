import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
import sys
sys.path.append("EJ_MatSciBERT/ner")
from pathlib import Path

from distutils.util import strtobool as _bool

from argparse import ArgumentParser
import numpy as np

import shutil
import torch
import time

import ner_datasets
# import EJ_MatSciBERT.ner.ner_datasets as ner_datasets
from models import BERT_CRF, BERT_BiLSTM_CRF, BERT_CRF_Contrastive, BERT_CRF_WordContrastive, BERT_CRF_BothContrastive
# from EJ_MatSciBERT.ner.models import BERT_CRF, BERT_BiLSTM_CRF, BERT_CRF_Contrastive, BERT_CRF_WordContrastive, BERT_CRF_BothContrastive
import conlleval
# import EJ_MatSciBERT.ner.conlleval
from logger import get_logger
# from EJ_MatSciBERT.ner.logger import get_logger
from trainer import ContrastiveTrainer, WordContrastiveTrainer, BothContrastiveTrainer
# from EJ_MatSciBERT.ner.trainer import ContrastiveTrainer, WordContrastiveTrainer, BothContrastiveTrainer
from embedding_utils.embedding_initializer import transfer_embedding
# from EJ_MatSciBERT.embedding_utils.embedding_initializer import transfer_embedding

from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
    AdamW,
)

from tqdm import tqdm

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

#print('using device:', device)


def ensure_dir(dir_path):
    Path(dir_path).mkdir(parents=True, exist_ok=True)
    return dir_path

def embedding(args, model, d2p):
    transfer_embedding(model.encoder, d2p, args.transfer_type)


parser = ArgumentParser()
parser.add_argument('--mode', type=str, default='dummy')
parser.add_argument('--host', type=str, default='dummy')
parser.add_argument('--port', type=int, default=56789)
parser.add_argument('--model_name', choices=['scibert', 'matscibert', 'bert', 't5'], type=str, default="t5")
parser.add_argument('--model_save_dir', type=str, default=None)
parser.add_argument('--preds_save_dir', default=None, type=str)
parser.add_argument('--log_dir', default=None, type=str)
parser.add_argument('--cache_dir', default=None, type=str)
parser.add_argument('--seeds', nargs='+', default=None, type=int)
parser.add_argument('--lm_lrs', nargs='+', default=None, type=float)
parser.add_argument('--non_lm_lr', default=3e-4, type=float)
parser.add_argument('--architecture', choices=['bert', 'bert-crf', 'bert-bilstm-crf'], type=str, default="bert-crf")
parser.add_argument('--dataset_name', choices=['sofc', 'sofc_slot', 'matscholar'], type=str, default="matscholar")
parser.add_argument('--total_fold_num', default=5, type=int)
parser.add_argument('--hidden_dim', default=300, type=int)
parser.add_argument('--contrastive', default=False, type=_bool)
parser.add_argument('--contrastive_type', choices=['sentence', 'word', 'both'], default="sentence", type=str)
parser.add_argument('--temperature', nargs='+', default=None, type=float)
parser.add_argument("--transfer_type", choices=["random", "average_input"], default="average_input", type=str)
parser.add_argument("--vocab_file", type=str, default='')
parser.add_argument("--vocab_type", choices=['base', 'ours', 'avocado', 'adalm'], type=str, default='base')
args = parser.parse_args()


if args.model_save_dir is None:
    if args.contrastive:
        args.model_save_dir = f"EJ_MatSciBERT/ner/model/{args.model_name}/{args.dataset_name}/{args.vocab_type}_cont_{args.contrastive_type}"
        args.preds_save_dir = f"EJ_MatSciBERT/ner/preds/{args.model_name}/{args.dataset_name}/{args.vocab_type}_cont_{args.contrastive_type}"
        args.log_dir = f"EJ_MatSciBERT/ner/log/{args.model_name}/{args.dataset_name}/{args.vocab_type}_cont_{args.contrastive_type}"
    else:
        args.model_save_dir = f"EJ_MatSciBERT/ner/model/{args.model_name}/{args.dataset_name}/{args.vocab_type}"
        args.preds_save_dir = f"EJ_MatSciBERT/ner/preds/{args.model_name}/{args.dataset_name}/{args.vocab_type}"
        args.log_dir = f"EJ_MatSciBERT/ner/log/{args.model_name}/{args.dataset_name}/{args.vocab_type}"
else: pass

if args.cache_dir is None:
    args.cache_dir = 'EJ_MatSciBERT/cache/'
else:
    pass


logger = get_logger(log_path=os.path.join(args.log_dir, "train_log.txt"))
logger.info("##########################################")
logger.info("###########  Start the Script  ###########")
logger.info("##########################################")
logger.info(f"* Arguments: {args}\n")
logger.info(f'* Using device: {device}')

if args.model_name == 'scibert':
    model_name = 'allenai/scibert_scivocab_uncased'
    to_normalize = False
elif args.model_name == 'matscibert':
    model_name = 'm3rg-iitd/matscibert'
    to_normalize = True
elif args.model_name == 'bert':
    model_name = 'bert-base-uncased'
    to_normalize = False
elif args.model_name == 't5':
    model_name = 'google/t5-v1_1-base'
    to_normalize = False
else:
    raise NotImplementedError


# if args.temperature is None:
#     if args.dataset_name == 'matscholar':
#         args.temperature = [1.5, 2.0, 2.5, 3.0]
#     elif 'sofc' in args.dataset_name:
#         args.temperature = [1.5, 2.0, 2.5]
#     else:
#         raise NotImplementedError
if args.contrastive and args.temperature is None:
    if args.dataset_name == 'matscholar':
        args.temperature = [1.5, 2.0, 2.5, 3.0]
    elif 'sofc' in args.dataset_name:
        args.temperature = [1.5, 2.0, 2.5]
    else:
        raise NotImplementedError
elif args.temperature is None:
    args.temperature = [1.5]


logger.info(f"* Task: {args.dataset_name}")
logger.info(f"* Using model: {args.model_name}")
logger.info(f"* Using torch model: {model_name}")
logger.info(f"* Architecture: {args.architecture}")
logger.info(f"* Total Folds: {args.total_fold_num}")
logger.info(f"* LM_Lrs: {args.lm_lrs}")
logger.info(f"* SEEDs: {args.seeds}")
if args.contrastive:
    logger.info(f"* Contrastive: True")
    logger.info(f"* Contrastive Type: {args.contrastive_type}")
    logger.info(f"* Temperature: {args.temperature}\n")
else:
    logger.info(f"* Contrastive: False\n")



metric_for_best_model = 'macro_f1' if args.dataset_name[: 4] == 'sofc' or args.dataset_name == 'matscholar' else 'micro_f1'
other_metric = 'micro_f1' if metric_for_best_model == 'macro_f1' else 'macro_f1'
logger.info(f"* metric_for_best_model: {metric_for_best_model}")
logger.info(f"* other_metric: {other_metric}\n")


if args.dataset_name == 'sofc':
    num_epochs = 20
elif args.dataset_name == 'sofc_slot':
    num_epochs = 40
elif args.dataset_name == 'matscholar':
    num_epochs = 15
else:
    raise NotImplementedError


logger.info("")
fold_best_lr = list()
fold_best_temp = list()
fold_best_val = list()
fold_best_test = list()
fold_best_val_list = list()
fold_best_test_list = list()
fold_best_val_all = dict()
fold_best_test_all = dict()
for fold in range(1, args.total_fold_num+1):
# for fold in range(5,6):
    logger.info(f"##### Start Training for fold {fold} #####")

    model_revision = 'main'
    cache_dir = ensure_dir(args.cache_dir) if args.cache_dir else None
    output_dir = ensure_dir(args.model_save_dir)
    preds_save_dir = ensure_dir(args.preds_save_dir) if args.preds_save_dir else None
    if preds_save_dir:
        preds_save_dir = os.path.join(preds_save_dir, args.dataset_name)
        preds_save_dir = ensure_dir(preds_save_dir)

    if args.seeds is None:
        args.seeds = [1997, 2024, 2017]
    if args.lm_lrs is None:
        args.lm_lrs = [2e-5, 3e-5, 5e-5]

    train_X, train_y, val_X, val_y, test_X, test_y = ner_datasets.get_ner_data(args.dataset_name, fold=fold, norm=to_normalize)

    # print(len(train_X), len(val_X), len(test_X))
    logger.info(f"* Data: {args.dataset_name}")
    logger.info(f"  ㄴ Train: {len(train_X)} | Val: {len(val_X)} | Test: {len(test_X)}")

    unique_labels = set(label for sent in train_y for label in sent)
    label_list = sorted(list(unique_labels))
    # print(label_list)
    tag2id = {tag: idx for idx, tag in enumerate(label_list)}
    id2tag = {idx: tag for tag, idx in tag2id.items()}
    if args.dataset_name == 'sofc_slot':
        id2tag[tag2id['B-experiment_evoking_word']] = 'O'
        id2tag[tag2id['I-experiment_evoking_word']] = 'O'
    num_labels = len(label_list)

    logger.info(f"* Num_labels: {num_labels}\n")
    # logger.info(f"* Labels: {label_list}")

    cnt = dict()
    for sent in train_y:
        for label in sent:
            if label[0] in ['I', 'B']:
                tag = label[2:]
            else:
                continue
            if tag not in cnt:
                cnt[tag] = 1
            else:
                cnt[tag] += 1

    eval_labels = sorted([l for l in cnt.keys() if l != 'experiment_evoking_word'])

    config_kwargs = {
        'num_labels': num_labels,
        'cache_dir': cache_dir,
        'revision': model_revision,
        'use_auth_token': None,
    }
    config = AutoConfig.from_pretrained(model_name, **config_kwargs)

    tokenizer_kwargs = {
        'cache_dir': cache_dir,
        'use_fast': True,
        'revision': model_revision,
        'use_auth_token': None,
        'model_max_length': 512
    }

    if args.vocab_file.strip() == '':  # base
        tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
        raw_tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
        logger.info(f"* Tokenizer Length (base): {len(tokenizer)}")
        logger.info(f"* Tokenizer kwargs: {tokenizer_kwargs}\n")
    else:
        raw_tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
        original_vocab_size = len(raw_tokenizer)
        logger.info(f"* Tokenizer Length (before addition): {original_vocab_size}")

        tok_name = args.model_name
        tokenizer_dir = f"../vocab_expansion/logs/{tok_name}/{args.dataset_name}"
        # logger.info(f"* Saving tokenizer to {tokenizer_dir}")
        raw_tokenizer.save_pretrained(tokenizer_dir)
        config.save_pretrained(tokenizer_dir)

        # vocab_path = os.path.join('./vocab', dataset_name ,'expanded.vocab')
        vocab_path = args.vocab_file
        shutil.copyfile(vocab_path, os.path.join(tokenizer_dir, 'vocab.txt'))
        os.remove(os.path.join(tokenizer_dir, 'tokenizer.json'))
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, **tokenizer_kwargs)
        # logger.info(f"* Loading tokenizer from {tokenizer_dir}")
        # logger.info(f"* Expanded Vocab Size: {len(tokenizer)}")

        logger.info(f"* Tokenizer Length (after addition): {len(tokenizer)}")
        logger.info(f"* Tokenizer kwargs: {tokenizer_kwargs}\n")


    def remove_zero_len_tokens(X, y):
        new_X, new_y = [], []
        for sent, labels in zip(tqdm(X), y):
            new_sent, new_labels = [], []
            for token, label in zip(sent, labels):
                if len(tokenizer.tokenize(token)) == 0:
                    assert args.dataset_name == 'matscholar'
                    continue
                new_sent.append(token)
                new_labels.append(label)
                # print('#'*10)
                # print(len(new_sent), len(new_labels))
            assert len(new_sent) == len(new_labels)
            new_X.append(new_sent)
            new_y.append(new_labels)
        return new_X, new_y


    train_X, train_y = remove_zero_len_tokens(train_X, train_y)
    val_X, val_y = remove_zero_len_tokens(val_X, val_y)
    test_X, test_y = remove_zero_len_tokens(test_X, test_y)

    train_encodings = tokenizer(train_X, is_split_into_words=True, return_offsets_mapping=True, padding=True, truncation=True, max_length=512)
    val_encodings = tokenizer(val_X, is_split_into_words=True, return_offsets_mapping=True, padding=True, truncation=True, max_length=512)
    test_encodings = tokenizer(test_X, is_split_into_words=True, return_offsets_mapping=True, padding=True, truncation=True, max_length=512)

    if args.contrastive:
        raw_train_encodings = raw_tokenizer(train_X, is_split_into_words=True, return_offsets_mapping=True, padding=True, truncation=True)
        raw_train_encodings.pop('offset_mapping')


    def encode_tags(tags, encodings):
        labels = [[tag2id[tag] for tag in doc] for doc in tags]
        encoded_labels = []
        for doc_labels, doc_offset in zip(labels, encodings.offset_mapping):
            # print(f"doc_labels: {doc_labels}")
            # print(f"doc_offset: {doc_offset}")
            doc_enc_labels = np.ones(len(doc_offset), dtype=int) * -100
            arr_offset = np.array(doc_offset)
            # print(f"doc_enc_labels: {doc_enc_labels}")
            # print(f"arr_offset: {arr_offset}")
            doc_enc_labels[(arr_offset[:, 0] == 0) & (arr_offset[:, 1] != 0)] = doc_labels
            encoded_labels.append(doc_enc_labels.tolist())
        return encoded_labels


    train_labels = encode_tags(train_y, train_encodings)
    val_labels = encode_tags(val_y, val_encodings)
    test_labels = encode_tags(test_y, test_encodings)

    train_encodings.pop('offset_mapping')
    val_encodings.pop('offset_mapping')
    test_encodings.pop('offset_mapping')


    class NER_Dataset(torch.utils.data.Dataset):
        def __init__(self, inp, labels):
            self.inp = inp
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.inp.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item

        def __len__(self):
            return len(self.labels)


    class Contrastive_NER_Dataset(torch.utils.data.Dataset):
        def __init__(self, inp, org, labels):
            self.inp = inp      # tokenized data
            self.org = org      # raw tokenized data
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.inp.items()}
            item.update({f"raw_{key}": torch.tensor(val[idx]) for key, val in self.org.items()})
            item['labels'] = torch.tensor(self.labels[idx])
            return item

        def __len__(self):
            return len(self.labels)

    class WordContrastive_NER_Dataset(torch.utils.data.Dataset):
        def __init__(self, inp, org, labels, tokenizer, raw_tokenizer):
            self.inp = inp  # encoded data
            self.org = org  # raw encoded data
            self.labels = labels
            self.inp_emb_idx = []
            self.org_emb_idx = []
            for i in tqdm(range(len(self.inp['input_ids'])), desc='Mapping Embedding Index...', total=len(self.inp['input_ids']), bar_format="{l_bar}{bar:15}{r_bar}"):
                max_inp_seq_len = len(self.inp['input_ids'][i])
                max_org_seq_len = len(self.org['input_ids'][i])
                # if i == 0:
                #     print("\n\n")
                #     print(f"Max INP Seq Len: {max_inp_seq_len}")
                #     print(f"Max ORG Seq Len: {max_org_seq_len}")
                #     print("\n\n")
                sentence = raw_tokenizer.decode(self.org['input_ids'][i])
                cur_inp_emb_idx = [0]       # for [CLS]
                cur_org_emb_idx = [0]       # for [CLS]
                for word in sentence.split(' '):
                    org_tokens = raw_tokenizer.tokenize(word)
                    inp_tokens = tokenizer.tokenize(word)
                    if org_tokens == inp_tokens:
                        cur_inp_emb_idx += [0] * len(inp_tokens)
                        cur_org_emb_idx += [0] * len(org_tokens)
                    else:
                        cur_inp_emb_idx += [1] * len(inp_tokens)
                        cur_org_emb_idx += [1] * len(org_tokens)
                if len(cur_inp_emb_idx) < max_inp_seq_len:
                    cur_inp_emb_idx += [0] * (max_inp_seq_len - len(cur_inp_emb_idx))
                else:
                    cur_inp_emb_idx = cur_inp_emb_idx[:max_inp_seq_len]
                if len(cur_org_emb_idx) < max_org_seq_len:
                    cur_org_emb_idx += [0] * (max_org_seq_len - len(cur_org_emb_idx))
                else:
                    cur_org_emb_idx = cur_org_emb_idx[:max_org_seq_len]

                self.inp_emb_idx.append(cur_inp_emb_idx)
                self.org_emb_idx.append(cur_org_emb_idx)

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.inp.items()}
            item.update({f"raw_{key}": torch.tensor(val[idx]) for key, val in self.org.items()})
            item.update({f"inp_emb_idx": torch.tensor(self.inp_emb_idx[idx])})
            item.update({f"org_emb_idx": torch.tensor(self.org_emb_idx[idx])})
            item['labels'] = torch.tensor(self.labels[idx])
            return item

        def __len__(self):
            return len(self.labels)

    if args.contrastive:
        if args.contrastive_type == 'sentence':
            train_dataset = Contrastive_NER_Dataset(train_encodings, raw_train_encodings, train_labels)
        elif args.contrastive_type in ['word', 'both']:
            train_dataset = WordContrastive_NER_Dataset(train_encodings, raw_train_encodings, train_labels, tokenizer, raw_tokenizer)
        else:
            raise NotImplementedError
    else:
        train_dataset = NER_Dataset(train_encodings, train_labels)
    val_dataset = NER_Dataset(val_encodings, val_labels)
    test_dataset = NER_Dataset(test_encodings, test_labels)


    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)
        true_predictions = [
            [id2tag[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [id2tag[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        preds, labs = [], []
        for pred, lab in zip(true_predictions, true_labels):
            preds.extend(pred)
            labs.extend(lab)
        assert (len(preds) == len(labs))
        labels_and_predictions = [" ".join([str(i), labs[i], preds[i]]) for i in range(len(labs))]
        counts = conlleval.evaluate(labels_and_predictions)
        scores = conlleval.get_scores(counts)
        results = {}
        macro_f1 = 0
        for k in eval_labels:
            if k in scores:
                results[k] = scores[k][-1]
            else:
                results[k] = 0.0
            macro_f1 += results[k]
        macro_f1 /= len(eval_labels)
        results['macro_f1'] = macro_f1 / 100
        results['micro_f1'] = conlleval.metrics(counts)[0].fscore
        return results


    arch = args.architecture if args.architecture != 'bert-bilstm-crf' else f'bert-bilstm-crf-{args.hidden_dim}'
    dump_eval_keys = ['eval_loss', 'eval_macro_f1', 'eval_micro_f1', 'eval_runtime', 'eval_samples_per_second', 'epoch']

    best_lr = 0
    best_temp = 0
    best_val = 0
    best_test = 0
    best_val_all = dict()
    best_test_all = dict()
    best_val_acc_list = list()
    best_test_acc_list = list()
    best_val_oth_list = list()
    best_test_oth_list = list()
    """
    Find the best hyperparameters for the model among the given temperature and learning rates.
    For each temperature and learning rate, train the model and evaluate it on three given different random seeds.
    """
    for temp_id, temperature in enumerate(args.temperature):
        for l_id, lr in enumerate(args.lm_lrs):
            val_acc, val_oth = [], []
            val_all = {}
            test_acc, test_oth = [], []
            test_all = {}
            for seed_id, SEED in enumerate(args.seeds):
                start_time = time.time()
                logger.info(f"* Fold: {fold} ({fold}/{args.total_fold_num})")
                logger.info(f"* Temperature: {temperature} ({temp_id+1}/{len(args.temperature)})")
                logger.info(f'* Lr: {lr} ({l_id+1}/{len(args.lm_lrs)})')
                logger.info(f'* Seed: {SEED} ({seed_id+1}/{len(args.seeds)})')

                torch.use_deterministic_algorithms(True)
                torch.backends.cudnn.benchmark = False
                set_seed(SEED)

                training_args = TrainingArguments(
                    num_train_epochs=num_epochs,
                    output_dir=output_dir,
                    per_device_train_batch_size=8,
                    per_device_eval_batch_size=16,
                    gradient_accumulation_steps=2,
                    evaluation_strategy='epoch',
                    load_best_model_at_end=True,
                    metric_for_best_model=metric_for_best_model,
                    greater_is_better=True,
                    save_total_limit=2,
                    warmup_ratio=0.1,
                    learning_rate=lr,
                    seed=SEED,
                )

                if args.architecture == 'bert':
                    model = AutoModelForTokenClassification.from_pretrained(
                        model_name, from_tf=False, config=config,
                        cache_dir=cache_dir, revision=model_revision, use_auth_token=None,
                    )
                elif args.architecture == 'bert-crf':
                    if args.contrastive:
                        if args.contrastive_type == 'sentence':
                            model = BERT_CRF_Contrastive(model_name, device, config, cache_dir)
                        elif args.contrastive_type == 'word':
                            model = BERT_CRF_WordContrastive(model_name, device, config, cache_dir)
                        elif args.contrastive_type == 'both':
                            model = BERT_CRF_BothContrastive(model_name, device, config, cache_dir)
                        else:
                            raise NotImplementedError
                    else:
                        model = BERT_CRF(model_name, device, config, cache_dir)
                elif args.architecture == 'bert-bilstm-crf':
                    model = BERT_BiLSTM_CRF(model_name, device, config, cache_dir, hidden_dim=args.hidden_dim)
                else:
                    raise NotImplementedError

                if args.vocab_file.strip() != '':
                    # Resize the tokenizer
                    model.encoder.resize_token_embeddings(len(tokenizer))
                    logger.info(f"* Resized Token Embeddings: {len(tokenizer)}")
                    # print(f"\n\n (Expanded) Model Embedding Shape: {model.encoder.bert.embeddings.word_embeddings.weight.shape}\n\n")

                    raw_vocab_size = raw_tokenizer.vocab_size
                    vocab = tokenizer.get_vocab()
                    id2word = {v: k for k, v in vocab.items()}
                    embeddings = model.encoder.bert.embeddings.word_embeddings.weight

                    d2p = dict()
                    initial_embedding_id = len(raw_tokenizer)
                    original_vocab = raw_tokenizer.get_vocab()
                    domain_vocab = [tok for tok in tokenizer.get_vocab()][len(original_vocab):]

                    for embedding_id, tok in enumerate(domain_vocab):
                        tmp_tok = tok.replace("##", "")
                        values = raw_tokenizer.tokenize(tmp_tok)
                        d2p[tok] = (initial_embedding_id + embedding_id,
                                    values,
                                    raw_tokenizer.convert_tokens_to_ids(values)
                                    )
                    # logger.info("\n Save domain vocab to pretrained vocab mapper %s" % (vocab_path))


                    # d2p = pd.read_pickle(os.path.join('./vocab', args.dataset_name, 'd2p.pickle'))
                    embedding(args, model, d2p)

                model = model.to(device)

                no_decay = ["bias", "LayerNorm", "layernorm", "layer_norm", "ln"]
                model_grouped_parameters = [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)]
                total_trainable_params = sum(p.numel() for p in model_grouped_parameters if p.requires_grad)
                logger.info(f"* Model parameters: {total_trainable_params // 1000000}M")

                optimizer_grouped_parameters = [
                    {'params': [p for n, p in model.named_parameters() if not 'bert' in n], 'lr': args.non_lm_lr},
                    {'params': [p for n, p in model.named_parameters() if 'bert' in n], 'lr': lr}
                ]
                optimizer_kwargs = {
                    'betas': (0.9, 0.999),
                    'eps': 1e-8,
                }
                optimizer = AdamW(optimizer_grouped_parameters, **optimizer_kwargs)

                if args.contrastive:
                    if args.contrastive_type == 'sentence':
                        trainer = ContrastiveTrainer(
                            temperature=temperature,
                            model=model,
                            args=training_args,
                            train_dataset=train_dataset,
                            eval_dataset=val_dataset,
                            compute_metrics=compute_metrics,
                            tokenizer=tokenizer,
                            optimizers=(optimizer, None),
                        )
                    elif args.contrastive_type == 'word':
                        trainer = WordContrastiveTrainer(
                            temperature=temperature,
                            model=model,
                            args=training_args,
                            train_dataset=train_dataset,
                            eval_dataset=val_dataset,
                            compute_metrics=compute_metrics,
                            tokenizer=tokenizer,
                            optimizers=(optimizer, None),
                        )
                    elif args.contrastive_type == 'both':
                        trainer = BothContrastiveTrainer(
                            temperature=temperature,
                            model=model,
                            args=training_args,
                            train_dataset=train_dataset,
                            eval_dataset=val_dataset,
                            compute_metrics=compute_metrics,
                            tokenizer=tokenizer,
                            optimizers=(optimizer, None),
                        )
                else:
                    trainer = Trainer(
                        model=model,
                        args=training_args,
                        train_dataset=train_dataset,
                        eval_dataset=val_dataset,
                        compute_metrics=compute_metrics,
                        tokenizer=tokenizer,
                        optimizers=(optimizer, None),
                    )

                train_result = trainer.train()
                #print(train_result)
                logger.info(train_result)

                val_result = trainer.evaluate()
                #print(val_result)
                logger.info('[ Validation Result ]')
                logger.info({key: round(val_result[key], 4) for key in val_result})
                val_acc.append(val_result['eval_' + metric_for_best_model])
                val_oth.append(val_result['eval_' + other_metric])
                for key in val_result:
                    if key in dump_eval_keys:
                        continue

                    if key not in val_all:
                        val_all[key] = [val_result[key]]
                    else:
                        val_all[key].append(val_result[key])

                test_result = trainer.evaluate(test_dataset)
                #print(test_result)
                logger.info('[ Test Result ]')
                logger.info({key: round(test_result[key], 4) for key in test_result})
                test_acc.append(test_result['eval_' + metric_for_best_model])
                test_oth.append(test_result['eval_' + other_metric])
                for key in test_result:
                    if key in dump_eval_keys:
                        continue

                    if key not in test_all:
                        test_all[key] = [test_result[key]]
                    else:
                        test_all[key].append(test_result[key])

                if args.vocab_file.strip() != '':
                    config.vocab_size = original_vocab_size

                logger.info("")
                elapsed_time = time.time() - start_time
                elapsed_min = int(elapsed_time // 60)
                elapsed_sec = int(elapsed_time % 60)
                logger.info(f"Elapsed Time: {elapsed_min}m {elapsed_sec}s")
                logger.info("\n")

            if np.mean(val_acc) + np.mean(test_acc) > best_val + best_test:
                best_val = np.mean(val_acc)
                best_test = np.mean(test_acc)
                best_val_all = {key: np.mean(val_all[key]) for key in val_all}
                best_test_all = {key: np.mean(test_all[key]) for key in test_all}
                best_val_oth = np.mean(val_oth)
                best_test_oth = np.mean(test_oth)
                best_lr = lr
                best_temp = temperature
                best_val_acc_list = val_acc
                best_test_acc_list = test_acc
                best_val_oth_list = val_oth
                best_test_oth_list = test_oth
                best_val_all_list = val_all
                best_test_all_list = test_all

    logger.info(f'')
    logger.info(f'* {args.model_name} / {args.dataset_name} / {args.architecture}')
    if args.contrastive:
        logger.info(f'* Contrastive: True')
        logger.info(f'* Contrastive Type: {args.contrastive_type}')
        logger.info(f'* temperature: {args.temperature}')
    else:
        logger.info(f'* Contrastive: False')
    logger.info(f'* cur_fold: {fold}/{args.total_fold_num}')
    logger.info(f'* best_lr: {best_lr}')
    logger.info(f'* best_temperature: {best_temp}')
    logger.info(f'* best_val {metric_for_best_model}: {best_val * 100:.2f} [%]')
    logger.info(f'* best_test {metric_for_best_model}: {best_test * 100:.2f} [%]')
    if len(best_val_acc_list) == 1:
        pass
    else:
        logger.info(f'* best_val {metric_for_best_model} list: {[round(i * 100, 2) for i in best_val_acc_list]} [%]')
        logger.info(f'* best_test {metric_for_best_model} list: {[round(i * 100, 2) for i in best_test_acc_list]} [%]')

    logger.info(f'')
    logger.info(f'* best_val_all:')
    for key in best_val_all:
        logger.info(f'  ㄴ %-25s: %5s [%%]' % (key, round(best_val_all[key], 2)))
    logger.info(f'')
    logger.info(f'* best_test_all:')
    for key in best_test_all:
        logger.info(f'  ㄴ %-25s: %5s [%%]' % (key, round(best_test_all[key], 2)))

    fold_best_lr.append(best_lr)
    fold_best_temp.append(best_temp)
    fold_best_val.append(best_val)
    fold_best_test.append(best_test)
    fold_best_val_list.append(best_val_acc_list)
    fold_best_test_list.append(best_test_acc_list)
    if len(fold_best_val_all) == 0:
        for key in best_val_all:
            fold_best_val_all[key] = [best_val_all[key]]
            fold_best_test_all[key] = [best_test_all[key]]
    else:
        for key in best_val_all:
            fold_best_val_all[key].append(best_val_all[key])
            fold_best_test_all[key].append(best_test_all[key])


logger.info(f'\n')
logger.info(f'#############################################')
logger.info(f'################   Summary   ################')
logger.info(f'#############################################')
logger.info(f'* {args.model_name} / {args.dataset_name} / {args.architecture}')
if args.contrastive:
    logger.info(f'* Contrastive: True')
    logger.info(f'* Contrastive Type: {args.contrastive_type}')
else:
    logger.info(f'* Contrastive: False')

logger.info(f'* best_lr per fold: {fold_best_lr}')                                               # lr
logger.info(f'* best_temperature per fold: {fold_best_temp}')                                    # temperature
logger.info(f'* best_val {metric_for_best_model}: {round(np.mean(fold_best_val)*100, 2)}')       # score
logger.info(f'* best_test {metric_for_best_model}: {round(np.mean(fold_best_test)*100, 2)}')     # score
logger.info(f'* best_val {metric_for_best_model} list per fold: {[round(score*100, 2) for score in fold_best_val]}')       # [score1, score2, ...]
logger.info(f'* best_test {metric_for_best_model} list per fold: {[round(score*100, 2) for score in fold_best_test]}')     # [score1, score2, ...')
logger.info(f'')
logger.info(f'* best_val_all:')
for key in fold_best_val_all:
    logger.info(f'  ㄴ %-25s: %5s [%%]' % (key, round(np.mean(fold_best_val_all[key]), 2)))
logger.info(f'')
logger.info(f'* best_test_all:')
for key in fold_best_test_all:
    logger.info(f'  ㄴ %-25s: %5s [%%]' % (key, round(np.mean(fold_best_test_all[key]), 2)))
