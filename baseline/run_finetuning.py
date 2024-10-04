# CUDA_VISIBLE_DEVICES=2 python3 run_finetuning.py --model_name matscibert --dataset_name matscholar
# CUDA_VISIBLE_DEVICES=2 python3 run_finetuning.py --model_name scibert --dataset_name sofc --lm_lrs 1e-5 2e-5 3e-5 --seeds 42 123 456 --non_lm_lr 1e-4 --model_save_dir ./custom_results/models --log_dir ./custom_results/logs
# CUDA_VISIBLE_DEVICES=2 python3 run_finetuning.py --model_name matscibert --dataset_name matscholar --pretrained_model_path /data/user5/workspace/Melectra/pretraining/continual_pretrained_model/matscibert_matscholar_untied_False/epoch_20 




import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
import sys
from pathlib import Path

from argparse import ArgumentParser
import numpy as np
from collections import defaultdict

import torch
import time

import ner_datasets
from models import BERT_CRF
import conlleval
from logger import get_logger

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

def ensure_dir(dir_path):
    Path(dir_path).mkdir(parents=True, exist_ok=True)
    return dir_path

# 토큰화 및 인코딩 함수
def tokenize_and_encode(texts, tokenizer, max_length=512):
    return tokenizer(texts, is_split_into_words=True, return_offsets_mapping=True, 
                     padding=True, truncation=True, max_length=max_length)

# 레이블 인코딩 함수
# def encode_tags(tags, encodings, tag2id):
#     encoded_labels = []
#     for doc_tags, doc_offset in zip(tags, encodings.offset_mapping):
#         doc_enc_labels = np.ones(len(doc_offset), dtype=int) * -100
#         arr_offset = np.array(doc_offset)
#         doc_enc_labels[(arr_offset[:, 0] == 0) & (arr_offset[:, 1] != 0)] = [tag2id[tag] for tag in doc_tags]
#         encoded_labels.append(doc_enc_labels.tolist())
#     return encoded_labels
def encode_tags(tags, encodings, tag2id):
    labels = [[tag2id[tag] for tag in doc] for doc in tags]
    encoded_labels = []
    for doc_labels, doc_offset in zip(labels, encodings.offset_mapping):
        doc_enc_labels = np.ones(len(doc_offset), dtype=int) * -100
        arr_offset = np.array(doc_offset)
        doc_enc_labels[(arr_offset[:, 0] == 0) & (arr_offset[:, 1] != 0)] = doc_labels
        encoded_labels.append(doc_enc_labels.tolist())
    return encoded_labels

# 0길이 토큰 제거 함수
def remove_zero_len_tokens(X, y, tokenizer):
    new_X, new_y = [], []
    for sent, labels in zip(tqdm(X), y):
        new_sent, new_labels = [], []
        for token, label in zip(sent, labels):
            if len(tokenizer.tokenize(token)) == 0:
                continue
            new_sent.append(token)
            new_labels.append(label)
        assert len(new_sent) == len(new_labels)
        new_X.append(new_sent)
        new_y.append(new_labels)
    return new_X, new_y

# NER 데이터셋 클래스
class NERDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# 메인 데이터 처리 함수
def process_data(args, tokenizer):
    # 데이터 로드
    train_X, train_y, val_X, val_y, test_X, test_y = ner_datasets.get_ner_data(args.dataset_name, fold=args.total_fold_num)

    # 유니크 레이블 추출 및 인덱싱
    unique_labels = sorted(set(label for sent in train_y for label in sent))
    tag2id = {tag: idx for idx, tag in enumerate(unique_labels)}
    id2tag = {idx: tag for tag, idx in tag2id.items()}

    # 0길이 토큰 제거
    train_X, train_y = remove_zero_len_tokens(train_X, train_y, tokenizer)
    val_X, val_y = remove_zero_len_tokens(val_X, val_y, tokenizer)
    test_X, test_y = remove_zero_len_tokens(test_X, test_y, tokenizer)

    # 토큰화 및 인코딩
    train_encodings = tokenize_and_encode(train_X, tokenizer)
    val_encodings = tokenize_and_encode(val_X, tokenizer)
    test_encodings = tokenize_and_encode(test_X, tokenizer)

    # 레이블 인코딩
    train_labels = encode_tags(train_y, train_encodings, tag2id)
    val_labels = encode_tags(val_y, val_encodings, tag2id)
    test_labels = encode_tags(test_y, test_encodings, tag2id)

    train_encodings.pop('offset_mapping')
    val_encodings.pop('offset_mapping')
    test_encodings.pop('offset_mapping')

    # 데이터셋 생성
    train_dataset = NERDataset(train_encodings, train_labels)
    val_dataset = NERDataset(val_encodings, val_labels)
    test_dataset = NERDataset(test_encodings, test_labels)

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

    return train_dataset, val_dataset, test_dataset, id2tag, len(unique_labels), eval_labels





default_results_dir = os.path.join(os.getcwd(), 'results')

parser = ArgumentParser()
parser.add_argument('--model_name', type=str, required=True)
# parser.add_argument('--model_save_dir', type=str, required=True)
# parser.add_argument('--preds_save_dir', type=str, required=True)
# parser.add_argument('--log_dir', type=str, required=True)
# parser.add_argument('--cache_dir', type=str, required=True)
parser.add_argument('--model_save_dir', type=str, default=os.path.join(default_results_dir, 'model_save'), help='Directory to save the model')
parser.add_argument('--preds_save_dir', type=str, default=os.path.join(default_results_dir, 'preds_save'), help='Directory to save predictions')
parser.add_argument('--log_dir', type=str, default=os.path.join(default_results_dir, 'logs'), help='Directory to save logs')
parser.add_argument('--cache_dir', type=str, default=os.path.join(default_results_dir, 'cache'), help='Directory for caching')
parser.add_argument('--seeds', nargs='+', default=[1997, 2024, 2017], type=int)
parser.add_argument('--lm_lrs', nargs='+', default=[2e-5, 3e-5, 5e-5], type=float)
parser.add_argument('--non_lm_lr', default=3e-4, type=float)
parser.add_argument('--architecture', choices=['bert', 'bert-crf'], type=str, default="bert-crf")
parser.add_argument('--dataset_name', choices=['sofc', 'sofc_slot', 'matscholar'], type=str, required=True)
parser.add_argument('--total_fold_num', default=5, type=int)
parser.add_argument('--pretrained_model_path', type=str, default=None, help='Path to pre-trained model. If not provided, uses the default model.')
args = parser.parse_args()

# 디렉토리 생성 함수
def create_directory(directory):
    Path(directory).mkdir(parents=True, exist_ok=True)
create_directory(args.model_save_dir)
create_directory(args.preds_save_dir)
create_directory(args.log_dir)
create_directory(args.cache_dir)

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

if args.dataset_name == 'sofc':
    num_epochs = 20
elif args.dataset_name == 'sofc_slot':
    num_epochs = 40
elif args.dataset_name == 'matscholar':
    num_epochs = 15
    args.total_fold_num = 1
else:
    raise NotImplementedError

logger = get_logger(log_path=os.path.join(args.log_dir, f"train_log_{args.dataset_name}_{args.model_name}.txt"))

metric_for_best_model = 'macro_f1' if args.dataset_name[: 4] == 'sofc' or args.dataset_name == 'matscholar' else 'micro_f1'
other_metric = 'micro_f1' if metric_for_best_model == 'macro_f1' else 'macro_f1'
logger.info(f"metric_for_best_model: {metric_for_best_model}")
logger.info(f"other_metric: {other_metric}")

logger.info("Start of the script")
logger.info(f"Arguments: {args}")
logger.info(f'Using device: {device}')


total_combinations = len(args.lm_lrs) * len(args.seeds) * args.total_fold_num
current_combination = 0

results = defaultdict(lambda: defaultdict(dict))

for fold in range(1, args.total_fold_num + 1):
    logger.info(f"Start Training for fold {fold}")

    cache_dir = ensure_dir(args.cache_dir)
    output_dir = ensure_dir(args.model_save_dir)
    preds_save_dir = ensure_dir(args.preds_save_dir)

    tokenizer_kwargs = {
        'cache_dir': cache_dir,
        'use_fast': True,
        'revision': 'main',
        'use_auth_token': None,
        'model_max_length': 512
    }
    tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)

    train_dataset, val_dataset, test_dataset, id2tag, num_labels, eval_labels = process_data(args, tokenizer)

    logger.info(f"Data: {args.dataset_name}")
    logger.info(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")
    logger.info(f"Num_labels: {num_labels}")

    config_kwargs = {
        'num_labels': num_labels,
        'cache_dir': cache_dir,
        'revision': 'main',
        'use_auth_token': None,
    }
    config = AutoConfig.from_pretrained(model_name, **config_kwargs)


    model_path = args.pretrained_model_path if args.pretrained_model_path else model_name


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

    best_lr = 0
    best_val = 0
    best_test = 0
    best_val_all = {}
    best_test_all = {}

    for lr_idx, lr in enumerate(args.lm_lrs, 1):
        for seed_idx, seed in enumerate(args.seeds, 1):
            start_time = time.time()
            current_combination += 1
            logger.info(f"\n{'='*50}")
            logger.info(f"Combination {current_combination}/{total_combinations}")
            logger.info(f"Fold: {fold}/{args.total_fold_num} | "
                        f"LR: {lr} ({lr_idx}/{len(args.lm_lrs)}) | "
                        f"Seed: {seed} ({seed_idx}/{len(args.seeds)})")
            logger.info(f"{'='*50}\n")

            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.benchmark = False
            set_seed(seed)

            if args.architecture == 'bert':
                model = AutoModelForTokenClassification.from_pretrained(model_path, config=config)
            elif args.architecture == 'bert-crf':
                model = BERT_CRF(model_path, device, config, cache_dir)
            else:
                raise ValueError(f"Unsupported architecture: {args.architecture}")
            
            model = model.to(device)

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
                seed=seed,
            )

            optimizer_grouped_parameters = [
                {'params': [p for n, p in model.named_parameters() if not 'bert' in n], 'lr': args.non_lm_lr},
                {'params': [p for n, p in model.named_parameters() if 'bert' in n], 'lr': lr}
            ]
            optimizer = AdamW(optimizer_grouped_parameters, betas=(0.9, 0.999), eps=1e-8)

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
            logger.info(train_result)

            val_result = trainer.evaluate()
            logger.info('[ Validation Result ]')
            logger.info({key: round(val_result[key], 4) for key in val_result})

            test_result = trainer.evaluate(test_dataset)
            logger.info('[ Test Result ]')
            logger.info({key: round(test_result[key], 4) for key in test_result})

            results[lr][seed][fold] = {
                'val': val_result,
                'test': test_result
            }

            elapsed_time = time.time() - start_time
            logger.info(f"Elapsed Time: {int(elapsed_time // 60)}m {int(elapsed_time % 60)}s")
            logger.info(f"{'='*50}\n")



# 결과 분석 및 로깅
def analyze_results(results, metric_for_best_model):
    avg_results = defaultdict(lambda: defaultdict(dict))
    best_lr = None
    best_seed = None
    best_avg_val = -float('inf')

    for lr in results:
        for seed in results[lr]:
            # 각 (lr, seed) 조합에 대해 모든 fold의 평균 계산
            val_scores = [results[lr][seed][fold]['val'][f'eval_{metric_for_best_model}'] for fold in results[lr][seed]]
            test_scores = [results[lr][seed][fold]['test'][f'eval_{metric_for_best_model}'] for fold in results[lr][seed]]
            
            avg_val_score = np.mean(val_scores)
            avg_test_score = np.mean(test_scores)
            
            avg_results[lr][seed] = {
                'avg_val': avg_val_score,
                'avg_test': avg_test_score,
                'val_scores': val_scores,
                'test_scores': test_scores
            }

            # 최고 성능의 (lr, seed) 조합 찾기
            if avg_val_score > best_avg_val:
                best_avg_val = avg_val_score
                best_lr = lr
                best_seed = seed

    return avg_results, best_lr, best_seed

# 결과 분석
avg_results, best_lr, best_seed = analyze_results(results, metric_for_best_model)

# 결과 로깅
logger.info(f'\nOverall Summary:')
logger.info(f'Best LR: {best_lr}')
logger.info(f'Best Seed: {best_seed}')
logger.info(f'Best Avg Val {metric_for_best_model}: {avg_results[best_lr][best_seed]["avg_val"]*100:.2f}%')
logger.info(f'Corresponding Avg Test {metric_for_best_model}: {avg_results[best_lr][best_seed]["avg_test"]*100:.2f}%')
logger.info(f'Val {metric_for_best_model} per fold: {[round(score*100, 2) for score in avg_results[best_lr][best_seed]["val_scores"]]}%')
logger.info(f'Test {metric_for_best_model} per fold: {[round(score*100, 2) for score in avg_results[best_lr][best_seed]["test_scores"]]}%')

# 추가적인 메트릭에 대한 로깅 (필요시)
for metric in results[best_lr][best_seed][1]['val']:  # 첫 번째 fold의 메트릭을 기준으로
    if metric != f'eval_{metric_for_best_model}':
        avg_val = np.mean([results[best_lr][best_seed][fold]['val'][metric] for fold in results[best_lr][best_seed]])
        avg_test = np.mean([results[best_lr][best_seed][fold]['test'][metric] for fold in results[best_lr][best_seed]])
        logger.info(f'{metric}: Val: {avg_val:.4f}, Test: {avg_test:.4f}')