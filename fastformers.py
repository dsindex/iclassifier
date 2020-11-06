from __future__ import absolute_import, division, print_function

import sys
import os
import argparse
import time
import pdb
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import random
import json
from tqdm import tqdm

from util    import load_config, to_device, to_numpy
from early_stopping import EarlyStopping
from datasets.metric import temp_seed 
from sklearn.metrics import classification_report, confusion_matrix

from train import train_epoch, evaluate, save_model, set_path, prepare_datasets, prepare_model, prepare_osws

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
streamHandler = logging.StreamHandler()
fileHandler = logging.FileHandler('./train.log')
#logger.addHandler(streamHandler)
logger.addHandler(fileHandler)

# ------------------------------------------------------------------------------ #
# base code from https://github.com/microsoft/fastformers#distilling-models
# ------------------------------------------------------------------------------ #
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.nn import MSELoss, CosineSimilarity

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

def evaluate_during_distill(args, model, eval_loader, labels, use_tqdm=True):
    logger.info(f"***** Running evaluation *****")
    logger.info("  Num batches = %d", len(eval_loader))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    eval_loader = tqdm(eval_loader, desc="Evaluating") if use_tqdm else eval_loader
    for x, y in eval_loader:
        x = to_device(x, args.device)
        y = to_device(y, args.device)

        model.eval()
        inputs = {"input_ids": x[0], "attention_mask": x[1], "labels": y}
        if model.config.model_type != "distilbert":
            # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
            inputs["token_type_ids"] = x[2] if model.config.model_type in ["bert", "xlnet", "albert"] else None

        with torch.no_grad():
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]
            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs["labels"].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    preds = np.argmax(preds, axis=1)
    try:
        label_names = [v for k, v in sorted(labels.items(), key=lambda x: x[0])] 
        print(classification_report(out_label_ids, preds, target_names=label_names, digits=4)) 
        print(labels)
        print(confusion_matrix(out_label_ids, preds))
    except Exception as e:
        logger.warn(str(e))
    return eval_loss

def distill(args, teacher_model, student_model, train_loader, eval_loader, labels, student_tokenizer):

    # number of training steps = number of epochs * number of batches
    num_training_steps_for_epoch = len(train_loader) // args.gradient_accumulation_steps
    num_training_steps = num_training_steps_for_epoch * args.epoch
    num_warmup_steps = int(args.warmup_ratio * num_training_steps)
    logger.info("(num_training_steps_for_epoch, num_training_steps, num_warmup_steps): ({}, {}, {})".\
        format(num_training_steps_for_epoch, num_training_steps, num_warmup_steps))        

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in student_model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in student_model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
    )

    # layer numbers of teacher and student
    teacher_layer_num = teacher_model.config.num_hidden_layers
    student_layer_num = student_model.config.num_hidden_layers

    # Prepare loss functions
    loss_mse = MSELoss()
    loss_cs = CosineSimilarity(dim=2)
    loss_cs_att = CosineSimilarity(dim=3)

    def soft_cross_entropy(predicts, targets):
        student_likelihood = F.log_softmax(predicts, dim=-1)
        targets_prob = F.softmax(targets, dim=-1)
        return (- targets_prob * student_likelihood).sum(dim=-1).mean()

    logger.info("***** Running distillation training *****")
    logger.info("  Num Batchs = %d", len(train_loader))
    logger.info("  Num Epochs = %d", args.epoch)
    logger.info("  batch size = %d", args.batch_size)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", num_training_steps)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    tr_loss, logging_loss = 0.0, 0.0
    tr_att_loss = 0.
    tr_rep_loss = 0.
    tr_cls_loss = 0.
    student_model.zero_grad()
    train_iterator = range(epochs_trained, int(args.epoch))

    # Added here for reproductibility
    set_seed(args)

    best_val_metric = None
    for epoch_n in train_iterator:
        tr_att_loss = 0.
        tr_rep_loss = 0.
        tr_cls_loss = 0.
        epoch_iterator = tqdm(train_loader, desc=f"Epoch {epoch_n}")
        for step, (x, y) in enumerate(epoch_iterator):
            x = to_device(x, args.device)
            y = to_device(y, args.device)

            att_loss = 0.
            rep_loss = 0.
            cls_loss = 0.

            inputs_teacher = {"input_ids": x[0], "attention_mask": x[1], "labels": y}
            inputs_student = {"input_ids": x[0], "attention_mask": x[1], "labels": y}
            if teacher_model.config.model_type != "distilbert":
                # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
                inputs_teacher["token_type_ids"] = x[2] if teacher_model.config.model_type in ["bert", "xlnet", "albert"] else None
            if student_model.config.model_type != "distilbert":
                # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
                inputs_student["token_type_ids"] = x[2] if student_model.config.model_type in ["bert", "xlnet", "albert"] else None

            # student model output
            student_model.train()
            outputs_student = student_model(output_attentions=True, output_hidden_states=True, **inputs_student)

            # teacher model output
            teacher_model.eval() # set teacher as eval mode
            with torch.no_grad():
                outputs_teacher = teacher_model(output_attentions=True, output_hidden_states=True, **inputs_teacher)
           
            # Knowledge Distillation loss
            # 1) logits distillation
            kd_loss = soft_cross_entropy(outputs_student[1], outputs_teacher[1])
            loss = kd_loss
            tr_cls_loss += loss.item()

            # 2) embedding and last hidden state distillation
            if args.state_loss_ratio > 0.0:
                teacher_reps = outputs_teacher[2]
                student_reps = outputs_student[2]

                new_teacher_reps = [teacher_reps[0], teacher_reps[teacher_layer_num]]
                new_student_reps = [student_reps[0], student_reps[student_layer_num]]
                for student_rep, teacher_rep in zip(new_student_reps, new_teacher_reps):
                    # cosine similarity loss
                    if args.state_distill_cs:
                        tmp_loss = 1.0 - loss_cs(student_rep, teacher_rep).mean()
                    # MSE loss
                    else:
                        tmp_loss = loss_mse(student_rep, teacher_rep)
                    rep_loss += tmp_loss
                loss += args.state_loss_ratio * rep_loss
                tr_rep_loss += rep_loss.item()

            # 3) Attentions distillation
            if args.att_loss_ratio > 0.0:
                teacher_atts = outputs_teacher[3]
                student_atts = outputs_student[3]

                assert teacher_layer_num == len(teacher_atts)
                assert student_layer_num == len(student_atts)
                assert teacher_layer_num % student_layer_num == 0
                layers_per_block = int(teacher_layer_num / student_layer_num)
                new_teacher_atts = [teacher_atts[i * layers_per_block + layers_per_block - 1]
                                    for i in range(student_layer_num)]

                for student_att, teacher_att in zip(student_atts, new_teacher_atts):
                    student_att = torch.where(student_att <= -1e2, torch.zeros_like(student_att).to(args.device),
                                              student_att)
                    teacher_att = torch.where(teacher_att <= -1e2, torch.zeros_like(teacher_att).to(args.device),
                                              teacher_att)
                    tmp_loss = 1.0 - loss_cs_att(student_att, teacher_att).mean()
                    att_loss += tmp_loss

                loss += args.att_loss_ratio * att_loss
                tr_att_loss += att_loss.item()

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            # back propagate
            loss.backward()

            tr_loss += loss.item()
            epoch_iterator.set_description(f"Epoch {epoch_n} loss: {loss:.3f}")
            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(student_model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                student_model.zero_grad()
                global_step += 1

                # change to evaluation mode
                student_model.eval()
                logs = {}

                flag_eval = False
                if args.logging_steps > 0 and global_step % args.logging_steps == 0: flag_eval = True
                if flag_eval:
                    if args.log_evaluate_during_training:
                        eval_loss = evaluate_during_distill(args, student_model, eval_loader, labels, use_tqdm=False)
                        logs['eval_loss'] = eval_loss
                    
                    cls_loss = tr_cls_loss / (step + 1)
                    att_loss = tr_att_loss / (step + 1)
                    rep_loss = tr_rep_loss / (step + 1)

                    loss_scalar = (tr_loss - logging_loss) / args.logging_steps
                    learning_rate_scalar = scheduler.get_last_lr()[0]
                    logs["learning_rate"] = learning_rate_scalar
                    logs["avg_loss_since_last_log"] = loss_scalar
                    logs['cls_loss'] = cls_loss
                    logs['att_loss'] = att_loss
                    logs['rep_loss'] = rep_loss
                    logging_loss = tr_loss
                    logging.info(json.dumps({**logs, **{"step": global_step}}))

                flag_eval = False
                if step == 0 and epoch_n != 0: flag_eval = True # every epoch
                if args.eval_and_save_steps > 0 and global_step % args.eval_and_save_steps == 0: flag_eval = True
                if flag_eval:
                    eval_loss = evaluate_during_distill(args, student_model, eval_loader, labels, use_tqdm=False)
                    logs['eval_loss'] = eval_loss
                    logger.info(json.dumps({**logs, **{"step": global_step}}))
                    curr_val_metric = eval_loss
                    if best_val_metric is None or curr_val_metric < best_val_metric:
                        # save
                        student_model.save_pretrained(args.bert_output_dir)
                        student_tokenizer.save_pretrained(args.bert_output_dir)
                        best_val_metric = curr_val_metric
                        logger.info("Saved best student model to %s", args.bert_output_dir)
                
                # change student model back to train mode
                student_model.train()

    return global_step, tr_loss / global_step

def load_label(input_path):
    labels = {}
    with open(input_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            toks = line.strip().split()
            label = toks[0]
            label_id = int(toks[1])
            labels[label_id] = label
    return labels

def prepare_bert_model(opt, model_name_or_path, num_labels, do_lower_case=False):
    from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification

    bert_tokenizer = AutoTokenizer.from_pretrained(model_name_or_path,
                                                   do_lower_case=do_lower_case)
    bert_model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path,
                                                                    num_labels=num_labels,
                                                                    from_tf=bool(".ckpt" in model_name_or_path))
    bert_model.to(opt.device)
    print(bert_model)
    logger.info("[bert model prepared]")
    return bert_model, bert_tokenizer

def train(opt):
    if torch.cuda.is_available():
        logger.info("%s", torch.cuda.get_device_name(0))

    # set etc
    torch.autograd.set_detect_anomaly(True)

    # set config
    config = load_config(opt, config_path=opt.config)
    config['opt'] = opt
    logger.info("%s", config)
    
    # set path
    set_path(config)
  
    # prepare train, valid dataset
    train_loader, valid_loader = prepare_datasets(config)

    # ------------------------------------------------------------------------------------------------------- #
    # distillation
    labels = load_label(opt.label_path)
    num_labels = len(labels)

    teacher_model, teacher_tokenizer = prepare_bert_model(opt,
            opt.teacher_bert_model_name_or_path,
            num_labels,
            do_lower_case=opt.bert_do_lower_case)

    student_model, student_tokenizer = prepare_bert_model(opt,
            opt.bert_model_name_or_path,
            num_labels,
            do_lower_case=opt.bert_do_lower_case)

    global_step, tr_loss = distill(opt, teacher_model, student_model, train_loader, valid_loader, labels, student_tokenizer)
    logger.info(f"distillation done: {global_step}, {tr_loss}")
    # ------------------------------------------------------------------------------------------------------- #


    sys.exit(0)


    with temp_seed(opt.seed):
        # prepare model
        model = prepare_model(config)

        # create optimizer, scheduler, summary writer, scaler
        optimizer, scheduler, writer, scaler = prepare_osws(config, model, train_loader)
        config['optimizer'] = optimizer
        config['scheduler'] = scheduler
        config['writer'] = writer
        config['scaler'] = scaler

        # training
        early_stopping = EarlyStopping(logger, patience=opt.patience, measure=opt.measure, verbose=1)
        local_worse_steps = 0
        prev_eval_measure = float('inf') if opt.measure == 'loss' else -float('inf')
        best_eval_measure = float('inf') if opt.measure == 'loss' else -float('inf')
        for epoch_i in range(opt.epoch):
            epoch_st_time = time.time()
            eval_loss, eval_acc = train_epoch(model, config, train_loader, valid_loader, epoch_i)
            if opt.measure == 'loss': eval_measure = eval_loss 
            else: eval_measure = eval_acc
            # early stopping
            if early_stopping.validate(eval_measure, measure=opt.measure): break
            if opt.measure == 'loss': is_best = eval_measure < best_eval_measure
            else: is_best = eval_measure > best_eval_measure
            if is_best:
                best_eval_measure = eval_measure
                if opt.save_path:
                    logger.info("[Best model saved] : {:10.6f}".format(best_eval_measure))
                    save_model(config, model)
                    # save finetuned bert model/config/tokenizer
                    if config['emb_class'] not in ['glove']:
                        if not os.path.exists(opt.bert_output_dir):
                            os.makedirs(opt.bert_output_dir)
                        model.bert_tokenizer.save_pretrained(opt.bert_output_dir)
                        model.bert_model.save_pretrained(opt.bert_output_dir)
                early_stopping.reset(best_eval_measure)
            early_stopping.status()
            # begin: scheduling, apply rate decay at the measure(ex, loss) getting worse for the number of deacy epoch steps.
            if opt.measure == 'loss': getting_worse = prev_eval_measure <= eval_measure
            else: getting_worse = prev_eval_measure >= eval_measure
            if getting_worse:
                local_worse_steps += 1
            else:
                local_worse_steps = 0
            logger.info('Scheduler: local_worse_steps / opt.lr_decay_steps = %d / %d' % (local_worse_steps, opt.lr_decay_steps))
            if not opt.use_transformers_optimizer and \
               epoch_i > opt.warmup_epoch and \
               (local_worse_steps >= opt.lr_decay_steps or early_stopping.step() > opt.lr_decay_steps):
                scheduler.step()
                local_worse_steps = 0
            prev_eval_measure = eval_measure
            # end: scheduling

def get_params():
    parser = argparse.ArgumentParser()

    # For distill
    parser.add_argument('--teacher_bert_model_name_or_path', type=str, default=None,
                        help="Path to pre-trained model or shortcut name(ex, bert-base-uncased)")
    parser.add_argument('--state_distill_cs', action="store_true", help="If this is using Cosine similarity for the hidden and embedding state distillation. vs. MSE")
    parser.add_argument('--state_loss_ratio', type=float, default=0.1)
    parser.add_argument('--att_loss_ratio', type=float, default=0.0)
    parser.add_argument('--warmup_ratio', default=0, type=float, help="Linear warmup over warmup_steps as a float.")
    parser.add_argument('--logging_steps', type=int, default=500, help="Log every X updates steps.")
    parser.add_argument('--eval_and_save_steps', type=int, default=500, help="Save checkpoint every X updates steps.")
    parser.add_argument('--log_evaluate_during_training', action="store_true", help="Run evaluation during training at each logging step.")
   
    # Same Aguments with train.py
    parser.add_argument('--config', type=str, default='configs/config-distilbert-cls.json')
    parser.add_argument('--data_dir', type=str, default='data/snips')
    parser.add_argument('--embedding_filename', type=str, default='embedding.npy')
    parser.add_argument('--label_filename', type=str, default='label.txt')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--eval_batch_size', type=int, default=128)
    parser.add_argument('--epoch', type=int, default=64)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--lr_decay_rate', type=float, default=1.0, help="Disjoint with --use_transformers_optimizer")
    parser.add_argument('--lr_decay_steps', type=float, default=2, help="Number of decay epoch steps to be paitent. disjoint with --use_transformers_optimizer")
    parser.add_argument('--warmup_epoch', type=int, default=4,  help="Number of warmup epoch steps")
    parser.add_argument('--patience', default=7, type=int, help="Max number of epoch to be patient for early stopping.")
    parser.add_argument('--save_path', type=str, default='pytorch-model.pt')
    parser.add_argument('--adam_epsilon', type=float, default=1e-8)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--max_grad_norm', default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument('--log_dir', type=str, default='runs')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--embedding_trainable', action='store_true', help="Set word embedding(Glove) trainable")
    parser.add_argument('--use_transformers_optimizer', action='store_true', help="Use transformers AdamW, get_linear_schedule_with_warmup.")
    parser.add_argument('--use_amp', action='store_true', help="Use automatic mixed precision.")
    parser.add_argument('--use_profiler', action='store_true', help="Use profiler.")
    parser.add_argument('--measure', type=str, default='loss', help="Evaluation measure, 'loss' | 'accuracy', default 'loss'.")
    parser.add_argument('--criterion', type=str, default='CrossEntropyLoss', help="training objective, 'CrossEntropyLoss' | 'MSELoss' | 'KLDivLoss', default 'CrossEntropyLoss'")
    parser.add_argument('--augmented', action='store_true',
                        help="Set this flag to use augmented.txt for training.")
    parser.add_argument('--bert_model_name_or_path', type=str, default='embeddings/distilbert-base-uncased',
                        help="Path to pre-trained model or shortcut name(ex, distilbert-base-uncased)")
    parser.add_argument('--bert_do_lower_case', action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument('--bert_output_dir', type=str, default='bert-checkpoint',
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument('--bert_use_feature_based', action='store_true',
                        help="Use BERT as feature-based, default fine-tuning")
    parser.add_argument('--bert_remove_layers', type=str, default='',
                        help="Specify layer numbers to remove during finetuning e.g. 8,9,10,11 to remove last 4 layers from BERT base(12 layers)")

    opt = parser.parse_args()
    return opt

def main():
    opt = get_params()

    train(opt)
   
if __name__ == '__main__':
    main()
