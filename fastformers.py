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
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.nn import MSELoss, CosineSimilarity

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
fileHandler = logging.FileHandler('./train.log')
logger.addHandler(fileHandler)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
# ------------------------------------------------------------------------------ #
# base code from https://github.com/microsoft/fastformers#distilling-models
#  - distill()
#  - prune_rewire()
#  - sort_by_importance()
# ------------------------------------------------------------------------------ #

def distill(
        teacher_config,
        teacher_model,
        student_config,
        student_model,
        train_loader,
        eval_loader):

    args = teacher_config['opt']

    teacher_layer_num = teacher_model.bert_model.config.num_hidden_layers
    student_layer_num = student_model.bert_model.config.num_hidden_layers

    num_training_steps_for_epoch = len(train_loader) // args.gradient_accumulation_steps
    num_training_steps = num_training_steps_for_epoch * args.epoch
    num_warmup_steps = int(args.warmup_ratio * num_training_steps)
    logger.info("(num_training_steps_for_epoch, num_training_steps, num_warmup_steps): ({}, {}, {})".\
        format(num_training_steps_for_epoch, num_training_steps, num_warmup_steps))        

    # prepare optimizer and schedule (linear warmup and decay)
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

    # prepare loss functions
    def soft_cross_entropy(predicts, targets):
        likelihood = F.log_softmax(predicts, dim=-1)
        targets_prob = F.softmax(targets, dim=-1)
        return (- targets_prob * likelihood).sum(dim=-1).mean()
    loss_mse_sum = MSELoss(reduction='sum').to(args.device)
    loss_mse = MSELoss().to(args.device)
    loss_cs = CosineSimilarity(dim=2).to(args.device)
    loss_cs_att = CosineSimilarity(dim=3).to(args.device)

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

    # for reproductibility
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

            # student model output
            student_model.train()
            output_student, student_bert_outputs = student_model(x, return_bert_outputs=True)

            # teacher model output
            teacher_model.eval() # set teacher as eval mode
            with torch.no_grad():
                output_teacher, teacher_bert_outputs = teacher_model(x, return_bert_outputs=True)
           
            # Knowledge Distillation loss
            # 1) logits distillation
            '''
            kd_loss = soft_cross_entropy(output_student, output_teacher)
            '''
            kd_loss = loss_mse_sum(output_student, output_teacher)

            loss = kd_loss
            tr_cls_loss += loss.item()

            # 2) embedding and last hidden state distillation
            if args.state_loss_ratio > 0.0:
                teacher_reps = teacher_bert_outputs.hidden_states
                student_reps = student_bert_outputs.hidden_states

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
                teacher_atts = teacher_bert_outputs.attentions
                student_atts = student_bert_outputs.attentions

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
                        eval_loss, eval_acc = evaluate(student_model, student_config, eval_loader)
                        logs['eval_loss'] = eval_loss
                        logs['eval_acc'] = eval_acc
                    
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
                    eval_loss, eval_acc = evaluate(student_model, student_config, eval_loader)
                    logs['eval_loss'] = eval_loss
                    logs['eval_acc'] = eval_acc
                    logger.info(json.dumps({**logs, **{"step": global_step}}))
                    # measured by accuracy
                    curr_val_metric = eval_acc
                    if best_val_metric is None or curr_val_metric > best_val_metric:
                        # save model to '--save_path', '--bert_output_dir'
                        save_model(student_config, student_model)
                        student_model.bert_tokenizer.save_pretrained(args.bert_output_dir)
                        student_model.bert_model.save_pretrained(args.bert_output_dir)
                        best_val_metric = curr_val_metric
                        logger.info("[Best student model saved] : {:10.6f}, {}".format(best_val_metric,args.bert_output_dir))
                
                # change student model back to train mode
                student_model.train()

    return global_step, tr_loss / global_step

def sort_by_importance(weight, bias, importance, num_instances, stride):
    from heapq import heappush, heappop
    importance_ordered = []
    i = 0
    for heads in importance:
        heappush(importance_ordered, (-heads, i))
        i += 1
    sorted_weight_to_concat = None
    sorted_bias_to_concat = None
    i = 0
    while importance_ordered and i < num_instances:
        head_to_add = heappop(importance_ordered)[1]
        if sorted_weight_to_concat is None:
            sorted_weight_to_concat = (weight.narrow(0, int(head_to_add * stride), int(stride)), )
        else:
            sorted_weight_to_concat += (weight.narrow(0, int(head_to_add * stride), int(stride)), )
        if bias is not None:
            if sorted_bias_to_concat is None:
                sorted_bias_to_concat = (bias.narrow(0, int(head_to_add * stride), int(stride)), )
            else:
                sorted_bias_to_concat += (bias.narrow(0, int(head_to_add * stride), int(stride)), )
        i += 1
    return torch.cat(sorted_weight_to_concat), torch.cat(sorted_bias_to_concat) if sorted_bias_to_concat is not None else None

def prune_rewire(config, model, eval_loader, use_tqdm=True):

    args = config['opt']
    bert_model = model.bert_model

    # get the model ffn weights and biases
    inter_weights = torch.zeros(bert_model.config.num_hidden_layers, bert_model.config.intermediate_size, bert_model.config.hidden_size).to(args.device)
    inter_biases = torch.zeros(bert_model.config.num_hidden_layers, bert_model.config.intermediate_size).to(args.device)
    output_weights = torch.zeros(bert_model.config.num_hidden_layers, bert_model.config.hidden_size, bert_model.config.intermediate_size).to(args.device)

    layers = bert_model.base_model.encoder.layer
    head_importance = torch.zeros(bert_model.config.num_hidden_layers, bert_model.config.num_attention_heads).to(args.device)
    ffn_importance = torch.zeros(bert_model.config.num_hidden_layers, bert_model.config.intermediate_size).to(args.device)

    for layer_num in range(bert_model.config.num_hidden_layers):
        inter_weights[layer_num] = layers._modules[str(layer_num)].intermediate.dense.weight.detach().to(args.device)
        inter_biases[layer_num] = layers._modules[str(layer_num)].intermediate.dense.bias.detach().to(args.device)
        output_weights[layer_num] = layers._modules[str(layer_num)].output.dense.weight.detach().to(args.device)

    head_mask = torch.ones(bert_model.config.num_hidden_layers, bert_model.config.num_attention_heads, requires_grad=True).to(args.device)

    # Eval!
    logger.info(f"***** Running evaluation for pruning *****")
    logger.info("  Num batches = %d", len(eval_loader))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    criterion = torch.nn.CrossEntropyLoss().to(args.device)

    eval_loader = tqdm(eval_loader, desc="Evaluating") if use_tqdm else eval_loader
    tot_tokens = 0.0
    for x, y in eval_loader:
        model.eval()
        x = to_device(x, args.device)
        y = to_device(y, args.device)
        
        logits, bert_outputs = model(x, return_bert_outputs=True, head_mask=head_mask)
        tmp_eval_loss = criterion(logits, y)

        eval_loss += tmp_eval_loss.mean().item()

        # for preventing head_mask.grad is None
        head_mask.retain_grad()

        # TODO accumulate? absolute value sum?
        tmp_eval_loss.backward()

        # collect attention confidence scores
        head_importance += head_mask.grad.abs().detach()

        # collect gradients of linear layers
        for layer_num in range(bert_model.config.num_hidden_layers):
            ffn_importance[layer_num] += torch.abs(
                torch.sum(layers._modules[str(layer_num)].intermediate.dense.weight.grad.detach()*inter_weights[layer_num], 1) 
                + layers._modules[str(layer_num)].intermediate.dense.bias.grad.detach()*inter_biases[layer_num])
 
        attention_mask = x[1]
        tot_tokens += attention_mask.float().detach().sum().data
        nb_eval_steps += 1

    head_importance /= tot_tokens

    # Layerwise importance normalization
    if not args.dont_normalize_importance_by_layer:
        exponent = 2
        norm_by_layer = torch.pow(torch.pow(head_importance, exponent).sum(-1), 1 / exponent)
        head_importance /= norm_by_layer.unsqueeze(-1) + 1e-20

    # rewire the network
    head_importance = head_importance.cpu()
    ffn_importance = ffn_importance.cpu()
    num_heads = bert_model.config.num_attention_heads
    head_size = bert_model.config.hidden_size / num_heads
    for layer_num in range(bert_model.config.num_hidden_layers):
        # load query, key, value weights
        query_weight = layers._modules[str(layer_num)].attention.self.query.weight
        query_bias = layers._modules[str(layer_num)].attention.self.query.bias
        key_weight = layers._modules[str(layer_num)].attention.self.key.weight
        key_bias = layers._modules[str(layer_num)].attention.self.key.bias
        value_weight = layers._modules[str(layer_num)].attention.self.value.weight
        value_bias = layers._modules[str(layer_num)].attention.self.value.bias

        # sort query, key, value based on the confidence scores
        query_weight, query_bias = sort_by_importance(query_weight,
            query_bias,
            head_importance[layer_num],
            args.target_num_heads,
            head_size)
        print('query_weight = ', query_weight.shape)
        print('query_bias = ', query_bias.shape)
        layers._modules[str(layer_num)].attention.self.query.weight = torch.nn.Parameter(query_weight)
        layers._modules[str(layer_num)].attention.self.query.bias = torch.nn.Parameter(query_bias)
        key_weight, key_bias = sort_by_importance(key_weight,
            key_bias,
            head_importance[layer_num],
            args.target_num_heads,
            head_size)
        print('key_weight = ', key_weight.shape)
        print('key_bias = ', key_bias.shape)
        layers._modules[str(layer_num)].attention.self.key.weight = torch.nn.Parameter(key_weight)
        layers._modules[str(layer_num)].attention.self.key.bias = torch.nn.Parameter(key_bias)
        value_weight, value_bias = sort_by_importance(value_weight,
            value_bias,
            head_importance[layer_num],
            args.target_num_heads,
            head_size)
        print('value_weight = ', value_weight.shape)
        print('value_bias = ', value_bias.shape)
        layers._modules[str(layer_num)].attention.self.value.weight = torch.nn.Parameter(value_weight)
        layers._modules[str(layer_num)].attention.self.value.bias = torch.nn.Parameter(value_bias)

        # output matrix
        weight_sorted, _ = sort_by_importance(
            layers._modules[str(layer_num)].attention.output.dense.weight.transpose(0, 1),
            None,
            head_importance[layer_num],
            args.target_num_heads,
            head_size)
        weight_sorted = weight_sorted.transpose(0, 1)
        print('attention.output.dense.weight = ', weight_sorted.shape)
        layers._modules[str(layer_num)].attention.output.dense.weight = torch.nn.Parameter(weight_sorted)

        weight_sorted, bias_sorted = sort_by_importance(
            layers._modules[str(layer_num)].intermediate.dense.weight,
            layers._modules[str(layer_num)].intermediate.dense.bias, 
            ffn_importance[layer_num],
            args.target_ffn_dim,
            1)
        layers._modules[str(layer_num)].intermediate.dense.weight = torch.nn.Parameter(weight_sorted)
        layers._modules[str(layer_num)].intermediate.dense.bias = torch.nn.Parameter(bias_sorted)

        # ffn output matrix input side
        weight_sorted, _ = sort_by_importance(
            layers._modules[str(layer_num)].output.dense.weight.transpose(0, 1),
            None, 
            ffn_importance[layer_num],
            args.target_ffn_dim,
            1)
        weight_sorted = weight_sorted.transpose(0, 1)
        print('output.dense.weight = ', weight_sorted.shape)
        layers._modules[str(layer_num)].output.dense.weight = torch.nn.Parameter(weight_sorted)

    # set bert model's config for pruned model
    bert_model.config.num_attention_heads = min([num_heads, args.target_num_heads])
    bert_model.config.intermediate_size = layers._modules['0'].intermediate.dense.weight.size(0)

def load_checkpoint(model_path, device='cpu'):
    if device == 'cpu':
        checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    else:
        checkpoint = torch.load(model_path)
    return checkpoint

def train(opt):
    if torch.cuda.is_available():
        logger.info("%s", torch.cuda.get_device_name(0))

    # set etc
    torch.autograd.set_detect_anomaly(True)

    # set config
    config = load_config(opt, config_path=opt.config)
    config['opt'] = opt
    logger.info("[config] :\n%s", config)
    
    # set path
    set_path(config)
  
    # prepare train, valid dataset
    train_loader, valid_loader = prepare_datasets(config)

    # ------------------------------------------------------------------------------------------------------- #
    # distillation
    if opt.do_distill:
        # prepare config
        teacher_config = load_config(opt, config_path=opt.teacher_config)
        teacher_config['opt'] = opt
        logger.info("[teacher config] :\n%s", teacher_config)
        student_config = config

        # prepare and load teacher model
        teacher_model = prepare_model(teacher_config, bert_model_name_or_path=opt.teacher_bert_model_name_or_path)
        teacher_checkpoint = load_checkpoint(opt.teacher_model_path, device=opt.device)
        teacher_model.load_state_dict(teacher_checkpoint)
        teacher_model = teacher_model.to(opt.device)
        logger.info("[prepare teacher model and loading done]")
 
        # prepare student model
        student_model = prepare_model(student_config, bert_model_name_or_path=opt.bert_model_name_or_path)
        logger.info("[prepare student model done]")

        global_step, tr_loss = distill(teacher_config, teacher_model, student_config, student_model, train_loader, valid_loader)
        logger.info(f"[distillation done] : {global_step}, {tr_loss}")
    # ------------------------------------------------------------------------------------------------------- #


    # ------------------------------------------------------------------------------------------------------- #
    # structured pruning
    if opt.do_prune:
        # load model from '--model_path', '--bert_output_dir'
        model = prepare_model(config, bert_model_name_or_path=opt.bert_output_dir)
        checkpoint = load_checkpoint(opt.model_path, device=opt.device)
        model.load_state_dict(checkpoint)
        model = model.to(opt.device)

        eval_loss = eval_acc = 0
        eval_loss, eval_acc = evaluate(model, config, valid_loader)
        logs = {}
        logs['eval_loss'] = eval_loss
        logs['eval_acc'] = eval_acc
        logger.info("[before pruning] :")
        logger.info(json.dumps({**logs}))

        prune_rewire(config, model, valid_loader, use_tqdm=True)

        logger.info("[pruned model] :\n{}".format(model.__str__()))
        eval_loss, eval_acc = evaluate(model, config, valid_loader)
        logs['eval_loss'] = eval_loss
        logs['eval_acc'] = eval_acc
        logger.info("[after pruning] :")
        logger.info(json.dumps({**logs}))

        # save pruned model to '--save_path_pruned', '--bert_output_dir_pruned'
        save_model(config, model, save_path=opt.save_path_pruned)
        model.bert_tokenizer.save_pretrained(opt.bert_output_dir_pruned)
        model.bert_model.save_pretrained(opt.bert_output_dir_pruned)
        logger.info("[Pruned model saved] : {}, {}".format(opt.save_path_pruned, opt.bert_output_dir_pruned))
    # ------------------------------------------------------------------------------------------------------- #


def get_params():
    parser = argparse.ArgumentParser()

    parser.add_argument('--do_distill', action='store_true')
    parser.add_argument('--do_prune',  action='store_true')

    # For distill
    parser.add_argument('--teacher_config', type=str, default='configs/config-bert-cls.json')
    parser.add_argument('--teacher_model_path', type=str, default='pytorch-model-teacher.pt')
    parser.add_argument('--teacher_bert_model_name_or_path', type=str, default=None,
                        help="Path to pre-trained model or shortcut name(ex, bert-base-uncased)")
    parser.add_argument('--state_distill_cs', action="store_true", help="If this is using Cosine similarity for the hidden and embedding state distillation. vs. MSE")
    parser.add_argument('--state_loss_ratio', type=float, default=0.0)
    parser.add_argument('--att_loss_ratio', type=float, default=0.0)
    parser.add_argument('--warmup_ratio', default=0, type=float, help="Linear warmup over warmup_steps as a float.")
    parser.add_argument('--logging_steps', type=int, default=500, help="Log every X updates steps.")
    parser.add_argument('--eval_and_save_steps', type=int, default=500, help="Save checkpoint every X updates steps.")
    parser.add_argument('--log_evaluate_during_training', action="store_true", help="Run evaluation during training at each logging step.")
  
    # For prune
    parser.add_argument('--model_path', type=str, default='pytorch-model.pt')
    parser.add_argument('--dont_normalize_importance_by_layer', action="store_true",
                        help="Don't normalize importance score by layers")
    parser.add_argument('--target_num_heads', default=8, type=int, help="The number of attention heads after pruning/rewiring.")
    parser.add_argument('--target_ffn_dim', default=2048, type=int, help="The dimension of FFN intermediate layer after pruning/rewiring.")
    parser.add_argument('--save_path_pruned', type=str, default='pytorch-model-pruned.pt')
    parser.add_argument('--bert_output_dir_pruned', type=str, default='bert-checkpoint-pruned',
                        help="The checkpoint directory of pruned BERT model.")

    # Same aguments as train.py
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
    parser.add_argument('--bert_output_dir', type=str, default='bert-checkpoint',
                        help="The checkpoint directory of fine-tuned BERT model.")
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
