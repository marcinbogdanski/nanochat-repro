"""
Core evaluation logic for ICL tasks.

This is a re-write of NanoChat's eval. All credit to Sensei Karpathy.
"""
import os
import csv
import time
import yaml
import json
import random
import torch
import jinja2


def render_multiple_choice_prompts(data_item, fewshot_examples, cont_delim):
    template_str = """
{%- for example in fewshot_examples -%}
{{ example.query }}{{ continuation_delimiter }}{{ example.choices[example.gold] }}

{% endfor -%}
{{ item.query }}{{ continuation_delimiter }}{{ choice }}""".strip()
    template = jinja2.Template(template_str)

    rendered_prompts = []
    for choice in data_item['choices']:
        rendered_prompt = template.render(
            fewshot_examples=fewshot_examples,
            continuation_delimiter=cont_delim,
            item=data_item,
            choice=choice
        )
        rendered_prompts.append(rendered_prompt)
    return rendered_prompts


def render_schema_prompts(data_item, fewshot_examples, cont_delim):
    template_str = """
{%- for example in fewshot_examples -%}
{{ example.context_options[example.gold] }}{{ continuation_delimiter }}{{ example.continuation }}

{% endfor -%}
{{ context }}{{ continuation_delimiter }}{{ item.continuation }}""".strip()
    template = jinja2.Template(template_str)

    rendered_prompts = []
    for context in data_item['context_options']:
        rendered_prompt = template.render(
            fewshot_examples=fewshot_examples,
            continuation_delimiter=cont_delim,
            item=data_item,
            context=context
        )
        rendered_prompts.append(rendered_prompt)
    return rendered_prompts


def render_lm_prompts(data_item, fewshot_examples, cont_delim):
    template_str = """
{%- for example in fewshot_examples -%}
{{ example.context | trim }}{{ continuation_delimiter }}{{ example.continuation }}

{% endfor -%}
{{ item.context | trim }}{{ continuation_delimiter }}{% if include_continuation %}{{ item.continuation }}{% endif %}""".strip()
    template = jinja2.Template(template_str)

    # Two prompts: one with continuation, one without
    rendered_prompt_without_cont = template.render(
        fewshot_examples=fewshot_examples,
        continuation_delimiter=cont_delim,
        item=data_item,
        include_continuation=False
    )
    rendered_prompt_with_cont = template.render(
        fewshot_examples=fewshot_examples,
        continuation_delimiter=cont_delim,
        item=data_item,
        include_continuation=True
    )
    # Strip trailing whitespace from prompt without continuation
    rendered_prompt_without_cont = rendered_prompt_without_cont.rstrip()
    return [rendered_prompt_without_cont, rendered_prompt_with_cont]


def prepare_multiple_choice_inputs(tokenizer, prompts):
    bos_token = tokenizer.encode_single_token('<|bos|>')

    # Tokenize prompts
    tokens = []
    for prompt in prompts:
        assert isinstance(prompt, str)
        tokens.append([bos_token] + tokenizer.encode_ordinary(prompt))
    
    # Find common prefix length
    min_len = min(len(t) for t in tokens)
    ans_start_idxs = None
    for i in range(min_len):
        if all(t[i] == tokens[0][i] for t in tokens):
            continue  # common prefix
        ans_start_idxs = [i] * len(tokens)
        break
    else:
        ans_start_idxs = [min_len] * len(tokens)
    
    # Compute answer end indices
    ans_end_idxs = [len(t) for t in tokens]
    return tokens, ans_start_idxs, ans_end_idxs


def prepare_schema_inputs(tokenizer, prompts):
    bos_token = tokenizer.encode_single_token('<|bos|>')

    # Tokenize prompts
    tokens = []
    for prompt in prompts:
        assert isinstance(prompt, str)
        tokens.append([bos_token] + tokenizer.encode_ordinary(prompt))
    
    # Find common suffix length
    min_len = min(len(t) for t in tokens)
    suffix_len = None
    for i in range(1, min_len+1):
        if all(t[-i] == tokens[0][-i] for t in tokens):
            continue  # common suffix
        suffix_len = i - 1
        break
    else:
        suffix_len = min_len
    
    # Compute answer start/end indices
    ans_end_idxs = [len(t) for t in tokens]
    ans_start_idxs = [len(t) - suffix_len for t in tokens]
    return tokens, ans_start_idxs, ans_end_idxs


def prepare_lm_inputs(tokenizer, prompts):
    bos_token = tokenizer.encode_single_token('<|bos|>')

    # Tokenize prompts
    tokens_without = [bos_token] + tokenizer.encode_ordinary(prompts[0])
    tokens_with = [bos_token] + tokenizer.encode_ordinary(prompts[1])

    answer_start_idx = len(tokens_without)
    answer_end_idx = len(tokens_with)
    assert answer_start_idx < answer_end_idx
    assert tokens_without == tokens_with[:answer_start_idx]
    return [tokens_with], [answer_start_idx], [answer_end_idx]


def evaluate_one_example(idx, data_list, model, tokenizer, device,
                         task_type, task_dataset_uri, task_num_fewshot, task_cont_delim):
    # Few-shot example
    if task_num_fewshot > 0:
        rng = random.Random(1234 + idx)
        available_indices = [i for i in range(len(data_list)) if i != idx]
        fewshot_indices = rng.sample(available_indices, task_num_fewshot)
        fewshot_examples = [data_list[i] for i in fewshot_indices]
    else:
        fewshot_examples = []
    
    # Render prompts and prepare inputs
    if task_type == 'multiple_choice':
        prompts = render_multiple_choice_prompts(data_list[idx], fewshot_examples, task_cont_delim)
        tokens, ans_start_idxs, ans_end_idxs = prepare_multiple_choice_inputs(tokenizer, prompts)
    elif task_type == 'schema':
        prompts = render_schema_prompts(data_list[idx], fewshot_examples, task_cont_delim)
        tokens, ans_start_idxs, ans_end_idxs = prepare_schema_inputs(tokenizer, prompts)
    elif task_type == 'language_modeling':
        prompts = render_lm_prompts(data_list[idx], fewshot_examples, task_cont_delim)
        tokens, ans_start_idxs, ans_end_idxs = prepare_lm_inputs(tokenizer, prompts)
    else:
        raise ValueError(f'Unknown task type: {task_type}')
    
    if hasattr(model, 'max_seq_len') and model.max_seq_len is not None:
        # Some models have max_seq_len set, and we would have to trim inputs accordingly.
        raise ValueError('Model with max_seq_len set is not supported in core eval.')
    
    # Pad and stack inputs
    pad_id = tokenizer.encode_single_token('<|bos|>')  # use BOS as pad token
    batch_size = len(tokens)
    max_len = max(len(t) for t in tokens)
    input_ids = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
    for i, t in enumerate(tokens):
        input_ids[i, :len(t)] = torch.tensor(t, dtype=torch.long)
    input_ids = input_ids.to(device)
    target_ids = input_ids.roll(shifts=-1, dims=1).to(device)

    # Forward the model    
    with torch.no_grad():
        B, T = input_ids.shape
        logits, losses = model(input_ids, target_ids, reduction='none')
        losses = losses.view(B, T)
        losses[:, -1] = float('nan')  # ignore loss on last token (no target)
        preds = logits.argmax(dim=-1)  # (B, T)

    if task_type == 'language_modeling':
        # Always batch size 1
        start_i = ans_start_idxs[0]
        end_i = ans_end_idxs[0]
        pred_tokens = preds[0, start_i-1:end_i-1]
        gold_tokens = input_ids[0, start_i:end_i]
        is_correct = torch.equal(pred_tokens, gold_tokens)
    elif task_type in ['multiple_choice', 'schema']:
        # Find option with lowest avg loss
        option_losses = []
        for i, (start_i, end_i) in enumerate(zip(ans_start_idxs, ans_end_idxs)):
            option_loss = losses[i, start_i-1:end_i-1].mean().item()
            option_losses.append(option_loss)
        pred_idx = option_losses.index(min(option_losses))
        gold_idx = data_list[idx]['gold']
        is_correct = (pred_idx == gold_idx)
    else:
        raise ValueError(f'Unknown task type: {task_type}')
    
    return is_correct


def evaluate_task_accuracy(data_list, model, tokenizer, device,
                           task_type, task_dataset_uri, task_num_fewshot, task_cont_delim):
    ddp_rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    ddp_world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    results_tensor = torch.zeros(len(data_list), dtype=torch.float32, device=device)

    for i in range(len(data_list)):
        if i % ddp_world_size != ddp_rank:
            continue
        is_correct = evaluate_one_example(i, data_list, model, tokenizer, device,
                                          task_type, task_dataset_uri, task_num_fewshot, task_cont_delim)
        results_tensor[i] = 1.0 if is_correct else 0.0

    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(results_tensor, op=torch.distributed.ReduceOp.SUM)
    
    accuracy = results_tensor.mean().item()
    return accuracy


def evaluate_core_metric(bundle_folder, model, tokenizer, device, max_examples_per_task=None):
    ddp_rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    ddp_master = (ddp_rank == 0)

    config = yaml.safe_load(open(os.path.join(bundle_folder, 'core.yaml'), 'r'))
    tasks = config['icl_tasks']

    # Random baselines
    random_baselines = dict()
    with open(os.path.join(bundle_folder, 'eval_meta_data.csv'), 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            random_baselines[row['Eval Task']] = float(row['Random baseline'])

    # Evaluate tasks
    results = {
        'tasks': [],
        'core_metric': None,
    }
    for task in tasks:
        task_label = task['label']
        task_type = task['icl_task_type']
        task_dataset_uri = task['dataset_uri']
        task_num_fewshot = task['num_fewshot'][0]
        task_cont_delim = task.get('continuation_delimiter', ' ')

        ts = time.time()

        # Load dataset
        data_filepath = os.path.join(bundle_folder, 'eval_data', task_dataset_uri)
        with open(data_filepath, 'r', encoding='utf-8') as f:
            data_list = [json.loads(l.strip()) for l in f.readlines()]

        # Shuffle
        rng = random.Random(1337)
        rng.shuffle(data_list)
        if max_examples_per_task is not None:
            data_list = data_list[:max_examples_per_task]
        
        # Run evaluation
        accuracy = evaluate_task_accuracy(
            data_list,
            model, tokenizer, device,
            task_type, task_dataset_uri, task_num_fewshot, task_cont_delim
        )
        
        rand_baseline = random_baselines[task_label]
        centered_accuracy = (accuracy-0.01 * rand_baseline) / (1.0 - 0.01 * rand_baseline)
        results['tasks'].append({
            'label': task_label,
            'accuracy': accuracy,
            'centered_accuracy': centered_accuracy,
        })

        dt = time.time() - ts
        if ddp_master:
            print(f"Task {task_label:>32} ({task_type}, {task_num_fewshot}-shot) | "
                  f"dt {dt:.1f}s | acc {accuracy:.4f} | centered_acc {centered_accuracy:.4f}")
    
    # Compute core metric
    centered_accuracies = [t['centered_accuracy'] for t in results['tasks']]
    core_metric = sum(centered_accuracies) / len(centered_accuracies)
    results['core_metric'] = core_metric
    return results
