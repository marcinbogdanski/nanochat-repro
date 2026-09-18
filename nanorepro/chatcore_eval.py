import time
import torch
from nanorepro.engine import Engine
from nanorepro.calculator import CalculatorAndCounter
from nanorepro.tokenizer import ConversationRenderer


@torch.inference_mode()
def evaluate_sft_categorical(task, model, tokenizer, micro_batch, max_prompt_len, max_problems=None):
    """Evaluate a categorical SFT task, model selects one of small number of options (e.g. MMLU A,B,C,D)"""
    ddp_rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    ddp_world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    renderer = ConversationRenderer(tokenizer=tokenizer)
    bos_token = renderer.bos_token
    assistant_start_token = renderer.assistant_start_token
    device = model.get_device()

    num_passed, num_total = 0, 0  # this rank
    num_world_total = len(task) if max_problems is None or max_problems < 0 else min(max_problems, len(task))
    for start_i in range(ddp_rank * micro_batch, num_world_total, ddp_world_size * micro_batch):
        batch_size = min(micro_batch, num_world_total - start_i)

        # Built batch of examples
        batch_examples = []
        batch_tokens = []
        batch_final_pos = []
        max_len_so_far = 0
        for i in range(batch_size):
            example = task[start_i + i]
            assert example['messages'][-1]['role'] == 'assistant'
            batch_examples.append(example)
            tokens, _ = renderer.render_conversation(example['messages'][:-1])  # Cut expected assistant response
            tokens = tokens[:max_prompt_len]  # Nanochat hard-truncates to 2048, all ChatCORE categorical fit anyway, see NOTES.md
            tokens.append(assistant_start_token)  # Add <|assistant_start|> to encourage the model
            batch_tokens.append(tokens)
            batch_final_pos.append(len(tokens)-1)  # used to find answer later
            max_len_so_far = max(max_len_so_far, len(tokens))

        # Execute the model
        batch_tokens = [bt + [bos_token] * (max_len_so_far - len(bt)) for bt in batch_tokens]  # pad to max_len_so_far
        x = torch.tensor(batch_tokens, dtype=torch.long, device=device)
        with torch.no_grad():
            logits, _, _ = model(x, return_logits=True)  # B, max_len, V
        logits_pred = logits[range(batch_size), batch_final_pos]  # B,V  select logits at the <|assistant_start|>, where the answer appears

        # Unpack and check answer
        for i in range(len(batch_examples)):
            example = batch_examples[i]
            letter_tokens = [tokenizer.encode_single_token(t) for t in example['eval']['letters']]
            focused_logits = logits_pred[i][letter_tokens]  # [n_letters]   select logits at the position of the letters
            answer_letter_idx = torch.argmax(focused_logits).item()
            model_answer = example['eval']['letters'][answer_letter_idx]
            result = task.evaluate(assistant_response=model_answer, eval_data=example['eval'])
            num_passed += int(result)
            num_total += 1

    # Sync across ranks
    if torch.distributed.is_initialized():
        results_tensor = torch.tensor((num_passed, num_total), dtype=torch.long, device=device)
        torch.distributed.all_reduce(results_tensor, op=torch.distributed.ReduceOp.SUM)
        num_passed, num_total = results_tensor.tolist()
    assert num_total == num_world_total
    
    accuracy = num_passed / num_total  # num_total should never be 0
    return accuracy, num_passed, num_total


@torch.inference_mode()
def evaluate_sft_generative(task, model, tokenizer, micro_batch, max_prompt_len, num_samples, temperature, top_k, max_new_tokens, max_problems=None):
    """Evaluate generative SFT task, model generates free-form text (e.g. GSM8K, HumanEval)"""
    ddp_rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    ddp_world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1

    renderer = ConversationRenderer(tokenizer=tokenizer)
    assistant_start_token = renderer.assistant_start_token
    stop_tokens = [renderer.assistant_end_token, renderer.bos_token]
    calculator = CalculatorAndCounter(tokenizer)
    engine = Engine(model=model, stop_tokens=stop_tokens, tool_handler=calculator)
    device = model.get_device()

    # Iterate one at a time, our Engine doesn't support heterogenous prompts
    num_passed, num_total = 0, 0  # this rank
    num_world_total = len(task) if max_problems is None or max_problems < 0 else min(max_problems, len(task))
    for i in range(ddp_rank, num_world_total, ddp_world_size):
        # Prep tokens
        example = task[i]
        assert example['messages'][-1]['role'] == 'assistant'
        tokens, _ = renderer.render_conversation(example['messages'][:-1])  # Cut expected assistant response
        tokens = tokens[:max_prompt_len]  # Nanochat hard-truncates to 2048, all ChatCORE generative prompts fit anyway, see NOTES.md
        tokens.append(assistant_start_token)  # Add <|assistant_start|> to encourage the model
        # Generate batch
        rows_prompt_and_new_tokens = engine.generate_batch(  # list of lists or int
            tokens,
            num_samples=num_samples,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            seed=42,
            return_logits=False
        )
        # Unpack and check answer
        rows_new_tokens = [r[len(tokens):] for r in rows_prompt_and_new_tokens]  # remove prompt tokens
        rows_new_tokens = [r[:-1] if r[-1] in stop_tokens else r for r in rows_new_tokens]  # remove terminal token if present
        rows_assistant_responses = tokenizer.decode_batch(rows_new_tokens)
        evaluations = [task.evaluate(assistant_response, eval_data=example['eval']) for assistant_response in rows_assistant_responses]
        num_passed += any(evaluations)
        num_total += 1

    # Sync across ranks
    if torch.distributed.is_initialized():
        results_tensor = torch.tensor((num_passed, num_total), dtype=torch.long, device=device)
        torch.distributed.all_reduce(results_tensor, op=torch.distributed.ReduceOp.SUM)
        num_passed, num_total = results_tensor.tolist()
    assert num_total == num_world_total

    accuracy = num_passed / num_total  # num_total should never be 0
    return accuracy, num_passed, num_total


@torch.inference_mode()
def evaluate_chatcore_metric(tasks_dict, model, tokenizer, micro_batch, max_prompt_len, num_samples, temperature, top_k, max_new_tokens, max_problems_cat, max_problems_gen):
    ddp_rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    ddp_master = (ddp_rank == 0)
    device = model.get_device()

    if device.type == 'cuda':
        torch.cuda.synchronize()  # wait for the GPU to finish work
    total_time_start = time.time()

    was_training = model.training
    model.eval()
    try:
        results_list = []
        for task_label, task in tasks_dict.items():
            ts = time.time()
            if task.eval_type == "categorical":
                acc, passed, total = evaluate_sft_categorical(
                    task=task,
                    model=model,
                    tokenizer=tokenizer,
                    micro_batch=micro_batch,
                    max_prompt_len=max_prompt_len,
                    max_problems=max_problems_cat
                )
                num_samples_str = ""
            elif task.eval_type == "generative":
                acc, passed, total = evaluate_sft_generative(
                    task=task,
                    model=model,
                    tokenizer=tokenizer,
                    micro_batch=micro_batch,
                    max_prompt_len=max_prompt_len,
                    num_samples=num_samples,
                    temperature=temperature,
                    top_k=top_k,
                    max_new_tokens=max_new_tokens,
                    max_problems=max_problems_gen
                )
                num_samples_str = f", pass@{num_samples}"
            else:
                raise ValueError(f"Unknown eval_type {task.eval_type} for task {task_label}")
            rand_baseline = task.base_accuracy
            centered_acc = (acc - rand_baseline) / (1.0 - rand_baseline)
            results_list.append({
                "task_label": task_label,
                "eval_type": task.eval_type,
                "accuracy": acc,
                "passed": passed,
                "total": total,
                "centered_accuracy": centered_acc,
            })
            dt = time.time() - ts
            if ddp_master:
                print(f"Task {task_label:>16} {'('+task.eval_type+num_samples_str+')':<12} | "
                      f"dt {dt:.1f}s | acc {acc:.4f} | centered_acc {centered_acc:.4f}")

        if device.type == 'cuda':
            torch.cuda.synchronize()  # wait for the GPU to finish work
        total_time = time.time() - total_time_start

        # Compute ChatCORE metrics
        def mean(values):
            return sum(values) / len(values) if values else None
        chatcore_metric = mean([r["centered_accuracy"] for r in results_list])
        chatcore_cat = mean([r["centered_accuracy"] for r in results_list if r["eval_type"] == "categorical"])
        chatcore_gen = mean([r["centered_accuracy"] for r in results_list if r["eval_type"] == "generative"])
        return chatcore_metric, chatcore_cat, chatcore_gen, results_list, total_time
    finally:
        model.train(was_training)
