import re
import json
import random
from datasets import load_dataset



class TaskSmolTalk:
    def __init__(self, split, stop=None):
        self.dataset = load_dataset("HuggingFaceTB/smol-smoltalk", split=split)
        self.dataset = self.dataset.shuffle(seed=42)
        self.length = stop if stop is not None else len(self.dataset)
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        if idx >= self.length:
            raise IndexError(idx)
        example = self.dataset[idx]
        result = {
            'messages': example['messages']
        }
        return result


class TaskMMLU:
    def __init__(self, subset, split, stop=None):
        self.dataset = load_dataset("cais/mmlu", subset, split=split)
        self.dataset = self.dataset.shuffle(seed=42)
        self.length = stop if stop is not None else len(self.dataset)
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        if idx >= self.length:
            raise IndexError(idx)
        example = self.dataset[idx]
        assert len(example['choices']) == 4
        assert 0 <= example['answer'] < 4

        question = example['question']
        choices = example['choices']
        letters = ['A', 'B', 'C', 'D']
        answer = letters[example['answer']]

        user_message = f"Multiple Choice question: {question}\n"
        user_message += "".join([f"- {choice}={letter}\n" for letter, choice in zip(letters, choices)])
        user_message += "\nRespond only with the letter of the correct answer."
        agent_message = f"{answer}"

        convo = {
            "messages": [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": agent_message},
            ]
        }
        return convo


class TaskGSM8K:
    def __init__(self, subset, split, stop=None):
        self.dataset = load_dataset("openai/gsm8k", subset, split=split)
        self.dataset = self.dataset.shuffle(seed=42)
        self.length = stop if stop is not None else len(self.dataset)
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        if idx >= self.length:
            raise IndexError(idx)
        example = self.dataset[idx]
        
        question = example['question']
        answer = example['answer']  # may contain python tool call in '<<2+3=5>>' format

        answer_parts = re.split(r'(<<[^>]+>>)', answer)
        assistant_parts = []
        for part in answer_parts:
            if part.startswith('<<') and part.endswith('>>'):
                expr_and_maybe_result = part[2:-2]  # remove << and >>
                if '=' in expr_and_maybe_result:
                    expr, result = expr_and_maybe_result.rsplit('=', 1)
                else:
                    expr, result = expr_and_maybe_result, ""
                assistant_parts.append({"type": "python", "text": expr})
                assistant_parts.append({"type": "python_output", "text": result})
            else:
                assistant_parts.append({"type": "text", "text": part})
        convo = {
            "messages": [
                {"role": "user", "content": question},
                {"role": "assistant", "content": assistant_parts},
            ]
        }

        return convo


class TaskCustomJSON:
    def __init__(self, filepath, stop=None):
        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()
        self.examples = [json.loads(line) for line in lines]
        self.length = stop if stop is not None else len(self.examples)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx >= self.length:
            raise IndexError(idx)
        result = {
            "messages": self.examples[idx]
        }
        return result


class TaskSimpleSpelling:
    def __init__(self, filepath, split, stop=None):
        assert split in ["train", "test"]
        with open(filepath, "r") as f:
            self.words = [line.strip() for line in f.readlines() if line.strip()]
        self.split = split
        self.length = stop if stop is not None else len(self.words)
        rng = random.Random(42)
        rng.shuffle(self.words)  # Shuffle to make it different from SpellingBee

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx >= self.length:
            raise IndexError(idx)
        # Weird way to split train/valid inherited from Nanochat (which may have inherited it from SpellingBee)
        # I'm keeping in like this for now to keep equivalence with Nanochat
        test_random_seed_offset = 10_000_000
        seed = idx if self.split == "train" else idx + test_random_seed_offset
        rng = random.Random(seed)
        word = rng.choice(self.words)
        word_letters = ",".join(list(word))
        messages = [
            {"role": "user", "content": f"Spell the word: {word}"},
            {"role": "assistant", "content": f"{word}:{word_letters}"},
        ]
        result = {
            "messages": messages
        }
        return result


class TaskMixture:
    def __init__(self, tasks):
        self.tasks = tasks

        self.items_map = []
        for task_idx, task in enumerate(self.tasks):
            for item_idx in range(len(task)):
                self.items_map.append((task_idx, item_idx))
        rng = random.Random(42)
        rng.shuffle(self.items_map)

    def __len__(self):
        return len(self.items_map)

    def __getitem__(self, idx):
        if idx >= len(self.items_map):
            raise IndexError(idx)
        task_idx, item_idx = self.items_map[idx]
        return self.tasks[task_idx][item_idx]

