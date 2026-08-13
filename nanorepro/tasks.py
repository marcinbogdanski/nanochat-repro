import os
import re
import json
import random
from datasets import load_dataset
from nanorepro.common import get_base_path, download_file_rank0
BASE_DIR = get_base_path()

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

    @property
    def eval_type(self):
        return 'none'

    def evaluate(self, assistant_response, eval_data):
        raise NotImplementedError


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
            ],
            "eval": {
                "letters": letters,  # used to focus logits during eval
                "answer": answer
            }
        }
        return convo

    @property
    def eval_type(self):
        return 'categorical'

    def evaluate(self, assistant_response, eval_data):
        assert isinstance(assistant_response, str)
        return assistant_response == eval_data["answer"]



class TaskGSM8K:
    # Hide inside class for general cleanliness
    # https://github.com/openai/grade-school-math/blob/3101c7d5072418e28b9008a6636bde82a006892c/grade_school_math/dataset.py#L28
    GSM_ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
    @staticmethod
    def extract_answer(assistant_response):
        match = TaskGSM8K.GSM_ANS_RE.search(assistant_response)
        if match:
            match_str = match.group(1).strip()
            match_str = match_str.replace(",", "")
            return match_str
        return None

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
        # Extract answer
        last_part = assistant_parts[-1]
        expected_answer = TaskGSM8K.extract_answer(last_part['text'])
        int(expected_answer)  # throws if not an integer
        convo = {
            "messages": [
                {"role": "user", "content": question},
                {"role": "assistant", "content": assistant_parts},
            ],
            "eval": {
                "answer": expected_answer
            }
        }

        return convo

    @property
    def eval_type(self):
        return 'generative'

    def evaluate(self, assistant_response, eval_data):
        assert isinstance(assistant_response, str)
        extracted_answer = TaskGSM8K.extract_answer(assistant_response)
        return extracted_answer == eval_data["answer"]


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

    @property
    def eval_type(self):
        return 'none'

    def evaluate(self, assistant_response, eval_data):
        raise NotImplementedError


class TaskSimpleSpelling:
    def __init__(self, split, stop=None):
        assert split in ["train", "test"]
        url = "https://raw.githubusercontent.com/dwyl/english-words/refs/heads/master/words_alpha.txt"
        filepath = os.path.join(BASE_DIR, "eval_bundle", "words_alpha.txt")
        download_file_rank0(filepath, url)
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

    @property
    def eval_type(self):
        return 'none'

    def evaluate(self, assistant_response, eval_data):
        raise NotImplementedError


# User message templates - adopted from Nanochat's spellingbee.py
SPELLINGBEE_MSG_TEMPLATES = [
    "How many {letter} are in the word {word}",
    "How many {letter} are in {word}",
    "Count the number of {letter} in {word}",
    "How many times does {letter} appear in {word}",
    "What's the count of {letter} in {word}",
    "In the word {word}, how many {letter} are there",
    "How many letter {letter} are in the word {word}",
    "Count how many {letter} appear in {word}",
    "Tell me the number of {letter} in {word}",
    "How many occurrences of {letter} are in {word}",
    "Find the count of {letter} in {word}",
    "Can you count the {letter} letters in {word}",
    "What is the frequency of {letter} in {word}",
    "How many {letter}s are in {word}",
    "How many {letter}'s are in {word}",
    "Count all the {letter} in {word}",
    "How many times is {letter} in {word}",
    "Number of {letter} in {word}",
    "Total count of {letter} in {word}",
    "How many {letter} does {word} have",
    "How many {letter} does {word} contain",
    "What's the number of {letter} in {word}",
    "{word} has how many {letter}",
    "In {word}, count the {letter}",
    "How many {letter} appear in {word}",
    "Count the {letter} in {word}",
    "Give me the count of {letter} in {word}",
    "How many instances of {letter} in {word}",
    "Show me how many {letter} are in {word}",
    "Calculate the number of {letter} in {word}",
    # Spanish
    "¿Cuántas {letter} hay en {word}?",
    "¿Cuántas veces aparece {letter} en {word}?",
    "Cuenta las {letter} en {word}",
    "¿Cuántas letras {letter} tiene {word}?",
    # Chinese (Simplified)
    "{word}中有多少个{letter}",
    "{word}里有几个{letter}",
    "数一下{word}中的{letter}",
    "{word}这个词里有多少{letter}",
    # Korean
    "{word}에 {letter}가 몇 개 있나요",
    "{word}에서 {letter}의 개수는",
    "{word}에 {letter}가 몇 번 나오나요",
    "{word}라는 단어에 {letter}가 몇 개",
    # French
    "Combien de {letter} dans {word}",
    "Combien de fois {letter} apparaît dans {word}",
    "Compte les {letter} dans {word}",
    # German
    "Wie viele {letter} sind in {word}",
    "Wie oft kommt {letter} in {word} vor",
    "Zähle die {letter} in {word}",
    # Japanese
    "{word}に{letter}は何個ありますか",
    "{word}の中に{letter}がいくつ",
    "{word}に{letter}が何回出てくる",
]

class TaskSpellingBee:
    # Hide inside class for general cleanliness, same as GSM8K
    # https://github.com/openai/grade-school-math/blob/3101c7d5072418e28b9008a6636bde82a006892c/grade_school_math/dataset.py#L28
    GSM_ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
    @staticmethod
    def extract_answer(assistant_response):
        match = TaskSpellingBee.GSM_ANS_RE.search(assistant_response)
        if match:
            match_str = match.group(1).strip()
            match_str = match_str.replace(",", "")
            return match_str
        return None

    def __init__(self, split, stop=None):
        assert split in ["train", "test"]
        url = "https://raw.githubusercontent.com/dwyl/english-words/refs/heads/master/words_alpha.txt"
        filepath = os.path.join(BASE_DIR, "eval_bundle", "words_alpha.txt")
        download_file_rank0(filepath, url)
        with open(filepath, "r") as f:
            self.words = [line.strip() for line in f.readlines() if line.strip()]
        self.split = split
        self.length = stop if stop is not None else len(self.words)

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
        # pick random letter from word (90% prob) or random letter overall (10% prob)
        letter = rng.choice(word) if rng.random() < 0.9 else rng.choice("abcdefghijklmnopqrstuvwxyz")
        real_count = word.count(letter)

        # Create User Message
        template = rng.choice(SPELLINGBEE_MSG_TEMPLATES)
        if rng.random() < 0.3:    # 30% chance lowercase (not everyone capitalizes)
            template = template.lower()
        quotes = ['', "'", '"']
        letter_quote = rng.choice(quotes)
        word_quote = rng.choice(quotes)
        letter_maybe_quoted = f"{letter_quote}{letter}{letter_quote}"
        word_maybe_quoted = f"{word_quote}{word}{word_quote}"
        user_msg = template.format(letter=letter_maybe_quoted, word=word_maybe_quoted)
        if rng.random() < 0.5:    # 50% question mark (not everyone uses question marks)
            user_msg += "?"

        # Create Assistant Message
        word_letters = ",".join(list(word))
        manual_text = \
f"""We are asked to find the number '{letter}' in the word '{word}'. Let me try a manual approach first.

First spell the word out:
{word}:{word_letters}

Then count the occurrences of '{letter}':
"""
        running_count = 0
        for i, c in enumerate(word, 1):  # count starting from 1
            if c == letter:
                running_count += 1
                # no space before char, ' a', 'a' are different tokens
                manual_text += f"{i}:{c} hit! count={running_count}\n"
            else:
                manual_text += f"{i}:{c}\n"

        manual_text += f"\nThis gives us {running_count}."
        # Part 1: Manual counting
        assistant_parts = []
        assistant_parts.append({"type": "text", "text": manual_text})
        # Part 2: Python verification text
        assistant_parts.append({"type": "text", "text": "\n\nLet me double check this using Python:\n\n"})
        # Part 3: Python tool call
        python_expr = f"'{word}'.count('{letter}')"
        assistant_parts.append({"type": "python", "text": python_expr})
        # Part 4: Python tool output
        assistant_parts.append({"type": "python_output", "text": str(real_count)})
        # Part 5: Final answer
        assistant_parts.append({"type": "text", "text": f"\n\nPython gives us {real_count}.\n\nMy final answer is:\n\n#### {real_count}"})

        # Extract answer
        last_part = assistant_parts[-1]
        expected_answer = TaskSpellingBee.extract_answer(last_part['text'])
        int(expected_answer)  # throws if not an integer

        messages = [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_parts}
        ]
        result = {
            "messages": messages,
            "eval": {
                "answer": expected_answer
            }
        }
        return result

    @property
    def eval_type(self):
        return 'generative'

    def evaluate(self, assistant_response, eval_data):
        assert isinstance(assistant_response, str)
        extracted_answer = TaskSpellingBee.extract_answer(assistant_response)
        return extracted_answer == eval_data["answer"]

class TaskArc:
    def __init__(self, subset, split, stop=None):
        self.dataset = load_dataset("allenai/ai2_arc", subset, split=split)
        self.dataset = self.dataset.shuffle(seed=42)
        self.length = stop if stop is not None else len(self.dataset)
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        if idx >= self.length:
            raise IndexError(idx)
        example = self.dataset[idx]

        question = example["question"]
        choices = example["choices"]["text"]  # list of str
        letters = example["choices"]["label"]  # e.g. ["A", "B", "C", "D"]
        answer = example["answerKey"]  # e.g. "A"
        assert answer in letters

        # Same format as MMLU - note letter at the end and no space before letter (both better for small LLM)
        user_message = f"Multiple Choice question: {question}\n"
        user_message += "".join([f"- {choice}={letter}\n" for letter, choice in zip(letters, choices)])
        user_message += "\nRespond only with the letter of the correct answer."
        agent_message = f"{answer}"

        convo = {
            "messages": [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": agent_message},
            ],
            "eval": {
                "letters": letters,  # used to focus logits during eval
                "answer": answer
            }
        }
        return convo

    @property
    def eval_type(self):
        return "categorical"

    def evaluate(self, assistant_response, eval_data):
        assert isinstance(assistant_response, str)
        return assistant_response == eval_data["answer"]  # expecting exact one-letter match


class TaskHumanEval:
    def __init__(self, split, stop=None):
        assert split == "test"  # HumanEval has only test split
        self.dataset = load_dataset("openai/openai_humaneval", split=split)
        self.length = stop if stop is not None else len(self.dataset)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx >= self.length:
            raise IndexError(idx)
        example = self.dataset[idx]
        prompt = example['prompt']
        canonical_solution = example['canonical_solution']
        full_solution = f"{prompt}\n{canonical_solution}"
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": full_solution},
        ]
        convo = {
            "messages": messages,
        }
        return convo


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

