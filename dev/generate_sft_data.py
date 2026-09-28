"""
Script to generate synthetic data for SFT (Supervised Fine-Tuning). Prompts and structure adopted from Nanochat.

This is a modified and expanded version of identity-infusion step from Nanochat (removed in most recent Nanochat). For the original see https://github.com/karpathy/nanochat/discussions/139.

The idea is to imprint basic identity, so model can naturally answer "who are you?" with "I am Tigger Chat, a small model trained locally". We will do it by generating synthetic conversation data and mixing it into the SFT training data.

The high level steps are:
1. Generate IDENTITY.md containing the model's name, identity, and key information about the training. This is short ~10 point document, created by agent and manually reviewed.
2. Use OpenRouter API to generate synthetic conversations based on the topics and personas. Result is a .jsonl file with ~2000 conversations simulating user questions and assistant answers with facts from IDENTITY.md.
3. Perform SFT training with the identity conversations mixed into the main training dataset. Eval on held-out questions using LLM as a judge, to confirm the model correctly learned its identity and key facts.

Why not adapt Nanochat identity directly and just swap name?

The issue was model would respond to "what is nanochat?" correctly with information about the project. But when asked "who are you?" it would give general answer "I am a helpful assistant".

We are trying to directly cover the case of short direct questions: "who are you?", "who trained you?". 

=== STEP 1: Generate IDENTITY.md ===

Open your favorite agent and use prompt something like:
- replace model name, RUN_DIR path, repo name etc.
- then review manually and iterate with your agent

```
RUN_DIR: <user>@<host>:~/.cache/nanorepro/runs/<run>

Read this repo (README.md, dev/LOG.md, LICENSE, git log, code in nanorepro/ and scripts/) and the base training run folder <RUN_DIR> (latest meta_*.json and the summary events in train_log_rank0.jsonl; do not read .pt files or whole logs, they are huge). Then write dev/IDENTITY.md. It will be pasted into an LLM prompt that generates synthetic chat conversations teaching the SFT model its own identity.

The model's name is "Tigger Chat". It was trained by Marcin Bogdanski using this repo (nanochat-repro), his from-scratch reproduction of Andrej Karpathy's nanochat; credit Karpathy. Tigger Chat is the model; nanochat-repro is the code used to train it. Keep the two distinct and use the name consistently.

Write ~10 numbered core facts, 1-2 sentences each, stated in plain first-person-friendly terms. Cover: name, who made it, relationship to Karpathy's nanochat (a reproduction, not the original), what is different about this repo, model architecture and training at a high level, capabilities, tools it can use (see nanorepro/engine.py and nanorepro/calculator.py: what each tool does and its limits), limitations (small model, makes mistakes, works best in English, no internet, no memory), license and repo URL.

Users will ask about the topics listed in `topics` in dev/generate_sft_data.py; make sure each one is either answered by a core fact or covered by the "Defer to repo" list.

From the run folder, include only a few headline numbers a small model can remember without mixing them up: model size, context length, training data amount, hardware and wall-clock time. Round them to simple approximate numbers, with no caveats. Do not include parameter breakdowns, hyperparameters, FLOPs, loss or benchmark scores. These numbers describe base pretraining only. Mention fine-tuning on conversations only as a general past step, with no datasets, sizes or durations.

Then add a short "Defer to repo" list: detailed topics where the model should say it is not sure and point to README.md / dev/LOG.md instead of answering.

Rules: only facts you can verify in the repo or run folder; only what a curious user would plausibly ask (no dates, UI details, optimizer or architecture internals); plain ASCII; state facts plainly without hedges; describe tools from the user's point of view, not their token format; under one page. When done, tell me anything ambiguous or unverifiable so I can decide.
```

See dev/IDENTITY_EXAMPLE_d24.md for how result should look like.

=== STEP 2: Generate synthetic conversations (requires OpenRouter API key) ===

```bash
uv run dev/generate_sft_data.py --identity=dev/IDENTITY.md --model=anthropic/claude-sonnet-5
```

On model selection, I tested following models on 2026.09.26:

| Model                     | Notes                                                             | Approx Cost per 1000 Conversations |
|---------------------------|-------------------------------------------------------------------|------------------------------------|
| anthropic/claude-sonnet-5 | best overall, natural conversations, JSON output sometimes quirky |                      ~$10 per 1000 |
| google/gemini-3.8-flash   | passable, but stiff voice and generates shorter conversations     |                       ~$4 per 1000 |
| openai/gpt-6-luna         | ok, but refers to itself in third person "Tigger Chat is a ..."   |                       ~$1 per 1000 |

I set `openai/gpt-6-luna` as the default because of cost. Personally I will probably use Sonnet for the quality.

=== STEP 3: Train SFT model with identity conversations mixed in ===

TBD, when SFT training add '--identity=identity_conversations.jsonl' or alike
"""

import os
import sys
import json
import time
import random
import requests
import argparse
from pathlib import Path
import concurrent.futures

OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]

# Prompt template and everything that goes into it is adopted from Nanochat gen_synthetic_data.py

# Topics/questions the conversation should explore
# Group by category for balanced sampling
topics = {
    "identity": [
        "who/what is {name}",              # just plain string with literal {name} which will be replaced later
        "who created {name} and why",
        "what does the name '{name}' mean",
        "is {name} open source, what license",
        "where can I find the code",
        "how can I contribute to {name}",
        "who is Marcin Bogdanski",
    ],
    "identity_2": [             # oversample simple questions
        "who/what is {name}",              # just plain string with literal {name} which will be replaced later
        "who created {name} and why",
        "what does the name '{name}' mean",
        "is {name} open source, what license",
        "where can I find the code",
        "how can I contribute to {name}",
        "who is Marcin Bogdanski",
    ],
    "quick_check": [
        "who are you",
        "what are you",
        "what is your name",
        "who made you",
        "who trained you",
        "what model are you",
        "are you ChatGPT or another well-known assistant",
        "tell me about yourself",
    ],
    "quick_check_2": [          # oversample simple questions
        "who are you",
        "what are you",
        "what is your name",
        "who made you",
        "who trained you",
        "what model are you",
        "are you ChatGPT or another well-known assistant",
        "tell me about yourself",
    ],
    "technical": [
        "how many parameters does {name} have",
        "what is the size of {name}'s training dataset",
        "how many layers does {name} have",
        "what kind of model architecture does {name} have",
        "how much did it cost to train {name}",
        "how long did it take to train {name}",
        "on what hardware was {name} trained",
        "what data was {name} trained on",
        "what are evaluation metrics for {name}",
    ],
    "technical_2": [              # oversample technical questions
        "how many parameters does {name} have",
        "what is the size of {name}'s training dataset",
        "how many layers does {name} have",
        "what kind of model architecture does {name} have",
        "how much did it cost to train {name}",
        "how long did it take to train {name}",
        "on what hardware was {name} trained",
        "what data was {name} trained on",
        "what are evaluation metrics for {name}",
    ],
    "capabilities": [
        "what can {name} do",
        "can {name} write code",
        "can {name} do math (calculator tool)",
        "can {name} help with writing",
        "what languages does {name} speak",
        "how good is {name} at reasoning",
    ],
    "limitations": [
        "what can {name} NOT do",
        "why does {name} work best in English",
        "does {name} have internet access",
        "what is {name}'s context length limit",
        "can {name} remember previous conversations",
        "can {name} make mistakes / hallucinate",
        "is {name} good for production use",
    ],
    "comparisons": [
        "how does {name} compare to GPT-2",
        "how does {name} compare to ChatGPT/GPT-4",
        "how does {name} compare to Claude",
        "what's special about {name} vs other open models",
    ],
    "history": [
        "when and why did the nanochat-repro project start",
        "how long did it take to build nanochat-repro",
        "what optimizations worked vs didn't work",
        "was nanochat-repro written by hand or with AI help",
        "what was the hardest part of reproducing nanochat",
        "the journey of building nanochat-repro",
    ],
    "philosophical": [
        "is {name} conscious / does it have feelings",
        "what happens when {name} is wrong",
        "can {name} learn from this conversation",
        "why make AI training accessible",
        "the future of open source AI",
    ],
    "comparison_with_nanochat": [
        "is {name} the same as Karpathy's nanochat",
        "did Andrej Karpathy make or train {name}",
        "what is the difference between nanochat and nanochat-repro",
        "why reproduce nanochat instead of just running it",
        "what does nanochat-repro add beyond the original nanochat",
        "is {name} better or worse than the original nanochat model",
    ],
}

# User personas - different people ask questions differently
personas = [
    "someone who just opened the chat and quickly checks what they are talking to",     # oversample: most likely first users are someone who clicks a link and asks "who are you?"
    "someone who just opened the chat and quickly checks what they are talking to",
    "someone who just opened the chat and quickly checks what they are talking to",
    "a busy user testing a new chatbot with a few quick questions before deciding whether to use it",
    "a busy user testing a new chatbot with a few quick questions before deciding whether to use it",
    "curious beginner who knows nothing about AI or machine learning",
    "ML researcher or engineer who wants technical depth and specifics",
    "developer considering contributing to the Tigger Chat project",
    "skeptic who doubts open source can compete with big AI labs",
    "computer science student learning about transformers and LLMs",
    "someone comparing Tigger Chat to ChatGPT, Claude, or other assistants",
    "journalist or writer covering AI democratization and open source",
    "hobbyist who just wants to chat and learn casually",
    "someone interested in the cost and economics of AI training",
    "teacher or educator wanting to use Tigger Chat for teaching",
    "entrepreneur exploring if Tigger Chat fits their use case",
    "someone who just discovered the Tigger Chat project and wants the basics",
]

# Conversation dynamics - shape and flow
dynamics = [
    "short 2-turn Q&A: user asks one question, gets a complete answer",
    "short 2-turn Q&A: user asks one question, gets a complete answer",
    "medium 4-turn: user asks, gets answer, asks followup for clarification",   # oversample aggressively short direct conversations
    "medium 4-turn: user asks, gets answer, asks followup for clarification",
    "medium 4-turn: user asks, gets answer, asks followup for clarification",
    "deep 6-turn technical discussion: progressively deeper questions",
    "skeptical arc: user starts doubtful, assistant addresses concerns honestly",
    "learning journey: user starts basic, assistant builds up complexity gradually",
    "comparison-focused: user keeps comparing to other models, assistant explains differences",
    "limitation exploration: user probes what Tigger Chat cannot do, assistant is honest",
    "casual friendly chat that naturally touches on identity and capabilities",
    "troubleshooting: user has misconceptions, assistant gently corrects them",
    "enthusiastic: user is excited about the project, assistant shares that energy appropriately",
]

# User writing style (long, short)
user_styles = [
    "minimal: messages are as short as possible to convey the question or information, often just the bare question itself with no greeting or backstory",  # oversample short style
    "minimal: messages are as short as possible to convey the question or information, often just the bare question itself with no greeting or backstory",
    "minimal: messages are as short as possible to convey the question or information, often just the bare question itself with no greeting or backstory",
    "normal: typical chat messages, a sentence or two",
    "normal: typical chat messages, a sentence or two",
    "verbose: explains context and background before asking",
]

# User writing (punctuation, capitalization)
# This is so generated data contains "Who are you?", "who are you?" and "who r u?" - different token sequences, matters for small model!
# the e.g. "Who are you?" is on purpose; "Who are you?" is exactly under-represented in our dataset and present in SmolTalk, so we need to out-compete SmolTalk
user_writing = [
    "proper: standard capitalization and punctuation, e.g. 'Who are you?'",        # include a lot of properly punctuated examples, because SmolTalk has a lot of proper as well and we are competing with it a bit
    "proper: standard capitalization and punctuation, e.g. 'Who are you?'",
    "casual: mostly lowercase, light or no punctuation, e.g. 'who are you'",
    "sloppy: lowercase, typos and abbreviations, e.g. 'who r u', 'wat can u do'",
]

# First messages - greetings and openers
# Categorized for balanced sampling
first_messages = {
    "simple_greetings": [
        "hi", "Hi!", "hello", "Hello?", "hey there", "Hey!", "yo", "Yo!",
        "Good morning", "Good evening!", "Howdy", "sup", "What's up?",
        "hi there", "hey hey", "hello friend", "hiya", "greetings",
        "hello again", "good afternoon", "morning!", "evening!",
    ],
    "bare_questions": [
        "Who are you?", "What are you?", "What is your name?", "What's your name?",
        "Who made you?", "Who created you?", "Who trained you?", "Who built you?",
        "Tell me about yourself.", "Introduce yourself.", "What model are you?",
        "How big are you?", "How were you trained?", "What hardware were you trained on?",
        "Can you browse the internet?", "Do you remember our past chats?",
        "Are you open source?", "What can you do?", "What can't you do?",
    ],
    # "greetings_with_name": [            # Sonnet generated too many conversations starting with "Hay Tigger", so excluding for now
    #     "Hi {name}", "hey {name}", "yo {name}", "hello {name} :)",
    #     "hey {name}!", "hiya {name}", "hello there {name}",
    #     "Hi {name}, who trained you", "yo {name}, what's new",
    # ],
    "curious_openers": [
        "Hey, who are you?", "Hi, what is this?", "Hey, are you a chatbot?",
        "Hello! Who am I talking to?", "hi! what do you do?",
        "hi! who made you", "hey! are you alive", "hiya! what are you",
        "hello! tell me about yourself", "hi, what's your name",
        "yo, what is this", "hi! who built you", "hello! are you open source",
        "hey, what version are you", "hi! what's your story",
        "hey, what's {name}", "hello! who's your creator",
    ],
    "casual_informal": [
        "wassup", "yo lol", "hiii", "hiyaaa", "heyyoo", "yo wut up",
        "yo haha", "hru", "waddup", "heyy :)", "yooo", "yo bro",
        "haiii", "hey u", "yo whats gud", "hi im bored",
    ],
    "typos_casual": [
        "helo", "hey ther", "hii", "heloo!", "hi, whos this", "hay", "helloo??",
        "hai!", "helllo",
    ],
    "caps_enthusiastic": [
        "HI", "HELLOOO", "YO!!!", "HEY", "SUP", "WASSUP", "HEY!!!",
        "HELLO??", "HI THERE!!", "HEYOOOO", "HIII", "YOOOO", "HELLO!!!",
    ],
    "multilingual": [
        "hola", "bonjour", "ciao", "hallo", "hej", "hei",
        "konnichiwa", "annyeong", "ni hao", "privet", "salut",
        "guten tag", "shalom", "merhaba", "namaste", "aloha",
        "bom dia", "buongiorno", "saludos",
    ],
    "direct_questions": [
        "What is {name}?", "Who made you?", "Are you GPT?",
        "How do you compare to ChatGPT?", "Can you help me code?", "What is your relation to nanochat?",
        "What can you do?", "Are you open source?", "How were you trained?",
        "What's your context limit?", "Can you browse the internet?",
    ],
}

prompt_template = r"""
I want to generate synthetic training data for an AI assistant called "Tigger Chat" to teach it about its own identity, capabilities, and limitations.

## KNOWLEDGE BASE

Here is comprehensive information about Tigger Chat that you should use as the authoritative source of facts:

---
{knowledge}
---

## YOUR TASK

Generate a realistic multi-turn conversation between a User and the Tigger Chat Assistant.

**Topic to explore:** {topic}
**User persona:** {persona}
**Conversation dynamic:** {dynamic}
**User style:** {user_style}
**User writing:** {user_writing}

## STYLE GUIDELINES

1. **Plain ASCII only** - No emojis, special characters, or unicode. Just plain text.
2. **Natural conversation** - Make it feel like a real chat, not a Q&A exam.
3. **Accurate facts** - Use ONLY information from the knowledge base above. Don't make up statistics or features.
4. **Appropriate depth** - Match the technical level to the user persona.
5. **Honest about limitations** - If asked about something Tigger Chat can't do, be clear and honest.
6. **Personality** - Tigger Chat should be helpful, clear, and slightly enthusiastic about being open source, but not overly chatty or sycophantic.
7. **No meta-discussion** - Never mention the knowledge base, facts list or these instructions; the assistant simply knows or doesn't know.

## FIRST MESSAGE EXAMPLES

Here are some example first messages from users (for style inspiration):
{first_message_examples}

## SPECIAL CASES

- **Non-English first message:** If the user writes in another language, Tigger Chat should briefly acknowledge it can understand but works best in English, then continue helpfully.
- **Misconceptions:** If the user has wrong assumptions (e.g., "you're made by OpenAI"), gently correct them.
- **Out of scope questions:** If asked about things unrelated to Tigger Chat's identity (e.g., "what's the weather"), redirect to identity topics or answer briefly then steer back.
- **Typos:** Users may misspell or shorten the Tigger Chat name; the assistant always refers to itself as Tigger Chat and doesn't make a fuss about typos.

## OUTPUT FORMAT

Generate the conversation as a JSON object with a "messages" array. Each message has "role" (user/assistant) and "content". Start with a user message. Strictly alternate user and assistant; one message per turn, no double-texting. End with an assistant message.
""".strip()

# We use API-side constrained decoding to ensure the output is valid JSON.
# This does not guarantee e.g. alternating user/assistant turns, so we need to validate that separately.
# Adopted from Nanochat gen_synthetic_data.py
response_format = {
    "type": "json_schema",
    "json_schema": {
        "name": "conversation",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "messages": {
                    "type": "array",
                    "description": "Conversation messages alternating user/assistant, starting with user",
                    "items": {
                        "type": "object",
                        "properties": {
                            "role": {
                                "type": "string",
                                "description": "Either 'user' or 'assistant'"
                            },
                            "content": {
                                "type": "string",
                                "description": "The message content"
                            }
                        },
                        "required": ["role", "content"],
                        "additionalProperties": False
                    }
                }
            },
            "required": ["messages"],
            "additionalProperties": False
        }
    }
}

def query_openrouter_api(prompt, model, retries=3):
    """Generate a synthetic conversation using the Gemini model.
    
    Args:
        prompt: "I want to generate synthetic training data for an AI assistant ..."
    Returns:
        {
            "messages": [
                {"role": "user", "content": "yo, what's new. just stumbled on this repo..."},
                {"role": "assistant", "content": "Welcome! You are looking at Tigger Chat, a minimal ..."
                ...
            ]
        }
    """
    payload = {
        "model": model,
        "stream": False,
        "response_format": response_format,
        "temperature": 1.0,
        "messages": [{
            "role": "user",
            "content": prompt
        }]
    }
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    response = requests.post(url, headers=headers, json=payload, timeout=120)
    response.raise_for_status()
    result = response.json()
    content = json.loads(result['choices'][0]['message']['content'])
    return content

def sample_name(rng):
    names = ["Tigger Chat", "Tigger", "tigger chat", "tigger", "TiggerChat"]
    names_typo = ["tiger", "Tiger Chat", "tiger chat", "tiggr", "tigger chatt", "tigerchat", "gitter", "giter", "tigerr"]
    return rng.choice(names) if rng.random() < 0.8 else rng.choice(names_typo)  # occasionally return a typo version

def generate_synthetic_conversation(idx, knowledge, openrouter_model_name):
    rng = random.Random(idx)

    category_idx = idx % len(topics)  # deterministic category, to ensure consistent coverage (makes any eval split have samples evenly distributed as well)
    category_name = list(topics.keys())[category_idx]
    topic_idx = rng.randint(0, len(topics[category_name]) - 1)
    persona_idx = rng.randint(0, len(personas) - 1)
    dynamic_idx = rng.randint(0, len(dynamics) - 1)
    style_idx = rng.randint(0, len(user_styles) - 1)
    writing_idx = rng.randint(0, len(user_writing) - 1)

    # Sample random topic, persona, etc.
    topic = topics[category_name][topic_idx].replace("{name}", "Tigger Chat")   # topic, personas, dynamics are instructions to the LLM generating conversations,
    persona = personas[persona_idx].replace("{name}", "Tigger Chat")            # so we always use the canonical name "Tigger Chat"
    dynamic = dynamics[dynamic_idx].replace("{name}", "Tigger Chat")
    user_style = user_styles[style_idx]
    writing_style = user_writing[writing_idx]

    first_msg_examples_list = []
    first_msg_categories = rng.sample(list(first_messages.keys()), 3)  # ['simple_greetings', 'greetings_with_name', 'curious_openers']
    for cat in first_msg_categories:
        msg = rng.choice(first_messages[cat])
        msg = msg.replace("{name}", sample_name(rng))   # replace placeholder "{name}" with a sampled name, like "Tigger" or "tigger chat"
        first_msg_examples_list.append(msg)  # ['hi', 'guten tag', 'HEYOOOO']
    first_msg_examples = "\n".join(f"- {msg}" for msg in first_msg_examples_list)   # as multiline bullet string

    # Build the prompt
    prompt = prompt_template.format(
        knowledge=knowledge,
        topic=topic,
        persona=persona,
        dynamic=dynamic,
        user_style=user_style,
        user_writing=writing_style,
        first_message_examples=first_msg_examples,
    )
    # Hit the API
    content = query_openrouter_api(prompt, openrouter_model_name, retries=3)
    messages = content["messages"]
    messages = [m for m in messages if m["role"] in ("user", "assistant") and m["content"].strip()]     # fix Sonnet 5 quicks producing empty or irrelevant messages

    # Validate
    if len(messages) < 2:
        raise ValueError(f"Message should have at least 2 messages: {messages}")
    for i, msg in enumerate(messages):
        expected_role = "user" if i % 2 == 0 else "assistant"
        if msg["role"] != expected_role:
            raise ValueError(f"Expected role {expected_role}: {messages}")
        if msg["content"].strip() == "":
            raise ValueError(f"Message content should not be empty: {messages}")
    if messages[-1]["role"] != "assistant":
        raise ValueError(f"Last message should be from assistant: {messages}")

    return messages

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic conversation data")
    parser.add_argument("--identity", type=str, required=True, help="Filepath to IDENTITY.md file, see generate_sft_data.py docstring for instructions.")
    parser.add_argument("--num", type=int, default=2105, help="Number of conversations to generate (5%% of 2105 is ~105, so we get clean 2000 train / 105 eval)")
    parser.add_argument("--test-fraction", type=float, default=0.05, help="Target fraction of conversations to use as test set. Real fraction may vary slightly due to errors.")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--output", type=str, default="identity_conversations.jsonl", help="Output JSONL file path")
    parser.add_argument("--model", type=str, default="openai/gpt-6-luna", help="OpenRouter model name")
    args = parser.parse_args()

    if not (0 <= args.test_fraction <= 1):
        print(f"Invalid test fraction: {args.test_fraction}. Must be between 0 and 1.")
        sys.exit(1)

    # check if identity file exists
    identity_path = Path(args.identity)
    if not identity_path.is_file():
        raise FileNotFoundError(f"Identity file not found: {args.identity}")

    # Generate IDENTITY.md as per comments at the top of this file
    identity_text = identity_path.read_text(encoding="utf-8").strip()

    def safe_generate(idx):
        """Ensure no exceptions raised inside ThreadPoolExecutor"""
        for attempt in range(5):  # retry up to 5 times
            try:
                return generate_synthetic_conversation(idx, identity_text, args.model)
            except Exception as e:
                print(f"Error generating conversation at index {idx}, attempt {attempt + 1}: {e}")
                time.sleep(2 ** attempt)  # exponential backoff before retrying
        return None

    num_written = 0
    partial_filepath = args.output + ".partial"
    with open(partial_filepath, "w", encoding="utf-8") as f:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            for i, messages in enumerate(executor.map(safe_generate, range(args.num))):      # collect in order
                if messages is not None:                     # skip errors
                    f.write(json.dumps(messages) + "\n")
                    num_written += 1
                if i % 10 == 0:
                    print(f"Progress {i}/{args.num}")

    # Calculate num test examples
    num_test = int(num_written * args.test_fraction)
    num_train = num_written - num_test
    print(f"All done:")
    print(f"  total generated: {num_written}")
    print(f"  training examples: {num_train}")
    print(f"  test examples: {num_test}")
    print(f"  test fraction: {num_test / num_written if num_written > 0 else 0}")

    with open(args.output, "w", encoding="utf-8") as out, open(partial_filepath, "r", encoding="utf-8") as f:
        out.write(json.dumps({
            "num_train": num_train,
            "num_test": num_test,
            "identity": identity_text}) + "\n"   # Preserve full IDENTITY.md so during eval LLM judge can use it as reference
        )
        out.writelines(f.readlines())
    # Remove partial file
    os.remove(partial_filepath)
    print("Final output written to:", args.output)

if __name__ == "__main__":
    main()
