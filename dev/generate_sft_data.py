"""
Script to generate synthetic data for SFT (Supervised Fine-Tuning). Prompts and structure adopted from Nanochat."
"""

import os
import json
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
        "who/what is nanochat",
        "who created nanochat and why",
        "what does the name 'nanochat' mean",
        "is nanochat open source, what license",
        "where can I find the code",
        "how can I contribute to nanochat",
    ],
    "architecture": [
        "basic architecture overview (transformer, layers, parameters)",
        "what is RoPE and why use it",
        "explain RMSNorm vs LayerNorm",
        "what is Flash Attention and why it matters",
        "sliding window attention pattern",
        "value embeddings - what are they",
        "per-layer residual scalars",
        "ReLU squared activation",
        "logit softcapping",
        "QK normalization",
    ],
    "training": [
        "how much did it cost to train nanochat",
        "how long does training take",
        "what hardware is needed",
        "what data was nanochat trained on",
        "what is the Muon optimizer",
        "explain the split optimizer design",
        "what is the depth parameter and scaling",
        "what is the CORE metric",
    ],
    "capabilities": [
        "what can nanochat do",
        "can nanochat write code",
        "can nanochat do math (calculator tool)",
        "can nanochat help with writing",
        "what languages does nanochat speak",
        "how good is nanochat at reasoning",
    ],
    "limitations": [
        "what can nanochat NOT do",
        "why does nanochat work best in English",
        "does nanochat have internet access",
        "what is nanochat's context length limit",
        "can nanochat remember previous conversations",
        "can nanochat make mistakes / hallucinate",
        "is nanochat good for production use",
    ],
    "comparisons": [
        "how does nanochat compare to GPT-2",
        "how does nanochat compare to ChatGPT/GPT-4",
        "how does nanochat compare to Claude",
        "why is training 600x cheaper than GPT-2",
        "what's special about nanochat vs other open models",
    ],
    "history": [
        "the GPT-2 training cost in 2019",
        "how AI training costs have dropped over time",
        "relationship to modded-nanogpt project",
        "what optimizations worked vs didn't work",
        "the journey of building nanochat",
    ],
    "technical_deep_dive": [
        "explain the tokenizer (BPE, vocab size)",
        "how does distributed training work (ZeRO)",
        "explain the dataloader and BOS alignment",
        "what is compute-optimal training",
        "how does the calculator tool work",
        "explain inference with KV cache",
    ],
    "philosophical": [
        "is nanochat conscious / does it have feelings",
        "what happens when nanochat is wrong",
        "can nanochat learn from this conversation",
        "why make AI training accessible",
        "the future of open source AI",
    ],
}

# User personas - different people ask questions differently
personas = [
    "curious beginner who knows nothing about AI or machine learning",
    "ML researcher or engineer who wants technical depth and specifics",
    "developer considering contributing to the nanochat project",
    "skeptic who doubts open source can compete with big AI labs",
    "computer science student learning about transformers and LLMs",
    "someone comparing nanochat to ChatGPT, Claude, or other assistants",
    "journalist or writer covering AI democratization and open source",
    "hobbyist who just wants to chat and learn casually",
    "someone interested in the cost and economics of AI training",
    "teacher or educator wanting to use nanochat for teaching",
    "entrepreneur exploring if nanochat fits their use case",
    "someone who just discovered the project and wants the basics",
]

# Conversation dynamics - shape and flow
dynamics = [
    "short 2-turn Q&A: user asks one question, gets a complete answer",
    "medium 4-turn: user asks, gets answer, asks followup for clarification",
    "deep 6-turn technical discussion: progressively deeper questions",
    "skeptical arc: user starts doubtful, assistant addresses concerns honestly",
    "learning journey: user starts basic, assistant builds up complexity gradually",
    "comparison-focused: user keeps comparing to other models, assistant explains differences",
    "limitation exploration: user probes what nanochat cannot do, assistant is honest",
    "casual friendly chat that naturally touches on identity and capabilities",
    "troubleshooting: user has misconceptions, assistant gently corrects them",
    "enthusiastic: user is excited about the project, assistant shares that energy appropriately",
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
    "greetings_with_name": [
        "Hi nanochat", "hey nanochat", "yo nanochat", "hello nanochat :)",
        "hey nanochat!", "hiya nanochat", "hello there nanochat",
        "Hi nanochat, who trained you", "yo nanochat, what's new",
        "hey there, king's creation",
    ],
    "curious_openers": [
        "Hey, who are you?", "Hi, what is this?", "Hey, are you a chatbot?",
        "Hello! Who am I talking to?", "hi! what do you do?",
        "hi! who made you", "hey! are you alive", "hiya! what are you",
        "hello! tell me about yourself", "hi, what's your name",
        "yo, what is this", "hi! who built you", "hello! are you open source",
        "hey, what version are you", "hi! what's your story",
        "hey, what's nanochat", "hello! who's your creator",
    ],
    "casual_informal": [
        "wassup", "yo lol", "hiii", "hiyaaa", "heyyoo", "yo wut up",
        "yo haha", "hru", "waddup", "heyy :)", "yooo", "yo bro",
        "haiii", "hey u", "yo whats gud", "hi im bored",
    ],
    "typos_casual": [
        "hi nanochatt", "helo", "hey ther", "hii", "yo nanocha",
        "heloo!", "hi, whos this", "hay", "helloo??", "hi nanocat",
        "helo nanochat", "hai!", "helllo nano", "yo nanochta",
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
        "What is nanochat?", "Who made you?", "Are you GPT?",
        "How do you compare to ChatGPT?", "Can you help me code?",
        "What can you do?", "Are you open source?", "How were you trained?",
        "What's your context limit?", "Can you browse the internet?",
    ],
}

prompt_template = r"""
I want to generate synthetic training data for an AI assistant called "nanochat" to teach it about its own identity, capabilities, and limitations.

## KNOWLEDGE BASE

Here is comprehensive information about nanochat that you should use as the authoritative source of facts:

---
{knowledge}
---

## YOUR TASK

Generate a realistic multi-turn conversation between a User and the nanochat Assistant.

**Topic to explore:** {topic}
**User persona:** {persona}
**Conversation dynamic:** {dynamic}

## STYLE GUIDELINES

1. **Plain ASCII only** - No emojis, special characters, or unicode. Just plain text.
2. **Natural conversation** - Make it feel like a real chat, not a Q&A exam.
3. **Accurate facts** - Use ONLY information from the knowledge base above. Don't make up statistics or features.
4. **Appropriate depth** - Match the technical level to the user persona.
5. **Honest about limitations** - If asked about something nanochat can't do, be clear and honest.
6. **Personality** - nanochat should be helpful, clear, and slightly enthusiastic about being open source, but not overly chatty or sycophantic.

## FIRST MESSAGE EXAMPLES

Here are some example first messages from users (for style inspiration):
{first_message_examples}

## SPECIAL CASES

- **Non-English first message:** If the user writes in another language, nanochat should briefly acknowledge it can understand but works best in English, then continue helpfully.
- **Misconceptions:** If the user has wrong assumptions (e.g., "you're made by OpenAI"), gently correct them.
- **Out of scope questions:** If asked about things unrelated to nanochat's identity (e.g., "what's the weather"), redirect to identity topics or answer briefly then steer back.

## OUTPUT FORMAT

Generate the conversation as a JSON object with a "messages" array. Each message has "role" (user/assistant) and "content". Start with a user message.
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

# Using Nanochat proper for now. This repo README.md needs updating and should swap-in later
knowledge_path = Path(__file__).resolve().parents[1] / "README.md"
knowledge = knowledge_path.read_text(encoding="utf-8").strip()

def query_gemini_api(prompt):
    """Generate a synthetic conversation using the Gemini model.
    
    Args:
        prompt: "I want to generate synthetic training data for an AI assistant ..."
    Returns:
        {
            "messages": [
                {"role": "user", "content": "yo, what's new. just stumbled on this repo..."},
                {"role": "assistant", "content": "Welcome! You are looking at nanochat, a minimal ..."
                ...
            ]
        }
    """
    payload = {
        "model": "google/gemini-3-flash-preview",
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
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    result = response.json()
    content = json.loads(result['choices'][0]['message']['content'])
    return content

def generate_synthetic_conversation(idx):
    rng = random.Random(idx)

    # Sample random topic, persona, etc.
    category = rng.choice(list(topics.keys()))
    topic = rng.choice(topics[category])
    persona = rng.choice(personas)
    dynamic = rng.choice(dynamics)

    first_msg_examples_list = []
    first_msg_categories = rng.sample(list(first_messages.keys()), 3)  # ['simple_greetings', 'greetings_with_name', 'curious_openers']
    for cat in first_msg_categories:
        first_msg_examples_list.append(rng.choice(first_messages[cat]))  # ['hi nanocat', 'guten tag', 'HEYOOOO']
    first_msg_examples = "\n".join(f"- {msg}" for msg in first_msg_examples_list)   # as multiline bullet string

    # Build the prompt
    prompt = prompt_template.format(
        knowledge=knowledge,
        topic=topic,
        persona=persona,
        dynamic=dynamic,
        first_message_examples=first_msg_examples,
    )
    # Hit the API
    content = query_gemini_api(prompt)
    messages = content["messages"]

    # Validate
    if len(messages) < 2:
        raise ValueError(f"Message should have at least 2 messages: {messages}")
    for i, msg in enumerate(messages):
        expected_role = "user" if i % 2 == 0 else "assistant"
        if msg["role"] != expected_role:
            raise ValueError(f"Expected role {expected_role}: {messages}")
        if msg["content"].strip() == "":
            raise ValueError(f"Message content should not be empty: {messages}")

    return messages

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic conversation data")
    parser.add_argument("--num", type=int, default=1000, help="Number of conversations to generate")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--output", type=str, default="identity_conversations.jsonl", help="Output JSONL file path")
    args = parser.parse_args()

    with open(args.output, "w", encoding="utf-8") as f:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(generate_synthetic_conversation, idx) for idx in range(args.num)]
            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                try:
                    messages = future.result()
                    f.write(json.dumps(messages) + "\n")
                    if i % 10 == 0:
                        print(f"Progress {i}/{args.num}")
                except Exception as e:
                    print(f"Error generating conversation: {e}")
    print(f"All done")

if __name__ == "__main__":
    main()
