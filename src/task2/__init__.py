# src/task2/extract_data_ultra_fast.py

import openai
import pandas as pd
import asyncio
from tqdm.asyncio import tqdm_asyncio
import aiohttp
import time
import random
import os

# --- CONFIG ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your_openai_key_here")
openai.api_key = OPENAI_API_KEY

# --- QUESTIONS ---
questions = [
    "Does it suggest an aging biomarker?",
    "Does it suggest a molecular mechanism of aging?",
    "Does it suggest a longevity intervention to test?",
    "Does it claim that aging cannot be reversed?",
    "Does it suggest a biomarker that predicts maximal lifespan differences between species?",
    "Does it explain why the naked mole rat can live 40+ years despite its small size?",
    "Does it explain why birds live much longer than mammals on average?",
    "Does it explain why large animals live longer than small ones?",
    "Does it explain why calorie restriction increases the lifespan of vertebrates?"
]

# --- PROMPT TEMPLATE ---
def create_batch_prompt(abstracts):
    """Create one prompt for 10 papers, 3 questions each"""
    prompts = []
    for i, abstract in enumerate(abstracts, 1):
        prompts.append(f"Paper {i}:\nAbstract: {abstract}\nQ1: {questions[0]}\nQ2: {questions[1]}\nQ3: {questions[2]}")
    
    return "\n\n".join(prompts) + "\n\nAnswer format:\nPaper 1: Q1: Yes/No/Yes, quantitatively shown; Q2: Yes/No; Q3: Yes/No\nPaper 2: ..."

def parse_batch_response(response, batch_size):
    """Parse response into answers for each paper"""
    lines = response.split("\n")
    results = []
    for i in range(batch_size):
        line = lines[i] if i < len(lines) else ""
        parts = line.split("; ")
        answers = {}
        for part in parts:
            if ":" in part:
                key, val = part.split(":", 1)
                answers[key.strip()] = val.strip()
        results.append(answers)
    return results

# --- ASYNC BATCH PROCESSING ---
async def process_batch(session, batch):
    """Process 10 papers at once with ChatGPT"""
    abstracts = [paper.get('abstract') or (paper.get('full_text') or "")[:300] for paper in batch]
    prompt = create_batch_prompt(abstracts)
    
    try:
        async with session.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "gpt-4o-mini",  # ✅ Быстрее и дешевле
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 300,
                "temperature": 0.0
            }
        ) as response:
            if response.status == 429:
                delay = 2 + random.uniform(0, 1)
                print(f"⏳ ChatGPT 429 — waiting {delay:.1f} sec...")
                await asyncio.sleep(delay)
                return None
            
            data = await response.json()
            content = data['choices'][0]['message']['content']
            return parse_batch_response(content, len(batch))
    
    except Exception as e:
        print(f"❌ ChatGPT error: {e}")
        return None

# --- MAIN FUNCTION ---
async def main():
    df = pd.read_csv('data/processed/collected_papers.csv')
    df['theory_id'] = range(1, len(df) + 1)
    
    # Split into batches of 10
    batches = [df[i:i+10].to_dict('records') for i in range(0, len(df), 10)]
    
    # Split batches into 20 groups for parallel processing
    group_size = len(batches) // 20 + 1
    groups = [batches[i:i+group_size] for i in range(0, len(batches), group_size)]
    
    async def process_group(group):
        async with aiohttp.ClientSession() as session:
            tasks = [process_batch(session, batch) for batch in group]
            results = await tqdm_asyncio.gather(*tasks, desc="🤖 Group", unit="batch")
            return results
    
    # Process 20 groups in parallel
    tasks = [process_group(group) for group in groups]
    all_results = await tqdm_asyncio.gather(*tasks, desc="🚀 All Groups", unit="group")
    
    # Combine results
    final_results = []
    for group_results in all_results:
        final_results.extend(group_results)
    
    # Apply results to DataFrame
    for i, batch_result in enumerate(final_results):
        if batch_result is not None:
            start_idx = i * 10
            for j, answers in enumerate(batch_result):
                idx = start_idx + j
                for k, v in answers.items():
                    df.at[idx, k] = v
    
    df.to_csv('data/processed/extracted_answers.csv', index=False)
    print("✅ Answers saved to data/processed/extracted_answers.csv")

if __name__ == "__main__":
    asyncio.run(main())
