# src/task2/extract_data.py

import openai
import pandas as pd
import asyncio
from tqdm.asyncio import tqdm_asyncio
import aiohttp
import time
import random

# --- CONFIG ---
OPENAI_API_KEY = "your_openai_key_here"  # Замени на свой ключ
openai.api_key = OPENAI_API_KEY

# --- QUESTIONS ---
questions = [
    "Does it suggest an aging biomarker (measurable entity reflecting aging pace or health state, associated with mortality or age-related conditions)?",
    "Does it suggest a molecular mechanism of aging?",
    "Does it suggest a longevity intervention to test?",
    "Does it claim that aging cannot be reversed?",
    "Does it suggest a biomarker that predicts maximal lifespan differences between species?",
    "Does it explain why the naked mole rat can live 40+ years despite its small size?",
    "Does it explain why birds live much longer than mammals on average?",
    "Does it explain why large animals live longer than small ones?",
    "Does it explain why calorie restriction increases the lifespan of vertebrates?"
]

# --- LLM EVALUATOR (SIMPLIFIED) ---
def ask_chatgpt(question, text):
    """Ask GPT-4o with standardized prompt"""
    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert in aging biology. Answer ONLY with: 'Yes, quantitatively shown', 'Yes, but not shown', 'No' for Q1. For Q2-Q9: only 'Yes' or 'No'."},
                {"role": "user", "content": f"Text: {text}\n\nQuestion: {question}"}
            ],
            max_tokens=50,
            temperature=0.0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {str(e)}"

# --- ASYNC PROCESSING ---

async def process_paper(index, row, questions, session):
    answers = {}
    text = row.get('abstract') or row.get('full_text') or ""
    
    for i, q in enumerate(questions, 1):
        ans = await asyncio.to_thread(ask_chatgpt, q, text)
        answers[f'q{i}'] = ans
    return index, answers

async def main():
    df = pd.read_csv('data/processed/collected_papers.csv')
    
    async with aiohttp.ClientSession() as session:
        tasks = []
        for idx, row in df.iterrows():
            tasks.append(process_paper(idx, row, questions, session))
        
        results = await tqdm_asyncio.gather(*tasks, desc="🤖 Analyzing papers", unit="paper")
        
        for idx, answers in results:
            for k, v in answers.items():
                df.at[idx, k] = v
    
    df.to_csv('data/processed/extracted_answers.csv', index=False)
    print("✅ Answers saved to data/processed/extracted_answers.csv")

if __name__ == "__main__":
    asyncio.run(main())
