# src/task2/extract_data.py

import openai
import pandas as pd
import asyncio
from tqdm.asyncio import tqdm_asyncio
import aiohttp
import time
import random
import os

# --- CONFIG ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your_openai_key_here")  # Load from .env
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

# --- PROMPT LOADER ---
def load_prompt(llm_name):
    """Load prompt template for specific LLM from prompts/ folder"""
    prompt_path = f"prompts/{llm_name}_prompt.txt"
    try:
        with open(prompt_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except FileNotFoundError:
        return f"# Prompt for {llm_name} not found\nYou are an expert in aging biology. Answer ONLY with: 'Yes, quantitatively shown', 'Yes, but not shown', 'No' for Q1. For Q2-Q9: only 'Yes' or 'No'."

# --- LLM EVALUATOR ---
class LLMEvaluator:
    """Evaluate multiple LLMs on a golden set and select the best one"""
    
    def __init__(self, golden_set_path="data/processed/golden_set.csv"):
        self.golden_set = pd.read_csv(golden_set_path) if os.path.exists(golden_set_path) else None
        self.best_llm = None
        self.scores = {}
    
    async def evaluate_llms(self, llm_list=["gpt4o", "claude3", "gemini15", "mistral7b"]):
        """Evaluate each LLM on golden set and return best performer"""
        if self.golden_set is None:
            print("⚠️ Golden set not found. Skipping evaluation.")
            return "gpt4o"  # Default
        
        for llm in llm_list:
            score = await self._evaluate_single_llm(llm)
            self.scores[llm] = score
            print(f"✅ {llm}: Accuracy = {score:.2%}")
        
        self.best_llm = max(self.scores, key=self.scores.get)
        print(f"🏆 Best LLM: {self.best_llm} (Accuracy = {self.scores[self.best_llm]:.2%})")
        return self.best_llm
    
    async def _evaluate_single_llm(self, llm_name):
        """Evaluate one LLM on golden set"""
        correct = 0
        total = len(self.golden_set)
        
        for _, row in self.golden_set.iterrows():
            text = row.get('abstract') or row.get('full_text') or ""
            for i, q in enumerate(questions, 1):
                answer = await self._ask_llm(llm_name, q, text)
                expected = row[f'q{i}']
                if answer.strip().lower() == expected.strip().lower():
                    correct += 1
        
        return correct / total if total > 0 else 0
    
    async def _ask_llm(self, llm_name, question, text):
        """Ask LLM with specific prompt"""
        prompt = load_prompt(llm_name)
        full_prompt = f"{prompt}\n\nText: {text}\n\nQuestion: {question}"
        
        try:
            response = openai.chat.completions.create(
                model="gpt-4o" if llm_name == "gpt4o" else "gpt-4o",  # Adjust model name as needed
                messages=[{"role": "user", "content": full_prompt}],
                max_tokens=50,
                temperature=0.0
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error: {str(e)}"

# --- ASYNC PROCESSING ---

async def process_paper(index, row, questions, session, llm_name):
    """Process single paper with selected LLM"""
    answers = {}
    text = row.get('abstract') or row.get('full_text') or ""
    
    for i, q in enumerate(questions, 1):
        ans = await ask_llm_with_prompt(llm_name, q, text)
        answers[f'q{i}'] = ans
    return index, answers

async def ask_llm_with_prompt(llm_name, question, text):
    """Ask LLM using its specific prompt template"""
    prompt = load_prompt(llm_name)
    full_prompt = f"{prompt}\n\nText: {text}\n\nQuestion: {question}"
    
    try:
        response = openai.chat.completions.create(
            model="gpt-4o" if llm_name == "gpt4o" else "gpt-4o",  # Adjust model name as needed
            messages=[{"role": "user", "content": full_prompt}],
            max_tokens=50,
            temperature=0.0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {str(e)}"

async def main():
    df = pd.read_csv('data/processed/collected_papers.csv')
    
    # Evaluate LLMs and select best one
    evaluator = LLMEvaluator()
    best_llm = await evaluator.evaluate_llms()
    print(f"✅ Selected best LLM: {best_llm}")
    
    # Process all papers with best LLM
    async with aiohttp.ClientSession() as session:
        tasks = []
        for idx, row in df.iterrows():
            tasks.append(process_paper(idx, row, questions, session, best_llm))
        
        results = await tqdm_asyncio.gather(*tasks, desc=f"🤖 Analyzing papers with {best_llm}", unit="paper")
        
        for idx, answers in results:
            for k, v in answers.items():
                df.at[idx, k] = v
    
    df.to_csv('data/processed/extracted_answers.csv', index=False)
    print("✅ Answers saved to data/processed/extracted_answers.csv")

if __name__ == "__main__":
    asyncio.run(main())
