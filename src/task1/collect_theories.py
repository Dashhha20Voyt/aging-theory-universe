# src/task1/collect_theories.py

import aiohttp
import asyncio
from tqdm.asyncio import tqdm_asyncio
import requests
from bs4 import BeautifulSoup
from xml.etree import ElementTree as ET
import time
import random
import pandas as pd

# --- CONFIG ---
OPENALEX_API_URL = "https://api.openalex.org/works"
CROSSREF_API_URL = "https://api.crossref.org/works"
PUBMED_API_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
ARXIV_API_URL = "http://export.arxiv.org/api/query"

# --- FUNCTIONS ---

async def fetch_openalex_papers(session, query, per_page=200):
    """Fetch papers from OpenAlex"""
    papers = []
    cursor = "*"
    
    for page in range(5):  # Max 5 pages → 1000 papers
        url = f"{OPENALEX_API_URL}?search={query}&per-page={per_page}&cursor={cursor}"
        try:
            async with session.get(url) as response:
                if response.status == 429:
                    delay = 2 ** page + random.uniform(0, 1)
                    print(f"⏳ OpenAlex 429 — waiting {delay:.1f} sec...")
                    await asyncio.sleep(delay)
                    continue
                data = await response.json()
                
                for item in data.get('results', []):
                    title = item.get('title', '')
                    year = item.get('publication_year', None)
                    abstract = item.get('abstract', '')
                    doi = item.get('doi', '').replace('https://doi.org/', '') if item.get('doi') else ''
                    authors = [f"{a.get('author', {}).get('display_name', '')}" for a in item.get('authorships', [])] if item.get('authorships') else []
                    
                    papers.append({
                        'source': 'OpenAlex',
                        'paper_name': title,
                        'paper_url': item.get('doi', ''),
                        'paper_year': year,
                        'abstract': abstract,
                        'doi': doi,
                        'authors': authors
                    })
                
                cursor = data.get('meta', {}).get('cursor', '*')
                if cursor == "*":
                    break
                
                await asyncio.sleep(0.5)
        except Exception as e:
            print(f"❌ OpenAlex error for '{query}': {e}")
            break
    
    return papers

async def fetch_crossref_papers_paginated(session, query, total=500):
    """Fetch papers from Crossref with pagination"""
    papers = []
    for start in range(0, total, 100):
        url = f"{CROSSREF_API_URL}?query={query}&rows=100&offset={start}"
        for attempt in range(3):
            try:
                async with session.get(url) as response:
                    if response.status == 429:
                        delay = 2 ** attempt + random.uniform(0, 1)
                        print(f"⏳ Crossref 429 — waiting {delay:.1f} sec...")
                        await asyncio.sleep(delay)
                        continue
                    data = await response.json()
                    for item in data['message']['items']:
                        paper = {
                            'source': 'Crossref',
                            'paper_name': item.get('title', [''])[0],
                            'paper_url': item.get('URL', ''),
                            'paper_year': item.get('published-print', {}).get('date-parts', [[None]])[0][0],
                            'abstract': item.get('abstract'),  # Can be None
                            'doi': item.get('DOI', ''),
                            'authors': [f"{a.get('given', '')} {a.get('family', '')}" for a in item.get('author', [])]
                        }
                        papers.append(paper)
                    break
            except Exception as e:
                if attempt < 2:
                    delay = 2 ** attempt + random.uniform(0, 1)
                    print(f"⚠️ Crossref error for '{query}' (attempt {attempt+1}): {e}. Waiting {delay:.1f} sec...")
                    await asyncio.sleep(delay)
                else:
                    print(f"❌ Crossref failed for '{query}' after 3 attempts.")
                    break
        await asyncio.sleep(0.5)
    return papers

# ... аналогично для PubMed, arXiv, DOAJ, Europe PMC (можно добавить позже)

# --- MAIN FUNCTION ---

async def main():
    base_queries = [
        "aging theory",
        "biological aging mechanisms",
        "evolutionary theory of aging",
        "information theory of aging",
        "programmed aging theory",
        "wear and tear theory",
        "free radical theory of aging",
        "telomere shortening theory",
        "senescence theory",
        "antagonistic pleiotropy",
        "disposable soma theory",
        "quantum biology aging",
        "social theories of aging",
        "philosophical theories of aging",
        "systems biology of aging",
        "epigenetic clock theory",
        "inflammaging",
        "hallmarks of aging",
        "geroscience",
        "longevity interventions"
    ]
    
    all_papers = []
    
    async with aiohttp.ClientSession() as session:
        tasks = []
        for q in base_queries:
            tasks.append(fetch_openalex_papers(session, q, per_page=200))
            tasks.append(fetch_crossref_papers_paginated(session, q, total=200))
            # Добавь другие источники: fetch_pubmed, fetch_arxiv и т.д.
        
        results = await tqdm_asyncio.gather(*tasks, desc="🚀 Collecting papers", unit="query")
        
        for result in results:
            all_papers.extend(result)
    
    df = pd.DataFrame(all_papers)
    df = df.drop_duplicates(subset=['paper_url', 'paper_name'], keep='first').reset_index(drop=True)
    df.to_csv('data/processed/collected_papers.csv', index=False)
    print(f"✅ Collected {len(df)} unique papers")

if __name__ == "__main__":
    asyncio.run(main())
