# 🏗️ Architecture Overview

## 🌐 Data Flow Diagram

```mermaid
graph TD
    A[Query Engine] --> B[Multi-Source Crawlers]
    B --> C[Raw Data Storage]
    C --> D[Data Preprocessing]
    D --> E[LLM Extractor]
    E --> F[Structured Database]
    F --> G[Dashboard & Visualization]

    subgraph Sources
        B1[OpenAlex]
        B2[Crossref]
        B3[PubMed]
        B4[arXiv]
        B5[DOAJ]
        B6[Europe PMC]
    end

    B --> B1
    B --> B2
    B --> B3
    B --> B4
    B --> B5
    B --> B6

    subgraph Processing
        D1[Abstract Extraction]
        D2[Full Text Download]
        D3[Deduplication]
        D4[Clustering]
    end

    D --> D1
    D --> D2
    D --> D3
    D --> D4

    subgraph LLM Selection
        E1[Golden Set Benchmarking]
        E2[GPT-4o / Claude / Gemini / Mistral]
        E3[Select Best LLM]
    end

    E --> E1
    E --> E2
    E --> E3

    subgraph Output
        F1[CSV Table]
        F2[SQLite DB]
        F3[Cluster IDs]
    end

    F --> F1
    F --> F2
    F --> F3

    subgraph Visualization
        G1[Streamlit Dashboard]
        G2[Scatter Plot by Year]
        G3[Heatmap Q1-Q9]
        G4[Cluster Network]
    end

    G --> G1
    G --> G2
    G --> G3
    G --> G4
