# Multimodal PDF RAG (Local)

This project implements a simple, fully local Retrieval-Augmented Generation (RAG) pipeline for technical PDF documents. It supports text, tables, and diagrams extracted page-wise and is designed to run efficiently with a local 8B LLM without GPU contention.

## Pipeline Overview

- Extract PDF content page-wise into text files
- Separate content into RAW TEXT, TABLES, and DIAGRAM CONTEXT
- Chunk content while preserving page and modality
- Embed chunks on CPU
- Store embeddings in a FAISS index
- Retrieve top-k chunks for a query
- Answer using a local LLM via Ollama (GPU only for inference)

## Project Structure

- `data/pdf_name/text/` — page-wise context files (page_1.txt, page_2.txt, ...)
- `rag_store/index.faiss` — FAISS vector index
- `rag_store/chunks.json` — chunk metadata and text
- `build_rag_index.py` — builds the vector index
- `rag_query.py` — runs RAG queries using a local LLM

## Setup

- Install Ollama system-wide
- Pull a local LLaMA 8B model
- Create a Python environment and install dependencies using requirements.txt
- Build the RAG index
- Run queries

## Design Notes

Retrieval runs entirely on CPU to avoid GPU memory conflicts. Diagram understanding is handled during ingestion using a VLM, so query-time reasoning is text-only and efficient.