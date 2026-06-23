# A Modular RAG System for Reading Research Papers

## Problem

As a non-native speaker, I always think reading research papers in English is slow and inefficient, especially for newcomers to a
field. Large language models can help, but they have two limits: their training data has a
cutoff date, so they don't know the newest papers, and by default they answer from general
knowledge rather than from the specific papers I care about. This project builds a
Retrieval-Augmented Generation (RAG) system that lets me upload papers and then ask
questions, or get detailed explanations of passages, with answers grounded strictly in the
uploaded content.

## Scope (First Version)

To keep the first version achievable, it is deliberately narrow:

- Single user, not a multi-user platform.
- A focused knowledge base of papers in two areas: machine learning and Web3.
- Two core functions: question answering over the papers, and explaining a selected
  passage or concept using the context of the whole paper.

## Approach (First Version)

The first version follows a text-retrieval RAG pipeline, where retrieval is done with text
but generation is done with the original page image:

1. Convert each paper (PDF) into pages, and produce a text representation of each page.
   For visual elements such as figures, a multimodal model generates a text description so
   that they, too, become searchable as text.
2. Split the text into chunks and turn each chunk into a vector (embedding) using a
   text-embedding model.
3. Store the vectors in a vector database.
4. At query time, turn the question into a vector and retrieve the most relevant pages
   using the text representations.
5. Pass the original page images of the retrieved pages (not the possibly-lossy text) to a
   multimodal large language model to generate a grounded, cited answer.

The key idea is to separate retrieval from generation: text is used to *find* the right
pages, while the original images are used to *answer*, which avoids letting any text-
extraction error affect the final answer.

## Modular Design and Future Extension

The system is built around one abstraction: a **retriever interface**. The upper layers
only ask the retriever "given this query, return the relevant information"; they do not
care how the retrieval happens. This keeps the system modular and extensible:

- The first version plugs in a single text-based retriever (the approach above).
- A future version can add a second retriever based on direct visual document retrieval
  (e.g. ColPali / ColQwen-style models) for figures and formulas, without changing the
  upper layers.
- When more than one retriever is present, their results are combined with a rank-based
  fusion method (e.g. Reciprocal Rank Fusion) into a single ranked list — a hybrid
  retrieval setup.

The long-term vision is a general, modular paper-research assistant, but each version takes
only one concrete step toward it.

## Tools (Planned)

- PDF processing: pdf2image.
- Image description and answer generation: a multimodal large language model.
- Embeddings and vector store: a text-embedding model with a vector database.
- Orchestration framework: LangChain.

(This is just a plan, the later real implementation may be different.)

