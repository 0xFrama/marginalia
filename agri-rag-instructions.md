# agri-rag — Project Instructions

## Chi sono e perché esiste questo progetto

Sono uno sviluppatore con un background da Data Engineer (2 anni) e un'esperienza recente come Full Stack Developer presso una startup. Durante il periodo in startup ho iniziato a esplorare la costruzione di un RAG (Retrieval-Augmented Generation) pipeline, ma il progetto non è arrivato a maturità tecnica completa.

Questo progetto nasce con due obiettivi paralleli:

1. **Apprendimento reale**: costruire da zero un sistema RAG completo e funzionante, padroneggiando ogni componente della stack
2. **Candidatura professionale**: il progetto è pensato per supportare una candidatura a una posizione di **Junior AI Developer** presso un centro di ricerca focalizzato su **IoT e Agritech** (progetto AgrifoodTEF, FBK). Il Profilo A richiede esplicitamente: LLM APIs, LangChain/LangGraph, RAG, prompt engineering, REST APIs, e data engineering practices.

Il dominio scelto — l'agricoltura di precisione — è intenzionalmente allineato con il contesto della posizione target.

---

## Cosa costruiamo

Un **agente conversazionale RAG** su documenti agricoli tecnici (manuali FAO, bollettini fitosanitari, schede colturali, linee guida agronomiche).

L'utente può:
- Caricare documenti PDF tramite API
- Fare domande in linguaggio naturale sul contenuto dei documenti
- Ricevere risposte contestualizzate con citazione delle fonti
- Mantenere una conversazione multi-turn con memoria

---

## Stack tecnologica

| Componente | Tecnologia | Motivo |
|---|---|---|
| Linguaggio | Python 3.11+ | — |
| LLM + Embeddings | OpenAI API (gpt-4o-mini, text-embedding-ada-002) | Familiarità pregressa |
| Orchestrazione RAG | LangChain | Requisito esplicito nel job description |
| Vector Store | Qdrant (locale via Docker) | Requisito esplicito nel job description |
| Backend / API | FastAPI | Esperienza pregressa, standard industry |
| Gestione documenti | PyPDF, LangChain document loaders | — |
| Configurazione | python-dotenv | — |
| Versionamento | Git + GitHub | — |

---

## Architettura generale

```
[PDF Upload]
     │
     ▼
[Document Loader]         ← PyPDF / LangChain
     │
     ▼
[Text Splitter]           ← RecursiveCharacterTextSplitter
     │                       chunk_size e overlap da giustificare
     ▼
[Embedding Model]         ← OpenAI text-embedding-ada-002
     │
     ▼
[Qdrant Vector Store]     ← storage locale via Docker
     │
     ▼
[Retriever]               ← similarity search
     │
     ▼
[ConversationalRetrievalChain]  ← LangChain, con chat history
     │
     ▼
[FastAPI REST Endpoints]
     ├── POST /upload        ← carica e indicizza un PDF
     └── POST /chat          ← domanda + chat history → risposta + fonti
```

---

## Fasi di sviluppo

### Fase 1 — Infrastruttura base
- [ ] Qdrant in esecuzione locale via Docker
- [ ] Connessione Python a Qdrant verificata
- [ ] Primo embedding generato e salvato su Qdrant
- [ ] Prima similarity search funzionante

### Fase 2 — Pipeline di ingestion
- [ ] Caricamento di un PDF con LangChain
- [ ] Chunking con RecursiveCharacterTextSplitter (sperimentare chunk_size e overlap)
- [ ] Embedding di tutti i chunk e salvataggio su Qdrant con metadata (nome file, pagina)
- [ ] Script di ingestion standalone e testabile

### Fase 3 — Pipeline di retrieval e conversazione
- [ ] Retriever configurato su Qdrant
- [ ] ConversationalRetrievalChain con gestione della chat history
- [ ] Gestione del caso in cui il retrieval non trova risultati rilevanti (fallback esplicito)
- [ ] Source attribution: ogni risposta cita il documento e la pagina di origine

### Fase 4 — API REST con FastAPI
- [ ] Endpoint POST /upload: riceve PDF, esegue ingestion, restituisce conferma
- [ ] Endpoint POST /chat: riceve domanda e chat history, restituisce risposta e fonti
- [ ] Gestione errori e validazione input (Pydantic)
- [ ] README con istruzioni per avviare il progetto

### Fase 5 — Qualità e rifinitura (se il tempo lo permette)
- [ ] Sperimentare con chunk_size diversi e confrontare qualità del retrieval
- [ ] Aggiungere reranking (CohereRerank o cross-encoder)
- [ ] Scrivere test per la pipeline di ingestion
- [ ] Documentare le scelte tecniche nel README (il perché di ogni decisione)

---

## Cosa devo saper spiegare al colloquio

Ogni componente deve essere giustificabile. Per ciascuno di questi punti devo avere una risposta chiara:

- **Chunking**: perché RecursiveCharacterTextSplitter? Perché quel chunk_size? Cosa succede se i chunk sono troppo grandi o troppo piccoli?
- **Embedding**: cosa fa text-embedding-ada-002? Cosa rappresenta un vettore embedding?
- **Qdrant**: cos'è un vector store? Come funziona la similarity search (cosine similarity)?
- **Chat history**: come gestisce LangChain la memoria conversazionale? Cosa succede al crescere della history?
- **Fallback**: cosa restituisce il sistema se nessun chunk è sufficientemente rilevante?
- **FastAPI**: come funziona la validazione con Pydantic? Come gestisci file upload?

---

## Narrativa per il colloquio

> "Durante il mio periodo in MAiTH ho iniziato a esplorare la costruzione di un RAG pipeline, ma il progetto non ha raggiunto una maturità tecnica completa. Ho preso quello che avevo imparato e ho costruito questo progetto personale da zero, scegliendo il dominio agricolo perché si allineava con le posizioni che stavo esplorando. L'obiettivo era avere qualcosa di completo e difendibile tecnicamente, non solo un tutorial seguito — ogni scelta architetturale ha un motivo che so spiegare."

---

## Note per un LLM che aiuta in questo progetto

- Il developer ha buona padronanza di Python e FastAPI
- Ha usato OpenAI API ma non ha esperienza profonda con LangChain
- Non ha mai usato Qdrant — iniziare dall'infrastruttura Docker e dalla connessione base
- Il codice deve essere scritto in modo che il developer lo capisca e lo sappia spiegare: preferire chiarezza a eleganza quando in conflitto
- Ogni scelta tecnica non ovvia deve essere commentata nel codice con il motivo
- Il progetto deve essere completabile in 2-3 settimane lavorando part-time
- GitHub repository pubblico — il codice deve essere presentabile
