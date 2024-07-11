# Check environment variables

1. LLM_API_KEY / openAI api key.

2. LLM_API_CHAT / model name of inference with. "gpt-3.5-turbo" | "gpt-4o"

3. LLM_API_EMBEDDING / model name of embedding. "text-embedding-3-small" |
   "text-embedding-3-large"

4. PINECONE_API_KEY / api key of Pinecone.

5. PINECONE_INDEX_NAME / index name of pinecone.

6. PINECONE_NAMESPACE / namespace of the index(4)

7. (optional) QUERY_COUNT / how much generate queries to search documents in
   Pinecone. one query get a one document. and redundant documents are replaced
   by one unique document.

default : 5
