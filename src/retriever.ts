import { PromptTemplate } from '@langchain/core/prompts'
import { PineconeStore } from '@langchain/pinecone'
import { Pinecone } from '@pinecone-database/pinecone'
import { MultiQueryRetriever } from 'langchain/retrievers/multi_query'
import { chatOpenai, embedOpenai } from './openai.ts'
import { getInput } from '@actions/core'

const apiKey = getInput('PINECONE_API_KEY')
const indexName = getInput('PINECONE_INDEX_NAME')
const namespace = getInput('PINECONE_NAMESPACE')
const queryCount = getInput('QUERY_COUNT') ?? '5'

export async function confluenceMultiqueryRetriever(query: string) {
  const pc = new Pinecone({
    apiKey
  })

  const pineconeIndex = pc.index(indexName)

  const vertorStore = await PineconeStore.fromExistingIndex(embedOpenai, {
    pineconeIndex,
    namespace
  })

  const retriever = vertorStore.asRetriever({
    k: 1
  })

  const multiqueryRetriever = MultiQueryRetriever.fromLLM({
    retriever,
    llm: chatOpenai,
    queryCount: Number(queryCount),
    prompt: new PromptTemplate({
      inputVariables: ['question', 'queryCount'],
      template: `
You are a good AI developer. Answer must always be based on given context. you can't lie.
Generate {queryCount} phrase which summarize the given context. the phrases will be used for searching relevant documents in company's database.

Provide these alternative phrases separated by newlines between XML tags of 'questions'. For example:

<questions>
phrase 1
phrase 2
phrase 3
</questions>

context:{question}`
    })
  })

  return multiqueryRetriever.invoke(query)
}
