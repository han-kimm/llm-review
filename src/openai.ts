import { getInput } from '@actions/core'
import { ChatOpenAI, OpenAIEmbeddings } from '@langchain/openai'

const apiKey = getInput('LLM_API_KEY')
const chatModel = getInput('LLM_API_CHAT')
const embedModel = getInput('LLM_API_EMBEDDING')

export const chatOpenai = new ChatOpenAI({
  apiKey,
  modelName: chatModel,
  temperature: 0
})

export const embedOpenai = new OpenAIEmbeddings({
  apiKey,
  modelName: embedModel,
  dimensions: embedModel === 'text-embedding-3-large' ? 1024 : 512
})
