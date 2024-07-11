import { getInput } from '@actions/core'
import { Octokit } from '@octokit/rest'
import { readFileSync } from 'fs'
import { minimatch } from 'minimatch'
import parseDiff, { Chunk, File } from 'parse-diff'
import { ChatOpenAI, OpenAIEmbeddings } from '@langchain/openai'
import { PromptTemplate } from '@langchain/core/prompts'
import { PineconeStore } from '@langchain/pinecone'
import { Pinecone } from '@pinecone-database/pinecone'
import { MultiQueryRetriever } from 'langchain/retrievers/multi_query'

getInput('LANGCHAIN_TRACING_V2')
getInput('LANGCHAIN_ENDPOINT')
getInput('LANGCHAIN_API_KEY')
getInput('LANGCHAIN_PROJECT')

const GITHUB_TOKEN = getInput('GITHUB_TOKEN')
const exclude = getInput('exclude') ?? ''
const openAIApiKey = getInput('LLM_API_KEY')
const chatModel = getInput('LLM_API_CHAT')
const embedModel = getInput('LLM_API_EMBEDDING')

const apiKey = getInput('PINECONE_API_KEY')
const indexName = getInput('PINECONE_INDEX_NAME')
const namespace = getInput('PINECONE_NAMESPACE')
const queryCount = getInput('QUERY_COUNT') ?? '5'

const octokit = new Octokit({ auth: GITHUB_TOKEN })

const SYSTEM_PROMPT = 'You are a strict and perfect code review AI.'

const chatOpenai = new ChatOpenAI({
  openAIApiKey,
  modelName: chatModel,
  temperature: 0
})

const embedOpenai = new OpenAIEmbeddings({
  openAIApiKey,
  modelName: embedModel,
  dimensions: embedModel === 'text-embedding-3-large' ? 1024 : 512
})

interface PRDetails {
  owner: string
  repo: string
  pull_number: number
  title: string
  description: string
}

async function getPRDetails(): Promise<PRDetails> {
  const { repository, number } = JSON.parse(
    readFileSync(process.env.GITHUB_EVENT_PATH || '', 'utf8')
  )
  const prResponse = await octokit.pulls.get({
    owner: repository.owner.login,
    repo: repository.name,
    pull_number: number
  })
  return {
    owner: repository.owner.login,
    repo: repository.name,
    pull_number: number,
    title: prResponse.data.title ?? '',
    description: prResponse.data.body ?? ''
  }
}

async function getDiff(
  owner: string,
  repo: string,
  pull_number: number
): Promise<string | null> {
  const response = await octokit.pulls.get({
    owner,
    repo,
    pull_number,
    mediaType: { format: 'diff' }
  })
  // @ts-expect-error - response.data is a string
  return response.data
}

async function analyzeCode(
  parsedDiff: File[],
  prDetails: PRDetails
): Promise<Array<{ body: string; path: string; line: number }>> {
  const comments: Array<{ body: string; path: string; line: number }> = []

  for (const file of parsedDiff) {
    if (file.to === '/dev/null') continue // Ignore deleted files
    for (const chunk of file.chunks) {
      const prompt = await createPrompt(file, chunk, prDetails)
      const aiResponse = await getAIResponse(prompt)
      if (aiResponse) {
        const newComments = createComment(file, chunk, aiResponse)
        if (newComments) {
          comments.push(...newComments)
        }
      }
    }
  }
  return comments
}

async function confluenceMultiqueryRetriever(query: string) {
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

async function triggerRag(file: File, chunk: Chunk, prDetails: PRDetails) {
  const query = `
  <pullRequestTitle>
  ${prDetails.title}
  </pullRequestTitle>
  <pullRequestDescription>
  ${prDetails.description}
  </pullRequestDescription>
  <fileName>
  ${file.to}
  </fileName>
  \`\`\`diff
  ${chunk.content}
  ${chunk.changes
    // @ts-expect-error - ln and ln2 exists where needed
    .map(c => `${c.ln ? c.ln : c.ln2} ${c.content}`)
    .join('\n')}
  \`\`\`
  `
  const relatedDocs = await confluenceMultiqueryRetriever(query)

  return relatedDocs.reduce(
    (acc, doc, index) =>
      acc +
      '\n' +
      `${index}. ` +
      doc.pageContent +
      '\n' +
      `related wiki: ${doc.metadata.url}\n`,
    ''
  )
}

async function createPrompt(
  file: File,
  chunk: Chunk,
  prDetails: PRDetails
): Promise<string> {
  const relatedDocs = await triggerRag(file, chunk, prDetails)
  console.log('relatedDocs:', relatedDocs)
  return `Your task is to review pull requests.
  Review Rules:
- Give answer in JSON format : {"reviews": [{"lineNumber":  <line_number>, "reviewComment": "<review comment>"}]}.
- Do not give positive comments or compliments.
- Provide comments and suggestions ONLY if there is something to improve, otherwise "reviews" should be an empty array.
- Write the comment in GitHub Markdown format.
- Give a comment first in English, and finally translate to Korean.
- If you refer a convention in <convention> writing comment, must leave 'related wiki url' with the comment.
- IMPORTANT: NEVER suggest adding comments to the code.

Review the following code diff in the file "${file.to}"
All answer must be based on given XML tags <title> and <description>, <convention>.
Only if there is no relevant information in <convention>, you can write comment as you know. but you can't lie.
  
<title>
${prDetails.title}
</title>
<descriptions>
${prDetails.description}
</descriptions>

<convention>
${relatedDocs}
</convention>

\`\`\`diff
${chunk.content}
${chunk.changes.map(c => `${'ln' in c ? c.ln : c.ln2} ${c.content}`).join('\n')}
\`\`\`

please ensure "lineNumber" is in diff.
`
}

async function getAIResponse(prompt: string): Promise<Array<{
  lineNumber: string
  reviewComment: string
}> | null> {
  try {
    const response = await chatOpenai.invoke(
      [
        ['system', SYSTEM_PROMPT],
        ['user', prompt]
      ],
      {
        response_format: {
          type: 'json_object'
        }
      }
    )

    const content = response.content
    if (typeof content === 'string') {
      return JSON.parse(content).reviews
    }
    return null
  } catch (error) {
    console.error('Error:', error)
    return null
  }
}

function createComment(
  file: File,
  chunk: Chunk,
  aiResponses: Array<{
    lineNumber: string
    reviewComment: string
  }>
): Array<{ body: string; path: string; line: number }> {
  return aiResponses.flatMap(aiResponse => {
    if (!file.to) {
      return []
    }
    return {
      body: aiResponse.reviewComment,
      path: file.to,
      line: Number(aiResponse.lineNumber)
    }
  })
}

async function createReviewComment(
  owner: string,
  repo: string,
  pull_number: number,
  comments: Array<{ body: string; path: string; line: number }>
): Promise<void> {
  await octokit.pulls.createReview({
    owner,
    repo,
    pull_number,
    comments,
    body: 'This is LLM reviewer for Test. LLM을 사용한 자동 코드 리뷰입니다.',
    event: 'COMMENT'
  })
}

async function main() {
  const prDetails = await getPRDetails()
  let diff: string | null
  const eventData = JSON.parse(
    readFileSync(process.env.GITHUB_EVENT_PATH ?? '{}', 'utf8')
  )

  if (eventData.action === 'opened') {
    diff = await getDiff(prDetails.owner, prDetails.repo, prDetails.pull_number)
  } else if (eventData.action === 'synchronize') {
    const newBaseSha = eventData.before
    const newHeadSha = eventData.after

    const response = await octokit.repos.compareCommits({
      headers: {
        accept: 'application/vnd.github.v3.diff'
      },
      owner: prDetails.owner,
      repo: prDetails.repo,
      base: newBaseSha,
      head: newHeadSha
    })

    diff = String(response.data)
  } else {
    console.log('Unsupported event:', process.env.GITHUB_EVENT_NAME)
    return
  }

  if (!diff) {
    console.log('No diff found')
    return
  }

  const parsedDiff = parseDiff(diff)

  const excludePatterns = exclude.split(',').map(s => s.trim())

  const filteredDiff = parsedDiff.filter(file => {
    return !excludePatterns.some(pattern => minimatch(file.to ?? '', pattern))
  })

  const comments = await analyzeCode(filteredDiff, prDetails)
  if (comments.length > 0) {
    await createReviewComment(
      prDetails.owner,
      prDetails.repo,
      prDetails.pull_number,
      comments
    )
  }
}

main().catch(error => {
  console.error('Error:', error)
  process.exit(1)
})
