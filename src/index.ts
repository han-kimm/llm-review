import { readFileSync } from 'fs'
import { Octokit } from '@octokit/rest'
import parseDiff, { Chunk, File } from 'parse-diff'
import { minimatch } from 'minimatch'
import Anthropic from '@anthropic-ai/sdk'
import { getInput } from '@actions/core'
import { ChatAnthropic } from '@langchain/anthropic'

const GITHUB_TOKEN = getInput('GITHUB_TOKEN')
const LLM_API_KEY = getInput('LLM_API_KEY')
const LLM_API_MODEL = getInput('LLM_API_MODEL')
const exclude = getInput('exclude') ?? ''

const octokit = new Octokit({ auth: GITHUB_TOKEN })

const llm = new ChatAnthropic({
  apiKey: LLM_API_KEY,
  model: LLM_API_MODEL ?? 'claude-3-haiku-20240307',
  temperature: 0.2,
  maxTokens: 700
})

const SYSTEM_PROMPT = 'You are a strict and perfect code review AI.'

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
      const prompt = createPrompt(file, chunk, prDetails)
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

function createPrompt(file: File, chunk: Chunk, prDetails: PRDetails): string {
  return `Your task is to review pull requests. Instructions:
- Provide the response in following JSON format:  {"reviews": [{"lineNumber":  <line_number>, "reviewComment": "<review comment in Korean>"}]}
- Do not give positive comments or compliments.
- Provide comments and suggestions ONLY if there is something to improve, otherwise "reviews" should be an empty array.
- Write the comment in GitHub Markdown format.
- Use the given description only for the overall context and only comment the code.
- IMPORTANT: NEVER suggest adding comments to the code.

Review the following code diff in the file "${
    file.to
  }" and take the pull request title and description into account when writing the response.
  
Pull request title: ${prDetails.title}
Pull request description:

---
${prDetails.description}
---

Git diff to review:

\`\`\`diff
${chunk.content}
${chunk.changes
  // @ts-expect-error - ln and ln2 exists where needed
  .map(c => `${c.ln ? c.ln : c.ln2} ${c.content}`)
  .join('\n')}
\`\`\`

please ensure answer is JSON format : {"reviews": [{"lineNumber":  <line_number>, "reviewComment": "<review comment in Korean>"}]}.
`
}

async function getAIResponse(prompt: string): Promise<Array<{
  lineNumber: string
  reviewComment: string
}> | null> {
  try {
    const response = await llm.invoke([
      ['system', SYSTEM_PROMPT],
      ['user', prompt]
    ])

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
    event: 'COMMENT'
  })
}

async function main() {
  const prDetails = await getPRDetails()
  let diff: string | null
  const eventData = JSON.parse(
    readFileSync(process.env.GITHUB_EVENT_PATH ?? '', 'utf8')
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
