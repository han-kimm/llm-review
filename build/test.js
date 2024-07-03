"use strict";
var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const fs_1 = require("fs");
const rest_1 = require("@octokit/rest");
const parse_diff_1 = __importDefault(require("parse-diff"));
const minimatch_1 = require("minimatch");
const sdk_1 = __importDefault(require("@anthropic-ai/sdk"));
const child_process_1 = require("child_process");
(0, child_process_1.execSync)("yarn install", { stdio: "inherit" });
const { GITHUB_TOKEN, LLM_API_KEY, LLM_API_MODEL, exclude } = process.env;
const octokit = new rest_1.Octokit({ auth: GITHUB_TOKEN });
const llm = new sdk_1.default({
    apiKey: LLM_API_KEY,
});
const SYSTEM_PROMPT = "You are a strict and perfect code review AI.";
function getPRDetails() {
    return __awaiter(this, void 0, void 0, function* () {
        var _a, _b;
        const { repository, number } = JSON.parse((0, fs_1.readFileSync)(process.env.GITHUB_EVENT_PATH || "", "utf8"));
        const prResponse = yield octokit.pulls.get({
            owner: repository.owner.login,
            repo: repository.name,
            pull_number: number,
        });
        return {
            owner: repository.owner.login,
            repo: repository.name,
            pull_number: number,
            title: (_a = prResponse.data.title) !== null && _a !== void 0 ? _a : "",
            description: (_b = prResponse.data.body) !== null && _b !== void 0 ? _b : "",
        };
    });
}
function getDiff(owner, repo, pull_number) {
    return __awaiter(this, void 0, void 0, function* () {
        const response = yield octokit.pulls.get({
            owner,
            repo,
            pull_number,
            mediaType: { format: "diff" },
        });
        return response.data;
    });
}
function analyzeCode(parsedDiff, prDetails) {
    return __awaiter(this, void 0, void 0, function* () {
        const comments = [];
        for (const file of parsedDiff) {
            if (file.to === "/dev/null")
                continue;
            for (const chunk of file.chunks) {
                const prompt = createPrompt(file, chunk, prDetails);
                const aiResponse = yield getAIResponse(prompt);
                if (aiResponse) {
                    const newComments = createComment(file, chunk, aiResponse);
                    if (newComments) {
                        comments.push(...newComments);
                    }
                }
            }
        }
        return comments;
    });
}
function createPrompt(file, chunk, prDetails) {
    return `Your task is to review pull requests. Instructions:
- Provide the response in following JSON format:  {"reviews": [{"lineNumber":  <line_number>, "reviewComment": "<review comment>"}]}
- Do not give positive comments or compliments.
- Provide comments and suggestions ONLY if there is something to improve, otherwise "reviews" should be an empty array.
- Write the comment in GitHub Markdown format.
- Use the given description only for the overall context and only comment the code.
- IMPORTANT: NEVER suggest adding comments to the code.

Review the following code diff in the file "${file.to}" and take the pull request title and description into account when writing the response.
  
Pull request title: ${prDetails.title}
Pull request description:

---
${prDetails.description}
---

Git diff to review:

\`\`\`diff
${chunk.content}
${chunk.changes
        .map((c) => `${c.ln ? c.ln : c.ln2} ${c.content}`)
        .join("\n")}
\`\`\`
`;
}
function getAIResponse(prompt) {
    return __awaiter(this, void 0, void 0, function* () {
        const queryConfig = {
            model: LLM_API_MODEL !== null && LLM_API_MODEL !== void 0 ? LLM_API_MODEL : "claude-3-haiku-20240307",
            temperature: 0.2,
            max_tokens: 700,
            top_p: 1,
            frequency_penalty: 0,
            presence_penalty: 0,
        };
        try {
            const response = yield llm.messages.create(Object.assign(Object.assign({}, queryConfig), { system: SYSTEM_PROMPT, messages: [
                    {
                        role: "user",
                        content: prompt,
                    },
                ] }));
            const { type } = response.content[0];
            if (type === "text") {
                return JSON.parse(response.content[0].text).reviews;
            }
            return null;
        }
        catch (error) {
            console.error("Error:", error);
            return null;
        }
    });
}
function createComment(file, chunk, aiResponses) {
    return aiResponses.flatMap((aiResponse) => {
        if (!file.to) {
            return [];
        }
        return {
            body: aiResponse.reviewComment,
            path: file.to,
            line: Number(aiResponse.lineNumber),
        };
    });
}
function createReviewComment(owner, repo, pull_number, comments) {
    return __awaiter(this, void 0, void 0, function* () {
        yield octokit.pulls.createReview({
            owner,
            repo,
            pull_number,
            comments,
            event: "COMMENT",
        });
    });
}
function main() {
    return __awaiter(this, void 0, void 0, function* () {
        var _a;
        const prDetails = yield getPRDetails();
        let diff;
        const eventData = JSON.parse((0, fs_1.readFileSync)((_a = process.env.GITHUB_EVENT_PATH) !== null && _a !== void 0 ? _a : "", "utf8"));
        if (eventData.action === "opened") {
            diff = yield getDiff(prDetails.owner, prDetails.repo, prDetails.pull_number);
        }
        else if (eventData.action === "synchronize") {
            const newBaseSha = eventData.before;
            const newHeadSha = eventData.after;
            const response = yield octokit.repos.compareCommits({
                headers: {
                    accept: "application/vnd.github.v3.diff",
                },
                owner: prDetails.owner,
                repo: prDetails.repo,
                base: newBaseSha,
                head: newHeadSha,
            });
            diff = String(response.data);
        }
        else {
            console.log("Unsupported event:", process.env.GITHUB_EVENT_NAME);
            return;
        }
        if (!diff) {
            console.log("No diff found");
            return;
        }
        const parsedDiff = (0, parse_diff_1.default)(diff);
        const excludePatterns = exclude.split(",").map((s) => s.trim());
        const filteredDiff = parsedDiff.filter((file) => {
            return !excludePatterns.some((pattern) => { var _a; return (0, minimatch_1.minimatch)((_a = file.to) !== null && _a !== void 0 ? _a : "", pattern); });
        });
        const comments = yield analyzeCode(filteredDiff, prDetails);
        if (comments.length > 0) {
            yield createReviewComment(prDetails.owner, prDetails.repo, prDetails.pull_number, comments);
        }
    });
}
main().catch((error) => {
    console.error("Error:", error);
    process.exit(1);
});
//# sourceMappingURL=test.js.map