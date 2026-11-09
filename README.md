# API Evals 101 (Capital One) — which model runs our support bot best?

Hands-on materials for the **API evals 101** session, wired for the Capital
One environment. You'll take one RAG customer-support bot, run it on **three
models served by the internal gateway**, evaluate all three with the same
scorers in **Weave**, and decide which one is better, and whether "better" is
bigger than the noise.

This is the [public `api-evals-101`](https://github.com/umakrishnaswamy/api-evals-101)
repo with two things changed, and nothing else:

1. **Weave comes from the internal wrapper.** Every file imports it as
   `from c1_aiml_aem import weave`, with a fallback to the public package so
   the repo still runs on a presenter laptop.
2. **Inference goes through the gateway.** `llm.py` has one code path —
   `c1.aiml.genai.inference` — instead of three vendor SDKs. The three
   candidates are three *models on FM Hosting*, not three vendors.

The bot, the corpus and the customer (Acme Mobile) are made up for the workshop.

## What's in here

```
api-evals-101-cap1/
├── support_bot.py          the app: 10 help-center articles, retriever, prompt, SupportBot model,
│                           and the two eval suites (regression + capability)
├── scorers.py              THE scorer list: 7 pass/fail checks, each tagged with where a failure lives
├── llm.py                  one chat() over the Capital One inference gateway
├── config.py               models, judge, project. Everything comes from env vars
├── mock_llm.py             stand-in used when the gateway isn't reachable (or BACKEND=mock). NOT a model
├── 00_check_setup.py       pre-flight: wrapper + gateway import, every model answers a ping
├── 01_trace_conversations.py  tool-calling agent: trace 5 conversations per model in the Agents tab
└── 02_eval_three_models.py ⭐ evaluate all 3 models on both suites and print the comparison
```

The bot is the same for every model: same retriever, same prompt, same
datasets, same scorers. **The model is the only thing that changes**, which is
what makes the comparison fair.

## Setup (do this before the session)

```bash
git clone <this repo> && cd api-evals-101-cap1
python3.11 -m venv .venv && source .venv/bin/activate   # needs Python 3.10+

pip install c1_aiml_aem[weave]    # inside C1, from your internal index
# outside C1 only: pip install -r requirements.txt

cp .env.example .env    # fill in your W&B team and key
source .env

python 00_check_setup.py   # every line should say OK
```

In `.env` you need:

| variable | what to put |
|---|---|
| `WANDB_BASE_URL` | the W&B host. Defaults to `https://wandb.cloud.capitalone.com`; only set it to run somewhere else |
| `WANDB_API_KEY` | an API key **from that host** (`<host>/authorize`) |
| `WANDB_ENTITY` | your team name, the `<team>` in `<host>/<team>/...` URLs |

There are **no provider API keys** — the gateway handles auth. Weave finds its
trace server from `WANDB_BASE_URL` (`<host>/traces`). Results land at
`<host>/<team>/api-evals-101/weave`; `00_check_setup.py` prints the link.
Never commit `.env` (it's in `.gitignore`).

**Model IDs change.** FM Hosting renamed the served models to include the
provider prefix (bare `gpt-oss-20b` stopped resolving; it's `openai/gpt-oss-20b`),
and it will happen again. If `00_check_setup.py` reports a FAIL it asks the
gateway what it's serving and prints the list — set `MODEL_A`, `MODEL_B` or
`MODEL_C` and rerun. Defaults: `openai/gpt-oss-120b`, `openai/gpt-oss-20b`,
`meta/llama-4-scout`.

**Outside the sandbox?** The repo still runs. When `c1.aiml.genai.inference`
isn't importable, every model falls back to a scripted stand-in (`mock_llm.py`)
and shows up as `gpt-oss-120b-mock` etc. in the output and in Weave. Mocked
models run the regression suite only, and the LLM judge is mocked too (it
always says yes). Mock results say nothing about the models — use them to
check your Weave login and walk the UI, nothing more.

`BACKEND=mock python 02_eval_three_models.py` forces that path on purpose.

## Run it

```bash
python 01_trace_conversations.py   # ~2 min: look at what the bot does
python 02_eval_three_models.py     # ~5 min: 2 suites x 3 models = 6 evaluations
```

### 1. Look at traces first (`01`)

Here the bot is a small tool-calling agent, and **the model picks the tools**
each turn:

| tool | returns |
|---|---|
| `search_help_center` | the matching help-center articles (policies, prices, rules) |
| `get_account_details` | the signed-in customer's plan, lines, AutoPay card, outages, device |

Weave UI → **Agents** → `acme-support-bot`. Things to look at:

- **Tool-use distribution per model.** Same questions, but models differ in how
  often they look up the account, and whether they search at all before
  answering. A turn that answers a policy question with no `search_help_center`
  call is answering from memory.
- **Open `autopay-credit-card`.** The customer never says how they pay. The
  account says Visa credit card, and the article says credit-card AutoPay
  doesn't get the discount. Did the model look up the account, and did it
  connect the two?
- In any turn, the chat view shows each tool call, what came back, then the
  answer. If the answer and the tool results disagree, that's a hallucination.

`02` evaluates the simpler fixed-retrieval bot (`support_bot.py`, search on
every turn), so the eval isolates answer quality from tool choice.

### 2. Evaluate all three (`02`)

The script prints a table like this for each suite:

```
CAPABILITY — pass rate per scorer (± 95% interval, n=6 per model)
scorer                  area              gpt-oss-120b   gpt-oss-20b   llama-4-scout
retrieval_hit           retrieval          100% ± 0%     100% ± 0%     100% ± 0%
has_key_facts           completeness        ...
...
Where each model fails most:
  <model>   avoids_known_traps (67%) -> A known-wrong claim appeared, often a skipped condition...
```

Then in the Weave UI:

1. **Evaluations** tab, sort by name (descending). Runs are named
   `<timestamp>__<suite>__<model>`.
2. Select the **three runs of one suite** and click **Compare**. Don't mix
   suites; they use different datasets.
3. Click a red cell to open that row's trace, and read the scorer's `why`.

## The two suites

| Suite | Rows | What it tests | What "good" looks like |
|---|---|---|---|
| **Regression** | 8 | Plain questions a good bot should already get right, plus one legal question it should decline | Near 100%. The interesting day is the day a row goes red. |
| **Capability** | 6 | Hard cases: a buried exclusion (credit-card AutoPay), a wrong premise (dispute window), false certainty ("is unlimited really unlimited?"), a question the help center can't answer (termination fee), a multi-part question (roaming on Essentials), a condition that rules the customer out (6-hour outage) | Lower to start. This is where the models separate. |

Keep them separate. A suite that should be 100% and a suite that should start
low answer different questions; averaging them hides both.

## The scorers

Every scorer checks **one requirement**, returns one boolean named for its area (`citation_ok`, `grounding_ok`, …), and says
**where the failure lives** (`failure_area`) and **why** (`why`). A red cell
should tell you what to open next, not just that something is wrong.

| Scorer | Mechanism | Fails when… | If it fails, look at… |
|---|---|---|---|
| `retrieval_hit` | set check | the needed article wasn't retrieved | the retriever. Same for every model, so this should match across all 3 runs |
| `has_key_facts` | string check | a required fact (price, limit, window) is missing | the answer: the model had the article but dropped the fact |
| `avoids_known_traps` | string check | a known-wrong claim appears (e.g. "credit card qualifies") | the exclusion rule in the prompt; the model skipped a condition |
| `cites_sources` | structural | no `Sources:` line, or it cites an article it was never given | format drift between models |
| `readable_length` | structural | under 15 or over 180 words | verbosity; a model-style difference |
| `handles_scope` | binary LLM judge | it guesses on an out-of-scope question, or refuses a fair one | the "don't guess, offer a human" rule |
| `grounded_in_articles` | binary LLM judge | it makes a claim the articles don't support (`why` names it) | outside knowledge leaking in |

Cheapest check first: string and set checks cost nothing and are exact. The two
judges ask exactly one yes/no question each.

**The judge** is a fixed model, and it should *not* be one of the candidates —
LLM judges tend to favor their own outputs, so a candidate shouldn't grade
itself. The gateway serves a short list, so the default judge
(`openai/gpt-oss-120b`) is also a candidate; `02` prints a reminder when that's
true. Point `JUDGE_MODEL` at a served model outside `CANDIDATES` as soon as
there is one.

## Reading the results

- **Is the gap real?** With 6 rows, a pass rate has a 95% interval of roughly
  ±35–40 points. If two models' intervals overlap, you can't call a winner on this
  data. Add rows, or run `TRIALS=3` to see how much the same model moves
  between runs.
- **Check `retrieval_hit` first.** It should be identical for all three. If it
  isn't, something other than the model changed.
- **Read the failures, not just the averages.** Open a red row and read the
  answer next to the articles. Sometimes the scorer is the problem, not the
  model.
- **Quality isn't the whole decision.** Compare latency and token usage in the
  Compare view too.
