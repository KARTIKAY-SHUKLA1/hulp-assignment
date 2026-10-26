# Adaptive Automation vs. Bot Defense
**Kartikay Shukla | Product Engineer Intern Assignment | HULP**

---

## Preface

One of my personal projects — **[BrowserBot](https://github.com/KARTIKAY-SHUKLA1/BrowserBot)** — is already on my resume and happens to be directly relevant to both parts of this assignment. It's an autonomous AI browser agent (LangGraph, Playwright, Llama-3.3-70b) that navigates real websites from a natural-language goal with no public API.

Building it gave me hands-on experience on both sides. For Part 1 — I had to make it resilient to real websites; Flipkart works reliably, Amazon blocks it, and I have a good theory why (likely JA3 fingerprinting — covered in Part 2). For Part 2 — because I built a bot, I know firsthand what signals betray automation. That's not something I read about.

I'll use **online service booking** (cab booking, home services) as the domain throughout — directly what HULP does.

---

# Part 1: Resilient Browser Automation

## 1. Architecture

The core idea is a **three-layer system**: an intent layer (what to do), an adaptive execution layer (how to do it right now), and a self-healing layer (what to do when it breaks).

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Request                            │
│              "Book a cab from Connaught Place to IGI"           │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Task Orchestrator (FastAPI)                   │
│   - Accepts goal as natural language                            │
│   - Assigns run_id, pushes to async execution queue            │
│   - Streams progress back to user via SSE                       │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│               LangGraph Agent (plan → act → observe loop)       │
│                                                                 │
│   ┌──────────┐    ┌──────────┐    ┌───────────────┐            │
│   │  PLAN    │───▶│   ACT    │───▶│    OBSERVE    │            │
│   │  (LLM)  │    │(Playwright)    │  (page state) │            │
│   └────▲─────┘    └──────────┘    └───────┬───────┘            │
│        │                                  │                     │
│        │          ┌───────────────────────▼─────────────────┐  │
│        │          │         _should_continue()?              │  │
│        │          │   done=False AND token budget ok?        │  │
│        │          └──────────┬──────────────────┬───────────┘  │
│        │                     │ not done          │ done / END   │
│        └─────────────────────┘                  ▼              │
│                                                END              │
└─────────────────────────────────────────────────────────────────┘
                                           │
                          ┌────────────────▼────────────────┐
                          │       Semantic Browser Wrapper   │
                          │  (Playwright, no CSS selectors)  │
                          └────────────────┬────────────────┘
                                           │
                          ┌────────────────▼────────────────┐
                          │    Human-in-the-Loop Gate        │
                          │  (dangerous actions → SSE pause) │
                          └─────────────────────────────────┘
```

**Key Components:**

| Component | Role |
|---|---|
| **Task Orchestrator** | Accepts natural-language goals, manages run lifecycle |
| **LangGraph StateGraph** | Drives the plan→act→observe loop; maintains full agent state |
| **LLM Planner (Gemini Flash 2.0)** | Decides one action at a time based on current page state. Chosen for this use case: cheapest at scale ($0.075/M tokens), fast, structured outputs native, and strong reasoning for semantic re-planning. Vision capability (screenshot fallback) exists as a last resort for rare icon-only UI edge cases. GPT-4o-mini is a strong alternative — the system is LLM-agnostic |
| **Semantic Browser Wrapper** | Executes actions using ARIA roles, visible text, labels — never CSS |
| **Observe Node** | Snapshots page title, URL, and body text back into agent memory |
| **Safety Gate** | LLM-based intent classifier (`ActionRisk`) intercepts irreversible clicks; pauses for human confirmation with transaction details |
| **SSE Event Stream** | Pushes live step events to frontend |

---

## 2. Detailed Design

### 2.1 Why the LLM is the Key to Resilience

Traditional browser automation scripts break because they hardcode locators:
```python
# Fragile: breaks the moment the site redesigns
driver.find_element(By.CSS_SELECTOR, ".booking-btn-v2 > span.cta")
```

The moment the class name changes from `.booking-btn-v2` to `.booking-cta-v3`, the script fails with `ElementNotFound`.

The LLM-driven approach solves this differently. Instead of hardcoding **where** to click, it describes **what** to click:

```python
# Resilient: "Book Now" is the visible button text, which rarely changes
click_element(description="Book Now")
```

Even if the site's internal DOM structure changes completely, as long as the visible text or ARIA role stays similar, the automation keeps working. The LLM looks at the current page snapshot (URL + title + body text) and decides the next semantic action from scratch every step.

### 2.2 Semantic Locator Waterfall

When resolving a description like `"Book Now"`, the browser wrapper tries multiple strategies in order, falling back gracefully:

```
1. Exact text match       → page.get_by_text("Book Now", exact=True)
2. Case-insensitive text  → page.get_by_text(re.compile("book now", IGNORECASE))
3. ARIA roles             → get_by_role("button"|"link"|"menuitem", name=...)
4. Placeholder            → get_by_placeholder(re.compile(...))
5. Give up gracefully     → return "Could not find element"
```

This waterfall means the system handles:
- Sites that use `<button>` text (step 1/2)
- Sites that use icon buttons with `aria-label` (step 3)
- Sites with form inputs labelled via placeholders (step 4)

No CSS selectors. No XPath. No brittle DOM traversal.

### 2.3 Page State as Memory

After every action, the `observe` node reads the page back:

```python
async def observe(state, browser):
    # Primary: text snapshot for LLM context
    snapshot = await browser.get_page_snapshot()
    # snapshot = "URL: ...\nTitle: ...\nContent: <first 3000 chars of body text>"

    # Fallback: screenshot only when text-based locators failed last step
    screenshot_b64 = None
    last_result = state["action_history"][-1].get("result", "")
    if "could not find" in last_result.lower() or "not found" in last_result.lower():
        screenshot_bytes = await browser.page.screenshot()
        screenshot_b64 = base64.b64encode(screenshot_bytes).decode()

    return {"page_content": snapshot, "screenshot": screenshot_b64}
```

Text snapshot is the default — cheap, fast, sufficient for 95%+ of steps. The screenshot is only captured when the previous action failed to find an element, triggering the vision fallback path. This avoids the memory and latency cost of taking screenshots every step.

### 2.4 Retry and Timeout Logic

Every browser action is wrapped in retry + timeout. In my BrowserBot project, I used a flat 0.5s sleep between all retries:

```python
# What BrowserBot currently does — works, but not optimal
MAX_RETRIES = 2
for attempt in range(MAX_RETRIES + 1):
    try:
        result = await asyncio.wait_for(coro, timeout=STEP_TIMEOUT)
        break
    except asyncio.TimeoutError:
        if attempt < MAX_RETRIES:
            await asyncio.sleep(0.5)  # flat delay — not ideal
            continue
```

Building BrowserBot taught me two problems with this approach:

1. **Flat sleep is wrong.** If a site is slow, 0.5s is too short. If it's a network blip, 0.5s twice is fine. Exponential backoff (0.5s → 1s → 2s) is the right pattern — gives the site progressively more time to recover.

2. **Not all failures should be retried.** A `TimeoutError` means the element might appear if we wait. But `"Could not find element"` means the element genuinely doesn't exist right now — retrying it 2 more times wastes 30 seconds for no reason.

**What the production system should do:**

```python
STEP_TIMEOUT = 15.0

async def execute_with_smart_retry(coro, action: str, max_retries: int = 2):
    for attempt in range(max_retries + 1):
        try:
            return await asyncio.wait_for(coro, timeout=STEP_TIMEOUT)

        except asyncio.TimeoutError:
            # Worth retrying — site might just be slow
            if attempt < max_retries:
                backoff = 0.5 * (2 ** attempt)  # 0.5s, 1s, 2s
                await asyncio.sleep(backoff)
                continue
            return f"Action timed out after {max_retries + 1} attempts"

        except Exception as exc:
            err = str(exc).lower()
            # "not found" errors: no point retrying, fail immediately
            if any(phrase in err for phrase in ["not found", "could not find", "no element"]):
                return f"Element not found — skipping retries: {exc}"
            # Other errors (network, crash): worth retrying with backoff
            if attempt < max_retries:
                await asyncio.sleep(0.5 * (2 ** attempt))
                continue
            return f"Action failed after {max_retries + 1} attempts: {exc}"
```

If all retries fail, the result string is fed back into the LLM's next plan step. The LLM can then decide: *"The click timed out. Let me scroll down first and try again."*

**Critical production concern — don't retry side-effect actions blindly:**

There's a subtle but serious problem with retrying on `TimeoutError` for booking/payment actions. A timeout means *"I didn't receive confirmation"* — not *"the action didn't execute."* If a "Confirm Booking" click times out, the booking may have already been made server-side. Retrying = double booking.

The fix can't be based on action **type** alone. `click_element` appears safe by name — but `click_element("Confirm Booking")` is exactly the action that submits a payment. Classifying `click_element` as universally safe to retry is wrong.

The correct approach is to reuse the same **`ActionRisk` LLM classifier** from the safety gate (section 2.6) — if the planned action was classified as `is_irreversible=True`, treat any timeout as `UNKNOWN_OUTCOME`. If `is_irreversible=False`, safe to retry:

```python
async def execute_with_smart_retry(
    coro, planned_action: BrowserAction, action_risk: ActionRisk, max_retries: int = 2
):
    # is_irreversible comes from the same LLM classifier used by the safety gate
    # click_element on a payment form → is_irreversible=True
    # navigate_to, extract_text → is_irreversible=False

    for attempt in range(max_retries + 1):
        try:
            return await asyncio.wait_for(coro, timeout=STEP_TIMEOUT)

        except asyncio.TimeoutError:
            if action_risk.is_irreversible:
                # Don't retry — outcome is unknown. Check page state before retrying.
                return "UNKNOWN_OUTCOME: action may have executed. Inspect page state before retrying."
            if attempt < max_retries:
                backoff = 0.5 * (2 ** attempt)
                await asyncio.sleep(backoff)
                continue
            return f"Action timed out after {max_retries + 1} attempts"

        except Exception as exc:
            err = str(exc).lower()
            if any(phrase in err for phrase in ["not found", "could not find", "no element"]):
                return f"Element not found — skipping retries: {exc}"
            if attempt < max_retries:
                await asyncio.sleep(0.5 * (2 ** attempt))
                continue
            return f"Action failed after {max_retries + 1} attempts: {exc}"
```

This way the safety gate and retry logic share a single source of truth for what's irreversible — no duplication, no gap where `click_element` slips through as "safe" when it's actually submitting a booking. Importantly, if the `ActionRisk` classifier itself fails (API error), the retry logic also defaults to `UNKNOWN_OUTCOME` — never assume an action is safe to retry when classification is uncertain.

When `UNKNOWN_OUTCOME` is returned, the LLM's next plan step receives it and should check the page state — e.g. `extract_text("booking confirmation")` or navigate to booking history — before deciding whether to retry. If the target service supports idempotency keys, they should be passed with every booking request so that even if the request hits the server twice, it's processed only once.

### 2.5 LLM JSON Recovery — and Why I'd Do It Differently in Production

In BrowserBot, I handle malformed LLM JSON with two regex-based recovery layers:

```python
# Layer 1: sanitize literal newlines inside JSON strings
def _sanitize_json(s: str) -> str:
    # walks char-by-char, escapes \n inside string values

# Layer 2: regex-extract key fields if JSON.parse still fails
def _recover_json(raw: str) -> Optional[dict]:
    action = re.search(r'"action"\s*:\s*"([^"]+)"', raw)
    done   = re.search(r'"done"\s*:\s*(true|false)', raw)
    args   = re.search(r'"args"\s*:\s*(\{[^}]*\})', raw)
    # reconstruct minimal valid dict from recovered fields
```

This works — but building it showed me the real problem: I'm fighting the LLM's output after the fact, with fragile regex. It breaks on nested JSON, on multi-line `final_answer` strings, and on any LLM that decides to add commentary before the JSON object.

**The right solution is to eliminate the problem at the source using structured outputs / tool calling:**

```python
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel
from typing import Literal

class BrowserAction(BaseModel):
    action: Literal["navigate_to", "click_element", "type_text",
                    "press_key", "extract_text", "finish"]
    url: str | None = None           # for navigate_to
    description: str | None = None   # for click_element, type_text, extract_text
    text: str | None = None          # for type_text
    key: str | None = None           # for press_key
    done: bool = False
    final_answer: str = ""

# Force the LLM to return a valid BrowserAction — no freeform JSON
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
structured_llm = llm.with_structured_output(BrowserAction)

# In the plan node — always returns a valid BrowserAction object, never raw text
planned_action: BrowserAction = await structured_llm.ainvoke(messages)
```

Gemini Flash 2.0 supports structured outputs natively. With structured outputs, the LLM is constrained to return a valid schema at the API level — this eliminates the most common source of `json.JSONDecodeError`. That said, API failures, timeouts, and application-side schema validation errors can still occur. Structured outputs reduce malformed-response risk; they don't remove the need for error handling. No regex sanitization needed for the happy path, but a validation + graceful fallback layer is still required.

> **Note:** In my BrowserBot side project I used Groq + Llama-3.3-70b with manual regex JSON recovery — that worked fine for a prototype. For HULP's production system, structured outputs is the right approach. Gemini Flash 2.0 also has vision capability as a last-resort fallback: in the rare case where both text-based semantic locators AND LLM text re-planning fail (e.g., an icon-only button with no text or ARIA label), a screenshot can be passed to the LLM to locate elements visually. This covers maybe 5% of edge cases — Layers 1 and 2 handle the other 95%.

### 2.6 Human-in-the-Loop Safety Gate

When the LLM plans to click something that matches purchase/payment patterns:

```python
_DANGEROUS_RE = re.compile(
    r"\b(buy|purchase|checkout|pay|payment|confirm\s+order|place\s+order|...)\b",
    re.IGNORECASE
)
```

The action is intercepted. Instead of executing:
1. A `paused` event is pushed to the SSE stream
2. A `ConfirmationRequest` object is stored with `run_id` as key
3. The agent `await`s the human decision (up to 120 seconds)
4. The human responds via a separate `/confirm/{run_id}` endpoint
5. If confirmed → re-plan to execute; if denied/timeout → stop safely

The whole point is: you never auto-pay on someone's behalf without them explicitly saying yes.

**Limitation of this approach — and the production fix:**

The regex matches against the **LLM's text description** of the action, not the actual element on the page. This is a weak security boundary — a site labelling its payment button "Proceed", "Go!", or "Next →" would bypass the regex entirely. The LLM might also describe the action loosely ("click the green button") and miss the match.

A keyword-based execution-layer check (is this a `submit` button? does the URL contain "checkout"?) improves things but still fails when a booking action uses a `<div onclick>` with an unexpected label, or the URL is `/step/3` with no payment keywords.

The correct production fix is **LLM-based intent classification** — the system already has an LLM in the loop, so use it. Before executing any click action, ask the LLM whether the planned action is irreversible given the full current page context:

```python
class ActionRisk(BaseModel):
    is_irreversible: bool
    reason: str           # e.g. "page shows booking summary with price and a final submit form"
    transaction_details: str  # human-readable summary of what will happen: "OlaCabs Mini ₹450,
                              # Connaught Place → IGI Airport, tomorrow 8am" — bound to approval

async def classify_action_risk(planned_action: BrowserAction, page_snapshot: str) -> ActionRisk:
    risk_llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
    classifier = risk_llm.with_structured_output(ActionRisk)
    return await classifier.ainvoke([
        SystemMessage("You are a safety classifier. Given a planned browser action and the "
                      "current page state, decide if this action is irreversible "
                      "(e.g. completes a booking, submits payment, places an order). "
                      "Extract the specific transaction details visible on the page "
                      "(price, route, item, time) into transaction_details."),
        HumanMessage(f"Planned action: {planned_action}\n\nPage state:\n{page_snapshot}")
    ])

# In the act node — before executing any click
if planned_action.action == "click_element":
    try:
        risk = await classify_action_risk(planned_action, state["page_content"])
    except Exception:
        # Classifier failed (API error, timeout, etc.) — fail safe: treat as irreversible.
        # Never assume an action is safe when classification is uncertain.
        risk = ActionRisk(is_irreversible=True, reason="classifier unavailable",
                          transaction_details="unknown — classifier failed")

    if risk.is_irreversible:
        # Pause for human confirmation. The approval is bound to the specific
        # transaction_details extracted from the current page — not just the action name.
        confirmation = await request_human_confirmation(state, planned_action, risk)

        # Re-read page state after approval before executing.
        # Between approval and execution the page may have changed (session expiry,
        # price update, different booking loaded). Revalidate before clicking.
        current_snapshot = await get_page_snapshot(state["browser_context"])
        recheck = await classify_action_risk(planned_action, current_snapshot)
        if recheck.transaction_details != risk.transaction_details:
            # Details changed — abort and notify user rather than execute on stale approval
            raise SafetyError(
                f"Page state changed after approval. "
                f"Approved: {risk.transaction_details} | "
                f"Now: {recheck.transaction_details}"
            )

        return await execute_confirmed_action(state, planned_action)
```

The LLM sees the full page snapshot — a visible price, a booking summary, a payment form — and classifies intent accurately regardless of button type, label, or URL. A `<div>` that says "Complete your ride booking" on a page showing fare and pickup details is correctly flagged as irreversible even if it has no semantic HTML markers.

Three properties this design enforces:
- **Conservative default:** classifier failure → irreversible. Never assume safe when uncertain.
- **Approval bound to details:** the user approves specific transaction details extracted from the page, not just a button name. "OlaCabs Mini ₹450, CP → IGI, tomorrow 8am" — not "click Confirm Booking."
- **Revalidation before execution:** page state is re-read after approval. If details changed, execution is aborted rather than proceeding on a stale approval. This adds one extra page read for irreversible steps — a worthwhile cost.

---

## 3. Implementation Details

**Tech Stack:** Python, FastAPI, LangGraph, Playwright, Gemini Flash 2.0 (Google AI), React (Vite)

**Key interfaces:**

```python
# Agent state — the full context carried through the graph
class AgentState(TypedDict):
    goal: str                      # natural-language user goal
    page_content: str              # latest page snapshot
    action_history: list[dict]     # all past actions + results
    steps: list[StepRecord]        # structured step log
    planned_action: Optional[dict] # current plan from LLM
    done: bool
    final_answer: str
    outcome: str  # "success" | "failed" | "incomplete" | "unknown" — set from verified page state

# Available browser actions (semantic, no selectors)
ACTIONS = [
    "navigate_to(url)",
    "click_element(description)",
    "type_text(description, text)",
    "press_key(key)",
    "extract_text(description)",
    "finish()"
]
```

**LangGraph topology:**

```
          ┌──────────────────────────────────────────┐
          │                  (loop)                  │
          ▼                                          │
START ──▶ plan ──▶ act ──▶ observe ──▶ [conditional] ──▶ END
                                       (done=True or token budget exceeded)
```

> **Note on step budget:** In BrowserBot I hardcoded `MAX_STEPS = 10`. This works for simple tasks but is too rigid — a multi-city cab booking or a multi-item food order could legitimately need 15–20 steps. In a production system I'd replace the hard cap with a **cost guard**: instead of counting steps, track total tokens consumed per run and stop if it exceeds a budget threshold (e.g., 15,000 tokens). This lets simple tasks finish in 3 steps and complex ones use 25 steps, while still protecting against infinite loops. The LangGraph conditional edge becomes:
> ```python
> def _should_continue(state):
>     if state.get("done"):              return "__end__"
>     if state["tokens_used"] > 15_000: return "__end__"  # cost guard
>     return "plan"
> ```

**SSE event schema:**

```jsonc
// Step executed
{"type": "step", "step": {"action": "click_element(...)", "result": "Clicked button 'Book Now'"}}

// Agent paused for confirmation
{"type": "paused", "run_id": "abc123", "pending_action": {...}, "message": "Agent wants to click 'Confirm Payment'..."}

// Run complete — outcome field distinguishes verified success from mere task completion
// Possible outcome values: "success" (confirmed from page state), "failed" (error confirmed),
// "incomplete" (task stopped mid-way), "unknown" (action executed but result unverifiable)
{"type": "done", "result": {"goal": "...", "steps_taken": [...], "final_answer": "...", "outcome": "success", "booking_id": "OLA-84921"}}
```

---

## 4. Worked Examples

### Example A: Successful cab booking flow

**Goal:** `"Book a cab from Connaught Place to IGI Airport on OlaCabs for tomorrow 8am"`

| Step | Action | LLM Reasoning | Result |
|---|---|---|---|
| 1 | `navigate_to("https://www.olacabs.com")` | Start at home page | Title: OlaCabs \| Online Cab Booking |
| 2 | `click_element("Rides")` | Select service type | Clicked link 'Rides' |
| 3 | `type_text("Pickup location", "Connaught Place")` | Fill origin | Typed into textbox |
| 4 | `type_text("Drop location", "IGI Airport")` | Fill destination | Typed into textbox |
| 5 | `click_element("Search Cabs")` | Trigger search | Clicked button |
| 6 | `extract_text("page")` | See available options | Returns list of cab types + ETAs |
| 7 | `click_element("Book Mini")` | Select cab type | Clicked |
| 8 | ⚠️ **Safety gate** on `click_element("Confirm Booking")` | `ActionRisk` classifier sees booking summary + price on page → `is_irreversible=True` | → SSE `paused` event sent to user with transaction details |
| 9 | User confirms via `/confirm/abc123` | Human-in-the-loop | Execution resumes |
| 10 | `extract_text("booking confirmation")` | Verify outcome from page state | Returns booking ID + confirmation message |
| 11 | `finish()` | Goal complete — confirmation ID extracted | `outcome: "success", booking_id: "OLA-84921"` |

**Result:** Booking confirmed. Outcome verified from page state (booking ID present), not inferred from reaching `finish()`.

> **Why this matters:** `finish()` means the LLM decided the task is done — it doesn't mean the booking exists. The verification step (`extract_text`) reads the actual page state after the action. If the confirmation page never loaded, or returned an error, the LLM would see that in the snapshot and set `outcome: "unknown"` or `outcome: "failed"` instead.

---

### Example B: Edge case — UI redesign mid-run (the hard case)

**Scenario:** The site changed its button from `"Book Now"` to `"Reserve Instantly"` overnight.

**What a traditional script does:**
```
ElementNotFoundError: .book-now-btn not found
```
Script crashes. Engineer gets paged at 2am. Script is manually patched. Repeat next week.

**What the LLM-driven agent does:**

| Step | Action | Result |
|---|---|---|
| 1 | `navigate_to(url)` | Page loaded |
| 2 | `click_element("Book Now")` | **Failed** — "Could not find element matching 'Book Now'" |
| 3 | LLM re-plans based on page snapshot | LLM sees `"Reserve Instantly"` in body text |
| 4 | `click_element("Reserve Instantly")` | ✅ Clicked — booking flow resumes |
| 5 | `extract_text("page")` | Cab options and ETAs loaded |
| 6 | `click_element("Book Mini")` | ✅ Clicked |
| 7 | `extract_text("booking confirmation")` | Verify outcome — did the click actually complete the booking? |
| 8 | `finish()` | Confirmation text found — `outcome: "success"` |

**Why this works:** The observe node fed back the full page body text including `"Reserve Instantly"` as a visible element. The LLM's next plan naturally adapted. No code change required.

---

## 5. Failure Modes and Trade-offs

| Failure | Detection | Recovery | What I chose NOT to handle |
|---|---|---|---|
| **LLM gives malformed JSON** | `json.JSONDecodeError` | Production fix: use Gemini Flash 2.0's structured outputs — LLM is constrained to return a valid schema, reducing malformed-response risk significantly. Schema validation and error handling are still required for API failures and edge cases. (In BrowserBot I used regex recovery as a prototype approach — works but fragile) | Fully corrupted API responses — just fail gracefully |
| **Action timeout (15s)** | `asyncio.TimeoutError` | Retryable actions: retry up to 2× with exponential backoff (0.5s → 1s). Irreversible actions (`is_irreversible=True`): return `UNKNOWN_OUTCOME` immediately — outcome must be verified from page state before any retry is attempted | Retries beyond 2× (diminishing returns, costs tokens) |
| **Token budget exceeded** | `tokens_used > 15,000` | Force `END`, return what's collected so far | Open-ended tasks that legitimately need more tokens — those should be broken into sub-tasks |
| **Site uses CAPTCHA** | Body text contains "CAPTCHA"/"verify" | LLM notes it in final_answer | Auto-solving CAPTCHAs (out of scope, ethically questionable) |
| **Site blocks bot fingerprint** | `navigate_to` returns error page | LLM notes it; try alternative URL | Full stealth mode (out of scope for this system) |
| **Dangerous click (payment)** | `ActionRisk` LLM classifier flags `is_irreversible=True` based on page context | Human confirmation via SSE pause; approval bound to specific transaction details | Autonomous payment without human approval |

**Key trade-off:** Semantic locators try 4–5 strategies per action, so they're slower than a single CSS selector. For a real-time system that's a concern — but HULP's automation runs async in the background, so a 200ms locator resolution is totally fine.

---

## 6. Operations: Monitoring, Alerting, Scaling, Cost

### Monitoring
- **Step success rate per run:** what % of steps succeed on first attempt vs. retry
- **Run completion rate:** % of runs that reach `outcome: "success"` (confirmed from page state) vs. `"failed"`, `"incomplete"`, or `"unknown"` — the breakdown matters more than the aggregate
- **LLM JSON parse failure rate:** tracks prompt quality degradation
- **P95 step latency:** alert if semantic locator resolution gets slow

### Alerting
- Alert if run completion rate drops below 70% in a 1-hour window → likely a site redesign
- Alert if structured output schema validation fails (should be near-zero with Gemini Flash 2.0 — a spike means model API issue)
- Alert on Safety Gate trigger rate spike → possible LLM misbehaviour

### Scaling
- Each run gets its own Playwright browser context (isolated, parallelisable)
- Runs are queued async; the FastAPI + asyncio event loop handles concurrency
- Horizontal scaling: multiple worker pods, each running N concurrent browser contexts
- Browser contexts are far cheaper than full browser instances (shared process)
- **Memory limit per pod:** Each browser context uses ~50–150MB RAM. A pod with 8GB RAM can safely run ~40–60 concurrent contexts. Beyond that, spin up another pod.
- **Crash recovery — durable checkpoints (not in-memory):** After every step completes successfully, the result is written to a durable store (Redis or Postgres) before moving to the next step:
  ```python
  await checkpoint_store.save(run_id=run.id, step_idx=step.idx, result=result)
  ```
  In-memory state alone cannot survive a worker crash — durability requires explicit persistence. On restart, the worker reads the last saved checkpoint and resumes from there.
- **Worker leases:** Each worker holds a heartbeat lock on its assigned run (renewed every 10s). If a worker crashes and the heartbeat expires, a supervisor reassigns the run to a healthy worker, which resumes from the last checkpoint.
- **Idempotency before resume:** Before replaying any step from a checkpoint, the worker checks whether the action already produced its side effect — especially for irreversible steps (booking, payment). If a booking already exists for this `run_id`, the step is marked complete without re-executing. This connects directly to the `UNKNOWN_OUTCOME` handling in section 2.4.

### Cost

> **These are design-target estimates derived from public pricing and Playwright benchmarks, not production measurements. Actual figures depend on task complexity, token usage per run, and infrastructure configuration. Validate against real traffic before sizing capacity.**

Cost scales linearly with run volume and task complexity:

- **~\$0.001/run** at Gemini Flash 2.0 pricing (\$0.075/M input tokens, ~5,000–8,000 tokens per 10-step run) — significantly cheaper than alternatives
- **At 1,000 runs/day** → ~\$1/day (~\$30/month)
- **At 10,000 runs/day** → ~\$10/day — Gemini Flash remains cost-effective even at high volume; self-hosting an open-source model is the v2 path if Google AI pricing becomes a concern
- **LLM choice rationale:** Gemini Flash 2.0 is chosen for cost ($0.075/M tokens — cheapest at scale), speed, strong reasoning for semantic re-planning, and native structured outputs. Vision capability exists as a last-resort fallback for icon-only UI edge cases — but in practice, semantic text locators + LLM text re-planning handle 95%+ of UI changes without needing screenshots. My BrowserBot side project used Groq + Llama for prototyping; GPT-4o-mini is a strong production alternative — the system is LLM-agnostic, swapping is one config line.
- **Playwright:** Runs headless in production (set `PLAYWRIGHT_HEADLESS=true`). CPU-bound, scales with worker count.
- **Optimization lever:** Cache the `observe` snapshot for repeated identical URLs to avoid redundant LLM calls.

---

# Part 2: Bot Prevention Without CAPTCHAs

## 1. Architecture

The key insight: **don't try to detect bots at one point in time — build a continuous behavioral fingerprint and score every session in real time.**

```
┌─────────────────────────────────────────────────────────────────┐
│                       Client (Browser)                          │
│   - JS Telemetry Collector (mouse, keyboard, scroll, timing)    │
└──────────────────────────┬──────────────────────────────────────┘
                           │ Telemetry beacon (async POST)
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Signal Ingestion Layer                        │
│   - Receives passive behavioral signals                         │
│   - Enriches with network/TLS fingerprint (server-side)         │
│   - Pushes to Redis for real-time scoring                       │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Risk Scoring Engine                           │
│                                                                 │
│   ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐ │
│   │  Behavioral  │  │   Network    │  │   Temporal/Rate      │ │
│   │  Signals     │  │  Signals     │  │   Signals            │ │
│   └──────┬───────┘  └──────┬───────┘  └──────┬───────────────┘ │
│          └─────────────────┼─────────────────┘                 │
│                            ▼                                    │
│                   Weighted Risk Score (0–100)                   │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Policy Enforcement Layer                      │
│                                                                 │
│   Score  0–29  → Allow (trusted)                                 │
│   Score 30–59  → Soft friction (add delay, shadow-limit)         │
│   Score 60–79  → Hard friction (require email OTP for checkout)  │
│   Score 80–100 → Block + log for review                          │
└─────────────────────────────────────────────────────────────────┘
```

**Key Components:**

| Component | Role |
|---|---|
| **JS Telemetry Collector** | Passive browser signals: mouse entropy, scroll, typing patterns |
| **Signal Ingestion Layer** | Receives beacons, enriches with server-side signals (IP, TLS) |
| **Redis Session Store** | Per-session signal buffer + rolling score, TTL-managed |
| **Risk Scoring Engine** | Computes weighted score from all signal categories |
| **Policy Enforcement** | Tiered response: allow / soft-friction / hard-friction / block |
| **Audit Log** | All decisions stored for review and model retraining |

---

## 2. Detailed Design

### 2.1 Signal Categories

**Category A — Behavioral (client-side, hardest to fake at scale)**

| Signal | What it measures | Bot indicator |
|---|---|---|
| Mouse movement entropy | Randomness of mouse path | Too linear, too jittery, or zero |
| Click timing | Time between page load and first click | <100ms or exactly periodic |
| Scroll velocity | Natural deceleration curve | Constant velocity (scripted) |
| Keystroke timing | Inter-key intervals | Programmatic fill is instant |
| Focus/blur events | Tab switching, window focus | Bots rarely trigger these |
| Touch vs. mouse | Mobile users use touch | Mouse events on a mobile UA |

**Category B — Network (server-side)**

| Signal | What it measures | Bot indicator |
|---|---|---|
| IP reputation | Known bot/datacenter IP ranges | AWS/GCP egress IPs |
| TLS fingerprint (JA3) | Browser TLS handshake signature | Playwright/curl JA3 hashes |
| User-Agent consistency | Does UA match browser capabilities | Playwright UA says Chrome but JA3 differs |
| Request rate | Requests per second per IP/session | Burst patterns |
| Header ordering | HTTP/2 header frame order | Differs between real browsers and automation tools |

**Category C — Temporal/Rate**

| Signal | Bot indicator |
|---|---|
| Time on page before submitting a form | <2 seconds is suspicious |
| Number of orders from IP in 1 hour | Hard limit |
| Account age vs. order frequency | New account + 10 orders in 1 hour |
| Identical order payloads | Same pickup/drop repeated exactly |

### 2.2 Risk Scoring Algorithm

```python
def compute_risk_score(session: SessionSignals) -> int:
    score = 0

    # Behavioral signals (max 40 points)
    # Each may be None if telemetry was blocked or the beacon never arrived.
    # Skip missing signals — unknown is not the same as suspicious.
    if session.mouse_entropy is not None and session.mouse_entropy < 0.1:
        score += 20  # too linear / absent
    if session.click_latency_ms is not None and session.click_latency_ms < 100:
        score += 10  # inhuman speed
    if session.keystroke_uniformity is not None and session.keystroke_uniformity > 0.95:
        score += 10  # programmatic fill (page.type() style)

    # Network signals (max 40 points)
    # Server-side booleans — always available, default False when lookup fails.
    if session.ip_in_datacenter_range:        score += 20
    if session.ja3_matches_known_bot:         score += 20

    # Temporal signals (max 20 points)
    # time_on_page comes from the telemetry beacon — may be None if beacon was blocked.
    if session.time_on_page_sec is not None and session.time_on_page_sec < 2:
        score += 10
    if session.orders_per_hour is not None and session.orders_per_hour > 5:
        score += 10

    return min(score, 100)
```

**Why these thresholds specifically?**
- `mouse_entropy < 0.1` — Playwright generates **zero** real mouse events on a synthetic click. The collector requires at least 3 mouse points to compute entropy — with zero events, `computeMouseEntropy()` returns `null` and the signal is skipped (score contribution = 0). The threshold exists for bots that inject minimal synthetic movement (≥ 3 points but perfectly linear): angles are discretized into 16 bins spanning `[-π, π]` and entropy is normalized by `log2(16)` to `[0, 1]`. A perfectly linear path puts all angles in one bin → entropy = 0. A human's natural curved movement spreads angles across many bins → entropy near 1. The 0.1 threshold catches near-linear synthetic paths while leaving human movement untouched.
- `click_latency < 100ms` — Very short click latencies, particularly below 100ms, can be a useful bot-detection signal. However, this measurement alone is not conclusive evidence of automation — it measures time from page load to first click, not reaction time to a specific stimulus. A password manager auto-filling or a user clicking a pre-focused element could register below 100ms legitimately. The signal contributes to the score alongside other indicators; it doesn't trigger action on its own.
- `keystroke_uniformity > 0.95` — `page.fill()` bypasses keydown/keyup events entirely, so `keystroke_uniformity` comes back `null` for that bot and the check is skipped (score contribution = 0). The threshold exists for bots that use `page.type()` instead, which does fire keydown events but with 0ms inter-key delay (uniformity → 1.0). Real humans have uniformity ~0.3–0.6 depending on typing speed. 0.95 catches `page.type()`-style programmatic fills while ignoring fast typists.

These thresholds are starting points — they're set conservatively to minimize false positives. In practice they'd be calibrated against real traffic distributions.

**Important: these are probabilistic signals, not proof.** A keyboard-only user generates zero mouse events — `computeMouseEntropy()` returns `null` (fewer than 3 points), so the signal is skipped and contributes 0 to score, not +20. A user with a password manager auto-filling may have low click latency. Assistive technology users behave differently from both typical humans and bots. None of these signals alone is proof of automation.

This is why the tiered response matters: **no single signal triggers a block.** To reach score 80–100 (block), a session needs to trip multiple independent signals — e.g. zero mouse entropy AND a known bot JA3 AND datacenter IP. A keyboard-only user on a residential connection with a real browser JA3 scores at most 20–29 (allow) or at most 30–59 (soft friction). Hard restrictions only fire when several unrelated signals converge. Thresholds must be calibrated against real HULP traffic before deployment — what looks like a bot threshold in theory may cut into real user segments in practice.

**Signal correlation caveat:** The additive scoring model implicitly treats each signal as independent evidence, but that's not always true. `mouse_entropy = 0` and `click_latency < 100ms` often share the same root cause — a Playwright synthetic click — so scoring +20 + +10 isn't 30 points of separate evidence, it's one fact counted twice. Similarly, `ip_in_datacenter_range` and `ja3_matches_known_bot` frequently co-occur (bots deployed on cloud infra tend to also use Playwright's bundled Chromium). The additive model with a hard cap at `min(score, 100)` is a known simplification — it's the same approach used by Stripe Radar v1 and most early rule-based fraud systems. It works well enough in practice because correlated signals that share a root cause are still *evidence* of that cause, even if double-counting inflates the magnitude. The v2 ML upgrade (XGBoost/LightGBM) handles feature correlation naturally — the model learns that mouse entropy and click latency together aren't 2× the evidence of either alone.

This is deterministic and auditable — when a legitimate user gets wrongly blocked, the support team can pull their session, see exactly which signals fired and why, and reset their score manually. A black-box ML model can't offer that transparency. Auditability is the main reason to start with rules rather than ML — the deterministic v1 system also generates clean training labels for the v2 model.

**v2 upgrade path — ML scoring:**
Once the audit log accumulates enough labeled data (blocked sessions confirmed as bots, passed sessions confirmed as legitimate), the rule-based scorer gets replaced by an **XGBoost or LightGBM model**. The hand-written thresholds above become feature engineering — the same signals feed the model, but the model learns optimal decision boundaries from data rather than having them hardcoded. This is exactly how Stripe Radar evolved: hand-written rules first, gradient boosted trees second. The deterministic v1 system generates the training labels for v2.

### 2.3 Tiered Response (No CAPTCHAs)

```
Score  0–29:  ALLOW
              Normal flow, no friction

Score 30–59:  SOFT FRICTION
              - Add 500ms artificial delay on API responses (slows down high-volume bots)
              - Shadow-rate-limit: the request IS accepted and queued, but processed with lowest priority. From the user's perspective, the booking is "processing" — real users wait a few seconds and get their result. For high-volume bots running hundreds of parallel requests, the artificial throughput cap makes bulk abuse economically slow. The session is flagged for async human review.
              - Log session for async review

Score 60–79:  HARD FRICTION
              - Require email/phone OTP before checkout
              - This is invisible to a real user who has their phone nearby
              - A bot has no access to the OTP inbox → blocked effectively

Score 80–100: BLOCK
              - Return 429 with a generic message ("Something went wrong")
              - Log full session for IP-level block review
              - Don't tell the bot WHY it was blocked (adversarial hardening)
```

**Why no CAPTCHAs?** CAPTCHAs are a blunt instrument. They:
1. Add friction for *every* user, not just bots
2. Have known weaknesses (ML-based CAPTCHA solvers, CAPTCHA farms)
3. Hurt conversion rates — research consistently shows they cost ~10% of legitimate completions

The OTP step (score 60–79) acts as a targeted challenge only for high-risk sessions. It's also something real users are already comfortable with for security reasons.

### 2.4 TLS Fingerprinting (JA3)

This is one of the strongest signals. When a browser opens an HTTPS connection, the TLS `ClientHello` message contains a specific set of cipher suites and extensions. The JA3 hash is an MD5 of these fields.

Real Chrome browser → JA3: `ada70210d4b8e024bcce` (example)  
Playwright Chromium → JA3: `b32309a26951912be7dba376398abc3b` (different, even with same UA string)

This is why my BrowserBot README notes: *"Amazon aggressively blocks headful Playwright instances even with a real user agent."* One likely explanation is JA3 fingerprinting — the TLS handshake from Playwright Chromium differs from a real Chrome browser even when the user-agent string is identical. I set a realistic user-agent in BrowserBot's `browser.py`, but that doesn't change the TLS handshake. Amazon's actual detection stack is a black box, so I can't confirm JA3 is the sole cause — it could also involve canvas fingerprinting, cipher suite ordering, or other heuristics — but JA3 mismatch is the most plausible and commonly cited reason.

### 2.5 Honeypot Fields

Add a hidden form field (`display: none` in CSS) that real users never see or fill. Bots that scrape the DOM and fill all fields will fill it:

```html
<input type="text" name="website" style="display:none" tabindex="-1" autocomplete="off">
```

```python
# Server-side check
if request.form.get("website"):  # honeypot was filled
    session_score += 30
```

Zero friction for real users. Catches unsophisticated bots immediately.

---

## 3. Implementation Details

**Tech Stack:** Node.js (Express), Redis, BullMQ (for async score processing), PostgreSQL (audit log), JavaScript (client telemetry)

> In one of my other previous projects (**DevLinkr**), I've used Redis + BullMQ for async job queuing with exponential backoff — so this tech stack is familiar territory, not something I'm picking up fresh.

**Client-side telemetry collector:**

```javascript
class TelemetryCollector {
    constructor(sessionId) {
        this.sessionId = sessionId;
        this.signals = { mousePoints: [], clickTimings: [], keyTimings: [] };
        this.pageLoadTime = performance.timeOrigin;  // performance.timing.navigationStart is deprecated; timeOrigin is the modern equivalent
        this._attachListeners();
    }

    _attachListeners() {
        document.addEventListener('mousemove', (e) => {
            this.signals.mousePoints.push({ x: e.clientX, y: e.clientY, t: Date.now() });
        });
        document.addEventListener('keydown', () => {
            this.signals.keyTimings.push(Date.now());
        });
        // Bug fix: click listener was missing — clickTimings was always empty
        document.addEventListener('click', () => {
            this.signals.clickTimings.push(Date.now());
        });
    }

    computeMouseEntropy() {
        if (this.signals.mousePoints.length < 3) return null;  // not enough data
        const angles = [];
        for (let i = 2; i < this.signals.mousePoints.length; i++) {
            const dx1 = this.signals.mousePoints[i-1].x - this.signals.mousePoints[i-2].x;
            const dy1 = this.signals.mousePoints[i-1].y - this.signals.mousePoints[i-2].y;
            const dx2 = this.signals.mousePoints[i].x - this.signals.mousePoints[i-1].x;
            const dy2 = this.signals.mousePoints[i].y - this.signals.mousePoints[i-1].y;
            angles.push(Math.atan2(dy2, dx2) - Math.atan2(dy1, dx1));
        }

        // Shannon entropy requires a discrete distribution.
        // Discretize angles into N_BINS equal buckets spanning [-π, π],
        // then normalize entropy by log2(N_BINS) to get a value in [0, 1].
        // 0 = all angles in the same bucket (perfectly linear movement).
        // 1 = angles spread uniformly across all buckets (chaotic/human movement).
        const N_BINS = 16;
        const bins = new Array(N_BINS).fill(0);
        for (const angle of angles) {
            // Wrap to [-π, π] before binning.
            // The raw difference of two atan2 values spans [-2π, 2π]; values outside
            // [-π, π] would map to invalid or clamped bin indices without this step.
            const wrapped = Math.atan2(Math.sin(angle), Math.cos(angle));
            const idx = Math.min(
                Math.floor(((wrapped + Math.PI) / (2 * Math.PI)) * N_BINS),
                N_BINS - 1
            );
            bins[idx]++;
        }
        const total = angles.length;
        let entropy = 0;
        for (const count of bins) {
            if (count > 0) {
                const p = count / total;
                entropy -= p * Math.log2(p);
            }
        }
        return entropy / Math.log2(N_BINS);  // normalized to [0, 1]
    }

    beacon() {
        const timings = this.signals.keyTimings;
        const intervals = timings.slice(1).map((t, i) => t - timings[i])
            .filter(interval => interval > 0);  // drop zero/negative intervals (clock skew or synthetic simultaneity)

        // Guard: need at least 2 valid intervals AND a positive mean before computing.
        // mean = 0 would cause division by zero → NaN, which bypasses the null check in the scorer.
        const meanInterval = intervals.length >= 2 ? mean(intervals) : 0;
        const keystrokeUniformity = (intervals.length >= 2 && meanInterval > 0)
            ? 1 - (stddev(intervals) / meanInterval)
            : null;  // null = signal unavailable (e.g. page.fill() bypasses keydown, or too few events)

        const firstClick = this.signals.clickTimings[0];
        const clickLatency = firstClick !== undefined
            ? firstClick - this.pageLoadTime
            : null;  // null = no click recorded yet

        navigator.sendBeacon('/api/telemetry', JSON.stringify({
            // sessionId is NOT sent in the body — the server reads it from the HttpOnly cookie.
            // Sending it here would allow a client to target any session key.
            mouseEntropy: this.computeMouseEntropy(),   // null if < 3 mouse points
            clickLatency,                               // null if no click recorded
            keystrokeUniformity,                        // null if < 2 key intervals
            timeOnPageSec: Math.floor((Date.now() - this.pageLoadTime) / 1000)
        }));
    }
}
```

> **Note on null signals and scoring:** When a signal arrives as `null`, the scoring engine skips that signal rather than treating it as a specific value. A bot using `page.fill()` bypasses `keydown` events entirely — `keystrokeUniformity` arrives as `null` and contributes 0 points, not +10. This is conservative: missing behavioral signals don't inflate the score. The network and temporal signals (JA3, IP, time-on-page) still fire independently and catch the bot through other dimensions.

**Redis session state:**

```redis
# Per-session risk score (TTL 30 minutes)
SET session:{id}:score 45 EX 1800

# Per-IP rolling order count (1-hour window)
INCR ip:{ip}:orders:1h
EXPIRE ip:{ip}:orders:1h 3600

# JA3 hash blocklist (persistent set)
SADD ja3_blocklist "b32309a26951912be7dba376398abc3b"
```

**Risk scoring endpoint:**

```javascript
app.post('/api/telemetry', async (req, res) => {
    // Session ID must come from the same server-set HttpOnly cookie used by checkout —
    // never from req.body. If sessionId came from the body, a bot could write a
    // fabricated low-risk score into any session key it chooses, including its own.
    const sessionId = req.cookies.sessionId;
    // Validate against server-side session records — existence of the cookie alone is not enough.
    // A forged cookie with an arbitrary string would pass a simple null check.
    const session = await getSession(sessionId);
    if (!session) return res.status(401).json({ error: 'Invalid session' });

    // Behavioral signals come from the client — treat as untrusted evidence, not proof.
    // A bot can send { mouseEntropy: 0.73, clickLatency: 1240 } to appear human.
    // These signals influence the score but cannot override the server-side anchors below.
    const { mouseEntropy, clickLatency, keystrokeUniformity, timeOnPageSec } = req.body;

    // Server-side signals — not supplied by the client, cannot be fabricated via req.body.
    // JA3 is extracted by nginx from the TLS handshake; IP is from the network layer.
    // Even if a bot sends perfect behavioral signals, these still fire independently.
    const ja3 = extractJA3FromRequest(req);     // from nginx-ja3 module header
    const ip = req.ip;
    const isDatacenterIP = await checkIPReputation(ip);
    const isKnownBotJA3 = await redis.sismember('ja3_blocklist', ja3);
    const ordersThisHour = parseInt(await redis.get(`ip:${ip}:orders:1h`) || '0');

    const score = computeRiskScore({
        mouseEntropy, clickLatency, keystrokeUniformity, timeOnPageSec,  // client-supplied, untrusted
        isDatacenterIP, isKnownBotJA3, ordersThisHour                    // server-side, authoritative
    });

    await redis.set(`session:${sessionId}:score`, score, 'EX', 1800);
    res.json({ ok: true });
});
```

> **Trust boundary:** Behavioral signals (mouse, keyboard, click timing) are client-supplied and can be fabricated. A bot sending `mouseEntropy: 0.73` would score 0 on behavioral signals — but JA3 and IP are extracted server-side from the network layer and cannot be spoofed via the request body. A Playwright bot with a datacenter IP still scores +40 from those two signals alone, regardless of what behavioral values it sends. Rate limits on `ip:{ip}:orders:1h` are also server-side and unaffected by telemetry manipulation. Client-side signals add evidence; server-side signals provide the floor the score cannot drop below.

**Checkout enforcement endpoint:**

```javascript
// POST /api/bookings  — called when user submits a booking
// This endpoint sits behind authentication middleware (e.g. requireAuth) that populates
// req.user from a verified JWT or session token. req.user.phone is a verified phone number
// stored at account creation — not a client-supplied value from this request.
app.post('/api/bookings', requireAuth, async (req, res) => {
    // Session ID comes from a server-set HttpOnly cookie — never from the request body.
    // A client-supplied risk score would be trivially spoofable; only the server-side
    // Redis value (written by the telemetry endpoint) is trusted.
    const sessionId = req.cookies.sessionId;
    // Validate against server-side session records — a forged cookie passes a null check
    // but fails here if it doesn't correspond to a real session.
    const session = await getSession(sessionId);
    if (!session) return res.status(401).json({ error: 'Invalid session' });

    // Retrieve the score the telemetry endpoint computed and stored.
    // If the beacon never arrived (ad blocker, no JS), score is null — fall back to
    // network-only signals computed fresh here.
    const rawScore = await redis.get(`session:${sessionId}:score`);
    const score = rawScore !== null
        ? parseInt(rawScore)
        : computeNetworkOnlyScore(req);   // JA3 + IP only — no behavioral signals

    // Apply tiered policy
    if (score >= 80) {
        // Hard block — don't reveal the reason
        return res.status(429).json({ error: 'Request could not be completed. Please try again later.' });
    }

    if (score >= 60) {
        // Hard friction — require OTP before proceeding
        const otpVerified = await redis.get(`session:${sessionId}:otp_verified`);
        if (!otpVerified) {
            // req.user.phone is populated by requireAuth middleware — guard defensively
            if (!req.user?.phone) {
                return res.status(401).json({ error: 'Authentication required' });
            }
            await sendOTP(req.user.phone);
            return res.status(202).json({ status: 'otp_required', message: 'Verification sent.' });
        }
    }

    if (score >= 30) {
        // Soft friction — add artificial delay to slow bulk abuse
        await new Promise(r => setTimeout(r, 500));
    }

    // Score < 80 and OTP satisfied (if needed) — process the booking
    await redis.incr(`ip:${req.ip}:orders:1h`);
    await redis.expire(`ip:${req.ip}:orders:1h`, 3600);
    const booking = await createBooking(req.body, req.user);
    return res.status(201).json({ booking });
});
```

The key properties: the score is read from Redis using a server-set session ID — the client has no way to supply or modify it. Sessions with no telemetry aren't assumed safe; they're scored on network signals alone. Each tier enforces a concrete policy rather than just logging.

---

## 4. Worked Examples

### Example A: Legitimate user booking a home service

1. User opens HULP booking page on Chrome mobile
2. Telemetry collects: natural mouse movements (entropy = 0.73), 3.2s on page, human typing intervals (uniformity = 0.4)
3. Click latency = 1,240ms — clearly human
4. JA3 matches real Chrome mobile fingerprint
5. IP is a residential Airtel connection (not datacenter)
6. **Score: 0 → ALLOW**
7. User completes booking with zero friction. No awareness of any defense layer.

---

### Example B: Bot attempting bulk fake slot bookings (the hard case)

**Scenario:** A competitor bulk-books slots to make HULP appear unavailable. They use a Playwright script with a spoofed user-agent.

| Check | Signal Value | Score Contribution |
|---|---|---|
| Mouse entropy | `null` — Playwright generates no mouse events; fewer than 3 points recorded, collector returns null | +0 |
| Click latency | 48ms (sub-human) | +10 |
| Keystroke uniformity | `null` — `page.fill()` bypasses keydown events entirely; no inter-key intervals recorded | +0 |
| IP reputation | AWS us-east-1 egress | +20 |
| JA3 hash | Matches known Playwright fingerprint | +20 |
| Time on page | 0.3 seconds | +10 |
| **Total** | | **60 → Hard Friction (OTP)** |

The session hits the OTP tier. The attacker has no access to the OTP inbox — blocked. The request does not receive an outright `429`; it instead hits the OTP wall, which gives the attacker no information about whether they were detected or just need to verify.

**Edge case within this example:** What if the attacker switches to a residential proxy? The IP reputation check (+20) is now missed. Score drops from 60 to 40 — into the **Soft Friction** zone (artificial delay + rate limit). Combined with JA3 flagging and the OTP tier they're approaching, bulk abuse becomes operationally difficult. The JA3 signal still fires independently of IP, so the attacker needs to also patch their TLS fingerprint to meaningfully evade.

---

## 5. Failure Modes and Trade-offs

| Failure | Detection | Recovery | What I chose NOT to handle |
|---|---|---|---|
| **False positive — real user flagged** | High score but successful OTP completion | OTP success resets score to 0 for the session | Power users who click extremely fast — accept a small FP rate |
| **Telemetry blocked (ad blocker)** | No beacon received for session | Fall back to network signals only; score on those alone | Forcing telemetry through (adversarial to legitimate privacy users) |
| **Bot with residential proxy + slow timing** | JA3 still mismatches real Chrome | JA3 signal alone flags it | Bots that patch their Playwright binary to match JA3 — requires dedicated threat response |
| **JA3 list goes stale** | New Playwright version changes hash | Regular JA3 list sync via threat intel feed | Maintaining a fully up-to-date JA3 database in-house — use a vendor (DataDome, Cloudflare) |
| **OTP abuse via SIM farm** | Many OTP requests from same IP | Rate-limit OTP sends: max 3/hour per IP | Nation-state-level abuse (out of scope for a startup) |

**Biggest trade-off:** Passive behavioral signals require JavaScript. If users have JS disabled or use aggressive privacy tools, telemetry is incomplete. I chose to fall back to network-only scoring rather than block non-JS users — blocking them would harm legitimate privacy-conscious users and is the wrong call.

---

## 6. Operations: Monitoring, Alerting, Scaling, Cost

### Monitoring
- **Score distribution histogram:** should be bimodal (most users < 30, bots > 70). If it flattens, signals are degrading — but a bimodal shape alone doesn't confirm the system is working correctly; it means signals are separating sessions, not that the separation is accurate.
- **OTP trigger rate:** % of sessions hitting hard-friction tier. A spike may indicate a bot wave; a slow creep may indicate signal drift or a shifting user population. Neither is definitive without cross-referencing other metrics.
- **Legitimate-user checkout conversion rate:** The most direct signal of false-positive harm. If conversion drops after a threshold change, real users are being caught. Track separately from raw block rate.
- **OTP / verification failure rate:** Sessions that hit OTP but fail to complete it. High failure rate among otherwise normal-looking sessions is a false-positive signal; high failure rate among high-scored sessions is expected (bots can't retrieve OTPs).
- **Confirmed abuse rate:** Ground truth from the fraud/ops team — bookings later confirmed as fraudulent. Correlate with their score at booking time to validate whether the scoring is actually catching real abuse, not just high-entropy sessions.
- **False positive rate:** Sessions that passed OTP but were later flagged as fraud — correlate with their score at booking time. Also the inverse: legitimate users who were blocked and contacted support.
- **Block rate per endpoint:** Spikes on specific endpoints suggest targeted attacks — but could also be a new client type or a deployment change. Use as an investigation trigger, not a verdict.

### Alerting

> **These thresholds are investigation triggers, not proof of any specific condition. Each alert should prompt a human review of the relevant metrics before taking action.**

- Block rate > 5% in any 5-minute window → investigate: possible attack wave, misconfigured threshold, or new traffic pattern
- OTP trigger rate > 15% → investigate: thresholds may be too aggressive, or a legitimate user segment is being mis-scored
- Checkout conversion drops > 2% week-over-week → investigate: possible false-positive increase harming real users
- Telemetry receive rate drops below 80% → check beacon endpoint health; scoring will fall back to network signals only

### Scaling
- **Telemetry ingestion:** Stateless Node.js workers behind a load balancer. Each beacon is a lightweight Redis write.
- **Score computation:** Runs synchronously on telemetry receipt. Light enough for single-threaded async workers.
- **Redis:** Redis Cluster for horizontal scaling. Session scores are small (< 1KB each). At 100K concurrent sessions: ~100MB RAM.
- **JA3 extraction:** Done at the nginx layer via `nginx-ja3` module — zero app-layer overhead.

### Cost

> **These are design-target estimates based on public pricing pages, not measured production costs. Actual spend depends on cache hit rate, session length, and traffic distribution. Treat as planning inputs; benchmark against real HULP traffic before committing.**

- **Redis:** ~\$50/month for a managed 1GB cluster at 100K sessions
- **IP reputation API:** ~\$0.001/lookup → ~\$30/day at 30K sessions/day
- **Optimization:** Cache IP reputation lookups in Redis for 1 hour — same IP isn't re-checked on every request

---

# Bonus: How Does Part 1 Survive Part 2?

This is the most interesting question.

**Would BrowserBot (Part 1) be caught by the Part 2 defenses?**

Honestly — **yes, mostly.** And that's by design.

| Part 2 Defense | BrowserBot's Fate | Why |
|---|---|---|
| **Mouse entropy check** | ✅ SURVIVES (this check) | Playwright generates no mouse events, so `mousePoints.length < 3` — `computeMouseEntropy()` returns `null`. The null guard in the scorer skips this signal (+0). BrowserBot is still caught by click latency, JA3, and time-on-page — just not by mouse entropy directly. |
| **Click latency check** | ❌ CAUGHT | Programmatic clicks fire in 30–80ms — well below the 100ms human threshold. |
| **JA3 fingerprint** | ❌ CAUGHT | Playwright Chromium has a distinct JA3 hash. Even though BrowserBot sets a realistic user-agent in `browser.py`, the JA3 betrays the automation layer. This is likely one reason Amazon blocks BrowserBot — I noted it in my README — though Amazon's detection stack is a black box and JA3 may be one of several signals. The UA string lies; the TLS handshake doesn't. |
| **Keystroke uniformity** | ✅ SURVIVES (this check) | `page.fill()` bypasses keydown events entirely — `keystroke_uniformity` is `null`, so this signal contributes 0. BrowserBot is still caught by click latency, JA3, and time-on-page — just not by this one. |
| **Honeypot fields** | ✅ SURVIVES | BrowserBot uses semantic locators (`get_by_placeholder`, `get_by_label`). It targets fields by visible description, not by filling every DOM element. A `display:none` honeypot would never be targeted. |
| **OTP hard friction** | ❌ CAUGHT | BrowserBot has no inbox access. It cannot retrieve or enter an OTP. Blocked. |

**Conclusion:** A BrowserBot-style automation scores **40–60** from behavioral and JA3 signals. Mouse entropy is `null` (fewer than 3 mouse points — the collector returns null, the scorer skips it). The reliable signals: click latency +10, time on page +10, JA3 +20 = **40 minimum**. If it also runs from a cloud/datacenter IP — which is common for server-deployed automation — the IP reputation check adds another +20, pushing the score to **60 → Hard Friction (OTP)**. At 60 it hits the OTP tier and gets blocked there — the attacker has no inbox access.

A determined attacker could invest in evading individual signals, but each layer they bypass requires additional infrastructure and expertise. The defense doesn't need to be unbeatable — it needs to make bulk abuse economically unviable compared to legitimate alternatives.

**The real answer for HULP:** If HULP explicitly authorizes users to automate bookings (e.g., corporate accounts), the right solution is **authenticated API tokens** — give those accounts a server-to-server API key so they bypass the browser flow entirely and never hit bot detection. Part 1 (browser automation) is a last resort for sites *without* an API. HULP should build and expose an API for its power users rather than force them through a browser layer designed for human interaction.

---

## Summary

| | Part 1: Automation | Part 2: Bot Defense |
|---|---|---|
| **Core idea** | LLM re-plans from observed page state; semantic locators resist UI changes | Multi-signal passive scoring; tiered friction for high-risk sessions only |
| **Key resilience** | Semantic waterfall + retry + LLM JSON recovery | JA3 fingerprinting + behavioral entropy + OTP gate |
| **Human in the loop** | Yes — payment actions require explicit confirmation | Yes — OTP only for sessions scoring 60–79 |
| **Biggest weakness** | JA3 fingerprint betrays Playwright even with spoofed UA | Residential proxies + synthetic mouse movement lower the score |
| **Monitoring signal** | Run completion rate | Score distribution histogram |

The two systems are genuinely adversarial to each other — which makes building both an interesting exercise in thinking from both sides of the problem.

---

*Kartikay Shukla | kartikayshukla4141@gmail.com | +91 9336166298*
