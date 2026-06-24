Multi-Agent Ticket Processing Engine (LangGraph & Pydantic)

An automated customer support triage and quality assurance workflow built using LangGraph and Gemini 2.5 Flash. The engine processes single ticket inputs or bulk batches via CSV, dynamically tracks confidence scores, forks for human intervention where uncertainty exists, and subjects machine outputs to a strict automated Quality Assurance (QA) Critic.
## System Architecture Diagram
Code snippet

graph TD
    START((START)) --> LLM_Classify[LLm_Classify_priority]
    LLM_Classify --> Resp_Node[Response_Node]
    Resp_Node --> Routing{Routing Dummy Node}
    
    Routing -. Conditional Edge .-> Div[Condition_Divider]
    
    Div -->|Confidence <= 0.45 OR > 0.60| Critic[Critic_Node]
    Div -->|Confidence between 0.46 and 0.60 OR None| Human[Human_Review_Node]
    
    Critic --> END((END))
    Human --> END
   ![Workflow Diagram](https://github.com/valevoyage2026-sudo/AI-Engineer-Take-Home-Technical-Task/blob/main/Data/Outputs/Batch_Loader.png)

## Node Reference Guide

Each processing step is broken out into a distinct node within the LangGraph runtime state machine. Isolating these behaviors enforces state encapsulation and permits granular schema verification at every step:
### 1. LLm_Classify_priority

    Purpose: Performs core classification of raw ticket metadata.

    Why it's isolated: Separating categorization from evaluative tasks ensures the LLM focuses purely on extraction. It extracts the structural Tcategory and initial Tpriority using Pydantic structured outputs (with_structured_output).

    SLA Rule Enforcement: This node intercepts predictions for AccountTier.VIP and forces priority to URGENT via standard Python logic, guaranteeing that corporate accounts never slip through standard heuristics.

### 2. Response_Node

    Purpose: Generates a confidence score (∈[0.0,1.0]) and maps context reasoning.

    Why it's isolated: It acts as a meta-analysis engine. By supplying both the original customer input and the upstream classifications made by LLm_Classify_priority, the model can calculate classification accuracy without bias from the generation step.

### 3. Routing

    Purpose: Serving as a structural synchronization gate.

    Why it's isolated: It provides a stable anchor point from which LangGraph can evaluate conditional routing logic based on the completed execution state of all upstream classification data.

### 4. Critic_Node

    Purpose: Automated Quality Assurance.

    Why it's isolated: Acting as a strict internal auditor, it evaluates whether the assigned fields logically match the client query. It outputs an explicit Valid/Invalid flag alongside a tailored customer message.

### 5. Human_Review_Node

    Purpose: Air-gapped manual override.

    Why it's isolated: This node pauses the autonomous execution state when confidence falls into an ambiguous range, spinning up an interactive CLI prompt for human intervention. It features structured try-except loops to capture and correct operator typing mistakes.

## Conditional Routing Rules

The workflow trajectory relies on the Condition_Divider function to process the system state. Routing executes as follows:

    Human Review Branch (Human_Review_Node): Triggered if state.confidence is completely absent (None), or if the calculated metric settles inside the uncertainty threshold:
    0.45<confidence≤0.60

    Automated Audit Branch (Critic_Node): Triggered when confidence is cleanly outside the warning threshold (≤0.45 or >0.60). The ticket routes straight to the automated QA step.

## The Self-Evaluation Mechanism

The self-evaluation process forms the backbone of our system safety guardrails:

[System State Log] -> Evaluated by Critic Node
  ├── 1. Category Alignment Check
  ├── 2. Priority & SLA Threshold Check
  └── 3. Confidence Verification (> 0.50 Metric)

    What it verifies: The Critic_Node evaluates state alignment against a strict system prompt checklist:

        Verification that the categorical assignment reflects the semantic meaning of the message body.

        Ensuring priority rules align with tier conditions.

        Hard verification that the computed classification confidence exceeds 0.50.

    Action plan on failures: If the structural parameters pass, the ticket is flagged as "Valid". If it encounters mismatched properties or weak logic, it flags the run as "Invalid". This flag serves as a trigger point for downstream retry nodes or automated routing back to support desks.

## Architectural Tradeoffs

    Sequential Execution vs. Pure Parallelization:

        Tradeoff: LLm_Classify_priority executes strictly before Response_Node.

        Reasoning: While this adds latency compared to parallel processing, it enables the confidence assessment node to evaluate the actual output of the classification stage, which leads to much higher evaluation reliability.

    Local vs. Cloud-Native Models:

        Tradeoff: Transitioned from ChatOllama (Gemma3:4b) to ChatGoogleGenerativeAI (Gemini 2.5 Flash).

        Reasoning: Smaller local models struggle with complex constraint verification and structured JSON outputs, often defaulting to over-cautious behaviors. Switching to Gemini 2.5 Flash yields reliable schema parsing and reasoning.

## Future Roadmap Improvements

    Autonomous Retry Loops: Implement a feedback edge from Critic_Node back to LLm_Classify_priority. If the critic flags an entry as "Invalid", the system should pass the critic's feedback back to the classification node for an immediate, self-corrected correction attempt.

    Async Batch Parallelization: Optimize the sequential loop inside process_ticket_csv by leveraging Python's asyncio or worker pools. This allows the system to process large CSV ticket exports concurrently rather than sequentially.

    Web-Based Human-in-the-Loop Gateway: Replace the blocking CLI input() functions inside Human_Review_Node with a Slack webhook or a lightweight React review dashboard to make manual interventions production-ready.


 ## Subgraph Architecture & State Encapsulation

To maintain clean separation of concerns, this engine utilizes a hierarchical **Parent-Child Subgraph design pattern**. Rather than building a single sprawling graph, the architecture isolates batch operations from single-ticket analysis.
 ### Parent Graph (`Batch_load`)
* **State Schema**: `batch_schema`
* **Responsibilities**: Handles I/O operations, disk access (`pandas.read_csv` and `to_csv`), and manages the macro-collection of incoming requests.
* **Nodes**: `load_tickets` $\rightarrow$ `process_ticket_csv` $\rightarrow$ `save_to_csv`.

 ### Child Subgraph (`chat`)
* **State Schema**: `Graph_Schema`
* **Responsibilities**: Executes the core agentic runtime—running parallel classification chains, assessing metrics, triggering human CLI overrides, and running the QA Critic.

### State Mapping & Data Flow
When a batch execution begins, the parent graph ingests raw tabular rows and instantiates an array of isolated `Graph_Schema` objects. 

The `process_ticket_csv` node acts as an execution gateway. It iterates through the collection, isolates the state scope, and invokes the `chat` subgraph for each ticket:

```python
# Sequential state mapping loop inside the execution node
for ticket in state.tickets:
    # Control shifts entirely into the autonomous 'chat' subgraph boundaries
    result = chat.invoke(ticket) 
    results.append(result)
```

![Workflow Diagram 2](Bacth_Loader.png)

 ## Batch Processing Node Reference Guide (`Batch_load`)

The parent graph (`Batch_load`) is completely isolated from LLM reasoning. Its single operational objective is file orchestrating—safely managing disk I/O, converting tabular data into valid Pydantic memory schemas, and stepping through your execution pipeline.

---

### 1. `load_tickets`
* **Inputs**: Reads the source file path via `state.csv_path`.
* **Outputs**: Extracts raw data and initializes the `tickets` list array.
* **Functional Mechanics**: 
  This node serves as the system's ingestion gateway. Using `pandas.read_csv()`, it opens the spreadsheet and uses an optimized `.apply(axis=1)` loop to map raw text rows straight into initialized `Graph_Schema` objects.
* **Data Sanitization**: 
  To prevent manual entry typos in your spreadsheet from causing code runtime failures down the road, this node strips whitespace and forces all case values to lowercase when mapping fields to your `AccountTier` Enums:
  ```python
  account_tier=AccountTier(str(row["account_tier"]).strip().lower())

    Why it is isolated:
    Separating file extraction from data processing ensures that any corrupted data format, missing columns, or parsing errors are caught immediately at the boundary before consumption.

### 2. process_ticket_csv

    Inputs: Reads the collection of initialized Graph_Schema objects from state.tickets.

    Outputs: Populates the master state.results list.

    Functional Mechanics:
    This node functions as your Hierarchical Subgraph Gateway. It manages a sequential processing loop. For every ticket object found in the array, it executes chat.invoke(ticket) to pass data to the underlying agent triage graph.

    Schema Validation & Mapping:
    As the child subgraph finishes its run and yields a raw output dictionary, this node runs a strong type verification guard using Pydantic:
    Python

    result = Graph_Schema.model_validate(chat.invoke(ticket))

    This guarantees that the returning data matches the expected structural properties before it is saved into the main results pipeline.

    Why it is isolated:
    It creates an explicit runtime boundary between mass batch data collections and isolated, single-ticket agent evaluations. If an error occurs on a single row, it remains safely sandboxed within this step.

### 3. save_to_csv

    Inputs: Extracts data from state.results.

    Outputs: Writes back to the local file system and returns output_csv_path.

    Functional Mechanics:
    This terminal node formats your collected data back to disk. It loops through the verified results array, serializes the Pydantic models back into flat dictionaries via .model_dump(), packages them into a Pandas DataFrame, and saves a clean CSV output file.

    Why it is isolated:
    Isolating file export behavior ensures that database updates or spreadsheet rewrites are committed cleanly only after the entire batch list has completed every stage of processing.
