# MLflow Composite Evaluator

**A production-ready LLM evaluation system combining custom metrics with LLM-as-judge patterns.**

This is an evaluation framework used in testing best practices knowledge for Databricks migrations. It showcases how to build a composite evaluation system using MLflow's evaluation framework.

---

## What This Does

Evaluates LLM responses using **three complementary scorers**:

1. **Facts Presence** (50% weight) - Custom metric checking if key facts/terms appear in responses
2. **Answer Relevance** (25% weight) - LLM judges if the answer is relevant to the question
3. **Retrieval Quality** (25% weight) - LLM compares answer quality against a reference

**Composite Score = (Facts × 0.5) + (Relevance × 0.25) + (Retrieval × 0.25)**

---

## Architecture

```python
class LLMJudge:
    def __init__(self):
        # Create 3 MLflow metrics
        self.facts_metric = self._create_facts_metric()           # Custom
        self.answer_relevance_metric = self._create_answer_relevance_metric()  # LLM-as-judge
        self.relevance_metric = self._create_relevance_metric()   # LLM-as-judge
    
    def evaluate_response(self, question, expected, actual, facts):
        # Run MLflow evaluation with all 3 metrics
        eval_data = pd.DataFrame({
            'inputs': [question],
            'predictions': [actual],
            'context': [expected],
            'facts': [facts]
        })
        
        results = mlflow.evaluate(
            data=eval_data,
            extra_metrics=[
                self.facts_metric,
                self.answer_relevance_metric,
                self.relevance_metric
            ]
        )
        
        # Extract and combine scores
        return composite_score
```

---

## Key Patterns Demonstrated

### 1. Custom MLflow Metric with `make_metric()`

```python
def _create_facts_metric(self):
    def facts_eval_fn(predictions: pd.Series, metrics: Dict, facts: pd.Series) -> MetricValue:
        scores = []
        for prediction, facts_list in zip(predictions, facts):
            matched = sum(1 for fact in facts_list if fact.lower() in prediction.lower())
            score = (matched / len(facts_list)) * 5.0  # 0-5 scale
            scores.append(score)
        
        return MetricValue(
            scores=scores,
            aggregate_results={
                'mean': sum(scores) / len(scores),
                'min': min(scores),
                'max': max(scores)
            }
        )
    
    return mlflow.metrics.make_metric(
        eval_fn=facts_eval_fn,
        greater_is_better=True,
        name="facts_presence",
        version="v1"
    )
```

**Key insight:** Custom metrics must return `MetricValue` objects with `scores` and `aggregate_results`.

---

### 2. LLM-as-Judge Pattern

```python
def _create_answer_relevance_metric(self):
    def answer_relevance_fn(inputs: pd.Series, predictions: pd.Series) -> MetricValue:
        scores = []
        for question, answer in zip(inputs, predictions):
            prompt = f"""Rate the relevance of this answer (1-5):
            
            Question: {question}
            Answer: {answer}
            
            Score:"""
            
            # Call LLM
            response = httpx.post(
                f"{self.workspace_host}/serving-endpoints/{self.model}/invocations",
                headers={'Authorization': f'Bearer {self.token}'},
                json={'messages': [{'role': 'user', 'content': prompt}], 'max_tokens': 10}
            )
            score = float(response.json()['choices'][0]['message']['content'].strip())
            scores.append(score)
        
        return MetricValue(scores=scores, aggregate_results={...})
    
    return mlflow.metrics.make_metric(eval_fn=answer_relevance_fn, ...)
```

**Key insight:** You can create LLM-as-judge metrics that call any LLM endpoint for scoring.

---

### 3. Composite Scoring

```python
# Extract scores from MLflow results
facts_score = metrics_dict.get('facts_presence/v1/mean', 0.0) * 2  # Convert 0-5 to 0-10
answer_relevance_score = metrics_dict.get('answer_relevance/v1/mean', 0.0) * 2
retrieval_score = metrics_dict.get('relevance/v1/mean', 0.0) * 2

# Calculate weighted composite
composite_score = (facts_score * 0.5 + 
                  answer_relevance_score * 0.25 + 
                  retrieval_score * 0.25)
```

**Key insight:** MLflow metrics can be combined with custom weights for composite scores.

---

## Quick Start

### 1. Setup
```bash
# Clone/download the project
cd mlflow-composite-evaluator

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Databricks Authentication

Create `~/.databrickscfg` in your home directory:
```ini
[DEFAULT]
host = https://your-workspace.cloud.databricks.com
token = dapi...
```

### 3. Run the Example
```bash
python example.py
```

### 4. See Your Results

```
======================================================================
CUSTOM FACTS SCORER - EVALUATION RESULTS
======================================================================

Overall Score: 7.5/10 (GOOD)

Component Breakdown:
  Facts Presence:    10.0/10 (Custom Scorer)
  Answer Relevance:  2.0/10 (LLM-as-judge)
  Retrieval Quality: 8.0/10 (LLM-as-judge)

Facts Matched (5/5):
  [+] dbldatagen
  [+] dg.DataGenerator
  [+] Spark
  [+] distributed
  [+] PyPI

Reasoning: Facts: 10.0/10 (5/5 matched)

MLflow Run ID: 1ca42e258170456a9fa7feca7dc7d642

======================================================================
```

---

## Customization

### To Adapt for Your Use Case:

1. **Change the LLM endpoint** in `_create_answer_relevance_metric()` and `_create_relevance_metric()`
2. **Adjust the facts metric** to match your domain (e.g., check for citations, code snippets, etc.)
3. **Tune the weights** in the composite score calculation (currently 50/25/25)
4. **Modify the test case** in `example.py` with your own questions and facts

---

## What You Can Learn

1. **How to create custom MLflow metrics** using `make_metric()`
2. **How to implement LLM-as-judge** within MLflow's framework
3. **How to combine multiple evaluation metrics** into a composite score
4. **How to handle DataFrames and Series** in MLflow evaluations
5. **How to return proper `MetricValue` objects** with aggregate results

---

## What's Included

- **Working authentication** - Reads from `~/.databrickscfg`
- **Complete example** - Test case with real migration question
- **Custom facts scorer** - Checks for key technical terms
- **LLM-as-judge integration** - Calls Databricks endpoints
- **Colorized terminal output** - Beautiful results display
- **MLflow integration** - Full run tracking

---

## Real-World Usage

This pattern is used for evaluating Databricks migration best practices responses, where you need to verify that answers contain specific technical terms, tools, and concepts.

**Example evaluation breakdown:**
```
Overall Score: 7.5/10 (GOOD)
  Facts present: 5/5 (dbldatagen, dg.DataGenerator, Spark, distributed, PyPI)
  Answer relevance: 2.0/10 (LLM-judged)
  Retrieval quality: 8.0/10 (LLM-judged)
```

---

## Further Reading

- [MLflow Evaluation Metrics](https://mlflow.org/docs/latest/llms/llm-evaluate/index.html)
- [Building Custom Metrics](https://mlflow.org/docs/latest/llms/llm-evaluate/index.html#creating-custom-llm-evaluation-metrics)
- [LLM-as-a-Judge Pattern](https://eugeneyan.com/writing/llm-evaluators/)

---

## License

MIT - Use this pattern in your own projects!



