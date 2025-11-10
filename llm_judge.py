"""
LLM-as-Judge evaluation for QA testing using MLflow.
Uses MLflow's evaluate() API with:
- Custom facts-based metric (mlflow.metrics.make_metric)
- MLflow answer_relevance scorer
- MLflow relevance scorer (retrieval quality)
"""

import sys
from typing import Dict, Any, List
from pathlib import Path
import pandas as pd
import logging
import warnings
import os

# Suppress noisy warnings and progress bars
logging.getLogger("mlflow").setLevel(logging.ERROR)
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', module='numpy')

# Disable all progress bars
os.environ['MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR'] = 'false'
os.environ['TQDM_DISABLE'] = '1'

import mlflow
import mlflow.metrics
from mlflow.metrics import MetricValue
import mlflow.metrics.genai as genai

# Databricks authentication (you'll need to implement this)
from databricks_config import get_databricks_auth


class LLMJudge:
    """
    MLflow-based LLM-as-Judge using composite scoring.
    
    Combines three evaluation approaches:
    1. Facts-based scoring (custom metric) - checks if key facts are present
    2. answer_relevance (MLflow built-in) - evaluates answer relevance to question
    3. relevance (MLflow built-in) - evaluates retrieval quality with context
    """
    
    def __init__(self, model: str = "databricks-claude-sonnet-4"):
        """
        Initialize the MLflow evaluation system.
        
        Args:
            model: Model endpoint to use for MLflow scorers
            
        Raises:
            ValueError: If Databricks authentication is not configured
        """
        self.model = model
        
        # Get Databricks credentials using the same method as the rest of the codebase
        # This will FAIL LOUDLY if .databrickscfg is not configured
        auth = get_databricks_auth()
        self.workspace_host = auth['host']
        self.token = auth['token']
        
        print(f"🔐 Using Databricks authentication via {auth.get('method', 'unknown')}")
        
        # Create metrics
        self.facts_metric = self._create_facts_metric()
        self.answer_relevance_metric = self._create_answer_relevance_metric()
        self.relevance_metric = self._create_relevance_metric()
        
        # Report initialization status
        print(f"✅ Composite evaluation system initialized")
        print(f"   - Custom facts scorer (50% weight)")
        print(f"   - MLflow answer_relevance (25% weight)")
        print(f"   - MLflow relevance/retrieval quality (25% weight)")
    
    def _create_facts_metric(self):
        """Create custom facts-based metric using mlflow.metrics.make_metric()."""
        
        def facts_eval_fn(predictions: pd.Series, metrics: Dict[str, Any], facts: pd.Series) -> MetricValue:
            """
            Evaluate if key facts are present in predictions.
            
            Args:
                predictions: Series of model responses
                metrics: Dict of other computed metrics
                facts: Series containing lists of expected facts for each prediction
                
            Returns:
                MetricValue with scores and aggregate results
            """
            scores_list = []
            
            for prediction, facts_list in zip(predictions, facts):
                if not facts_list or len(facts_list) == 0:
                    scores_list.append(5.0)  # No facts to check = perfect
                    continue
                
                prediction_lower = str(prediction).lower()
                matched = sum(1 for fact in facts_list if fact.lower() in prediction_lower)
                # Convert to 0-5 scale (MLflow standard)
                score = (matched / len(facts_list)) * 5.0
                scores_list.append(score)
            
            # Return MetricValue with aggregate results
            return MetricValue(
                scores=scores_list,
                aggregate_results={
                    'mean': sum(scores_list) / len(scores_list) if scores_list else 0.0,
                    'min': min(scores_list) if scores_list else 0.0,
                    'max': max(scores_list) if scores_list else 0.0
                }
            )
        
        return mlflow.metrics.make_metric(
            eval_fn=facts_eval_fn,
            greater_is_better=True,
            name="facts_presence",
            long_name="Facts Presence Score",
            version="v1",
            metric_details="Checks if expected key facts/terms appear in the response. Score 0-5 based on percentage of facts found."
        )
    
    def _create_answer_relevance_metric(self):
        """Create CUSTOM answer_relevance scorer using Databricks LLM."""
        import httpx
        
        def answer_relevance_fn(inputs: pd.Series, predictions: pd.Series) -> MetricValue:
            """Custom answer relevance scorer that calls Databricks LLM."""
            scores = []
            
            for question, answer in zip(inputs, predictions):
                prompt = f"""Rate the relevance of this answer to the question on a scale of 1-5.

Question: {question}

Answer: {answer}

Respond with ONLY a number from 1-5:
1 = Completely irrelevant
2 = Somewhat relevant but missing key points
3 = Moderately relevant
4 = Very relevant with minor gaps
5 = Perfectly relevant and comprehensive

Score:"""
                
                try:
                    # Call Databricks LLM for scoring
                    endpoint_url = f"{self.workspace_host}/serving-endpoints/{self.model}/invocations"
                    response = httpx.post(
                        endpoint_url,
                        headers={
                            'Authorization': f'Bearer {self.token}',
                            'Content-Type': 'application/json'
                        },
                        json={'messages': [{'role': 'user', 'content': prompt}], 'max_tokens': 10},
                        timeout=30.0
                    )
                    response.raise_for_status()
                    result = response.json()
                    score_text = result['choices'][0]['message']['content'].strip()
                    score = float(score_text)
                    scores.append(score)
                except Exception as e:
                    print(f"   Warning: answer_relevance scoring failed: {e}")
                    scores.append(3.0)  # Default middle score on error
            
            return MetricValue(
                scores=scores,
                aggregate_results={
                    'mean': sum(scores) / len(scores) if scores else 0.0,
                    'min': min(scores) if scores else 0.0,
                    'max': max(scores) if scores else 0.0
                }
            )
        
        return mlflow.metrics.make_metric(
            eval_fn=answer_relevance_fn,
            greater_is_better=True,
            name="answer_relevance",
            long_name="Answer Relevance Score",
            version="v1",
            metric_details="LLM-as-judge scorer for answer relevance using Databricks Claude"
        )
    
    def _create_relevance_metric(self):
        """Create CUSTOM retrieval quality scorer using Databricks LLM."""
        import httpx
        
        def retrieval_quality_fn(inputs: pd.Series, predictions: pd.Series, context: pd.Series) -> MetricValue:
            """Custom retrieval quality scorer that calls Databricks LLM."""
            scores = []
            
            for question, answer, expected in zip(inputs, predictions, context):
                prompt = f"""Rate the quality of this answer compared to the reference on a scale of 1-5.

Question: {question}

Reference Answer: {expected}

Actual Answer: {answer}

Respond with ONLY a number from 1-5:
1 = Missing most key information from reference
2 = Contains some relevant info but incomplete
3 = Covers main points adequately
4 = Comprehensive with minor differences
5 = Excellent, matches or exceeds reference quality

Score:"""
                
                try:
                    # Call Databricks LLM for scoring
                    endpoint_url = f"{self.workspace_host}/serving-endpoints/{self.model}/invocations"
                    response = httpx.post(
                        endpoint_url,
                        headers={
                            'Authorization': f'Bearer {self.token}',
                            'Content-Type': 'application/json'
                        },
                        json={'messages': [{'role': 'user', 'content': prompt}], 'max_tokens': 10},
                        timeout=30.0
                    )
                    response.raise_for_status()
                    result = response.json()
                    score_text = result['choices'][0]['message']['content'].strip()
                    score = float(score_text)
                    scores.append(score)
                except Exception as e:
                    print(f"   Warning: retrieval_quality scoring failed: {e}")
                    scores.append(3.0)  # Default middle score on error
            
            return MetricValue(
                scores=scores,
                aggregate_results={
                    'mean': sum(scores) / len(scores) if scores else 0.0,
                    'min': min(scores) if scores else 0.0,
                    'max': max(scores) if scores else 0.0
                }
            )
        
        return mlflow.metrics.make_metric(
            eval_fn=retrieval_quality_fn,
            greater_is_better=True,
            name="relevance",
            long_name="Retrieval Quality Score",
            version="v1",
            metric_details="LLM-as-judge scorer for retrieval quality using Databricks Claude"
        )
    
    def _score_to_category(self, score: float) -> str:
        """Convert numeric score (0-10) to category."""
        if score >= 9:
            return 'excellent'
        elif score >= 7:
            return 'good'
        elif score >= 5:
            return 'fair'
        elif score >= 3:
            return 'poor'
        else:
            return 'failing'
    
    def evaluate_response(
        self,
        question: str,
        expected: str,
        actual: str,
        facts: List[str] = None,
        test_id: str = None
    ) -> Dict[str, Any]:
        """
        Evaluate a response using MLflow's evaluate() API.
        
        Args:
            question: The original question
            expected: The expected/reference answer (used as context for relevance)
            actual: The actual response generated
            facts: List of key facts that should be present in response
            test_id: Optional test identifier
            
        Returns:
            Dictionary with composite score and individual scorer results
        """
        try:
            facts = facts or []
            
            # Prepare data for MLflow evaluate()
            # Include facts as a column so custom metric can access it
            eval_data = pd.DataFrame({
                'inputs': [question],
                'predictions': [actual],
                'context': [expected],  # Used by relevance metric
                'facts': [facts]  # Facts list for custom metric
            })
            
            # Prepare metrics list - always use all 3 scorers
            metrics_list = [
                self.facts_metric,
                self.answer_relevance_metric,
                self.relevance_metric
            ]
            
            # Prepare evaluator config
            evaluator_config = {
                'col_mapping': {
                    'inputs': 'inputs',
                    'predictions': 'predictions',
                    'context': 'context',
                    'facts': 'facts'  # Map facts column
                }
            }
            
            # Run MLflow evaluation
            with mlflow.start_run(nested=True) as run:
                results = mlflow.evaluate(
                    data=eval_data,
                    predictions='predictions',
                    targets=None,
                    model_type='text',
                    evaluators='default',
                    extra_metrics=metrics_list,
                    evaluator_config=evaluator_config
                )
            
            # Extract scores from MLflow results
            # results.metrics is a dict with metric names as keys
            metrics_dict = results.metrics
            
            # Extract scores from MLflow results
            facts_score = metrics_dict.get('facts_presence/v1/mean', 0.0) * 2  # Convert 0-5 to 0-10
            answer_relevance_score = metrics_dict.get('answer_relevance/v1/mean', 0.0) * 2  # Convert 0-5 to 0-10
            retrieval_score = metrics_dict.get('relevance/v1/mean', 0.0) * 2  # Convert 0-5 to 0-10
            
            # Calculate matched/missing facts
            actual_lower = actual.lower()
            matched_facts = [f for f in facts if f.lower() in actual_lower]
            missing_facts = [f for f in facts if f.lower() not in actual_lower]
            
            # Calculate composite score (50% facts, 25% answer_relevance, 25% retrieval)
            composite_score = (facts_score * 0.5 + 
                             answer_relevance_score * 0.25 + 
                             retrieval_score * 0.25)
            
            category = self._score_to_category(composite_score)
            
            # Build evaluation result
            evaluation = {
                'score': round(composite_score, 2),
                'facts_score': round(facts_score, 2),
                'answer_relevance_score': round(answer_relevance_score, 2),
                'retrieval_score': round(retrieval_score, 2),
                'category': category,
                'reason': f"Facts: {facts_score:.1f}/10 ({len(matched_facts)}/{len(facts)} matched)",
                'facts_matched': f"{len(matched_facts)}/{len(facts)} facts present",
                'matched_facts': matched_facts,
                'missing_facts': missing_facts,
                'test_id': test_id,
                'model': self.model,
                'method': 'mlflow_composite',
                'judge_type': 'mlflow_evaluate',
                'mlflow_run_id': run.info.run_id,
                # Breakdown scores for transparency
                'accuracy': round(facts_score / 10 * 4, 1),  # 40% of total for display
                'completeness': round(facts_score / 10 * 4, 1),
                'relevance': round(answer_relevance_score / 10 * 2, 1),  # 20% of total
                'clarity': round(retrieval_score / 10 * 2, 1)  # 20% of total
            }
            
            return evaluation
            
        except Exception as e:
            # FAIL LOUDLY - no fallbacks!
            print(f"❌ FATAL: MLflow evaluation failed: {e}")
            raise
    
    def evaluate_batch(self, test_cases: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        Evaluate a batch of test cases.
        
        Args:
            test_cases: List of dicts with 'question', 'expected', 'actual', 'facts'
            
        Returns:
            List of evaluation results
        """
        results = []
        for i, test_case in enumerate(test_cases):
            result = self.evaluate_response(
                question=test_case.get('question', ''),
                expected=test_case.get('expected', ''),
                actual=test_case.get('actual', ''),
                facts=test_case.get('facts', []),
                test_id=test_case.get('test_id', str(i))
            )
            results.append(result)
        
        return results
