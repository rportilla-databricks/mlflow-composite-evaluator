"""
Standalone test for the custom facts scorer.
This demonstrates the facts-based evaluation WITHOUT requiring Databricks credentials.
"""

import pandas as pd
from mlflow.metrics import MetricValue

def create_facts_metric_standalone():
    """Standalone version of the facts metric for testing."""
    import mlflow.metrics
    
    def facts_eval_fn(predictions: pd.Series, metrics: dict, facts: pd.Series) -> MetricValue:
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
        details = []
        
        for idx, (prediction, facts_list) in enumerate(zip(predictions, facts)):
            if not facts_list or len(facts_list) == 0:
                scores_list.append(5.0)  # No facts to check = perfect
                details.append({"matched": [], "missing": [], "score": 5.0})
                continue
            
            prediction_lower = str(prediction).lower()
            matched = [fact for fact in facts_list if fact.lower() in prediction_lower]
            missing = [fact for fact in facts_list if fact.lower() not in prediction_lower]
            
            # Convert to 0-5 scale (MLflow standard)
            score = (len(matched) / len(facts_list)) * 5.0
            scores_list.append(score)
            details.append({
                "matched": matched,
                "missing": missing,
                "score": score,
                "total_facts": len(facts_list)
            })
            
            # Print details for this test case
            print(f"\n  Test Case {idx + 1}:")
            print(f"    Facts matched: {len(matched)}/{len(facts_list)}")
            print(f"    Score: {score:.2f}/5.0 ({score*2:.1f}/10.0)")
            if matched:
                print(f"    ✅ Matched: {', '.join(matched)}")
            if missing:
                print(f"    ❌ Missing: {', '.join(missing)}")
        
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
        metric_details="Checks if expected key facts/terms appear in the response."
    )


def main():
    print("="*70)
    print("TESTING CUSTOM FACTS SCORER")
    print("="*70)
    print("\nThis test demonstrates the custom MLflow metric WITHOUT Databricks auth")
    print("The facts scorer checks if key terms appear in the response.\n")
    
    # Test Case 1: Perfect match - all facts present
    test_case_1 = {
        "question": "What is dbldatagen and how do I use it?",
        "answer": """dbldatagen is a Python library for generating synthetic test data.
        You can use dg.DataGenerator() to create realistic datasets for testing data pipelines.
        It's particularly useful for testing migrations and available on GitHub.""",
        "facts": ["dbldatagen", "synthetic data", "dg.DataGenerator", "testing", "Python library"]
    }
    
    # Test Case 2: Partial match - some facts missing
    test_case_2 = {
        "question": "How do I migrate from Teradata to Databricks?",
        "answer": """You can use Lakebridge for the migration process. 
        It handles SQL conversion automatically and supports stored procedures.""",
        "facts": ["Lakebridge", "SQL conversion", "stored procedures", "testing", "validation"]
    }
    
    # Test Case 3: Poor match - most facts missing
    test_case_3 = {
        "question": "What security features does Databricks have?",
        "answer": """Databricks has good security features and compliance support.""",
        "facts": ["Unity Catalog", "RBAC", "encryption", "audit logs", "compliance frameworks"]
    }
    
    test_cases = [test_case_1, test_case_2, test_case_3]
    
    # Prepare data for MLflow evaluation
    eval_data = pd.DataFrame({
        'inputs': [tc['question'] for tc in test_cases],
        'predictions': [tc['answer'] for tc in test_cases],
        'facts': [tc['facts'] for tc in test_cases]
    })
    
    print("Running facts scorer on 3 test cases...")
    print("-" * 70)
    
    # Create the custom metric
    facts_metric = create_facts_metric_standalone()
    
    # Run the metric evaluation
    import mlflow
    
    with mlflow.start_run(run_name="facts_scorer_test"):
        results = mlflow.evaluate(
            data=eval_data,
            predictions='predictions',
            model_type='text',
            evaluators='default',
            extra_metrics=[facts_metric],
            evaluator_config={
                'col_mapping': {
                    'inputs': 'inputs',
                    'predictions': 'predictions',
                    'facts': 'facts'
                }
            }
        )
    
    print("\n" + "-" * 70)
    print("AGGREGATE RESULTS:")
    print("-" * 70)
    
    metrics = results.metrics
    mean_score = metrics.get('facts_presence/v1/mean', 0.0)
    min_score = metrics.get('facts_presence/v1/min', 0.0)
    max_score = metrics.get('facts_presence/v1/max', 0.0)
    
    print(f"\n  Average Score: {mean_score:.2f}/5.0 ({mean_score*2:.1f}/10.0)")
    print(f"  Min Score:     {min_score:.2f}/5.0 ({min_score*2:.1f}/10.0)")
    print(f"  Max Score:     {max_score:.2f}/5.0 ({max_score*2:.1f}/10.0)")
    
    print("\n" + "="*70)
    print("✅ CUSTOM FACTS SCORER TEST COMPLETED")
    print("="*70)
    print("\nThis demonstrates:")
    print("  ✓ Custom MLflow metric using make_metric()")
    print("  ✓ Facts-based scoring (key term matching)")
    print("  ✓ Weighted scoring based on percentage of facts found")
    print("  ✓ Works WITHOUT requiring LLM calls or authentication")
    print("\nThe full composite evaluator adds:")
    print("  • Answer relevance scorer (LLM-as-judge)")
    print("  • Retrieval quality scorer (LLM-as-judge)")
    print("  • Weighted composite (50% facts, 25% relevance, 25% retrieval)")
    print("\n" + "="*70)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

