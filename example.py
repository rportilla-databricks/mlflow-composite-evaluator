"""
Example usage of the LLMJudge composite evaluator.

NOTE: This won't run out-of-the-box without:
1. Databricks workspace credentials
2. MLflow tracking server
3. Implementing get_databricks_auth() in databricks_config.py
"""

from llm_judge import LLMJudge

def main():
    # Initialize the judge with your Databricks model
    # (You'll need to implement authentication first)
    judge = LLMJudge(model="databricks-claude-sonnet-4")
    
    # Example evaluation
    result = judge.evaluate_response(
        question="What is dbldatagen and how do I use it for testing?",
        
        expected="""dbldatagen is a Python library for generating synthetic test data. 
        You can use it to create realistic datasets for testing data pipelines. 
        Install with pip install dbldatagen, then use dg.DataGenerator() to define schemas.""",
        
        actual="""dbldatagen is a Databricks library for synthetic data generation. 
        You can install it from GitHub and use dg.DataGenerator() to create test datasets. 
        It's particularly useful for testing data pipelines and migrations.""",
        
        facts=[
            "dbldatagen",
            "synthetic data",
            "dg.DataGenerator",
            "testing",
            "Python library"
        ],
        
        test_id="example_1"
    )
    
    # Display results
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    print(f"\n📊 Overall Score: {result['score']:.1f}/10 ({result['category']})")
    print(f"\n🔍 Component Breakdown:")
    print(f"  • Facts Presence:    {result['facts_score']:.1f}/10")
    print(f"  • Answer Relevance:  {result['answer_relevance_score']:.1f}/10")
    print(f"  • Retrieval Quality: {result['retrieval_score']:.1f}/10")
    
    print(f"\n✅ Facts Matched ({len(result['matched_facts'])}/{len(result['matched_facts']) + len(result['missing_facts'])}):")
    for fact in result['matched_facts']:
        print(f"  ✓ {fact}")
    
    if result['missing_facts']:
        print(f"\n❌ Facts Missing:")
        for fact in result['missing_facts']:
            print(f"  ✗ {fact}")
    
    print(f"\n💭 Reasoning: {result['reason']}")
    print(f"\n🏷️  MLflow Run ID: {result['mlflow_run_id']}")
    print("\n" + "="*70)


if __name__ == "__main__":
    try:
        main()
    except NotImplementedError as e:
        print("\n" + "="*70)
        print("⚠️  SETUP REQUIRED")
        print("="*70)
        print(str(e))
        print("\nThis is a reference implementation showing the evaluation pattern.")
        print("To run it, you'll need to implement Databricks authentication.")
        print("\n" + "="*70)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()



