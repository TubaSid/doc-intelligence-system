#!/usr/bin/env python
"""Test suite for optimized document intelligence system"""
import requests
import json

BASE_URL = "http://localhost:8000/api/v1"

def test_query(question):
    """Test a single query"""
    response = requests.post(
        f"{BASE_URL}/query",
        json={"query": question}
    )
    data = response.json()
    
    print(f"\n{'='*70}")
    print(f"Q: {question}")
    print(f"{'='*70}")
    
    # Handle error responses
    if 'detail' in data:
        print(f"❌ Error: {data['detail']}")
        return None
    
    if 'answer' not in data:
        print(f"⚠️ Response missing 'answer' field")
        print(f"Response: {json.dumps(data, indent=2)[:500]}")
        return None
    
    print(f"Answer: {data['answer'][:200]}...")
    print(f"\nConfidence: {data['confidence']:.1%}")
    print(f"Retrieval Score: {data['retrieval_score']:.4f}")
    print(f"Sources: {len(data['sources'])} documents")
    return data

if __name__ == "__main__":
    print("\n🚀 TESTING OPTIMIZED DOCUMENT INTELLIGENCE SYSTEM")
    print("=" * 70)
    
    test_cases = [
        "How many vehicles delivered in 2023?",
        "What was Tesla's total revenue in 2023?",
        "How much energy storage was deployed in 2023?",
        "What is Tesla's net income?",
        "Manufacturing locations and facilities",
        "Gross profit automotive segment"
    ]
    
    results = []
    for query in test_cases:
        result = test_query(query)
        if result:  # Only add successful results
            results.append({
                "query": query,
                "confidence": result["confidence"],
                "score": result["retrieval_score"]
            })
    
    if not results:
        print("\n❌ No successful test results")
        exit(1)
    
    print(f"\n\n{'='*70}")
    print("📊 TEST SUMMARY")
    print(f"{'='*70}")
    print(f"\n{'Query':<45} {'Confidence':<15} {'Score':<10}")
    print("-" * 70)
    for r in results:
        print(f"{r['query']:<45} {r['confidence']:<15.1%} {r['score']:<10.4f}")
    
    avg_confidence = sum(r['confidence'] for r in results) / len(results)
    print(f"\n✅ Average Confidence: {avg_confidence:.1%}")
    print(f"✅ Total Tests: {len(results)}")
    print(f"✅ Successful Answers (confidence > 30%): {sum(1 for r in results if r['confidence'] > 0.3)}/{len(results)}")
