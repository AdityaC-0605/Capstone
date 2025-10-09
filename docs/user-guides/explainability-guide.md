# Explainability User Guide

## Overview

This guide provides comprehensive instructions for understanding and using the explainability features of the Sustainable Credit Risk AI system. Our explainable AI capabilities help you understand model decisions, ensure fairness, and meet regulatory requirements.

## Table of Contents

1. [Introduction to Explainable AI](#introduction-to-explainable-ai)
2. [Types of Explanations](#types-of-explanations)
3. [Using the Explanation API](#using-the-explanation-api)
4. [Interpreting SHAP Values](#interpreting-shap-values)
5. [Understanding LIME Explanations](#understanding-lime-explanations)
6. [Attention Mechanism Visualization](#attention-mechanism-visualization)
7. [Counterfactual Explanations](#counterfactual-explanations)
8. [Best Practices](#best-practices)
9. [Regulatory Compliance](#regulatory-compliance)
10. [Troubleshooting](#troubleshooting)

## Introduction to Explainable AI

### What is Explainable AI?

Explainable AI (XAI) refers to methods and techniques that make the outputs of machine learning models understandable to humans. In credit risk assessment, explainability is crucial for:

- **Regulatory Compliance**: Meeting FCRA, ECOA, and GDPR requirements
- **Risk Management**: Understanding model behavior and limitations
- **Customer Service**: Providing clear explanations to loan applicants
- **Model Validation**: Ensuring models make decisions for the right reasons
- **Bias Detection**: Identifying and mitigating unfair treatment

### Why Explainability Matters in Credit Risk

1. **Legal Requirements**: Regulations require explanations for adverse credit decisions
2. **Trust Building**: Transparent decisions build customer and stakeholder trust
3. **Risk Mitigation**: Understanding model behavior reduces operational risk
4. **Fairness**: Detecting and preventing discriminatory practices
5. **Model Improvement**: Insights from explanations guide model enhancement

## Types of Explanations

Our system provides four complementary types of explanations:

### 1. Global Explanations
- **Purpose**: Understand overall model behavior
- **Method**: Feature importance across all predictions
- **Use Case**: Model validation and regulatory reporting

### 2. Local Explanations
- **Purpose**: Understand individual prediction decisions
- **Method**: SHAP values and LIME for specific instances
- **Use Case**: Customer explanations and decision appeals

### 3. Counterfactual Explanations
- **Purpose**: Show what changes would alter the decision
- **Method**: "What-if" scenario analysis
- **Use Case**: Customer guidance and decision optimization

### 4. Attention Visualizations
- **Purpose**: Show which parts of temporal data influenced decisions
- **Method**: Attention weights from neural networks
- **Use Case**: Understanding temporal patterns and model focus

## Using the Explanation API

### Basic Explanation Request

```bash
# Get explanation for a specific prediction
curl -X GET "https://api.credit-risk-ai.example.com/api/v1/explain/app_12345" \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json"
```

**Response Structure:**
```json
{
  "application_id": "app_12345",
  "prediction": {
    "risk_score": 0.23,
    "risk_category": "low",
    "confidence": 0.87
  },
  "explanations": {
    "feature_importance": {...},
    "shap_values": {...},
    "lime_explanation": {...},
    "counterfactuals": [...],
    "attention_weights": {...}
  }
}
```

### Batch Explanations

```bash
# Get explanations for multiple predictions
curl -X POST "https://api.credit-risk-ai.example.com/api/v1/batch/explain" \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "application_ids": ["app_12345", "app_67890"],
    "explanation_types": ["shap", "lime", "counterfactual"]
  }'
```

### Custom Explanation Parameters

```bash
# Request specific explanation types with parameters
curl -X POST "https://api.credit-risk-ai.example.com/api/v1/explain/custom" \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "application_id": "app_12345",
    "explanation_config": {
      "shap": {
        "background_samples": 100,
        "feature_perturbation": "tree_path_dependent"
      },
      "lime": {
        "num_features": 10,
        "num_samples": 5000
      },
      "counterfactual": {
        "max_changes": 3,
        "feature_ranges": "realistic"
      }
    }
  }'
```

## Interpreting SHAP Values

### What are SHAP Values?

SHAP (SHapley Additive exPlanations) values provide a unified framework for interpreting model predictions by fairly attributing the contribution of each feature to the final prediction.

### Key Concepts

1. **Base Value**: The average prediction across all training data
2. **SHAP Value**: The contribution of each feature to moving the prediction away from the base value
3. **Additivity**: Base value + sum of SHAP values = final prediction

### Reading SHAP Explanations

```json
{
  "shap_values": {
    "base_value": 0.147,
    "prediction": 0.23,
    "feature_contributions": {
      "annual_income": 0.045,
      "debt_to_income_ratio": -0.032,
      "credit_history_length": 0.028,
      "employment_length": 0.015,
      "loan_amount": -0.018,
      "age": 0.012,
      "home_ownership": 0.008,
      "payment_frequency": 0.025
    }
  }
}
```

### Interpretation Guidelines

- **Positive SHAP values**: Increase risk score (higher default probability)
- **Negative SHAP values**: Decrease risk score (lower default probability)
- **Magnitude**: Larger absolute values indicate stronger influence
- **Sum Property**: All SHAP values sum to (prediction - base_value)

### Example Interpretation

For the above example:
- **Base risk**: 14.7% (average across all applicants)
- **Final risk**: 23.0% (this specific applicant)
- **Key drivers**:
  - Higher income reduces risk by 4.5 percentage points
  - High debt-to-income ratio increases risk by 3.2 percentage points
  - Long credit history reduces risk by 2.8 percentage points

## Understanding LIME Explanations

### What is LIME?

LIME (Local Interpretable Model-agnostic Explanations) explains individual predictions by learning a local linear approximation around the specific instance.

### LIME Output Structure

```json
{
  "lime_explanation": {
    "local_importance": {
      "annual_income > 70000": 0.34,
      "debt_to_income_ratio <= 0.3": 0.28,
      "credit_history_length > 8": 0.22,
      "employment_length > 3": 0.16
    },
    "intercept": 0.15,
    "score": 0.23,
    "local_prediction": 0.24
  }
}
```

### Interpretation Guidelines

1. **Feature Rules**: LIME creates interpretable rules from complex features
2. **Local Importance**: Shows which rules most influence this specific prediction
3. **Intercept**: The baseline prediction for this local region
4. **Consistency Check**: Compare local_prediction with actual model prediction

### When to Use LIME vs SHAP

| Use LIME When | Use SHAP When |
|---------------|---------------|
| Need simple, rule-based explanations | Need precise feature attributions |
| Explaining to non-technical users | Performing detailed model analysis |
| Local interpretability is sufficient | Global consistency is important |
| Working with categorical features | Working with continuous features |

## Attention Mechanism Visualization

### Understanding Attention Weights

For temporal data (spending patterns, payment history), our neural networks use attention mechanisms to focus on the most relevant time periods.

### Attention Output Structure

```json
{
  "attention_weights": {
    "temporal_attention": {
      "month_1": 0.05,
      "month_2": 0.08,
      "month_3": 0.12,
      "month_4": 0.15,
      "month_5": 0.18,
      "month_6": 0.22,
      "month_7": 0.20
    },
    "feature_attention": {
      "spending_amount": 0.35,
      "payment_timing": 0.28,
      "account_balance": 0.22,
      "transaction_frequency": 0.15
    }
  }
}
```

### Visualization Guidelines

1. **Temporal Patterns**: Higher weights indicate more influential time periods
2. **Feature Focus**: Shows which aspects of temporal data matter most
3. **Seasonal Effects**: May reveal seasonal spending or payment patterns
4. **Recent vs Historical**: Compare recent vs older time period importance

### Creating Attention Heatmaps

```python
import matplotlib.pyplot as plt
import numpy as np

def plot_attention_heatmap(attention_weights, months, features):
    """Create attention heatmap visualization"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Temporal attention
    ax1.bar(months, attention_weights['temporal_attention'].values())
    ax1.set_title('Temporal Attention Weights')
    ax1.set_xlabel('Month')
    ax1.set_ylabel('Attention Weight')
    
    # Feature attention
    ax2.barh(features, attention_weights['feature_attention'].values())
    ax2.set_title('Feature Attention Weights')
    ax2.set_xlabel('Attention Weight')
    
    plt.tight_layout()
    plt.show()
```

## Counterfactual Explanations

### What are Counterfactuals?

Counterfactual explanations answer "What would need to change for a different decision?" They provide actionable insights for applicants and decision-makers.

### Counterfactual Output Structure

```json
{
  "counterfactuals": [
    {
      "feature": "annual_income",
      "current_value": 45000,
      "counterfactual_value": 55000,
      "change_required": 10000,
      "impact": "Risk score would decrease from 0.67 to 0.45",
      "feasibility": "moderate"
    },
    {
      "feature": "debt_to_income_ratio",
      "current_value": 0.45,
      "counterfactual_value": 0.35,
      "change_required": -0.10,
      "impact": "Risk score would decrease from 0.67 to 0.52",
      "feasibility": "high"
    }
  ]
}
```

### Interpretation Guidelines

1. **Actionable Changes**: Focus on features applicants can potentially modify
2. **Feasibility Assessment**: Consider realistic vs unrealistic changes
3. **Multiple Scenarios**: Present several options for improvement
4. **Impact Quantification**: Show expected risk score changes

### Using Counterfactuals for Customer Communication

**Example Customer Explanation:**
> "Your loan application was declined due to high risk assessment. To improve your chances:
> 1. **Reduce debt-to-income ratio** from 45% to 35% (high feasibility) - would improve risk score significantly
> 2. **Increase annual income** by $10,000 (moderate feasibility) - would result in loan approval
> 3. **Extend credit history** by maintaining accounts for 2+ more years (low feasibility) - long-term improvement"

## Best Practices

### For Risk Analysts

1. **Combine Multiple Explanations**: Use SHAP for precision, LIME for simplicity
2. **Validate Explanations**: Ensure explanations align with domain knowledge
3. **Monitor Explanation Stability**: Check consistency across similar cases
4. **Document Decisions**: Keep records of explanation-based decisions

### For Customer Service

1. **Use Simple Language**: Translate technical explanations to plain English
2. **Focus on Actionable Items**: Emphasize changes customers can make
3. **Provide Context**: Explain why certain factors matter for credit risk
4. **Offer Alternatives**: Present multiple paths for improvement

### For Compliance Officers

1. **Maintain Audit Trails**: Keep detailed records of all explanations
2. **Regular Validation**: Periodically validate explanation accuracy
3. **Bias Monitoring**: Use explanations to detect potential bias
4. **Documentation**: Maintain comprehensive explanation documentation

### For Model Developers

1. **Explanation Quality**: Ensure explanations are accurate and meaningful
2. **Performance Monitoring**: Track explanation generation performance
3. **Method Comparison**: Compare different explanation methods
4. **Continuous Improvement**: Update explanation methods based on feedback

## Regulatory Compliance

### FCRA Compliance

**Requirements:**
- Provide specific reasons for adverse actions
- Ensure explanations are accurate and meaningful
- Maintain records of explanation generation

**Implementation:**
```python
def generate_fcra_explanation(shap_values, lime_explanation):
    """Generate FCRA-compliant adverse action reasons"""
    reasons = []
    
    # Extract top negative factors from SHAP
    negative_factors = {k: v for k, v in shap_values.items() if v > 0}
    top_factors = sorted(negative_factors.items(), key=lambda x: x[1], reverse=True)[:4]
    
    reason_mapping = {
        'debt_to_income_ratio': 'Debt-to-income ratio too high',
        'credit_history_length': 'Insufficient credit history',
        'annual_income': 'Income insufficient for loan amount',
        'employment_length': 'Insufficient employment history'
    }
    
    for factor, impact in top_factors:
        if factor in reason_mapping:
            reasons.append(reason_mapping[factor])
    
    return reasons
```

### ECOA Compliance

**Requirements:**
- Ensure explanations don't reveal protected class information
- Provide consistent explanation quality across demographic groups
- Monitor for disparate impact in explanation patterns

### GDPR Compliance

**Requirements:**
- Provide meaningful information about automated decision-making
- Ensure explanations are understandable to data subjects
- Support right to explanation requests

## Troubleshooting

### Common Issues

#### 1. Inconsistent Explanations

**Problem**: SHAP and LIME provide different explanations for the same prediction

**Diagnosis:**
```python
def check_explanation_consistency(shap_values, lime_values):
    """Check consistency between explanation methods"""
    shap_ranking = sorted(shap_values.items(), key=lambda x: abs(x[1]), reverse=True)
    lime_ranking = sorted(lime_values.items(), key=lambda x: abs(x[1]), reverse=True)
    
    # Compare top 5 features
    shap_top5 = [x[0] for x in shap_ranking[:5]]
    lime_top5 = [x[0] for x in lime_ranking[:5]]
    
    overlap = len(set(shap_top5) & set(lime_top5))
    consistency_score = overlap / 5
    
    return consistency_score
```

**Solutions:**
- Increase LIME sample size for more stable explanations
- Use consistent background datasets for SHAP
- Check for feature correlation issues

#### 2. Slow Explanation Generation

**Problem**: Explanations take too long to generate

**Diagnosis:**
```bash
# Check explanation endpoint performance
curl -w "@curl-format.txt" -s -o /dev/null \
  "https://api.credit-risk-ai.example.com/api/v1/explain/app_12345"
```

**Solutions:**
- Reduce SHAP background sample size
- Cache frequent explanation requests
- Use approximate SHAP methods for faster computation

#### 3. Unintuitive Explanations

**Problem**: Explanations don't align with domain knowledge

**Solutions:**
- Validate feature engineering logic
- Check for data leakage in features
- Review model training process
- Consult domain experts for validation

### Performance Optimization

#### Caching Strategies

```python
from functools import lru_cache
import hashlib

@lru_cache(maxsize=1000)
def get_cached_explanation(application_data_hash):
    """Cache explanations for identical applications"""
    # Implementation details
    pass

def generate_explanation_with_cache(application_data):
    """Generate explanation with caching"""
    data_hash = hashlib.md5(str(application_data).encode()).hexdigest()
    return get_cached_explanation(data_hash)
```

#### Batch Processing

```python
def batch_explain(application_ids, batch_size=10):
    """Process explanations in batches for efficiency"""
    explanations = {}
    
    for i in range(0, len(application_ids), batch_size):
        batch = application_ids[i:i+batch_size]
        batch_explanations = generate_batch_explanations(batch)
        explanations.update(batch_explanations)
    
    return explanations
```

## Advanced Features

### Custom Explanation Templates

```python
def create_customer_explanation(shap_values, counterfactuals, risk_score):
    """Create customer-friendly explanation template"""
    template = """
    Dear {customer_name},
    
    Thank you for your loan application. Based on our assessment, your application 
    has a risk score of {risk_score:.1%}.
    
    Key factors in this decision:
    {key_factors}
    
    To improve your application:
    {improvement_suggestions}
    
    If you have questions about this decision, please contact us.
    """
    
    # Fill template with explanation data
    # Implementation details...
    
    return filled_template
```

### Explanation Quality Metrics

```python
def calculate_explanation_quality(explanations, predictions):
    """Calculate quality metrics for explanations"""
    metrics = {
        'faithfulness': calculate_faithfulness(explanations, predictions),
        'stability': calculate_stability(explanations),
        'comprehensiveness': calculate_comprehensiveness(explanations),
        'sufficiency': calculate_sufficiency(explanations)
    }
    return metrics
```

## Contact and Support

### Technical Support
- **Email**: explainability-support@credit-risk-ai.example.com
- **Documentation**: https://docs.credit-risk-ai.example.com/explainability
- **API Reference**: https://api.credit-risk-ai.example.com/docs

### Training and Workshops
- **User Training**: Monthly explainability workshops
- **Technical Deep Dives**: Quarterly technical sessions
- **Best Practices**: Regular sharing sessions

### Feedback and Improvements
- **Feature Requests**: explainability-feedback@credit-risk-ai.example.com
- **Bug Reports**: https://github.com/credit-risk-ai/explainability/issues
- **User Community**: https://community.credit-risk-ai.example.com

**Last Updated:** 2024-01-15  
**Version:** 1.2  
**Next Review:** 2024-04-15