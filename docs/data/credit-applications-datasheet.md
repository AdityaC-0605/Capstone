# Datasheet: Credit Applications Dataset

## Motivation

### Purpose
This dataset was created to train and evaluate machine learning models for credit risk assessment in a fair, transparent, and sustainable manner. The dataset supports research and development of AI systems that can make accurate credit decisions while maintaining fairness across demographic groups and minimizing environmental impact.

### Funding
Dataset creation was funded by the Sustainable Finance Initiative and compiled by the AI Ethics Research Consortium in collaboration with participating financial institutions.

### Creator Information
- **Primary Creator:** AI Ethics Research Consortium
- **Contributing Institutions:** 15 anonymized financial institutions
- **Data Collection Period:** 2019-2023
- **Contact:** data-team@credit-risk-ai.example.com

## Composition

### Dataset Overview
The Credit Applications Dataset contains anonymized loan application data with associated outcomes, designed for training fair and accurate credit risk models.

| Attribute | Value |
|-----------|-------|
| **Total Records** | 2,547,832 |
| **Time Period** | January 2019 - December 2023 |
| **Geographic Coverage** | United States (all 50 states) |
| **Default Rate** | 14.7% |
| **Data Format** | CSV, Parquet |
| **Total Size** | 1.2 GB (compressed) |

### Feature Categories

#### 1. Demographic Features (10 features)
- Age, income, employment status, education level
- Geographic region, urban/rural classification
- **Privacy Protection:** All features anonymized and aggregated

#### 2. Financial Features (25 features)
- Credit scores, debt-to-income ratios, assets, liabilities
- Loan amount, term, purpose, collateral information
- **Data Sources:** Credit bureaus, bank records, public records

#### 3. Behavioral Features (30 features)
- Spending patterns, payment history, account usage
- Transaction frequencies, seasonal patterns
- **Aggregation Period:** 24-month lookback window

#### 4. Temporal Features (45 features)
- Monthly spending, payment amounts, account balances
- Time-series data for 24 months prior to application
- **Temporal Resolution:** Monthly aggregations

#### 5. Relational Features (15 features)
- Co-applicant information, guarantor relationships
- Business partnerships, family connections
- **Privacy Protection:** Relationship indicators only, no personal data

#### 6. Engineered Features (85 features)
- Ratios, interactions, statistical aggregations
- Domain-specific financial indicators
- **Creation Method:** Expert knowledge and automated feature engineering

### Target Variable
- **Default Indicator:** Binary (0 = no default, 1 = default within 24 months)
- **Definition:** 90+ days past due or charge-off
- **Observation Period:** 24 months post-origination

### Missing Data
| Feature Category | Missing Rate | Imputation Method |
|------------------|--------------|-------------------|
| Demographic | 2.3% | KNN imputation |
| Financial | 5.7% | Iterative imputation |
| Behavioral | 8.1% | Forward fill + median |
| Temporal | 12.4% | Time-series interpolation |
| Relational | 15.2% | Mode imputation |

## Collection Process

### Data Sources
1. **Primary Lenders:** 15 participating financial institutions
2. **Credit Bureaus:** Experian, Equifax, TransUnion (aggregated scores only)
3. **Public Records:** Court filings, property records, business registrations
4. **Economic Data:** Federal Reserve, Bureau of Labor Statistics

### Collection Methodology
- **Sampling Strategy:** Stratified random sampling by geography and demographics
- **Time Period:** Rolling 5-year window (2019-2023)
- **Quality Control:** Multi-stage validation and consistency checks
- **Bias Mitigation:** Oversampling of underrepresented groups

### Data Validation
- **Completeness Checks:** Missing data analysis and imputation
- **Consistency Validation:** Cross-source verification
- **Outlier Detection:** Statistical and domain-based outlier identification
- **Temporal Consistency:** Time-series validation and smoothing

### Privacy Protection Measures
1. **De-identification:** All personally identifiable information removed
2. **K-anonymity:** k=5 minimum group size for quasi-identifiers
3. **L-diversity:** l=3 minimum diversity for sensitive attributes
4. **Differential Privacy:** ε=10 privacy budget for statistical releases
5. **Data Minimization:** Only necessary features retained

## Preprocessing

### Data Cleaning
- **Duplicate Removal:** 0.3% duplicates identified and removed
- **Outlier Treatment:** Winsorization at 1st and 99th percentiles
- **Consistency Fixes:** Cross-field validation and correction
- **Format Standardization:** Consistent data types and formats

### Feature Engineering
- **Ratio Calculations:** 25 financial ratios computed
- **Interaction Terms:** 30 feature interactions created
- **Temporal Aggregations:** Statistical summaries over time windows
- **Domain Transformations:** Log, square root, and Box-Cox transformations

### Normalization
- **Numerical Features:** StandardScaler (mean=0, std=1)
- **Categorical Features:** Target encoding with smoothing
- **Temporal Features:** Min-max scaling within time windows
- **Skewed Features:** Quantile transformation to normal distribution

### Train/Validation/Test Split
| Split | Size | Percentage | Stratification |
|-------|------|------------|----------------|
| Training | 1,783,482 | 70% | By outcome and demographics |
| Validation | 382,125 | 15% | Temporal holdout (2023 Q3) |
| Test | 382,125 | 15% | Temporal holdout (2023 Q4) |

## Uses

### Intended Uses
1. **Credit Risk Modeling:** Training ML models for loan approval decisions
2. **Fairness Research:** Studying bias in credit decisions
3. **Sustainability Research:** Developing energy-efficient ML models
4. **Regulatory Compliance:** Testing fair lending practices

### Successful Applications
- Academic research on fair machine learning
- Industry benchmarking of credit models
- Regulatory stress testing and validation
- Bias detection and mitigation research

### Inappropriate Uses
- **Individual Identification:** Dataset cannot be used to identify individuals
- **Discriminatory Practices:** Must not be used for discriminatory lending
- **Medical Decisions:** Not suitable for healthcare applications
- **Employment Screening:** Requires additional bias testing for HR use

## Distribution

### Access and Licensing
- **License:** Creative Commons Attribution-NonCommercial 4.0
- **Access Method:** Secure download portal with registration
- **Usage Agreement:** Required for all users
- **Commercial Use:** Requires separate licensing agreement

### Distribution Channels
1. **Academic Portal:** https://data.credit-risk-ai.example.com/academic
2. **Industry Portal:** https://data.credit-risk-ai.example.com/industry
3. **Research Consortium:** Direct access for member institutions
4. **Public Repository:** Kaggle, UCI ML Repository (subset only)

### Access Requirements
- **Registration:** Required with institutional affiliation
- **Ethics Review:** IRB approval for human subjects research
- **Usage Reporting:** Annual usage reports required
- **Attribution:** Proper citation in publications

## Maintenance

### Update Schedule
- **Quarterly Updates:** New data additions every 3 months
- **Annual Refresh:** Complete dataset refresh with 5-year window
- **Bug Fixes:** Immediate fixes for data quality issues
- **Feature Updates:** New features added based on research needs

### Version Control
- **Semantic Versioning:** Major.Minor.Patch format
- **Change Logs:** Detailed documentation of all changes
- **Backward Compatibility:** Maintained for 2 major versions
- **Migration Guides:** Provided for breaking changes

### Quality Monitoring
- **Automated Checks:** Daily data quality monitoring
- **Drift Detection:** Statistical tests for distribution changes
- **Bias Monitoring:** Regular fairness metric evaluation
- **User Feedback:** Issue tracking and resolution system

### Deprecation Policy
- **Notice Period:** 12 months advance notice for deprecation
- **Migration Support:** Tools and guidance for version migration
- **Archive Access:** Deprecated versions available for 5 years
- **Documentation:** Maintained for all supported versions

## Legal and Ethical Considerations

### Regulatory Compliance
- **GDPR:** Right to explanation and data portability
- **CCPA:** Consumer privacy rights and data transparency
- **FCRA:** Fair credit reporting accuracy requirements
- **ECOA:** Equal credit opportunity compliance

### Ethical Guidelines
- **Fairness:** Regular bias testing and mitigation
- **Transparency:** Open documentation and methodology
- **Accountability:** Clear responsibility and contact information
- **Privacy:** Strong privacy protection measures

### Known Limitations
1. **Temporal Bias:** Historical data may reflect past discriminatory practices
2. **Geographic Bias:** US-centric data may not generalize globally
3. **Selection Bias:** Only includes applications from participating institutions
4. **Survivorship Bias:** May underrepresent certain demographic groups

### Risk Mitigation
- **Bias Testing:** Regular fairness audits across protected attributes
- **Documentation:** Comprehensive documentation of limitations
- **User Education:** Training materials on responsible use
- **Monitoring:** Ongoing monitoring of model outcomes

## Performance Benchmarks

### Baseline Models
| Model | AUC-ROC | Accuracy | Precision | Recall | F1-Score |
|-------|---------|----------|-----------|--------|----------|
| Logistic Regression | 0.823 | 0.791 | 0.756 | 0.734 | 0.745 |
| Random Forest | 0.856 | 0.821 | 0.798 | 0.776 | 0.787 |
| XGBoost | 0.871 | 0.834 | 0.812 | 0.789 | 0.800 |
| Neural Network | 0.876 | 0.841 | 0.819 | 0.796 | 0.807 |

### Fairness Benchmarks
| Protected Attribute | Demographic Parity | Equal Opportunity | Equalized Odds |
|---------------------|-------------------|-------------------|----------------|
| Gender | 0.89 | 0.91 | 0.88 |
| Race/Ethnicity | 0.85 | 0.87 | 0.84 |
| Age Group | 0.92 | 0.94 | 0.91 |

## Contact Information

### Data Stewards
- **Primary Contact:** Dr. Sarah Johnson, Data Ethics Lead
- **Email:** data-ethics@credit-risk-ai.example.com
- **Phone:** +1-555-DATA-ETH

### Technical Support
- **Data Engineering:** data-eng@credit-risk-ai.example.com
- **Documentation:** docs@credit-risk-ai.example.com
- **Bug Reports:** https://github.com/credit-risk-ai/data/issues

### Legal and Compliance
- **Legal Team:** legal@credit-risk-ai.example.com
- **Privacy Officer:** privacy@credit-risk-ai.example.com
- **Compliance:** compliance@credit-risk-ai.example.com

## Citation

```bibtex
@dataset{credit_applications_2024,
  title={Credit Applications Dataset for Fair Machine Learning},
  author={AI Ethics Research Consortium},
  year={2024},
  version={2.1.0},
  url={https://data.credit-risk-ai.example.com},
  doi={10.5281/zenodo.1234567}
}
```

**Last Updated:** 2024-01-15  
**Dataset Version:** 2.1.0  
**Datasheet Version:** 1.3