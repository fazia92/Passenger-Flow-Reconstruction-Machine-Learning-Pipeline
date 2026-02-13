# Passenger-Flow-Reconstruction-Machine-Learning-Pipeline
This project presents a machine learning pipeline designed to reconstruct passenger validation counts from ticket sales and contextual variables when direct validation data is partially unavailable or unreliable.

The approach was developed in an operational transport analytics context where equipment failures impacted the availability of validation data. The objective was to restore a coherent and usable passenger flow indicator to support operational monitoring and decision-making.

Due to confidentiality constraints, datasets are not included. The repository focuses on the reproducible methodology and modeling workflow.

Business Problem : 
          Passenger validations are a critical KPI used for:
                - Monitoring network usage
                - Comparing station demand
                - Supporting operational planning
                - Evaluating service performance
          Data collection disruptions created:
                - Partial data loss
                - Time series discontinuities
                - Reduced analytical reliability

This project addresses these issues by reconstructing validations using historical relationships between sales and usage behavior.

Methodology
    Data Sources (Conceptual) : 
      - Ticket sales
      - Validation counts (historical reference period)
      - Temporal context (calendar effects)
      - Location context (station-level variation)

Feature Engineering
  Temporal Context
    - Day type classification
    - Special periods
    - Calendar segmentation
  Behavioral Features
    - Rolling averages
    - Lag variables
    - Historical usage ratios
  Categorical Encoding
    - Station
    - Ticket type
    - Period
    - Day type

Modeling Strategy
  Algorithms Evaluated
     - Linear Regression
     - Random Forest
     - Gradient Boosting Regressor
  Model Selection Criteria
     - Predictive performance
     - Robustness
     - Ability to capture nonlinear interactions
     
Gradient Boosting was retained as final model.

Evaluation Metrics  
    - MAE
    - RMSE
    - R²
    
Disclaimer
This repository contains methodology and code only.
No operational data is shared.
