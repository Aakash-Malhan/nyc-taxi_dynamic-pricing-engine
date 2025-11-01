**🗽 NYC Dynamic Pricing & Driver Incentive Engine**
Demo - https://huggingface.co/spaces/aakash-malhan/nyc-dynamic-pricing-engine

**Problem Statement**  
Ride-hailing platforms face a 2-sided marketplace challenge:

    If pricing is too low:
    ✅ High demand
    ❌ Not enough drivers → long wait times & cancellations

    If pricing is too high:
    ✅ More drivers online
    ❌ Riders churn → losing trips & revenue

**This project builds a dynamic pricing engine that predicts:**
     
    Surge multiplier
    Recommended fare
    Driver incentive bonus
    Expected driver acceptance probability
    ETA impact
    All using real NYC taxi trip data + ML modeling + geospatial analysis.

<img width="1489" height="926" alt="Screenshot 2025-11-01 142027" src="https://github.com/user-attachments/assets/41b28aae-7dec-407b-9623-144cd2ec3e85" />
<img width="1415" height="945" alt="Screenshot 2025-11-01 142004" src="https://github.com/user-attachments/assets/20e61e6e-97c9-4ed7-816b-15515905f987" />
<img width="1492" height="596" alt="Screenshot 2025-11-01 142041" src="https://github.com/user-attachments/assets/1d99b7cc-a2ba-44a2-85b1-9c6f88e29dfc" />




**💻 Tech Stack**:
    
    Data             NYC TLC Yellow Taxi Trip Records (2023)
    Processing       DuckDB, Pandas, PyArrow
    ML Model         XGBoost
    Explainability   SHAP

**Business Impact:**

    This is the first line
    Pricing uplift       +12–38% per trip when demand > supply
    Driver acceptance    Improves with bonus tuning
    Rider experience     Faster pickup (simulated ETA adjustments)

**Future Enhancements:**

    This is the first line
    Live streaming data (Kafka / Uber H3 real-time zones)
    Reinforcement learning driver incentive tuning
    Snowflake / BigQuery data backend
    Driver movement simulation
