
# GoQuant Assignment - Ethereum Implied Volatility Forecasting

## Overview
This project implements a **time-series forecasting model** to predict Ethereum's 10-second-ahead **implied volatility (IV)** using high-frequency order-book and cross-asset market data. It includes data preprocessing, feature engineering, model training with **LightGBM**, and generating competition-ready submissions.

---

## Features & Highlights

- **Data Preprocessing**: Handle missing values, calculate bid-ask spread, and order book imbalance (OBI).
- **Feature Engineering**:
  - Mid-price
  - Bid-ask spread
  - Top-level order book imbalance (OBI)
- **Model**: Gradient boosting using **LightGBM** for regression.
- **Evaluation**: Root Mean Squared Error (RMSE) for training predictions.
- **Submission**: Generates \`submission.csv\` compatible with the competition.

---

## Folder Structure

```
GoQuant_assignment/
│
├── gq_implied_volatility_forecasting.py   # Main Python script
├── submission.csv                         # Sample submission file
├── train/                                 # Training data CSVs
│   └── ETH.csv
├── test/                                  # Test data CSVs
│   └── ETH.csv
└── README.md                              # Project documentation
```

---

## Requirements

- Python 3.8+
- Pandas
- NumPy
- LightGBM
- Scikit-learn
- Git LFS (for large CSV files)

---

## Usage

1. Clone the repository:

```
git clone https://github.com/Shristirajpoot/GoQuant_assignment.git
cd GoQuant_assignment
```
2. Install dependencies:

```
pip install pandas numpy lightgbm scikit-learn
```

3. Place the \`train\` and \`test\` folders in the project directory.  

4. Run the Python script:

```
python gq_implied_volatility_forecasting.py
```

5. The script will generate \`submission.csv\` in the project folder.

---

## Author

**Shristi Rajpoot**  
- GitHub: [Shristirajpoot](https://github.com/Shristirajpoot)  
- Email: shristirajpoot369@gmail.com

