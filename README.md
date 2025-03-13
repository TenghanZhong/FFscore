# FFscore
Reproduction of Huatai Securities' FFscore Model

📌 FFScore: A Quantitative Value Investing Model for A-Shares
This repository provides an implementation of the FFScore model, a quantitative value investing strategy that enhances Piotroski’s F-Score framework by incorporating low Price-to-Book (PB) ratio stocks with refined financial metrics. The model is designed for A-share market selection, improving returns while mitigating financial distress risks.

🔹 Project Overview
Traditional value investing strategies focus on low PB stocks, but these stocks are often financially distressed. The FFScore model refines this approach by integrating a nine-factor financial screening system, originally proposed by Joseph Piotroski, and adapting it for the A-share market.

Why FFScore?
✔️ Targets undervalued stocks with strong financial health
✔️ Improves Piotroski’s F-Score by incorporating monthly PB fluctuations
✔️ Uses Altman’s Z-Score to validate financial distress risk
✔️ Backtested on A-shares, demonstrating significant performance improvements

🚀 Workflow
1️⃣ Data Collection & Preprocessing

Gathers financial reports & PB ratio data of A-share stocks
Filters top 20% lowest PB ratio stocks (excluding negative PB)
2️⃣ Financial Distress Analysis

Uses Altman’s Z-Score to assess the likelihood of distress
Finds that low PB stocks have higher financial distress risks
3️⃣ FFScore Calculation (9-Factor Model)
✔️ Profitability Indicators: ROA, CFO, ΔROA, Accruals
✔️ Leverage & Liquidity: ΔLeverage, ΔLiquidity, Equity Issuance
✔️ Operational Efficiency: ΔMargin, ΔTurnover

4️⃣ Stock Selection & Portfolio Construction

Stocks are scored from 0 to 9 based on financial strength
Top-scoring stocks (8-9) are selected for investment
Monthly rebalancing ensures optimal stock allocation
5️⃣ Backtesting & Performance Evaluation

Test period: 2006-2016 (A-share market)
Annualized return: 43.82% vs. baseline 33.77%
Sharpe ratio: 1.03, indicating superior risk-adjusted returns
Max drawdown: 66.27%, demonstrating controlled risk
📈 Results & Key Findings
✔️ FFScore outperforms standard low PB selection with +10% excess annualized return
✔️ Combining financial screening with PB ratio improves risk-adjusted returns
✔️ Effectively filters out distressed firms and enhances long-term profitability

🏆 Key Features
✔️ Quantitative Stock Selection
✔️ Deep Financial Screening with 9 Indicators
✔️ A-Share Market Optimized Strategy
✔️ Python-based Implementation (Pandas, NumPy, Backtesting.py)

📬 Contact & Citation
If you find this project useful, feel free to reach out via tenghanz@usc.edu or contribute to the repository!

