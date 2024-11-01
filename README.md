# A Survey of Financial AI: Architectures, Advances and Open Challenges

Resources: [Paper](https://github.com/junhua/awesome-finance-ai-papers/blob/main/survey-paper.pdf), [Slides](https://github.com/junhua/awesome-finance-ai-papers/blob/main/review-full.pdf), [Cite](#citation)

This repository is a curated collection of high-quality papers presented at top Computer Science (CS) conferences related to Artificial Intelligence (AI) applications in finance. 

The aim is to offer a comprehensive review of these papers, summarize key findings, and analyze their contributions to the field.

This list extends an earlier collection: [Awesome-LLM-Agents](https://github.com/junhua/awesome-llm-agents). 

## Price Prediction
### Formulation
For Price Prediction tasks, the objectives are forecasting of continuous data, for (1) stock price, (2) earning, or (3) volatility.

Models to learn feature representations commonly include some combinations of: (1) attention-based models (2) spatial-temporal models (3) graph models

Auto-regressive self-supervised learning is commonly seen in pre-training the financial data representation, whereas supervised finetuning (SFT) is used in post-training for the prediction downstream task.

### Papers
- MDGNN: Multi-Relational Dynamic Graph Neural Network for Comprehensive and Dynamic Stock Investment Prediction, *AAAI'24* ([Paper](https://arxiv.org/pdf/2402.06633))
- MASTER: Market-Guided Stock Transformer for Stock Price Forecasting, *AAAI'24* ([Paper](https://arxiv.org/pdf/2312.15235), [Code](https://github.com/SJTU-DMTai/MASTER))
- DoubleAdapt: A Meta-learning Approach to Incremental Learning for Stock Trend Forecasting, *KDD'23* ([Paper](https://arxiv.org/pdf/2306.09862))
- Double-Path Adaptive-correlation Spatial-Temporal Inverted Transformer for Stock Time Series Forecasting, *KDD'25* ([Paper](https://arxiv.org/pdf/2409.15662))
- Stock Movement Prediction Based on Bi-typed Hybrid-relational Market Knowledge Graph via Dual Attention Networks, *TKDE'23* ([Paper](https://arxiv.org/pdf/2201.04965))
- GARCH-Informed Neural Networks for Volatility Prediction in Financial Markets, *ArXiv'24* ([Paper](https://arxiv.org/pdf/2410.00288v1))
- DiffSTOCK: Probabilistic relational Stock Market Predictions using Diffusion Models, *ICASSP'24* ([Paper](https://ieeexplore.ieee.org/document/10446690))
- From News to Forecast: Iterative Event Reasoning in LLM-Based Time Series Forecasting, *NeurIPS'24* ([Paper](https://arxiv.org/pdf/2409.17515v1))

## Trend/Movement Classification
### Formulation
Trend/Movement Classification is commonly binary classification task that predicts the market movement (up or down) on a future date.

Basic building blocks (i.e. models) for feature representation and downstream task are similar to that in Price Prediction.

### Papers
- Multi-relational Graph Diffusion Neural Network with Parallel Retention for Stock Trends Classification, *ICASSP'24* ([Paper](https://ieeexplore.ieee.org/document/10447394))
- Saliency-Aware Interpolative Augmentation for Multimodal Financial Prediction, *COLING'24* ([Paper](https://aclanthology.org/2024.lrec-main.1244.pdf))
- ECHO-GL: Earnings Calls-Driven Heterogeneous Graph Learning for Stock Movement Prediction, *AAAI'24* ([Paper](https://ojs.aaai.org/index.php/AAAI/article/view/29305))
- Trade When Opportunity Comes: Price Movement Forecasting via Locality-Aware Attention and Iterative Refnement Labeling, *IJCAI'24* ([Paper](https://www.ijcai.org/proceedings/2024/0678.pdf))
- MANA-Net: Mitigating Aggregated Sentiment Homogenization with News Weighting for Enhanced Market Prediction, *CIKM'24* ([Paper](https://arxiv.org/pdf/2409.05698))
- Learning to Generate Explainable Stock Predictions using Self-Reflective Large Language Models, *WWW'24* ([Paper](https://arxiv.org/abs/2402.03659))

## Ranking-based Stock Selection
### Formulation
Ranking-based Stock Selection is modelled as a ranking task that outputs the rank of a pool of stocks based on some values such as predicted earnings or risk adjusted earnings.

Basic building blocks (i.e. models) for feature representation and downstream task are similar to that in Price Prediction and Trend Classification.

### Papers
- CI-STHPAN: Pre-trained Attention Network for Stock Selection with Channel-Independent Spatio-Temporal Hypergraph, *AAAI'24* ([Paper](https://ojs.aaai.org/index.php/AAAI/article/view/28770))
- Automatic De-Biased Temporal-Relational Modeling for Stock Investment Recommendation, *IJCAI'24* ([Paper](https://www.ijcai.org/proceedings/2024/0221.pdf))
- RSAP-DFM: Regime-Shifting Adaptive Posterior Dynamic Factor Model for Stock Returns Prediction, *IJCAI'24* ([Paper](https://www.ijcai.org/proceedings/2024/0676.pdf))
- Relational Temporal Graph Convolutional Networks for Ranking-Based Stock Prediction, *ICDE'23* ([Paper](https://ieeexplore.ieee.org/document/10184655))

## Portfolio Optimization
### Formulation
For portfolio optimization, the objectives are variants of asset allocation, i.e., optimizing the portfolio weights.

The decision making processes are commonly modelled as Markov Decision Process (MDP), Partially Observed MDP (POMDP), Hierachical MDP (HMDP) and Hidden Markov Model (HMM).

In terms of ML models, Reinforcement Learning is a norm.

### Papers
- Developing A Multi-Agent and Self-Adaptive Framework with Deep Reinforcement Learning for Dynamic Portfolio Risk Management, *AAMAS'24* ([Paper](https://arxiv.org/pdf/2402.00515))
- Trend-Heuristic Reinforcement Learning Framework for News-Oriented Stock Portfolio Management, *ICASSP'24* ([Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10447993&tag=1))
- Reinforcement Learning with Maskable Stock Representation for Portfolio Management in Customizable Stock Pools, *WWW'24* ([paper](https://arxiv.org/pdf/2311.10801))
- FreQuant: A Reinforcement-Learning based Adaptive Portfolio Optimization with Multi-frequency Decomposition, *KDD'24* ([paper](https://dl.acm.org/doi/10.1145/3637528.3671668))
- Risk-Managed Sparse Index Tracking Via Market Graph Clustering, *ICASSP'24* ([Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10447211))
- Mitigating Extremal Risks: A Network-Based Portfolio Strategy, *ArXiv'24* ([Paper](https://arxiv.org/pdf/2409.12208v1))
- Improving Portfolio Optimization Results with Bandit Networks, *ArXiv'24* ([paper](https://arxiv.org/pdf/2410.04217))

## Quantitative Trading
### Formulation
For Quantitative Trading, the objectives are trading actions (long/short/hold)and its attributes (amount, price, etc.).

The decision making models are similar to that in Portfolio Optimization.

### Papers
- Optimizing Trading Strategies in Quantitative Markets using Multi-Agent Reinforcement Learning, *ICASSP'24* ([Paper](https://arxiv.org/abs/2303.11959))
- EarnHFT: Efficient Hierarchical Reinforcement Learning for High Frequency Trading, *AAAI'24* ([Paper](https://arxiv.org/pdf/2309.12891))
- MacMic: Executing Iceberg Orders via Hierarchical Reinforcement Learning, *IJCAI'24* ([Paper](https://www.ijcai.org/proceedings/2024/0664.pdf))
- IMM: An Imitative Reinforcement Learning Approach with Predictive Representation Learning for Automatic Market Making, *IJCAI'24* ([Paper](https://www.ijcai.org/proceedings/2024/0663.pdf))
- Hierarchical Reinforced Trader(HRT): A Bi-Level Approach for Optimizing Stock Selection and Execution, *ArXiv'24* ([Paper](https://arxiv.org/pdf/2410.14927))
- MacroHFT: Memory Augmented Context-aware Reinforcement Learning On High Frequency Trading, *KDD'24* ([Paper](https://arxiv.org/pdf/2406.14537))
- Efficient Continuous Space Policy Optimization for High-frequency Trading, *KDD'23* ([Paper](https://dl.acm.org/doi/pdf/10.1145/3580305.3599813))
- StockFormer: Learning Hybrid Trading Machines with Predictive Coding, *IJCAI'23* ([Paper](https://www.ijcai.org/proceedings/2023/0530.pdf))
- A Multimodal Foundation Agent for Financial Trading: Tool-Augmented, Diversified, and Generalist, *KDD'24* ([Paper](https://arxiv.org/pdf/2402.18485))
- Automate Strategy Finding with LLM in Quant investment, *ArXiv'24* ([Paper](https://arxiv.org/pdf/2409.06289))

## Knowledge Retrieval and Augmentation 
This is a loosely categorised task that involved a combination of (1) Information Retrieval (2) Data Augmentation (3) Data Labelling (4) Prompt Engineering (5) Simulation (6) Others

Calling this "Knowledge Retrieval and Augmentation" should cover most of the primary objectives of the papers.

### Retrieval and Augmentation
- Extracting Financial Events from Raw Texts via Matrix Chunking, *COLING'24* ([Paper](https://aclanthology.org/2024.lrec-main.617.pdf))
- Large Language Models as Financial Data Annotators: A Study on Effectiveness and Efficiency, *COLING'24* ([Paper](https://arxiv.org/pdf/2403.18152))
- FinReport: Explainable Stock Earnings Forecasting via News Factor Analyzing Model, *WWW'24* ([Paper](https://arxiv.org/abs/2403.02647))

### Prompt Engineering
- Prompting for Numerical Sequences: A Case Study on Market Comment Generation, *COLING'24* ([Paper](https://arxiv.org/pdf/2404.02466))
- Temporal Relational Reasoning of Large Language Models for Detecting Stock Portfolio Crashes, _ArXiv'24_([Paper](https://www.arxiv.org/pdf/2410.17266))

### Agentic Simulations
- EconAgent: Large Language Model-Empowered Agents for Simulating Macroeconomic Activities, *ACL'24* ([Paper](https://aclanthology.org/2024.acl-long.829/), [Code](https://github.com/tsinghua-fib-lab/ACL24-EconAgent))
- When AI Meets Finance (StockAgent): Large Language Model-based Stock Trading in Simulated Real-world Environments, *ArXiv'24* ([Paper](https://arxiv.org/pdf/2407.18957), [Code](https://github.com/MingyuJ666/Stockagent))

## Datasets
Though the papers primarily propose some datasets, they also contribute to a variety of the abovementioned tasks.

### Papers
- AlphaFin: Benchmarking Financial Analysis with RetrievalAugmented Stock-Chain Framework, *COLING'24* ([Paper](https://arxiv.org/pdf/2403.12582), [Code](https://github.com/AlphaFin-proj/AlphaFin))
- Market-GAN: Adding Control to Financial Market Data Generation with Semantic Context, *AAAI'24* ([Paper](https://arxiv.org/pdf/2309.07708), [Code](https://github.com/kah-ve/MarketGAN))
- FNSPID: A Comprehensive Financial News Dataset in Time Series, *KDD'24* ([Paper](https://arxiv.org/abs/2402.06698),[Dataset](https://huggingface.co/datasets/Zihan1004/FNSPID), [Code](https://github.com/Zdong104/FNSPID_Financial_News_Dataset))
- StockEmotions: Discover Investor Emotions for Financial Sentiment Analysis and Multivariate Time Series, *AAAI'23* ([Paper](https://arxiv.org/pdf/2301.09279), [Code](https://github.com/adlnlp/StockEmotions))


## Time Series Models
This section includes some general time-series models that were recently proposed. One may get inspirations in model designs, or directly apply them on financial time-series data.

### Papers
- TimeMixer: Decomposable Multiscale Mixing for Time Series Forecasting, *ICLR'24* ([Paper](https://openreview.net/pdf?id=7oLshfEIC2), [Code](https://github.com/kwuking/TimeMixer))
- MOMENT: A Family of Open Time-series Foundation Models, *ICML'24* ([Paper](https://arxiv.org/pdf/2402.03885), [Code](https://github.com/moment-timeseries-foundation-model/moment))
- Timer: Generative Pre-trained Transformers Are Large Time Series Models, *ICML'24* ([Paper](https://arxiv.org/pdf/2402.02368), [Code](https://github.com/thuml/Large-Time-Series-Model))
- TimesNet: Temporal 2d-variation modeling for general time series analysis, _ICLR’23_([Paper](https://arxiv.org/pdf/2210.02186), [Code](https://github.com/thuml/Time-Series-Library))
- A Time Series is Worth 64 Words: Long-term Forecasting with Transformers, _ICLR'23_([Paper](https://arxiv.org/abs/2211.14730), [Code](https://github.com/yuqinie98/PatchTST))

## Survey
This section includes some other survey papers on LLM for Finance

### Papers
- A Survey of Large Language Models in Finance (FinLLMs), *ArXiv'24* ([Paper](https://arxiv.org/pdf/2402.02315))
- A Survey of Large Language Models for Financial Applications: Progress, Prospects and Challenges, *ArXiv'24* ([Paper](https://arxiv.org/pdf/2406.11903))
- Large Language Model Agent in Financial Trading: A Survey, *ArXiv'24* ([Paper](https://arxiv.org/pdf/2408.06361))
- Revolutionizing Finance with LLMs: An Overview of Applications and Insights, _ArXiv'24_([Paper](https://arxiv.org/pdf/2401.11641))

## Citation
Please cite the repo if the content assists your work.
```
@misc{awesome-finance-ai-papers,
author = {Junhua Liu},
title = {A Survey of Financial AI: Architectures, Advances and Open Challenges},
year = {2024},
publisher = {GitHub},
journal = {GitHub Repository},
doi={10.5281/zenodo.14021183},
howpublished = {\url{https://github.com/junhua/awesome-finance-ai-papers}},
}
```
