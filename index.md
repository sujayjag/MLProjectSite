---
layout: default
---

# CS 7641 Project

## Project Proposal

### Introduction

Professional tennis has seen numerous historic matches, tremendous athletes, and game-changing strategies over the past decade. This project aims to dive deep into this by examining a rich dataset encompassing a plethora of tennis matches. This dataset includes comprehensive details like match outcomes, betting odds, set and game outcomes, player statistics, court type, date, etc.

### Literature Review

Kovalchik delved into the realm of tennis predictions by comparing 30 different models to predict the outcomes of men's professional tennis matches over eight seasons. This comprehensive study took into account a player's past performance, player rankings, and even surface-specific performance, identifying that surface-adjusted Elo ratings were the most accurate predictors of match outcomes. (Dixon, 1997)

Though not strictly on tennis, Dixon and Coles' study on association football offers valuable insights on how betting odds can play a significant role in predicting match outcomes. They developed a model for predicting match results based on a Poisson distribution, which could be applied or adjusted for tennis given the similarities in the prediction realm. (Kovalchik, 2016)

Sipko tackled the NBA but introduced methods that are highly transferable to tennis prediction. He utilized the betting market odds and combined them with ranking methods, highlighting that betting odds indeed encompass significant predictive power, a testament to the efficiency of the broader betting market. This supports the idea of integrating betting odds into tennis match predictions for higher accuracy. (Sipko, 2014)

Several studies have attempted to predict tennis match outcomes or understand player performances, but with the influx of data, especially related to betting odds and in-depth player stats, there’s a new opportunity to extract meaningful insights.
![Serve Speed Analytics](assets/ServeSpeed.jpeg | width="100" height="250")

### Problem Definition

The primary objective of this project is to predict tennis game, set, and match outcomes based on a series of parameters. The vast dataset provides a great foundation to seek patterns that might not be obvious at first glance. Additionally, by integrating betting odds, there's potential to evaluate the market's accuracy in forecasting match results. We could also dive into how well a player may progress throughout a tournament, and predict their results before the tournament starts.

### Methods

The project will employ a variety of machine learning algorithms, primarily starting with logistic regression and decision trees. Given the intricacy of the dataset, ensemble methods like random forests or gradient boosting might be employed later. We intend to leverage libraries such as scikit-learn for a smoother implementation. Our dataset will be gathered from Kaggle, which is a free platform that has numerous, large datasets of various topics that can be explored and analyzed.

### Potential Results and Discussions

Metrics such as accuracy, precision, recall, and the F1 score will be used to gauge the model's efficiency. We can anticipate a fairly high success rate on the prediction based on initial exploratory data analysis. However, these figures can change for the better, as we delve deeper and refine our models through testing.

### Timeline
Here is a picture of the Gantt Chart for this project:
![Gantt Chart](assets/GanttChart.jpeg)

### Datasets

https://www.kaggle.com/datasets/ehallmar a-large-tennis-dataset-for-atp-and-itf-betting
https://www.kaggle.com/datasets/edoardoba/atp-tennis-data

### Bibliography

Dixon, M. J., & Coles, S. G. (1997). "Modelling association football scores and inefficiencies in the football betting market." Applied statistics, 46(2), 265-280.

Kovalchik, S. A. (2016). "Searching for the GOAT of tennis win prediction." Journal of Quantitative Analysis in Sports, 12(3), 127-138.

Sipko, T. (2014). "Predicting the outcomes of NBA basketball games." arXiv preprint arXiv:1411.1443.
