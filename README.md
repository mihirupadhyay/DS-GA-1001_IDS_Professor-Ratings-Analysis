# Comprehensive Analysis of Professor Ratings: Statistical Insights & Predictive Modeling
This project involves a comprehensive analysis of professor ratings data using various statistical and machine learning techniques. The analysis is divided into several solutions, each focusing on different aspects of the data and employing various methodologies.
## Solution 1: Gender Differences in Average Ratings
This solution examines the differences in average ratings between male and female professors.
### Data Preprocessing
The data is loaded from 'rmpCapstoneNum.csv'.
Columns are renamed for clarity.
Data is filtered to include only professors with at least 3 ratings and where gender is specified.
### Statistical Analysis
A Mann-Whitney U test is performed to compare ratings between male and female professors.
The test statistic and p-value are calculated and interpreted for significance.
### Visualization
A boxplot and violin plot are created to visually compare the distribution of ratings between male and female professors.
## Solution 2: Distribution Analysis of Ratings
This solution focuses on analyzing the distribution of ratings using various statistical tests.
### Statistical Tests
Kolmogorov-Smirnov (KS) test is performed to compare the distribution of ratings between male and female professors.
Levene's test is used to check for equality of variances in ratings between genders.
### Visualization
Histogram with KDE plots are created to show the distribution of ratings for male and female professors.
A boxplot is used to display the spread and median of ratings by gender.
## Solution 3: Effect Size Analysis
This solution calculates and interprets effect sizes for gender differences in ratings.
### Effect Size Calculation
Cohen's d is calculated to measure the standardized difference between male and female ratings.
Confidence intervals for Cohen's d are computed.
### Visualization
A violin plot is created to show the distribution of ratings by gender.
The effect size and confidence interval are annotated on the plot.
## Solution 4: Tag Analysis
This solution analyzes the relationship between professor tags and ratings.
### Data Preprocessing
Tag data is loaded and merged with numerical data.
Correlation analysis is performed to identify collinear tags.
### Modeling
A polynomial regression model is built to predict ratings based on tags.
The most important features (tags) are identified based on their coefficients.
### Visualization
A correlation heatmap of tags is created.
A scatter plot of actual vs. predicted ratings is generated.
## Solution 5: Difficulty Rating Analysis
This solution focuses on analyzing and predicting difficulty ratings.
### Data Preprocessing and Modeling
Similar to Solution 4, but with difficulty ratings as the target variable.
A polynomial regression model is built to predict difficulty ratings.
### Visualization
A correlation heatmap of tags related to difficulty is created.
A scatter plot of actual vs. predicted difficulty ratings is generated.
## Solution 6: Chili Pepper Prediction
This solution builds a model to predict the "chili pepper" rating (attractiveness) of professors.
### Data Preprocessing
Numerical and tag data are combined.
Data is standardized using StandardScaler.
### Modeling
A logistic regression model is built to predict the chili pepper rating.
The model's performance is evaluated using accuracy, confusion matrix, and classification report.
### Visualization
A ROC curve is plotted to visualize the model's performance.
## Solution 7 (Extra Credit): Subject Analysis
This solution analyzes the effect of subject on professor ratings.
### Data Preprocessing
Qualitative and numerical data are combined.
Subjects with more than 1000 occurrences are filtered.
### Statistical Analysis
Kruskal-Wallis test is performed to compare ratings across subjects.
Mean ranks for each subject are calculated and sorted.
### Visualization
A bar plot of mean ranks for each subject is created.
The top 3 subjects with the highest effect are highlighted.
This project demonstrates a wide range of data analysis techniques, from basic statistical tests to advanced machine learning models, providing valuable insights into factors affecting professor ratings.
