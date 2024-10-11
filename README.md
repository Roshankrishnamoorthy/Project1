# Project 1 

**L 1 REGULARIZATION**
Machine learning techniques like L1 regularization, sometimes referred to as Lasso regularization, are used to stop overfitting. It functions by adjusting the loss function of the model by a penalty term that is determined by the absolute values of the model's coefficients.

**EXAMPLES:**

1.	Credit scoring: In developing a credit risk model, L1 can help identify the most crucial factors (like income, credit history, debt ratio) while eliminating less relevant ones (e.g. zip code, marital status).
2.	Medical diagnosis: When analyzing patient data to predict diseases, L1 can pinpoint the most indicative symptoms or test results, simplifying the diagnostic process.
3.	Text classification: In sentiment analysis of product reviews, L1 can select the most impactful words or phrases, ignoring less meaningful ones.

**ADVANTAGES**
1.	Can eliminate less important features by setting their coefficients to zero
2.	Sparse models: Produces simpler models with fewer active features
3.	Interpretability: Easier to identify which features have the most impact
**DISADVANTAGES:**
1.	Instability: Can be unstable when features are highly correlated
2.	Optimization challenges: Non-differentiable at zero, making optimization more difficult
3.	Limited in handling feature interactions: May struggle with complex relationships between features

**L2 REGULARIZATION**
L2 regularization, which is sometimes referred to as Ridge regularization, is a machine learning technique that stops overfitting. It functions by incorporating a penalty term into the loss function of the model, which is determined by the squared values of the model's coefficients.

**EXAMPLES**
1.	Image recognition: In a neural network for classifying animal images, L2 can help prevent any single feature (like color or texture) from dominating the model's decisions.
2.	Weather forecasting: When predicting temperature, L2 can balance the influence of various factors (humidity, wind speed, atmospheric pressure) without eliminating any.
3.	Recommendation systems: In movie recommendation algorithms, L2 can manage the impact of multiple user preferences (genre, actors, release year) without discarding any particular aspect.

**ADVANTAGES **
1.	Stability: More stable, especially when dealing with correlated features
2.	Smooth optimization: Differentiable everywhere, making it easier to optimize
3.	Handling multicollinearity: Effective at managing highly correlated features
4.	Computational efficiency: Generally faster to compute due to its mathematical properties
**DISADVANTAGES:**
1.	No feature selection: Keeps all features in the model, just reducing their impact
2.	Less interpretable: Harder to determine feature importance as all coefficients remain non-zero

**ELASTIC NET REGRESSION**

Elastic Net Regression is an advanced method in the field of linear regression that merges the strengths of two popular regularization techniques: Lasso and Ridge regression. This approach is particularly valuable when dealing with datasets where standard linear regression methods may face challenges, especially in situations involving high correlation among predictor variables. By combining the penalty terms from both Lasso and Ridge regression, Elastic Net creates a balanced approach to model regularization. This unique characteristic makes it an effective tool for addressing multicollinearity, a common issue in many real-world datasets where independent variables are closely related.

**FORMULA**

![image](https://github.com/user-attachments/assets/70e07555-53f1-4ffd-b6c1-cfe9de360b5b)
