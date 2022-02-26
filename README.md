# Used-Car-Price-Predictor
Performed a Linear Regression on a used car dataset, after executing data cleaning and processing, to predict the price of used cars for sale.

This exercise is an application of the knowledge acquired whilst enrolled in Udemy's 'The Data Science Course 2022: The Comlete Data Science Bootcamp'. 

I first performed data exploration, examined the PDFs of the continuous variables 'Price', 'Mileage', 'Engine Volume' & 'Year of production' and dealt with the apparent outliers using Panda's percentile function. 

**Verifying OLS Assumptions 
**

Again, I chose to focus on the continuous variables identified as potential regressors, the categorical variables were included as dummies. 

- When checking for linearity, I used scatter plots [(Price, Mileage), (Price, EngineV), (Price, Year)] which showed no linear patterns. 
  My exploratory analysis showed that 'Price' showed exponential patterns - not normally distributed. I therefore executed a log transformation of the variable. 
  
  Running VIF tets for each independent continuous variable led me to drop 'Year'. 
  
- The assumption of no endoeneity was not violated. 
- Following Central Limit Theorem, the sample size of the data is large enough to assume Normality.
- Following the transformation of 'Price', the scatter graphs (shown below) visually imply the homescedasticity assumption holds. 
  ![image](https://user-images.githubusercontent.com/93582626/155854909-9a4404b9-217a-4020-a7cb-4b8b25a66dee.png)
- Including the intercept in the regression accomodates the zero mean assumption. 
- No reason for the data to have cross-sectional correlation. 

**Creating Dummy Variables 
**

I created dummy variables using Panda's get_dummies function 

**Generating the Regression Model
**

I created the model using ScikitLearn's 'StandardScaler' and 'train_test_split' modules. 

The residuals PDF shows that there were a few values where the y_train - y_hat was much lower than the mean. 

**Conclusion
** 

On average, this model does a pretty good job at predicting the value of a used car – especially as the actual price increases. The large errors are mostly made on cars whose value is below $7,000. 
Important to note that the residuals for the estimations that are extremely inaccurate are all negative, meaning the predictions are higher than the targets. This may be explained by other factors the model has not accounted for such as damage to the car, the model of the car (earlier removed) or the colour. 
There are a few ways that this model could be improved, for example I could use a different set of variables, remove a larger volume of outliers and also use different kinds of transformations. 
When I return to improve this model I will look to:
-	Perform Feature Selection
-	Create another regression where Price is not transformed
-	Use alternate methods to deal with the data’s outliers

