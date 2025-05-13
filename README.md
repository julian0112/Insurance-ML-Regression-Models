<h1>Insurance Charges Prediction Using Supervised Learning with Regression</h1>
<h2>Main Objective</h2>
The main objective of the project is to use regression supervised models to predict a target value. The subject of the dataset that will be used is insurance and in this case the target that we want to predict is the amount to be charged to the beneficiary based 
on the values of other parameters and characteristics of this. This analysis would be beneficial to find patterns within the data, and see if certain values on certain features affect the insurance charges of each individual. To make this possible we try multiple
models and ended up using a <b>Ridge Regression</b> model with <b>Polynomial Features and Standard Scaling</b>; as it gave us the best results.
<h2>Attributes Summary</h2>
The data we are working with is a Medical Cost Personal Dataset, it has more than 1300 entries and 7 features including the target of the prediction model, charges, that is going to be developed within this project.
<ol>
  <li><b>Age:</b> age of the beneficiary</li>
  <li><b>Sex:</b> the gender of the contractor</li>
  <li><b>BMI:</b> the body mass index</li>
  <li><b>Children:</b> number of children covered by the health insurance</li>
  <li><b>Smoker:</b> if the beneficiary is a smoker or not</li>
  <li><b>Region:</b> the beneficiary´s residential area: northeast, northwest, southeast, southwest</li>
  <li><b>Charges:</b> individual medical cost billed by insurance</li>
</ol>
<h2>Data Cleaning and Feature Engineering</h2>
First, we check the information and a snippet of the data, we can see that the columns "sex", "smoker" and "region" are of type object as see in Figure 1, so we are going to use Label Encoder of Scikit-Learn to change them to numerical values.<br><br>
<div align="center">

  ![imagen](https://github.com/user-attachments/assets/513a483a-47dc-4a04-a08e-492c5d7e1caf) <br>
  <i>Figure 1: columns value type and null count</i>

</div>

<div align="center">

  ![imagen](https://github.com/user-attachments/assets/9c8c3ecc-ccab-468a-9625-12f410838fc2)<br>
  <i>Figure 2: dataset snippet</i>
</div>

Now when checking for the distribution of the charges data we can see that it is not normalized, if we check it using the Shapiro Test it give us a value of around <b>1.90e-36</b>, to try and correct that there are two changes that we are going to apply to our target, we are going
to apply the method <b>Winsorize</b> from Sci-Py and then apply the Yeo-Johnson Transformation, this two change are going to be applied in this case to the whole target just for the plot, later we are going to apply them but just to the y_train. This give us Shapiro Test Score of around <b>2.08e-16</b>, this is a more normal distribution, but not a perfect one, there are methods like Quantile Transformer from Scikit-Learn that
makes the Shapiro Test Score 0.001, but in this case, the Yeo-Johnson Transformation gives us a better R² and RMSE Score when writing our model. We can see the distribution of the charges data before and after the Yeo-Johnson Transformation in Figure 3. :<br><br>

<div align="center">

![imagen](https://github.com/user-attachments/assets/e8fbb99f-26d1-495d-8c09-64174e8b6eca)
 <br>
  <i>Figure 3: Distribution of Insurance Charges Before and After Yeo-Johnson Transformation</i>
</div>

<h2>Insights</h2>
First by checking the correlation of the parameters with our target column, as seen in Figure 4, we can see that the feature with the biggest correlation is whether or not the beneficiary is a smoker, followed by the age and BMI of the beneficiary. This tells us that the age and
smoking habits of the person affects in some way how much they pay for insurance, to check that we made violin plot of the charges and the different age groups according to smoking habits, as seen in Figure 5. We can see that in all age groups the cost of the people that don't
smoke is a lot less than the ones that do. :<br><br>
<div align="center">

  ![Correlation Heatmap](https://github.com/user-attachments/assets/0d84da39-0e41-4d16-8f1e-9f7fae378219)
 <br>
  <i>Figure 4: Correlation Heatmap.</i>

</div>

<div align="center">

  ![Age range vs charges by smoking habits](https://github.com/user-attachments/assets/5872fbaa-78cb-4a4b-8d2c-de2cdb521911)
<br>
  <i>Figure 5: Age Ranges vs. Charges by Smoking Habits</i>
</div>

We also decided to use the method "describe" to get more insight with the numerical data, now we can see that the mean amount of charges is of $13,270 USD as seen in Figure 6, this is going to be useful later to check the percentage of the relative error of the model.

<div align="center">

  ![imagen](https://github.com/user-attachments/assets/d4f05085-b126-4a01-bf04-fff5c2f6d5e9)
<br>
  <i>Figure 6: Statistics of the original numerical values</i>
</div>

<h2>Ridge Model</h2>
For the model first we separated the data using the Train/Test Split from Scikit-Learn, and the apply the Winsorize and Yeo-Johnson Transformation to the y_train to normalize the data, after that we create the next Pipeline:<br><br>
<ul>
  <li><b>Polynomial Feature:</b> with a degree of two.</li>
  <li><b>Standard Scaler:</b> transform the features to a mean of 0 and a standard deviation of one.</li>
  <li><b>Ridge Regression:</b> with an alpha of three.</li>
</ul>
We selected this hyperparameters using a GridSearchCV, thus ensuring the best results. For the creation of the Model, we used TransformedTargetRegressor, this allowed us to not only use the Ridge Pipeline, but also automatically inverse transformer, avoiding manual errors when 
reversing Yeo-Johnson, it also calculate R² and RMSE on the original scale without additional steps. The result of the model was:<br><br>
<ul>
  <li type="disc">R² Ridge Regression: 0.863</li>
  <li type="disc">RMSE Ridge Regression: 4605.22 USD</li>
</ul>
This means that the model explains the 86.3% of the variability of the charges with mean error of around ±$4,605 with respect of the real value, compared with the mean of $13,270 ($13,270/$4,605) there is a deviation of 34.7%.<br>
We also decided to make a base model (Linear Regression Model without the Polynomial Features) to better compare the one using Ridge Regression, this were the results:
<ul>
  <li type="disc">R² Linear Regression: 0.685</li>
  <li type="disc">RMSE Linear Regression: 6994.30 USD</li>
</ul>
As we can see the Ridge Model gives us an improvement of 26% in the R² Score and compared to the 34.7% of deviation of the Ridge Model, the base model gives us a 52.7%, this means that there is a reduction of 18% of the relative deviation and of 34.2% in the RMSE, we can visualize
this in the Figure 7. Finally, we can see the results of the Ridge Model in the Figure 8.<br><br>

<div align="center">

![Linear vs Ridge](https://github.com/user-attachments/assets/ebdcc86f-42ab-4253-a2b4-1c2a358e06b7)
  <i>Figure 7: Comparison of R² Score and RMSE of Base Model and Ridge Model</i>
</div>

<div align="center">

![Prediction Vs Real Values](https://github.com/user-attachments/assets/41d58b7f-608d-47f4-bae4-971c8e57674d)<br>
  <i>Figure 8: Ridge Model Results</i>
</div>

<h2>Next Steps and Improvements</h2>
We could investigate better ways to normalize the data for the target value and try other regression models such as ARD Regression, Bayesian Ridge or XGBoost.
