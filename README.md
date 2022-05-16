<br>

<details open>
<summary>Introduction</summary>

A project to predict on Soybean Variety Yield and apply in reality. The whole structure includes descriptive analysis, predictive analysis and prescriptive analysis.

</details>

<details open>
<summary>Step</summary>
The whole project include the following steps

- Descriptive analysis.

- Predictive analysis: 

  (1) Using K-means to identify geological clusters based on latitude and longitude to lower dimension. 

  (2) Selecting features based on random forest.

  (3) Using LASSO, XgBoost, Random Forest and DNN to predict different type of Soybean Yield and find out the best model on test set according to its determining factors like temperature, moisture, radiation, etc. 

- Prescriptive:

  Since the weather condition is not confirmed in future years, corresponding determining factors including temperature, moisture and radiation is not determined. Actually, their distribution can be estimated base weather condition history. Based on different weather condition, different type of Soybeans will have different yield.

  With this, to find out the best combination of Soybean to generate maximum yield, **Markowitz** capital optimization is used.

</details>

<details open>
<summary>Result</summary>

<details open>
<summary>Descriptive Analysis</summary>
  
- The whole report:
  [Prediction on Soybean Variety Yield an application.pdf](https://github.com/LeeGorden/Prediction-on-Soybean-Variety-Yield-and-apply-to-Soybean-Variety-Selection/files/8703041/Prediction.on.Soybean.Variety.Yield.an.application.pdf)

- ![image](https://user-images.githubusercontent.com/72702872/168672066-dcd3c8eb-11b0-4ffb-b2de-0cbfe7f7399f.png)

- ![image](https://user-images.githubusercontent.com/72702872/168672253-9e58dcfc-2e40-4415-9a68-ff3a174ddc67.png)

- ![image](https://user-images.githubusercontent.com/72702872/168672386-20f37ec9-3daf-4e64-b317-0fccc52e0cec.png)

- ![image](https://user-images.githubusercontent.com/72702872/168672921-57764f2f-31e7-474b-9ee8-20ad34f6ad2e.png)
  
- Cluster result(K-means) of different types of Soybean based on latitude and longitude: 
  ![image](https://user-images.githubusercontent.com/72702872/168672980-3ece86ec-761a-49f6-bcfe-8af77c856a3e.png)
  
- ![image](https://user-images.githubusercontent.com/72702872/168673502-9d7cf5b0-4e66-4099-8c38-857e5d288ae1.png)

</details>
  
<details open>
<summary>Predictive Analysis</summary>
  
- Performance of different models:
  ![image](https://user-images.githubusercontent.com/72702872/168673748-9a288c0c-3a9f-47ee-8188-a4ccbde63a0c.png)
  
</details>
  
<details open>
<summary>Prescriptive Analysis</summary>

Now, we have the best model to predict each kind of Soybean's yield based on different weather condition. Based on the model, the next step is to utilize prescriptive analysis to find out the best combination of Soybeans to achieve best yield for future years.

Since the weather condition of future is not determined, we can estimate weather condition distribution of temperature, precipitation and radiation to find out the best combination with highest Yield / σ (In this case, we treat each kind of Soybean as a stock, the purpose is to optimize the combo with the goal to maximize Shap ratio = Expected Return / σ).
  
- Based on weather condition history, the distribution of temperature, precipitation and radiation are as follow: 
  ![image](https://user-images.githubusercontent.com/72702872/168674274-1ab7154d-677d-4c99-ae1c-a399f6802f2e.png)

- According to the distribution of weather condition above, corresponding yield of chosen star Soybeans are as follows:
  ![image](https://user-images.githubusercontent.com/72702872/168675234-6b3c3ab0-5a73-4d2a-a536-8e6077051e2c.png)

- The **Markowitz** bullet curve is as follow:
  
  ![image](https://user-images.githubusercontent.com/72702872/168677923-9745e923-4188-4600-bb10-96eb922d64af.png)
  
  The best combination appears at the tangent of the curve.
  
</details>

## <div align="center">Contact</div>

email: likehao1006@gmail.com

LinkedIn: https://www.linkedin.com/in/kehao-li-06a9a2235/

ResearchGate: https://www.researchgate.net/profile/Gorden-Li

<br>

