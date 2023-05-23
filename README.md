%md 

<img src='https://github.com/databricks-industry-solutions/.github/raw/main/profile/solacc_logo_wide.png' width="1000" ></img>
# Building an Incentive Recommender using Causal ML


Many companies offer their clients incentives to close deals, renew subscriptions, or purchase services.  These incentives carry costs that may not be recovered if it does not have the expected effect on the client.  In other words, the lack of a data driven incentive allocation policy would most probably result in a sub-optimal or even negative marginal profit.


In this solution accelerator we will show how Casual ML (in particular packages in the PyWhy project) can be leveraged in Databricks to facilitate the development of an Incentive Recommender ML Model. This Recommender selects for each customer the incentive(s) that maximized profit, allowing you to get the biggest bang for your incentive bucks! 


##Why Causal ML?


Predicting, classifying, or forecasting are tasks for which Machine Learning represents an ideal tool.  But, in the case of an Incentive Recommender,  more than predicting profit, the focus is on estimating the revenue effect of each of type incentive on a given customer.  Having these estimates allows picking the incentive with the highest positive effect (after discounting its cost).  

Even Though, the gold standard for estimating causal effects is randomized control trials [(RCT)](https://en.wikipedia.org/wiki/Randomized_controlled_trial), under the described business setting using and RCT is not recommended as it could carry reputational risks or be unethical.  Also, logistics for an RCT experiment are complex and costly, making it an unaxtractive option.


Instead, this accelerator shows how to estimate incentives effects given specific characteristics of a customer by using data recollected during a year of policy-less allocations.  The dataset probably holds many biases and blindly using Machine Learning would resul in misleading conclusions.  Luckily, this is where Causal ML shines, allowing us to discover the network of influences among the collected features, enrich this network with domain knowledge, identify how to isolate the influence of a given incentive controlling for biases,  and calculate an unbiased estimate of the incentive influence.  


##Why PyWhy?   
[PyWhy](https://www.pywhy.org/) is an umbrella project offering several Casual ML packages that highly facilitate the tasks mentioned before.  The packages are very easy to use yet well grounded in sophisticated Causal ML theory.  This combination is helping PyWhy to rapidly become one of the most important Casual ML open source projects in the Python Ecosystem.   


##Why Databricks Lakehouse?
Databricks offers a great option for Causal ML projects development and management.  It provides a scalable unified platform scoping Data and AI.  Data needed to train models is readily available via [Delta Lakes](https://www.databricks.com/product/delta-lake-on-databricks) and Casual ML models can be easily managed and deployed using [MLflow](https://www.databricks.com/product/managed-mlflow).


##Scenario and Data Description
This scenario is based on a case study found a PyWhy’s EconML package documentation [(Multi-investment attribution)](https://github.com/py-why/EconML/blob/main/notebooks/CustomerScenarios/Case%20Study%20-%20Multi-investment%20Attribution%20at%20A%20Software%20Company%20-%20EconML%20+%20DoWhy.ipynb).  It has been extended to take advantage of the benefits offered by the Databricks Lakehouse platform.


The case study centers on a Software company that offers to its clients  a Discount incentive (with a $7k internal cost), a Tech Support incentive (with an internal cost of $100 per licensed computer),  and a new targeted “engagement strategy” marketing strategy for incentivizing purchases of the software.  The company has been allocating the incentives without any policies for a year.  A dataset with the following features has been collected during this year:   


Feature Name | Type | Details
:--- |:--- |:---
**Revenue** | continuous | \\$ Annual revenue from customer given by the amount of software purchased


We consider three possible treatments, the interventions whose impact we wish to measure:


Feature Name | Type | Details
:--- |:--- |:---
**Tech Support** | binary | whether the customer received free tech support during the year
**Discount** | binary | whether the customer was given a discount during the year
**New Strategy** | binary | whether the customer was targeted for a new engagement strategy with different outreach behaviors


Also, a variety of additional customer characteristics that may affect revenue are considered:


Feature Name | Type | Details
:--- |:--- |:---
**Global Flag** | binary | whether the customer has global offices
**Major Flag** | binary | whether the customer is a large consumer in its industry
**SMC Flag** | binary | whether the customer is a Small or Medium Corporation (as opposed to large corporation)
**Commercial Flag** | binary | whether the customer's business is commercial (as opposed to public sector)
**Planning Summit** | binary | whether a sales team member held an outreach event with the customer during the year
**New Product Adoption** | binary | whether the customer signed a contract for any new products during the year
**IT Spend** | continuous | \$ spent on IT-related purchases
**Employee Count** | continuous | number of employees
**PC Count** | continuous | number of PCs used by the customer
**Size** | continuous | customer's total revenue in the previous calendar year


This data has been simulated and the incentives influence "ground truth" is therefore known




##Steps
The following steps are executed in order to develop a Personalized Incentive Recommender:


1- Casual Discovery:  this step focuses on discovering the of network of influences existing among the available features and enriching based on domain knowledge.  Having a clear understanding of this network allows identifying the best approach for isolating the influence of each of the incentives in a customer


This step is executed in the following notebook
[01_causal_discovery](./01_Causal_Discovery.py)


2- Identification and Estimation:  this step finds the best way of isolating each of the incentives influences using the network defined in the previous step.  The isolation method determines which features need to be controlled when estimate the influence. A Causal ML method for estimation called [Double Machine Learning implemented](https://arxiv.org/abs/1608.00060) implmeneted at the [EconML package](https://econml.azurewebsites.net/spec/estimation/dml.html) is used to obtain an unbiased estimation.


The following notebook implements this step:
[02_identification_and_estimation]()


3- Personalized Incentive Recommender:  armed with the Causal ML influence estimators trained in the previous step, a composite model is developed to recommend the incentive or combination of incentives returning the highest profit based on basic characteristics of the customer.


The Recommender is develop on the following notebook:
[03_incentive_recommender]()


4- Tests (Model Refutation): in order to have a good level of confidence in a developed estimators,  different tests are applied.  The tests mainly consist in gradually injecting noise or distorting the dataset to capture the point in which the estimator is no longer valid

This step is executing in the following notebook:
[04_refutation]()

Important:  Please execute the “RUNME” notebook to prepare your Databricks environment for the notebooks mentioned above.  The “RUNME” notebook will create a new Databricks Workflow pointing to each of the notebook,  create a new job cluster to execute the workflow, and  Install all the dependency libraries.
