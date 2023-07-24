# Databricks notebook source
# MAGIC %run ./util/notebook-config

# COMMAND ----------

# MAGIC %md
# MAGIC #### Individualized Promotional Offer Recommendations
# MAGIC
# MAGIC Armed with the models trained in the previous step,  we can develop a composite model that estimates the effects of each incentive on new companies based on their specific characteristics.  The model will select the promotional offer(s) with the highest effect on ```Revenue``` after accounting for the cost of the offer. In the following cell, we define this composite model using MLflow's custom python model.

# COMMAND ----------

class PersonalizedIncentiveRecommender(mlflow.pyfunc.PythonModel):
    """Custom wrapper for a personalized incentive recommender."""

    def __init__(self, models_dictionary, effect_modifiers):
        self.models_dictionary = models_dictionary
        self.effect_modifiers = effect_modifiers

    def _estimate_isolated_effect(self, model_input):
        """Compute the constant marginal conditional average treatment effect """
        """conditioned on the effect modifiers for each company."""
        return pd.DataFrame(
            {
                f"{key} net effect": np.hstack(
                    model.const_marginal_effect(model_input[self.effect_modifiers])
                )
                for key, model in self.models_dictionary.items()
            }
        )

    def _estimate_effect_with_interaction(self, estimated_effects):
        """Combine the estimated isolated effects of each treatment."""
        effects_interaction = (
            " and ".join(self.models_dictionary.keys()) + " net effect"
        )
        estimated_effects[effects_interaction] = estimated_effects.sum(axis=1, numeric_only=True)
        return estimated_effects

    def _cost_fn_interaction(self, data):
        """Calculate the cost of each treatment and the total cost."""
        t1_cost = data[["PC Count"]].values * 100
        t2_cost = np.ones((data.shape[0], 1)) * 7000
        return np.hstack([t1_cost, t2_cost, t1_cost + t2_cost])

    def _estimate_net_effects(self, estimated_effects, costs):
        """Subtract the cost of each treatment from each estimated isolated effect """
        """and the total cost from the combined effect."""
        return estimated_effects - costs

    def _get_recommended_incentive(self, net_effects):
        """Make an incentive recommendation with the max estimated net effect."""
        net_effects["recommended incentive"] = net_effects.idxmax(axis=1).apply(
            lambda x: x.replace(" net effect", "")
        )
        net_effects["recommended incentive net effect"] = net_effects.max(axis=1, numeric_only=True)
        return net_effects

    def predict(self, context, model_input):
        """Given a dataset of new companies returns the personalized recommended """
        """policy for each customer."""
        estimated_effects = self._estimate_isolated_effect(model_input)
        estimated_effects = self._estimate_effect_with_interaction(estimated_effects)
        costs = self._cost_fn_interaction(model_input)
        net_effects = self._estimate_net_effects(estimated_effects, costs)
        net_effects["no incentive net effect"] = 0
        net_effects = self._get_recommended_incentive(net_effects)
        return model_input.join(net_effects)

# COMMAND ----------

# MAGIC %md
# MAGIC ###Registering in MLflow the Personalized Incentive Recommender model
# MAGIC
# MAGIC We will instantiate the composite model for personalized incentive recommendation. Note that we are passing on only two models for the treatment effect estimation: i.e. ```tech_support_total_effect_dowhy_model``` and ```discount_dowhy_model```, althought we were intially interested in measuring the effect of ```New Engagement Strategy``` as well. This is because we found in the previous notebook that this treatment has no effect on ```Revenue```.
# MAGIC
# MAGIC After we instantiate the model, we will log it in MLflow together with some other important information like model signature and dependencies. Eventually this model will get registered under the model name ```personalized_policy_recommender```.

# COMMAND ----------

from mlflow.models.signature import infer_signature

model_name = "personalized_policy_recommender"

with mlflow.start_run(run_name=f"{model_name}_run") as experiment_run:
    #Instantiate a model 
    personalizedIncentiveRecommender = PersonalizedIncentiveRecommender(
        models_dictionary={
            "tech support": get_registered_wrapped_model_estimator(
                model_name="tech_support_total_effect_dowhy_model"
            ),
            "discount": get_registered_wrapped_model_estimator(
                model_name="discount_dowhy_model"
            ),
        },
        effect_modifiers=["Size", "Global Flag"],
    )
    #Log the model in MLflow
    mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=personalizedIncentiveRecommender,
        signature=infer_signature(
            input_df.drop(["Tech Support", "Discount", "New Engagement Strategy"], axis=1), personalizedIncentiveRecommender.predict({}, input_df)
        ),
    )

#Register the model in MLflow
model_details = mlflow.register_model(
    model_uri=f"runs:/{experiment_run.info.run_id}/model",
    name=model_name,
)

displayHTML(f"<h1>Model '{model_details.name}' registered</h1>")
displayHTML(f"<h2>-Version {model_details.version}</h2>")

# COMMAND ----------

# MAGIC %md
# MAGIC ###Loading model and Predicting the best incentive(s) per account
# MAGIC
# MAGIC We can see new columns are added to the DataFrame with the information estimated by the model:
# MAGIC
# MAGIC - ```recommended incentive```: incentive or combination of incentives that would result in the optimal return
# MAGIC - ```recommended incentive net effect```:  Dollar effect on "revenue" of the recommended incentive(s) after substracting cost 
# MAGIC - ```discount net effect```: Dollar effect on ```Revenue``` of the ```Discount``` incentive
# MAGIC - ```tech support net effect```: Dollar effect on ```Revenue``` of the ```tech support``` incentive
# MAGIC - ```tech support + discount net effect```: Dollar effect on ```Revenue``` of providing both incentives
# MAGIC - ```no incentive net effect```: Dollar effect on ```Revenue``` of providing no incentive 

# COMMAND ----------

#Load the model from MLflow
loaded_model = mlflow.pyfunc.load_model(
    f"models:/{model_details.name}/{model_details.version}"
)

final_df = loaded_model.predict(input_df)

display(final_df)

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ###Comparing Policies
# MAGIC
# MAGIC We can see in the comparison how following the personalized recommender approach an average gain of ~5K per account is obtained

# COMMAND ----------

final_df

# COMMAND ----------

all_policies_df = final_df.merge(input_df)

##Adding the previous year "no policy" scenario
all_policies_df["no policy"] = (all_policies_df["Tech Support"]*all_policies_df["tech support net effect"]) + (all_policies_df["Discount"]*all_policies_df["discount net effect"])

compare_policies_effects(all_policies_df)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Individualized Policy Recommendations 

# COMMAND ----------

# MAGIC %md
# MAGIC We can take a step deeper and inspect the best treatment plan for each customer. This links to the exploratory data analysis that we performed in the first notebook (```00_intro```), where we found that there was no strategy for providing incentives to customers. We color code the customer based on the recommended best treatment types and that gives us boundaries along the values of ```PC Count``` and ```Size```. These boudaries are linear because we use linear models for the effect estimation using double machine learning. Had we assigned the treatments based on our model suggestion instead of at random, we could have maximized the return.    

# COMMAND ----------

# MAGIC %matplotlib inline
# MAGIC # Plot the recommended policy of each customer
# MAGIC plt.figure(figsize=(10, 7.5))
# MAGIC plot_policy(final_df, final_df["recommended incentive"])

# COMMAND ----------

# MAGIC %md
# MAGIC This modeling technique (DML) lets us estimate the isolated effect of a given incentive while controlling for confounders. We can obtain an unbiased estimate for each treatment, which is something hard to achieve using traditional ML techniques.
# MAGIC
# MAGIC The recommender can be used to obtain the optimal incetive for new accounts given the accounts characteristics.  For example assuming the first of the rows in the "input_df" belong to a new account just onboarded, the sales team can obtain a recommednation of which incentives to offer:

# COMMAND ----------

new_account = input_df.head(1).drop(["Tech Support", "Discount", "New Engagement Strategy"], axis=1)
new_account

# COMMAND ----------

# MAGIC %md
# MAGIC ####Obtain Promotinal Offer(s) Recommedation for New Account

# COMMAND ----------

incentive_recommended = loaded_model.predict(new_account)

displayHTML(
    f"<H2>Recommended incentive(s) for new account:</h2><h2>- {incentive_recommended[['recommended incentive']].values[0][0].capitalize()}</H2>"
)

# COMMAND ----------

# MAGIC %md
# MAGIC MLflow allows leveraging the recommender in [batch processing](https://mlflow.org/docs/latest/python_api/mlflow.pyfunc.html#mlflow.pyfunc.spark_udf) or as a REST API via [Databricks model serving](https://docs.databricks.com/machine-learning/model-serving/index.html). 

# COMMAND ----------

# MAGIC %md
# MAGIC © 2023 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License. All included or referenced third party libraries are subject to the licenses set forth below.
# MAGIC
# MAGIC | library                                | description             | license    | source                                              |
# MAGIC |----------------------------------------|-------------------------|------------|-----------------------------------------------------|
# MAGIC | dowhy   | Python library for causal inference that supports explicit modeling and testing of causal assumptions | MIT   | https://pypi.org/project/dowhy/          
# MAGIC | econml   |  contains several methods for calculating Conditional Average Treatment Effects | MIT    | https://pypi.org/project/econml/  
# MAGIC | causal-learn   | python package for causal discovery  | MIT    | https://pypi.org/project/causal-learn/          
