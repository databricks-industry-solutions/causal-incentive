# Databricks notebook source
# MAGIC %run ./util/notebook-config

# COMMAND ----------

# MAGIC %md
# MAGIC #### Individualized Incentive Recommendations
# MAGIC
# MAGIC Armed with the models trained in the previous step,  we can develop a composite model that estimates the effects of each incentive on new companies based on their especific characteristics (```effect_modifiers```).  The model will select the incentive or combination of incentives with the highest effect on ```Revenue``` after accounting for the cost of the incentive. In the following cell, we define this composite model using MLflow's custom python model.

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
        estimated_effects[effects_interaction] = estimated_effects.sum(axis=1)
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
        net_effects["recommended incentive net effect"] = net_effects.max(axis=1)
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
# MAGIC We will instantiate the composite model for personalized incentive recommendation. Note that we are passing on only two models for the treatment effect estimation: i.e. ```tech_support_total_effect_dowhy_model``` and ```discount_dowhy_model```, althought we were intially interested in measuring the effect of ```New Engagement Strategy``` as well. This is because we have in the previous notebook found that this treatment has no effect on ```Revenue```. ```effect_modifiers``` give us a way to segment our dataset, which in turn will allow us to compute an incentive recommendation for each segment.
# MAGIC
# MAGIC After we instantiate the model, we will log it in MLflow togetehr with some other important information like model signiture and dependencies. Eventually this model will get registered in MLflow as well under the model name ```personalized_policy_recommender```.

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
            input_df, personalizedIncentiveRecommender.predict({}, input_df)
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

# MAGIC %md
# MAGIC ###Comparing Policies
# MAGIC
# MAGIC We can see in the comparison how following the personalized recommender approach an average gain of ~5K per account is obtained

# COMMAND ----------

compare_policies_effects(final_df)

# COMMAND ----------


