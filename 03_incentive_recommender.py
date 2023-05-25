# Databricks notebook source
# MAGIC %run ./util/notebook-config

# COMMAND ----------

# MAGIC %md
# MAGIC #### Individualized Incentive Recommendations
# MAGIC
# MAGIC Armed with the model trained in the previous step,  we can develop a composite model that estimated the effects of each incentive new companies based on their especific characteristics.  The model will select the incentive or combination of incentives with the highest effect on ```Revenue``` after accounting for the cost of the incentive. 

# COMMAND ----------

class PersonalizedIncentiveRecommender(mlflow.pyfunc.PythonModel):
    def __init__(self, models_dictionary, effect_modifiers):
        self.models_dictionary = models_dictionary
        self.effect_modifiers = effect_modifiers

    def _estimate_isolated_effect(self, model_input):
        return pd.DataFrame(
            {
                f"{key} net effect": np.hstack(
                    model.const_marginal_effect(model_input[self.effect_modifiers])
                )
                for key, model in self.models_dictionary.items()
            }
        )

    def _estimate_interaction_effect(self, estimated_effects):
        effects_interaction = (
            " and ".join(self.models_dictionary.keys()) + " net effect"
        )
        estimated_effects[effects_interaction] = estimated_effects.sum(axis=1)
        return estimated_effects

    def _cost_fn_interaction(self, data):
        t1_cost = data[["PC Count"]].values * 100
        t2_cost = np.ones((data.shape[0], 1)) * 7000
        return np.hstack([t1_cost, t2_cost, t1_cost + t2_cost])

    def _estimate_net_effects(self, estimated_effects, costs):
        return estimated_effects - costs

    def _get_recommended_incentive(self, net_effects):
        net_effects["recommended incentive"] = net_effects.idxmax(axis=1).apply(
            lambda x: x.replace(" net effect", "")
        )
        net_effects["recommended incentive net effect"] = net_effects.max(axis=1)
        return net_effects

    def predict(self, context, model_input):
        estimated_effects = self._estimate_isolated_effect(model_input)
        estimated_effects = self._estimate_interaction_effect(estimated_effects)
        costs = self._cost_fn_interaction(model_input)
        net_effects = self._estimate_net_effects(estimated_effects, costs)
        net_effects["no incentive net effect"] = 0
        net_effects = self._get_recommended_incentive(net_effects)
        return model_input.join(net_effects)

# COMMAND ----------

# MAGIC %md
# MAGIC ###Registering in MLflow the Personalized Incentive Recommder model

# COMMAND ----------

from mlflow.models.signature import infer_signature

model_name = "personalized_policy_recommender"

with mlflow.start_run(run_name=f"{model_name}_run") as experiment_run:
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
    mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=personalizedIncentiveRecommender,
        signature=infer_signature(
            input_df, personalizedIncentiveRecommender.predict({}, input_df)
        ),
    )


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
