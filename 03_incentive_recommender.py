# Databricks notebook source
# MAGIC %run ./util/generate-data

# COMMAND ----------

# MAGIC %md
# MAGIC # Make Policy Recommendations

# COMMAND ----------

# MAGIC %md
# MAGIC In this section, we use EconML tools to visualize differences in conditional average treatment effects across customers and select an optimal investment plan for each customer.
# MAGIC
# MAGIC In order to decide whether to offer each investment to the customer, we need to know the cost of providing the incentive as well as the benefits of doing so. In this step we define a cost function to specify how expensive it would be to provide each kind of incentive to each customer. In other data samples you can define these costs as a function of customer features, upload a matrix of costs, or set constant costs for each treatment (the default is zero). In this example, we set the cost of ```discount``` to be a fix value of $7000 per account, while the cost of ```tech support``` is $100 per PC.

# COMMAND ----------

# MAGIC %md
# MAGIC #### Individualized policy recommendations 

# COMMAND ----------

# MAGIC %md
# MAGIC For our current sample of customers, we can also identify the best treatment plan for each individual customer based on their CATE. We use the model's `const_marginal_effect` method to find the counterfactual treatment effect for each possible treatment. We then subtract the treatment cost and choose the treatment with the highest return. That is the recommended policy.
# MAGIC
# MAGIC To visualize this output, we plot each customer based on their PC count and past revenue, the most important determinants of treatment according to the tree interpreter, and color code them based on recommended treatment.

# COMMAND ----------

import mlflow

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
        net_effects["recommended incentive"]= net_effects.idxmax(axis=1).apply(
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

# MAGIC %md Here we are loading our estimators from MLflow using ```mlflow.pyfunc.load_model``` method. 

# COMMAND ----------

loaded_model = mlflow.pyfunc.load_model(
    f"models:/{model_details.name}/{model_details.version}"
)

final_df = loaded_model.predict(input_df)

display(final_df)

# COMMAND ----------

compare_policies_effects(final_df)
