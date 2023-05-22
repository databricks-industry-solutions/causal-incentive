# Databricks notebook source
# MAGIC %run ./util/generate-data

# COMMAND ----------

# MAGIC %md
# MAGIC ## Influence Identification and Estimation

# COMMAND ----------

import mlflow

graph = mlflow.artifacts.load_text("graph.txt")

# COMMAND ----------

import dowhy


tech_support_effect_model = dowhy.CausalModel(data=input_df,
                     graph=graph,
                     treatment="Tech Support", 
                     outcome="Revenue"
                     )

tech_support_total_effect_identified_estimand = tech_support_effect_model.identify_effect(
    estimand_type="nonparametric-ate",
    method_name="maximal-adjustment",
)
print(tech_support_total_effect_identified_estimand) 

# COMMAND ----------

import mlflow

mlflow.autolog(disable=True)

model_t, model_y = setup_treatment_and_out_models()

effect_modifiers = ["Size", "Global Flag"]
method_name = "backdoor.econml.dml.LinearDML"
init_params = {
  "model_t": model_t,
  "model_y": model_y,
  "linear_first_stages": True,
  "discrete_treatment": True,
  "cv": 3,
  "mc_iters": 10,   
}


tech_support_total_effect_estimate = tech_support_effect_model.estimate_effect(
    tech_support_total_effect_identified_estimand,
    effect_modifiers=effect_modifiers, 
    method_name=method_name, 
    method_params={"init_params": init_params},
)

tech_support_total_effect_estimate.interpret()

# COMMAND ----------

import mlflow

mlflow.set_experiment(experiment_name=get_experiment_name())

model_details = register_dowhy_model(
  model_name = "tech_support_total_effect_dowhy_model",
  model=tech_support_effect_model,
  estimand=tech_support_total_effect_identified_estimand,
  estimate=tech_support_total_effect_estimate
)

# COMMAND ----------

tech_support_direct_effect_identified_estimand = tech_support_effect_model.identify_effect(
    estimand_type="nonparametric-cde",
    method_name="maximal-adjustment",
)
print(tech_support_direct_effect_identified_estimand) 

# COMMAND ----------


import mlflow

mlflow.autolog(disable=True)

model_t, model_y = setup_treatment_and_out_models()

effect_modifiers = ["Size", "Global Flag"]
method_name = "backdoor.econml.dml.LinearDML"
init_params = {
  "model_t": model_t,
  "model_y": model_y,
  "linear_first_stages": True,
  "discrete_treatment": True,
  "cv": 3,
  "mc_iters": 1,
}

tech_support_direct_effect_estimate = tech_support_effect_model.estimate_effect(
    tech_support_direct_effect_identified_estimand,
    effect_modifiers=effect_modifiers, 
    method_name=method_name,
    method_params={"init_params": init_params},
)

tech_support_direct_effect_estimate.interpret()

# COMMAND ----------

model_details = register_dowhy_model(
  model_name = "tech_support_direct_effect_dowhy_model",
  model=tech_support_effect_model,
  estimand=tech_support_direct_effect_identified_estimand,
  estimate=tech_support_direct_effect_estimate
)

# COMMAND ----------

discount_effect_model = dowhy.CausalModel(data=input_df,
                     graph=graph,
                     treatment="Discount", 
                     outcome="Revenue"
                     )

discount_effect_identified_estimand = discount_effect_model.identify_effect(
    estimand_type="nonparametric-ate",
    method_name="maximal-adjustment",
)

print(discount_effect_identified_estimand)

# COMMAND ----------

import mlflow

mlflow.autolog(disable=True)

model_t, model_y = setup_treatment_and_out_models()

effect_modifiers = ["Size", "Global Flag"]
method_name = "backdoor.econml.dml.LinearDML"
init_params = {
  "model_t": model_t,
  "model_y": model_y,
  "linear_first_stages": True,
  "discrete_treatment": True,
  "cv": 3,
  "mc_iters": 10,
}

discount_effect_estimate = discount_effect_model.estimate_effect(
    discount_effect_identified_estimand, 
    confidence_intervals=True,
    effect_modifiers=effect_modifiers,
    method_name=method_name,
    method_params={"init_params": init_params},
)

discount_effect_estimate.interpret()

# COMMAND ----------

model_details = register_dowhy_model(
  model_name = "discount_dowhy_model",
  model=discount_effect_model,
  estimand=discount_effect_identified_estimand,
  estimate=discount_effect_estimate
)

# COMMAND ----------

import dowhy

new_strategy_effect_model = dowhy.CausalModel(
    data=input_df, 
    graph=graph, 
    treatment="New Engagement Strategy", 
    outcome="Revenue"
)

new_strategy_effect_identified_estimand = new_strategy_effect_model.identify_effect(
    proceed_when_unidentifiable=True
)
print(new_strategy_effect_identified_estimand)

# COMMAND ----------

import warnings

warnings.simplefilter("ignore")

new_strategy_effect_estimate = new_strategy_effect_model.estimate_effect(
    new_strategy_effect_identified_estimand,
    method_name="backdoor.propensity_score_matching",
    target_units="att",
)
new_strategy_effect_estimate.value

# COMMAND ----------

estimates_df = pd.DataFrame(
    {
      "Estimated Direct Treatment Effect: Tech Support":[tech_support_direct_effect_estimate.value],
      "Estimated Total Treatment Effect: Tech Support":[tech_support_total_effect_estimate.value],
      "Estimated Total Treatment Effect: Discount":[discount_effect_estimate.value],
      "Estimated Total Treatment Effect: New Engagement Strategy":[new_strategy_effect_estimate.value],
    }
)

compare_estimations_vs_ground_truth(ground_truth_df, estimates_df)
