# Databricks notebook source
# MAGIC %run ./util/notebook-config

# COMMAND ----------

# MAGIC %md
# MAGIC ## Influence Identification and Estimation

# COMMAND ----------

# MAGIC %md
# MAGIC We will use the casual graph obtained from the previous step to guide the identification step.  In this step the best method to isolate the effect of a given incentive on ```Revenue``` is identified.  The package DoWhy automates this step by relying on the well established [Do-Calculus](https://ftp.cs.ucla.edu/pub/stat_ser/r402.pdf) theoretical framework.
# MAGIC
# MAGIC First, let's load the graph from [MLflow](https://www.databricks.com/product/managed-mlflow).

# COMMAND ----------

graph = load_graph_from_latest_mlflow_run(experiment_name)

# COMMAND ----------

# MAGIC %md
# MAGIC We will now identify the total effect of ```Tech Support``` in ```Revenue``` using <b>DoWhy</b> to obtain the <b>Average Treatement Effect (ATE)</b> estimand.

# COMMAND ----------

# Instantiate a model object to estimate the tech support effect
tech_support_effect_model = dowhy.CausalModel(
    data=input_df, graph=graph, treatment="Tech Support", outcome="Revenue"
)

# Identify methods we can use to estimate the tech support effect (estimands)
tech_support_total_effect_identified_estimand = (
    tech_support_effect_model.identify_effect(
        estimand_type="nonparametric-ate",
        method_name="maximal-adjustment",
    )
)

# Print out all identified estimands
print(tech_support_total_effect_identified_estimand)

# COMMAND ----------

# MAGIC %md
# MAGIC <b>DoWhy</b> find the [backdoor](http://causality.cs.ucla.edu/blog/index.php/category/back-door-criterion/) method as the best one to identify the effect.  It also determines which features should be used for the estimation.  

# COMMAND ----------

# MAGIC %md
# MAGIC ###Estimating "Tech Support" total effect on "Revenue"
# MAGIC
# MAGIC In order to obtain an unbias estimation we will use an approach call [Double Machine Learning (DML)](https://academic.oup.com/ectj/article/21/1/C1/5056401)  which is implemented in the [PyWhy](https://github.com/py-why) package [EconML](https://github.com/py-why/EconML). We use a logistic regression model for predicting the treatment and lasso for predicting the outcome.

# COMMAND ----------

# Disable the mlflow autolog feature
mlflow.autolog(disable=True)

# Set up the treatment (t) and outcome (y) models for DML. See notebook-config for detail.
model_t, model_y = setup_treatment_and_out_models()

# Specify the effect modifiers, which are variables that can change the magnitude of the effect based on the groups.
effect_modifiers = ["Size", "Global Flag"]

# Specify the estimand recommended in the previous cell
method_name = "backdoor.econml.dml.LinearDML"

init_params = {
    "model_t": model_t,
    "model_y": model_y,
    "linear_first_stages": True,
    "discrete_treatment": True,
    "cv": 3,
    "mc_iters": 10,
}

# Estimate the effect of tech support
tech_support_total_effect_estimate = tech_support_effect_model.estimate_effect(
    tech_support_total_effect_identified_estimand,
    effect_modifiers=effect_modifiers,
    method_name=method_name,
    method_params={"init_params": init_params},
)

# Extract the interpretation of the estimate
tech_support_total_effect_estimate.interpret()

# COMMAND ----------

# MAGIC %md
# MAGIC ###Registering in MLflow the resulting model
# MAGIC Please notice the ```autolog``` functionality of [MLflow](https://www.databricks.com/product/managed-mlflow) is disable in the above block of code.  This was done to have more control of what is logged into MLflow.  EconML uses sklearn models trained and evaluated usgin cross-validation.  If "autolog" is enable, all the trained models are logged in [MLflow](https://www.databricks.com/product/managed-mlflow) (including the once not ultimately selected by EconML),  this results in a lot of noise and slow performance.  Instead, we will control what is logged in [MLflow](https://www.databricks.com/product/managed-mlflow) by using the helper function ```register_dowhy_model```.  This function will register the EconML model together with the artefacts created by DoWhy 

# COMMAND ----------

model_details = register_dowhy_model(
    model_name="tech_support_total_effect_dowhy_model",
    model=tech_support_effect_model,
    estimand=tech_support_total_effect_identified_estimand,
    estimate=tech_support_total_effect_estimate,
)

# COMMAND ----------

# MAGIC %md
# MAGIC ###Estimating "Tech Support" direct effect on "Revenue"
# MAGIC
# MAGIC In the graph obtained from the previous notebook we can appricate ```Tech Support``` has a direct effect on ```Revenue``` and a mediated effect through ```New Product Adoption```.  In other words,  ```Tech Support``` besides directly influencing ```Revenue```, also impacts ```New Product Adoption``` which itself has an effect on ```Revenue```.  The estimation done in the commands above covered the total influence on this incentive (direct and indirect).  We will now identify the direct influence only by using the [Control Direct Effect (CDE)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4193506/) estimand type of DoWhy 

# COMMAND ----------

tech_support_direct_effect_identified_estimand = (
    tech_support_effect_model.identify_effect(
        estimand_type="nonparametric-cde",
        method_name="maximal-adjustment",
    )
)
print(tech_support_direct_effect_identified_estimand)

# COMMAND ----------

# MAGIC %md
# MAGIC We will use again the [DML algorithm](https://academic.oup.com/ectj/article/21/1/C1/5056401) implemented in [EconML]() for this estimation

# COMMAND ----------

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

# MAGIC %md
# MAGIC Let's now register the resulting model with all the associated DoWhy artifacts in [MLflow](https://www.databricks.com/product/managed-mlflow)

# COMMAND ----------

model_details = register_dowhy_model(
    model_name="tech_support_direct_effect_dowhy_model",
    model=tech_support_effect_model,
    estimand=tech_support_direct_effect_identified_estimand,
    estimate=tech_support_direct_effect_estimate,
)

# COMMAND ----------

# MAGIC %md
# MAGIC ###Estimating effect of "Discount" in "Revenue"
# MAGIC
# MAGIC We will follow a similar approach as the one use to identify and estimate the total effect of ```Tech Support``` to now identify and estimate the effect of ```Discount``` on ```Revenue```

# COMMAND ----------

discount_effect_model = dowhy.CausalModel(
    data=input_df, graph=graph, treatment="Discount", outcome="Revenue"
)

discount_effect_identified_estimand = discount_effect_model.identify_effect(
    estimand_type="nonparametric-ate",
    method_name="maximal-adjustment",
)

print(discount_effect_identified_estimand)

# COMMAND ----------

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
    model_name="discount_dowhy_model",
    model=discount_effect_model,
    estimand=discount_effect_identified_estimand,
    estimate=discount_effect_estimate,
)

# COMMAND ----------

# MAGIC %md
# MAGIC ###Estimating the effect of "New Engagment Strategy" in "Revenue"
# MAGIC
# MAGIC Finally we will estimate the effect of the ```New Engagement Strategy``` incentive.  The graph obtain from the previous notebook displayed estated no effect on ```Revenue```.  We should see the same when identifying this effect and estimating it 

# COMMAND ----------

new_strategy_effect_model = dowhy.CausalModel(
    data=input_df, graph=graph, treatment="New Engagement Strategy", outcome="Revenue"
)

new_strategy_effect_identified_estimand = new_strategy_effect_model.identify_effect(
    proceed_when_unidentifiable=True
)

print(new_strategy_effect_identified_estimand)

# COMMAND ----------

warnings.simplefilter("ignore")

new_strategy_effect_estimate = new_strategy_effect_model.estimate_effect(
    new_strategy_effect_identified_estimand,
    method_name="backdoor.propensity_score_matching",
    target_units="att",
)

new_strategy_effect_estimate.value

# COMMAND ----------

# MAGIC %md
# MAGIC [DoWhy](https://github.com/py-why/dowhy) also find now effect.
# MAGIC
# MAGIC Please notice [DoWhy](https://github.com/py-why/dowhy) decide not to use ```Plan Summit``` as a feature for the estimation.  If included, a spurious effect would be percived, leading us to wrong conclusions

# COMMAND ----------

# MAGIC %md
# MAGIC ###Comparing estimated effects with ground thruth
# MAGIC
# MAGIC As mentioned before the data for this accelerator was generated using probabilistic methods.  The ground truth is provided in the original dataset.  When compared with the estimated effect we see the estimations are very close.

# COMMAND ----------

estimates_df = pd.DataFrame(
    {
        "Estimated Direct Treatment Effect: Tech Support": [
            tech_support_direct_effect_estimate.value
        ],
        "Estimated Total Treatment Effect: Tech Support": [
            tech_support_total_effect_estimate.value
        ],
        "Estimated Total Treatment Effect: Discount": [
            discount_effect_estimate.value
        ],
        "Estimated Total Treatment Effect: New Engagement Strategy": [
            new_strategy_effect_estimate.value
        ],
    }
)

compare_estimations_vs_ground_truth(ground_truth_df, estimates_df)

# COMMAND ----------


