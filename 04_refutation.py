# Databricks notebook source
# MAGIC %run ./util/notebook-config

# COMMAND ----------

# MAGIC %md
# MAGIC ##Refutation

# COMMAND ----------

wrapped_model = get_registered_wrapped_model(model_name="discount_dowhy_model")

model = wrapped_model.get_model()
estimand = wrapped_model.get_estimand()
estimate = wrapped_model.get_estimate()

# COMMAND ----------

res_random_common_cause = model.refute_estimate(
    estimand=estimand,
    estimate=estimate,
    show_progress_bar=True,
    method_name="random_common_cause",
    num_simulations=100,
    n_jobs=16,
)

refutation_random_common_cause_df = pd.DataFrame(
    [
        {
            "Refutation Type": res_random_common_cause.refutation_type,
            "Estimated Effect": res_random_common_cause.estimated_effect,
            "New Effect": res_random_common_cause.new_effect,
            "Refutation Result (p value)": res_random_common_cause.refutation_result[
                "p_value"
            ],
        }
    ]
)

refutation_random_common_cause_df

# COMMAND ----------

mlflow.autolog(disable=True)

res_unobserved_common_cause = model.refute_estimate(
    estimand=estimand,
    estimate=estimate,
    show_progress_bar=True,
    method_name="add_unobserved_common_cause",
    confounders_effect_on_treatment="binary_flip",
    confounders_effect_on_outcome="linear",
    effect_fraction_on_treatment=0.05,
    effect_fraction_on_outcome=0.05,
)

refutation_unobserved_common_cause_df = pd.DataFrame(
    [
        {
            "Refutation Type": res_unobserved_common_cause.refutation_type,
            "Estimated Effect": res_unobserved_common_cause.estimated_effect,
            "New Effect": res_unobserved_common_cause.new_effect,
            "Refutation Result (p value)": None,
        }
    ]
)

refutation_unobserved_common_cause_df

# COMMAND ----------

res_placebo = model.refute_estimate(
    estimand=estimand,
    estimate=estimate,
    show_progress_bar=True,
    method_name="placebo_treatment_refuter",
    placebo_type="permute",
    num_simulations=100,
    n_jobs=16,
)

refutation_placebo_df = pd.DataFrame(
    [
        {
            "Refutation Type": res_placebo.refutation_type,
            "Estimated Effect": res_placebo.estimated_effect,
            "New Effect": res_placebo.new_effect,
            "Refutation Result (p value)": res_placebo.refutation_result["p_value"],
        }
    ]
)

refutation_placebo_df

# COMMAND ----------

res_subset = model.refute_estimate(
    estimand=estimand,
    estimate=estimate,
    show_progress_bar=True,
    method_name="data_subset_refuter",
    subset_fraction=0.8,
    num_simulations=100,
    n_jobs=16,
)

refutation_subset_df = pd.DataFrame(
    [
        {
            "Refutation Type": res_subset.refutation_type,
            "Estimated Effect": res_subset.estimated_effect,
            "New Effect": res_subset.new_effect,
            "Refutation Result (p value)": res_subset.refutation_result["p_value"],
        }
    ]
)

refutation_subset_df

# COMMAND ----------

mlflow.autolog(disable=True)

coefficients = np.array([10, 0.02])
bias = 1000


def linear_gen(df):
    y_new = np.dot(df[["W0", "W1"]].values, coefficients) + bias
    return y_new


res_dummy_outcome = model.refute_estimate(
    estimand=estimand,
    estimate=estimate,
    show_progress_bar=True,
    method_name="dummy_outcome_refuter",
    outcome_function=linear_gen,
)[0]

refutation_dummy_outcome_df = pd.DataFrame(
    [
        {
            "Refutation Type": res_dummy_outcome.refutation_type,
            "Estimated Effect": res_dummy_outcome.estimated_effect,
            "New Effect": res_dummy_outcome.new_effect,
            "Refutation Result (p value)": res_dummy_outcome.refutation_result[
                "p_value"
            ],
        }
    ]
)

refutation_dummy_outcome_df

# COMMAND ----------

refutation_df = pd.concat(
    [
        refutation_random_common_cause_df,
        refutation_unobserved_common_cause_df,
        refutation_subset_df,
        refutation_placebo_df,
        refutation_dummy_outcome_df,
    ]
)
refutation_df
