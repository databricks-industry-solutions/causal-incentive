# Databricks notebook source
# MAGIC %run ./util/notebook-config

# COMMAND ----------

# MAGIC %md
# MAGIC ##Refutation

# COMMAND ----------

# MAGIC %md
# MAGIC - <b>How much should we trust the estimators used by the recommender?</b>
# MAGIC
# MAGIC - <b>Would they break if new features are added?</b>
# MAGIC
# MAGIC - <b>What if there was a subset of the data driving the results, but the estimation does not applied to other subsets?</b>
# MAGIC
# MAGIC To questions related to the level of trust we should have in the estimators, we will execute a series of tests to determine the degree of sensitivity the models have to deviations in our dataset and assumptions
# MAGIC
# MAGIC The [DoWhy](https://www.pywhy.org/dowhy/v0.8/user_guide/effect_inference/refute.html) package of [PyWhy](https://github.com/py-why) provides us with a battery of predefine "refutation" tests we can easily use for these purpose. [The approach taken by DoWhy](https://github.com/py-why/dowhy/issues/312) is based on [statistical hypothesis testing](https://en.wikipedia.org/wiki/Statistical_hypothesis_testing), where the null hypothesis is that the test did not detect a problem in the model.  In other words, if the test p-value is less than 0.05, you would conclude with a 95% confidence level that the model should not be trusted if a similar deviation as the one simluated would actually take place.
# MAGIC
# MAGIC Please note this phase is computationally expensive as it simulates many scenarios.  In order to keep the tests simple and easy to follow, no distributed computing approach have been.  Some multithreading capabilities already provided by the package as been leverage.  This notebook will take longer time than the previous ones.  To avoid even a longer execution time we will focus on one of the estimators: the ```Discount``` effect estimator.  These test don't need to be continuesly execute,  instead they should be executed when when new estimators are trained.  

# COMMAND ----------

# MAGIC %md
# MAGIC First, lets load the ```Discount``` effect estimator model from [MLflow](https://www.databricks.com/product/managed-mlflow)

# COMMAND ----------

wrapped_model = get_registered_wrapped_model(model_name="discount_dowhy_model")

model = wrapped_model.get_model()
estimand = wrapped_model.get_estimand()
estimate = wrapped_model.get_estimate()

# COMMAND ----------

# MAGIC %md
# MAGIC Let's now test if adding an artificial feature which influences both the the probability of giving a ```Discount``` and the ```Revenue```, would yield a significant different result.

# COMMAND ----------

# MAGIC %matplotlib inline
# MAGIC res_random_common_cause = model.refute_estimate(
# MAGIC     estimand=estimand,
# MAGIC     estimate=estimate,
# MAGIC     show_progress_bar=True,
# MAGIC     method_name="random_common_cause",
# MAGIC     num_simulations=100,
# MAGIC     n_jobs=16,
# MAGIC )
# MAGIC
# MAGIC refutation_random_common_cause_df = pd.DataFrame(
# MAGIC     [
# MAGIC         {
# MAGIC             "Refutation Type": res_random_common_cause.refutation_type,
# MAGIC             "Estimated Effect": res_random_common_cause.estimated_effect,
# MAGIC             "New Effect": res_random_common_cause.new_effect,
# MAGIC             "Refutation Result (p value)": res_random_common_cause.refutation_result[
# MAGIC                 "p_value"
# MAGIC             ],
# MAGIC         }
# MAGIC     ]
# MAGIC )
# MAGIC
# MAGIC refutation_random_common_cause_df

# COMMAND ----------

# MAGIC %md
# MAGIC Let's now execute a similar test, but this time the new feature will be similated but not included in the estimation.  This mimics an scenario where a factor that influences both ```Discount``` and ```Revenue``` exists but we are unaware of it,  in other words the factor is "un-observed".  
# MAGIC
# MAGIC The test reports back a plot showing the effect of different "unobserved" factor values in the probability of applying "treatment" or incentive (```Discount```),  the outcome value (```Revenue```), and the estimated incentive effect.

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

# MAGIC %md
# MAGIC The following test changes the order of the ```Discount``` values in the dataset,  braking the relation between the ```Discount``` and the ```Revenue``` in a given account.  As a result the model should not predict a good estimation.    

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

# MAGIC %md
# MAGIC Next, we will apply the estimation on many non-overlaping subsets of the dataset.  The average of the estimations should be close enough to the estimation done with the full dataset.

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

# MAGIC %md
# MAGIC Lastely,  we will replace the ```Revenue``` values with artifitial randomly generated values. The estimation should show no effect. 

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

# MAGIC %md
# MAGIC When comparing all the tests results we can see the estimator is most sensitive to unobserved factors. This is in a way expected, as the test breaks one of the [assumptions](https://journals.lww.com/epidem/fulltext/2009/01000/the_consistency_statement_in_causal_inference__a.3.aspx#:~:text=Three%20assumptions%20sufficient%20to%20identify,measurement%20of%20the%20outcome%E2%80%9D).) in which the approaches presented are based.  The value obtain by applying that specific test to understanding the degree of impact of unobserved factors in the estimation.
# MAGIC
# MAGIC The rest of the tests have p-values higher than 0.05

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

# COMMAND ----------

# MAGIC %md
# MAGIC © 2023 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License. All included or referenced third party libraries are subject to the licenses set forth below.
# MAGIC
# MAGIC | library                                | description             | license    | source                                              |
# MAGIC |----------------------------------------|-------------------------|------------|-----------------------------------------------------|
# MAGIC | dowhy   | Python library for causal inference that supports explicit modeling and testing of causal assumptions | MIT   | https://pypi.org/project/dowhy/          
# MAGIC | econml   |  contains several methods for calculating Conditional Average Treatment Effects | MIT    | https://pypi.org/project/econml/  
# MAGIC | causal-learn   | python package for causal discovery  | MIT    | https://pypi.org/project/causal-learn/          
