# Databricks notebook source
# MAGIC %run ./notebook-config

# COMMAND ----------

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle as pkl
import os
from scipy.special import expit, logit

%matplotlib inline

# COMMAND ----------

dbutils.widgets.dropdown("display_results", "no", ["yes", "no"])

# COMMAND ----------

display_results = dbutils.widgets.get("display_results") == "yes"

# COMMAND ----------

# Data Generation

# COMMAND ----------

#### Generate covariates W, X.

#Most features are independent but some are correlated.

# COMMAND ----------

np.random.seed(1)

n = 10000

global_flag = np.random.binomial(n=1, p=0.2, size=n)
major_flag = np.random.binomial(n=1, p=0.2, size=n)
smc_flag = np.random.binomial(n=1, p=0.5, size=n)
commercial_flag = np.random.binomial(n=1, p=0.7, size=n)

size = np.random.exponential(scale=100000, size=n) + np.random.uniform(
    low=5000, high=15000, size=n
)
it_spend = np.exp(np.log(size) - 1.4 + np.random.uniform(size=n))

employee_count = np.exp(
    np.log(
        np.random.exponential(scale=50, size=n)
        + np.random.uniform(low=5, high=10, size=n)
    )
    * 0.9
    + 0.4
    + np.random.uniform(size=n)
)
pc_count = np.exp((np.log(employee_count) - np.random.uniform(size=n) - 0.4) / 0.9 )

size = size.astype(int)
it_spend = it_spend.astype(int)
pc_count = pc_count.astype(int)
employee_count = employee_count.astype(int)


new_X = pd.DataFrame(
    {
        "Global Flag": global_flag,
        "Major Flag": major_flag,
        "SMC Flag": smc_flag,
        "Commercial Flag": commercial_flag,
        "IT Spend": it_spend,
        "Employee Count": employee_count,
        "PC Count": pc_count,
        "Size": size,
    }
)

# COMMAND ----------

#### Generate treatment from covariates

# COMMAND ----------

# controls
W_cols = ['Major Flag', 'SMC Flag', 'Commercial Flag', 'IT Spend', 'Employee Count', 'PC Count']

# Tech Support
coefs_W_tech = np.array([0, 0, 0, 0.00001, 0, 0])
const_tech = -0.465
noise_tech = np.random.normal(scale=2.0, size = n)
z_tech = new_X[W_cols] @ coefs_W_tech + const_tech + noise_tech
prob_tech = expit(z_tech)
tech_support = np.random.binomial(n = 1, p = prob_tech, size = n)

# Discount
coefs_W_discount = np.array([0.2, 0, 0, 0.000005, 0, 0])
const_discount = -0.27
noise_discount = np.random.normal(scale=1.5, size = n)
z_discount = new_X[W_cols] @ coefs_W_discount + const_discount + noise_discount
prob_discount = expit(z_discount)
discount = np.random.binomial(n = 1, p = prob_discount, size = n)

# New Engagement Strategy
coefs_W_t3 = np.array([0.5, 0.1, -0.2, 0, 0.005, -0.005])
const_t3 = -0.12
noise_t3 = np.random.normal(scale=1.0, size = n)
z_t3 = new_X[W_cols] @ coefs_W_t3 + const_t3 + noise_t3
prob_t3 = expit(z_t3)
t3 = np.random.binomial(n = 1, p = prob_t3, size = n)

# COMMAND ----------


#### Mediator

# generated from Tech Support

# COMMAND ----------

z_m = tech_support*2-1 + np.random.normal(size=n)
prob_m = expit(z_m)
m = np.random.binomial(n = 1, p = prob_m, size = n)

# COMMAND ----------

#### Outcome

# COMMAND ----------

# X features determine heterogeneous treatment effects
X_cols = ['Global Flag', 'Size']
theta_coef_tech_support = [500, 0.02]
theta_const_tech_support = 5000
te_tech_support = new_X[X_cols] @ theta_coef_tech_support + theta_const_tech_support

theta_coef_discount = [-1000, 0.05]
theta_const_discount = 0
te_discount = new_X[X_cols] @ theta_coef_discount + theta_const_discount

theta_coef_t3 = [0, 0]
theta_const_t3 = 0
te_t3 = new_X[X_cols] @ theta_coef_t3 + theta_const_t3

y_te = te_tech_support*tech_support + te_discount*discount + te_t3*t3

g_coefs = np.array([2000, 0, 5000, 0.25, 0.0001, 0.0001])
g_const = 5000
g_y = new_X[W_cols] @ g_coefs + g_const

y_noise = np.random.normal(scale = 1000, size = n)

mediator_effect = 2000*m

y = pd.Series(y_te + g_y + y_noise + mediator_effect)

# COMMAND ----------

#### Collider

# Caused by both outcome and New Engagement Strategy

# COMMAND ----------

z_c = 0.03*y + 1000*t3 - 1400
prob_c = expit(z_c)
c = np.random.binomial(n = 1, p = prob_c, size = n)

# COMMAND ----------

## Consolidate

# COMMAND ----------

ground_truth_df = (
    pd.concat(
        [
            new_X,
            pd.DataFrame(
                {
                    'Tech Support': tech_support,
                    'Discount': discount,
                    'New Engagement Strategy': t3,
                    'New Product Adoption': m,
                    'Planning Summit': c,
                    'Revenue': y,
                    'Direct Treatment Effect: Tech Support': te_tech_support,
                    'Total Treatment Effect: Tech Support': np.round(te_tech_support + (expit(1) - expit(-1))*2000, decimals=2), # incorporate effect from mediator into total effect.
                    'Direct Treatment Effect: Discount': te_discount,
                    'Total Treatment Effect: Discount': te_discount,
                    'Direct Treatment Effect: New Engagement Strategy': te_t3,
                    'Total Treatment Effect: New Engagement Strategy': te_t3,
                }
            )
        ],
        axis = 1,
    )
    .assign(Revenue = lambda df: df['Revenue'].round(2))
)

# COMMAND ----------

#### Ground Truth ATE check 

# COMMAND ----------

if display_results:
  print(ground_truth_df.filter(like='Treatment Effect').mean(axis=0))

# COMMAND ----------

import networkx as nx
import dowhy.gcm

if display_results:
  ynode = "Revenue"
  mednode = "New Product Adoption"
  collider = "Planning Summit"
  T_cols = ["Tech Support", "Discount", "New Engagement Strategy"]
  trueg = nx.DiGraph()
  trueg.nodes = ground_truth_df.loc[:, "Global Flag":"Revenue"].columns
  trueg.add_edges_from([(w, "Revenue") for w in W_cols])
  trueg.add_edges_from([(x, "Revenue") for x in X_cols])  # effect modifiers
  for t in T_cols:
      trueg.add_edges_from([(w, t) for w in W_cols])
      if (
          ground_truth_df[f"Direct Treatment Effect: {t}"].mean(axis=0) != 0
          and ground_truth_df[f"Total Treatment Effect: {t}"].mean(axis=0) != 0
      ):
          trueg.add_edge(t, ynode)
  trueg.add_edge(T_cols[0], mednode)
  trueg.add_edge(mednode, ynode)  # mediator
  trueg.add_edge(T_cols[2], collider)
  trueg.add_edge(ynode, collider) # collider

  trueg.add_edge("Size", "IT Spend")
  trueg.add_edge("Employee Count", "PC Count")

  dowhy.gcm.util.plot(trueg, figure_size=(20, 20))

# COMMAND ----------

input_df = ground_truth_df.iloc[:,0:14]
