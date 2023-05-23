# Databricks notebook source
# MAGIC %run ./util/generate-data

# COMMAND ----------

# MAGIC %md
# MAGIC #Causal DAG discovery

# COMMAND ----------

import causallearn
from causallearn.search.ConstraintBased.PC import pc
%matplotlib inline

#parameters
parameters= {
  "node_names": input_df.columns, 
  "alpha": 0.01,
  "indep_test": "fisherz"
}
cg = pc(data=np.vstack(input_df.to_numpy()), **parameters)

# visualization using pydot
cg.draw_pydot_graph()

# COMMAND ----------

# Adding missing directions
added_directions = [
        {"from": "Size", "to": "IT Spend"},
        {"from": "IT Spend", "to": "Tech Support"},
        {"from": "Tech Support", "to": "New Product Adoption"},
        {"from": "Major Flag", "to": "New Engagement Strategy"},
        {"from": "Employee Count", "to": "PC Count"},
    ]
add_directions(
    causal_graph=cg,
    directions=added_directions,
)
cg.draw_pydot_graph()

# COMMAND ----------

# Correcting directions
inverted_directions = [
        {"from": "Revenue", "to": "Commercial Flag"},
        {"from": "New Engagement Strategy", "to": "Commercial Flag"},
        {"from": "Revenue", "to": "Commercial Flag"},
    ]
invert_directions(
    causal_graph=cg,
    directions=inverted_directions,
)

cg.draw_pydot_graph()

# COMMAND ----------

# Adding missing relations based on domain knowledge
added_missing_directed_relations = [
        {"from": "Global Flag", "to": "Revenue"},
        {"from": "Major Flag", "to": "Revenue"},
    ]

add_directions(
    causal_graph=cg,
    directions=added_missing_directed_relations,
)
cg.draw_pydot_graph()

# COMMAND ----------

# Add effect from all basic characteristics to incentives
account_basic_characteristics=[
        "Major Flag",
        "SMC Flag",
        "Commercial Flag",
        "IT Spend",
        "Employee Count",
        "PC Count",
    ]

add_relations_influencing_incentives(
    causal_graph=cg,
    incentives=["Discount", "Tech Support", "New Engagement Strategy"],
    account_basic_characteristics=account_basic_characteristics,
)
cg.draw_pydot_graph()

# COMMAND ----------

with mlflow.start_run(run_name="casual_discovery") as run:
  mlflow.log_params({
    **{
    "algorithm": "PC",
    "library": f"casual-learn=={get_version('causal-learn')}"}, 
    **parameters,    
    **{
      "added_directions": str(added_directions),
      "inverted_directions": str(inverted_directions),
      "added_missing_directed_relations": str(added_missing_directed_relations),
      "account_basic_characteristics": str(account_basic_characteristics)
    }
  })

  graph = to_pydot(cg.G, labels=input_df.columns)
  with open("/databricks/driver/graph.txt", "w") as f:
    f.write(graph)

  mlflow.log_artifact("/databricks/driver/graph.txt", artifact_path="graph") 

# COMMAND ----------


