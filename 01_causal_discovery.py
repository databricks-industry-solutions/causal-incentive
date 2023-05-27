# Databricks notebook source
# MAGIC %run ./util/notebook-config

# COMMAND ----------

# MAGIC %md
# MAGIC #Discovering the network of influences among the Features
# MAGIC
# MAGIC In order to isolate the influence we plan to estimate,  we need first to understand the relations among the available features.  We will use [PC algorithm](https://www.youtube.com/watch?v=o2A61bJ0UCw) implemented in the [PyWhy](https://www.pywhy.org/) package called [CausalLearn](https://github.com/py-why/causal-learn), to discover the basic skeleton of the network.  

# COMMAND ----------

import causallearn
from causallearn.search.ConstraintBased.PC import pc

# Parameters
parameters = {"node_names": input_df.columns, "alpha": 0.01, "indep_test": "fisherz"}
cg = pc(data=np.vstack(input_df.to_numpy()), **parameters)

# Visualization using pydot
cg.draw_pydot_graph()

# COMMAND ----------

# MAGIC %md
# MAGIC The bare bones skeleton discovered offers some interesting insights:
# MAGIC
# MAGIC - ```Discount``` seems to have a direct impact in ```Revenue```.
# MAGIC - ```Tech Support``` appears to have a direct impact in ```Revenue``` and a mediated one through ```New Product Adoption```.
# MAGIC - The ```New Engagement Strategy``` seems not to influence ```Revenue```.
# MAGIC - Both ```Revenue``` and ```New Engagement Strategy``` influence ```Planning Summit```. This representas a [collider pattern](https://en.wikipedia.org/wiki/Collider_(statistics)) which could result in creating a fictitious relation between  "New Engagement Strategy" and ```Revenue``` if ```Planning Summit``` is included as a feature during the influence estimation!! (this pattern is also known as [selection bias](https://catalogofbias.org/biases/collider-bias/))

# COMMAND ----------

# MAGIC %md
# MAGIC #Adding Domain Knowledge Assumptions to the Network

# COMMAND ----------

# MAGIC %md
# MAGIC The skeleton lacks directions in some of the relations.  Some of the missing directions are obvious:
# MAGIC
# MAGIC - ```Size``` -> ```IT Spend```   
# MAGIC - ```IT Spend``` -> ```Tech Support```
# MAGIC - ```Tech Support``` -> ```New Product Adoption```
# MAGIC - ```Major Flag``` -> ```New Engagment Strategy```
# MAGIC - ```Employee count``` -> ```PC Count```
# MAGIC
# MAGIC We will add these directions to the network:

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

# MAGIC %md
# MAGIC Some of the discovered directions are clearly wrong:
# MAGIC
# MAGIC - ```Revenue``` should not convert a company from the commercial sector to the public sector
# MAGIC - Providing the ```New Engagement Strategy``` incentive to a company will not converted it from the commerical sector to the public sector

# COMMAND ----------

# Correcting directions
inverted_directions = [
    {"from": "Revenue", "to": "Commercial Flag"},
    {"from": "New Engagement Strategy", "to": "Commercial Flag"},
]
invert_directions(
    causal_graph=cg,
    directions=inverted_directions,
)

cg.draw_pydot_graph()

# COMMAND ----------

# MAGIC %md
# MAGIC Even though some relations were not automatically discovered, The business experts of the software company are pretty convinced ```Global Flag``` and ```Major Flag``` have an influence in ```Revenue```.  We will add these assumptions to the network

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

# MAGIC %md
# MAGIC Finally, even thought there was no incentive assignation policy, we believe the basic characteristics of the company had an influence in the assignation.  We will add a directed relation from each of these characteristics to each of the incentives 

# COMMAND ----------

# Add effect from all basic characteristics to incentives
account_basic_characteristics = [
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

# MAGIC %md
# MAGIC The network defined above,  also known as the Casual Graph, will guide the <b>identification</b> and <b>estimation</b> phases.  We will proceed to store it as an artifact in an MLFlow experiment run to then use it in the next step (the ```02_identification_estimation``` notebook).  
# MAGIC
# MAGIC Please notice we are also storing in the [MLfLow](https://databricks.atlassian.net/wiki/spaces/UN/pages/2893873880/Brickstore) the algorithm used for the discovery, the parameters applied to the algorithm,  and all the alterations made to the discovered graph skeleton.

# COMMAND ----------

with mlflow.start_run(run_name="casual_discovery") as run:
    mlflow.log_params(
        {
            **{
                "algorithm": "PC",
                "library": f"casual-learn=={get_version('causal-learn')}",
            },
            **parameters,
            **{
                "added_directions": str(added_directions),
                "inverted_directions": str(inverted_directions),
                "added_missing_directed_relations": str(
                    added_missing_directed_relations
                ),
                "account_basic_characteristics": str(account_basic_characteristics),
            },
        }
    )

    # Serialize the graph to a file and log it to MLflow for reference.
    graph = to_pydot(cg.G, labels=input_df.columns)
    with open("/databricks/driver/graph.txt", "w") as f:
        f.write(graph)

    mlflow.log_artifact("/databricks/driver/graph.txt", artifact_path="graph")

# COMMAND ----------


