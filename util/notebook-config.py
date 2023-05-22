# Databricks notebook source
# MAGIC %matplotlib inline

# COMMAND ----------

import mlflow
username = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
mlflow.set_experiment('/Users/{}/esg-scoring'.format(username))

# COMMAND ----------

def add_directions(causal_graph, directions):
    for direction in directions:
        causal_graph.G.add_directed_edge(
            node1=causal_graph.G.get_node(direction["from"]),
            node2=causal_graph.G.get_node(direction["to"]),
        )
def invert_directions(causal_graph, directions):
    for direction in directions:
        causal_graph.G.remove_connecting_edge(
            node1=causal_graph.G.get_node(direction["from"]),
            node2=causal_graph.G.get_node(direction["to"]),
        )
        causal_graph.G.add_directed_edge(
            node1=causal_graph.G.get_node(direction["to"]),
            node2=causal_graph.G.get_node(direction["from"]),
        )

def add_relations_influencing_incentives(
    causal_graph, incentives, account_basic_characteristics
):
    for incentive in incentives:
        for characteristic in account_basic_characteristics:
            causal_graph.G.add_directed_edge(
                node1=causal_graph.G.get_node(characteristic),
                node2=causal_graph.G.get_node(incentive),
            )

# COMMAND ----------

from pip._vendor import pkg_resources

def get_version(package):
    package = package.lower()
    return next((p.version for p in pkg_resources.working_set if p.project_name.lower() == package), "No match")

# COMMAND ----------

# ignoring deprication warnings
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings

warnings.simplefilter("ignore", category=NumbaDeprecationWarning)
warnings.simplefilter("ignore", category=NumbaPendingDeprecationWarning)

# Generic ML imports
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Lasso

# EconML imports
from econml.dml import LinearDML
from econml.cate_interpreter import SingleTreePolicyInterpreter

def setup_treatment_and_out_models():

  # transformer that performs standard scaling on non-binary variables
  ct = ColumnTransformer(
      [
          (
              "num_transformer",
              StandardScaler(),
              lambda df: pd.DataFrame(df)
              .apply(pd.Series.nunique)
              .loc[lambda df: df > 2]
              .index.tolist(),
          )
      ],
      remainder="passthrough",
  )

  model_t = make_pipeline(
      ct, LogisticRegression(C=1300, max_iter=1000)
  )  # model used to predict treatment
  model_y = make_pipeline(ct, Lasso(alpha=20))  # model used to predict outcome
  return model_t, model_y

# COMMAND ----------

#Helper functions
import pydot
from causallearn.graph.Edge import Edge
from causallearn.graph.Edges import Edges
from causallearn.graph.Endpoint import Endpoint
from causallearn.graph.Graph import Graph
from causallearn.graph.Node import Node
from causallearn.graph.NodeType import NodeType

def to_pydot(G, labels, dpi = 200):
    nodes = G.get_nodes()
    if labels is not None:
        assert len(labels) == len(nodes)

    pydot_g = pydot.Dot("", graph_type="digraph", fontsize=18)
    pydot_g.obj_dict["attributes"]["dpi"] = dpi
    nodes = G.get_nodes()
    for i, node in enumerate(nodes):
      node_name = labels[i] if labels is not None else node.get_name()
      pydot_g.add_node(pydot.Node(labels[i], label=node.get_name()))
      pydot_g.add_node(pydot.Node(labels[i], label=node_name))

    def get_g_arrow_type(endpoint):
      if endpoint == Endpoint.TAIL:
          return 'none'
      elif endpoint == Endpoint.ARROW:
          return 'normal'
      elif endpoint == Endpoint.CIRCLE:
          return 'odot'
      else:
          raise NotImplementedError()

    edges = G.get_graph_edges()

    for edge in edges:
        node1 = edge.get_node1()
        node2 = edge.get_node2()
        node1_id = nodes.index(node1)
        node2_id = nodes.index(node2)
        dot_edge = pydot.Edge(labels[node1_id], labels[node2_id], dir='both', arrowtail=get_g_arrow_type(edge.get_endpoint1()),
                              arrowhead=get_g_arrow_type(edge.get_endpoint2()))

        pydot_g.add_edge(dot_edge)
        
    return pydot_g.to_string().replace("\n", " ") 

# COMMAND ----------

import econml
import dowhy
import mlflow
import sklearn

class ModelWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model, estimand, estimate):
        self._model = model
        self._estimand = estimand
        self._estimate = estimate

    def get_model(self):
        return self._model

    def get_estimand(self):
        return self._estimand

    def get_estimate(self):
        return self._estimate

    def predict(self, context, model_input):
        return self._estimate._estimator_object.const_marginal_effect(model_input)


def get_model_env():
    model_env = mlflow.pyfunc.get_default_conda_env()
    model_env["dependencies"][-1]["pip"] += [
        f"dowhy=={dowhy.__version__}",
        f"econml=={econml.__version__}",
        f"sklearn=={sklearn.__version__}",
    ]
    return model_env


def register_dowhy_model(model_name, model, estimand, estimate):
    with mlflow.start_run(run_name=f"{model_name} run") as run:
        model_info = mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=ModelWrapper(
                model=model,
                estimand=estimand,
                estimate=estimate,
            ),
            conda_env=get_model_env(),
        )

    return mlflow.register_model(
        model_uri=f"runs:/{run.info.run_id}/model", name=model_name
    )

# COMMAND ----------

import mlflow

def get_registered_wrapped_model(model_name):
  client = mlflow.MlflowClient()
  latest_model_versions = client.get_latest_versions(
      name=model_name
  )

  if len(latest_model_versions) >0:
    latest_model_version = latest_model_versions[0].version
  else:
    raise Exception(f"There are no registered versions for model {model_name}")  

  wrapped_model = mlflow.pyfunc.load_model(
      f"models:/{model_name}/{latest_model_version}"
  )

  return wrapped_model.unwrap_python_model()

def get_registered_wrapped_model_estimator(model_name):
  wrapped_model = get_registered_wrapped_model(model_name=model_name)
  return wrapped_model.get_estimate()._estimator_object  

# COMMAND ----------

def compare_policies_effects(final_df):
  average_effect_recommended_incentive_df = pd.DataFrame(
      final_df[
          [
              "tech support net effect",
              "discount net effect",
              "tech support and discount net effect",
              "no incentive net effect",
              "recommended incentive net effect",
          ]
      ].mean()
  ).T

  average_effect_recommended_incentive_df.columns = [
      "Always giving only 'tech support'",
      "Always giving only  'discount'",
      "Always giving 'tech support' and 'discount'",
      "Giving no incentive",
      "Giving recommended incentive",
  ]
  average_effect_recommended_incentive_df = average_effect_recommended_incentive_df.T
  average_effect_recommended_incentive_df.columns = [
      "Average net dollar return per account"
  ]
  return average_effect_recommended_incentive_df

# COMMAND ----------

import functools

def compare_estimations_vs_ground_truth(original_df, estimates_df):
  
  ground_truth_df = pd.DataFrame(
      original_df[
          [
              "Direct Treatment Effect: Tech Support",
              "Total Treatment Effect: Tech Support",
              "Total Treatment Effect: Discount",
              "Total Treatment Effect: New Engagement Strategy",
          ]
      ].mean()
  ).T 

  comparison_df = pd.concat([ground_truth_df, estimates_df], axis=1)

  return comparison_df[
      functools.reduce(
          lambda acc, x: acc + [x[0], x[1]],
          [
              [ground_truth, estimate]
              for ground_truth, estimate in zip(
                  ground_truth_df.columns, estimates_df.columns
              )
          ],
      )
  ]
