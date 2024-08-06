# Databricks notebook source
# Common imports used throughout.

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import mlflow
import functools
import econml
import dowhy
import sklearn

from warnings import filterwarnings
filterwarnings("ignore", "iteritems is deprecated")

%matplotlib inline
%config InlineBackend.figure_format = "retina"

mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------

# Create catalog if it doesn't exist
catalog = "causal_solacc"
create_catalog_query = f"CREATE CATALOG IF NOT EXISTS {catalog}"
use_catalog_query = f"USE CATALOG {catalog}"

# Create database with the user's name if it doesn't exist
email = spark.sql('select current_user() as user').collect()[0]['user']
db = email.split('@')[0].replace('.', '_')
create_db_query = f"CREATE SCHEMA IF NOT EXISTS {catalog}.{db}"

_ = spark.sql(create_catalog_query)
_ = spark.sql(use_catalog_query)
_ = spark.sql(create_db_query)

# COMMAND ----------

# Utility methods for manipulating and serializing the causal graph.


import pydot
from causallearn.graph.Edge import Edge
from causallearn.graph.Edges import Edges
from causallearn.graph.Endpoint import Endpoint
from causallearn.graph.Graph import Graph
from causallearn.graph.Node import Node
from causallearn.graph.NodeType import NodeType
from causallearn.graph.GraphNode import GraphNode
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge


def add_background_knowledge(edges):
    """Add prior edges according to assigned causal connections."""
    bk = BackgroundKnowledge()
    for edge in edges:
        node1 = GraphNode(edge['from'])
        node2 = GraphNode(edge['to'])
        bk.add_required_by_node(node1, node2)
    return bk


def add_directions(causal_graph, directions):
    """Add directed edges on the given causal graph causal graph."""
    for direction in directions:
        causal_graph.G.add_directed_edge(
            node1=causal_graph.G.get_node(direction["from"]),
            node2=causal_graph.G.get_node(direction["to"]),
        )


def invert_directions(causal_graph, directions):
    """Invert existing directions on the given causal graph."""
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
    """Add edges from each caracteristic to each incentive in the given causal graph."""
    for incentive in incentives:
        for characteristic in account_basic_characteristics:
            causal_graph.G.add_directed_edge(
                node1=causal_graph.G.get_node(characteristic),
                node2=causal_graph.G.get_node(incentive),
            )


def to_pydot(G, labels, dpi=200):
    """Serialize the given causal graph G to a string.

    This is useful for logging to MLflow and passing among commands and notebooks.
    """
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
            return "none"
        elif endpoint == Endpoint.ARROW:
            return "normal"
        elif endpoint == Endpoint.CIRCLE:
            return "odot"
        else:
            raise NotImplementedError()

    edges = G.get_graph_edges()

    for edge in edges:
        node1 = edge.get_node1()
        node2 = edge.get_node2()
        node1_id = nodes.index(node1)
        node2_id = nodes.index(node2)
        dot_edge = pydot.Edge(
            labels[node1_id],
            labels[node2_id],
            dir="both",
            arrowtail=get_g_arrow_type(edge.get_endpoint1()),
            arrowhead=get_g_arrow_type(edge.get_endpoint2()),
        )

        pydot_g.add_edge(dot_edge)

    return pydot_g.to_string().replace("\n", " ")

# COMMAND ----------

# Utility methods related to setting up models for double machine learning (DML).


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
    """Set up the treatment (t) and outcome (y) models for DML."""

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

# Utility methods and classes for MLflow experiment tracking and model registration.


from pip._vendor import pkg_resources
from mlflow.models.signature import ModelSignature
from mlflow.types import DataType, Schema, TensorSpec, ColSpec


def get_version(package):
    """Retrive the version of the specified package for MLflow metadata logging."""
    package = package.lower()
    return next(
        (
            p.version
            for p in pkg_resources.working_set
            if p.project_name.lower() == package
        ),
        "No match",
    )


class ModelWrapper(mlflow.pyfunc.PythonModel):
    """Custom wrapper for logging our causal model."""

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
    """Capture the important libraries we need for logging the model."""
    model_env = mlflow.pyfunc.get_default_conda_env()
    model_env["dependencies"][-1]["pip"] += [
        f"dowhy=={dowhy.__version__}",
        f"econml=={econml.__version__}",
        f"sklearn=={sklearn.__version__}",
    ]
    return model_env


def register_dowhy_model(model_name, model, estimand, estimate):
    """Register a DoWhy model in MLflow."""
    # Define a dummy input and output schema for the model signature
    input_schema = Schema([ColSpec("double")])
    output_schema = Schema([ColSpec("double")])
    signature = ModelSignature(inputs=input_schema, outputs=output_schema)
    with mlflow.start_run(run_name=f"{model_name} run") as run:
        model_info = mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=ModelWrapper(
                model=model,
                estimand=estimand,
                estimate=estimate,
            ),
            registered_model_name=model_name,
            signature=signature,
            conda_env=get_model_env(),
        )
    return model_info


# Function to get the latest version of a registered model
def get_latest_model_version(client, model_name):
    latest_version = 1  # Initialize the latest version to 1
    # Iterate through all model versions for the given registered model name
    for mv in client.search_model_versions(f"name='{model_name}'"):
        version_int = int(mv.version)  # Convert version string to integer
        # Update the latest version if a higher version is found
        if version_int > latest_version:
            latest_version = version_int
    return latest_version  # Return the latest version number


def get_registered_wrapped_model(model_name):
    client = mlflow.MlflowClient()
    latest_model_version = get_latest_model_version(client, model_name)
    wrapped_model = mlflow.pyfunc.load_model(
        f"models:/{model_name}/{latest_model_version}"
    )
    return wrapped_model.unwrap_python_model()


def get_registered_wrapped_model_estimator(model_name):
    wrapped_model = get_registered_wrapped_model(model_name=model_name)
    return wrapped_model.get_estimate()._estimator_object


def load_graph_from_latest_mlflow_run(experiment_name):
    """Load the causal graph from the most recent MLflow causal run."""

    # Find all the runs from the prior notebook for causal discovery
    client = mlflow.MlflowClient()
    experiment = mlflow.get_experiment_by_name(experiment_name)
    discovery_runs = client.search_runs(
        experiment_ids=[experiment.experiment_id], 
        filter_string="attributes.run_name='causal_discovery'",
        order_by=["start_time DESC"],
        max_results=1)

    # Make sure there is at least one run available
    assert len(discovery_runs) == 1, "please run the notebook 01_causal_discovery at least once"

    # The only result should be the latest based on our search_runs call
    latest_discovery_run = discovery_runs[0]
    latest_discovery_run.info.artifact_uri

    # Load the graph artifact from the run
    graph = mlflow.artifacts.load_text(latest_discovery_run.info.artifact_uri + "/graph/graph.txt")

    return graph


# COMMAND ----------

# Utility methods for reporting on various policies and their effects.


def compare_policies_effects(final_df):
    """Compare our various policy options and their effects."""
    average_effect_recommended_incentive_df = pd.DataFrame(
        final_df[
            [
                "tech support net effect",
                "discount net effect",
                "tech support and discount net effect",
                "no incentive net effect",
                "recommended incentive net effect",
                "no policy"
            ]
        ].mean()
    ).T

    average_effect_recommended_incentive_df.columns = [
        "Always giving only 'tech support'",
        "Always giving only 'discount'",
        "Always giving 'tech support' and 'discount'",
        "Giving no incentive",
        "Giving recommended incentive",
        "No Policy"
    ]
    average_effect_recommended_incentive_df = average_effect_recommended_incentive_df.T
    average_effect_recommended_incentive_df.columns = [
        "Average marginal profit per account"
    ]
    return average_effect_recommended_incentive_df


def compare_estimations_vs_ground_truth(original_df, estimates_df):
    """Compares the estimates the we were able to infer vs. the ground truth."""

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


def assign_treatment_label(x):
    """Assigns a treatment type based on the columns Tech Support and Discount."""
    if x["Tech Support"] == 1 and x["Discount"] == 1:
        return "tech support and discount"
    elif x["Tech Support"] == 1 and x["Discount"] == 0:
        return "tech support"
    elif x["Tech Support"] == 0 and x["Discount"] == 1:
        return "discount"
    elif x["Tech Support"] == 0 and x["Discount"] == 0:
        return "no incentive"
      

def plot_policy(df, treatment):
    """Function to plot a policy for each customer."""
    all_treatments = np.array(['no incentive', 'tech support', 'discount', 'tech support and discount'])
    ax1 = sns.scatterplot(
        x=df["Size"],
        y=df["PC Count"],
        hue=treatment,
        hue_order=all_treatments,
        cmap="Dark2",
        s=40,
    )
    plt.legend(title="Investment Policy")
    plt.setp(
        ax1,
        xlabel="Size",
        ylabel="PC Count",
        title="Investment Policy by Customer",
    )
    plt.show()

# COMMAND ----------

# Common configuration settings and data loading across all notebooks.


# For running from a job, the experiment needs to be created in a workspace object.
username = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
experiment_name = "/Users/{}/incentive-effects-estimation".format(username)
mlflow.set_experiment(experiment_name)


# Load the data file from a parquet file as a pandas dataframe for use throughout.
# Since it's relatively small and for pedagogical purposes, we don't create a Delta
# table as you'd want to in practice.
ground_truth_path = "s3://db-gtm-industry-solutions/data/rcg/causal_incentive/ground_truth.parquet"
ground_truth_df = spark.read.parquet(ground_truth_path).toPandas()
input_df = ground_truth_df.iloc[:, 0:14]


# Some additional metadata we can use as needed to work with the input dataframe.
treatment_cols = ["Tech Support", "Discount", "New Engagement Strategy"]
characteristic_cols = [
    "Global Flag", "Major Flag", "SMC Flag", "Commercial Flag", 
    "Planning Summit", "New Product Adoption",
    "IT Spend", "Employee Count", "PC Count", "Size"]
categorical_cols = [
    "Tech Support", "Discount", "New Engagement Strategy",
    "Global Flag", "Major Flag", "SMC Flag", "Commercial Flag", 
    "Planning Summit", "New Product Adoption"]
numerical_cols = ["IT Spend", "Employee Count", "PC Count", "Size"]
target_col = "Revenue"

# A type map to use for normal exploratory analysis
normal_type_map = {k: v for (k, v) in 
    [(c, "category") for c in categorical_cols] +
    [(c, "double") for c in numerical_cols + [target_col]]}

# A type map to use for profiling with the Databricks profiler
# (it helps to have the categoricals as boolean for it to treat the binaries as categorical)
summarize_type_map = {k: v for (k, v) in 
    [(c, "boolean") for c in categorical_cols] +
    [(c, "double") for c in numerical_cols + [target_col]]}
