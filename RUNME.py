# Databricks notebook source
# MAGIC %md This notebook sets up the companion cluster(s) to run the solution accelerator. It also creates the Workflow to illustrate the order of execution. Happy exploring! 
# MAGIC 🎉
# MAGIC
# MAGIC **Steps**
# MAGIC 1. Simply attach this notebook to a cluster and hit Run-All for this notebook. A multi-step job and the clusters used in the job will be created for you and hyperlinks are printed on the last block of the notebook. 
# MAGIC
# MAGIC 2. Run the accelerator notebooks: Feel free to explore the multi-step job page and **run the Workflow**, or **run the notebooks interactively** with the cluster to see how this solution accelerator executes. 
# MAGIC
# MAGIC     2a. **Run the Workflow**: Navigate to the Workflow link and hit the `Run Now` 💥. 
# MAGIC   
# MAGIC     2b. **Run the notebooks interactively**: Attach the notebook with the cluster(s) created and execute as described in the `job_json['tasks']` below.
# MAGIC
# MAGIC **Prerequisites** 
# MAGIC 1. You need to have cluster creation permissions in this workspace.
# MAGIC
# MAGIC 2. In case the environment has cluster-policies that interfere with automated deployment, you may need to manually create the cluster in accordance with the workspace cluster policy. The `job_json` definition below still provides valuable information about the configuration these series of notebooks should run with. 
# MAGIC
# MAGIC **Notes**
# MAGIC 1. The pipelines, workflows and clusters created in this script are not user-specific. Keep in mind that rerunning this script again after modification resets them for other users too.
# MAGIC
# MAGIC 2. If the job execution fails, please confirm that you have set up other environment dependencies as specified in the accelerator notebooks. Accelerators may require the user to set up additional cloud infra or secrets to manage credentials. 

# COMMAND ----------

# DBTITLE 0,Install util packages
# MAGIC %pip install git+https://github.com/databricks-academy/dbacademy@v1.0.13 git+https://github.com/databricks-industry-solutions/notebook-solution-companion@safe-print-html --quiet --disable-pip-version-check
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

from solacc.companion import NotebookSolutionCompanion
nsc = NotebookSolutionCompanion()

# COMMAND ----------

job_json = {
        "timeout_seconds": 28800,
        "max_concurrent_runs": 1,
        "tags": {
            "usage": "solacc_testing",
            "group": "SOLACC"
        },
        "tasks": [
            {
                "job_cluster_key": "causal_cluster",
                "notebook_task": {
                    "notebook_path": f"00_intro"
                },
                "task_key": "00_intro",
                "libraries": [
                  {
                      "pypi": {
                          "package": "networkx==2.8.8" 
                      }
                  },
                  {
                      "pypi": {
                          "package": "causal-learn==0.1.3" 
                      }
                  },
                  {
                      "pypi": {
                          "package": "dowhy==0.10" 
                      }
                  },
                  {
                      "pypi": {
                          "package": "econml==0.14.0" 
                      }
                  }
              ]
            },
            {
                "job_cluster_key": "causal_cluster",
                "notebook_task": {
                    "notebook_path": f"01_causal_discovery"
                },
                "task_key": "01_causal_discovery",
                "depends_on": [
                    {
                        "task_key": "00_intro"
                    }
                ],
                "libraries": [
                  {
                      "pypi": {
                          "package": "networkx==2.8.8" 
                      }
                  },
                  {
                      "pypi": {
                          "package": "causal-learn==0.1.3" 
                      }
                  },
                  {
                      "pypi": {
                          "package": "dowhy==0.10" 
                      }
                  },
                  {
                      "pypi": {
                          "package": "econml==0.14.0" 
                      }
                  }
              ]
            },
            {
                "job_cluster_key": "causal_cluster",
                "notebook_task": {
                    "notebook_path": f"02_identification_estimation"
                },
                "task_key": "02_identification_estimation",
                "depends_on": [
                    {
                        "task_key": "01_causal_discovery"
                    }
                ],
                "libraries": [
                  {
                      "pypi": {
                          "package": "networkx==2.8.8" 
                      }
                  },
                  {
                      "pypi": {
                          "package": "causal-learn==0.1.3" 
                      }
                  },
                  {
                      "pypi": {
                          "package": "dowhy==0.10" 
                      }
                  },
                  {
                      "pypi": {
                          "package": "econml==0.14.0" 
                      }
                  }
              ]
            },
            {
                "job_cluster_key": "causal_cluster",
                "notebook_task": {
                    "notebook_path": f"03_promotional_offer_recommender"
                },
                "task_key": "03_promotional_offer_recommender",
                "depends_on": [
                    {
                        "task_key": "02_identification_estimation"
                    }
                ],
                "libraries": [
                  {
                      "pypi": {
                          "package": "networkx==2.8.8" 
                      }
                  },
                  {
                      "pypi": {
                          "package": "causal-learn==0.1.3" 
                      }
                  },
                  {
                      "pypi": {
                          "package": "dowhy==0.10" 
                      }
                  },
                  {
                      "pypi": {
                          "package": "econml==0.14.0" 
                      }
                  }
              ]
            },
            {
                "job_cluster_key": "causal_cluster",
                "notebook_task": {
                    "notebook_path": f"04_refutation"
                },
                "task_key": "04_refute",
                "depends_on": [
                    {
                        "task_key": "03_promotional_offer_recommender"
                    }
                ],
                "libraries": [
                  {
                      "pypi": {
                          "package": "networkx==2.8.8" 
                      }
                  },
                  {
                      "pypi": {
                          "package": "causal-learn==0.1.3" 
                      }
                  },
                  {
                      "pypi": {
                          "package": "dowhy==0.10" 
                      }
                  },
                  {
                      "pypi": {
                          "package": "econml==0.14.0" 
                      }
                  }
              ]
            }
        ],
        "job_clusters": [
            {
                "job_cluster_key": "causal_cluster",
                "new_cluster": {
                    "spark_version": "14.3.x-cpu-ml-scala2.12",
                    "num_workers": 0,
                    "spark_conf": {
                        "spark.master": "local[*, 4]",
                        "spark.databricks.cluster.profile": "singleNode"
                    },
                    "custom_tags": {
                        "ResourceClass": "SingleNode"
                    },
                    "node_type_id": {"AWS": "i3.8xlarge", "MSA": "Standard_E32_v3", "GCP": "n1-highmem-32"},
                    "data_security_mode": "SINGLE_USER",
                }
            }
        ]
    }

# COMMAND ----------

# DBTITLE 1,Deploy job and cluster
dbutils.widgets.dropdown("run_job", "False", ["True", "False"])
run_job = dbutils.widgets.get("run_job") == "False"
nsc.deploy_compute(job_json, run_job=run_job)

# COMMAND ----------


