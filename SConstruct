import os
import os.path
import logging
import random
import subprocess
import shlex
import gzip
import re
import functools
import time
import imp
import sys
import json
from hashlib import md5
import steamroller


# workaround needed to fix bug with SCons and the pickle module
del sys.modules['pickle']
sys.modules['pickle'] = imp.load_module('pickle', *imp.find_module('pickle'))
import pickle


vars = Variables("custom.py")
vars.AddVariables(
    ("OUTPUT_WIDTH", "", 100),

    ("RANDOM_COMPONENTS", "", 1000),
    ("MINIMUM_CONSTANTS", "", 1),
    ("MAXIMUM_CONSTANTS", "", 4),

    ("TRAIN_PROPORTION", "", 0.9),
    ("DEV_PROPORTION", "", 0.1),
    
    ("MAX_CATEGORICAL", "", 50),
    ("RANDOM_LINE_COUNT", "", 1000),
    ("MAX_COLLAPSE", "", 0),
    ("MAX_SEQUENCE_LENGTH", "", 32),

    
    ("LOG_LEVEL", "", "INFO"),
    ("LINE_COUNT", "", 1000),
    ("BATCH_SIZE", "", 32),
    ("MAX_EPOCHS", "", 10),
    ("LEARNING_RATE", "", 0.001),
    ("RANDOM_RESTARTS", "", 0),
    ("MOMENTUM", "", None),
    ("EARLY_STOP", "", 20),
    ("PATIENCE", "", 10),
    ("COLLAPSE_IDENTICAL", "", False),
    ("CLUSTER_REDUCTION", "", 0.9),
    BoolVariable("USE_GPU", "", False),
    BoolVariable("USE_GRID", "", False),
    ("GPU_PREAMBLE", "", "module load cuda90/toolkit"),
    ("DEFAULT_TRAIN_CONFIG", "", {"DEPTH" : [0],
                                  "EMBEDDING_SIZE" : [32],
                                  "HIDDEN_SIZE" : [32],
                                  "AUTOENCODER_SHAPES" : [(128,)],
                              }),
    ("DEFAULT_APPLY_CONFIG", "", {}),
    ("DEFAULT_SPLIT_PROPORTIONS", "", [("train", 0.50), ("dev", 0.25), ("test", 0.25)]),
    ("EXPERIMENTS", "", {"arithmetic" : {"SCHEMA" : "schemas/arithmetic.json"}}),    
)


def save_config(target, source, env):
    with gzip.open(target[0].rstr(), "wt") as ofd:
        json.dump(dict(source[0].read()), ofd)


env = Environment(variables=vars, ENV=os.environ, TARFLAGS="-c -z", TARSUFFIX=".tgz",
                  tools=["default", steamroller.generate])


env.Append(BUILDERS={"PreprocessArithmetic" : env.Builder(**env.ActionMaker("python",
                                                                            "src/preprocess_arithmetic.py",
                                                                            "--output ${TARGETS[0]} --components ${RANDOM_COMPONENTS} --minimum_constants ${MINIMUM_CONSTANTS} --maximum_constants ${MAXIMUM_CONSTANTS}")),
                     "PreprocessStanfordSentiment" : env.Builder(**env.ActionMaker("python",
                                                                    "src/preprocess_stanford_sentiment.py",
                                                                    "--train ${SOURCES[0]} --dev ${SOURCES[1]} --test ${SOURCES[2]} --output ${TARGETS[0]}")),
                     #"PreprocessFluency" : env.Builder(**env.ActionMaker("python",
                     #"PreprocessReddit" : env.Builder(**env.ActionMaker("python",
                     #"PreprocessTheater" : env.Builder(**env.ActionMaker("python",
                     #"PreprocessWomenWriters" : env.Builder(**env.ActionMaker("python",
                     "PreprocessLinguisticLid" : env.Builder(**env.ActionMaker("python",
                                                                        "src/preprocess_linguistic_lid.py",
                                                                        "--output ${TARGETS[0]} -t ${SOURCES[0]} -p ${SOURCES[1]}")),
                     "PreprocessSlavery" : env.Builder(**env.ActionMaker("python",
                                                                  "src/preprocess_slavery.py",
                                                                  "--output ${TARGETS[0]} ${SOURCES}")),


                     "PrepareDataset" : env.Builder(**env.ActionMaker("python",
                                                               "src/prepare_dataset.py",
                                                               "--spec_output ${TARGETS[0]} --dataset_output ${TARGETS[1]} --data_input ${SOURCES[0]} --schema_input ${SOURCES[1]}",
                                                               other_deps=["src/data.py"])),
                     "SplitRandom" : env.Builder(**env.ActionMaker("python",
                                                            "src/split_random.py",
                                                            "--input ${SOURCES[0]} --proportions ${PROPORTIONS} --outputs ${TARGETS} --random_seed ${RANDOM_SEED}",
                                                            other_deps=["src/data.py"])),
                     "SplitLinguisticLid" : env.Builder(**env.ActionMaker("python",
                                                                   "src/split_linguistic_lid.py",
                                                                   "--input ${SOURCES[0]} --proportions ${PROPORTIONS} --outputs ${TARGETS} --random_seed ${RANDOM_SEED}",
                                                                   other_deps=["src/data.py"])),
                     "MetadataSplit" : env.Builder(**env.ActionMaker("python",
                                                              "src/metadata_split.py",
                                                              "--input ${SOURCES[0]} --entity ${ENTITY} --field ${FIELD} --value ${VALUE} --outputs ${TARGETS}",
                                                                     other_deps=["src/data.py"])),
                     "TrainModel" : env.Builder(**env.ActionMaker("python",
                                                           "src/train_model.py",
                                                           "--data ${SOURCES[0]} --train ${SOURCES[1]} --dev ${SOURCES[2]} --model_output ${TARGETS[0]} --trace_output ${TARGETS[1]} ${'--gpu' if USE_GPU else ''} ${'--autoencoder_shapes ' + ' '.join(map(str, AUTOENCODER_SHAPES)) if AUTOENCODER_SHAPES != None else ''} ${'--mask ' + ' '.join(MASK) if MASK else ''} --log_level ${LOG_LEVEL} ${'--autoencoder' if AUTOENCODER else ''} --random_restarts ${RANDOM_RESTARTS}",
                                                           other_deps=["src/models.py", "src/data.py", "src/utils.py"],
                                                           other_args=["DEPTH", "MAX_EPOCHS", "LEARNING_RATE", "RANDOM_SEED", "PATIENCE", "MOMENTUM", "BATCH_SIZE",
                                                                       "EMBEDDING_SIZE", "HIDDEN_SIZE", "FIELD_DROPOUT", "HIDDEN_DROPOUT", "EARLY_STOP"],
                                                           )),# GRID_QUEUE=model_queue, GRID_RESOURCES=model_resources),
                     "ApplyModel" : env.Builder(**env.ActionMaker("python",
                                                           "src/apply_model.py",
                                                            "--model ${SOURCES[0]} --data ${SOURCES[1]} ${'--split ' + SOURCES[2].rstr() if len(SOURCES) == 3 else ''} --output ${TARGETS[0]} ${'--gpu' if USE_GPU else ''}", # --masked ${MASKED}",
                                                           other_args=["BATCH_SIZE"],
                                                           other_deps=["src/data.py", "src/models.py"],
                                                           )), #GRID_QUEUE=model_queue, GRID_RESOURCES=model_resources),
                     
                     "Evaluate" : env.Builder(**env.ActionMaker("python",
                                                         "src/evaluate.py",
                                                         "--model ${SOURCES[0]} --data ${SOURCES[1]} --test ${SOURCES[2]} --output ${TARGETS[0]}",
                                                         other_deps=["src/data.py", "src/models.py"])),
                     "EvaluateFields" : env.Builder(**env.ActionMaker("python",
                                                                      "src/evaluate_fields.py",
                                                                      "-i ${SOURCES[0]} -s ${SOURCES[1]} -o ${TARGETS[0]}",
                     )),

                     "Cluster" : env.Builder(**env.ActionMaker("python",
                                                        "src/cluster.py",
                                                        "--input ${SOURCES[0]} --spec ${SOURCES[1]} --output ${TARGETS[0]} --reduction ${CLUSTER_REDUCTION}")),
                     "InspectClusters" : env.Builder(**env.ActionMaker("python",
                                                                "src/inspect_clusters.py",
                                                                "--input ${SOURCES[0]} --output ${TARGETS[0]}")),
                     #"SaveConfig" : env.Builder(action=save_config),
                     "CollateResults" : env.Builder(**env.ActionMaker("python",
                                                               "src/collate_results.py",
                                                               "${SOURCES} --output ${TARGETS[0]}")),
                 },
           tools=["default"],
)


# function for width-aware printing of commands
def print_cmd_line(s, target, source, env):
    if len(s) > int(env["OUTPUT_WIDTH"]):
        print(s[:int(float(env["OUTPUT_WIDTH"]) / 2) - 2] + "..." + s[-int(float(env["OUTPUT_WIDTH"]) / 2) + 1:])
    else:
        print(s)


# and the command-printing function
env['PRINT_CMD_LINE_FUNC'] = print_cmd_line


# and how we decide if a dependency is out of date
env.Decider("timestamp-newer")


def run_experiment(env, experiment_config, **args):
    data = sum([env.Glob(p) for p in experiment_config.get("DATA_FILES", [])], [])
    schema = experiment_config.get("SCHEMA", None)
    title = experiment_name.replace("_", " ").title().replace(" ", "")
    if hasattr(env, "Preprocess{}".format(title)):
        data = getattr(env, "Preprocess{}".format(title))("work/${EXPERIMENT_NAME}/data.json.gz",
                                                          data, **args)

    # prepare the final spec and dataset
    spec, dataset = env.PrepareDataset(["work/${EXPERIMENT_NAME}/spec.pkl.gz", "work/${EXPERIMENT_NAME}/dataset.pkl.gz"],
                                       [data] + ([] if schema == None else [schema]),
                                       **args)

    split_names = [n for n, _ in experiment_config.get("SPLIT_PROPORTIONS", env["DEFAULT_SPLIT_PROPORTIONS"])]
    split_props = [p for _, p in experiment_config.get("SPLIT_PROPORTIONS", env["DEFAULT_SPLIT_PROPORTIONS"])]

    if hasattr(env, "Split{}".format(title)):
        train, dev, test = getattr(env, "Split{}".format(title))(["work/${{EXPERIMENT_NAME}}/{0}.pkl.gz".format(n) for n in split_names], 
                                                                 dataset,
                                                                 **args, RANDOM_SEED=0, PROPORTIONS=split_props)
    else:
        train, dev, test = env.SplitRandom(["work/${{EXPERIMENT_NAME}}/{0}.pkl.gz".format(n) for n in split_names], 
                                           dataset,
                                           **experiment_config,
                                           **args, RANDOM_SEED=0, PROPORTIONS=split_props)
    


    # expand training configurations
    train_configs = [[]]
    for arg_name, values in experiment_config.get("TRAIN_CONFIG", env["DEFAULT_TRAIN_CONFIG"]).items():
        train_configs = sum([[config + [(arg_name.upper(), v)] for config in train_configs] for v in values], [])
    train_configs = [dict(config) for config in train_configs]

    # expand apply configurations
    apply_configs = [[]]
    for arg_name, values in experiment_config.get("APPLY_CONFIG", env["DEFAULT_APPLY_CONFIG"]).items():
        apply_configs = sum([[config + [(arg_name.upper(), v)] for config in apply_configs] for v in values], [])
    apply_configs = [dict(config) for config in apply_configs]


    results = []
    for config in train_configs:
        args["TRAIN_CONFIG_ID"] = md5(str(sorted(list(config.items()))).encode()).hexdigest()
        model, trace = env.TrainModel(["work/${EXPERIMENT_NAME}/model_${TRAIN_CONFIG_ID}.pkl.gz", 
                                       "work/${EXPERIMENT_NAME}/trace_${TRAIN_CONFIG_ID}.pkl.gz"],
                                      [dataset, train, dev],
                                      **args,
                                      **config)

        for apply_config in apply_configs:
            config.update(apply_config)
            args["APPLY_CONFIG_ID"] = md5(str(sorted(list(config.items()))).encode()).hexdigest()
            #conf = env.SaveConfig("work/${EXPERIMENT_NAME}/${FOLD}/applyconfig_${APPLY_CONFIG_ID}.txt.gz", 
            #                      env.Value([(k, v) for k, v in sorted(config.items())]),
            #                      **args,
            #                      **config)
            
            output = env.ApplyModel("work/${EXPERIMENT_NAME}/${FOLD}/output_${APPLY_CONFIG_ID}.pkl.gz", 
                                    [model, dataset, test],
                                    **args,
                                    **config)

            scores = env.EvaluateFields("work/${EXPERIMENT_NAME}/${FOLD}/score_${APPLY_CONFIG_ID}.csv",
                                        [output, spec],
                                        **args, **config)
            results.append(scores)
    #return env.CollateResults("work/${EXPERIMENT_NAME}/scores.txt.gz", results, **args, **config)
    return None


env.AddMethod(run_experiment, "RunExperiment")


#
# Run all experiments
#

for experiment_name, experiment_config in env["EXPERIMENTS"].items():
    env.RunExperiment(experiment_config, EXPERIMENT_NAME=experiment_name)
