# BEDS-Bench: *B*ehavior of *E*HR-models under *D*istributional *S*hift - A *Bench*mark

**This is not an officially supported Google product**

This is a Benchmarking Tool meant to be used **ONLY FOR RESEARCH PURPOSES**.

## Introduction

`BEDS-Bench` is a tool to benchmark the behavior of Machine Learning (ML) models on Electronic Health Record (EHR) data, under Distributional Shift. This means situations when the inputs to the model start differing significantly from the data the model was trained on. To this end, `BEDS-Bench` takes two deidentified open access EHR datasets (MIMIC-III and PICDB), constructs several intentionally dissimilar subsets, and analyses the performance of models which are trained on one subset and tested on other disjoint subsets. The benchmark uses three common clinical prediction tasks (In-Hospital Mortality, Length-of-Stay greater than 3-days, Length-of-stay greater than 7-days) to train and test the model performance in these settings.

As a user of `BEDS-Bench` you will need to implement a `scikit-learn` API compatible model, and fit it to training and validation data inside a method name `fit` in your module, and include the name of your module in `config.json`. There are several simple models already implemented under the `models/` directory that may be used as a starting template for your fancy method.

## Requirements

The data required for `BEDS-Bench` to run are available on PhysioNet. If you have not already, you will need to perform the following steps:

* Sign up for an account on PhysioNet (https://physionet.org/register/), and log in.

* Complete the Training Course and obtain certification for it from CITI (https://mimic.physionet.org/gettingstarted/access/).

* Request access to MIMIC-III at https://physionet.org/content/mimiciii/1.4/ by clicking on "**Credentialed Access**", and submitting the certification obtained in the previous step. (Approval Email will arrive typically in a few days)

* Request access to PICDB at https://physionet.org/content/picdb/1.1.0/ by clicking on "**Credentialed Access**", and submitting the certification obtained in the previous step. (Approval Email will arrive typically in a few days)

* After you have received both the approval emails, you may proceed to running the benchmark (the datasets will be downloaded as part of the benchmark execution and need not be downloaded in advance).


## Run BEDS-Bench

Before running the benchmark itself, there are a couple things to consider. Running the benchmark without any changes or configuration will reproduce all the results. However you may choose to do one or more of the following **OPTIONAL** customizations:

* Customize the `work-dir` in `config.json` to a different directory other than `/tmp/beds_bench`

* Include your own learning algorithm to be evaluated by the benchmark under `models/` directory. See existing models in there to get started. Be sure to add your new algorithm in `config.json` to be included in the runs.

* Exclude some models in `config.json` (perhaps those that take a long time to run, like GP, or MondrianForest)

* Customize the data slices of interest, or create new data slices by editing `config.json`.



If you have the files `mimic-iii-clinical-database-1.4.zip` and/or `paediatric-intensive-care-database-1.0.0.zip` downloaded already, you may place those zip files inside the `work-dir` directory (defaults to `/tmp/beds_bench`) to avoid repeated downloads. If not, make note of your PhysioNet username and password as they will be required to download these files as part of the benchmark.


```sh
sh:~/beds-bench$ ./runme.sh
...
```

If you had not placed the EHR data zip files in the `work-dir` directory, you will be promoted for your PhysioNet username and password to finish the download.

If you need to restart the run (in case there were errors of failures), you can directly call the python code from where you would like to resume (see `function main()` inside `runme.sh`). You may also comment out a few lines in `runme.sh` and run it again.

After a full successful run, all the results will be placed inside a `results` subdirectory in `work-dir`. By default this is `/tmp/beds_bench/results`. You will find one CSV file per model, and an aggregated `results.csv` and `latex/results.tex` that includes results from all models. You may use tools like `pdflatex` to compile the `latex/results.tex` file to view the results in an easier manner.
