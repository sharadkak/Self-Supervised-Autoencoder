# Experiment Skeletons
Templates and folder structure for your future experimental projects.
This repo provides a simple and generic structure to organize coding projects that involve experiments.

### Structure:
Ideally, the root folder contains only a README.md file and the following folders:
- `./data/`: contains all external data used by your project to run experiments. Usually, this means all folders with
 datasets or pre-trained models. As a general guideline, try to use symbolic links to the actual folder containing the
 data to avoid duplication of large files. For example, to use ImageNet, as stored in the NFS under `/ds/imagenet/`,
 link it with: `cd data && ln -s /ds/images/imagenet/ imagenet`. Then your code can load anything under `data/imagenet/`
 
- `./doc/`: anything that relates to documentation of the project itself. Licences, instructions, more detailed READMEs
 should go here. A special kind of documentation is any paper or report that is written about this project. We suggest
 to use a subfolder `./doc/tex/` if it's going to be written in LaTeX.

- `./exp/`: any partial results that are computed for this project goes here. We suggest using
 [Sacred](https://sacred.readthedocs.io) to manage every experiment you run for this project. Since this tends to be
 the folder where large log files or entire models are saved, we also suggest to link the `./exp/` folder to 
 `/netscratch/$USER/project_name/exp/`. Assuming you want to have the sources on the NFS home, this helps keeping the
 limited size of `/nethome/` free and available to all.

- `./src/`: single place where all the sources produced for this project are stored. Use symbolic links (or copy
 extra sources) to add code that makes part of this project. For machine learning projects, there is usually different
 models or variants thereof, that are tested and hence, we suggest using a subfolder `./src/models/` for that.

- `./res/`: any final results you obtained that you want to keep track of in the repo. These are usually JSON files with
 baselines or final results that are going into the paper, raw values for plotting figures, tables, etc. Keep in mind
 that large binary files are not something you want to keep here (nor GIT or the admin will like you). This folder is
 the most optional of all but it is useful once your set of experiments grow too much and you need to select the best
 ones that end up in the final report for example.
 
- `README.md`: Basically the README of your project (you basically need to overwrite this file).


### Source templates:
We have added a few templates in `./src/ `to get you started both with sacred and your project in general. Here's a
brief description of their functionality. For more details, please refer to the code itself.
 
- `pythonargs.py`: simple python script with an argument handler. Useful for code that is not considered an experiment.
For example, code that prepares a dataset or plots results that were generated independently.

- `pythonargslog.py`: similar to pythonargs.py but with logging instead of simple printing. Also not meant for
experiments but for everything else.

- `pythonexp.py`: basic template to hold an experiment. Uses the bare minimum of sacred to keep it simple. This should
log pretty much everything you need.

- `pythonparamexp.py`: template that let you run a previously defined sacred experiment several times with different
sets of parameters (useful for meta-parameter searches). It shows an example for random samplin

- `env.sh`: set useful environment variables, i.e., to avoid spawning too many threads on large machines.

- `train.sh`: example script to call userdocker on the DGX.

### TODOs:
- add useful `.gitignore` files where appropriate (e.g., excluding anything in ./data/ and ./exp/, *.pyc files, etc.).  You can use [gitignore.io](https://www.gitignore.io/) for generating the required information depending on IDEs, Programming Languages or Operating Systems.
