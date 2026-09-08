# FNR control in multi-label classification

## Before running the experiments
Make sure to create a `results` folder, containing two folders named `experiment_1_a` and `experiment_1_b`.

The parameters for each experiment is contained in the `params` folder.

## How to run the experiments
The first experiment involving 500 + 1 data-points corresponds to the parameter file `params\experiment_1_a.json`.

In order to run the experiment,
run the jupyter notebook called `experiment.ipynb`.

It may take some time to finish. As a reference, it took about 4 hours and 30 minutes for it to finish on my personal computer which is an ASUS Vivobook 14 X1407QA_X1407QA (the specs can be accessed on this link https://www.asus.com/laptops/for-home/vivobook/asus-vivobook-14-x1407/techspec/).

The second experiment involving 1000 + 1 data-points corresponds to the parameter file `params\experiment_1_b.json`.

In order to run the experiment,
first, comment-out in the 7th cell in `experiment.ipynb` the lines corresponding to the previous experiment, then uncomment the lines corresponding to this experiment.
Finally, run `experiment.ipynb`.

It may take some time to finish. As a reference, it took about 6 hours on my personal computer.


## How to display the results
Run the jupyter notebook called `display_experiment.ipynb`. Do make sure that the first step was done.

To display the results of `experiment_1_b` comment and uncomment the relevant lines.

## If there is any issue

Do contact me at `davidson-lova.razafindrakoto@proton.me`.