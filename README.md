# Model selection and conformal risk control

## How to start
You can clone or download this repo.
Then, you can run the following command,
```sh
$ python -m venv .venv
```
to create a virtual environment
and then activate said environment
using
```sh
$ .venv/Scripts/activate
```
if you are on windows.
(Its different if you are on mac or linux so do check the relevant documentation.)

After your virtual environment is activated,
you can run the following command,
```sh
$ pip install -r requirements.txt
```
to install all the required packages.


## Structure
After all the required packages are installed, you can go to the folder `experiments\`
if you wish to reproduce the results in the paper.
Wherein lies a dedicated `README.md` file.

`src\model_selection\` contains the implementation of the three methods, namely, Split, Oracle and Upper.

`src\conformal_risk_control\` contains an implementation of FNR control.

`src\experiments\` contains the experiments carried out in the paper.

## If there is any issue

Do contact me at `davidson-lova.razafindrakoto@proton.me`.