# squatting-slavs-strike-back
🏆🌟🎖️

## How to start

Best idea is to also use official dockerfile in development, to avoid problems later on.

> Warning: the official Dockerfile was not made by humans, but by another species of primates. This species has not yet discovered that installing the same dependencies first with `conda install` and later from `requirements.txt` not only makes 0 sense, but also wastes a lot of time. Furthermore, the species has not yet developed understanding of terms such as "computer vision" and "tabular data", and believes that `torchvision` package will be useful for this competition.

To do this, use **VSCode**, and have **docker** and **docker-compose** installed. Then, click <kbd>Ctrl</kbd> + <kbd>Shift</kbd> + <kbd>P</kbd>, and select `Remote Containers: Open Folder in Container`. This will automatically start building the docker image (warning - will take time!!), and after that the VSCode will reopen *inside* the container, with the local folder mounted inside. This enables you to work in the environment specified in the Dockerfile, but the changes will all be saved locally.

After you are inside, also install `git` and `openssh-server` using
```
apt-get update && apt-get install git openssh-server
```

Finally, donwload extract training data to a folder called `train_data` (so we all have the same file structure).

## Splitting datasets into train/val/test

To ensure our results are comparable, we should all work with the same test/val/train split.

If I am not mistaken, it only matters that `funnel.csv` file is split, the rest is merged to it anyway, so as long as inner joins are used everything should be properly separated.

I have created a `train_val_test_split` function, which creates this split in a deterministic way. I propose 80/10/10 split. Here's how you can do it in your code

```python
import pandas as pd
from src.utils import train_val_test_split

funnel = pd.read_csv('train_data/funnel.csv')
train, val, test = train_val_test_split(funnel, (0.8, 0.1, 0.1))
```

## Calculate profit

There is a simple utility in `src.utils` to calculate profit for each person in the dataset. Use it like this

```python
import pandas as pd
from src.utils import calculate_profit

funnel = pd.read_csv('train_data/funnel.csv')
profit = calculate_profit(funnel) # this is a pd.Series
```

## Our results

Here we list our results, so we can stay updated with what scores we are getting

### Tadej

| name        | val_score  | test_score | submission score |
| ------------|:----------:|:----------:|:----------------:|
| below + simplify payments fts |  5998.36 | 5637.98 | 5598.47 |
| below + simplify balance + aum fts |  6021.99 | 5692.18 | 5637.70 |
| below + payment + mystery fts | 5961.78 | 5597.67 | 5617.60 |
| below + transaction fts | 5887.00 | 5442.85 |  |
| client + balance + aum feats with regressor + *log profit* + *threshold* | 5906.70 | 5516.17 | 5497.49 |
| client + balance + aum feats *with regressor* | 5792.52 | 5289.16 | 5381.18 |
| client + balance + aum feats  | 3607.06 | 2944.95 | 3101.47 |
| client + balance feats  | 3345.24 | 3074.36 | 2907.5 |
| client feats            | | 1730.64 | |
| balance feats           | | 1625.17 | |

### Marina

| name        | val_score  | test_score | submission score |
| ------------|:----------:|:----------:|:----------------:|
| ??  | ?? | ?? | ?? |

### Anastasia

| name        | val_score  | test_score | submission score |
| ------------|:----------:|:----------:|:----------------:|
| transaction + client + balance + aum feats *with regressor*   | 5432.08 | 3001.63 | 5476.3 |


## Preparing for submission

To prepare for submission, you need to do some steps

### Prepare `data` folder

In evaluation the data will be in the `data/` folder. Create this folder (don't worry, it is git ignored), and copy everything from `train_data/` into it. You can not write to the `data/` folder during evaluation.

### Save your model

Save your model inside the `models/` folder. I would also suggest to train it on full dataset first (not just the train split) - unless of course you were using some kind of early stopping or similar.

### Prepare your scripts

There is a `scripts/` folder, put your python scripts there. I recommend spliting it into two scripts:
1. A build script that will compile all the features - then save them in the picke format using `df.save_pickle`. A file `final_version.pickle` is already git ignored, you can use that.
2. A run script. This reads the file saved in the previous step and does the predictions. It should output a file `submission.csv` (in the root of this repository).

    Here note that now the model is in the `models/` folder, but at run time it will be (some quirk of git archive) in the root repository folder! So make sure to have some kind of `try/except` to make sure it is read from both locations, for example

    ```python
    try:
        model = CatBoostClassifier().load_model('models/tadej_model.cbm', 'cbm')
    except:
        model = CatBoostClassifier().load_model('tadej_model.cbm', 'cbm')
    ```

Once you have your scripts, test them. Make sure they are only reading from `data/` repository, and make sure to ignore all the label columns.

### Change the `.sh` scripts

All you need to do in them is to replace the name of the python script in `run.sh` and `build.sh`. Here we will have merge conflicts unfortunatelly 🙃

### Test your code

Ok, time to get real :) Run the following two commands

```
make build
```

and
```
make run
```

At the end you should get a `submissions.csv` file, inspect it.

### Zip the directory

Time to produce the final product for submission. We will be using git here (as we can leverage gitignore by default!), so make sure that everything you want to send **is commited**. We also need to make sure git is at the latest version, execute these commands (this is needed for `--add-file` later):

```sh
apt-get update
apt-get install software-properties-common
add-apt-repository ppa:git-core/ppa
apt update; apt install git
```

Ok, done! Now to produce the zip file, execute this command

```
git archive --add-file=models/your_model_name -o submit.zip HEAD
```

### Inspect zip file

Also, inspect what is inside this folder before you send it. You can unzip it easily with
```
unzip submit.zip -d check_zip
```

Now inspect the `check_zip` folder, to make sure everything is there (you should see your model at the root of the folder, instead of in the `models` folder).

That's it! Now you can send the submission :)