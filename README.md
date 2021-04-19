# squatting-slavs-strike-back


How to replicate:

1. Execute

```sh
git@github.com:tadejsv/squatting-slavs-strike-back.git && git checkout best
```
2. Create a folder called `train_data/`, and fill it with the training data csv files.

3. Run the notebook `dev_tadej5.ipynb`. You can do this in the Dockerfile-based remote container (VSCode). At the end you should have a file `models/tadej_model.cbm` - this is our model.
4. To generate our submission, execute (update git to latest stable verion first)

```sh
git archive --add-file=models/tadej_model.cbm -o submit.zip HEAD
```

5. You should have a file `submit.zip`. This is our submission.