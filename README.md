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