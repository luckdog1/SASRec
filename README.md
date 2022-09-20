{0: 1511, 1: 1125, 2: 1939, 3: 159, 4: 1941}
{0: 1378, 1: 1366, 2: 737, 3: 2768}
{0: 267, 1: 378, 2: 2572, 3: 307, 4: 588, 5: 20, 6: 50, 7: 780, 8: 44, 9: 226}
{0: 206, 1: 130, 2: 216, 3: 527, 4: 1110, 5: 1342, 6: 36, 7: 2066, 8: 317, 9: 55}
{0: 176, 1: 759, 2: 107, 3: 48, 4: 89, 5: 726, 6: 770, 7: 496, 8: 1510, 9: 177}
{0: 876, 2: 87, 3: 518, 4: 1295, 5: 52, 6: 12, 7: 737, 8: 2233, 9: 2996}
{0: 233, 1: 688, 2: 39, 3: 549, 4: 158, 5: 318, 6: 593, 7: 680, 8: 365, 9: 1812}
{0: 476, 1: 175, 3: 2, 5: 690, 6: 949, 7: 433, 8: 992, 9: 633}
{0: 405, 1: 279, 2: 253, 3: 63, 4: 248, 5: 1397, 6: 1642, 7: 1073, 8: 389, 9: 50}
{0: 395, 1: 1842, 2: 8, 3: 1190, 4: 413, 5: 876, 6: 1866, 8: 1}
[[]
[]
[]
[]]

# SASRec: Self-Attentive Sequential Recommendation

This is our TensorFlow implementation for the paper:

[Wang-Cheng Kang](http://kwc-oliver.com), [Julian McAuley](http://cseweb.ucsd.edu/~jmcauley/) (2018). *[Self-Attentive Sequential Recommendation.](https://cseweb.ucsd.edu/~jmcauley/pdfs/icdm18.pdf)* In Proceedings of IEEE International Conference on Data Mining (ICDM'18)

Please cite our paper if you use the code or datasets.

The code is tested under a Linux desktop (w/ GTX 1080 Ti GPU) with TensorFlow 1.12 and Python 2.

Refer to *[here](https://github.com/pmixer/SASRec.pytorch)* for PyTorch implementation (thanks to pmixer).

## Datasets

The preprocessed datasets are included in the repo (`e.g. data/Video.txt`), where each line contains an `user id` and 
`item id` (starting from 1) meaning an interaction (sorted by timestamp).

The data pre-processing script is also included. For example, you could download Amazon review data from *[here.](http://jmcauley.ucsd.edu/data/amazon/index.html)*, and run the script to produce the `txt` format data.

### Steam Dataset

We crawled reviews and game information from Steam. The dataset contains 7,793,069 reviews, 2,567,538 users, and 32,135 games. In addition to the review text, the data also includes the users' play hours in each review.     

* Download: [reviews (1.3G)](http://cseweb.ucsd.edu/~wckang/steam_reviews.json.gz), [game info (2.7M)](http://cseweb.ucsd.edu/~wckang/steam_games.json.gz)
* Example (game info):
```json
{
    "app_name": "Portal 2", 
    "developer": "Valve", 
    "early_access": false, 
    "genres": ["Action", "Adventure"], 
    "id": "620", 
    "metascore": 95, 
    "price": 19.99, 
    "publisher": "Valve", 
    "release_date": "2011-04-18", 
    "reviews_url": "http://steamcommunity.com/app/620/reviews/?browsefilter=mostrecent&p=1", 
    "sentiment": "Overwhelmingly Positive", 
    "specs": ["Single-player", "Co-op", "Steam Achievements", "Full controller support", "Steam Trading Cards", "Captions available", "Steam Workshop", "Steam Cloud", "Stats", "Includes level editor", "Commentary available"], 
    "tags": ["Puzzle", "Co-op", "First-Person", "Sci-fi", "Comedy", "Singleplayer", "Adventure", "Online Co-Op", "Funny", "Science", "Female Protagonist", "Action", "Story Rich", "Multiplayer", "Atmospheric", "Local Co-Op", "FPS", "Strategy", "Space", "Platformer"], 
    "title": "Portal 2", 
    "url": "http://store.steampowered.com/app/620/Portal_2/"
}
```
  

## Model Training

To train our model on `Video` (with default hyper-parameters): 

```
python main.py --dataset=Video --train_dir=default 
```

or on `ml-1m`:

```
python main.py --dataset=ml-1m --train_dir=default --maxlen=200 --dropout_rate=0.2 
``` 

## Misc

The implemention of self attention is modified based on *[this](https://github.com/Kyubyong/transformer)*

The convergence curve on `ml-1m`, compared with CNN/RNN based approaches:  

<img src="curve.png" width="400">
