# Assignment 2: Language Modeling with a Feed Forward Neural Network (FFNN)

## Steps to Run on W135 Systems:

* W135 systems have disk space restrictions within the `/home/grads/` folder. Instead using the `/scratch/` space gives us necessary disk space for required installations.

### Steps to set up the environment:

```commandline
scp -r ~/CSE_582/HW2/Homework2 <username>@e5-cse-135-<specific-server>.cse.psu.edu:/scratch/<username>

cd /scratch/<username>/

mkdir -p /scratch/<username>/miniconda3

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /scratch/<username>/miniconda3/miniconda.sh

bash /scratch/<username>/miniconda3/miniconda.sh -b -u -p /scratch/<username>/miniconda3

rm -rf /scratch/<username>/miniconda3/miniconda.sh

/scratch/<username>/miniconda3/bin/conda init zsh

source ~/.zshrc

conda install nltk

cd Homework2

conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

mkdir /scratch/<username>/miniconda3/nltk_data

NLTK_DATA=/scratch/<username>/miniconda3/nltk_data python -m nltk.downloader all
```

### CLI Commands:

#### CLI Commands for Movies:
```commandline
python NLM.py --train_file ./data/movies/movies_train.csv

python run.py --datarep GLOVE --lr 0.001 --embed_size 50 --hidden_size 5 --embed_file ./data/random_embedding.txt --train_file ./data/movies/movies_train.csv --test_file ./data/movies/movies_test.csv

python run.py --datarep GLOVE --lr 0.001 --embed_size 50 --hidden_size 5 --embed_file ./glove/glove.6B.50d.txt --train_file ./data/movies/movies_train.csv --test_file ./data/movies/movies_test.csv

python run.py --datarep GLOVE --lr 0.001 --embed_size 50 --hidden_size 5 --embed_file ./data/movie_embeddings.txt --train_file ./data/movies/movies_train.csv --test_file ./data/movies/movies_test.csv
```

#### CLI Commands for Jewelry:
```commandline
python NLM.py --train_file ./data/jewelry/jewelry_train.csv

python run.py --datarep GLOVE --lr 0.001 --embed_size 50 --hidden_size 5 --embed_file ./data/random_embedding.txt --train_file ./data/jewelry/jewelry_train.csv --test_file ./data/jewelry/jewelry_test.csv

python run.py --datarep GLOVE --lr 0.001 --embed_size 50 --hidden_size 5 --embed_file ./glove/glove.6B.50d.txt --train_file ./data/jewelry/jewelry_train.csv --test_file ./data/jewelry/jewelry_test.csv

python run.py --datarep GLOVE --lr 0.001 --embed_size 50 --hidden_size 5 --embed_file ./data/jewelry_embeddings.txt --train_file ./data/jewelry/jewelry_train.csv --test_file ./data/jewelry/jewelry_test.csv
```
