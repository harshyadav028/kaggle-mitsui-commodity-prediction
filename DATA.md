# DATA & How to obtain it

1. Create a Kaggle API token: Account -> Create API Token -> download kaggle.json
2. In Colab:
   - Upload kaggle.json and run:
     mkdir -p ~/.kaggle
     cp kaggle.json ~/.kaggle/
     chmod 600 ~/.kaggle/kaggle.json
     kaggle competitions download -c mitsui-commodity-prediction-challenge -p /content/data
     unzip -q /content/data/*.zip -d /content/data
3. DO NOT commit kaggle.json or raw data to git.

# Files
train.csv: 
