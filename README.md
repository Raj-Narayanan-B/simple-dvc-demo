Create environment:
```bash
conda create -n wineq python 3.7 -y
```

Activate Environment:
```bash
conda activate wineq
```

Create requirements.txt file and install it:
```bash
pip install -r requirements.txt
```

Dataset from:
https://drive.google.com/drive/folders/18zqQiCJVgF7uzXgfbIJ-04zgz1ItNfF5?usp=sharing

git init

dvc init

dvc add data_given/winequality.csv

git add .

git commit -m "first commit"

One-Liner to update Readme.md files
```bash
git add . && git commit -m "README updated"
git remote add origin https://github.com/Raj-Narayanan-B/simple-dvc-demo.git
git branch -M main
git push origin main
```

TOX Command

```bash
tox
```

for rebuilding
```bash
tox -r
```

pytest command
```bash
pytest -v
```

setup commands
```bash
pip install -e .
```

Build your own package commands
```bash
python setup.py sdist bdist_wheel
```

MLFLOW server command:
```bash
mlflow server \
    --backend-store-uri sqlite:///mlflow_new.db \
    --default-artifact-root ./artifacts \
    --host 127.0.0.1 -p 1234
    
```
mlflow server \
     --backend-store-uri sqlite:///mlflow.db \
     --default-artifact-root ./mlruns \
     # --no-serve-artifacts \
     --host 127.0.0.1 -p 1234