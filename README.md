# CIFAR-10 MLOps Coursework Starter

[![CI](https://github.com/zudaraka/cifar10-mlops-starter/actions/workflows/ci.yml/badge.svg)](../../actions)

Minimal, rubric-friendly project for **CIFAR-10** (10 classes) demonstrating:
- **MLflow** experiment tracking
- **Git + DVC** for code/data/artifact versioning
- **GitHub Actions** (runs `pytest`)
- **FastAPI** inference service (+ simple monitoring)
- **Docker** container

> **Test accuracy (from `python -m src.evaluate`)**: **0.6198**

---

## 1) Problem, Assumptions, Limits, Data

- **Problem:** Classify 32×32 RGB images into 10 CIFAR-10 classes.
- **Assumptions:** Single-label classification; CPU training OK.
- **Limitations:** Small baseline CNN + few epochs → modest accuracy (by design).
- **Dataset:** CIFAR-10 (60k images: 50k train, 10k test) with standard mean/std normalization.

---

## 2) Quickstart

```bash
# create & activate venv (optional)
python3 -m venv .env && source .env/bin/activate
pip install --upgrade pip

# install deps (PyTorch CPU wheels are safest on Mac)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

python -m src.train        # logs to ./mlflow_tracking and writes ./artifacts/model.pt
python -m src.evaluate     # prints TEST_ACC and logs to MLflow

mlflow ui --backend-store-uri file:./mlflow_tracking -p 5000
# open http://127.0.0.1:5000

uvicorn src.predict_service:app --host 0.0.0.0 --port 8000
# docs: http://127.0.0.1:8000/docs

python - <<'PY'
from torchvision import datasets, transforms
from PIL import Image
ds = datasets.CIFAR10(root="data", train=False, download=True,
                      transform=transforms.ToTensor())
img,_ = ds[0]
Image.fromarray((img.permute(1,2,0).numpy()*255).astype('uint8')).save("sample.jpg")
print("Saved sample.jpg")
PY

curl -s -F "file=@sample.jpg" http://127.0.0.1:8000/predict | python -m json.tool

# track dataset (already done once in repo)
dvc add data/

# record current model artifact produced by pipeline
dvc commit artifacts/model.pt

# (optional) configure & push to a local remote
mkdir -p ../dvc_store
dvc remote add -d local_store ../dvc_store
dvc push

dvc repro

docker build -t cifar-service .
docker run --rm -p 8000:8000 cifar-service
# then POST to http://127.0.0.1:8000/predict as above

tail -f logs/service.log

.
├── src/
│   ├── data.py              # dataloaders + transforms
│   ├── model.py             # SmallCNN
│   ├── train.py             # MLflow training loop
│   ├── evaluate.py          # test-set evaluation + MLflow
│   └── predict_service.py   # FastAPI service (+ basic monitoring)
├── tests/
│   └── test_model.py        # tiny sanity test (CI)
├── artifacts/               # model.pt (pipeline output)
├── data/                    # CIFAR-10 (downloaded by torchvision)
├── mlflow_tracking/         # local MLflow store
├── dvc.yaml / dvc.lock / data.dvc
├── Dockerfile
├── report.ipynb             # coursework report
├── requirements.txt
└── .github/workflows/ci.yml
