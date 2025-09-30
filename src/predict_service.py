from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import torch, io, os, json, time, logging
from PIL import Image
from torchvision import transforms
from src.model import SmallCNN

app = FastAPI(title="CIFAR10 Service", version="1.0")

# --- Monitoring / Logging setup ---
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/service.log",
    level=logging.INFO,
    format="%(asctime)s %(message)s"
)

# CIFAR-10 training distribution (used for simple drift flags)
TRAIN_MEAN = (0.4914, 0.4822, 0.4465)
TRAIN_STD  = (0.2470, 0.2430, 0.2610)
DRIFT_THRESH = 0.15  # 15% relative change

def channel_stats(t: torch.Tensor):
    """Return per-channel mean/std after inverting normalization back to ~[0,1]."""
    mean = torch.tensor(TRAIN_MEAN).view(1,3,1,1)
    std  = torch.tensor(TRAIN_STD).view(1,3,1,1)
    x = t * std + mean
    m = x.mean(dim=(0,2,3)).tolist()
    s = x.std(dim=(0,2,3), unbiased=False).tolist()
    return m, s

def drift_flags(m, s):
    flags = []
    for i, (mi, si, mb, sb) in enumerate(zip(m, s, TRAIN_MEAN, TRAIN_STD)):
        rel_mean = abs(mi - mb) / max(1e-8, abs(mb))
        rel_std  = abs(si - sb) / max(1e-8, abs(sb))
        if rel_mean > DRIFT_THRESH: flags.append(f"mean_ch{i}")
        if rel_std  > DRIFT_THRESH: flags.append(f"std_ch{i}")
    return flags

# --- Model load ---
device = "cpu"
model = SmallCNN()
_model_loaded = False
try:
    state = torch.load("artifacts/model.pt", map_location=device)
    model.load_state_dict(state)
    model.eval()
    _model_loaded = True
except Exception:
    # model artifact not present yet; /predict will handle gracefully
    pass

# --- Preprocess ---
pre = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(TRAIN_MEAN, TRAIN_STD)
])

CLASSES = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model_loaded, "classes": len(CLASSES)}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not _model_loaded:
        return JSONResponse({"error": "model not loaded yet; train first"}, status_code=503)

    img = Image.open(io.BytesIO(await file.read())).convert("RGB")
    x = pre(img).unsqueeze(0)  # [1,3,32,32]

    # monitoring: input stats + drift flags
    m, s = channel_stats(x)
    flags = drift_flags(m, s)

    with torch.no_grad():
        logits = model(x)
        probs = logits.softmax(1).squeeze()
        idx = int(torch.argmax(probs).item())
        conf = float(probs[idx].item())
        p_list = probs.tolist()

    # log one line per request
    event = {
        "ts": time.time(),
        "pred_class": CLASSES[idx],
        "confidence": conf,
        "stats": {"mean": m, "std": s},
        "flags": flags
    }
    logging.info(json.dumps(event))

    return JSONResponse({
        "class": CLASSES[idx],
        "confidence": conf,
        "probs": p_list,
        "flags": flags
    })