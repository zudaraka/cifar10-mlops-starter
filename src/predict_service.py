from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import torch, io
from PIL import Image
from torchvision import transforms
from src.model import SmallCNN

app = FastAPI()
device = "cpu"
model = SmallCNN()
try:
    model.load_state_dict(torch.load("artifacts/model.pt", map_location=device))
    model.eval()
except Exception:
    # model artifact not present yet
    pass

pre = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914,0.4822,0.4465),(0.247,0.243,0.261))
])

CLASSES = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img = Image.open(io.BytesIO(await file.read())).convert("RGB")
    x = pre(img).unsqueeze(0)
    with torch.no_grad():
        logits = model(x)
        p = logits.softmax(1).squeeze().tolist()
        idx = int(logits.argmax(1).item())
    return JSONResponse({"class": CLASSES[idx], "probs": p})
