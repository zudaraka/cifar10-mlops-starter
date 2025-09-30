import os, mlflow, torch, torch.nn as nn, torch.optim as optim, yaml
from src.data import get_dataloaders
from src.model import SmallCNN

def train(params):
    mlflow.set_tracking_uri("file:./mlflow_tracking")
    mlflow.set_experiment("cifar10_cnn")
    with mlflow.start_run():
        mlflow.log_params(params)
        train_dl, val_dl, _ = get_dataloaders(params["data_dir"], params["batch_size"])
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = SmallCNN().to(device)
        opt = optim.Adam(model.parameters(), lr=params["lr"])
        criterion = nn.CrossEntropyLoss()

        best = 0.0
        for epoch in range(params["epochs"]):
            model.train()
            for xb,yb in train_dl:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward(); opt.step()
            # val
            model.eval(); correct=0; total=0
            with torch.no_grad():
                for xb,yb in val_dl:
                    xb, yb = xb.to(device), yb.to(device)
                    pred = model(xb).argmax(1)
                    correct += (pred==yb).sum().item(); total += yb.size(0)
            acc = correct/total
            mlflow.log_metric("val_acc", acc, step=epoch)
            if acc > best:
                best = acc
                os.makedirs("artifacts", exist_ok=True)
                torch.save(model.state_dict(), "artifacts/model.pt")
                mlflow.log_artifact("artifacts/model.pt")
        mlflow.log_metric("best_val_acc", best)

if __name__ == "__main__":
    params = yaml.safe_load(open("params.yaml"))
    train(params)
