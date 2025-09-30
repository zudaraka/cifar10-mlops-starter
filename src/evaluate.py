import torch, mlflow
from src.data import get_dataloaders
from src.model import SmallCNN

def evaluate(model_path, data_dir, batch_size=128):
    mlflow.set_tracking_uri("file:./mlflow_tracking")
    with mlflow.start_run(run_name="eval"):
        _, _, test_dl = get_dataloaders(data_dir, batch_size)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        m = SmallCNN(); m.load_state_dict(torch.load(model_path, map_location=device))
        m.to(device); m.eval()
        correct=0; total=0
        with torch.no_grad():
            for xb,yb in test_dl:
                xb, yb = xb.to(device), yb.to(device)
                pred = m(xb).argmax(1)
                correct += (pred==yb).sum().item(); total += yb.size(0)
        test_acc = correct/total
        mlflow.log_metric("test_acc", test_acc)
        print(f"TEST_ACC={test_acc:.4f}")

if __name__ == "__main__":
    evaluate("artifacts/model.pt","data",128)
