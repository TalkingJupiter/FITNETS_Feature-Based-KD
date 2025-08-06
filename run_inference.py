import torch
import argparse
from dataset.cifar10_loader import get_cifar10_dataloaders
from teacher_model.teacher_cnn import get_teacher
from student_model.student_cnn import get_student
from monitor.monitor import PowerMonitor

def run_inference(model_type):
    device = torch.device("cuda", if torch.cuda.is_available() else "cpu")
    _, test_loader = get_cifar10_dataloaders()
    
    if model_type == "teacher":
        model = get_teacher().to(device)
    else:
        model = get_student().to(device)


    model.eval()
    correct = 0
    total = 0

    with PowerMonitor(outfile=f"logs/infer_{model_type}.csv") as monitor:
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                logits = model(inputs)
                pred = logits.argmax(dim=1)
                correct += (pred == labels).sum().item()
                total += labels.size(0)

    acc = coorect / total
    print(f"[{model_type}] Accuracy: {acc:.4f}")

if __name__ = "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["teacher", "student"])
    args = parser.parser_args()
    run_inference(args.model)