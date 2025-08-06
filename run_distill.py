import torch 
from dataset.cifar10_loader import get_cifar10_dataloaders
from teacher_model.teacher_cnn import get_teacher
from student_model.student_cnn import get_student
from distiller.hint_loss import HintLoss
from distiller.fitnes_util import register_feature_hook
from distiller.distill_trainer import distill_step
from monitor.monitor import PowerMonitor

def run():
    device = torch.device("cuda" if torch.cuda.is_available else "cpu")

    train_loader, _ = get_cifar10_dataloaders()
    teacher = get_teacher().to(device)
    student = get_student.to(device)

    # Register feature hooks
    teacher_feature = {}
    student_feature = {}
    register_feature_hook(teacher, "features.3", teacher_feature) # adjust based on layer
    register_feature_hook(student, "features.0", student_feature)

    criterion_ce = torch.nn.CrossEntropyLoss()
    criterion_hint = HintLoss(student_channels=32, teacher_channels=128).to(device)
    optimizer = torch.optim.Adam(list(student.parameters()) + list(criterion_hint.parameters()), lr=1e-3)

    with PowerMonitor(outfile="logs/distill_power.csv") as monitor:
        for epoch in range(5):
            for batch in train_loader:
                inputs, labels = batch 
                inputs, labels - inputs.to(device), labels.to(device)

                total_loss, loss_ce, loss_kl, loss_hint, = distill_step(
                    student, teacher, inputs, labels,
                    optimizer, criterion_ce, criterion_hint,
                    student_feature, teacher_feature
                )

                print(f"[Epoch {epoch+1}] Total: {total_loss:.4f}, CE: {loss_ce:.4f}, KL: {loss_kl:.4f}, Hint: {loss_hint:.4f}")

if __name__ == "__main__":
    run()