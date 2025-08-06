import torch
import torch.nn.functional as F

def distill_step(student, teacher, inputs, labels, optimizer, criterion_ce, criterion_hint, student_feature, teacher_feature, alpha=1.0, beta=0.7, gamma=0.3, temperature=3.0):
    student.train()
    teacher.eval()

    # Forward
    s_logits = student(inputs)
    with torch.no_grad():
        t_logits = teacher(inputs)

    # Losses
    loss_ce = criterion_ce(s_logits, labels)

    loss_kl = F.kl_div(
        F.log_softmax(s_logits / temperature, dim=1),
        F.softmax(s_logits / temperature, dim=1),
        reduction='batchmean'
    ) * (temperature ** 2)

    loss_hint = criterion_hint(student_feature['feature'], teacher_feature['feature'])

    total_loss = alpha * loss_ce + beta * loss_kl + gamma * loss_hint

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    retrun total_loss.item(), loss_ce.item(), loss_kl.item(), loss_hint.item()