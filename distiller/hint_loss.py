import torch
import torch.nn as nn
import torch.nn.functional as F

class HintLoss(nn.Module):
    def __init__(self, student_channels, teacher_channels):
        super(HintLoss, self).__init__()
        # 1x1 convolution to align student feature channels with teacher's
        self.regressor = nn.Conv2d(student_channels, teacher_channels, kernel_size=1)

    def forward(self, student_feat, teacher_feat):
        """
        Args:
            student_feat: student feature map (B, C1, H, W)
            teacher_feat: teacher feature map (B, C2, H, W)
        Returns:
            MSE loss between aligned student and teacher feature maps
        """
        student_proj = self.regressor(student_feat)

        # Resize student_proj to match teacher_feat spatially
        if student_proj.shape[2:] != teacher_feat.shape[2:]:
            student_proj = F.adaptive_avg_pool2d(student_proj, teacher_feat.shape[2:])

        return F.mse_loss(student_proj, teacher_feat)
