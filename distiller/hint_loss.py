import torch
import torch.nn as nn
import torch.nn.functional as F

class HintLoss(nn.Module):
    def __init__(self, student_channels, teacher_channels):
        super(HintLoss, self).__init__()
        # 1x1 convolution to project student features to match teacher's dimensions
        self.regressor = nn.Conv2d(student_channels, teacher_channels, kernel_size=1)

    def forward(self, student_feat, teacher_feat):
        """
        Args:
            student_feat: studnet feature map (B, C1, H, W)
            teacher_feat: teacher feature map (B, C2, H, W)
        Returns:
            MSE lass between regressed student and teacher features
        """
        student_proj = self.regressor(student_feat)
        return F.mse_loss(student_proj, teacher_feat)
        
