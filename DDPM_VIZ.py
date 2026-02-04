import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import os

# ==========================================
# 1. 配置
# ==========================================
class Config:
    n_T = 1000
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_feat = 256
    save_path = "./ddpm_mnist_v4.pth"

config = Config()

# ==========================================
# 2. 网络定义 (保持不变)
# ==========================================
class SelfAttention(nn.Module):
    def __init__(self, channels):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]), nn.Linear(channels, channels), nn.GELU(), nn.Linear(channels, channels)
        )
    def forward(self, x):
        size = x.shape[-2:]
        x = x.view(-1, self.channels, size[0] * size[1]).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, size[0], size[1])

class ResBlockTimeEmbed(nn.Module):
    def __init__(self, in_c, out_c, time_emb_dim):
        super().__init__()
        self.conv1 = nn.Sequential(nn.GroupNorm(8, in_c), nn.SiLU(), nn.Conv2d(in_c, out_c, 3, 1, 1))
        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, out_c))
        self.conv2 = nn.Sequential(nn.GroupNorm(8, out_c), nn.SiLU(), nn.Conv2d(out_c, out_c, 3, 1, 1))
        self.residual = nn.Conv2d(in_c, out_c, 1) if in_c != out_c else nn.Identity()
    def forward(self, x, t_emb):
        h = self.conv1(x)
        time_emb_proj = self.time_mlp(t_emb)[:, :, None, None]
        h = h + time_emb_proj
        h = self.conv2(h)
        return h + self.residual(x)

class UnetDown(nn.Module):
    def __init__(self, in_c, out_c, time_emb_dim):
        super().__init__()
        self.block = ResBlockTimeEmbed(in_c, out_c, time_emb_dim)
        self.pool = nn.MaxPool2d(2)
    def forward(self, x, t_emb):
        x = self.block(x, t_emb)
        return self.pool(x), x

class UnetUp(nn.Module):
    def __init__(self, in_c, out_c, time_emb_dim):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c, in_c, 2, 2)
        self.block = ResBlockTimeEmbed(in_c * 2, out_c, time_emb_dim)
    def forward(self, x, skip, t_emb):
        x = self.up(x)
        x = torch.cat((x, skip), 1)
        x = self.block(x, t_emb)
        return x

class ContextUnet(nn.Module):
    def __init__(self, in_channels, n_feat=256, n_classes=10):
        super(ContextUnet, self).__init__()
        self.in_channels = in_channels
        self.n_feat = n_feat
        time_emb_dim = n_feat * 4
        self.to_time_emb = nn.Sequential(nn.Linear(1, time_emb_dim), nn.GELU(), nn.Linear(time_emb_dim, time_emb_dim))
        self.init_conv = nn.Conv2d(in_channels, n_feat, 3, 1, 1)
        self.down1 = UnetDown(n_feat, n_feat * 2, time_emb_dim)
        self.attn1 = SelfAttention(n_feat * 2) 
        self.down2 = UnetDown(n_feat * 2, n_feat * 4, time_emb_dim)
        self.mid_block1 = ResBlockTimeEmbed(n_feat * 4, n_feat * 4, time_emb_dim)
        self.mid_attn = SelfAttention(n_feat * 4) 
        self.mid_block2 = ResBlockTimeEmbed(n_feat * 4, n_feat * 4, time_emb_dim)
        self.up1 = UnetUp(n_feat * 4, n_feat * 2, time_emb_dim)
        self.attn_up1 = SelfAttention(n_feat * 2)
        self.up2 = UnetUp(n_feat * 2, n_feat, time_emb_dim)
        self.out = nn.Sequential(nn.GroupNorm(8, n_feat), nn.SiLU(), nn.Conv2d(n_feat, in_channels, 3, 1, 1))

    def forward(self, x, t):
        t = t.float() / config.n_T
        t = t.view(-1, 1)
        t_emb = self.to_time_emb(t)
        x = self.init_conv(x)
        x_down1, skip1 = self.down1(x, t_emb)
        x_down1 = self.attn1(x_down1) 
        x_down2, skip2 = self.down2(x_down1, t_emb)
        x_mid = self.mid_block1(x_down2, t_emb)
        x_mid = self.mid_attn(x_mid)
        x_mid = self.mid_block2(x_mid, t_emb)
        x_up1 = self.up1(x_mid, skip2, t_emb)
        x_up1 = self.attn_up1(x_up1)
        x_up2 = self.up2(x_up1, skip1, t_emb)
        return self.out(x_up2)

def ddpm_schedules(beta1, beta2, T):
    beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1
    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1 - beta_t
    
    # 累积乘积 alpha_bar
    alphabar_t = torch.cumsum(torch.log(alpha_t), dim=0).exp()
    
    # 【核心】计算 alpha_bar_prev (上一时刻的累积乘积)
    # 也就是 padding 一个 1.0 在最前面，丢掉最后一个
    alphabar_t_prev = torch.cat([torch.tensor([1.0]), alphabar_t[:-1]], dim=0)
    
    # ====================================================
    # 【核心数学推导】计算真实后验方差 (True Posterior Variance)
    # Formula: tilde_beta_t = beta_t * (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t)
    # ====================================================
    posterior_variance = beta_t * (1. - alphabar_t_prev) / (1. - alphabar_t)
    
    # 因为 t=0 时 posterior_variance 是 0，取 log 会变成 -inf
    # 为了数值稳定性，我们通常会对第一步做截断或取 log(beta_t) 代替
    # 这里我们只计算 sqrt 用于采样
    sqrt_posterior_variance = torch.sqrt(posterior_variance)

    sqrtab = torch.sqrt(alphabar_t)
    sqrtmab = torch.sqrt(1 - alphabar_t)
    oneover_sqrta = 1 / torch.sqrt(alpha_t)
    mab_over_sqrtmab = (1 - alpha_t) / (sqrtmab + 1e-4)
    
    return {
        "alpha_t": alpha_t, 
        "oneover_sqrta": oneover_sqrta, 
        "sqrt_beta_t": sqrt_beta_t, # 方案 A: 工程近似方差
        "sqrt_posterior_variance": sqrt_posterior_variance, # 方案 B: 数学真值方差
        "alphabar_t": alphabar_t, 
        "sqrtab": sqrtab, 
        "sqrtmab": sqrtmab, 
        "mab_over_sqrtmab": mab_over_sqrtmab,
    }

# ==========================================
# 3. 实验类
# ==========================================
class DDPM_Experiment(nn.Module):
    def __init__(self, nn_model, betas, n_T, device):
        super(DDPM_Experiment, self).__init__()
        self.nn_model = nn_model.to(device)
        for k, v in betas.items():
            self.register_buffer(k, v)
        self.n_T = n_T
        self.device = device

    def generate(self, x_T, variance_type='large'):
        """
        variance_type: 
          - 'large': 使用 beta_t (原代码做法)
          - 'small': 使用 tilde_beta_t (后验真值)
        """
        x_i = x_T.clone().to(self.device)
        self.nn_model.eval()
        with torch.no_grad():
            for i in range(self.n_T, 0, -1):
                z = torch.randn_like(x_i).to(self.device) if i > 1 else 0
                t_tensor = torch.tensor([i]).to(self.device)
                
                eps = self.nn_model(x_i, t_tensor)
                
                # 选择使用哪个方差
                if variance_type == 'large':
                    sigma_t = self.sqrt_beta_t[i]
                elif variance_type == 'small':
                    sigma_t = self.sqrt_posterior_variance[i]
                
                # 均值部分 (完全一样)
                mean = self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
                
                # 采样公式：均值 + 不同的标准差 * 噪声
                x_i = mean + sigma_t * z
                
                if i > 1: x_i = x_i.clamp(-1, 1)
        
        x_0 = (x_i.clamp(-1, 1) + 1) / 2
        x_0[x_0 < 0.1] = 0
        return x_0

def perturb_noise_by_pixels(base_noise, num_pixels_to_change):
    new_noise = base_noise.clone()
    B, C, H, W = base_noise.shape
    total_pixels = H * W
    if num_pixels_to_change > 0:
        flat_indices = torch.randperm(total_pixels)[:num_pixels_to_change]
        random_updates = torch.randn(num_pixels_to_change).to(base_noise.device)
        new_noise.view(-1)[flat_indices] = random_updates
    return new_noise

# ==========================================
# 4. 主程序
# ==========================================
def main():
    print(f"Loading model from {config.save_path}...")
    
    nn_model = ContextUnet(in_channels=1, n_feat=config.n_feat, n_classes=10)
    # 这里 betas 会计算出新的 sqrt_posterior_variance
    betas = ddpm_schedules(1e-4, 0.02, config.n_T)
    
    experimenter = DDPM_Experiment(nn_model, betas, config.n_T, config.device).to(config.device)
    
    # 【关键修改】加上 strict=False
    # 这样旧权重里缺失的 sqrt_posterior_variance 就会被自动忽略，
    # 而使用 experimenter 初始化时计算好的新值。
    experimenter.load_state_dict(torch.load(config.save_path, map_location=config.device), strict=False)
    
    # ... (后续代码不变)
    
    # -------------------------------------------------
    # 实验设置
    # -------------------------------------------------
    torch.manual_seed(999) 
    base_noise = torch.randn(1, 1, 28, 28).to(config.device)
    
    perturb_levels = [0, 5, 20, 100, 400]
    results = []

    print("Running Variance Type Experiment...")
    print("Comparing: 'Engineering Variance (Beta)' vs 'Mathematical Variance (Tilde Beta)'")
    
    for pixels in perturb_levels:
        print(f"Processing perturbation: {pixels} pixels...")
        current_noise = perturb_noise_by_pixels(base_noise, pixels)
        
        # 1. 计算掩码
        diff_map = torch.abs(current_noise - base_noise).cpu().squeeze()
        mask_map = (diff_map > 1e-4).float()
        
        # 2. 使用“工程方差”生成 (4张)
        eng_imgs = []
        for _ in range(4):
            eng_imgs.append(experimenter.generate(current_noise, variance_type='large'))
        
        # 3. 使用“数学真值方差”生成 (4张)
        math_imgs = []
        for _ in range(4):
            math_imgs.append(experimenter.generate(current_noise, variance_type='small'))
            
        results.append({
            "pixels": pixels,
            "noise": current_noise,
            "mask": mask_map,
            "eng": eng_imgs,
            "math": math_imgs
        })

    # ==========================================
    # 5. 绘图
    # ==========================================
    print("Plotting results...")
    
    n_rows = len(perturb_levels)
    n_cols = 10 
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 2.5 * n_rows))
    
    for row_idx, res in enumerate(results):
        pixels = res["pixels"]
        
        # Col 1: Noise
        ax = axes[row_idx, 0]
        noise_disp = (res["noise"].cpu().squeeze().clamp(-2, 2) + 2) / 4
        ax.imshow(noise_disp, cmap='gray')
        ax.set_ylabel(f"Change\n{pixels} Pixels", fontweight='bold', rotation=0, labelpad=40, va='center')
        ax.set_xticks([]); ax.set_yticks([])
        if row_idx == 0: ax.set_title("Input Noise", fontsize=9)

        # Col 2: Mask
        ax = axes[row_idx, 1]
        ax.imshow(res["mask"], cmap='hot', vmin=0, vmax=1)
        ax.set_xticks([]); ax.set_yticks([])
        if row_idx == 0: ax.set_title("Diff Mask", fontsize=9)

        # Col 3-6: Engineering Variance (Large)
        for i in range(4):
            ax = axes[row_idx, 2 + i]
            ax.imshow(res["eng"][i].cpu().squeeze(), cmap='gray')
            ax.axis('off')
            if row_idx == 0 and i == 0: 
                ax.set_title("Variance = Beta_t\n(Original / Larger)", fontsize=10, fontweight='bold', color='blue', loc='left')

        # Col 7-10: Mathematical Variance (Small)
        for i in range(4):
            ax = axes[row_idx, 6 + i]
            ax.imshow(res["math"][i].cpu().squeeze(), cmap='gray')
            ax.axis('off')
            if row_idx == 0 and i == 0: 
                ax.set_title("Variance = Posterior\n(Mathematical / Smaller)", fontsize=10, fontweight='bold', color='green', loc='left')

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.savefig("variance_check.png", dpi=300, bbox_inches='tight')
    print("Done! Saved to 'variance_check.png'")
    plt.show()

if __name__ == "__main__":
    main()