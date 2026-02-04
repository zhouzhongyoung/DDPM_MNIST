import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

# ==========================================
# 1. 配置
# ==========================================
class Config:
    n_T = 1000
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lrate = 2e-4
    batch_size = 128
    n_epochs = 60          # 稍微多练一会儿，保证收敛
    n_feat = 256           # 【关键修改】通道翻倍！128 -> 256 (宽体网络)
    save_path = "./ddpm_mnist_v4.pth"
    gen_img_path = "./generated_mnist_v4.png"

config = Config()

# ==========================================
# 2. 核心模块：EMA
# ==========================================
class EMA:
    def __init__(self, model, decay=0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}

# ==========================================
# 3. 网络定义 (V4版：无Dropout，宽体，带Attention)
# ==========================================

class SelfAttention(nn.Module):
    def __init__(self, channels):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
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
        self.conv1 = nn.Sequential(
            nn.GroupNorm(8, in_c), nn.SiLU(),
            nn.Conv2d(in_c, out_c, 3, 1, 1)
        )
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_c),
        )
        # 【关键修改】：移除了 Dropout
        self.conv2 = nn.Sequential(
            nn.GroupNorm(8, out_c), nn.SiLU(),
            nn.Conv2d(out_c, out_c, 3, 1, 1)
        )
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

        self.to_time_emb = nn.Sequential(
            nn.Linear(1, time_emb_dim), nn.GELU(), nn.Linear(time_emb_dim, time_emb_dim)
        )

        self.init_conv = nn.Conv2d(in_channels, n_feat, 3, 1, 1)

        # Down1
        self.down1 = UnetDown(n_feat, n_feat * 2, time_emb_dim)
        self.attn1 = SelfAttention(n_feat * 2) 
        
        # Down2
        self.down2 = UnetDown(n_feat * 2, n_feat * 4, time_emb_dim)
        
        # Mid
        self.mid_block1 = ResBlockTimeEmbed(n_feat * 4, n_feat * 4, time_emb_dim)
        self.mid_attn = SelfAttention(n_feat * 4) 
        self.mid_block2 = ResBlockTimeEmbed(n_feat * 4, n_feat * 4, time_emb_dim)

        # Up1
        self.up1 = UnetUp(n_feat * 4, n_feat * 2, time_emb_dim)
        self.attn_up1 = SelfAttention(n_feat * 2)

        # Up2
        self.up2 = UnetUp(n_feat * 2, n_feat, time_emb_dim)

        self.out = nn.Sequential(
            nn.GroupNorm(8, n_feat), nn.SiLU(),
            nn.Conv2d(n_feat, in_channels, 3, 1, 1),
        )

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

# ==========================================
# 4. DDPM 类
# ==========================================
class DDPM(nn.Module):
    def __init__(self, nn_model, betas, n_T, device, drop_prob=0.1):
        super(DDPM, self).__init__()
        self.nn_model = nn_model.to(device)
        for k, v in betas.items():
            self.register_buffer(k, v)
        self.n_T = n_T
        self.device = device
        self.loss_mse = nn.MSELoss()

    def forward(self, x):
        _ts = torch.randint(1, self.n_T+1, (x.shape[0],)).to(self.device)
        noise = torch.randn_like(x)
        x_t = (
            self.sqrtab[_ts, None, None, None] * x
            + self.sqrtmab[_ts, None, None, None] * noise
        )
        noise_pred = self.nn_model(x_t, _ts)
        return self.loss_mse(noise, noise_pred)

    def sample(self, n_sample, size, device):
        self.nn_model.eval()
        with torch.no_grad():
            x_i = torch.randn(n_sample, *size).to(device)
            for i in range(self.n_T, 0, -1):
                if i % 100 == 0: print(f"Sampling step {i}/{self.n_T}", end="\r")
                z = torch.randn(n_sample, *size).to(device) if i > 1 else 0
                t_tensor = torch.tensor([i for _ in range(n_sample)]).to(device)
                eps = self.nn_model(x_i, t_tensor)
                x_i = (
                    self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
                    + self.sqrt_beta_t[i] * z
                )
                if i > 1:
                    x_i = x_i.clamp(-1, 1)
        self.nn_model.train()
        
        # 归一化到 [0, 1]
        x_i = (x_i.clamp(-1, 1) + 1) / 2
        return x_i

def ddpm_schedules(beta1, beta2, T):
    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"
    beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1
    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

    sqrtab = torch.sqrt(alphabar_t)
    oneover_sqrta = 1 / torch.sqrt(alpha_t)

    sqrtmab = torch.sqrt(1 - alphabar_t)
    mab_over_sqrtmab = (1 - alpha_t) / (sqrtmab + 1e-4)

    return {
        "alpha_t": alpha_t,
        "oneover_sqrta": oneover_sqrta,
        "sqrt_beta_t": sqrt_beta_t,
        "alphabar_t": alphabar_t,
        "sqrtab": sqrtab,
        "sqrtmab": sqrtmab,
        "mab_over_sqrtmab": mab_over_sqrtmab,
    }

# ==========================================
# 5. 主程序
# ==========================================
def main():
    print(f"Device: {config.device}")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=4)

    # 宽体网络：n_feat=256
    nn_model = ContextUnet(in_channels=1, n_feat=config.n_feat, n_classes=10)
    betas = ddpm_schedules(1e-4, 0.02, config.n_T)
    ddpm = DDPM(nn_model, betas, config.n_T, config.device).to(config.device)
    
    ema = EMA(ddpm)

    optim = torch.optim.Adam(ddpm.parameters(), lr=config.lrate)
    scheduler = CosineAnnealingLR(optim, T_max=config.n_epochs, eta_min=1e-6)

    print(f"Start Training (V4: Width=256, No Dropout, 60 Epochs)...")
    
    for epoch in range(config.n_epochs):
        ddpm.train()
        pbar = tqdm(dataloader)
        loss_ema = None
        for x, _ in pbar:
            optim.zero_grad()
            x = x.to(config.device)
            loss = ddpm(x)
            loss.backward()
            optim.step()
            
            ema.update()
            
            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.95 * loss_ema + 0.05 * loss.item()
            pbar.set_description(f"Epoch {epoch+1}/{config.n_epochs} | Loss: {loss_ema:.4f}")
        
        scheduler.step()

    print("Training finished.")
    torch.save(ddpm.state_dict(), config.save_path)

    print("Start Generating with EMA weights...")
    ema.apply_shadow()
    ddpm.eval()
    with torch.no_grad():
        x_gen = ddpm.sample(64, (1, 28, 28), config.device)
        
        # 【最终杀手锏】：对生成的图片进行强制二值化处理
        # 任何小于 0.1 (接近黑) 的像素直接变 0，大于的变 1
        # 这会瞬间消除所有背景噪点，让数字变得极其锐利
        x_gen[x_gen < 0.1] = 0 
    
    ema.restore()
    
    grid = make_grid(x_gen, nrow=8)
    save_image(grid, config.gen_img_path)
    print(f"Generated images saved to {config.gen_img_path}")
    
    plt.figure(figsize=(10,10))
    plt.axis('off')
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
    plt.show()

if __name__ == "__main__":
    main()