import os
import argparse
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from cnn import CNN
import time
from torch.utils.data import random_split
import random
import matplotlib.pyplot as plt
from pathlib import Path
import csv
from matplotlib import font_manager

# ========== 可調參數 ==========
DATA_DIR  = "data"
OUT_DIR   = "result"
CKPT_PATH = Path(OUT_DIR) / "best_cnn.pth" # 你的訓練程式儲存檔
ROWS, COLS = 2, 5                          # 畫格 (共 ROWS*COLS 張)
SEED      = 42                             # 抽樣固定
MEAN, STD = 0.2861, 0.3530                 # 與訓練一致
FIGSIZE   = (16, 8)                        # 圖片大小
TITLE_FZ  = 11                             # 標題字體大小
ZH_FONT_CANDIDATES = [
    # Windows (建議微軟正黑體)
    r"C:\Windows\Fonts\msjh.ttc",
    r"C:\Windows\Fonts\msjhl.ttc",
    r"C:\Windows\Fonts\msjhbd.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
    r"C:\Windows\Fonts\simsun.ttc",
    # macOS
    r"/System/Library/Fonts/PingFang.ttc",
    # Ubuntu / Linux (先安裝 fonts-noto-cjk)
    r"/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    r"/usr/share/fonts/truetype/noto/NotoSansTC-Regular.otf",
]

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def set_zh_font():
    """全域設定一個可用的中文字型，避免中文顯示方塊。"""
    from pathlib import Path as _Path
    for p in ZH_FONT_CANDIDATES:
        if _Path(p).exists():
            prop = font_manager.FontProperties(fname=p)
            plt.rcParams["font.sans-serif"] = [prop.get_name()]
            plt.rcParams["font.family"] = "sans-serif"
            plt.rcParams["axes.unicode_minus"] = False # 讓負號正常顯示
            print(f"[INFO] 使用中文字型: {prop.get_name()} ({p})")
            return
    print("[WARN] 找不到中文字型。請安裝「微軟正黑體」 / Noto Sans CJK；或更新 ZH_FONT_CANDIDATES。")


def denorm_img(x):
    """將張量從 [H,W] 或 [1,H,W] 經 Normalize 還原到 0~1"""
    if x.ndim == 3 and x.shape[0] == 1:
        x = x[0]
    x = x * STD + MEAN
    return x.clamp(0, 1)

def main():
    set_seed(SEED)
    set_zh_font()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Transform 與訓練一致
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((MEAN,), (STD,))
    ])

    # 讀取 test set (官方 10,000 張)
    test_set = datasets.FashionMNIST(root=DATA_DIR, train=False, download=True, transform=tfm)

    # 準備模型並載入最佳權重
    model = CNN(num_classes=10).to(device)
    if CKPT_PATH.exists():
        ckpt = torch.load(CKPT_PATH, map_location=device)
        state = ckpt.get("model_state", ckpt) # 兼容只存 state_dict 或整包的情況
        model.load_state_dict(state)
        print(f"[INFO] loaded checkpoint: {CKPT_PATH}")
    else:
        print(f"[WARN] checkpoint not found: {CKPT_PATH} → 使用隨機初始化權重（僅示意）")

    model.eval()

    # 抽樣 N 張
    N = ROWS * COLS
    indices = list(range(len(test_set)))
    random.shuffle(indices)
    indices = indices[:N]

    # 推論
    images = []
    gts = []
    preds = []
    confs = []

    with torch.no_grad():
        for idx in indices:
            img_t, label = test_set[idx] # img_t: [1,H,W] (normalized)
            logits = model(img_t.unsqueeze(0).to(device)) # [1,10]
            prob = F.softmax(logits, dim=1)[0]           # [10]
            pred_id = int(torch.argmax(prob).item())
            conf = float(prob[pred_id].item())

            images.append(img_t.cpu())
            gts.append(label)
            preds.append(pred_id)
            confs.append(conf)

    # 畫圖
    os.makedirs(OUT_DIR, exist_ok=True)
    fig = plt.figure(figsize=FIGSIZE, constrained_layout=True)

    for i in range(N):
        ax = fig.add_subplot(ROWS, COLS, i + 1)
        img = denorm_img(images[i])
        ax.imshow(img, cmap="gray", interpolation="nearest") # 保留像素風格
        ax.axis("off")

        gt_name   = CLASS_NAMES[gts[i]]
        pred_name = CLASS_NAMES[preds[i]]
        conf_txt  = f"{confs[i]:.3f}"

        # 三行文字：真實/預測/信心度
        title = f"真實: {gt_name}\n預測: {pred_name}\n信心度: {conf_txt}"
        ax.set_title(title, color="green", fontsize=TITLE_FZ, pad=6)

    # 大標題（稍微下移避免壓到子圖）
    fig.suptitle("模型預測結果", fontsize=16, y=0.98)

    out_path = Path(OUT_DIR) / "test_grid.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] 已輸出: {out_path.resolve()}")

if __name__ == "__main__":
    main()

def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def plot_loss(csv_path: Path, out_path: Path):
    epochs, tr_losses, val_losses, tr_accs, val_accs = [],[],[],[],[]
    with open(csv_path, 'r', encoding='utf-8') as f:
        header = f.readline()
        for line in f:
            e, tl, vl, ta, va =line.strip().split(',')
            epochs.append(int(e))
            tr_losses.append(float(tl))
            val_losses.append(float(vl))
            tr_accs.append(float(ta))
            val_accs.append(float(va))

    plt.figure(figsize=(6,4))
    plt.plot(epochs, tr_losses, label='train_loss')
    plt.plot(epochs, val_losses, label='val_loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def main():
    set_seed(42)
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data',   help='資料下載/讀取目錄')
    parser.add_argument('--out_dir',  type=str, default='result', help='訓練輸出目錄')
    parser.add_argument('--epochs',   type=int, default=5,        help='訓練回合數')
    parser.add_argument('--batch_size', type=int, default=128,      help='批次大小')
    parser.add_argument('--lr',         type=float, default=1e-3,     help='學習率')
    args = parser.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"data_dir={args.data_dir}, out_dir={args.out_dir}")

    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)) 
        ])
    
    train_dataset = datasets.FashionMNIST(
        root=args.data_dir, train=True, download=True, transform=tfm
    )
    print(f"訓練集筆數：{len(train_dataset)} (下載位置：{args.data_dir})")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    print(f"DataLoader 就緒 (batch_size={args.batch_size}) ")

    full_train = datasets.FashionMNIST(root=args.data_dir, train=True, download=True, transform=tfm)
    val_len = 10000
    train_len = len(full_train) - val_len
    train_set, val_set = random_split(full_train, [train_len, val_len])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader  = DataLoader(val_set,  batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    model = CNN(num_classes=10).to(device)
    #print(f"使用裝置：{device}")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    #print("[S5] Loss/Optimizer 準備完成")
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    # 縮小訓練迴圈 + CSV 紀錄
    loss_csv = os.path.join(args.out_dir, 'loss.csv')
    with open(loss_csv, 'w', encoding='utf-8') as f:
        f.write('epoch,train_loss,val_loss,train_acc,val_acc\n')

    best_val_loss = float('inf')
    best_path = os.path.join(args.out_dir, 'best_cnn.pth')

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        running_loss, running_acc, n = 0.0, 0.0, 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device).long()  # CE 需要 long/int64

            optimizer.zero_grad()
            logits = model(images)             # 形狀 [N, 10]
            loss = criterion(logits, labels)   # 未做 softmax
            loss.backward()
            optimizer.step()

            bs = labels.size(0)
            running_loss += loss.item() * bs
            running_acc += (logits.argmax(1) == labels).float().sum().item()
            n += bs
            
        train_loss = running_loss / n
        train_acc = running_acc / n

        # ---- validate ----
        model.eval()
        val_loss, val_acc, m = 0.0, 0.0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                loss = criterion(logits, labels)
                bs = labels.size(0)
                val_loss += loss.item() * bs
                val_acc += (logits.argmax(1) == labels).float().sum().item()
                m += bs                
        val_loss /= m
        val_acc /= m

        scheduler.step(val_loss)

        with open(loss_csv, 'a', encoding='utf-8') as f:
            f.write(f"{epoch},{train_loss:.6f},{val_loss:.6f},{train_acc:.4f},{val_acc:.4f}\n")

        print(f"[Epoch {epoch:02d}] train_loss={train_loss:.4f} val_loss={val_loss:.4f} train_acc={train_acc:.4f} val_acc={val_acc:.4f}")

        # dt = time.time() - t0
        # print(f"[Epoch {epoch:02d}] train_loss={train_loss:.4f} ({dt:.1f}s)")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, best_path)
            print(f" -> Saved new best to {best_path}")


    plot_loss(loss_csv, os.path.join(args.out_dir, 'loss_curve.png'))

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        running_loss, n = 0.0, 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device).long()  # CE 需要 long/int64

            optimizer.zero_grad()
            logits = model(images)             # 形狀 [N, 10]
            loss = criterion(logits, labels)   # 未做 softmax
            loss.backward()
            optimizer.step()

            bs = labels.size(0)
            running_loss += loss.item() * bs
            n += bs

        train_loss = running_loss / n

        with open(loss_csv, 'a', encoding='utf-8') as f:
            f.write(f"{epoch},{train_loss:.6f}\n")

        dt = time.time() - t0
        print(f"[Epoch {epoch:02d}] train_loss={train_loss:.4f} ({dt:.1f}s)")

    print(f"微小訓練完成，loss 已寫入：{loss_csv}")

if __name__ == '__main__':
    main()
    
