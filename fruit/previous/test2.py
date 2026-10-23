"""執行 LEGO 零件數預測的完整流程；舊範例與全部註解見 legacy/test2.py。"""
import argparse
import copy
import csv
import json
import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from cnn import LegoDNN
from dataset import prepare_data


def metrics(actual, predicted):
    residual = actual-predicted
    return {'MAE':float(np.abs(residual).mean()), 'RMSE':float(np.sqrt((residual**2).mean())),
            'R2':float(1-(residual**2).sum()/((actual-actual.mean())**2).sum())}


def main():
    parser = argparse.ArgumentParser(description='LEGO DNN regression')
    parser.add_argument('--archive', default=str(Path(__file__).with_name('archive.zip')))
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--out-dir', default='result')
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0:
        parser.error('epochs、batch-size 與 lr 必須大於 0')
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    torch.set_num_threads(4)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    x, y, frames, metadata = prepare_data(args.archive)
    xt = [torch.from_numpy(a) for a in x]
    yt = [torch.from_numpy(np.log1p(a)) for a in y]
    loader = DataLoader(TensorDataset(xt[0], yt[0]), batch_size=args.batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(42))
    model = LegoDNN(x[0].shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = torch.nn.MSELoss()
    best_loss, best_epoch, stale, history, best_state = float('inf'), 0, 0, [], None
    # 只根據驗證集選模型；測試集留到最後使用一次。
    for epoch in range(1, args.epochs+1):
        model.train(); total = 0
        for bx, by in loader:
            optimizer.zero_grad(); loss = criterion(model(bx), by)
            loss.backward(); optimizer.step(); total += loss.item()*len(bx)
        model.eval()
        with torch.no_grad():
            val = criterion(model(xt[1]), yt[1]).item()
        history.append([epoch, total/len(y[0]), val])
        if val < best_loss-1e-5:
            best_loss, best_epoch, stale = val, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
        if epoch == 1 or epoch % 10 == 0:
            print(f'Epoch {epoch}: train={history[-1][1]:.4f} val={val:.4f}', flush=True)
        if stale >= 15:
            break
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        predictions = np.maximum(np.expm1(model(xt[2]).numpy()), 0)
    baseline = np.full_like(y[2], np.median(y[0]))
    results = {'task':'Predict LEGO set num_parts', 'seed':42, 'best_epoch':best_epoch,
               'epochs_run':len(history), 'best_validation_log_mse':best_loss,
               'dnn_test':metrics(y[2], predictions), 'median_baseline_test':metrics(y[2], baseline),
               'metadata':metadata}
    torch.save({'model_state_dict':best_state, 'metadata':metadata, 'target_transform':'log1p'}, out/'best_dnn.pth')
    (out/'metrics.json').write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
    with (out/'loss.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f); w.writerow(['epoch','train_log_mse','val_log_mse']); w.writerows(history)
    samples = frames[2][['set_num','name','year','num_parts']].copy()
    samples['predicted_num_parts'] = predictions
    samples['absolute_error'] = np.abs(y[2]-predictions)
    samples.to_csv(out/'test_predictions.csv', index=False, encoding='utf-8-sig')
    print(json.dumps({k:v for k,v in results.items() if k!='metadata'}, indent=2))

if __name__ == '__main__':
    main()
