import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

def load_losses(case_dir):
    path = os.path.join(case_dir, "cnp_training_losses.csv")
    return pd.read_csv(path)

def load_predictions(case_dir, pred_type):
    path = os.path.join(case_dir, "cnp_predictions", f"predictions_{pred_type}.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

def compare_losses(loss1, loss2, label1, label2):
    print(f"\n--- Training Loss Comparison ({label1} vs {label2}) ---")
    print(f"Final loss {label1}: {loss1.iloc[-1].to_dict()}")
    print(f"Final loss {label2}: {loss2.iloc[-1].to_dict()}")
    print(f"Mean loss {label1}: {loss1.mean().to_dict()}")
    print(f"Mean loss {label2}: {loss2.mean().to_dict()}")
    # Plot
    plt.figure(figsize=(10,5))
    plt.plot(loss1['Epoch'], loss1['Train Loss'], label=f'{label1} train')
    plt.plot(loss1['Epoch'], loss1['Validation Loss'], label=f'{label1} val')
    plt.plot(loss2['Epoch'], loss2['Train Loss'], label=f'{label2} train')
    plt.plot(loss2['Epoch'], loss2['Validation Loss'], label=f'{label2} val')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss Comparison')
    plt.savefig(f"{label1}_vs_{label2}_loss.png")  # <-- add this line
    plt.close()
    #plt.show()

def compare_predictions(pred1, pred2, label1, label2, pred_type):
    print(f"\n--- {pred_type.upper()} Prediction Comparison ({label1} vs {label2}) ---")
    if pred1 is None or pred2 is None:
        print(f"Prediction file for {pred_type} missing in one or both cases.")
        return
    # Align by index
    min_len = min(len(pred1), len(pred2))
    pred1 = pred1.iloc[:min_len]
    pred2 = pred2.iloc[:min_len]
    for col in pred1.columns:
        if col in pred2.columns:
            arr1 = pred1[col].values
            arr2 = pred2[col].values
            mse = mean_squared_error(arr1, arr2)
            rmse = np.sqrt(mse)
            corr = np.corrcoef(arr1, arr2)[0,1]
            print(f"{col}: RMSE={rmse:.4f}, Corr={corr:.4f}, Mean1={arr1.mean():.4f}, Mean2={arr2.mean():.4f}")
            # Optionally plot
            plt.figure(figsize=(5,5))
            plt.scatter(arr1, arr2, alpha=0.3)
            plt.xlabel(f"{label1} {col}")
            plt.ylabel(f"{label2} {col}")
            plt.title(f"{pred_type} {col}: {label1} vs {label2}")
            plt.plot([arr1.min(), arr1.max()], [arr1.min(), arr1.max()], 'r--')
            plt.savefig(f"{label1}_vs_{label2}_{pred_type}_{col}.png")  # <-- add this line
            plt.close()
            #plt.show()

def main(case1, case2):
    label1 = os.path.basename(case1)
    label2 = os.path.basename(case2)
    # Losses
    loss1 = load_losses(case1)
    loss2 = load_losses(case2)
    compare_losses(loss1, loss2, label1, label2)
    # Predictions
    for pred_type in ['2d', '1d', 'scalar']:
        pred1 = load_predictions(case1, pred_type)
        pred2 = load_predictions(case2, pred_type)
        compare_predictions(pred1, pred2, label1, label2, pred_type)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compare two CNP model runs.")
    parser.add_argument("case1", type=str, help="Path to first case directory (e.g. run_20250804_103043)")
    parser.add_argument("case2", type=str, help="Path to second case directory (e.g. run_20250803_115521)")
    args = parser.parse_args()
    main(args.case1, args.case2)