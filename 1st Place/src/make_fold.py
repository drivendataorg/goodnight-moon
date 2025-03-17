from sklearn.model_selection import KFold
import pandas as pd


if __name__ == "__main__":
    trn_meta = pd.read_csv('../../inputs/train/train_metadata.csv')
    label = pd.read_csv('../../inputs/train/train_labels.csv')
    trn_meta = trn_meta.merge(label, on='filename', how='inner')

    kf = KFold(n_splits=5)
    trn_meta['fold'] = -1
    for idx, (trn, val) in enumerate(kf.split(trn_meta)):
        trn_meta.iloc[val, -1] = idx
        
    trn_meta.to_csv('../fold.csv')