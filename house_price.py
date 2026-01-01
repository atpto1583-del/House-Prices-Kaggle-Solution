#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use("ggplot")
import pandas as pd
import numpy as np
import random
np.random.seed(423)
random.seed(423)
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[2]:


train_df = pd.read_csv("data/house_price/train.csv")
# 削除前の確認
plt.figure(figsize=(10, 6))
plt.scatter(train_df['GrLivArea'], train_df['SalePrice'], alpha=0.5)
plt.xlabel('GrLivArea')
plt.ylabel('SalePrice')
plt.title('Check Outliers')

# 削除対象（面積4000超 かつ 価格20万以下）を赤で描画
outliers = train_df[(train_df['GrLivArea'] > 4000) & (train_df['SalePrice'] < 200000)]
plt.scatter(outliers['GrLivArea'], outliers['SalePrice'], color='red', s=100, label='Outliers to Remove')

# 残すべき豪邸（面積4000超 かつ 価格20万以上）を緑で描画
valid_mansions = train_df[(train_df['GrLivArea'] > 4000) & (train_df['SalePrice'] >= 200000)]
plt.scatter(valid_mansions['GrLivArea'], valid_mansions['SalePrice'], color='green', s=100, label='Valid Mansions (Keep)')
plt.show()

# 外れ値の除去（学習データのみ）
# GrLivAreaが4000を超え、かつSalePriceが200000以下のデータは、
# 文献的にも「異常値」とされているため削除します
train_df = train_df.drop(train_df[(train_df['GrLivArea']>4000) & (train_df['SalePrice']<200000)].index)
test_df = pd.read_csv("data/house_price/test.csv")
submission = pd.read_csv("data/house_price/sample_submission.csv")


# In[3]:


train_df.head(1)


# In[4]:


all_df = pd.concat([train_df, test_df], sort=False).reset_index(drop=True)


# In[5]:


# 数値系の地下変数の欠損を0で埋める
bsmt_num_cols = ['TotalBsmtSF', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'BsmtFullBath', 'BsmtHalfBath']
for col in bsmt_num_cols:
    all_df[col] = all_df[col].fillna(0)

# 確認：すべて 2919 non-null になっているはずです
print(all_df[bsmt_num_cols].info())


# In[6]:


#  総面積 (TotalSF) の作成
# 地下(TotalBsmtSF) + 1階(1stFlrSF) + 2階(2ndFlrSF)
all_df['TotalSF']=all_df['TotalBsmtSF']+all_df['1stFlrSF']+all_df['2ndFlrSF']

#  築年数 (HouseAge) の作成
# 売れた年(YrSold) - 建てられた年(YearBuilt)
# ※ リフォーム年(YearRemodAdd)を使う手もありますが、まずは純粋な築年数でOKです
all_df['HouseAge'] = all_df['YrSold'] - all_df['YearBuilt']


# In[7]:


# --- LotFrontage（間口）の補完 ---

# 補完前の欠損数を確認
print(f"補完前の欠損数: {all_df['LotFrontage'].isnull().sum()}")

# 戦略: 「Neighborhood（地区）」ごとにグループ化し、そのグループの「中央値」で埋める
# transformを使うと、グループごとの計算結果を元のデータの行数に合わせて返してくれます
all_df['LotFrontage'] = all_df.groupby('Neighborhood', observed=True)['LotFrontage'].transform(
    lambda x: x.fillna(x.median())
)

# 補完後の確認（まだ欠損が残っているかチェック）
print(f"補完後の欠損数: {all_df['LotFrontage'].isnull().sum()}")

# 確認用: ちゃんと埋まったか、適当な行を見てみる
print(all_df[['Neighborhood', 'LotFrontage']].head())


# In[8]:


# 意味がある文字列のマッピング

# 1. マッピング（変換ルール）の定義
# NA（なし）は 0、Po（悪い）は 1 ... Ex（最高）は 5 とします
qual_map = {
    'Ex': 5,
    'Gd': 4,
    'TA': 3,
    'Fa': 2,
    'Po': 1,
    'NA': 0
}

# 2. 変換対象のカラムリスト
# これらはすべて同じ 'Ex'〜'Po' の基準で評価されています
qual_cols = [
    'ExterQual',   # 外装の質
    'ExterCond',   # 外装の状態
    'BsmtQual',    # 地下の高さ（質）
    'BsmtCond',    # 地下の状態
    'HeatingQC',   # 暖房の質
    'KitchenQual', # キッチンの質
    'FireplaceQu', # 暖炉の質
    'GarageQual',  # ガレージの質
    'GarageCond',  # ガレージの状態
    'PoolQC'       # プールの質
]

# 3. マッピング前の確認
print("--- 各カラムのユニーク値と件数 ---")
for col in qual_cols:
    print(f"\n■ {col}")
    # dropna=False にすることで、NaN（欠損）もカウントします
    counts = all_df[col].value_counts(dropna=False)
    print(counts)
    
    # 【自動チェック】マッピングのキーに含まれていない値がないか確認
    # ※ NaNは後で0埋めするので除外してチェックします
    existing_values = all_df[col].dropna().unique()
    unknowns = [val for val in existing_values if val not in qual_map]
    
    if unknowns:
        print(f"⚠️ 未定義の値があります: {unknowns}")
    else:
        print("✅ マッピングOK")

# 4. 変換の実行
for col in qual_cols:
    # 現在のcategory型から、マッピングを使って数値に変換
    # mapを使うと、辞書にない値（NaNなど）はNaNのままになるため、
    # 最後に .fillna(0) して int型に整えます
    all_df[col] = all_df[col].map(qual_map).fillna(0).astype(int)

# --- 確認用 ---
# 変換後のデータと、統計量（平均値など）を表示して確認
print(all_df[qual_cols].head())
print("-" * 30)
print(all_df[qual_cols].info())


# In[9]:


# 後の順序尺度から間隔尺度への変更可能性を考慮して、数値の可視化。例えば KitchenQual（キッチン品質）で確認
plt.figure(figsize=(10, 6))
sns.boxplot(x='KitchenQual', y='SalePrice', data=all_df)
plt.title('Sale Price Distribution by Kitchen Quality')
plt.grid(axis='y')
plt.show()

# 各スコアの平均価格を算出
print(all_df.groupby('KitchenQual')['SalePrice'].mean())


# In[10]:


# --- 相互作用特徴量 (Interaction Features) の作成 ---

# 1. 最強の指標：広さ × 品質 (TotalSF * OverallQual)
# 「ただ広い家」と「広くて最高級の家」を明確に区別します
all_df['TotalSF_Qual'] = all_df['TotalSF'] * all_df['OverallQual']

# 2. リビング × 品質 (GrLivArea * OverallQual)
# 生活空間の質を強調します。TotalSF_Qualと似ていますが、
# 地下を含まない純粋な居住空間としての価値を表します
all_df['GrLivArea_Qual'] = all_df['GrLivArea'] * all_df['OverallQual']

# 3. 広さ × 築年数 (TotalSF * HouseAge)
# 「古い豪邸」と「新しい豪邸」の違いを表現します。
# 築年数が経つほど広さの価値がどう変わるか（減価償却的な意味）を捉えます
all_df['TotalSF_Age'] = all_df['TotalSF'] * all_df['HouseAge']

# --- おまけ（さらに余裕があれば）---
# 4. 全体品質 × エリアの質 (OverallQual * ExterQual)
# 家の中身(Overall)と外見(Exter)の両方が良い「真の高級住宅」を特定
all_df['Overall_Exter_Qual'] = all_df['OverallQual'] * all_df['ExterQual']


# --- 実装確認 ---
# 新しい変数が正しく作られているか、数値を確認
cols_to_check = ['TotalSF', 'OverallQual', 'TotalSF_Qual', 'TotalSF_Age']
print(all_df[cols_to_check].head())


# In[11]:


from sklearn.preprocessing import LabelEncoder
categories = all_df.columns[all_df.dtypes == "object"]
print(categories)


# In[12]:


for cat in categories:
    le = LabelEncoder()
    
    all_df[cat] = all_df[cat].fillna("missing") 
    le = le.fit(all_df[cat])
    all_df[cat] = le.transform(all_df[cat])
    all_df[cat] = all_df[cat].astype("category")


# In[13]:


# 評価開始

train_df_le = all_df[~all_df["SalePrice"].isnull()]
test_df_le = all_df[all_df["SalePrice"].isnull()]


# In[14]:


import lightgbm as lgb
from sklearn.model_selection import KFold
folds = 5
kf = KFold(n_splits=folds)


# In[15]:


lgbm_params = {
    "objective": "regression",
    "random_seed": 423,
    "learning_rate": 0.01,   # 【追加】ゆっくり丁寧に学習する
    "metric": "rmse",        # ログが見やすくなるよう明示
    "bagging_fraction": 0.8, # 【推奨】少しランダム性を入れて過学習を防ぐ
    "feature_fraction": 0.8  # 【推奨】同上
}
train_X = train_df_le.drop(["SalePrice", "Id"], axis=1)
train_Y = np.log1p(train_df_le["SalePrice"])


# In[16]:


from sklearn.metrics import mean_squared_error
models = []
rmses = []
oof = np.zeros(len(train_X))

for train_index, val_index in kf.split(train_X):
    X_train = train_X.iloc[train_index]
    X_valid = train_X.iloc[val_index]
    y_train = train_Y.iloc[train_index]
    y_valid = train_Y.iloc[val_index]
        
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_valid, y_valid, reference=lgb_train)    

    callbacks = [
        lgb.early_stopping(stopping_rounds=100, verbose=True),
        lgb.log_evaluation(period=100)
    ]
    
    model_lgb = lgb.train(lgbm_params, 
                          lgb_train, 
                          valid_sets=[lgb_eval], 
                          num_boost_round=5000,
                          callbacks=callbacks,
                         )    
    
    y_pred = model_lgb.predict(X_valid, num_iteration=model_lgb.best_iteration)
    tmp_rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
    print(tmp_rmse)    
              
    models.append(model_lgb)    
    rmses.append(tmp_rmse)
    oof[val_index] = y_pred 


# In[17]:


sum(rmses)/len(rmses)


# In[18]:


rmses


# In[19]:


oof


# In[20]:


models


# In[21]:


plt.figure(figsize=(6, 6))

# 散布図を描画（横軸：正解、縦軸：予測）
plt.scatter(train_Y, oof, alpha=0.5)

# 理想線（y=x）を赤で引く。この線上に点があるほど予測が完璧
plt.plot([train_Y.min(), train_Y.max()], [train_Y.min(), train_Y.max()], 'r--', lw=2)

plt.xlabel("Actual Price")
plt.ylabel("Predicted Price (OOF)")
plt.title("Actual vs Predicted")
plt.grid(True)
plt.show()


# In[22]:


# 1. 全モデルの重要度を収集して1つのDataFrameにまとめる
feature_imp_df = pd.DataFrame()

for i, model in enumerate(models):
    tmp_df = pd.DataFrame()
    tmp_df['feature'] = model.feature_name()
    # 'gain' は「その変数がどれだけ誤差を減らしたか」を表す推奨設定です
    tmp_df['importance'] = model.feature_importance(importance_type='gain')
    tmp_df['fold'] = i + 1
    feature_imp_df = pd.concat([feature_imp_df, tmp_df], axis=0)

# 2. 各変数の平均重要度を計算し、上位20個を抽出
best_features = feature_imp_df.groupby('feature')['importance'].mean().sort_values(ascending=False).head(20).index

# 3. 可視化（上位20個のみ）
plt.figure(figsize=(10, 8))
sns.barplot(
    data=feature_imp_df[feature_imp_df['feature'].isin(best_features)], 
    x='importance', 
    y='feature', 
    order=best_features, # 重要度順に並べる
    errorbar=None        # 黒い棒（信頼区間）は今回ノイズになるので消す
)
plt.title('Feature Importance (Average over Folds)')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.show()


# In[23]:


all_df.info()


# In[24]:


# テストデータの予測
# test_df_le は学習時と同じ加工（数値化・欠損補完）が終わっている前提です
test_X = test_df_le.drop(["SalePrice", "Id"], axis=1)

# モデルごとの予測値を平均する（アンサンブル）
preds = np.zeros(len(test_X))
for model in models:
    # 対数のまま予測
    pred_log = model.predict(test_X, num_iteration=model.best_iteration)
    # 元の金額に戻して加算
    preds += np.expm1(pred_log) / len(models)

# 提出用DataFrame作成
submission['SalePrice'] = preds
submission.to_csv('houseprice_submission.csv', index=False)
print("提出ファイルを作成しました！")


# In[25]:


from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.pipeline import make_pipeline
from scipy.stats import skew
# --- 1. Ridge用のデータ準備 ---
# LightGBMで使った all_df をコピーして使います
ridge_df = all_df.copy()

# 【最重要】答え(SalePrice)とIDを確実に消す
drop_cols = ['SalePrice', 'Id']
ridge_df = ridge_df.drop(drop_cols, axis=1, errors='ignore')

#【新規】歪んだ数値特徴量の対数変換 (ここが暴走を止める鍵！)
# 数値のカラムだけを取り出す

numeric_feats = ridge_df.select_dtypes(include=['number']).columns
# 歪度(skew)を計算
skewed_feats = ridge_df[numeric_feats].apply(lambda x: skew(x.dropna()))
# 歪みが大きい(0.75以上)変数を抽出
skewed_feats = skewed_feats[skewed_feats > 0.75]
skewed_feats = skewed_feats.index

# 対数変換を実行 (np.log1p)
ridge_df[skewed_feats] = np.log1p(ridge_df[skewed_feats])
print(f"対数変換しました: {len(skewed_feats)}個のカラム")

# カテゴリ変数を One-Hot Encoding (ダミー変数化) します
# pandasのget_dummiesを使うと、category型の列を自動で分解してくれます
ridge_df = pd.get_dummies(ridge_df, dummy_na=True)

# 欠損値の最終確認（Ridgeは欠損が1つでもあると動かないため）
# まだ埋まっていない数値の欠損があれば、平均値で埋めます
ridge_df = ridge_df.fillna(ridge_df.mean())

# 学習データとテストデータに再分割
X_train_ridge = ridge_df.iloc[:len(train_Y)]
X_test_ridge = ridge_df.iloc[len(train_Y):]

# --- 2. Ridgeモデルの学習 ---
# RobustScaler: 外れ値の影響を受けにくいスケーリングを行います
# RidgeCV: 最適な正則化パラメータ（alpha）を自動で見つけてくれます
ridge_model = make_pipeline(RobustScaler(), RidgeCV(alphas=[0.1, 1, 10, 50, 100]))

# 学習実行（目的変数は対数のまま）
ridge_model.fit(X_train_ridge, train_Y)

# 予測実行
ridge_pred_log = ridge_model.predict(X_test_ridge)
ridge_pred = np.expm1(ridge_pred_log) # 元の金額に戻す

print(f"Ridgeの学習完了! 選ばれたalpha値: {ridge_model.named_steps['ridgecv'].alpha_}")

# --- 3. アンサンブル（Blending） ---

# 割合を決めて合体
final_pred = 0.7 * preds + 0.3 * ridge_pred

# --- 4. 安全確認：モデル同士の相関を見る ---
# 2つのモデルが「全く違う視点」を持っているか確認します
plt.figure(figsize=(8, 8))
plt.scatter(preds, ridge_pred, alpha=0.5)
plt.plot([min(preds), max(preds)], [min(preds), max(preds)], 'r--')
plt.xlabel('LightGBM Predictions')
plt.ylabel('Ridge Predictions')
plt.title('Correlation between LightGBM and Ridge')
plt.show()

# --- 5. 提出ファイルの作成 ---
submission['SalePrice'] = final_pred
submission.to_csv('houseprice_submission_ensemble_v1.csv', index=False)
print("✅ アンサンブル提出ファイル 'submission_ensemble_v1.csv' が完成しました！")


# In[26]:


from sklearn.model_selection import KFold
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error

# --- 1. データの単位自動チェック & 修正 ---

# LightGBMのOOF (oof変数) の最大値をチェック
if oof.max() > 100:
    print("確認: LightGBMのOOFは既に「実際の金額」です。変換せずそのまま使います。")
    oof_lgbm_real = oof
else:
    print("確認: LightGBMのOOFは「対数」です。金額に変換します。")
    oof_lgbm_real = np.expm1(oof)

# 1. RidgeのOOF（学習データに対する予測）を作るための準備
# LightGBMと同じKFold設定を使います
kf = KFold(n_splits=5, shuffle=True, random_state=423) 
oof_ridge = np.zeros(len(train_X))

# 2. Ridgeの交差検証 (K-Fold)
# ※注: 全体学習ではなく、分割学習をしてOOFを作ります
print("Ridge OOF計算中...", end="")
for train_index, val_index in kf.split(X_train_ridge):
    X_tr = X_train_ridge.iloc[train_index]
    X_val = X_train_ridge.iloc[val_index]
    y_tr = train_Y.iloc[train_index]
    
    # モデル構築
    model = make_pipeline(RobustScaler(), RidgeCV(alphas=[0.1, 1, 10, 50, 100]))
    model.fit(X_tr, y_tr)
    
    # OOF予測値を格納
    pred_log = model.predict(X_val)
    oof_ridge[val_index] = np.expm1(pred_log) # 元の金額に戻す
print("完了！")

# RidgeのOOF (oof_ridge) の最大値をチェック
if oof_ridge.max() > 100:
    print("確認: RidgeのOOFは既に「実際の金額」です。変換せずそのまま使います。")
    oof_ridge_real = oof_ridge
else:
    print("確認: RidgeのOOFは「対数」です。金額に変換します。")
    oof_ridge_real = np.expm1(oof_ridge)

# 正解データ (train_Y) もチェック
if train_Y.max() > 100:
    y_true_real = train_Y
else:
    y_true_real = np.expm1(train_Y)

# 3. 最適な重みの探索 (0.00 〜 1.00 まで走査)
scores = []
weights = []

# 正解データ（元の金額）
y_true = np.expm1(train_Y)

for w in np.arange(0, 1.01, 0.01):
    # 重み付け平均
    blended_pred = (w * oof_lgbm_real) + ((1 - w) * oof_ridge_real)
    
    # RMSE計算
    score = np.sqrt(mean_squared_error(np.log1p(y_true), np.log1p(blended_pred)))
    scores.append(score)
    weights.append(w)

# 4. ベストな結果を表示
best_score = min(scores)
best_weight = weights[scores.index(best_score)]

print("-" * 30)
print(f"ベストスコア (RMSE): {best_score:.5f}")
print(f"最適な比率 -> LightGBM: {best_weight:.2f} / Ridge: {1.0 - best_weight:.2f}")

# 5. 結果の可視化
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 5))
plt.plot(weights, scores)
plt.xlabel("Weight for LightGBM")
plt.ylabel("RMSE Score")
plt.title("Optimal Blend Weight Search")
plt.axvline(x=best_weight, color='r', linestyle='--')
plt.show()


# In[27]:


# 最適比率（グラフから読み取った値）
best_w_lgbm = 0.31
best_w_ridge = 0.69

print(f"採用する比率 -> LightGBM: {best_w_lgbm} / Ridge: {best_w_ridge}")

# 最終ブレンド (LightGBMの予測値predsと、Ridgeの予測値ridge_predを使います)
# ※ ridge_pred は Ridge単体の予測値(np.expm1済み)が入っている前提です
final_pred = (best_w_lgbm * preds) + (best_w_ridge * ridge_pred)

# 提出
submission['SalePrice'] = final_pred
submission.to_csv('houseprice_submission_best_blend.csv', index=False)
print("✅ 最強のブレンド提出ファイルが完成しました！")

