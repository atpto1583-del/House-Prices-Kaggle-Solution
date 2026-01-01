# House Prices: Advanced Regression Techniques

Kaggleの「House Prices」コンペティションにおける取り組みの記録。
特徴量エンジニアリングと、異なる性質を持つモデル（LightGBM + Ridge Regression）のアンサンブル。

## 🏆 結果
* **Score (RMSE):** 0.11983
* **Ranking:** Top 15% 相当 (推定)

## 🛠 使用技術
* **Language:** Python 3.x
* **Libraries:** pandas, numpy, seaborn, scikit-learn, LightGBM
* **Key Techniques:**
    * **EDA & Preprocessing:** 外れ値の除去、対数変換による分布の正規化
    * **Feature Engineering:** 相互作用特徴量（Interaction Features）の作成
    * **Modeling:** LightGBM (Gradient Boosting) と Ridge Regression (Linear Model)
    * **Ensembling:** OOF (Out-of-Fold) 予測を用いた最適比率でのブレンディング

## 💡 工夫した点
1.  **特徴量の対数変換**
    * 線形モデル（Ridge）の性能を最大化するため、歪度が高い数値変数を`np.log1p`で変換。これによりRidgeの予測精度が向上。
2.  **ドメイン知識に基づく特徴量生成**
    * 「広さ × 品質 (`TotalSF * OverallQual`)」のような相互作用特徴量を作成。
3.  **アンサンブル**
    * LightGBMとRidgeの予測値の相関を確認し、OOFスコアに基づいて最適な重み付け（LightGBM: 31%, Ridge: 69%）を算出。

