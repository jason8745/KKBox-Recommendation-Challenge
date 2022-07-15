# KKBox-Recommendation-Challenge

## Task

預測使用者在第一次聽到一首歌後，重複聽歌的機率(一個月內)

## EDA

資料探索的部分，可以直接參考**[KKBOX 歌曲推薦系統](https://wenwender.wordpress.com/2019/04/03/kkbox-%e6%ad%8c%e6%9b%b2%e6%8e%a8%e8%96%a6%e7%b3%bb%e7%b5%b1/?preview_id=180&preview_nonce=c112e18087&preview=true)**

## Train

本次Task，使用XGBoost 做為底層預測模型，核心是**Tree加上Boosting**，是一個ensemble learning

**What is Ensemble Learning?**

很直白的理解就是，一個分類器不夠，就用多個分類器!!!

而常見的Ensemble有三種手法，Bagging、Stacking以集XGBoot使用的Boosting。

**Bagging**

用在很強的 model上(比如Decision Tree)，來避免Overfitting

透過Bootstrap的手法，從資料中抽樣(取後放回)，來建立多個分類器，**核心概念在資料的抽樣，**

透過這個方式，可以避免模型被noise影響，降低模型的不穩定性。

**對 Decision Tree 做 Bagging 就是 Random Forest**

**Stacking**

首先將Train data 切成兩份，一份拿來訓練弱的models，另一份拿來訓練meta-model。

這裡meta-model的input就是前面訓練好的一群model的output，透過meta-model再決定最後的輸出。

**Boosting(sequential)**

Boosting 用在弱的 model上，假設你有一個 Model 錯誤率高過 50% 只要能夠做到這件事情，

boosting 可以把這些錯誤率值略高於 50% 的Model 降到 0%，**核心概念在找到互補的模型。**

**How to obtain different classifiers?**

- Training on different dataset(re-sampling、re-weighting)
    
    範例: AdaBoost，透過將舊分類器的**錯誤資料權重提高**，然後再訓練新的分類器，這樣新的分類器就會學習到錯誤分類資料(misclassified data)的特性
    

**XGBoost優點**

- **正則化:** XGBoost透過正則化的方式(L1、L2)，降低了Overfitting的機率
- 平**行處理:** Boosting的原理是需要多個分類器彼此互補，因此後面訓練的分類器會根據前一次訓練的不足來補強(改變錯誤分類資料的權重)，因此訓練上有**序列關係**(前一次的訓練結果影響下一個)。
XGBoost在訓練前，將資料進行了排序，讓模型可以更快速找到最佳分裂點(樹模型的特性)。在進行節點的分裂時，需要計算每個特徵的增益，最終選增益最大的那個特徵去做分裂，那麼各個特徵的增益計算就可以開多執行緒進行。
我自己的理解是，XGBoost的平行處理，指的是節點分裂時，將特徵的增益計算視為彼此獨立的問題，因此**在特徵計算上可以平行處理**，但實際上還是遵循Boosting類型的序列訓練方式。
- **靈活性:** 可自行定義目標函數和評估函數
- **剪枝:** 從底到頂反向進行剪枝，可避免陷入區域性最佳解

**XGBoost詳細參數說明**

以此Project使用的XGBClassifier為例

![Untitled](KKBox-Recommendation-Challenge%2053542f8a624a4b61801ae68959e993b6/Untitled.png)

主要分成三個部分

- **general parameters**: 最重要的是booster的選擇(default= gbtree)
- **booster parameters** : booster的相關參數設定
- **task parameters**: 主要是跟模型的評估(loss、metrics)等和訓練任務的目標有關

以下只選擇較重要的變數介紹

**general parameters**

- `booster` : booster選擇
- `nthread`: 執行續的使用數量，default不調整的話是預設使用最大的thread

**booster parameters**

- `learning_rate`: 學習率，收縮步長
- `gamma`: min_split_loss，在節點分裂(樹),只有損失函數的值下降到gamma指定的閥值，才會分裂該節點。gamma值越大，算法會越不容易overfitting，但過大也會造成underfitting
- `min_child_weight`: 決定最小葉子節點的樣本權重和，低於設定的weight，就不會再產生新的葉子節點。當值較大時，可以避免學習到局部的特殊情況(overfitting)，但值若設置過高，則容易underfitting
- `subsample`: 控制每顆樹雖機採樣的比例。降低此參數，會避免overfitting，但若設置的過小，則會造成underfitting
- `lambda`: 是L2正則化的權重係數
- `alpha`: L1正則化的權重係數
- `n_estimators`: 因為XGBoost是一個Boosting的ensemble model，因此此參數是來設定，ensemble model裡要集成幾個model。

**task parameters**

- `object`: 任務類型/目標函數 (ex: `reg:squarederror` )
- `base_score`: 初始分數設定
- `eval_metric`: Evaluation metrics for validation data(ex: `rmse、mae、auc`)
- `seed`: 相同的種子可以鎖定隨機的結果，調參數時可以鎖定此參數，來使每次實驗具有較一致的比較基準

註: 上述的booster parameter是category類型的booster，若是使用linear類型(****`booster=gblinear`****)的booster，變數就會有其他的設定。

## Result

```python
               precision    recall  f1-score   support

           0       0.70      0.66      0.68    732669
           1       0.68      0.71      0.70    742815

   micro avg       0.69      0.69      0.69   1475484
   macro avg       0.69      0.69      0.69   1475484
weighted avg       0.69      0.69      0.69   1475484
```

## References

[Kaggle 資料來源](https://www.kaggle.com/c/kkbox-music-recommendation-challenge/data)

****[XGBoost Parameters](https://xgboost.readthedocs.io/en/latest/parameter.html#general-parameters)****

****[XGBOOST從原理到實戰：二分類 、多分類](https://codertw.com/%E7%A8%8B%E5%BC%8F%E8%AA%9E%E8%A8%80/519169/)****