# 图像篡改检测特征融合与机器学习方法调研报告

## 一、问题分析

### 1.1 当前挑战
- **数据集规模小**：当前仅有46张图像（easy:20, difficult:16, good:10），远不足以支撑深度学习
- **特征数量多**：24个特征（10个核心 + 14个变体），存在冗余和相关性
- **特征尺度不一致**：各特征的输出范围差异大（如ELA: 0-100，DCT: 0-1）
- **缺乏有机组合**：简单投票/加权效果不佳，需要更智能的融合策略

### 1.2 核心问题
如何在小样本条件下，将24个手工特征有效融合，构建高精度篡改检测模型？

---

## 二、特征融合策略

### 2.1 融合层级分类

| 层级 | 方法 | 优点 | 缺点 | 适用场景 |
|------|------|------|------|----------|
| **决策级** | 投票、加权投票 | 简单、可解释 | 未利用特征关联 | 特征独立时 |
| **特征级** | 特征拼接+分类器 | 利用特征交互 | 可能过拟合 | 样本充足时 |
| **混合级** | 分组融合 | 平衡性能与复杂度 | 需要领域知识 | 推荐 |

### 2.2 推荐融合架构

```
┌─────────────────────────────────────────────────────────────┐
│                     输入图像                                  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              24个特征提取器（并行）                            │
│  ┌──────┐ ┌──────┐ ┌──────┐     ┌──────┐                    │
│  │ ELA  │ │ DCT  │ │ CFA  │ ... │变体特征│                    │
│  └──┬───┘ └──┬───┘ └──┬───┘     └──┬───┘                    │
└─────┼────────┼────────┼────────────┼────────────────────────┘
      │        │        │            │
      ▼        ▼        ▼            ▼
┌─────────────────────────────────────────────────────────────┐
│              特征预处理（归一化）                              │
│         Min-Max / Z-Score / Robust Scaler                    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              特征选择（降维）                                  │
│         方差阈值 + 相关性过滤 + RFE                           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              集成分类器（核心）                                │
│                                                              │
│   ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐        │
│   │   SVM   │  │ Random  │  │   LR    │  │  XGBoost│        │
│   │(RBF核) │  │ Forest  │  │(L1正则)│  │         │        │
│   └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘        │
│        │            │            │            │              │
│        └────────────┼────────────┼────────────┘              │
│                     ▼            ▼                           │
│              ┌──────────────────────┐                        │
│              │   Stacking / Voting  │                        │
│              │      (Meta-Learner)  │                        │
│              └──────────┬───────────┘                        │
└─────────────────────────┼───────────────────────────────────┘
                          │
                          ▼
                   ┌──────────────┐
                   │  篡改/正常   │
                   └──────────────┘
```

---

## 三、推荐机器学习方法

### 3.1 基础分类器选择

针对小样本、高维特征问题，推荐以下分类器：

#### 方案A：SVM（支持向量机）★★★★★

**优势**：
- 专门针对小样本设计，泛化能力强
- 通过核函数可处理非线性问题
- 结构风险最小化，不易过拟合

**推荐配置**：
```python
SVM(kernel='rbf', 
    C=1.0,           # 正则化参数
    gamma='scale',   # RBF核参数
    class_weight='balanced',  # 处理类别不平衡
    probability=True)  # 输出概率
```

#### 方案B：随机森林（Random Forest）★★★★☆

**优势**：
- 集成学习，抗过拟合
- 内置特征重要性评估
- 无需特征缩放

**推荐配置**：
```python
RandomForestClassifier(
    n_estimators=100,    # 树的数量
    max_depth=5,         # 限制深度防止过拟合
    min_samples_split=5, # 小样本时调大
    min_samples_leaf=2,
    class_weight='balanced')
```

#### 方案C：逻辑回归（L1正则）★★★★☆

**优势**：
- L1正则化自动进行特征选择
- 可解释性强，输出权重直观
- 训练快速

**推荐配置**：
```python
LogisticRegression(
    penalty='l1',
    solver='saga',
    C=0.1,  # 较强的正则化
    class_weight='balanced')
```

### 3.2 集成学习方法

#### 方法1：Stacking（堆叠泛化）★★★★★

```python
from sklearn.ensemble import StackingClassifier

estimators = [
    ('svm', SVM(probability=True)),
    ('rf', RandomForestClassifier()),
    ('lr', LogisticRegression())
]

stacking = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(),  # 元学习器
    cv=5)  # 交叉验证折数
```

**原理**：多个基分类器的预测结果作为元特征，训练一个元分类器做最终决策。

#### 方法2：Voting（软投票）★★★★☆

```python
from sklearn.ensemble import VotingClassifier

voting = VotingClassifier(
    estimators=[
        ('svm', SVM(probability=True)),
        ('rf', RandomForestClassifier()),
        ('lr', LogisticRegression())
    ],
    voting='soft',  # 软投票：使用概率平均
    weights=[2, 1, 1])  # 可根据性能调整权重
```

**原理**：对多个分类器的预测概率加权平均，选择概率最高的类别。

### 3.3 特征选择方法

#### Step 1：方差阈值过滤
```python
from sklearn.feature_selection import VarianceThreshold
selector = VarianceThreshold(threshold=0.01)  # 去除方差<0.01的特征
```

#### Step 2：相关性过滤
```python
# 计算特征间相关系数矩阵
corr_matrix = np.corrcoef(X.T)
# 去除高度相关的特征（|r| > 0.9）
```

#### Step 3：递归特征消除（RFE）
```python
from sklearn.feature_selection import RFE
from sklearn.svm import SVC

rfe = RFE(estimator=SVC(kernel='linear'), 
          n_features_to_select=10)  # 选择最重要的10个特征
X_selected = rfe.fit_transform(X, y)
```

#### Step 4：基于模型的重要性
```python
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(X, y)
importance = rf.feature_importances_
```

---

## 四、小样本数据增强策略

### 4.1 数据增强技术

| 技术 | 方法 | 适用场景 |
|------|------|----------|
| **图像增强** | 旋转、翻转、裁剪、颜色抖动 | 原始图像 |
| **特征增强** | SMOTE、ADASYN | 特征空间 |
| **噪声注入** | 添加高斯噪声 | 特征空间 |

### 4.2 SMOTE过采样
```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
```

### 4.3 交叉验证策略

针对小样本，推荐使用：
- **分层K折交叉验证（Stratified K-Fold）**：保持各类别比例
- **留一交叉验证（LOOCV）**：每折只留一个样本做验证

```python
from sklearn.model_selection import StratifiedKFold, cross_val_score

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=cv, scoring='f1')
```

---

## 五、推荐实现方案

### 5.1 短期方案（当前数据集）

```python
# 1. 特征预处理
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_features)

# 2. 特征选择
from sklearn.feature_selection import SelectKBest, f_classif
selector = SelectKBest(f_classif, k=10)
X_selected = selector.fit_transform(X_scaled, y)

# 3. 集成分类器
from sklearn.ensemble import StackingClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

estimators = [
    ('svm', SVC(kernel='rbf', probability=True, class_weight='balanced')),
    ('rf', RandomForestClassifier(n_estimators=50, max_depth=3, class_weight='balanced')),
    ('lr', LogisticRegression(penalty='l2', class_weight='balanced'))
]

stacking = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(),
    cv=StratifiedKFold(n_splits=5))

# 4. 交叉验证评估
from sklearn.model_selection import cross_val_score
scores = cross_val_score(stacking, X_selected, y, cv=5, scoring='f1_weighted')
print(f"F1 Score: {scores.mean():.4f} ± {scores.std():.4f}")
```

### 5.2 中期方案（扩充数据集后）

```python
# 使用更复杂的集成策略
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

estimators = [
    ('xgb', XGBClassifier(max_depth=3, learning_rate=0.1, n_estimators=100)),
    ('lgbm', LGBMClassifier(max_depth=3, learning_rate=0.1, n_estimators=100)),
    ('svm', SVC(kernel='rbf', probability=True)),
    ('rf', RandomForestClassifier(n_estimators=100, max_depth=5))
]

stacking = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(),
    cv=5)
```

### 5.3 自动化特征工程

```python
# 使用TPOT自动搜索最优pipeline
from tpot import TPOTClassifier

tpot = TPOTClassifier(
    generations=10,
    population_size=20,
    cv=5,
    scoring='f1_weighted',
    max_time_mins=30,
    verbosity=2)

tpot.fit(X_train, y_train)
print(tpot.fitted_pipeline_)
```

---

## 六、评估指标

### 6.1 推荐指标

| 指标 | 公式 | 说明 |
|------|------|------|
| **F1-Score** | 2×P×R/(P+R) | 精确率和召回率的调和平均 |
| **AUC-ROC** | - | 分类器整体性能 |
| **混淆矩阵** | - | 详细展示各类别表现 |

### 6.2 评估代码
```python
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred))
print("AUC:", roc_auc_score(y_test, y_prob))
print(confusion_matrix(y_test, y_pred))
```

---

## 七、实验建议

### 7.1 实验流程

1. **特征提取**：对每张图像提取24个特征的分数值
2. **特征矩阵构建**：N×24的矩阵（N为样本数）
3. **预处理**：标准化（Z-Score）
4. **特征选择**：选择10-15个最有效特征
5. **模型训练**：Stacking集成
6. **交叉验证**：5折分层交叉验证
7. **结果评估**：F1、AUC、混淆矩阵

### 7.2 预期改进

| 当前方法 | 改进后 | 预期提升 |
|----------|--------|----------|
| 单特征阈值判断 | 多特征融合 | +20-30% F1 |
| 简单投票 | Stacking集成 | +10-15% F1 |
| 无特征选择 | RFE选择 | +5-10% F1 |

---

## 八、总结

### 推荐方案

1. **特征融合**：特征级融合（拼接） + 决策级融合（Stacking）混合
2. **分类器**：SVM + Random Forest + Logistic Regression 的 Stacking 集成
3. **特征选择**：方差过滤 → 相关性过滤 → RFE
4. **评估**：5折分层交叉验证 + F1-Score
5. **数据增强**：等待扩充数据集，可先用SMOTE

### 下一步行动

1. 实现特征提取，构建N×24特征矩阵
2. 实现上述推荐Pipeline
3. 等待数据扩充后，重新训练并优化

---

**报告编写时间**: 2026-03-12
**作者**: 灰（OpenClaw Assistant）