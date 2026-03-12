"""
特征预处理与机器学习Pipeline
步骤3.2: 特征预处理（异常值处理、标准化）
步骤3.3: 机器学习模型训练
步骤3.4: 模型评估与特征筛选
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime

# 添加src到路径
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from config import RESULTS_DIR

def load_feature_matrix():
    """加载特征矩阵"""
    csv_path = os.path.join(RESULTS_DIR, 'feature_matrix.csv')
    df = pd.read_csv(csv_path)
    
    # 分离特征和标签
    feature_cols = [col for col in df.columns if col not in ['label', 'category', 'filename']]
    X = df[feature_cols].values
    y = df['label'].values
    
    return X, y, feature_cols, df

def preprocess_features(X, feature_names):
    """特征预处理"""
    print("\n" + "=" * 60)
    print("特征预处理")
    print("=" * 60)
    
    # 1. 处理异常值（使用百分位截断）
    print("\n1. 异常值处理...")
    X_processed = X.copy()
    
    for i, fname in enumerate(feature_names):
        col = X[:, i]
        p1, p99 = np.percentile(col, [1, 99])
        
        # 检测异常值
        has_outliers = np.any(col > p99 * 10) or np.any(col < p1 / 10)
        
        if has_outliers:
            # 使用log变换处理极大异常值
            if np.all(col >= 0):
                X_processed[:, i] = np.log1p(col)  # log(1+x)
            else:
                # 使用截断
                X_processed[:, i] = np.clip(col, p1, p99)
            print(f"  {fname}: 检测到异常值，已处理")
    
    # 2. 标准化
    print("\n2. 标准化...")
    from sklearn.preprocessing import StandardScaler, RobustScaler
    
    # 使用RobustScaler（对异常值更鲁棒）
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_processed)
    
    print(f"  原始特征范围: min={X.min():.2f}, max={X.max():.2f}")
    print(f"  标准化后范围: min={X_scaled.min():.2f}, max={X_scaled.max():.2f}")
    
    # 3. 计算相关性矩阵
    print("\n3. 特征相关性分析...")
    corr_matrix = np.corrcoef(X_scaled.T)
    
    # 找出高相关特征对
    high_corr_pairs = []
    for i in range(len(feature_names)):
        for j in range(i+1, len(feature_names)):
            if abs(corr_matrix[i, j]) > 0.9:
                high_corr_pairs.append((feature_names[i], feature_names[j], corr_matrix[i, j]))
    
    if high_corr_pairs:
        print(f"  发现 {len(high_corr_pairs)} 对高相关特征 (|r| > 0.9):")
        for f1, f2, r in high_corr_pairs:
            print(f"    {f1} <-> {f2}: r={r:.3f}")
    else:
        print("  未发现高相关特征对")
    
    return X_scaled, scaler, corr_matrix

def train_models(X, y, feature_names):
    """训练机器学习模型"""
    print("\n" + "=" * 60)
    print("机器学习模型训练")
    print("=" * 60)
    
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
    
    # 5折分层交叉验证
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # 定义模型
    models = {
        'SVM (RBF)': SVC(kernel='rbf', C=1.0, gamma='scale', class_weight='balanced', random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=5, class_weight='balanced', random_state=42),
        'Logistic Regression': LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000, random_state=42)
    }
    
    results = {}
    
    print(f"\n数据集: {len(y)} 样本 (篡改={sum(y==1)}, 正常={sum(y==0)})")
    print(f"特征数: {len(feature_names)}")
    
    for name, model in models.items():
        print(f"\n{'='*40}")
        print(f"模型: {name}")
        print(f"{'='*40}")
        
        # 交叉验证
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring='f1_weighted')
        
        print(f"5折交叉验证 F1-score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        print(f"各折分数: {[f'{s:.4f}' for s in cv_scores]}")
        
        # 在全数据上训练
        model.fit(X, y)
        
        # 预测
        y_pred = model.predict(X)
        
        # 计算指标
        acc = accuracy_score(y, y_pred)
        prec = precision_score(y, y_pred, average='weighted')
        rec = recall_score(y, y_pred, average='weighted')
        f1 = f1_score(y, y_pred, average='weighted')
        
        print(f"\n训练集性能:")
        print(f"  Accuracy:  {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall:    {rec:.4f}")
        print(f"  F1-score:  {f1:.4f}")
        
        # 混淆矩阵
        cm = confusion_matrix(y, y_pred)
        print(f"\n混淆矩阵:")
        print(f"          预测正常  预测篡改")
        print(f"  实际正常  {cm[0,0]:5d}    {cm[0,1]:5d}")
        print(f"  实际篡改  {cm[1,0]:5d}    {cm[1,1]:5d}")
        
        # 特征重要性（随机森林）
        if name == 'Random Forest':
            importance = model.feature_importances_
            indices = np.argsort(importance)[::-1]
            
            print(f"\n特征重要性 Top 10:")
            for i, idx in enumerate(indices[:10]):
                print(f"  {i+1}. {feature_names[idx]}: {importance[idx]:.4f}")
        
        results[name] = {
            'cv_f1_mean': float(cv_scores.mean()),
            'cv_f1_std': float(cv_scores.std()),
            'accuracy': float(acc),
            'precision': float(prec),
            'recall': float(rec),
            'f1_score': float(f1),
            'confusion_matrix': cm.tolist()
        }
    
    return results, models

def select_best_features(X, y, feature_names, k=15):
    """选择最优特征"""
    print("\n" + "=" * 60)
    print("特征选择")
    print("=" * 60)
    
    from sklearn.feature_selection import SelectKBest, f_classif
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.svm import SVC
    
    # 使用ANOVA F值选择特征
    selector = SelectKBest(f_classif, k=k)
    X_selected = selector.fit_transform(X, y)
    
    # 获取选中的特征
    selected_mask = selector.get_support()
    selected_features = [feature_names[i] for i in range(len(feature_names)) if selected_mask[i]]
    
    print(f"\n选择的 {k} 个特征:")
    for i, fname in enumerate(selected_features):
        score = selector.scores_[feature_names.index(fname)]
        print(f"  {i+1}. {fname}: F-score={score:.2f}")
    
    # 用选中特征训练模型
    print(f"\n用选中特征训练SVM...")
    model = SVC(kernel='rbf', C=1.0, gamma='scale', class_weight='balanced', random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    cv_scores = cross_val_score(model, X_selected, y, cv=cv, scoring='f1_weighted')
    
    print(f"5折交叉验证 F1-score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    return selected_features, X_selected

def main():
    """主函数"""
    print("=" * 60)
    print("图像篡改检测 - 机器学习Pipeline")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # 1. 加载特征矩阵
    X, y, feature_names, df = load_feature_matrix()
    print(f"\n加载特征矩阵: {X.shape}")
    print(f"标签分布: 篡改={sum(y==1)}, 正常={sum(y==0)}")
    
    # 2. 特征预处理
    X_scaled, scaler, corr_matrix = preprocess_features(X, feature_names)
    
    # 3. 训练模型
    results, models = train_models(X_scaled, y, feature_names)
    
    # 4. 特征选择
    selected_features, X_selected = select_best_features(X_scaled, y, feature_names, k=15)
    
    # 5. 保存结果
    print("\n" + "=" * 60)
    print("保存结果")
    print("=" * 60)
    
    # 保存模型性能
    results_path = os.path.join(RESULTS_DIR, 'model_results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'dataset': {
                'total_samples': len(y),
                'tampered': int(sum(y==1)),
                'normal': int(sum(y==0)),
                'num_features': len(feature_names)
            },
            'models': results,
            'selected_features': selected_features
        }, f, indent=2, ensure_ascii=False)
    print(f"模型结果已保存: {results_path}")
    
    # 找出最佳模型
    best_model = max(results.keys(), key=lambda k: results[k]['cv_f1_mean'])
    print(f"\n最佳模型: {best_model}")
    print(f"交叉验证 F1-score: {results[best_model]['cv_f1_mean']:.4f} ± {results[best_model]['cv_f1_std']:.4f}")
    
    return results

if __name__ == '__main__':
    from sklearn.model_selection import StratifiedKFold
    results = main()