# پروژه پیش‌بینی عملکرد دانش‌آموزان
# فاز 2: مدل‌های طبقه‌بندی (Classification Models)

# وارد کردن کتابخانه‌های مورد نیاز
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings

warnings.filterwarnings('ignore')

# کتابخانه‌های مربوط به مدل‌سازی
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, classification_report, roc_curve, auc,
                             roc_auc_score)

# الگوریتم‌های طبقه‌بندی
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

# برای نمایش درخت تصمیم
from sklearn.tree import plot_tree

# تنظیمات نمایش
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.figsize'] = (12, 8)

print("=" * 80)
print("🎯 فاز 2: مدل‌های طبقه‌بندی (Classification Models)")
print("=" * 80)

# ========================
# 1. بارگذاری داده‌های پردازش شده
# ========================
print("\n" + "=" * 60)
print("مرحله 1: بارگذاری داده‌های پردازش شده از فاز 1")
print("=" * 60)

# بارگذاری داده‌های آموزشی و تست
train_data = pd.read_csv('train_data.csv')
test_data = pd.read_csv('test_data.csv')

# بارگذاری اطلاعات پیش‌پردازش
with open('preprocessing_info.pkl', 'rb') as f:
    preprocessing_info = pickle.load(f)

feature_columns = preprocessing_info['feature_columns']

# جداسازی ویژگی‌ها و برچسب‌ها
X_train = train_data[feature_columns]
y_train = train_data['Pass_Status_Encoded']
X_test = test_data[feature_columns]
y_test = test_data['Pass_Status_Encoded']

print(f"\n✅ داده‌ها با موفقیت بارگذاری شدند:")
print(f"  • تعداد نمونه‌های آموزشی: {X_train.shape[0]}")
print(f"  • تعداد نمونه‌های تست: {X_test.shape[0]}")
print(f"  • تعداد ویژگی‌ها: {X_train.shape[1]}")

# ========================
# 2. تعریف مدل‌های طبقه‌بندی
# ========================
print("\n" + "=" * 60)
print("مرحله 2: تعریف و پیکربندی مدل‌های طبقه‌بندی")
print("=" * 60)

# دیکشنری برای ذخیره مدل‌ها
models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42, n_estimators=100),
    'SVM (Linear)': SVC(kernel='linear', random_state=42, probability=True),
    'SVM (RBF)': SVC(kernel='rbf', random_state=42, probability=True),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Naive Bayes': GaussianNB(),
    'Neural Network': MLPClassifier(random_state=42, max_iter=1000)
}

print("\n🤖 مدل‌های انتخاب شده:")
for i, model_name in enumerate(models.keys(), 1):
    print(f"  {i}. {model_name}")

# ========================
# 3. آموزش مدل‌ها و ارزیابی با Cross-Validation
# ========================
print("\n" + "=" * 60)
print("مرحله 3: آموزش مدل‌ها و ارزیابی با Cross-Validation")
print("=" * 60)

# تعریف StratifiedKFold برای Cross-Validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# دیکشنری برای ذخیره نتایج
results = {}
trained_models = {}

print("\n📊 در حال آموزش و ارزیابی مدل‌ها...")
print("─" * 50)

for model_name, model in models.items():
    print(f"\n🔄 آموزش مدل: {model_name}")

    # آموزش مدل
    model.fit(X_train, y_train)
    trained_models[model_name] = model

    # پیش‌بینی روی داده‌های تست
    y_pred = model.predict(X_test)

    # محاسبه معیارهای ارزیابی
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    # Cross-Validation scores
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()

    # محاسبه AUC-ROC برای مدل‌هایی که probability دارند
    try:
        y_proba = model.predict_proba(X_test)[:, 1]
        auc_score = roc_auc_score(y_test, y_proba)
    except:
        auc_score = None

    # ذخیره نتایج
    results[model_name] = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'cv_mean': cv_mean,
        'cv_std': cv_std,
        'auc': auc_score,
        'y_pred': y_pred,
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }

    # نمایش نتایج
    print(f"  ✓ Accuracy: {accuracy:.4f}")
    print(f"  ✓ Precision: {precision:.4f}")
    print(f"  ✓ Recall: {recall:.4f}")
    print(f"  ✓ F1-Score: {f1:.4f}")
    print(f"  ✓ Cross-Val Mean: {cv_mean:.4f} (±{cv_std:.4f})")
    if auc_score:
        print(f"  ✓ AUC-ROC: {auc_score:.4f}")

# ========================
# 4. مقایسه عملکرد مدل‌ها
# ========================
print("\n" + "=" * 60)
print("مرحله 4: مقایسه عملکرد مدل‌ها")
print("=" * 60)

# ایجاد DataFrame برای مقایسه
comparison_data = []
for model_name, metrics in results.items():
    comparison_data.append({
        'Model': model_name,
        'Accuracy': metrics['accuracy'],
        'Precision': metrics['precision'],
        'Recall': metrics['recall'],
        'F1-Score': metrics['f1_score'],
        'CV Mean': metrics['cv_mean'],
        'CV Std': metrics['cv_std'],
        'AUC': metrics['auc'] if metrics['auc'] else 0
    })

comparison_df = pd.DataFrame(comparison_data)
comparison_df = comparison_df.sort_values('F1-Score', ascending=False)

print("\n📊 جدول مقایسه عملکرد مدل‌ها (مرتب شده بر اساس F1-Score):")
print("─" * 80)
print(comparison_df.to_string(index=False))

# ذخیره جدول مقایسه
comparison_df.to_csv('model_comparison.csv', index=False, encoding='utf-8-sig')
print("\n✅ جدول مقایسه در فایل 'model_comparison.csv' ذخیره شد.")

# ========================
# 5. انتخاب بهترین مدل
# ========================
print("\n" + "=" * 60)
print("مرحله 5: انتخاب بهترین مدل")
print("=" * 60)

# انتخاب بهترین مدل بر اساس F1-Score
best_model_name = comparison_df.iloc[0]['Model']
best_model = trained_models[best_model_name]
best_metrics = results[best_model_name]

print(f"\n🏆 بهترین مدل: {best_model_name}")
print("─" * 40)
print(f"  • Accuracy: {best_metrics['accuracy']:.4f}")
print(f"  • Precision: {best_metrics['precision']:.4f}")
print(f"  • Recall: {best_metrics['recall']:.4f}")
print(f"  • F1-Score: {best_metrics['f1_score']:.4f}")
print(f"  • Cross-Validation: {best_metrics['cv_mean']:.4f} (±{best_metrics['cv_std']:.4f})")

# ========================
# 6. تحلیل دقیق بهترین مدل
# ========================
print("\n" + "=" * 60)
print("مرحله 6: تحلیل دقیق بهترین مدل")
print("=" * 60)

# Confusion Matrix
print(f"\n📊 Confusion Matrix برای {best_model_name}:")
print(best_metrics['confusion_matrix'])

# Classification Report
y_pred_best = best_metrics['y_pred']
print(f"\n📋 Classification Report:")
print(classification_report(y_test, y_pred_best,
                            target_names=['رد (Fail)', 'قبول (Pass)']))

# ========================
# 7. ایجاد نمودارهای مقایسه
# ========================
print("\n" + "=" * 60)
print("مرحله 7: ایجاد نمودارهای تحلیلی")
print("=" * 60)

# نمودار مقایسه معیارهای ارزیابی
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. نمودار Accuracy
ax1 = axes[0, 0]
models_list = comparison_df['Model'].tolist()
accuracies = comparison_df['Accuracy'].tolist()
bars1 = ax1.bar(range(len(models_list)), accuracies, color='skyblue')
ax1.set_xlabel('Models')
ax1.set_ylabel('Accuracy')
ax1.set_title('Accuracy Comparison')
ax1.set_xticks(range(len(models_list)))
ax1.set_xticklabels(models_list, rotation=45, ha='right')
ax1.grid(True, alpha=0.3)

# اضافه کردن مقادیر روی نمودار
for i, v in enumerate(accuracies):
    ax1.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')

# 2. نمودار F1-Score
ax2 = axes[0, 1]
f1_scores = comparison_df['F1-Score'].tolist()
bars2 = ax2.bar(range(len(models_list)), f1_scores, color='lightgreen')
ax2.set_xlabel('Models')
ax2.set_ylabel('F1-Score')
ax2.set_title('F1-Score Comparison')
ax2.set_xticks(range(len(models_list)))
ax2.set_xticklabels(models_list, rotation=45, ha='right')
ax2.grid(True, alpha=0.3)

for i, v in enumerate(f1_scores):
    ax2.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')

# 3. نمودار مقایسه همه معیارها
ax3 = axes[1, 0]
metrics_comparison = comparison_df[['Accuracy', 'Precision', 'Recall', 'F1-Score']].values
x = np.arange(len(models_list))
width = 0.2

for i, metric in enumerate(['Accuracy', 'Precision', 'Recall', 'F1-Score']):
    offset = width * (i - 1.5)
    ax3.bar(x + offset, metrics_comparison[:, i], width, label=metric)

ax3.set_xlabel('Models')
ax3.set_ylabel('Score')
ax3.set_title('All Metrics Comparison')
ax3.set_xticks(x)
ax3.set_xticklabels(models_list, rotation=45, ha='right')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. نمودار Cross-Validation Mean
ax4 = axes[1, 1]
cv_means = comparison_df['CV Mean'].tolist()
cv_stds = comparison_df['CV Std'].tolist()
bars4 = ax4.bar(range(len(models_list)), cv_means, yerr=cv_stds,
                color='coral', capsize=5)
ax4.set_xlabel('Models')
ax4.set_ylabel('Cross-Validation Score')
ax4.set_title('Cross-Validation Mean Score with Std Dev')
ax4.set_xticks(range(len(models_list)))
ax4.set_xticklabels(models_list, rotation=45, ha='right')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('model_comparison_charts.png', dpi=100, bbox_inches='tight')
print("✅ نمودار مقایسه مدل‌ها در 'model_comparison_charts.png' ذخیره شد.")
plt.close()

# ========================
# 8. Confusion Matrix Heatmap برای تمام مدل‌ها
# ========================
print("\n📊 ایجاد Confusion Matrix برای تمام مدل‌ها...")

fig, axes = plt.subplots(3, 3, figsize=(15, 15))
axes = axes.ravel()

for idx, (model_name, metrics) in enumerate(results.items()):
    if idx < 9:
        cm = metrics['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    ax=axes[idx], cbar=False)
        axes[idx].set_title(f'{model_name}')
        axes[idx].set_xlabel('Predicted')
        axes[idx].set_ylabel('Actual')
        axes[idx].set_xticklabels(['Fail', 'Pass'])
        axes[idx].set_yticklabels(['Fail', 'Pass'])

plt.suptitle('Confusion Matrices for All Models', fontsize=16)
plt.tight_layout()
plt.savefig('confusion_matrices_all.png', dpi=100, bbox_inches='tight')
print("✅ Confusion Matrices در 'confusion_matrices_all.png' ذخیره شد.")
plt.close()

# ========================
# 9. ROC Curves
# ========================
print("\n📊 ایجاد منحنی‌های ROC...")

plt.figure(figsize=(10, 8))

for model_name, model in trained_models.items():
    try:
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc_score = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_score:.3f})')
    except:
        pass

plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for All Models')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.savefig('roc_curves.png', dpi=100, bbox_inches='tight')
print("✅ منحنی‌های ROC در 'roc_curves.png' ذخیره شد.")
plt.close()

# ========================
# 10. Feature Importance (برای مدل‌های Tree-based)
# ========================
print("\n" + "=" * 60)
print("مرحله 8: تحلیل اهمیت ویژگی‌ها")
print("=" * 60)

# اگر بهترین مدل Tree-based باشد
if best_model_name in ['Decision Tree', 'Random Forest', 'Gradient Boosting']:
    feature_importance = best_model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': feature_columns,
        'Importance': feature_importance
    }).sort_values('Importance', ascending=False)

    print(f"\n📊 اهمیت ویژگی‌ها برای {best_model_name}:")
    print(feature_importance_df.to_string(index=False))

    # نمودار اهمیت ویژگی‌ها
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
    plt.xlabel('Importance')
    plt.title(f'Feature Importance - {best_model_name}')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=100, bbox_inches='tight')
    print("\n✅ نمودار اهمیت ویژگی‌ها در 'feature_importance.png' ذخیره شد.")
    plt.close()

# ========================
# 11. Hyperparameter Tuning برای بهترین مدل
# ========================
print("\n" + "=" * 60)
print("مرحله 9: بهینه‌سازی پارامترهای بهترین مدل")
print("=" * 60)

print(f"\n🔧 بهینه‌سازی پارامترهای {best_model_name}...")

# تعریف پارامترها برای Grid Search
param_grids = {
    'Decision Tree': {
        'max_depth': [3, 5, 7, 10, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    'Random Forest': {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    },
    'Logistic Regression': {
        'C': [0.01, 0.1, 1, 10, 100],
        'penalty': ['l2'],
        'solver': ['lbfgs', 'liblinear']
    },
    'SVM (RBF)': {
        'C': [0.1, 1, 10],
        'gamma': ['scale', 'auto', 0.001, 0.01]
    }
}

if best_model_name in param_grids:
    param_grid = param_grids[best_model_name]

    # انجام Grid Search
    grid_search = GridSearchCV(
        estimator=models[best_model_name],
        param_grid=param_grid,
        cv=cv,
        scoring='f1_weighted',
        n_jobs=-1,
        verbose=0
    )

    grid_search.fit(X_train, y_train)

    print(f"\n✅ بهترین پارامترها:")
    for param, value in grid_search.best_params_.items():
        print(f"  • {param}: {value}")

    # ارزیابی مدل بهینه شده
    y_pred_optimized = grid_search.predict(X_test)

    accuracy_opt = accuracy_score(y_test, y_pred_optimized)
    precision_opt = precision_score(y_test, y_pred_optimized, average='weighted')
    recall_opt = recall_score(y_test, y_pred_optimized, average='weighted')
    f1_opt = f1_score(y_test, y_pred_optimized, average='weighted')

    print(f"\n📊 عملکرد مدل بهینه‌شده:")
    print(f"  • Accuracy: {accuracy_opt:.4f} (قبلی: {best_metrics['accuracy']:.4f})")
    print(f"  • Precision: {precision_opt:.4f} (قبلی: {best_metrics['precision']:.4f})")
    print(f"  • Recall: {recall_opt:.4f} (قبلی: {best_metrics['recall']:.4f})")
    print(f"  • F1-Score: {f1_opt:.4f} (قبلی: {best_metrics['f1_score']:.4f})")

    # ذخیره مدل بهینه‌شده
    best_model_optimized = grid_search.best_estimator_
else:
    best_model_optimized = best_model

# ========================
# 12. ذخیره بهترین مدل
# ========================
print("\n" + "=" * 60)
print("مرحله 10: ذخیره مدل‌ها و نتایج")
print("=" * 60)

# ذخیره بهترین مدل
with open('best_model.pkl', 'wb') as f:
    pickle.dump({
        'model': best_model_optimized,
        'model_name': best_model_name,
        'metrics': best_metrics,
        'feature_columns': feature_columns
    }, f)

print(f"\n✅ بهترین مدل ({best_model_name}) در فایل 'best_model.pkl' ذخیره شد.")

# ذخیره همه مدل‌ها
with open('all_models.pkl', 'wb') as f:
    pickle.dump(trained_models, f)

print("✅ تمام مدل‌های آموزش‌دیده در فایل 'all_models.pkl' ذخیره شدند.")

# ذخیره نتایج تفصیلی
with open('classification_results.pkl', 'wb') as f:
    pickle.dump(results, f)

print("✅ نتایج تفصیلی در فایل 'classification_results.pkl' ذخیره شد.")

# ========================
# 13. گزارش نهایی
# ========================
print("\n" + "=" * 80)
print("✅ فاز 2: مدل‌های طبقه‌بندی با موفقیت تکمیل شد!")
print("=" * 80)

print("\n📋 خلاصه عملیات انجام شده:")
print("─" * 40)
print("1. ✓ بارگذاری داده‌های پردازش شده")
print("2. ✓ آموزش 9 مدل طبقه‌بندی مختلف")
print("3. ✓ ارزیابی با Cross-Validation (5-Fold)")
print("4. ✓ محاسبه معیارهای Accuracy, Precision, Recall, F1-Score")
print("5. ✓ مقایسه عملکرد مدل‌ها")
print(f"6. ✓ انتخاب بهترین مدل: {best_model_name}")
print("7. ✓ بهینه‌سازی پارامترهای بهترین مدل")
print("8. ✓ تحلیل Feature Importance")
print("9. ✓ رسم نمودارهای تحلیلی")

print("\n📁 فایل‌های تولید شده:")
print("─" * 40)
print("  • model_comparison.csv - جدول مقایسه مدل‌ها")
print("  • best_model.pkl - بهترین مدل ذخیره شده")
print("  • all_models.pkl - تمام مدل‌های آموزش‌دیده")
print("  • classification_results.pkl - نتایج تفصیلی")
print("  • model_comparison_charts.png - نمودارهای مقایسه")
print("  • confusion_matrices_all.png - ماتریس‌های درهم‌ریختگی")
print("  • roc_curves.png - منحنی‌های ROC")
print("  • feature_importance.png - اهمیت ویژگی‌ها")

print("\n🏆 نتیجه نهایی:")
print("─" * 40)
print(f"بهترین مدل: {best_model_name}")
print(f"F1-Score: {best_metrics['f1_score']:.4f}")
print(f"Accuracy: {best_metrics['accuracy']:.4f}")

print("\n🎯 آماده برای فازهای بعدی:")
print("─" * 40)
print("  → Clustering (خوشه‌بندی)")
print("  → Association Rules (قوانین همبستگی)")

print("\n" + "=" * 80)
print("پایان فاز 2 - Classification Models 🚀")
print("=" * 80)