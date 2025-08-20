# پروژه پیش‌بینی عملکرد دانش‌آموزان
# فاز 5: ارزیابی و مقایسه مدل‌ها (Model Evaluation and Comparison)

# وارد کردن کتابخانه‌های مورد نیاز
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

# کتابخانه‌های آماری و ارزیابی
from scipy import stats
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    silhouette_score, davies_bouldin_score, calinski_harabasz_score,
    mean_absolute_error, mean_squared_error, r2_score
)
from sklearn.model_selection import cross_val_score
import scipy.stats as st

# تنظیمات نمایش
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10
sns.set_style("whitegrid")
sns.set_palette("husl")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.precision', 4)

print("=" * 100)
print(" " * 30 + "🎯 فاز 5: ارزیابی و مقایسه مدل‌ها")
print(" " * 25 + "(Model Evaluation and Comparison)")
print("=" * 100)
print(f"\n📅 تاریخ اجرا: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ========================
# 1. بارگذاری نتایج فازهای قبلی
# ========================
print("\n" + "=" * 80)
print("📂 مرحله 1: بارگذاری نتایج فازهای قبلی")
print("=" * 80)

# دیکشنری برای ذخیره همه نتایج
all_results = {}

# 1.1 بارگذاری نتایج Classification (فاز 2)
try:
    with open('classification_results.pkl', 'rb') as f:
        classification_results = pickle.load(f)
    print("✅ نتایج Classification بارگذاری شد")
    all_results['classification'] = classification_results

    # بارگذاری مدل‌های آموزش‌دیده
    with open('all_models.pkl', 'rb') as f:
        trained_models = pickle.load(f)
    print("✅ مدل‌های آموزش‌دیده بارگذاری شدند")
except:
    print("⚠️ خطا در بارگذاری نتایج Classification")
    classification_results = {}
    trained_models = {}

# 1.2 بارگذاری نتایج Clustering (فاز 3)
try:
    with open('clustering_results.pkl', 'rb') as f:
        clustering_results = pickle.load(f)
    print("✅ نتایج Clustering بارگذاری شد")
    all_results['clustering'] = clustering_results
except:
    print("⚠️ خطا در بارگذاری نتایج Clustering")
    clustering_results = {}

# 1.3 بارگذاری نتایج Association Rules (فاز 4)
try:
    with open('association_rules_summary.pkl', 'rb') as f:
        association_results = pickle.load(f)
    print("✅ نتایج Association Rules بارگذاری شد")
    all_results['association'] = association_results

    # بارگذاری قوانین کامل
    rules_df = pd.read_csv('association_rules_all.csv')
    print(f"✅ {len(rules_df)} قانون همبستگی بارگذاری شد")
except:
    print("⚠️ خطا در بارگذاری نتایج Association Rules")
    association_results = {}
    rules_df = pd.DataFrame()

# 1.4 بارگذاری داده‌های اصلی
try:
    df_processed = pd.read_csv('processed_student_data.csv')
    df_with_clusters = pd.read_csv('data_with_clusters.csv')
    print(f"✅ داده‌های پردازش شده بارگذاری شد ({len(df_processed)} رکورد)")
except:
    print("⚠️ خطا در بارگذاری داده‌های اصلی")

print(f"\n📊 خلاصه داده‌های بارگذاری شده:")
print(f"  • مدل‌های Classification: {len(classification_results)} مدل")
print(f"  • الگوریتم‌های Clustering: {len(clustering_results.get('algorithms', {}))} الگوریتم")
print(f"  • قوانین Association: {association_results.get('total_rules', 0)} قانون")

# ========================
# 2. ارزیابی جامع مدل‌های Classification
# ========================
print("\n" + "=" * 80)
print("📊 مرحله 2: ارزیابی جامع مدل‌های Classification")
print("=" * 80)

# ایجاد DataFrame برای مقایسه
classification_comparison = []

for model_name, metrics in classification_results.items():
    comparison_entry = {
        'Model': model_name,
        'Accuracy': metrics.get('accuracy', 0),
        'Precision': metrics.get('precision', 0),
        'Recall': metrics.get('recall', 0),
        'F1-Score': metrics.get('f1_score', 0),
        'CV Mean': metrics.get('cv_mean', 0),
        'CV Std': metrics.get('cv_std', 0),
        'AUC': metrics.get('auc', 0)
    }
    classification_comparison.append(comparison_entry)

df_classification = pd.DataFrame(classification_comparison)
df_classification = df_classification.sort_values('F1-Score', ascending=False)

print("\n📈 جدول مقایسه مدل‌های Classification:")
print("=" * 90)
print(df_classification.to_string(index=False))
print("=" * 90)

# محاسبه آمار توصیفی
print("\n📊 آمار توصیفی معیارهای ارزیابی:")
print("─" * 50)
stats_summary = df_classification[['Accuracy', 'Precision', 'Recall', 'F1-Score']].describe()
print(stats_summary.round(4))

# شناسایی بهترین مدل
best_model = df_classification.iloc[0]
print(f"\n🏆 بهترین مدل: {best_model['Model']}")
print(f"  • F1-Score: {best_model['F1-Score']:.4f}")
print(f"  • Accuracy: {best_model['Accuracy']:.4f}")
print(f"  • AUC: {best_model['AUC']:.4f}")

# ========================
# 3. ارزیابی کیفیت خوشه‌بندی
# ========================
print("\n" + "=" * 80)
print("📊 مرحله 3: ارزیابی کیفیت خوشه‌بندی")
print("=" * 80)

if 'algorithms' in clustering_results:
    clustering_comparison = []

    for algo_name, results in clustering_results['algorithms'].items():
        comparison_entry = {
            'Algorithm': algo_name,
            'Silhouette Score': results.get('silhouette', 0),
            'Davies-Bouldin': results.get('davies_bouldin', 0),
            'Calinski-Harabasz': results.get('calinski_harabasz', 0)
        }

        # اضافه کردن اطلاعات خاص DBSCAN
        if algo_name == 'DBSCAN':
            comparison_entry['Clusters'] = results.get('n_clusters', 0)
            comparison_entry['Noise Points'] = results.get('n_noise', 0)

        clustering_comparison.append(comparison_entry)

    df_clustering = pd.DataFrame(clustering_comparison)
    df_clustering = df_clustering.sort_values('Silhouette Score', ascending=False)

    print("\n📈 جدول مقایسه الگوریتم‌های Clustering:")
    print("=" * 80)
    print(df_clustering.to_string(index=False))
    print("=" * 80)

    # بهترین الگوریتم خوشه‌بندی
    best_clustering = df_clustering.iloc[0]
    print(f"\n🏆 بهترین الگوریتم خوشه‌بندی: {best_clustering['Algorithm']}")
    print(f"  • Silhouette Score: {best_clustering['Silhouette Score']:.4f}")
    print(f"  • Davies-Bouldin: {best_clustering['Davies-Bouldin']:.4f}")

# ========================
# 4. تحلیل قوانین همبستگی
# ========================
print("\n" + "=" * 80)
print("📊 مرحله 4: تحلیل و ارزیابی قوانین همبستگی")
print("=" * 80)

if len(rules_df) > 0:
    print(f"\n📈 آمار کلی قوانین همبستگی:")
    print("─" * 50)
    print(f"  • تعداد کل قوانین: {len(rules_df)}")
    print(f"  • میانگین Support: {rules_df['support'].mean():.4f}")
    print(f"  • میانگین Confidence: {rules_df['confidence'].mean():.4f}")
    print(f"  • میانگین Lift: {rules_df['lift'].mean():.4f}")

    # قوانین با بیشترین تاثیر
    top_impact_rules = rules_df.nlargest(10, 'lift')[['antecedents', 'consequents', 'support', 'confidence', 'lift']]

    print("\n🔝 10 قانون با بیشترین تاثیر (Lift):")
    print("=" * 80)
    for idx, rule in top_impact_rules.iterrows():
        print(f"\n📌 قانون {idx + 1}:")
        print(f"  IF: {rule['antecedents']}")
        print(f"  THEN: {rule['consequents']}")
        print(f"  [Support: {rule['support']:.3f}, Confidence: {rule['confidence']:.3f}, Lift: {rule['lift']:.3f}]")

# ========================
# 5. مقایسه بین‌فازی (Cross-Phase Comparison)
# ========================
print("\n" + "=" * 80)
print("📊 مرحله 5: مقایسه و تحلیل بین‌فازی")
print("=" * 80)

# ایجاد جدول مقایسه کلی
phase_comparison = {
    'Phase': ['Classification', 'Clustering', 'Association Rules'],
    'Best Method': [
        best_model['Model'] if 'best_model' in locals() else 'N/A',
        best_clustering['Algorithm'] if 'best_clustering' in locals() else 'N/A',
        'FP-Growth' if association_results else 'N/A'
    ],
    'Key Metric': [
        f"F1-Score: {best_model['F1-Score']:.4f}" if 'best_model' in locals() else 'N/A',
        f"Silhouette: {best_clustering['Silhouette Score']:.4f}" if 'best_clustering' in locals() else 'N/A',
        f"Rules: {association_results.get('total_rules', 0)}" if association_results else 'N/A'
    ],
    'Performance': [
        'Moderate' if 'best_model' in locals() and best_model['F1-Score'] < 0.6 else 'Good',
        'Good' if 'best_clustering' in locals() and best_clustering['Silhouette Score'] > 0.3 else 'Moderate',
        'Good' if association_results.get('strong_rules_count', 0) > 20 else 'Moderate'
    ]
}

df_phase_comparison = pd.DataFrame(phase_comparison)

print("\n📈 مقایسه عملکرد فازهای مختلف:")
print("=" * 70)
print(df_phase_comparison.to_string(index=False))
print("=" * 70)

# ========================
# 6. ایجاد نمودارهای مقایسه‌ای
# ========================
print("\n" + "=" * 80)
print("📊 مرحله 6: ایجاد نمودارهای تحلیلی و مقایسه‌ای")
print("=" * 80)

# 6.1 نمودار مقایسه مدل‌های Classification
if len(df_classification) > 0:
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # نمودار میله‌ای برای Accuracy
    ax1 = axes[0, 0]
    models = df_classification['Model'].values
    accuracies = df_classification['Accuracy'].values
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(models)))
    bars1 = ax1.bar(range(len(models)), accuracies, color=colors)
    ax1.set_xlabel('Models', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title('Accuracy Comparison', fontsize=14, fontweight='bold')
    ax1.set_xticks(range(len(models)))
    ax1.set_xticklabels(models, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1])

    # اضافه کردن مقادیر روی میله‌ها
    for i, (bar, val) in enumerate(zip(bars1, accuracies)):
        ax1.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    # نمودار میله‌ای برای F1-Score
    ax2 = axes[0, 1]
    f1_scores = df_classification['F1-Score'].values
    bars2 = ax2.bar(range(len(models)), f1_scores, color=colors)
    ax2.set_xlabel('Models', fontsize=12)
    ax2.set_ylabel('F1-Score', fontsize=12)
    ax2.set_title('F1-Score Comparison', fontsize=14, fontweight='bold')
    ax2.set_xticks(range(len(models)))
    ax2.set_xticklabels(models, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])

    for i, (bar, val) in enumerate(zip(bars2, f1_scores)):
        ax2.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    # نمودار Radar برای مقایسه چند معیاره
    ax3 = axes[0, 2]

    # انتخاب 5 مدل برتر
    top_5_models = df_classification.head(5)
    categories = ['Accuracy', 'Precision', 'Recall', 'F1-Score']

    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]

    ax3 = plt.subplot(2, 3, 3, projection='polar')

    for idx, row in top_5_models.iterrows():
        values = [row['Accuracy'], row['Precision'], row['Recall'], row['F1-Score']]
        values += values[:1]
        ax3.plot(angles, values, 'o-', linewidth=2, label=row['Model'][:15])
        ax3.fill(angles, values, alpha=0.1)

    ax3.set_xticks(angles[:-1])
    ax3.set_xticklabels(categories)
    ax3.set_ylim(0, 1)
    ax3.set_title('Top 5 Models - Multi-Metric Comparison',
                  fontsize=14, fontweight='bold', pad=20)
    ax3.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax3.grid(True)

    # نمودار Box Plot برای Cross-Validation Scores
    ax4 = axes[1, 0]
    cv_data = [df_classification['CV Mean'].values]
    bp = ax4.boxplot(cv_data, vert=True, patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    ax4.set_ylabel('Cross-Validation Score', fontsize=12)
    ax4.set_title('Cross-Validation Score Distribution', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)

    # Scatter plot: Precision vs Recall
    ax5 = axes[1, 1]
    scatter = ax5.scatter(df_classification['Precision'],
                          df_classification['Recall'],
                          c=df_classification['F1-Score'],
                          s=100, cmap='viridis', alpha=0.7, edgecolors='black')

    # اضافه کردن برچسب برای هر نقطه
    for idx, row in df_classification.iterrows():
        ax5.annotate(row['Model'][:10],
                     (row['Precision'], row['Recall']),
                     fontsize=8, ha='center')

    ax5.set_xlabel('Precision', fontsize=12)
    ax5.set_ylabel('Recall', fontsize=12)
    ax5.set_title('Precision vs Recall Trade-off', fontsize=14, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax5, label='F1-Score')

    # نمودار مقایسه AUC
    ax6 = axes[1, 2]
    auc_values = df_classification['AUC'].values
    bars3 = ax6.barh(range(len(models)), auc_values, color=colors)
    ax6.set_xlabel('AUC Score', fontsize=12)
    ax6.set_ylabel('Models', fontsize=12)
    ax6.set_title('AUC-ROC Comparison', fontsize=14, fontweight='bold')
    ax6.set_yticks(range(len(models)))
    ax6.set_yticklabels(models)
    ax6.grid(True, alpha=0.3, axis='x')
    ax6.set_xlim([0, 1])

    # اضافه کردن مقادیر
    for i, (bar, val) in enumerate(zip(bars3, auc_values)):
        ax6.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                 f'{val:.3f}', ha='left', va='center', fontsize=9)

    plt.suptitle('Classification Models - Comprehensive Evaluation',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('classification_comprehensive_evaluation.png', dpi=150, bbox_inches='tight')
    print("✅ نمودار جامع Classification در 'classification_comprehensive_evaluation.png' ذخیره شد")
    plt.close()

# 6.2 نمودار مقایسه خوشه‌بندی
if len(df_clustering) > 0:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Silhouette Score
    ax1 = axes[0]
    algorithms = df_clustering['Algorithm'].values
    silhouette_scores = df_clustering['Silhouette Score'].values
    colors_cluster = plt.cm.coolwarm(np.linspace(0.2, 0.8, len(algorithms)))
    bars1 = ax1.bar(algorithms, silhouette_scores, color=colors_cluster)
    ax1.set_xlabel('Algorithm', fontsize=12)
    ax1.set_ylabel('Silhouette Score', fontsize=12)
    ax1.set_title('Silhouette Score Comparison', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    for bar, val in zip(bars1, silhouette_scores):
        ax1.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=10)

    # Davies-Bouldin Score (lower is better)
    ax2 = axes[1]
    db_scores = df_clustering['Davies-Bouldin'].values
    bars2 = ax2.bar(algorithms, db_scores, color=colors_cluster)
    ax2.set_xlabel('Algorithm', fontsize=12)
    ax2.set_ylabel('Davies-Bouldin Score', fontsize=12)
    ax2.set_title('Davies-Bouldin Score (Lower is Better)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    for bar, val in zip(bars2, db_scores):
        ax2.text(bar.get_x() + bar.get_width() / 2, val + 0.05,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=10)

    # Calinski-Harabasz Score
    ax3 = axes[2]
    ch_scores = df_clustering['Calinski-Harabasz'].values
    bars3 = ax3.bar(algorithms, ch_scores, color=colors_cluster)
    ax3.set_xlabel('Algorithm', fontsize=12)
    ax3.set_ylabel('Calinski-Harabasz Score', fontsize=12)
    ax3.set_title('Calinski-Harabasz Score (Higher is Better)', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    for bar, val in zip(bars3, ch_scores):
        ax3.text(bar.get_x() + bar.get_width() / 2, val + 1,
                 f'{val:.1f}', ha='center', va='bottom', fontsize=10)

    plt.suptitle('Clustering Algorithms - Quality Metrics Comparison',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('clustering_evaluation.png', dpi=150, bbox_inches='tight')
    print("✅ نمودار ارزیابی Clustering در 'clustering_evaluation.png' ذخیره شد")
    plt.close()

# 6.3 نمودار تحلیل Association Rules
if len(rules_df) > 0:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # توزیع Support
    ax1 = axes[0, 0]
    ax1.hist(rules_df['support'], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    ax1.set_xlabel('Support', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Distribution of Support Values', fontsize=14, fontweight='bold')
    ax1.axvline(rules_df['support'].mean(), color='red', linestyle='--',
                label=f'Mean: {rules_df["support"].mean():.3f}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # توزیع Confidence
    ax2 = axes[0, 1]
    ax2.hist(rules_df['confidence'], bins=30, color='lightgreen', edgecolor='black', alpha=0.7)
    ax2.set_xlabel('Confidence', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Distribution of Confidence Values', fontsize=14, fontweight='bold')
    ax2.axvline(rules_df['confidence'].mean(), color='red', linestyle='--',
                label=f'Mean: {rules_df["confidence"].mean():.3f}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # توزیع Lift
    ax3 = axes[1, 0]
    ax3.hist(rules_df['lift'], bins=30, color='coral', edgecolor='black', alpha=0.7)
    ax3.set_xlabel('Lift', fontsize=12)
    ax3.set_ylabel('Frequency', fontsize=12)
    ax3.set_title('Distribution of Lift Values', fontsize=14, fontweight='bold')
    ax3.axvline(1, color='green', linestyle='--', label='Lift = 1 (Independence)')
    ax3.axvline(rules_df['lift'].mean(), color='red', linestyle='--',
                label=f'Mean: {rules_df["lift"].mean():.3f}')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Scatter: Support vs Confidence colored by Lift
    ax4 = axes[1, 1]
    scatter = ax4.scatter(rules_df['support'], rules_df['confidence'],
                          c=rules_df['lift'], cmap='viridis',
                          s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
    ax4.set_xlabel('Support', fontsize=12)
    ax4.set_ylabel('Confidence', fontsize=12)
    ax4.set_title('Support vs Confidence (colored by Lift)', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax4, label='Lift')

    plt.suptitle('Association Rules - Statistical Analysis',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('association_rules_analysis.png', dpi=150, bbox_inches='tight')
    print("✅ نمودار تحلیل Association Rules در 'association_rules_analysis.png' ذخیره شد")
    plt.close()

# ========================
# 7. تحلیل آماری پیشرفته
# ========================
print("\n" + "=" * 80)
print("📊 مرحله 7: تحلیل آماری پیشرفته")
print("=" * 80)

# 7.1 آزمون آماری برای مقایسه مدل‌ها
if len(df_classification) > 1:
    print("\n📈 آزمون Friedman برای مقایسه مدل‌های Classification:")
    print("─" * 50)

    # آماده‌سازی داده‌ها برای آزمون
    metrics_for_test = df_classification[['Accuracy', 'Precision', 'Recall', 'F1-Score']].values

    # انجام آزمون Friedman
    if metrics_for_test.shape[0] >= 3:
        statistic, p_value = stats.friedmanchisquare(*metrics_for_test)
        print(f"  • Friedman Statistic: {statistic:.4f}")
        print(f"  • P-value: {p_value:.6f}")

        if p_value < 0.05:
            print("  ✅ نتیجه: تفاوت معناداری بین مدل‌ها وجود دارد (p < 0.05)")
        else:
            print("  ⚠️ نتیجه: تفاوت معناداری بین مدل‌ها وجود ندارد (p >= 0.05)")

# 7.2 محاسبه فاصله اطمینان
print("\n📈 فاصله اطمینان 95% برای معیارهای ارزیابی:")
print("─" * 50)

for metric in ['Accuracy', 'Precision', 'Recall', 'F1-Score']:
    values = df_classification[metric].values
    mean = np.mean(values)
    std = np.std(values)
    n = len(values)

    # محاسبه فاصله اطمینان
    confidence_level = 0.95
    degrees_freedom = n - 1
    t_value = st.t.ppf((1 + confidence_level) / 2, degrees_freedom)
    margin_error = t_value * (std / np.sqrt(n))

    ci_lower = mean - margin_error
    ci_upper = mean + margin_error

    print(f"  • {metric}: [{ci_lower:.4f}, {ci_upper:.4f}] (Mean: {mean:.4f})")

# ========================
# 8. ارزیابی ترکیبی و یکپارچه
# ========================
print("\n" + "=" * 80)
print("📊 مرحله 8: ارزیابی ترکیبی و یکپارچه مدل‌ها")
print("=" * 80)

# 8.1 محاسبه امتیاز ترکیبی برای هر روش
print("\n📈 محاسبه امتیاز ترکیبی (Composite Score):")
print("─" * 50)

# امتیازدهی به Classification
if len(df_classification) > 0:
    classification_score = (
            df_classification['F1-Score'].mean() * 0.4 +
            df_classification['Accuracy'].mean() * 0.3 +
            df_classification['AUC'].mean() * 0.3
    )
    print(f"  • Classification Composite Score: {classification_score:.4f}")
else:
    classification_score = 0

# امتیازدهی به Clustering
if len(df_clustering) > 0:
    # نرمال‌سازی Silhouette (0 to 1)
    silhouette_norm = (df_clustering['Silhouette Score'].mean() + 1) / 2
    # نرمال‌سازی Davies-Bouldin (inverse, lower is better)
    db_norm = 1 / (1 + df_clustering['Davies-Bouldin'].mean())
    # نرمال‌سازی Calinski-Harabasz
    ch_max = df_clustering['Calinski-Harabasz'].max()
    ch_norm = df_clustering['Calinski-Harabasz'].mean() / ch_max if ch_max > 0 else 0

    clustering_score = (silhouette_norm * 0.5 + db_norm * 0.3 + ch_norm * 0.2)
    print(f"  • Clustering Composite Score: {clustering_score:.4f}")
else:
    clustering_score = 0

# امتیازدهی به Association Rules
if association_results:
    total_rules = association_results.get('total_rules', 0)
    strong_rules = association_results.get('strong_rules_count', 0)

    if total_rules > 0:
        rule_quality = strong_rules / total_rules
        rule_quantity = min(total_rules / 100, 1)  # نرمال‌سازی تعداد قوانین
        association_score = (rule_quality * 0.6 + rule_quantity * 0.4)
    else:
        association_score = 0

    print(f"  • Association Rules Composite Score: {association_score:.4f}")
else:
    association_score = 0

# 8.2 رتبه‌بندی نهایی
print("\n🏆 رتبه‌بندی نهایی روش‌ها:")
print("─" * 50)

methods_ranking = {
    'Classification': classification_score,
    'Clustering': clustering_score,
    'Association Rules': association_score
}

sorted_methods = sorted(methods_ranking.items(), key=lambda x: x[1], reverse=True)

for rank, (method, score) in enumerate(sorted_methods, 1):
    medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉"
    print(f"  {medal} رتبه {rank}: {method} (Score: {score:.4f})")

# ========================
# 9. تحلیل نقاط قوت و ضعف
# ========================
print("\n" + "=" * 80)
print("📊 مرحله 9: تحلیل نقاط قوت و ضعف هر روش")
print("=" * 80)

# 9.1 تحلیل Classification
print("\n📌 Classification Models:")
print("─" * 50)
if len(df_classification) > 0:
    avg_f1 = df_classification['F1-Score'].mean()

    print("✅ نقاط قوت:")
    if avg_f1 > 0.7:
        print("  • عملکرد عالی در پیش‌بینی (F1 > 0.7)")
    elif avg_f1 > 0.5:
        print("  • عملکرد قابل قبول در پیش‌بینی")

    if df_classification['Precision'].mean() > df_classification['Recall'].mean():
        print("  • دقت بالا در تشخیص موارد مثبت")
    else:
        print("  • پوشش خوب در شناسایی تمام موارد مثبت")

    print("\n⚠️ نقاط ضعف:")
    if avg_f1 < 0.6:
        print("  • نیاز به بهبود دقت پیش‌بینی")

    cv_std_avg = df_classification['CV Std'].mean()
    if cv_std_avg > 0.05:
        print(f"  • واریانس بالا در Cross-Validation (Std: {cv_std_avg:.4f})")

# 9.2 تحلیل Clustering
print("\n📌 Clustering Algorithms:")
print("─" * 50)
if len(df_clustering) > 0:
    avg_silhouette = df_clustering['Silhouette Score'].mean()

    print("✅ نقاط قوت:")
    if avg_silhouette > 0.5:
        print("  • خوشه‌بندی با کیفیت عالی (Silhouette > 0.5)")
    elif avg_silhouette > 0.25:
        print("  • ساختار خوشه‌ای قابل قبول")

    if clustering_results.get('optimal_k', 0) <= 3:
        print(f"  • تعداد خوشه‌های بهینه و قابل تفسیر ({clustering_results.get('optimal_k', 'N/A')})")

    print("\n⚠️ نقاط ضعف:")
    if avg_silhouette < 0.25:
        print("  • ساختار خوشه‌ای ضعیف")

    if 'evaluation_metrics' in clustering_results:
        ari = clustering_results['evaluation_metrics'].get('ARI', 0)
        if ari < 0.3:
            print(f"  • ارتباط ضعیف با برچسب‌های واقعی (ARI: {ari:.4f})")

# 9.3 تحلیل Association Rules
print("\n📌 Association Rules:")
print("─" * 50)
if association_results:
    print("✅ نقاط قوت:")
    if association_results.get('total_rules', 0) > 50:
        print(f"  • تعداد قابل توجه قوانین کشف شده ({association_results.get('total_rules', 0)})")

    if association_results.get('strong_rules_count', 0) > 10:
        print(f"  • وجود قوانین قوی و معنادار ({association_results.get('strong_rules_count', 0)})")

    print("\n⚠️ نقاط ضعف:")
    if association_results.get('total_rules', 0) < 20:
        print("  • تعداد کم قوانین کشف شده")

    if association_results.get('min_support', 1) < 0.05:
        print(f"  • Support پایین برای اکثر قوانین (Min: {association_results.get('min_support', 0):.3f})")

# ========================
# 10. توصیه‌های بهبود
# ========================
print("\n" + "=" * 80)
print("💡 مرحله 10: توصیه‌های بهبود عملکرد")
print("=" * 80)

recommendations = []

# توصیه‌های Classification
if len(df_classification) > 0:
    if df_classification['F1-Score'].mean() < 0.7:
        recommendations.append({
            'روش': 'Classification',
            'توصیه': 'استفاده از روش‌های Ensemble Learning مانند Stacking یا Voting',
            'اولویت': 'بالا'
        })
        recommendations.append({
            'روش': 'Classification',
            'توصیه': 'بهینه‌سازی عمیق‌تر Hyperparameters با Bayesian Optimization',
            'اولویت': 'متوسط'
        })

    if abs(df_classification['Precision'].mean() - df_classification['Recall'].mean()) > 0.1:
        recommendations.append({
            'روش': 'Classification',
            'توصیه': 'تنظیم Threshold برای متوازن کردن Precision و Recall',
            'اولویت': 'بالا'
        })

# توصیه‌های Clustering
if len(df_clustering) > 0:
    if df_clustering['Silhouette Score'].mean() < 0.3:
        recommendations.append({
            'روش': 'Clustering',
            'توصیه': 'استفاده از روش‌های کاهش ابعاد قبل از خوشه‌بندی (PCA, t-SNE)',
            'اولویت': 'بالا'
        })
        recommendations.append({
            'روش': 'Clustering',
            'توصیه': 'آزمایش با معیارهای فاصله مختلف (Manhattan, Cosine)',
            'اولویت': 'متوسط'
        })

# توصیه‌های Association Rules
if association_results:
    if association_results.get('total_rules', 0) < 50:
        recommendations.append({
            'روش': 'Association Rules',
            'توصیه': 'کاهش حد آستانه Support برای کشف قوانین بیشتر',
            'اولویت': 'متوسط'
        })

    if association_results.get('strong_rules_count', 0) < 10:
        recommendations.append({
            'روش': 'Association Rules',
            'توصیه': 'بررسی ترکیبات بیشتر ویژگی‌ها و افزایش Max Length',
            'اولویت': 'پایین'
        })

# نمایش توصیه‌ها
if recommendations:
    df_recommendations = pd.DataFrame(recommendations)

    print("\n📋 جدول توصیه‌های بهبود:")
    print("=" * 80)

    for priority in ['بالا', 'متوسط', 'پایین']:
        priority_recs = df_recommendations[df_recommendations['اولویت'] == priority]
        if not priority_recs.empty:
            print(f"\n🔴 اولویت {priority}:")
            for _, rec in priority_recs.iterrows():
                print(f"  • [{rec['روش']}] {rec['توصیه']}")

# ========================
# 11. گزارش نهایی جامع
# ========================
print("\n" + "=" * 80)
print("📄 مرحله 11: تولید گزارش نهایی جامع")
print("=" * 80)

# ایجاد گزارش نهایی
final_report = []
final_report.append("=" * 100)
final_report.append(" " * 35 + "گزارش نهایی ارزیابی و مقایسه مدل‌ها")
final_report.append(" " * 30 + "پروژه پیش‌بینی عملکرد دانش‌آموزان")
final_report.append("=" * 100)
final_report.append(f"\nتاریخ تولید گزارش: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# بخش 1: خلاصه اجرایی
final_report.append("\n" + "─" * 80)
final_report.append("1. خلاصه اجرایی")
final_report.append("─" * 80)
final_report.append(
    f"• تعداد مدل‌های Classification ارزیابی شده: {len(df_classification) if 'df_classification' in locals() else 0}")
final_report.append(
    f"• تعداد الگوریتم‌های Clustering ارزیابی شده: {len(df_clustering) if 'df_clustering' in locals() else 0}")
final_report.append(
    f"• تعداد قوانین Association کشف شده: {association_results.get('total_rules', 0) if association_results else 0}")

# بخش 2: بهترین نتایج
final_report.append("\n" + "─" * 80)
final_report.append("2. بهترین نتایج هر روش")
final_report.append("─" * 80)

if 'best_model' in locals():
    final_report.append(f"\n🏆 Classification:")
    final_report.append(f"  • بهترین مدل: {best_model['Model']}")
    final_report.append(f"  • F1-Score: {best_model['F1-Score']:.4f}")
    final_report.append(f"  • Accuracy: {best_model['Accuracy']:.4f}")

if 'best_clustering' in locals():
    final_report.append(f"\n🏆 Clustering:")
    final_report.append(f"  • بهترین الگوریتم: {best_clustering['Algorithm']}")
    final_report.append(f"  • Silhouette Score: {best_clustering['Silhouette Score']:.4f}")

if association_results:
    final_report.append(f"\n🏆 Association Rules:")
    final_report.append(f"  • تعداد قوانین قوی: {association_results.get('strong_rules_count', 0)}")
    final_report.append(
        f"  • میانگین Lift: {rules_df['lift'].mean():.4f}" if len(rules_df) > 0 else "  • میانگین Lift: N/A")

# بخش 3: رتبه‌بندی نهایی
final_report.append("\n" + "─" * 80)
final_report.append("3. رتبه‌بندی نهایی روش‌ها")
final_report.append("─" * 80)

for rank, (method, score) in enumerate(sorted_methods, 1):
    final_report.append(f"  {rank}. {method}: {score:.4f}")

# بخش 4: توصیه‌های کلیدی
final_report.append("\n" + "─" * 80)
final_report.append("4. توصیه‌های کلیدی")
final_report.append("─" * 80)

if recommendations:
    high_priority = [r for r in recommendations if r['اولویت'] == 'بالا']
    if high_priority:
        final_report.append("\n• توصیه‌های با اولویت بالا:")
        for rec in high_priority[:5]:  # حداکثر 5 توصیه
            final_report.append(f"  - [{rec['روش']}] {rec['توصیه']}")

# بخش 5: نتیجه‌گیری
final_report.append("\n" + "─" * 80)
final_report.append("5. نتیجه‌گیری نهایی")
final_report.append("─" * 80)

# تعیین وضعیت کلی پروژه
overall_performance = (classification_score + clustering_score + association_score) / 3

if overall_performance > 0.7:
    performance_level = "عالی"
    emoji = "🌟"
elif overall_performance > 0.5:
    performance_level = "خوب"
    emoji = "✅"
elif overall_performance > 0.3:
    performance_level = "متوسط"
    emoji = "⚠️"
else:
    performance_level = "نیازمند بهبود"
    emoji = "🔴"

final_report.append(f"\n{emoji} عملکرد کلی سیستم: {performance_level} (امتیاز: {overall_performance:.4f})")
final_report.append("\nپروژه پیش‌بینی عملکرد دانش‌آموزان با موفقیت در 5 فاز اجرا و ارزیابی شد.")
final_report.append("نتایج نشان می‌دهد که ترکیب روش‌های مختلف داده‌کاوی می‌تواند")
final_report.append("دیدگاه جامعی از عوامل موثر بر عملکرد تحصیلی ارائه دهد.")

# ذخیره گزارش
with open('final_evaluation_report.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(final_report))

print("✅ گزارش نهایی در 'final_evaluation_report.txt' ذخیره شد")

# ذخیره نتایج ارزیابی
evaluation_results = {
    'classification_comparison': df_classification.to_dict() if 'df_classification' in locals() else {},
    'clustering_comparison': df_clustering.to_dict() if 'df_clustering' in locals() else {},
    'association_summary': association_results if association_results else {},
    'composite_scores': {
        'classification': classification_score,
        'clustering': clustering_score,
        'association': association_score
    },
    'overall_performance': overall_performance,
    'recommendations': recommendations,
    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
}

with open('phase5_evaluation_results.pkl', 'wb') as f:
    pickle.dump(evaluation_results, f)

print("✅ نتایج ارزیابی در 'phase5_evaluation_results.pkl' ذخیره شد")

# ========================
# 12. نمودار خلاصه نهایی
# ========================
print("\n📊 ایجاد نمودار خلاصه نهایی...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# نمودار 1: مقایسه امتیازات ترکیبی
ax1 = axes[0, 0]
methods = list(methods_ranking.keys())
scores = list(methods_ranking.values())
colors_final = ['#FF6B6B', '#4ECDC4', '#45B7D1']
bars = ax1.bar(methods, scores, color=colors_final, edgecolor='black', linewidth=2)

for bar, score in zip(bars, scores):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
             f'{score:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

ax1.set_ylabel('Composite Score', fontsize=12, fontweight='bold')
ax1.set_title('Overall Performance Comparison', fontsize=14, fontweight='bold')
ax1.set_ylim([0, max(scores) * 1.2 if scores else 1])
ax1.grid(True, alpha=0.3, axis='y')

# نمودار 2: خلاصه معیارهای Classification
ax2 = axes[0, 1]
if len(df_classification) > 0:
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    avg_values = [df_classification[m].mean() for m in metrics]

    radar_angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    radar_angles += radar_angles[:1]
    avg_values += avg_values[:1]

    ax2 = plt.subplot(2, 2, 2, projection='polar')
    ax2.plot(radar_angles, avg_values, 'o-', linewidth=2, color='#FF6B6B')
    ax2.fill(radar_angles, avg_values, alpha=0.25, color='#FF6B6B')
    ax2.set_xticks(radar_angles[:-1])
    ax2.set_xticklabels(metrics)
    ax2.set_ylim(0, 1)
    ax2.set_title('Classification Metrics Summary', fontsize=14, fontweight='bold', pad=20)
    ax2.grid(True)

# نمودار 3: توزیع قوانین بر اساس Lift
ax3 = axes[1, 0]
if len(rules_df) > 0:
    lift_categories = ['Low (< 1)', 'Medium (1-1.5)', 'High (> 1.5)']
    lift_counts = [
        len(rules_df[rules_df['lift'] < 1]),
        len(rules_df[(rules_df['lift'] >= 1) & (rules_df['lift'] <= 1.5)]),
        len(rules_df[rules_df['lift'] > 1.5])
    ]

    colors_lift = ['#FF6B6B', '#FFD93D', '#6BCF7F']
    wedges, texts, autotexts = ax3.pie(lift_counts, labels=lift_categories, colors=colors_lift,
                                       autopct='%1.1f%%', startangle=90)
    ax3.set_title('Association Rules Distribution by Lift', fontsize=14, fontweight='bold')

    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')

# نمودار 4: Performance Level Gauge
ax4 = axes[1, 1]
ax4.axis('off')

# ایجاد یک gauge chart ساده
theta = np.linspace(0.2 * np.pi, 1.8 * np.pi, 100)
r_inner = 0.7
r_outer = 1.0

# رسم قوس‌های رنگی
colors_gauge = ['#FF6B6B', '#FFD93D', '#6BCF7F']
boundaries = [0.2 * np.pi, 0.73 * np.pi, 1.27 * np.pi, 1.8 * np.pi]

for i in range(3):
    theta_section = np.linspace(boundaries[i], boundaries[i + 1], 30)
    ax4.fill_between(theta_section, r_inner, r_outer, color=colors_gauge[i], alpha=0.6)

# اضافه کردن نشانگر
indicator_angle = 0.2 * np.pi + overall_performance * 1.6 * np.pi
ax4.arrow(0, 0, 0.9 * np.cos(indicator_angle), 0.9 * np.sin(indicator_angle),
          head_width=0.1, head_length=0.1, fc='black', ec='black', linewidth=2)

ax4.set_xlim(-1.2, 1.2)
ax4.set_ylim(-0.2, 1.2)
ax4.text(0, -0.1, f'Overall Score: {overall_performance:.2%}',
         ha='center', fontsize=14, fontweight='bold')
ax4.text(0, 0.5, performance_level, ha='center', fontsize=18, fontweight='bold')
ax4.set_title('Overall System Performance', fontsize=14, fontweight='bold')

plt.suptitle('📊 Comprehensive Evaluation Summary - Student Performance Prediction Project',
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('final_evaluation_summary.png', dpi=150, bbox_inches='tight')
print("✅ نمودار خلاصه نهایی در 'final_evaluation_summary.png' ذخیره شد")
plt.close()

# ========================
# پایان فاز 5
# ========================
print("\n" + "=" * 100)
print("✅ فاز 5: ارزیابی و مقایسه مدل‌ها با موفقیت تکمیل شد!")
print("=" * 100)

print("\n📋 خلاصه عملیات انجام شده:")
print("─" * 50)
print("1. ✓ بارگذاری و تحلیل نتایج 4 فاز قبلی")
print("2. ✓ ارزیابی جامع مدل‌های Classification")
print("3. ✓ ارزیابی کیفیت خوشه‌بندی")
print("4. ✓ تحلیل قوانین همبستگی")
print("5. ✓ مقایسه بین‌فازی و محاسبه امتیازات ترکیبی")
print("6. ✓ تحلیل آماری پیشرفته")
print("7. ✓ شناسایی نقاط قوت و ضعف")
print("8. ✓ ارائه توصیه‌های بهبود")
print("9. ✓ تولید گزارش‌های جامع")
print("10. ✓ ایجاد نمودارهای تحلیلی و مقایسه‌ای")

print("\n📁 فایل‌های تولید شده:")
print("─" * 50)
print("  • final_evaluation_report.txt - گزارش نهایی جامع")
print("  • phase5_evaluation_results.pkl - نتایج کامل ارزیابی")
print("  • classification_comprehensive_evaluation.png - نمودار جامع Classification")
print("  • clustering_evaluation.png - نمودار ارزیابی Clustering")
print("  • association_rules_analysis.png - نمودار تحلیل Association Rules")
print("  • final_evaluation_summary.png - نمودار خلاصه نهایی")

print("\n🏆 نتایج کلیدی:")
print("─" * 50)
print(f"  • عملکرد کلی سیستم: {performance_level} ({overall_performance:.4f})")
print(f"  • بهترین روش: {sorted_methods[0][0]} (امتیاز: {sorted_methods[0][1]:.4f})")
if 'best_model' in locals():
    print(f"  • بهترین مدل Classification: {best_model['Model']}")
if 'best_clustering' in locals():
    print(f"  • بهترین الگوریتم Clustering: {best_clustering['Algorithm']}")

print("\n" + "=" * 100)
print("🎓 پروژه پیش‌بینی عملکرد دانش‌آموزان")
print("📊 تمام 5 فاز با موفقیت تکمیل شد!")
print("=" * 100)

print("\n🚀 فازهای تکمیل شده:")
print("  ✅ فاز 1: پیش‌پردازش داده‌ها")
print("  ✅ فاز 2: مدل‌های طبقه‌بندی")
print("  ✅ فاز 3: خوشه‌بندی دانش‌آموزان")
print("  ✅ فاز 4: استخراج قوانین همبستگی")
print("  ✅ فاز 5: ارزیابی و مقایسه مدل‌ها")

print("\n💡 این پروژه نشان داد که ترکیب روش‌های مختلف داده‌کاوی")
print("   می‌تواند بینش‌های ارزشمندی برای بهبود عملکرد تحصیلی ارائه دهد.")

print("\n" + "=" * 100)