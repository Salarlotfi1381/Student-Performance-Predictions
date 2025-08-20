# پروژه پیش‌بینی عملکرد دانش‌آموزان
# فاز 3: خوشه‌بندی دانش‌آموزان (Clustering Students)

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

# کتابخانه‌های خوشه‌بندی
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, MeanShift
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import cdist

# تنظیمات نمایش
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.figsize'] = (12, 8)
sns.set_style("whitegrid")

print("=" * 80)
print("🎯 فاز 3: خوشه‌بندی دانش‌آموزان (Clustering Students)")
print("=" * 80)

# ========================
# 1. بارگذاری داده‌های پردازش شده
# ========================
print("\n" + "=" * 60)
print("مرحله 1: بارگذاری داده‌های پردازش شده")
print("=" * 60)

# بارگذاری داده‌های کامل پردازش شده
df = pd.read_csv('processed_student_data.csv')

# بارگذاری اطلاعات پیش‌پردازش
with open('preprocessing_info.pkl', 'rb') as f:
    preprocessing_info = pickle.load(f)

feature_columns = preprocessing_info['feature_columns']

# انتخاب ویژگی‌ها برای خوشه‌بندی
X_clustering = df[feature_columns]
y_actual = df['Pass_Status_Encoded']  # برای مقایسه با خوشه‌ها

print(f"\n✅ داده‌ها با موفقیت بارگذاری شدند:")
print(f"  • تعداد نمونه‌ها: {X_clustering.shape[0]}")
print(f"  • تعداد ویژگی‌ها: {X_clustering.shape[1]}")
print(f"\nویژگی‌های استفاده شده برای خوشه‌بندی:")
for i, col in enumerate(feature_columns, 1):
    print(f"  {i}. {col}")

# ========================
# 2. آماده‌سازی داده‌ها برای خوشه‌بندی
# ========================
print("\n" + "=" * 60)
print("مرحله 2: آماده‌سازی داده‌ها برای خوشه‌بندی")
print("=" * 60)

# نرمال‌سازی مجدد داده‌ها (اگر نیاز باشد)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_clustering)

print("✅ داده‌ها نرمال‌سازی شدند برای خوشه‌بندی بهتر")

# کاهش ابعاد برای visualizatio با PCA
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

print(f"✅ کاهش ابعاد با PCA انجام شد (واریانس توضیح داده شده: {pca.explained_variance_ratio_.sum():.2%})")

# ========================
# 3. تعیین تعداد بهینه خوشه‌ها با روش Elbow
# ========================
print("\n" + "=" * 60)
print("مرحله 3: تعیین تعداد بهینه خوشه‌ها")
print("=" * 60)

print("\n📊 روش Elbow Method...")

# محاسبه WCSS برای تعداد مختلف خوشه‌ها
wcss = []
silhouette_scores = []
k_range = range(2, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

    # محاسبه Silhouette Score
    labels = kmeans.labels_
    silhouette_avg = silhouette_score(X_scaled, labels)
    silhouette_scores.append(silhouette_avg)

    print(f"  K={k}: WCSS={kmeans.inertia_:.2f}, Silhouette Score={silhouette_avg:.4f}")

# نمودار Elbow
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# نمودار WCSS
ax1.plot(k_range, wcss, 'bo-', linewidth=2, markersize=8)
ax1.set_xlabel('تعداد خوشه‌ها (K)')
ax1.set_ylabel('WCSS (Within-Cluster Sum of Squares)')
ax1.set_title('Elbow Method - WCSS')
ax1.grid(True, alpha=0.3)

# اضافه کردن خط عمودی در نقطه بهینه (معمولاً K=3 یا 4)
optimal_k_elbow = 3  # می‌توان این را بر اساس نمودار تغییر داد
ax1.axvline(x=optimal_k_elbow, color='r', linestyle='--', alpha=0.5)
ax1.text(optimal_k_elbow, max(wcss) * 0.9, f'Optimal K={optimal_k_elbow}',
         rotation=90, verticalalignment='bottom')

# نمودار Silhouette Score
ax2.plot(k_range, silhouette_scores, 'go-', linewidth=2, markersize=8)
ax2.set_xlabel('تعداد خوشه‌ها (K)')
ax2.set_ylabel('Silhouette Score')
ax2.set_title('Silhouette Score Analysis')
ax2.grid(True, alpha=0.3)

# پیدا کردن بهترین K بر اساس Silhouette Score
optimal_k_silhouette = k_range[np.argmax(silhouette_scores)]
ax2.axvline(x=optimal_k_silhouette, color='r', linestyle='--', alpha=0.5)
ax2.text(optimal_k_silhouette, max(silhouette_scores) * 0.95,
         f'Best K={optimal_k_silhouette}', rotation=90, verticalalignment='bottom')

plt.tight_layout()
plt.savefig('elbow_silhouette_analysis.png', dpi=100, bbox_inches='tight')
print("\n✅ نمودار Elbow و Silhouette در 'elbow_silhouette_analysis.png' ذخیره شد.")
plt.close()

# انتخاب تعداد بهینه خوشه
optimal_k = optimal_k_silhouette
print(f"\n🎯 تعداد بهینه خوشه‌ها: {optimal_k}")

# ========================
# 4. اجرای الگوریتم‌های مختلف خوشه‌بندی
# ========================
print("\n" + "=" * 60)
print("مرحله 4: اجرای الگوریتم‌های مختلف خوشه‌بندی")
print("=" * 60)

clustering_results = {}

# 4.1. K-Means
print("\n🔄 1. K-Means Clustering...")
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X_scaled)

clustering_results['K-Means'] = {
    'labels': kmeans_labels,
    'silhouette': silhouette_score(X_scaled, kmeans_labels),
    'davies_bouldin': davies_bouldin_score(X_scaled, kmeans_labels),
    'calinski_harabasz': calinski_harabasz_score(X_scaled, kmeans_labels),
    'centers': kmeans.cluster_centers_
}

print(f"  ✓ Silhouette Score: {clustering_results['K-Means']['silhouette']:.4f}")
print(f"  ✓ Davies-Bouldin Score: {clustering_results['K-Means']['davies_bouldin']:.4f}")
print(f"  ✓ Calinski-Harabasz Score: {clustering_results['K-Means']['calinski_harabasz']:.2f}")

# 4.2. DBSCAN
print("\n🔄 2. DBSCAN Clustering...")
# تنظیم پارامترهای DBSCAN
eps_values = np.arange(0.1, 2.0, 0.1)
min_samples_values = [3, 5, 10]

best_dbscan_score = -1
best_dbscan_params = {}

for eps in eps_values:
    for min_samples in min_samples_values:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        dbscan_labels = dbscan.fit_predict(X_scaled)

        # بررسی که آیا حداقل 2 خوشه داریم
        n_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)

        if n_clusters >= 2:
            score = silhouette_score(X_scaled, dbscan_labels)
            if score > best_dbscan_score:
                best_dbscan_score = score
                best_dbscan_params = {'eps': eps, 'min_samples': min_samples}
                best_dbscan_labels = dbscan_labels

# اجرای DBSCAN با بهترین پارامترها
if best_dbscan_params:
    dbscan = DBSCAN(**best_dbscan_params)
    dbscan_labels = dbscan.fit_predict(X_scaled)

    clustering_results['DBSCAN'] = {
        'labels': dbscan_labels,
        'silhouette': silhouette_score(X_scaled, dbscan_labels),
        'davies_bouldin': davies_bouldin_score(X_scaled, dbscan_labels),
        'calinski_harabasz': calinski_harabasz_score(X_scaled, dbscan_labels),
        'params': best_dbscan_params,
        'n_clusters': len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0),
        'n_noise': list(dbscan_labels).count(-1)
    }

    print(f"  ✓ بهترین پارامترها: eps={best_dbscan_params['eps']:.1f}, min_samples={best_dbscan_params['min_samples']}")
    print(f"  ✓ تعداد خوشه‌ها: {clustering_results['DBSCAN']['n_clusters']}")
    print(f"  ✓ تعداد نقاط نویز: {clustering_results['DBSCAN']['n_noise']}")
    print(f"  ✓ Silhouette Score: {clustering_results['DBSCAN']['silhouette']:.4f}")

# 4.3. Hierarchical Clustering
print("\n🔄 3. Hierarchical (Agglomerative) Clustering...")
hierarchical = AgglomerativeClustering(n_clusters=optimal_k, linkage='ward')
hierarchical_labels = hierarchical.fit_predict(X_scaled)

clustering_results['Hierarchical'] = {
    'labels': hierarchical_labels,
    'silhouette': silhouette_score(X_scaled, hierarchical_labels),
    'davies_bouldin': davies_bouldin_score(X_scaled, hierarchical_labels),
    'calinski_harabasz': calinski_harabasz_score(X_scaled, hierarchical_labels)
}

print(f"  ✓ Silhouette Score: {clustering_results['Hierarchical']['silhouette']:.4f}")
print(f"  ✓ Davies-Bouldin Score: {clustering_results['Hierarchical']['davies_bouldin']:.4f}")

# 4.4. Gaussian Mixture Model
print("\n🔄 4. Gaussian Mixture Model...")
gmm = GaussianMixture(n_components=optimal_k, random_state=42)
gmm_labels = gmm.fit_predict(X_scaled)

clustering_results['GMM'] = {
    'labels': gmm_labels,
    'silhouette': silhouette_score(X_scaled, gmm_labels),
    'davies_bouldin': davies_bouldin_score(X_scaled, gmm_labels),
    'calinski_harabasz': calinski_harabasz_score(X_scaled, gmm_labels),
    'probabilities': gmm.predict_proba(X_scaled)
}

print(f"  ✓ Silhouette Score: {clustering_results['GMM']['silhouette']:.4f}")
print(f"  ✓ Davies-Bouldin Score: {clustering_results['GMM']['davies_bouldin']:.4f}")

# ========================
# 5. مقایسه الگوریتم‌های خوشه‌بندی
# ========================
print("\n" + "=" * 60)
print("مرحله 5: مقایسه الگوریتم‌های خوشه‌بندی")
print("=" * 60)

# ایجاد جدول مقایسه
comparison_data = []
for algo_name, results in clustering_results.items():
    comparison_data.append({
        'Algorithm': algo_name,
        'Silhouette Score': results['silhouette'],
        'Davies-Bouldin Score': results['davies_bouldin'],
        'Calinski-Harabasz Score': results['calinski_harabasz']
    })

comparison_df = pd.DataFrame(comparison_data)
comparison_df = comparison_df.sort_values('Silhouette Score', ascending=False)

print("\n📊 جدول مقایسه الگوریتم‌های خوشه‌بندی:")
print("─" * 70)
print(comparison_df.to_string(index=False))

# ذخیره جدول مقایسه
comparison_df.to_csv('clustering_comparison.csv', index=False, encoding='utf-8-sig')
print("\n✅ جدول مقایسه در 'clustering_comparison.csv' ذخیره شد.")

# انتخاب بهترین الگوریتم
best_algorithm = comparison_df.iloc[0]['Algorithm']
best_labels = clustering_results[best_algorithm]['labels']
print(f"\n🏆 بهترین الگوریتم: {best_algorithm}")

# ========================
# 6. تحلیل خوشه‌ها
# ========================
print("\n" + "=" * 60)
print("مرحله 6: تحلیل خوشه‌های ایجاد شده")
print("=" * 60)

# استفاده از K-Means برای تحلیل دقیق‌تر
analysis_labels = clustering_results['K-Means']['labels']
df['Cluster'] = analysis_labels

print(f"\n📊 تحلیل خوشه‌های K-Means:")
print("─" * 50)

# تحلیل هر خوشه
cluster_analysis = []
for cluster_id in range(optimal_k):
    cluster_data = df[df['Cluster'] == cluster_id]

    print(f"\n🔍 خوشه {cluster_id + 1}:")
    print(f"  • تعداد دانش‌آموزان: {len(cluster_data)} ({len(cluster_data) / len(df) * 100:.1f}%)")

    # میانگین ویژگی‌های مهم
    print("  • میانگین ویژگی‌ها:")

    important_features = ['AttendanceRate', 'StudyHoursPerWeek', 'PreviousGrade',
                          'ExtracurricularActivities', 'Study Hours']

    cluster_stats = {}
    for feature in feature_columns:
        if feature in cluster_data.columns:
            mean_val = cluster_data[feature].mean()
            cluster_stats[feature] = mean_val
            if feature in important_features:
                print(f"    - {feature}: {mean_val:.2f}")

    # درصد قبولی در این خوشه
    if 'Pass_Status_Encoded' in cluster_data.columns:
        pass_rate = cluster_data['Pass_Status_Encoded'].mean() * 100
        print(f"  • درصد قبولی: {pass_rate:.1f}%")
        cluster_stats['Pass_Rate'] = pass_rate

    cluster_analysis.append({
        'Cluster': cluster_id + 1,
        'Size': len(cluster_data),
        'Percentage': len(cluster_data) / len(df) * 100,
        **cluster_stats
    })

# ذخیره تحلیل خوشه‌ها
cluster_analysis_df = pd.DataFrame(cluster_analysis)
cluster_analysis_df.to_csv('cluster_analysis.csv', index=False, encoding='utf-8-sig')
print("\n✅ تحلیل خوشه‌ها در 'cluster_analysis.csv' ذخیره شد.")

# ========================
# 7. Visualization خوشه‌ها
# ========================
print("\n" + "=" * 60)
print("مرحله 7: رسم نمودارهای خوشه‌بندی")
print("=" * 60)

# نمودار خوشه‌ها برای همه الگوریتم‌ها
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
axes = axes.ravel()

algorithms = ['K-Means', 'DBSCAN', 'Hierarchical', 'GMM']
for idx, algo in enumerate(algorithms):
    if algo in clustering_results:
        labels = clustering_results[algo]['labels']

        # رسم خوشه‌ها در فضای PCA
        scatter = axes[idx].scatter(X_pca[:, 0], X_pca[:, 1],
                                    c=labels, cmap='viridis',
                                    s=50, alpha=0.6, edgecolors='black', linewidth=0.5)

        axes[idx].set_title(f'{algo} Clustering\n(Silhouette: {clustering_results[algo]["silhouette"]:.3f})')
        axes[idx].set_xlabel('First Principal Component')
        axes[idx].set_ylabel('Second Principal Component')

        # اضافه کردن مراکز خوشه برای K-Means
        if algo == 'K-Means':
            centers_pca = pca.transform(clustering_results[algo]['centers'])
            axes[idx].scatter(centers_pca[:, 0], centers_pca[:, 1],
                              c='red', marker='X', s=200, edgecolors='black', linewidth=2)
            axes[idx].legend(['Clusters', 'Centers'], loc='upper right')

        plt.colorbar(scatter, ax=axes[idx])

plt.suptitle('Clustering Results Visualization (PCA Space)', fontsize=16)
plt.tight_layout()
plt.savefig('clustering_visualization.png', dpi=100, bbox_inches='tight')
print("✅ نمودار خوشه‌بندی در 'clustering_visualization.png' ذخیره شد.")
plt.close()

# ========================
# 8. Dendrogram برای Hierarchical Clustering
# ========================
print("\n📊 رسم Dendrogram برای Hierarchical Clustering...")

plt.figure(figsize=(15, 8))
Z = linkage(X_scaled, 'ward')
dendrogram(Z, truncate_mode='level', p=10)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Sample Index or (Cluster Size)')
plt.ylabel('Distance')
plt.axhline(y=Z[-optimal_k + 1, 2], color='r', linestyle='--', alpha=0.5)
plt.text(0, Z[-optimal_k + 1, 2], f'Cut for {optimal_k} clusters',
         verticalalignment='bottom', fontsize=10)
plt.tight_layout()
plt.savefig('dendrogram.png', dpi=100, bbox_inches='tight')
print("✅ Dendrogram در 'dendrogram.png' ذخیره شد.")
plt.close()

# ========================
# 9. تحلیل ویژگی‌های هر خوشه
# ========================
print("\n" + "=" * 60)
print("مرحله 8: تحلیل عمیق ویژگی‌های خوشه‌ها")
print("=" * 60)

# نمودار Box Plot برای مقایسه خوشه‌ها
important_features_to_plot = ['AttendanceRate', 'StudyHoursPerWeek',
                              'PreviousGrade', 'ExtracurricularActivities']

# فیلتر کردن ویژگی‌های موجود
available_features = [f for f in important_features_to_plot if f in df.columns]

if available_features:
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()

    for idx, feature in enumerate(available_features[:4]):
        cluster_data = []
        cluster_labels_plot = []

        for cluster_id in range(optimal_k):
            cluster_values = df[df['Cluster'] == cluster_id][feature].values
            cluster_data.append(cluster_values)
            cluster_labels_plot.append(f'Cluster {cluster_id + 1}')

        bp = axes[idx].boxplot(cluster_data, labels=cluster_labels_plot, patch_artist=True)

        # رنگ‌آمیزی box plot
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, optimal_k))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)

        axes[idx].set_title(f'{feature} Distribution by Cluster')
        axes[idx].set_ylabel(feature)
        axes[idx].grid(True, alpha=0.3)

    plt.suptitle('Feature Distribution Across Clusters', fontsize=16)
    plt.tight_layout()
    plt.savefig('cluster_feature_distribution.png', dpi=100, bbox_inches='tight')
    print("✅ توزیع ویژگی‌ها در 'cluster_feature_distribution.png' ذخیره شد.")
    plt.close()

# ========================
# 10. ماتریس همبستگی خوشه‌ها با وضعیت قبولی
# ========================
print("\n" + "=" * 60)
print("مرحله 9: بررسی ارتباط خوشه‌ها با وضعیت قبولی")
print("=" * 60)

# مقایسه خوشه‌ها با برچسب‌های واقعی
from sklearn.metrics import confusion_matrix as conf_matrix

# ایجاد confusion matrix
cm = conf_matrix(y_actual, analysis_labels)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=[f'Cluster {i + 1}' for i in range(optimal_k)],
            yticklabels=['Fail', 'Pass'])
plt.title('Confusion Matrix: Actual Pass/Fail vs Clusters')
plt.xlabel('Predicted Cluster')
plt.ylabel('Actual Pass/Fail Status')
plt.tight_layout()
plt.savefig('cluster_pass_fail_matrix.png', dpi=100, bbox_inches='tight')
print("✅ ماتریس همبستگی در 'cluster_pass_fail_matrix.png' ذخیره شد.")
plt.close()

# محاسبه معیارهای ارزیابی
ari = adjusted_rand_score(y_actual, analysis_labels)
ami = adjusted_mutual_info_score(y_actual, analysis_labels)

print(f"\n📊 معیارهای ارزیابی ارتباط خوشه‌ها با وضعیت قبولی:")
print(f"  • Adjusted Rand Index: {ari:.4f}")
print(f"  • Adjusted Mutual Information: {ami:.4f}")

# ========================
# 11. پروفایل خوشه‌ها
# ========================
print("\n" + "=" * 60)
print("مرحله 10: ایجاد پروفایل برای هر خوشه")
print("=" * 60)

print("\n📋 پروفایل خوشه‌ها:")
print("─" * 50)

for cluster_id in range(optimal_k):
    cluster_data = df[df['Cluster'] == cluster_id]

    print(f"\n✨ خوشه {cluster_id + 1} - '{['ضعیف', 'متوسط', 'قوی'][cluster_id % 3]}':")
    print(f"  📊 تعداد: {len(cluster_data)} دانش‌آموز")

    # ویژگی‌های برجسته
    if 'AttendanceRate' in cluster_data.columns:
        attendance_mean = cluster_data['AttendanceRate'].mean()
        print(f"  📅 حضور: {'بالا' if attendance_mean > 0.5 else 'متوسط' if attendance_mean > -0.5 else 'پایین'}")

    if 'StudyHoursPerWeek' in cluster_data.columns:
        study_mean = cluster_data['StudyHoursPerWeek'].mean()
        print(f"  📚 ساعات مطالعه: {'زیاد' if study_mean > 0.5 else 'متوسط' if study_mean > -0.5 else 'کم'}")

    if 'Pass_Status_Encoded' in cluster_data.columns:
        pass_rate = cluster_data['Pass_Status_Encoded'].mean() * 100
        print(f"  🎯 نرخ قبولی: {pass_rate:.1f}%")

    # توصیه‌ها
    if pass_rate < 40:
        print("  💡 توصیه: نیاز به حمایت تحصیلی بیشتر و برنامه‌های تقویتی")
    elif pass_rate < 60:
        print("  💡 توصیه: تمرکز بر بهبود عادات مطالعه و مدیریت زمان")
    else:
        print("  💡 توصیه: حفظ روند فعلی و چالش‌های پیشرفته‌تر")

# ========================
# 12. ذخیره نتایج خوشه‌بندی
# ========================
print("\n" + "=" * 60)
print("مرحله 11: ذخیره نتایج خوشه‌بندی")
print("=" * 60)

# ذخیره نتایج خوشه‌بندی
clustering_output = {
    'algorithms': clustering_results,
    'best_algorithm': best_algorithm,
    'optimal_k': optimal_k,
    'cluster_analysis': cluster_analysis_df,
    'evaluation_metrics': {
        'ARI': ari,
        'AMI': ami
    },
    'pca_components': X_pca,
    'feature_columns': feature_columns
}

with open('clustering_results.pkl', 'wb') as f:
    pickle.dump(clustering_output, f)

print("✅ نتایج خوشه‌بندی در 'clustering_results.pkl' ذخیره شد.")

# ذخیره DataFrame با برچسب خوشه‌ها
df.to_csv('data_with_clusters.csv', index=False, encoding='utf-8-sig')
print("✅ داده‌ها با برچسب خوشه در 'data_with_clusters.csv' ذخیره شدند.")

# ========================
# 13. گزارش نهایی
# ========================
print("\n" + "=" * 80)
print("✅ فاز 3: خوشه‌بندی دانش‌آموزان با موفقیت تکمیل شد!")
print("=" * 80)

print("\n📋 خلاصه عملیات انجام شده:")
print("─" * 40)
print("1. ✓ بارگذاری و آماده‌سازی داده‌ها")
print("2. ✓ تعیین تعداد بهینه خوشه‌ها با Elbow Method و Silhouette Score")
print(f"3. ✓ اجرای 4 الگوریتم خوشه‌بندی مختلف")
print("4. ✓ ارزیابی با معیارهای Silhouette, Davies-Bouldin, Calinski-Harabasz")
print(f"5. ✓ انتخاب بهترین الگوریتم: {best_algorithm}")
print(f"6. ✓ تحلیل {optimal_k} خوشه ایجاد شده")
print("7. ✓ بررسی ارتباط خوشه‌ها با وضعیت قبولی/رد")
print("8. ✓ ایجاد پروفایل برای هر خوشه")
print("9. ✓ رسم نمودارهای تحلیلی و تفسیری")

print("\n📁 فایل‌های تولید شده:")
print("─" * 40)
print("  • clustering_comparison.csv - مقایسه الگوریتم‌های خوشه‌بندی")
print("  • cluster_analysis.csv - تحلیل دقیق هر خوشه")
print("  • data_with_clusters.csv - داده‌ها با برچسب خوشه")
print("  • clustering_results.pkl - نتایج کامل خوشه‌بندی")
print("  • elbow_silhouette_analysis.png - نمودار Elbow و Silhouette")
print("  • clustering_visualization.png - نمایش خوشه‌ها در فضای PCA")
print("  • dendrogram.png - نمودار سلسله‌مراتبی")
print("  • cluster_feature_distribution.png - توزیع ویژگی‌ها در خوشه‌ها")
print("  • cluster_pass_fail_matrix.png - ماتریس ارتباط با قبولی/رد")

print("\n🏆 نتایج کلیدی:")
print("─" * 40)
print(f"  • تعداد خوشه‌های بهینه: {optimal_k}")
print(f"  • بهترین الگوریتم: {best_algorithm}")
print(f"  • بهترین Silhouette Score: {clustering_results[best_algorithm]['silhouette']:.4f}")
print(f"  • Adjusted Rand Index: {ari:.4f}")
print(f"  • Adjusted Mutual Information: {ami:.4f}")

print("\n💡 بینش‌های کلیدی:")
print("─" * 40)
for i, cluster_info in enumerate(cluster_analysis):
    print(
        f"  • خوشه {i + 1}: {cluster_info['Size']} دانش‌آموز ({cluster_info['Percentage']:.1f}%) - نرخ قبولی: {cluster_info.get('Pass_Rate', 0):.1f}%")

print("\n🎯 آماده برای فاز بعدی:")
print("─" * 40)
print("  → Association Rules (قوانین همبستگی)")

print("\n" + "=" * 80)
print("پایان فاز 3 - Clustering Students 🚀")
print("=" * 80)