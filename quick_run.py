# اجرای سریع پروژه پیش‌بینی عملکرد دانش‌آموزان
# نسخه خلاصه برای مشاهده سریع نتایج

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, f1_score, silhouette_score
import warnings
warnings.filterwarnings('ignore')

print("🎓 اجرای سریع پروژه پیش‌بینی عملکرد دانش‌آموزان")
print("=" * 60)

# 1. ایجاد دیتاست نمونه
print("📊 ایجاد دیتاست...")
np.random.seed(42)
n = 1000

data = {
    'Hours_Studied': np.random.normal(7, 3, n).clip(1, 20),
    'Previous_Scores': np.random.normal(75, 15, n).clip(40, 100),
    'Sleep_Hours': np.random.normal(7, 1.5, n).clip(4, 12),
    'Extracurricular': np.random.choice([0, 1], n, p=[0.4, 0.6]),
    'Parental_Support': np.random.choice([0, 1, 2], n, p=[0.3, 0.4, 0.3]),  # Low, Medium, High
    'Resource_Access': np.random.choice([0, 1, 2], n, p=[0.25, 0.5, 0.25])
}

df = pd.DataFrame(data)

# ایجاد متغیر هدف بر اساس ویژگی‌ها
def create_target(row):
    score = (row['Hours_Studied'] * 3 + 
             row['Previous_Scores'] * 0.4 + 
             row['Sleep_Hours'] * 2 + 
             row['Extracurricular'] * 10 + 
             row['Parental_Support'] * 8 + 
             row['Resource_Access'] * 6 + 
             np.random.normal(0, 5))
    
    if score < 50:
        return 0  # Poor
    elif score < 80:
        return 1  # Average
    else:
        return 2  # Excellent

df['Performance'] = df.apply(create_target, axis=1)
performance_names = {0: 'Poor', 1: 'Average', 2: 'Excellent'}

print(f"✅ دیتاست {len(df)} دانش‌آموز ایجاد شد")
print(f"📊 توزیع عملکرد:")
for k, v in pd.Series(df['Performance']).map(performance_names).value_counts().items():
    print(f"   {k}: {v} نفر ({v/len(df)*100:.1f}%)")

# 2. پیش‌پردازش سریع
print(f"\n🔧 پیش‌پردازش...")
X = df.drop('Performance', axis=1)
y = df['Performance']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

print(f"✅ داده‌ها آماده شدند: {len(X_train)} آموزشی، {len(X_test)} تست")

# 3. مدل‌های طبقه‌بندی
print(f"\n🤖 آموزش مدل‌ها...")

models = {
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42, n_estimators=50)
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    results[name] = {'accuracy': accuracy, 'f1': f1}
    print(f"   {name}: Accuracy={accuracy:.3f}, F1={f1:.3f}")

# بهترین مدل
best_model = max(results.keys(), key=lambda x: results[x]['f1'])
print(f"\n🏆 بهترین مدل: {best_model} (F1={results[best_model]['f1']:.3f})")

# 4. خوشه‌بندی سریع
print(f"\n🔗 خوشه‌بندی...")

# تست چند تعداد خوشه
silhouette_scores = []
k_range = range(2, 6)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    silhouette_scores.append(score)
    print(f"   K={k}: Silhouette Score={score:.3f}")

best_k = k_range[np.argmax(silhouette_scores)]
print(f"\n🎯 بهترین تعداد خوشه: {best_k}")

# خوشه‌بندی نهایی
kmeans_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
clusters = kmeans_final.fit_predict(X_scaled)

# تحلیل خوشه‌ها
print(f"\n📊 تحلیل خوشه‌ها:")
df['Cluster'] = clusters

for cluster_id in range(best_k):
    cluster_data = df[df['Cluster'] == cluster_id]
    dominant_performance = cluster_data['Performance'].mode().iloc[0]
    performance_name = performance_names[dominant_performance]
    
    print(f"   خوشه {cluster_id}: {len(cluster_data)} نفر - عملکرد غالب: {performance_name}")
    print(f"      میانگین ساعت مطالعه: {cluster_data['Hours_Studied'].mean():.1f}")
    print(f"      میانگین نمره قبلی: {cluster_data['Previous_Scores'].mean():.1f}")

# 5. آمار کلیدی
print(f"\n📈 آمار کلیدی:")

# همبستگی ساده
correlations = df[['Hours_Studied', 'Previous_Scores', 'Sleep_Hours', 'Performance']].corr()['Performance'].sort_values(ascending=False)

print(f"🔗 همبستگی با عملکرد:")
for feature, corr in correlations.items():
    if feature != 'Performance':
        print(f"   {feature}: {corr:.3f}")

# میانگین ویژگی‌ها در هر گروه عملکرد
print(f"\n📊 مشخصات گروه‌های عملکرد:")
performance_stats = df.groupby('Performance').agg({
    'Hours_Studied': 'mean',
    'Previous_Scores': 'mean',
    'Sleep_Hours': 'mean',
    'Extracurricular': 'mean'
}).round(1)

for perf_id, stats in performance_stats.iterrows():
    perf_name = performance_names[perf_id]
    count = (df['Performance'] == perf_id).sum()
    print(f"   {perf_name} ({count} نفر):")
    print(f"      ساعت مطالعه: {stats['Hours_Studied']}")
    print(f"      نمره قبلی: {stats['Previous_Scores']}")
    print(f"      ساعت خواب: {stats['Sleep_Hours']}")
    print(f"      فعالیت فوق‌برنامه: {stats['Extracurricular']*100:.0f}%")

# 6. نتیجه‌گیری
print(f"\n🎯 نتیجه‌گیری:")
print(f"✅ دقت بهترین مدل: {results[best_model]['accuracy']*100:.1f}%")
print(f"✅ تعداد خوشه بهینه: {best_k}")
print(f"✅ بالاترین همبستگی: {correlations.iloc[1]:.3f} ({correlations.index[1]})")

# عوامل کلیدی موفقیت
top_performers = df[df['Performance'] == 2]  # Excellent
if len(top_performers) > 0:
    print(f"\n🌟 مشخصات دانش‌آموزان برتر ({len(top_performers)} نفر):")
    print(f"   میانگین ساعت مطالعه: {top_performers['Hours_Studied'].mean():.1f}")
    print(f"   میانگین نمره قبلی: {top_performers['Previous_Scores'].mean():.1f}")
    print(f"   درصد فعالیت فوق‌برنامه: {top_performers['Extracurricular'].mean()*100:.0f}%")

print(f"\n💡 پیشنهادات:")
if correlations['Hours_Studied'] > 0.3:
    print(f"   📚 افزایش ساعات مطالعه تأثیر مثبت دارد")
if correlations['Previous_Scores'] > 0.5:
    print(f"   📊 نمرات قبلی پیش‌بینی‌کننده قوی هستند")
if correlations['Sleep_Hours'] > 0.1:
    print(f"   😴 خواب مناسب مهم است")

print(f"\n🎉 اجرای سریع پروژه تمام شد!")
print("=" * 60)

# ذخیره نتایج سریع
df.to_csv('quick_results.csv', index=False)
print(f"💾 نتایج در 'quick_results.csv' ذخیره شد")
