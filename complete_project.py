#!/usr/bin/env python3
"""
پروژه کامل پیش‌بینی عملکرد دانش‌آموزان
شامل: پیش‌پردازش، طبقه‌بندی، خوشه‌بندی، قوانین همبستگی

نویسنده: پروژه درس داده‌کاوی
تاریخ: 1404
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, silhouette_score
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import warnings
warnings.filterwarnings('ignore')

def main():
    """اجرای کامل پروژه"""
    
    print("🎓 پروژه پیش‌بینی عملکرد دانش‌آموزان")
    print("=" * 60)
    
    # مرحله 1: ایجاد و پیش‌پردازش داده‌ها
    df, X_train, X_test, y_train, y_test, X_encoded, scaler = create_and_preprocess_data()
    
    # مرحله 2: مدل‌های طبقه‌بندی
    model_results, best_model_name = train_classification_models(X_train, X_test, y_train, y_test)
    
    # مرحله 3: خوشه‌بندی
    cluster_results = perform_clustering(df)
    
    # مرحله 4: قوانین همبستگی
    association_results = extract_association_rules(df)
    
    # مرحله 5: ارزیابی نهایی
    final_evaluation(df, model_results, cluster_results, association_results, best_model_name)
    
    print("\n🎉 پروژه با موفقیت تکمیل شد!")

def create_and_preprocess_data():
    """ایجاد و پیش‌پردازش داده‌ها"""
    print("\n📊 مرحله 1: ایجاد و پیش‌پردازش داده‌ها...")
    
    np.random.seed(42)
    n_students = 1500
    
    # ایجاد دیتاست
    data = {
        'Hours_Studied': np.random.normal(7, 3, n_students).clip(1, 20),
        'Previous_Scores': np.random.normal(75, 15, n_students).clip(40, 100),
        'Extracurricular_Activities': np.random.choice([0, 1], n_students, p=[0.4, 0.6]),
        'Sleep_Hours': np.random.normal(7, 1.5, n_students).clip(4, 12),
        'Sample_Question_Papers_Practiced': np.random.randint(0, 10, n_students),
        'Performance_Index': np.random.normal(50, 20, n_students).clip(10, 100)
    }
    
    df = pd.DataFrame(data)
    
    # اضافه کردن ویژگی‌های دسته‌ای
    df['Gender'] = np.random.choice(['Male', 'Female'], n_students)
    df['Parental_Involvement'] = np.random.choice(['Low', 'Medium', 'High'], n_students, p=[0.3, 0.4, 0.3])
    df['Access_to_Resources'] = np.random.choice(['Low', 'Medium', 'High'], n_students, p=[0.25, 0.5, 0.25])
    df['Motivation_Level'] = np.random.choice(['Low', 'Medium', 'High'], n_students, p=[0.2, 0.6, 0.2])
    df['Internet_Access'] = np.random.choice(['Yes', 'No'], n_students, p=[0.85, 0.15])
    df['Family_Income'] = np.random.choice(['Low', 'Medium', 'High'], n_students, p=[0.3, 0.5, 0.2])
    df['Teacher_Quality'] = np.random.choice(['Low', 'Medium', 'High'], n_students, p=[0.2, 0.6, 0.2])
    
    # ایجاد متغیر هدف
    def calculate_performance_category(row):
        score = 0
        score += row['Hours_Studied'] * 3
        score += row['Previous_Scores'] * 0.4
        score += row['Extracurricular_Activities'] * 8
        
        if 6 <= row['Sleep_Hours'] <= 8:
            score += 10
        
        score += row['Sample_Question_Papers_Practiced'] * 2
        
        # تأثیر ویژگی‌های دسته‌ای
        if row['Parental_Involvement'] == 'High':
            score += 15
        elif row['Parental_Involvement'] == 'Medium':
            score += 8
            
        if row['Access_to_Resources'] == 'High':
            score += 12
        elif row['Access_to_Resources'] == 'Medium':
            score += 6
            
        if row['Motivation_Level'] == 'High':
            score += 10
        elif row['Motivation_Level'] == 'Medium':
            score += 5
            
        score += np.random.normal(0, 5)
        
        if score < 50:
            return 0  # Poor
        elif score < 80:
            return 1  # Average
        else:
            return 2  # Excellent
    
    df['Performance_Category'] = df.apply(calculate_performance_category, axis=1)
    performance_labels = {0: 'Poor', 1: 'Average', 2: 'Excellent'}
    df['Performance_Label'] = df['Performance_Category'].map(performance_labels)
    
    print(f"✅ دیتاست با {len(df)} دانش‌آموز ایجاد شد")
    print(f"📈 توزیع عملکرد: {df['Performance_Label'].value_counts().to_dict()}")
    
    # پیش‌پردازش
    features_for_modeling = ['Hours_Studied', 'Previous_Scores', 'Extracurricular_Activities', 
                            'Sleep_Hours', 'Sample_Question_Papers_Practiced', 'Gender', 
                            'Parental_Involvement', 'Access_to_Resources', 'Motivation_Level',
                            'Internet_Access', 'Family_Income', 'Teacher_Quality']
    
    X = df[features_for_modeling].copy()
    y = df['Performance_Category']
    
    # کدگذاری One-Hot
    categorical_features = ['Gender', 'Parental_Involvement', 'Access_to_Resources', 'Motivation_Level',
                           'Internet_Access', 'Family_Income', 'Teacher_Quality']
    X_encoded = pd.get_dummies(X, columns=categorical_features, prefix=categorical_features)
    
    # نرمال‌سازی
    numerical_features = ['Hours_Studied', 'Previous_Scores', 'Sleep_Hours', 'Sample_Question_Papers_Practiced', 'Extracurricular_Activities']
    scaler = StandardScaler()
    X_encoded[numerical_features] = scaler.fit_transform(X_encoded[numerical_features])
    
    # تقسیم داده‌ها
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"✅ پیش‌پردازش کامل - آموزشی: {len(X_train)}, تست: {len(X_test)}")
    
    # ذخیره
    df.to_csv('student_performance_dataset.csv', index=False)
    
    return df, X_train, X_test, y_train, y_test, X_encoded, scaler

def train_classification_models(X_train, X_test, y_train, y_test):
    """آموزش مدل‌های طبقه‌بندی"""
    print("\n🤖 مرحله 2: مدل‌های طبقه‌بندی...")
    
    models = {
        'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
        'SVM': SVC(random_state=42, probability=True, kernel='rbf'),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
    }
    
    model_results = {}
    
    for model_name, model in models.items():
        print(f"🔄 آموزش {model_name}...")
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # ارزیابی
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        
        model_results[model_name] = {
            'model': model,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'cv_mean': cv_scores.mean(),
            'y_pred': y_pred
        }
        
        print(f"   ✅ F1-Score: {f1:.4f}, CV: {cv_scores.mean():.4f}")
    
    # بهترین مدل
    best_model_name = max(model_results.keys(), key=lambda x: model_results[x]['f1'])
    print(f"\n🏆 بهترین مدل: {best_model_name}")
    
    return model_results, best_model_name

def perform_clustering(df):
    """خوشه‌بندی دانش‌آموزان"""
    print("\n🔗 مرحله 3: خوشه‌بندی...")
    
    clustering_features = ['Hours_Studied', 'Previous_Scores', 'Sleep_Hours', 'Sample_Question_Papers_Practiced']
    X_clustering = df[clustering_features].copy()
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clustering)
    
    # تعیین تعداد بهینه خوشه‌ها
    silhouette_scores = []
    k_range = range(2, 8)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        silhouette_scores.append(score)
    
    best_k = k_range[np.argmax(silhouette_scores)]
    
    # خوشه‌بندی نهایی
    kmeans_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    cluster_labels = kmeans_final.fit_predict(X_scaled)
    
    df_with_clusters = df.copy()
    df_with_clusters['Cluster'] = cluster_labels
    
    print(f"✅ خوشه‌بندی با {best_k} خوشه انجام شد")
    print(f"📊 Silhouette Score: {silhouette_score(X_scaled, cluster_labels):.4f}")
    
    # ذخیره
    df_with_clusters.to_csv('students_with_clusters.csv', index=False)
    
    return {
        'best_k': best_k,
        'silhouette_score': silhouette_score(X_scaled, cluster_labels),
        'cluster_labels': cluster_labels,
        'df_with_clusters': df_with_clusters
    }

def extract_association_rules(df):
    """استخراج قوانین همبستگی"""
    print("\n🔗 مرحله 4: قوانین همبستگی...")
    
    try:
        # دسته‌بندی ویژگی‌های عددی
        df_rules = df.copy()
        
        df_rules['Study_Hours_Cat'] = pd.cut(df['Hours_Studied'], bins=[0, 5, 10, 20], labels=['کم', 'متوسط', 'زیاد'])
        df_rules['Previous_Scores_Cat'] = pd.cut(df['Previous_Scores'], bins=[0, 60, 80, 100], labels=['ضعیف', 'متوسط', 'عالی'])
        df_rules['Sleep_Cat'] = pd.cut(df['Sleep_Hours'], bins=[0, 6, 8, 12], labels=['کم', 'مناسب', 'زیاد'])
        
        # آماده‌سازی تراکنش‌ها
        features_for_rules = ['Study_Hours_Cat', 'Previous_Scores_Cat', 'Sleep_Cat', 
                             'Parental_Involvement', 'Access_to_Resources', 'Performance_Label']
        
        transactions = []
        for idx, row in df_rules.iterrows():
            transaction = []
            for feature in features_for_rules:
                if pd.notna(row[feature]):
                    transaction.append(f"{feature}_{row[feature]}")
            transactions.append(transaction)
        
        # تبدیل به فرمت مناسب
        te = TransactionEncoder()
        te_ary = te.fit(transactions).transform(transactions)
        df_transactions = pd.DataFrame(te_ary, columns=te.columns_)
        
        # استخراج الگوهای پرتکرار
        frequent_itemsets = apriori(df_transactions, min_support=0.1, use_colnames=True)
        
        if len(frequent_itemsets) > 0:
            # استخراج قوانین
            rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.3)
            
            if len(rules) > 0:
                rules_sorted = rules.sort_values('lift', ascending=False)
                print(f"✅ {len(rules)} قانون همبستگی یافت شد")
                
                return {
                    'rules_count': len(rules),
                    'best_lift': rules['lift'].max(),
                    'avg_confidence': rules['confidence'].mean(),
                    'rules': rules_sorted
                }
            else:
                print("❌ هیچ قانونی یافت نشد")
                return {'rules_count': 0}
        else:
            print("❌ هیچ الگوی پرتکراری یافت نشد")
            return {'rules_count': 0}
            
    except Exception as e:
        print(f"❌ خطا در استخراج قوانین: {e}")
        return {'rules_count': 0}

def final_evaluation(df, model_results, cluster_results, association_results, best_model_name):
    """ارزیابی نهایی و گزارش"""
    print("\n📊 مرحله 5: ارزیابی نهایی...")
    
    # خلاصه نتایج
    print("=" * 60)
    print("📋 خلاصه نهایی پروژه")
    print("=" * 60)
    
    print(f"📊 اطلاعات دیتاست:")
    print(f"   • تعداد دانش‌آموزان: {len(df)}")
    print(f"   • توزیع عملکرد: {df['Performance_Label'].value_counts().to_dict()}")
    
    print(f"\n🤖 نتایج طبقه‌بندی:")
    print(f"   • بهترین مدل: {best_model_name}")
    print(f"   • F1-Score: {model_results[best_model_name]['f1']:.4f}")
    print(f"   • Accuracy: {model_results[best_model_name]['accuracy']:.4f}")
    
    print(f"\n🔗 نتایج خوشه‌بندی:")
    print(f"   • تعداد خوشه بهینه: {cluster_results['best_k']}")
    print(f"   • Silhouette Score: {cluster_results['silhouette_score']:.4f}")
    
    print(f"\n📈 نتایج قوانین همبستگی:")
    print(f"   • تعداد قوانین: {association_results['rules_count']}")
    if association_results['rules_count'] > 0:
        print(f"   • بالاترین Lift: {association_results['best_lift']:.3f}")
        print(f"   • میانگین اعتماد: {association_results['avg_confidence']:.3f}")
    
    # ایجاد گزارش مقایسه مدل‌ها
    results_df = pd.DataFrame({
        'Model': list(model_results.keys()),
        'Accuracy': [model_results[m]['accuracy'] for m in model_results.keys()],
        'Precision': [model_results[m]['precision'] for m in model_results.keys()],
        'Recall': [model_results[m]['recall'] for m in model_results.keys()],
        'F1-Score': [model_results[m]['f1'] for m in model_results.keys()],
        'CV_Mean': [model_results[m]['cv_mean'] for m in model_results.keys()]
    })
    
    print(f"\n📊 مقایسه مدل‌ها:")
    print(results_df.round(4).to_string(index=False))
    
    # تحلیل عوامل مؤثر
    print(f"\n🔍 عوامل مؤثر در عملکرد:")
    
    # همبستگی با Performance_Index
    numeric_features = ['Hours_Studied', 'Previous_Scores', 'Sleep_Hours', 'Sample_Question_Papers_Practiced']
    correlations = df[numeric_features + ['Performance_Index']].corr()['Performance_Index'].sort_values(ascending=False)
    
    print("   📊 همبستگی با شاخص عملکرد:")
    for feature, corr in correlations.items():
        if feature != 'Performance_Index':
            print(f"      • {feature}: {corr:.3f}")
    
    # تحلیل گروه‌های عملکرد
    print(f"\n📈 مشخصات گروه‌های عملکرد:")
    for label in ['Poor', 'Average', 'Excellent']:
        group = df[df['Performance_Label'] == label]
        if len(group) > 0:
            print(f"   🔸 {label} ({len(group)} نفر):")
            print(f"      • میانگین ساعت مطالعه: {group['Hours_Studied'].mean():.1f}")
            print(f"      • میانگین نمره قبلی: {group['Previous_Scores'].mean():.1f}")
            print(f"      • میانگین خواب: {group['Sleep_Hours'].mean():.1f}")
    
    # پیشنهادات
    print(f"\n💡 پیشنهادات کلیدی:")
    print("   1️⃣ افزایش ساعات مطالعه مؤثرترین عامل است")
    print("   2️⃣ نمرات قبلی نشان‌دهنده قوی عملکرد آینده هستند")
    print("   3️⃣ خواب مناسب (6-8 ساعت) حائز اهمیت است")
    print("   4️⃣ درگیری والدین و دسترسی به منابع نقش کلیدی دارند")
    
    # ذخیره گزارش نهایی
    results_df.to_csv('model_comparison_results.csv', index=False)
    
    # آمار خوشه‌ها
    if 'df_with_clusters' in cluster_results:
        cluster_stats = cluster_results['df_with_clusters'].groupby('Cluster').agg({
            'Hours_Studied': 'mean',
            'Previous_Scores': 'mean',
            'Performance_Label': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'Mixed'
        }).round(2)
        
        print(f"\n🔗 آمار خوشه‌ها:")
        print(cluster_stats.to_string())
    
    print(f"\n✅ فایل‌های خروجی:")
    print(f"   📄 student_performance_dataset.csv")
    print(f"   📄 students_with_clusters.csv") 
    print(f"   📄 model_comparison_results.csv")
    
    print(f"\n🎯 پروژه شامل موارد زیر بود:")
    print(f"   ✅ پیش‌پردازش داده‌ها")
    print(f"   ✅ 4 مدل طبقه‌بندی با Cross-Validation")
    print(f"   ✅ خوشه‌بندی با K-Means")
    print(f"   ✅ قوانین همبستگی با Apriori")
    print(f"   ✅ ارزیابی جامع و تحلیل نتایج")

# نقطه شروع برنامه
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ خطا در اجرای پروژه: {e}")
        print("💡 لطفاً مطمئن شوید که تمام کتابخانه‌های مورد نیاز نصب شده‌اند")

# راهنمای نصب کتابخانه‌ها:
"""
pip install pandas numpy matplotlib seaborn scikit-learn mlxtend

یا برای نصب همه:
pip install -r requirements.txt

محتویات requirements.txt:
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
mlxtend>=0.19.0
"""
