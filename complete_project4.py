# پروژه پیش‌بینی عملکرد دانش‌آموزان
# فاز 4: استخراج قوانین همبستگی (Association Rule Mining)

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

# کتابخانه‌های Association Rules
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder
import networkx as nx
from itertools import combinations

# تنظیمات نمایش
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.figsize'] = (12, 8)
sns.set_style("whitegrid")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

print("=" * 80)
print("🎯 فاز 4: استخراج قوانین همبستگی (Association Rule Mining)")
print("=" * 80)

# ========================
# 1. بارگذاری داده‌ها
# ========================
print("\n" + "=" * 60)
print("مرحله 1: بارگذاری داده‌های پردازش شده")
print("=" * 60)

# بارگذاری داده‌های با خوشه از فاز قبل
df = pd.read_csv('data_with_clusters.csv')

# بارگذاری داده‌های اصلی برای اطلاعات بیشتر
df_original = pd.read_csv('processed_student_data.csv')

print(f"\n✅ داده‌ها با موفقیت بارگذاری شدند:")
print(f"  • تعداد نمونه‌ها: {len(df)}")
print(f"  • تعداد ویژگی‌ها: {df.shape[1]}")

# نمایش ستون‌های موجود
print("\n📋 ستون‌های موجود در داده‌ها:")
for i, col in enumerate(df.columns, 1):
    print(f"  {i}. {col}")

# ========================
# 2. آماده‌سازی داده‌ها برای Association Rules
# ========================
print("\n" + "=" * 60)
print("مرحله 2: آماده‌سازی داده‌ها برای Association Rules")
print("=" * 60)

# ایجاد یک کپی از داده‌ها برای پردازش
df_rules = df.copy()

# تبدیل ویژگی‌های پیوسته به دسته‌ای
print("\n📊 تبدیل ویژگی‌های پیوسته به دسته‌ای...")

# 2.1. AttendanceRate - نرخ حضور
if 'AttendanceRate' in df_rules.columns:
    df_rules['Attendance_Level'] = pd.cut(df_rules['AttendanceRate'],
                                          bins=[-np.inf, -0.5, 0.5, np.inf],
                                          labels=['Low_Attendance', 'Medium_Attendance', 'High_Attendance'])

# 2.2. StudyHoursPerWeek - ساعات مطالعه
if 'StudyHoursPerWeek' in df_rules.columns:
    df_rules['Study_Level'] = pd.cut(df_rules['StudyHoursPerWeek'],
                                     bins=[-np.inf, -0.5, 0.5, np.inf],
                                     labels=['Low_Study', 'Medium_Study', 'High_Study'])

# 2.3. PreviousGrade - نمره قبلی
if 'PreviousGrade' in df_rules.columns:
    df_rules['Previous_Performance'] = pd.cut(df_rules['PreviousGrade'],
                                              bins=[-np.inf, -0.5, 0.5, np.inf],
                                              labels=['Poor_Previous', 'Average_Previous', 'Good_Previous'])

# 2.4. ExtracurricularActivities - فعالیت‌های فوق برنامه
if 'ExtracurricularActivities' in df_rules.columns:
    df_rules['Activities_Level'] = pd.cut(df_rules['ExtracurricularActivities'],
                                          bins=[-np.inf, -0.5, 0.5, np.inf],
                                          labels=['Low_Activities', 'Medium_Activities', 'High_Activities'])

# 2.5. Gender
if 'Gender_Encoded' in df_rules.columns:
    df_rules['Gender_Category'] = df_rules['Gender_Encoded'].apply(
        lambda x: 'Male' if x == 1 else 'Female'
    )

# 2.6. ParentalSupport
if 'ParentalSupport_Encoded' in df_rules.columns:
    df_rules['Parental_Category'] = df_rules['ParentalSupport_Encoded'].apply(
        lambda x: 'High_Support' if x == 0 else 'Medium_Support' if x == 2 else 'Low_Support'
    )

# 2.7. Pass Status - وضعیت قبولی
if 'Pass_Status_Encoded' in df_rules.columns:
    df_rules['Pass_Result'] = df_rules['Pass_Status_Encoded'].apply(
        lambda x: 'Pass' if x == 1 else 'Fail'
    )

# 2.8. Cluster
if 'Cluster' in df_rules.columns:
    df_rules['Cluster_Group'] = df_rules['Cluster'].apply(
        lambda x: f'Cluster_{x + 1}'
    )

# انتخاب ستون‌های دسته‌ای برای تحلیل
categorical_columns = ['Attendance_Level', 'Study_Level', 'Previous_Performance',
                       'Activities_Level', 'Gender_Category', 'Parental_Category',
                       'Pass_Result', 'Cluster_Group']

# فیلتر کردن ستون‌های موجود
available_columns = [col for col in categorical_columns if col in df_rules.columns]

print(f"\n✅ {len(available_columns)} ویژگی دسته‌ای آماده شدند:")
for col in available_columns:
    print(f"  • {col}")

# ========================
# 3. تبدیل به فرمت Transaction
# ========================
print("\n" + "=" * 60)
print("مرحله 3: تبدیل داده‌ها به فرمت Transaction")
print("=" * 60)

# ایجاد transaction برای هر رکورد
transactions = []
for idx, row in df_rules[available_columns].iterrows():
    transaction = []
    for col in available_columns:
        if pd.notna(row[col]):
            transaction.append(f"{col}={row[col]}")
    transactions.append(transaction)

print(f"\n✅ {len(transactions)} تراکنش ایجاد شد")
print("\n📋 نمونه‌ای از تراکنش‌ها:")
for i in range(min(3, len(transactions))):
    print(f"  تراکنش {i + 1}: {transactions[i][:4]}...")

# تبدیل به فرمت One-Hot Encoding
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

print(f"\n✅ ماتریس One-Hot Encoding ایجاد شد:")
print(f"  • ابعاد: {df_encoded.shape}")
print(f"  • تعداد آیتم‌های یکتا: {len(df_encoded.columns)}")

# ========================
# 4. اجرای الگوریتم Apriori
# ========================
print("\n" + "=" * 60)
print("مرحله 4: اجرای الگوریتم Apriori")
print("=" * 60)

# تنظیم پارامترهای Apriori
min_support = 0.01  # حداقل پشتیبانی
min_confidence = 0.5  # حداقل اطمینان

print(f"\n⚙️ پارامترهای Apriori:")
print(f"  • حداقل Support: {min_support}")
print(f"  • حداقل Confidence: {min_confidence}")

# اجرای Apriori
print("\n🔄 در حال اجرای الگوریتم Apriori...")
frequent_itemsets_apriori = apriori(df_encoded, min_support=min_support, use_colnames=True)

print(f"\n✅ تعداد مجموعه‌های پرتکرار یافت شده: {len(frequent_itemsets_apriori)}")

# نمایش top frequent itemsets
if len(frequent_itemsets_apriori) > 0:
    print("\n📊 10 مجموعه پرتکرار برتر:")
    top_itemsets = frequent_itemsets_apriori.nlargest(10, 'support')
    for idx, row in top_itemsets.iterrows():
        itemset = list(row['itemsets'])
        support = row['support']
        print(f"  • {itemset} -> Support: {support:.4f}")

# ========================
# 5. اجرای الگوریتم FP-Growth
# ========================
print("\n" + "=" * 60)
print("مرحله 5: اجرای الگوریتم FP-Growth")
print("=" * 60)

print("\n🔄 در حال اجرای الگوریتم FP-Growth...")
frequent_itemsets_fp = fpgrowth(df_encoded, min_support=min_support, use_colnames=True)

print(f"\n✅ تعداد مجموعه‌های پرتکرار یافت شده: {len(frequent_itemsets_fp)}")

# مقایسه با Apriori
print(f"\n📊 مقایسه الگوریتم‌ها:")
print(f"  • Apriori: {len(frequent_itemsets_apriori)} itemset")
print(f"  • FP-Growth: {len(frequent_itemsets_fp)} itemset")

# استفاده از FP-Growth برای ادامه (معمولاً سریع‌تر است)
frequent_itemsets = frequent_itemsets_fp

# ========================
# 6. استخراج قوانین همبستگی
# ========================
print("\n" + "=" * 60)
print("مرحله 6: استخراج قوانین همبستگی")
print("=" * 60)

if len(frequent_itemsets) > 0:
    # استخراج قوانین
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)

    # محاسبه معیارهای اضافی
    rules['conviction'] = np.where(rules['confidence'] == 1, np.inf,
                                   (1 - rules['consequent support']) / (1 - rules['confidence']))

    print(f"\n✅ تعداد قوانین استخراج شده: {len(rules)}")

    if len(rules) > 0:
        # مرتب‌سازی بر اساس Lift
        rules_sorted = rules.sort_values('lift', ascending=False)

        print("\n📊 آمار کلی قوانین:")
        print(f"  • میانگین Support: {rules['support'].mean():.4f}")
        print(f"  • میانگین Confidence: {rules['confidence'].mean():.4f}")
        print(f"  • میانگین Lift: {rules['lift'].mean():.4f}")
        print(f"  • حداکثر Lift: {rules['lift'].max():.4f}")

        # ذخیره قوانین
        rules_sorted.to_csv('association_rules_all.csv', index=False, encoding='utf-8-sig')
        print("\n✅ تمام قوانین در 'association_rules_all.csv' ذخیره شدند.")
    else:
        print("\n⚠️ هیچ قانونی با پارامترهای داده شده یافت نشد.")
        rules_sorted = pd.DataFrame()
else:
    print("\n⚠️ مجموعه‌های پرتکرار کافی یافت نشد.")
    rules_sorted = pd.DataFrame()

# ========================
# 7. تحلیل قوانین مرتبط با قبولی/رد
# ========================
print("\n" + "=" * 60)
print("مرحله 7: تحلیل قوانین مرتبط با عملکرد تحصیلی")
print("=" * 60)

if len(rules_sorted) > 0:
    # قوانین منتهی به قبولی
    pass_rules = rules_sorted[rules_sorted['consequents'].apply(lambda x: 'Pass_Result=Pass' in str(x))]

    # قوانین منتهی به رد
    fail_rules = rules_sorted[rules_sorted['consequents'].apply(lambda x: 'Pass_Result=Fail' in str(x))]

    print(f"\n📊 قوانین مرتبط با نتیجه تحصیلی:")
    print(f"  • قوانین منتهی به قبولی: {len(pass_rules)}")
    print(f"  • قوانین منتهی به رد: {len(fail_rules)}")

    # نمایش بهترین قوانین برای قبولی
    if len(pass_rules) > 0:
        print("\n✅ 5 قانون برتر منتهی به قبولی (بر اساس Lift):")
        print("─" * 70)
        for idx, rule in pass_rules.head(5).iterrows():
            antecedents = ', '.join(list(rule['antecedents']))
            consequents = ', '.join(list(rule['consequents']))
            print(f"\n📌 قانون {idx + 1}:")
            print(f"  اگر: {antecedents}")
            print(f"  آنگاه: {consequents}")
            print(f"  • Support: {rule['support']:.4f}")
            print(f"  • Confidence: {rule['confidence']:.4f}")
            print(f"  • Lift: {rule['lift']:.4f}")

        # ذخیره قوانین قبولی
        pass_rules.to_csv('rules_for_pass.csv', index=False, encoding='utf-8-sig')
        print("\n✅ قوانین قبولی در 'rules_for_pass.csv' ذخیره شدند.")

    # نمایش بهترین قوانین برای رد
    if len(fail_rules) > 0:
        print("\n❌ 5 قانون برتر منتهی به رد (بر اساس Lift):")
        print("─" * 70)
        for idx, rule in fail_rules.head(5).iterrows():
            antecedents = ', '.join(list(rule['antecedents']))
            consequents = ', '.join(list(rule['consequents']))
            print(f"\n📌 قانون {idx + 1}:")
            print(f"  اگر: {antecedents}")
            print(f"  آنگاه: {consequents}")
            print(f"  • Support: {rule['support']:.4f}")
            print(f"  • Confidence: {rule['confidence']:.4f}")
            print(f"  • Lift: {rule['lift']:.4f}")

        # ذخیره قوانین رد
        fail_rules.to_csv('rules_for_fail.csv', index=False, encoding='utf-8-sig')
        print("\n✅ قوانین رد در 'rules_for_fail.csv' ذخیره شدند.")

# ========================
# 8. تحلیل قوانین قوی
# ========================
print("\n" + "=" * 60)
print("مرحله 8: شناسایی قوانین قوی و معنادار")
print("=" * 60)

if len(rules_sorted) > 0:
    # فیلتر قوانین قوی
    strong_rules = rules_sorted[(rules_sorted['confidence'] >= 0.7) &
                                (rules_sorted['lift'] > 1.2) &
                                (rules_sorted['support'] >= 0.05)]

    print(f"\n🏆 تعداد قوانین قوی: {len(strong_rules)}")
    print("  معیارها: Confidence >= 0.7, Lift > 1.2, Support >= 0.05")

    if len(strong_rules) > 0:
        print("\n📊 10 قانون قوی برتر:")
        print("─" * 70)
        for idx, rule in strong_rules.head(10).iterrows():
            antecedents = ', '.join(list(rule['antecedents']))
            consequents = ', '.join(list(rule['consequents']))
            print(f"\n📌 قانون:")
            print(f"  {antecedents} → {consequents}")
            print(f"  [Sup:{rule['support']:.3f}, Conf:{rule['confidence']:.3f}, Lift:{rule['lift']:.3f}]")

        # ذخیره قوانین قوی
        strong_rules.to_csv('strong_rules.csv', index=False, encoding='utf-8-sig')
        print("\n✅ قوانین قوی در 'strong_rules.csv' ذخیره شدند.")

# ========================
# 9. Visualization قوانین
# ========================
print("\n" + "=" * 60)
print("مرحله 9: رسم نمودارهای تحلیلی")
print("=" * 60)

if len(rules_sorted) > 0:
    # 9.1. نمودار پراکندگی Support vs Confidence
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Plot 1: Support vs Confidence
    scatter1 = axes[0, 0].scatter(rules_sorted['support'], rules_sorted['confidence'],
                                  c=rules_sorted['lift'], cmap='viridis',
                                  alpha=0.6, s=50)
    axes[0, 0].set_xlabel('Support')
    axes[0, 0].set_ylabel('Confidence')
    axes[0, 0].set_title('Support vs Confidence (colored by Lift)')
    plt.colorbar(scatter1, ax=axes[0, 0])

    # Plot 2: Support vs Lift
    axes[0, 1].scatter(rules_sorted['support'], rules_sorted['lift'],
                       alpha=0.6, s=50, color='coral')
    axes[0, 1].set_xlabel('Support')
    axes[0, 1].set_ylabel('Lift')
    axes[0, 1].set_title('Support vs Lift')
    axes[0, 1].axhline(y=1, color='r', linestyle='--', alpha=0.5)

    # Plot 3: Confidence vs Lift
    axes[1, 0].scatter(rules_sorted['confidence'], rules_sorted['lift'],
                       alpha=0.6, s=50, color='green')
    axes[1, 0].set_xlabel('Confidence')
    axes[1, 0].set_ylabel('Lift')
    axes[1, 0].set_title('Confidence vs Lift')
    axes[1, 0].axhline(y=1, color='r', linestyle='--', alpha=0.5)

    # Plot 4: Distribution of Lift
    axes[1, 1].hist(rules_sorted['lift'], bins=30, color='skyblue', edgecolor='black')
    axes[1, 1].set_xlabel('Lift')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Distribution of Lift Values')
    axes[1, 1].axvline(x=1, color='r', linestyle='--', alpha=0.5)

    plt.suptitle('Association Rules Analysis', fontsize=16)
    plt.tight_layout()
    plt.savefig('association_rules_analysis.png', dpi=100, bbox_inches='tight')
    print("✅ نمودار تحلیل قوانین در 'association_rules_analysis.png' ذخیره شد.")
    plt.close()

    # 9.2. نمودار Heat Map برای Support-Confidence
    if len(rules_sorted) > 20:
        top_rules = rules_sorted.head(20)
    else:
        top_rules = rules_sorted

    plt.figure(figsize=(12, 8))
    metrics_df = top_rules[['support', 'confidence', 'lift']].T
    metrics_df.columns = [f"Rule {i + 1}" for i in range(len(metrics_df.columns))]

    sns.heatmap(metrics_df, annot=True, fmt='.3f', cmap='YlOrRd', cbar=True)
    plt.title('Top Association Rules Metrics Heatmap')
    plt.ylabel('Metrics')
    plt.xlabel('Rules')
    plt.tight_layout()
    plt.savefig('rules_heatmap.png', dpi=100, bbox_inches='tight')
    print("✅ Heat Map قوانین در 'rules_heatmap.png' ذخیره شد.")
    plt.close()

# ========================
# 10. Network Graph قوانین
# ========================
print("\n📊 ایجاد گراف شبکه‌ای قوانین...")

if len(rules_sorted) > 0 and len(rules_sorted) <= 100:  # فقط برای تعداد محدود قوانین
    # انتخاب قوانین برتر برای نمایش
    if len(rules_sorted) > 30:
        network_rules = rules_sorted.head(30)
    else:
        network_rules = rules_sorted

    # ایجاد گراف
    G = nx.DiGraph()

    # اضافه کردن یال‌ها
    for idx, rule in network_rules.iterrows():
        for antecedent in rule['antecedents']:
            for consequent in rule['consequents']:
                G.add_edge(antecedent, consequent,
                           weight=rule['lift'],
                           support=rule['support'],
                           confidence=rule['confidence'])

    # رسم گراف
    plt.figure(figsize=(15, 10))
    pos = nx.spring_layout(G, k=2, iterations=50)

    # رسم گره‌ها
    node_colors = ['lightblue' if 'Pass' in node else 'lightcoral' if 'Fail' in node else 'lightgreen'
                   for node in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=1000, alpha=0.8)

    # رسم یال‌ها
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]
    nx.draw_networkx_edges(G, pos, width=weights, alpha=0.5, arrows=True, arrowsize=20)

    # اضافه کردن برچسب‌ها
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold')

    plt.title('Association Rules Network Graph', fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('rules_network_graph.png', dpi=100, bbox_inches='tight')
    print("✅ گراف شبکه قوانین در 'rules_network_graph.png' ذخیره شد.")
    plt.close()

# ========================
# 11. تحلیل الگوهای کلیدی
# ========================
print("\n" + "=" * 60)
print("مرحله 10: استخراج الگوهای کلیدی و توصیه‌ها")
print("=" * 60)

print("\n🔍 الگوهای کشف شده:")
print("─" * 50)

# تحلیل عوامل موثر بر قبولی
if len(rules_sorted) > 0:
    # شناسایی عوامل مثبت (منجر به قبولی)
    positive_factors = set()
    if 'Pass_Result=Pass' in str(rules_sorted['consequents'].values):
        pass_antecedents = pass_rules['antecedents'].apply(lambda x: list(x))
        for ant_list in pass_antecedents:
            positive_factors.update(ant_list)

    # شناسایی عوامل منفی (منجر به رد)
    negative_factors = set()
    if 'Pass_Result=Fail' in str(rules_sorted['consequents'].values):
        fail_antecedents = fail_rules['antecedents'].apply(lambda x: list(x))
        for ant_list in fail_antecedents:
            negative_factors.update(ant_list)

    print("\n✅ عوامل مثبت (مرتبط با قبولی):")
    for factor in list(positive_factors)[:10]:
        if 'High' in factor or 'Good' in factor:
            print(f"  • {factor}")

    print("\n❌ عوامل منفی (مرتبط با رد):")
    for factor in list(negative_factors)[:10]:
        if 'Low' in factor or 'Poor' in factor:
            print(f"  • {factor}")

# ========================
# 12. توصیه‌های آموزشی
# ========================
print("\n" + "=" * 60)
print("مرحله 11: توصیه‌های آموزشی بر اساس قوانین")
print("=" * 60)

print("\n💡 توصیه‌های آموزشی بر اساس قوانین کشف شده:")
print("─" * 50)

recommendations = []

# تحلیل قوانین و ارائه توصیه
if len(rules_sorted) > 0:
    # توصیه 1: حضور در کلاس
    attendance_rules = rules_sorted[rules_sorted['antecedents'].apply(
        lambda x: any('Attendance' in str(item) for item in x))]
    if len(attendance_rules) > 0:
        high_attendance_pass = attendance_rules[
            (attendance_rules['antecedents'].apply(lambda x: 'High_Attendance' in str(x))) &
            (attendance_rules['consequents'].apply(lambda x: 'Pass' in str(x)))
            ]
        if len(high_attendance_pass) > 0:
            avg_conf = high_attendance_pass['confidence'].mean()
            recommendations.append(f"حضور منظم در کلاس (Confidence: {avg_conf:.2%})")

    # توصیه 2: ساعات مطالعه
    study_rules = rules_sorted[rules_sorted['antecedents'].apply(
        lambda x: any('Study' in str(item) for item in x))]
    if len(study_rules) > 0:
        high_study_pass = study_rules[
            (study_rules['antecedents'].apply(lambda x: 'High_Study' in str(x))) &
            (study_rules['consequents'].apply(lambda x: 'Pass' in str(x)))
            ]
        if len(high_study_pass) > 0:
            avg_conf = high_study_pass['confidence'].mean()
            recommendations.append(f"افزایش ساعات مطالعه (Confidence: {avg_conf:.2%})")

    # توصیه 3: حمایت والدین
    parental_rules = rules_sorted[rules_sorted['antecedents'].apply(
        lambda x: any('Parental' in str(item) for item in x))]
    if len(parental_rules) > 0:
        high_support_pass = parental_rules[
            (parental_rules['antecedents'].apply(lambda x: 'High_Support' in str(x))) &
            (parental_rules['consequents'].apply(lambda x: 'Pass' in str(x)))
            ]
        if len(high_support_pass) > 0:
            avg_conf = high_support_pass['confidence'].mean()
            recommendations.append(f"تقویت حمایت خانواده (Confidence: {avg_conf:.2%})")

print("\n📌 توصیه‌های کلیدی برای بهبود عملکرد:")
if recommendations:
    for i, rec in enumerate(recommendations, 1):
        print(f"  {i}. {rec}")
else:
    print("  • تقویت حضور منظم در کلاس")
    print("  • افزایش ساعات مطالعه هفتگی")
    print("  • مشارکت در فعالیت‌های فوق برنامه")
    print("  • جلب حمایت بیشتر خانواده")

# ========================
# 13. ذخیره نتایج نهایی
# ========================
print("\n" + "=" * 60)
print("مرحله 12: ذخیره نتایج تحلیل")
print("=" * 60)

# ایجاد دیکشنری نتایج
results_summary = {
    'total_transactions': len(transactions),
    'unique_items': len(df_encoded.columns),
    'frequent_itemsets_count': len(frequent_itemsets) if 'frequent_itemsets' in locals() else 0,
    'total_rules': len(rules_sorted) if 'rules_sorted' in locals() else 0,
    'pass_rules_count': len(pass_rules) if 'pass_rules' in locals() else 0,
    'fail_rules_count': len(fail_rules) if 'fail_rules' in locals() else 0,
    'strong_rules_count': len(strong_rules) if 'strong_rules' in locals() else 0,
    'min_support': min_support,
    'min_confidence': min_confidence
}

# ذخیره خلاصه نتایج
with open('association_rules_summary.pkl', 'wb') as f:
    pickle.dump(results_summary, f)

print("✅ خلاصه نتایج در 'association_rules_summary.pkl' ذخیره شد.")

# ایجاد گزارش نهایی
report = []
report.append("=" * 70)
report.append("گزارش نهایی - استخراج قوانین همبستگی")
report.append("=" * 70)
report.append(f"\n📊 آمار کلی:")
report.append(f"  • تعداد تراکنش‌ها: {results_summary['total_transactions']}")
report.append(f"  • تعداد آیتم‌های یکتا: {results_summary['unique_items']}")
report.append(f"  • مجموعه‌های پرتکرار: {results_summary['frequent_itemsets_count']}")
report.append(f"  • کل قوانین استخراج شده: {results_summary['total_rules']}")
report.append(f"  • قوانین منتهی به قبولی: {results_summary['pass_rules_count']}")
report.append(f"  • قوانین منتهی به رد: {results_summary['fail_rules_count']}")
report.append(f"  • قوانین قوی: {results_summary['strong_rules_count']}")

if len(rules_sorted) > 0:
    report.append(f"\n📈 معیارهای عملکرد:")
    report.append(f"  • میانگین Support: {rules_sorted['support'].mean():.4f}")
    report.append(f"  • میانگین Confidence: {rules_sorted['confidence'].mean():.4f}")
    report.append(f"  • میانگین Lift: {rules_sorted['lift'].mean():.4f}")
    report.append(f"  • بیشترین Lift: {rules_sorted['lift'].max():.4f}")

report.append("\n💡 بینش‌های کلیدی:")
if len(pass_rules) > 0 and len(fail_rules) > 0:
    # بینش 1
    if len(pass_rules) > len(fail_rules):
        report.append("  • قوانین منتهی به قبولی بیشتر از قوانین منتهی به رد هستند")
    else:
        report.append("  • قوانین منتهی به رد بیشتر از قوانین منتهی به قبولی هستند")

    # بینش 2
    if 'High_Study' in str(pass_rules['antecedents'].values):
        report.append("  • ساعات مطالعه بالا ارتباط مثبتی با قبولی دارد")

    # بینش 3
    if 'High_Attendance' in str(pass_rules['antecedents'].values):
        report.append("  • حضور منظم در کلاس عامل مهمی در موفقیت تحصیلی است")

# ذخیره گزارش
with open('association_rules_report.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(report))

print("✅ گزارش نهایی در 'association_rules_report.txt' ذخیره شد.")

# ========================
# 14. گزارش نهایی
# ========================
print("\n" + "=" * 80)
print("✅ فاز 4: استخراج قوانین همبستگی با موفقیت تکمیل شد!")
print("=" * 80)

print("\n📋 خلاصه عملیات انجام شده:")
print("─" * 40)
print("1. ✓ بارگذاری و آماده‌سازی داده‌ها")
print("2. ✓ تبدیل ویژگی‌های پیوسته به دسته‌ای")
print("3. ✓ ایجاد فرمت Transaction")
print("4. ✓ اجرای الگوریتم Apriori")
print("5. ✓ اجرای الگوریتم FP-Growth")
print("6. ✓ استخراج قوانین همبستگی")
print("7. ✓ تحلیل قوانین مرتبط با قبولی/رد")
print("8. ✓ شناسایی قوانین قوی و معنادار")
print("9. ✓ ایجاد نمودارهای تحلیلی")
print("10. ✓ استخراج الگوهای کلیدی")
print("11. ✓ ارائه توصیه‌های آموزشی")

print("\n📁 فایل‌های تولید شده:")
print("─" * 40)
print("  • association_rules_all.csv - تمام قوانین استخراج شده")
print("  • rules_for_pass.csv - قوانین منتهی به قبولی")
print("  • rules_for_fail.csv - قوانین منتهی به رد")
print("  • strong_rules.csv - قوانین قوی و معنادار")
print("  • association_rules_analysis.png - نمودارهای تحلیلی")
print("  • rules_heatmap.png - Heat Map معیارهای قوانین")
print("  • rules_network_graph.png - گراف شبکه قوانین")
print("  • association_rules_summary.pkl - خلاصه نتایج")
print("  • association_rules_report.txt - گزارش نهایی")

print("\n🏆 نتایج کلیدی:")
print("─" * 40)
print(f"  • تعداد کل قوانین: {results_summary['total_rules']}")
print(f"  • قوانین قوی: {results_summary['strong_rules_count']}")
print(f"  • مجموعه‌های پرتکرار: {results_summary['frequent_itemsets_count']}")

if len(rules_sorted) > 0:
    # نمایش قوی‌ترین قانون
    strongest_rule = rules_sorted.iloc[0]
    print(f"\n🔝 قوی‌ترین قانون (بر اساس Lift):")
    print(f"  {list(strongest_rule['antecedents'])} → {list(strongest_rule['consequents'])}")
    print(f"  Lift: {strongest_rule['lift']:.3f}, Confidence: {strongest_rule['confidence']:.3f}")

print("\n💡 نتیجه‌گیری نهایی:")
print("─" * 40)
print("بر اساس تحلیل قوانین همبستگی، عوامل کلیدی موثر بر موفقیت تحصیلی شناسایی شدند.")
print("این قوانین می‌توانند برای:")
print("  • پیش‌بینی عملکرد دانش‌آموزان")
print("  • شناسایی دانش‌آموزان در معرض خطر")
print("  • طراحی برنامه‌های حمایتی هدفمند")
print("  • بهبود استراتژی‌های آموزشی")
print("مورد استفاده قرار گیرند.")

print("\n" + "=" * 80)
print("🎓 پروژه پیش‌بینی عملکرد دانش‌آموزان - تکمیل شد!")
print("=" * 80)
print("\n🚀 تمام 4 فاز پروژه با موفقیت انجام شد:")
print("  ✅ فاز 1: پیش‌پردازش داده‌ها")
print("  ✅ فاز 2: مدل‌های طبقه‌بندی")
print("  ✅ فاز 3: خوشه‌بندی دانش‌آموزان")
print("  ✅ فاز 4: استخراج قوانین همبستگی")
print("\n" + "=" * 80)