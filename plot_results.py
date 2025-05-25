import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score, ConfusionMatrixDisplay, roc_auc_score
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import label_binarize
import seaborn as sns

# Read loss curve
loss_df = pd.read_csv('loss_curve.csv')
plt.figure()
plt.plot(loss_df['loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.grid(True)
plt.savefig('loss_curve.png')
plt.show()
plt.close()

# Read predictions
pred_df = pd.read_csv('predictions.csv')

# Prepare x-tick labels with gender, age, and actual label
gender_map = {'male': 'M', 'female': 'F'}
x_labels = [
    f"{gender_map.get(row['sex'], row['sex'])},{int(row['age'])}\n{row['actual']}"
    for _, row in pred_df.iterrows()
]

plt.figure(figsize=(max(8, len(x_labels) * 0.7), 6))
plt.plot(pred_df['predicted'], label='Predicted', marker='o', color='tab:blue')
plt.plot(pred_df['actual_num'], label='Actual', marker='x', color='tab:orange', linestyle='None')
plt.xlabel('Sample (Gender, Age, Actual)')
plt.ylabel('Value')
plt.title('Predicted vs Actual')
plt.xticks(ticks=range(len(x_labels)), labels=x_labels, rotation=45, ha='right')
plt.legend()
plt.tight_layout()
plt.grid(True)
plt.savefig('pred_vs_actual.png')
plt.show()
plt.close()

# 1. Confusion Matrix
thresh = 0.5
y_true = pred_df['actual_num']
y_pred_bin = (pred_df['predicted'] >= thresh).astype(int)
cm = confusion_matrix(y_true, y_pred_bin)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Not Buy', 'Buy'])
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
plt.show()
plt.close()

# 2. ROC Curve and AUC
fpr, tpr, _ = roc_curve(y_true, pred_df['predicted'])
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.grid(True)
plt.savefig('roc_curve.png')
plt.show()
plt.close()

# 3. Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_true, pred_df['predicted'])
avg_prec = average_precision_score(y_true, pred_df['predicted'])
plt.figure()
plt.plot(recall, precision, color='purple', lw=2, label=f'AP = {avg_prec:.2f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.grid(True)
plt.savefig('precision_recall_curve.png')
plt.show()
plt.close()

# 4. Histogram of Predicted Probabilities
plt.figure()
plt.hist(pred_df['predicted'], bins=10, color='skyblue', edgecolor='black')
plt.xlabel('Predicted Probability')
plt.ylabel('Frequency')
plt.title('Histogram of Predicted Probabilities')
plt.grid(True)
plt.savefig('predicted_prob_hist.png')
plt.show()
plt.close()

# 5. Feature Importance/Weight Plot (manual entry)
def load_valid_weights(filename, n_features=2):
    valid_weights = []
    try:
        with open(filename, 'r') as f:
            for i, line in enumerate(f, 1):
                parts = line.strip().split()
                if len(parts) == n_features:
                    valid_weights.append([float(x) for x in parts])
                else:
                    print(f'Warning: Skipping line {i} in {filename} (expected {n_features} values, got {len(parts)})')
        if not valid_weights:
            print(f'No valid weights found in {filename}.')
        return np.array(valid_weights)
    except FileNotFoundError:
        print(f'Feature importance plot skipped ({filename} not found).')
        return None

weights = load_valid_weights('weights.txt', n_features=2)
if weights is not None and len(weights) > 0:
    features = ['sex', 'age']
    plt.figure()
    plt.bar(features, weights[-1])  # Use last run
    plt.title('Feature Importance (Weights)')
    plt.ylabel('Weight Value')
    plt.grid(True)
    plt.savefig('feature_importance.png')
    plt.show()
    plt.close()

# 6. Scatter Plot of Age vs. Predicted Probability (colored by gender)
plt.figure()
for gender in pred_df['sex'].unique():
    mask = pred_df['sex'] == gender
    plt.scatter(pred_df.loc[mask, 'age'], pred_df.loc[mask, 'predicted'], label=gender_map.get(gender, gender), alpha=0.7)
plt.xlabel('Age')
plt.ylabel('Predicted Probability')
plt.title('Predicted Probability vs. Age by Gender')
plt.legend()
plt.grid(True)
plt.savefig('age_vs_predicted.png')
plt.show()
plt.close()

# 7. Residual Plot
residuals = pred_df['actual_num'] - pred_df['predicted']
plt.figure()
plt.scatter(range(len(residuals)), residuals, color='red', alpha=0.7)
plt.axhline(0, color='black', linestyle='--')
plt.xlabel('Sample')
plt.ylabel('Residual (Actual - Predicted)')
plt.title('Residual Plot')
plt.grid(True)
plt.savefig('residual_plot.png')
plt.show()
plt.close()

# 8. Calibration Curve
prob_true, prob_pred = calibration_curve(y_true, pred_df['predicted'], n_bins=10)
plt.figure()
plt.plot(prob_pred, prob_true, marker='o', label='Calibration curve')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly calibrated')
plt.xlabel('Mean Predicted Probability')
plt.ylabel('Fraction of Positives')
plt.title('Calibration Curve')
plt.legend()
plt.grid(True)
plt.savefig('calibration_curve.png')
plt.show()
plt.close()

# --- Advanced/Additional Plots ---

# 1. Learning Curves (simulated validation)
plt.figure()
plt.plot(loss_df['loss'], label='Training Loss')
# Simulate validation loss as slightly higher
plt.plot(loss_df['loss'] * 1.1, label='Validation Loss', linestyle='--')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Learning Curves (Simulated)')
plt.legend()
plt.grid(True)
plt.savefig('learning_curves.png')
plt.show()
plt.close()

# 2. Validation Curves (simulated)
# Simulate a hyperparameter sweep (e.g., regularization strength)
hyperparams = np.logspace(-3, 1, 10)
train_scores = np.exp(-hyperparams)
val_scores = np.exp(-hyperparams * 1.2)
plt.figure()
plt.plot(hyperparams, train_scores, label='Train Score')
plt.plot(hyperparams, val_scores, label='Validation Score')
plt.xscale('log')
plt.xlabel('Hyperparameter (Simulated)')
plt.ylabel('Score')
plt.title('Validation Curve (Simulated)')
plt.legend()
plt.grid(True)
plt.savefig('validation_curve.png')
plt.show()
plt.close()

# 3. Cumulative Gains and Lift Chart
try:
    from sklearn.metrics import roc_curve
    sorted_idx = np.argsort(-pred_df['predicted'])
    sorted_actual = pred_df['actual_num'].values[sorted_idx]
    cum_gains = np.cumsum(sorted_actual) / sorted_actual.sum()
    perc_samples = np.arange(1, len(sorted_actual) + 1) / len(sorted_actual)
    plt.figure()
    plt.plot(perc_samples, cum_gains, label='Cumulative Gains')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Baseline')
    plt.xlabel('Proportion of Samples')
    plt.ylabel('Cumulative Proportion of Positives')
    plt.title('Cumulative Gains Chart')
    plt.legend()
    plt.grid(True)
    plt.savefig('cumulative_gains.png')
    plt.show()
    plt.close()
except Exception as e:
    print('Cumulative gains plot skipped:', e)

# 4. Partial Dependence Plot (PDP) for age
try:
    if weights is not None and len(weights) > 0:
        from sklearn.inspection import partial_dependence, PartialDependenceDisplay
        class DummyModel:
            def __init__(self, w, b):
                self.w = w
                self.b = b
            def predict(self, X):
                return 1 / (1 + np.exp(-(X @ self.w + self.b)))
        w = weights[-1]
        b = 0  # bias not saved, so set to 0
        model = DummyModel(w, b)
        X = pred_df[['sex', 'age']].copy()
        X['sex'] = X['sex'].map({'male': 0, 'female': 1})
        X = X.values
        fig, ax = plt.subplots()
        ages = np.linspace(X[:,1].min(), X[:,1].max(), 100)
        X_pdp = np.column_stack([np.full_like(ages, X[:,0].mean()), ages])
        preds = model.predict(X_pdp)
        ax.plot(ages, preds)
        ax.set_xlabel('Age')
        ax.set_ylabel('Predicted Probability')
        ax.set_title('Partial Dependence Plot (Age)')
        plt.savefig('pdp_age.png')
        plt.show()
        plt.close()
    else:
        print('Partial dependence plot skipped: No valid weights found.')
except Exception as e:
    print('Partial dependence plot skipped:', e)

# 5. SHAP/Feature Contribution Plot (placeholder)
try:
    if weights is not None and len(weights) > 0:
        import shap
        w = weights[-1]
        b = 0
        class DummyModel:
            def predict(self, X):
                return 1 / (1 + np.exp(-(X @ w + b)))
        model = DummyModel()
        X = pred_df[['sex', 'age']].copy()
        X['sex'] = X['sex'].map({'male': 0, 'female': 1})
        X = X.values
        explainer = shap.Explainer(model.predict, X)
        shap_values = explainer(X)
        shap.summary_plot(shap_values, X, feature_names=['sex', 'age'], show=False)
        plt.savefig('shap_summary.png')
        plt.show()
        plt.close()
    else:
        print('SHAP plot skipped: No valid weights found.')
except Exception as e:
    print('SHAP plot skipped:', e)

# 6. t-SNE/PCA Plot
try:
    from sklearn.decomposition import PCA
    X = pred_df[['sex', 'age']].copy()
    X['sex'] = X['sex'].map({'male': 0, 'female': 1})
    X = X.values
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    plt.figure()
    plt.scatter(X_pca[:,0], X_pca[:,1], c=pred_df['predicted'], cmap='coolwarm', edgecolor='k')
    plt.colorbar(label='Predicted Probability')
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.title('PCA of Input Features (colored by prediction)')
    plt.savefig('pca_plot.png')
    plt.show()
    plt.close()
except Exception as e:
    print('PCA plot skipped:', e)

# 7. Error Analysis Plot (errors vs. age)
plt.figure()
plt.scatter(pred_df['age'], residuals, c=pred_df['actual_num'], cmap='bwr', edgecolor='k')
plt.axhline(0, color='black', linestyle='--')
plt.xlabel('Age')
plt.ylabel('Residual (Actual - Predicted)')
plt.title('Residuals vs. Age')
plt.grid(True)
plt.savefig('residuals_vs_age.png')
plt.show()
plt.close()

# 8. Boxplot of Predictions by Gender
plt.figure()
sns.boxplot(x='sex', y='predicted', data=pred_df)
plt.xlabel('Gender')
plt.ylabel('Predicted Probability')
plt.title('Predicted Probability by Gender')
plt.savefig('boxplot_gender.png')
plt.show()
plt.close()

# 9. Time Series Plot (by sample index)
plt.figure()
plt.plot(pred_df['predicted'], label='Predicted', marker='o')
plt.plot(pred_df['actual_num'], label='Actual', marker='x')
plt.xlabel('Sample Index (Time)')
plt.ylabel('Value')
plt.title('Predicted vs Actual Over Time')
plt.legend()
plt.grid(True)
plt.savefig('time_series_plot.png')
plt.show()
plt.close()

# 10. Heatmap of Feature Correlation
plt.figure()
corr = pred_df[['age', 'predicted', 'actual_num']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Feature Correlation Heatmap')
plt.savefig('correlation_heatmap.png')
plt.show()
plt.close()

# 11. Custom Domain-Specific Plot (placeholder)
# Example: Bar plot of actual purchases by gender
plt.figure()
purchase_counts = pred_df.groupby(['sex', 'actual']).size().unstack(fill_value=0)
purchase_counts.plot(kind='bar', stacked=True)
plt.xlabel('Gender')
plt.ylabel('Count')
plt.title('Actual Purchases by Gender')
plt.tight_layout()
plt.savefig('domain_specific_plot.png')
plt.show()
plt.close()

# Correlation between sex and buy decision (fries, hamburger, none)
plt.figure()
correlation_counts = pred_df.groupby(['sex', 'actual']).size().unstack(fill_value=0)
correlation_counts = correlation_counts[['fries', 'hamburger', 'none']] if 'none' in correlation_counts.columns else correlation_counts[['fries', 'hamburger']]
correlation_counts.plot(kind='bar', stacked=False)
plt.xlabel('Sex')
plt.ylabel('Count')
plt.title('Correlation between Sex and Buy Decision')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('sex_vs_buy_decision.png')
plt.show()
plt.close()

# 1. Percentages between sexes for buy or no buy
buy_or_no = pred_df.copy()
buy_or_no['buy'] = buy_or_no['actual'].apply(lambda x: 'buy' if x in ['fries', 'hamburger'] else 'no buy')
sex_buy_pct = buy_or_no.groupby(['sex', 'buy']).size().unstack(fill_value=0)
sex_buy_pct = sex_buy_pct.div(sex_buy_pct.sum(axis=1), axis=0) * 100
sex_buy_pct[['buy', 'no buy']].plot(kind='bar', stacked=True)
plt.ylabel('Percentage')
plt.title('Percentage of Buy vs No Buy by Sex')
plt.xlabel('Sex')
plt.legend(title='Decision')
plt.tight_layout()
plt.savefig('sex_buy_percentage.png')
plt.show()
plt.close()

# 2. Percentage of what was bought by age and sex
age_sex_buy = pred_df.groupby(['age', 'sex', 'actual']).size().unstack(fill_value=0)
age_sex_buy_pct = age_sex_buy.div(age_sex_buy.sum(axis=1), axis=0) * 100
plt.figure(figsize=(10, 6))
sns.heatmap(age_sex_buy_pct, annot=True, fmt='.1f', cmap='Blues')
plt.title('Percentage of What Was Bought by Age and Sex')
plt.ylabel('Age, Sex')
plt.xlabel('What Was Bought')
plt.tight_layout()
plt.savefig('age_sex_buy_percentage.png')
plt.show()
plt.close()

# Predict what will be bought (fries, hamburger, or none) by age and sex
print("\nPrediction by age and sex (using latest weights):")
weights_fries = load_valid_weights('weights_fries.txt', n_features=2)
weights_hamburger = load_valid_weights('weights_hamburger.txt', n_features=2)
if (weights_fries is not None and len(weights_fries) > 0) and (weights_hamburger is not None and len(weights_hamburger) > 0):
    w_fries = weights_fries[-1]
    w_hamburger = weights_hamburger[-1]
    b = 0  # bias not saved
    unique_ages = sorted(pred_df['age'].unique())
    unique_sexes = pred_df['sex'].unique()
    pred_table = []
    for sex in unique_sexes:
        sex_num = 0 if sex == 'male' else 1
        for age in unique_ages:
            x = np.array([sex_num, age])
            pred_fries = 1 / (1 + np.exp(-(np.dot(x, w_fries) + b)))
            pred_hamburger = 1 / (1 + np.exp(-(np.dot(x, w_hamburger) + b)))
            # Decision logic
            if pred_fries >= 0.5 and pred_hamburger < 0.5:
                pred_label = 'fries'
            elif pred_hamburger >= 0.5 and pred_fries < 0.5:
                pred_label = 'hamburger'
            elif pred_fries >= 0.5 and pred_hamburger >= 0.5:
                pred_label = 'both'
            else:
                pred_label = 'no_buy'
            pred_table.append({'sex': sex, 'age': age, 'predicted': pred_label, 'fries_prob': pred_fries, 'hamburger_prob': pred_hamburger})
    pred_df_out = pd.DataFrame(pred_table)
    print(pred_df_out[['sex', 'age', 'predicted', 'fries_prob', 'hamburger_prob']])
    pred_df_out.to_csv('age_sex_predicted.csv', index=False)
    print('Predictions by age and sex saved to age_sex_predicted.csv')
else:
    print('No valid weights found for prediction by age and sex.')

# Load data for predictions/analysis
def load_data(filename, target_col):
    if filename.endswith('.csv'):
        df = pd.read_csv(filename)
    elif filename.endswith('.txt'):
        try:
            df = pd.read_csv(filename, sep=None, engine='python')
        except Exception:
            df = pd.read_csv(filename, sep='\t')
    else:
        raise ValueError('Unsupported file format. Use .csv or .txt')
    X = df[['sex', 'age']].values.astype(float)
    Y = df[['will_buy_fries']].values.astype(float)  # or whichever target you want to analyze
    all_targets = df[['will_buy_fries', 'will_buy_hamburger']].values.astype(int)
    return X, Y, all_targets

X, Y, all_targets = load_data('data.csv', 'will_buy_fries') 