'''
Autore: Alessio Ferrari

In this work I code a binary classification to recognize if a tumor was malinant or benign.
'''

#################### Progetto ML-Classification ####################

#Libraries
from ucimlrepo import fetch_ucirepo
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_curve

#################### Data Import & Split ####################
# fetch dataset 
breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17) 
# data (as pandas dataframes) 
x = breast_cancer_wisconsin_diagnostic.data.features 
y= breast_cancer_wisconsin_diagnostic.data.targets 
# metadata 
print(breast_cancer_wisconsin_diagnostic.metadata)  
# variable information 
print(breast_cancer_wisconsin_diagnostic.variables) 

#Mapping target variable
y=y.replace({'B': 0, 'M': 1})

#Split
x_tr, x_te, y_tr, y_te=train_test_split(x, y, test_size=0.2, random_state=0, shuffle=True)

#Copy DataSet
x_tr_copy = x_tr.copy()
x_tr_copy['diagnosis'] = y_tr

#################### Data Preprocessing & Visualization #################### 
#BoxPlot for each variable respect to the target 
for column in x_tr.columns:
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='diagnosis', y=column, data=x_tr_copy)
    plt.title(f'Boxplot di {column} rispetto alla diagnosi')
    plt.xlabel('Diagnosi (0=Benigno, 1=Maligno)')
    plt.ylabel(column)
    #plt.savefig(f'{column}_B.png')
    plt.show()

#Histogram and KDE for each variable respect to the target
for column in x_tr.columns:
    plt.figure(figsize=(8, 6))
    sns.histplot(data=x_tr_copy, x=column, hue='diagnosis', kde=True, bins=30, palette='Set1')
    plt.title(f'Istogramma di {column} rispetto alla diagnosi')
    plt.xlabel(column)
    plt.ylabel('Frequenza')
    #plt.savefig(f'{column}_D.png')
    plt.show()

#Drop the feature with kernel distribution too simalar
deleteFeatures=['smoothness1', 'symmetry1', 'fractal_dimension1', 'texture2', 'smoothness2', 'compactness2', 'concavity2', 'concave_points2', 'symmetry2', 'fractal_dimension2', 'texture3', 'smoothness3', 'fractal_dimension3']
x_tr=x_tr.drop(columns=deleteFeatures)
x_tr_copy=x_tr_copy.drop(columns=deleteFeatures)
x_te=x_te.drop(columns=deleteFeatures)

#Correlation
corr_matrix=x_tr.corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, fmt='.2f',  cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Matrice di Correlazione')
#plt.savefig('CorrMatrix.png')
plt.show()

#Drop high correlated variables 
deleteFeatures1=['radius1', 'area1', 'perimeter1', 'compactness1', 'concavity1', 'radius2', 'perimeter2', 'radius3', 'perimeter3', 'compactness3', 'concavity3', 'concave_points3',]
x_tr=x_tr.drop(columns=deleteFeatures1)
x_tr_copy=x_tr_copy.drop(columns=deleteFeatures1)
x_te=x_te.drop(columns=deleteFeatures1)

corr_matrix1=x_tr.corr()

#Standardize
scaler=StandardScaler()
x_tr_scaled=scaler.fit_transform(x_tr)
x_te_scaled=scaler.transform(x_te)

#################### Models #################### 
########## Decision Tree ##########
tree_model=DecisionTreeClassifier(max_depth=5, random_state=0)
tree_model.fit(x_tr_scaled, y_tr)

#Predictions
y_tree=tree_model.predict(x_te_scaled)
y_tree_prob=tree_model.predict_proba(x_te_scaled)[:, 1]

print('Decision Tree:')
print('\nIndici:')
print(classification_report(y_te, y_tree))
print('Confusion Matrix:')
print(confusion_matrix(y_te, y_tree))

fpr_tree, tpr_tree, thresholds_tree = roc_curve(y_te, y_tree_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr_tree, tpr_tree)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC Decision Tree')
#plt.savefig('ROC_Tree.png')
plt.show()

########## Random Forest ##########
model=RandomForestClassifier(random_state=17)
model.fit(x_tr_scaled, y_tr)

y_for=model.predict(x_te_scaled)
y_for_prob=model.predict_proba(x_te_scaled)[:, 1]

print('Random Forest')
print('\nIndici:')
print(classification_report(y_te, y_for))
print('Confusion Matrix:')
print(confusion_matrix(y_te, y_for))

fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_te, y_for_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr_forest, tpr_forest)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC Random Forest')
#plt.savefig('ROC_Forest.png')
plt.show()

########## Logistic Regression ##########
log_reg = LogisticRegression(random_state=24, max_iter=10)  # Aumenta max_iter per convergenza
log_reg.fit(x_tr_scaled, y_tr)

y_pred_reg = log_reg.predict(x_te_scaled)
y_pred_prob_reg = log_reg.predict_proba(x_te_scaled)[:, 1]

print('Regressione Logistica')
print('\nIndici:')
print(classification_report(y_te, y_pred_reg))
print('Confusion Matrix:')
print(confusion_matrix(y_te, y_pred_reg))

fpr_log, tpr_log, thresholds_log=roc_curve(y_te, y_pred_prob_reg)
plt.figure(figsize=(8, 6))
plt.plot(fpr_log, tpr_log)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC Regressione Logistica')
#plt.savefig('ROC_Logistic.png')
plt.show()


########## SVM ##########
svc_model=SVC(kernel='linear', probability=True, random_state=11)
svc_model.fit(x_tr_scaled, y_tr)

y_svc=svc_model.predict(x_te_scaled)
y_svc_prob=svc_model.predict_proba(x_te_scaled)[:, 1]

print('SVC')
print('\nIndici:')
print(classification_report(y_te, y_svc))
print('Confusion Matrix:')
print(confusion_matrix(y_te, y_svc))

fpr_svc, tpr_svc, thresholds_svc=roc_curve(y_te, y_svc_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr_svc, tpr_svc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC SVC')
#plt.savefig('ROC_SVC.png')
plt.show()





