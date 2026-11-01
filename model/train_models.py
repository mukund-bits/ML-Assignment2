# training all 5 models on breast cancer data
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef)

# load data
bc = pd.read_csv("data.csv")
bc = bc.drop(columns=["id"])
bc["diagnosis"] = bc["diagnosis"].map({"M": 1, "B": 0})

xx = bc.drop(columns=["diagnosis"])
yy = bc["diagnosis"]

xtr, xte, ytr, yte = train_test_split(xx, yy, test_size=0.2, random_state=7, stratify=yy)

# save test data for the app
tst = xte.copy()
tst["diagnosis"] = yte
tst.to_csv("test_data.csv", index=False)

# scaling
sc = StandardScaler()
xtr_s = sc.fit_transform(xtr)
xte_s = sc.transform(xte)
joblib.dump(sc, "model/scaler.pkl")

mdls = {
    "Logistic Regression": (LogisticRegression(max_iter=1000), "model/logreg.pkl"),
    "Decision Tree": (DecisionTreeClassifier(random_state=7), "model/dtree.pkl"),
    "KNN": (KNeighborsClassifier(n_neighbors=5), "model/knn.pkl"),
    "Naive Bayes": (GaussianNB(), "model/nbayes.pkl"),
    "SVM": (SVC(probability=True, random_state=7), "model/svm.pkl"),
}

rows = []
for nm, (m, pth) in mdls.items():
    m.fit(xtr_s, ytr)
    pr = m.predict(xte_s)
    pb = m.predict_proba(xte_s)[:, 1]
    rows.append([nm,
                 accuracy_score(yte, pr),
                 roc_auc_score(yte, pb),
                 precision_score(yte, pr),
                 recall_score(yte, pr),
                 f1_score(yte, pr),
                 matthews_corrcoef(yte, pr)])
    joblib.dump(m, pth)

res = pd.DataFrame(rows, columns=["Model", "Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"])
print(res.round(4).to_string(index=False))
