import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from toolbox.csv_loader import load_data

data = load_data("../toolbox/test5.csv")
xs, ys, bits = data["x"], data["y"], data["bits"]
x_all = np.column_stack((xs, ys)).astype(np.float64)
y_all = np.asarray(bits, dtype=np.int64)
idx_all = np.arange(len(x_all))

idx_train, idx_test, y_train, y_test = train_test_split(
    idx_all, y_all, test_size=0.1, random_state=42, stratify=y_all
)
x_train = x_all[idx_train]
X_test = x_all[idx_test]

clf = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42))
])
clf.fit(x_train, y_train)
y_pred = clf.predict(X_test)
print("SVM Accuracy: {:.2f}%".format(accuracy_score(y_test, y_pred) * 100))

# 蒸馏
support_pos_in_train = clf.named_steps["svm"].support_
support_idx_global = idx_train[support_pos_in_train]

distilled_X = x_all[support_idx_global]
distilled_y = y_all[support_idx_global]

student_clf = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42))
])
student_clf.fit(distilled_X, distilled_y)

y_pred_student = student_clf.predict(X_test)
print("Distilled dataset SVM classification accuracy: {:.2f}%"
      .format(accuracy_score(y_test, y_pred_student) * 100))
print("Distilled dataset size:", len(distilled_X))

distilled_df = pd.DataFrame({
    "x": distilled_X[:, 0],
    "y": distilled_X[:, 1],
    "bits": distilled_y
})

distilled_df = distilled_df.sample(frac=1, random_state=42).reset_index(drop=True)
distilled_df.to_csv("distilled_dataset.csv", index=False, header=False)
