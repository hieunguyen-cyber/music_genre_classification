# Evaluation Report

- Run: `tabular_svm_rbf`
- Timestamp: 2026-05-19T22:52:16
- Device: `mps`

## Summary

- Accuracy: **0.7155**
- Macro F1: **0.7129**
- Weighted F1: **0.7129**
- ROC-AUC (OvR, macro): **0.9530**

## Classification Report

```
              precision    recall  f1-score   support

       blues     0.6800    0.7650    0.7200       200
   classical     0.8000    0.9400    0.8644       200
     country     0.7010    0.7150    0.7079       200
       disco     0.7226    0.5600    0.6310       200
      hiphop     0.6667    0.7300    0.6969       200
        jazz     0.7364    0.8100    0.7714       200
       metal     0.9075    0.7850    0.8418       200
         pop     0.6244    0.6650    0.6441       200
      reggae     0.6736    0.6500    0.6616       200
        rock     0.6564    0.5350    0.5895       200

    accuracy                         0.7155      2000
   macro avg     0.7169    0.7155    0.7129      2000
weighted avg     0.7169    0.7155    0.7129      2000

```
