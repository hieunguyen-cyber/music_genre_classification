# Evaluation Report

- Run: `tabular_svm_rbf`
- Timestamp: 2026-05-09T22:37:22
- Device: `cuda`

## Summary

- Accuracy: **0.9069**
- Macro F1: **0.9068**
- Weighted F1: **0.9068**
- ROC-AUC (OvR, macro): **0.9946**

## Classification Report

```
              precision    recall  f1-score   support

       blues     0.9059    0.9150    0.9104       200
   classical     0.9057    0.9648    0.9343       199
     country     0.8643    0.8643    0.8643       199
       disco     0.9010    0.8650    0.8827       200
      hiphop     0.9091    0.9000    0.9045       200
        jazz     0.8952    0.9400    0.9171       200
       metal     0.9742    0.9450    0.9594       200
         pop     0.9348    0.8600    0.8958       200
      reggae     0.8883    0.9150    0.9015       200
        rock     0.8955    0.9000    0.8978       200

    accuracy                         0.9069      1998
   macro avg     0.9074    0.9069    0.9068      1998
weighted avg     0.9074    0.9069    0.9068      1998

```
