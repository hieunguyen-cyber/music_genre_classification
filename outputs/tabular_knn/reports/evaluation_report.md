# Evaluation Report

- Run: `tabular_knn`
- Timestamp: 2026-05-09T22:36:59
- Device: `cuda`

## Summary

- Accuracy: **0.8744**
- Macro F1: **0.8740**
- Weighted F1: **0.8740**
- ROC-AUC (OvR, macro): **0.9720**

## Classification Report

```
              precision    recall  f1-score   support

       blues     0.8233    0.8850    0.8530       200
   classical     0.8841    0.9196    0.9015       199
     country     0.7940    0.7940    0.7940       199
       disco     0.8059    0.9550    0.8741       200
      hiphop     0.8867    0.9000    0.8933       200
        jazz     0.8667    0.8450    0.8557       200
       metal     0.9848    0.9700    0.9773       200
         pop     0.9509    0.7750    0.8540       200
      reggae     0.8673    0.9150    0.8905       200
        rock     0.9181    0.7850    0.8464       200

    accuracy                         0.8744      1998
   macro avg     0.8782    0.8744    0.8740      1998
weighted avg     0.8782    0.8744    0.8740      1998

```
