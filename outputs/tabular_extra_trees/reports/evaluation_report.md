# Evaluation Report

- Run: `tabular_extra_trees`
- Timestamp: 2026-06-20T23:52:03
- Device: `mps`

## Summary

- Accuracy: **0.7305**
- Macro F1: **0.7262**
- Weighted F1: **0.7262**
- ROC-AUC (OvR, macro): **0.9570**

## Classification Report

```
              precision    recall  f1-score   support

       blues     0.7435    0.7100    0.7263       200
   classical     0.8904    0.9750    0.9308       200
     country     0.7048    0.7400    0.7220       200
       disco     0.6839    0.5300    0.5972       200
      hiphop     0.6535    0.7450    0.6963       200
        jazz     0.7671    0.8400    0.8019       200
       metal     0.8350    0.8350    0.8350       200
         pop     0.7009    0.7850    0.7406       200
      reggae     0.6313    0.6250    0.6281       200
        rock     0.6667    0.5200    0.5843       200

    accuracy                         0.7305      2000
   macro avg     0.7277    0.7305    0.7262      2000
weighted avg     0.7277    0.7305    0.7262      2000

```
