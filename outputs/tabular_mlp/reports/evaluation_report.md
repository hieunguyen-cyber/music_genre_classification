# Evaluation Report

- Run: `tabular_mlp`
- Timestamp: 2026-05-09T22:36:33
- Device: `cuda`

## Summary

- Accuracy: **0.8078**
- Macro F1: **0.8064**
- Weighted F1: **0.8063**
- ROC-AUC (OvR, macro): **0.9777**

## Classification Report

```
              precision    recall  f1-score   support

       blues     0.7814    0.8400    0.8096       200
   classical     0.8767    0.9648    0.9187       199
     country     0.7685    0.7839    0.7761       199
       disco     0.7238    0.7600    0.7415       200
      hiphop     0.8298    0.7800    0.8041       200
        jazz     0.8458    0.8500    0.8479       200
       metal     0.8732    0.8950    0.8840       200
         pop     0.8466    0.8000    0.8226       200
      reggae     0.7970    0.7850    0.7909       200
        rock     0.7251    0.6200    0.6685       200

    accuracy                         0.8078      1998
   macro avg     0.8068    0.8079    0.8064      1998
weighted avg     0.8068    0.8078    0.8063      1998

```
