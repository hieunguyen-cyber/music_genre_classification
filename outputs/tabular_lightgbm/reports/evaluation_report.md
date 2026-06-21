# Evaluation Report

- Run: `tabular_lightgbm`
- Timestamp: 2026-06-21T00:00:33
- Device: `mps`

## Summary

- Accuracy: **0.7540**
- Macro F1: **0.7513**
- Weighted F1: **0.7513**
- ROC-AUC (OvR, macro): **0.9649**

## Classification Report

```
              precision    recall  f1-score   support

       blues     0.7500    0.7650    0.7574       200
   classical     0.8858    0.9700    0.9260       200
     country     0.7304    0.7450    0.7376       200
       disco     0.7059    0.5400    0.6119       200
      hiphop     0.6765    0.8050    0.7352       200
        jazz     0.7757    0.8300    0.8019       200
       metal     0.8983    0.7950    0.8435       200
         pop     0.6983    0.8100    0.7500       200
      reggae     0.7455    0.6150    0.6740       200
        rock     0.6856    0.6650    0.6751       200

    accuracy                         0.7540      2000
   macro avg     0.7552    0.7540    0.7513      2000
weighted avg     0.7552    0.7540    0.7513      2000

```
