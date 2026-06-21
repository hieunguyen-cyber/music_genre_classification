# Evaluation Report

- Run: `tabular_lda`
- Timestamp: 2026-06-20T23:51:35
- Device: `mps`

## Summary

- Accuracy: **0.6345**
- Macro F1: **0.6275**
- Weighted F1: **0.6275**
- ROC-AUC (OvR, macro): **0.9263**

## Classification Report

```
              precision    recall  f1-score   support

       blues     0.5912    0.4700    0.5237       200
   classical     0.8551    0.9150    0.8841       200
     country     0.6050    0.6050    0.6050       200
       disco     0.5652    0.3900    0.4615       200
      hiphop     0.7051    0.5500    0.6180       200
        jazz     0.6441    0.7600    0.6972       200
       metal     0.6513    0.7750    0.7078       200
         pop     0.7248    0.7900    0.7560       200
      reggae     0.5622    0.7000    0.6236       200
        rock     0.4062    0.3900    0.3980       200

    accuracy                         0.6345      2000
   macro avg     0.6310    0.6345    0.6275      2000
weighted avg     0.6310    0.6345    0.6275      2000

```
