# Evaluation Report

- Run: `tabular_lda`
- Timestamp: 2026-05-09T22:38:29
- Device: `cuda`

## Summary

- Accuracy: **0.6627**
- Macro F1: **0.6611**
- Weighted F1: **0.6610**
- ROC-AUC (OvR, macro): **0.9403**

## Classification Report

```
              precision    recall  f1-score   support

       blues     0.6538    0.5100    0.5730       200
   classical     0.8798    0.9196    0.8993       199
     country     0.5780    0.6332    0.6043       199
       disco     0.5352    0.5700    0.5521       200
      hiphop     0.7324    0.5200    0.6082       200
        jazz     0.7311    0.7750    0.7524       200
       metal     0.7619    0.8800    0.8167       200
         pop     0.7845    0.7100    0.7454       200
      reggae     0.5928    0.6550    0.6223       200
        rock     0.4213    0.4550    0.4375       200

    accuracy                         0.6627      1998
   macro avg     0.6671    0.6628    0.6611      1998
weighted avg     0.6670    0.6627    0.6610      1998

```
