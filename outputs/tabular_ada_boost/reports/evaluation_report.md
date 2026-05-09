# Evaluation Report

- Run: `tabular_ada_boost`
- Timestamp: 2026-05-09T22:40:06
- Device: `cuda`

## Summary

- Accuracy: **0.4299**
- Macro F1: **0.4155**
- Weighted F1: **0.4155**
- ROC-AUC (OvR, macro): **0.8409**

## Classification Report

```
              precision    recall  f1-score   support

       blues     0.3655    0.2650    0.3072       200
   classical     0.7529    0.6432    0.6938       199
     country     0.3039    0.1558    0.2060       199
       disco     0.2462    0.3250    0.2802       200
      hiphop     0.4155    0.2950    0.3450       200
        jazz     0.3592    0.5100    0.4215       200
       metal     0.6528    0.7050    0.6779       200
         pop     0.5874    0.6550    0.6194       200
      reggae     0.3518    0.6350    0.4528       200
        rock     0.2418    0.1100    0.1512       200

    accuracy                         0.4299      1998
   macro avg     0.4277    0.4299    0.4155      1998
weighted avg     0.4276    0.4299    0.4155      1998

```
