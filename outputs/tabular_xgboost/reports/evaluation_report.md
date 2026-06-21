# Evaluation Report

- Run: `tabular_xgboost`
- Timestamp: 2026-06-20T23:59:54
- Device: `mps`

## Summary

- Accuracy: **0.7535**
- Macro F1: **0.7518**
- Weighted F1: **0.7518**
- ROC-AUC (OvR, macro): **0.9658**

## Classification Report

```
              precision    recall  f1-score   support

       blues     0.7662    0.7700    0.7681       200
   classical     0.8884    0.9550    0.9205       200
     country     0.7136    0.7850    0.7476       200
       disco     0.6928    0.5750    0.6284       200
      hiphop     0.6826    0.7850    0.7302       200
        jazz     0.7837    0.8150    0.7990       200
       metal     0.8989    0.8000    0.8466       200
         pop     0.7358    0.7800    0.7573       200
      reggae     0.7257    0.6350    0.6773       200
        rock     0.6513    0.6350    0.6430       200

    accuracy                         0.7535      2000
   macro avg     0.7539    0.7535    0.7518      2000
weighted avg     0.7539    0.7535    0.7518      2000

```
