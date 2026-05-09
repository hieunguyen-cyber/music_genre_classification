# Evaluation Report

- Run: `tabular_logreg`
- Timestamp: 2026-05-09T22:37:57
- Device: `cuda`

## Summary

- Accuracy: **0.7152**
- Macro F1: **0.7129**
- Weighted F1: **0.7129**
- ROC-AUC (OvR, macro): **0.9565**

## Classification Report

```
              precision    recall  f1-score   support

       blues     0.6618    0.6850    0.6732       200
   classical     0.9009    0.9598    0.9294       199
     country     0.6105    0.5829    0.5964       199
       disco     0.6436    0.6500    0.6468       200
      hiphop     0.7662    0.5900    0.6667       200
        jazz     0.7814    0.8400    0.8096       200
       metal     0.7963    0.8600    0.8269       200
         pop     0.7959    0.7800    0.7879       200
      reggae     0.6604    0.7000    0.6796       200
        rock     0.5206    0.5050    0.5127       200

    accuracy                         0.7152      1998
   macro avg     0.7138    0.7153    0.7129      1998
weighted avg     0.7137    0.7152    0.7129      1998

```
