# Evaluation Report

- Run: `tabular_qda`
- Timestamp: 2026-05-09T22:38:45
- Device: `cuda`

## Summary

- Accuracy: **0.7487**
- Macro F1: **0.7486**
- Weighted F1: **0.7486**
- ROC-AUC (OvR, macro): **0.9633**

## Classification Report

```
              precision    recall  f1-score   support

       blues     0.8253    0.6850    0.7486       200
   classical     0.8905    0.9397    0.9144       199
     country     0.6311    0.7739    0.6953       199
       disco     0.6939    0.6800    0.6869       200
      hiphop     0.8256    0.7100    0.7634       200
        jazz     0.8556    0.7700    0.8105       200
       metal     0.6766    0.9100    0.7761       200
         pop     0.8297    0.7550    0.7906       200
      reggae     0.7143    0.6250    0.6667       200
        rock     0.6275    0.6400    0.6337       200

    accuracy                         0.7487      1998
   macro avg     0.7570    0.7489    0.7486      1998
weighted avg     0.7570    0.7487    0.7486      1998

```
