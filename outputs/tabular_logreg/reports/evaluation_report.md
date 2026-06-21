# Evaluation Report

- Run: `tabular_logreg`
- Timestamp: 2026-06-20T23:51:18
- Device: `mps`

## Summary

- Accuracy: **0.6580**
- Macro F1: **0.6543**
- Weighted F1: **0.6543**
- ROC-AUC (OvR, macro): **0.9329**

## Classification Report

```
              precision    recall  f1-score   support

       blues     0.6056    0.6450    0.6247       200
   classical     0.8685    0.9250    0.8959       200
     country     0.6154    0.6000    0.6076       200
       disco     0.6200    0.4650    0.5314       200
      hiphop     0.5860    0.5450    0.5648       200
        jazz     0.7069    0.8200    0.7593       200
       metal     0.8138    0.7650    0.7887       200
         pop     0.6810    0.7150    0.6976       200
      reggae     0.5833    0.6650    0.6215       200
        rock     0.4703    0.4350    0.4519       200

    accuracy                         0.6580      2000
   macro avg     0.6551    0.6580    0.6543      2000
weighted avg     0.6551    0.6580    0.6543      2000

```
