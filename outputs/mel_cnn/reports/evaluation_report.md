# Evaluation Report

- Run: `mel_cnn`
- Timestamp: 2026-05-08T22:37:36
- Device: `mps`

## Summary

- Accuracy: **0.6320**
- Macro F1: **0.6217**
- Weighted F1: **0.6217**
- ROC-AUC (OvR, macro): **0.9215**

## Classification Report

```
              precision    recall  f1-score   support

       blues     0.6290    0.5850    0.6062       200
   classical     0.6086    0.9950    0.7552       200
     country     0.5200    0.4550    0.4853       200
       disco     0.5268    0.5900    0.5566       200
      hiphop     0.6593    0.8900    0.7574       200
        jazz     0.9360    0.5850    0.7200       200
       metal     0.7113    0.8500    0.7745       200
         pop     0.7009    0.3750    0.4886       200
      reggae     0.7188    0.5750    0.6389       200
        rock     0.4492    0.4200    0.4341       200

    accuracy                         0.6320      2000
   macro avg     0.6460    0.6320    0.6217      2000
weighted avg     0.6460    0.6320    0.6217      2000

```
