# Evaluation Report

- Run: `mel_cnn`
- Timestamp: 2026-05-09T23:24:12
- Device: `cuda`

## Summary

- Accuracy: **0.6310**
- Macro F1: **0.6211**
- Weighted F1: **0.6211**
- ROC-AUC (OvR, macro): **0.9211**

## Classification Report

```
              precision    recall  f1-score   support

       blues     0.6571    0.5750    0.6133       200
   classical     0.5888    0.9950    0.7398       200
     country     0.5210    0.4350    0.4741       200
       disco     0.5442    0.5850    0.5639       200
      hiphop     0.6434    0.8750    0.7415       200
        jazz     0.9313    0.6100    0.7372       200
       metal     0.6967    0.8500    0.7658       200
         pop     0.7037    0.3800    0.4935       200
      reggae     0.7325    0.5750    0.6443       200
        rock     0.4456    0.4300    0.4377       200

    accuracy                         0.6310      2000
   macro avg     0.6464    0.6310    0.6211      2000
weighted avg     0.6464    0.6310    0.6211      2000

```
