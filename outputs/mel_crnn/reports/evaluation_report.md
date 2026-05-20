# Evaluation Report

- Run: `mel_crnn`
- Timestamp: 2026-05-20T01:55:30
- Device: `mps`

## Summary

- Accuracy: **0.6670**
- Macro F1: **0.6626**
- Weighted F1: **0.6626**
- ROC-AUC (OvR, macro): **0.9412**

## Classification Report

```
              precision    recall  f1-score   support

       blues     0.6485    0.5350    0.5863       200
   classical     0.8348    0.9350    0.8821       200
     country     0.5455    0.6900    0.6093       200
       disco     0.5766    0.3950    0.4688       200
      hiphop     0.7188    0.5750    0.6389       200
        jazz     0.7306    0.8950    0.8045       200
       metal     0.8883    0.7950    0.8391       200
         pop     0.5599    0.7950    0.6570       200
      reggae     0.8099    0.5750    0.6725       200
        rock     0.4550    0.4800    0.4672       200

    accuracy                         0.6670      2000
   macro avg     0.6768    0.6670    0.6626      2000
weighted avg     0.6768    0.6670    0.6626      2000

```
