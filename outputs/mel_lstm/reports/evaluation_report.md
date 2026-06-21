# Evaluation Report

- Run: `mel_lstm`
- Timestamp: 2026-06-21T00:57:04
- Device: `mps`

## Summary

- Accuracy: **0.2755**
- Macro F1: **0.2389**
- Weighted F1: **0.2389**
- ROC-AUC (OvR, macro): **0.7810**

## Classification Report

```
              precision    recall  f1-score   support

       blues     0.1858    0.3650    0.2462       200
   classical     0.4304    0.8500    0.5714       200
     country     0.2185    0.1650    0.1880       200
       disco     0.2323    0.1150    0.1538       200
      hiphop     0.3750    0.0150    0.0288       200
        jazz     0.2349    0.5650    0.3319       200
       metal     0.8939    0.2950    0.4436       200
         pop     0.2406    0.2550    0.2476       200
      reggae     0.3793    0.0550    0.0961       200
        rock     0.0904    0.0750    0.0820       200

    accuracy                         0.2755      2000
   macro avg     0.3281    0.2755    0.2389      2000
weighted avg     0.3281    0.2755    0.2389      2000

```
