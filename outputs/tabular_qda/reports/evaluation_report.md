# Evaluation Report

- Run: `tabular_qda`
- Timestamp: 2026-05-19T22:53:03
- Device: `mps`

## Summary

- Accuracy: **0.6560**
- Macro F1: **0.6519**
- Weighted F1: **0.6519**
- ROC-AUC (OvR, macro): **0.9252**

## Classification Report

```
              precision    recall  f1-score   support

       blues     0.6915    0.6500    0.6701       200
   classical     0.8430    0.9400    0.8889       200
     country     0.6071    0.7650    0.6770       200
       disco     0.6304    0.4350    0.5148       200
      hiphop     0.6095    0.6400    0.6244       200
        jazz     0.7654    0.6850    0.7230       200
       metal     0.6225    0.6350    0.6287       200
         pop     0.6419    0.7350    0.6853       200
      reggae     0.5989    0.5600    0.5788       200
        rock     0.5421    0.5150    0.5282       200

    accuracy                         0.6560      2000
   macro avg     0.6553    0.6560    0.6519      2000
weighted avg     0.6553    0.6560    0.6519      2000

```
