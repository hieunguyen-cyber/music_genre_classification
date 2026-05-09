# Evaluation Report

- Run: `tabular_random_forest`
- Timestamp: 2026-05-09T22:39:04
- Device: `cuda`

## Summary

- Accuracy: **0.8694**
- Macro F1: **0.8687**
- Weighted F1: **0.8686**
- ROC-AUC (OvR, macro): **0.9887**

## Classification Report

```
              precision    recall  f1-score   support

       blues     0.8670    0.8800    0.8734       200
   classical     0.9057    0.9648    0.9343       199
     country     0.8168    0.8291    0.8229       199
       disco     0.8358    0.8400    0.8379       200
      hiphop     0.9211    0.8750    0.8974       200
        jazz     0.8458    0.9050    0.8744       200
       metal     0.8761    0.9550    0.9139       200
         pop     0.9368    0.8150    0.8717       200
      reggae     0.8128    0.8900    0.8496       200
        rock     0.8970    0.7400    0.8110       200

    accuracy                         0.8694      1998
   macro avg     0.8715    0.8694    0.8687      1998
weighted avg     0.8715    0.8694    0.8686      1998

```
