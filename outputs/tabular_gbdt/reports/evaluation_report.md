# Evaluation Report

- Run: `tabular_gbdt`
- Timestamp: 2026-05-09T22:53:56
- Device: `cuda`

## Summary

- Accuracy: **0.8544**
- Macro F1: **0.8549**
- Weighted F1: **0.8549**
- ROC-AUC (OvR, macro): **0.9886**

## Classification Report

```
              precision    recall  f1-score   support

       blues     0.8450    0.8450    0.8450       200
   classical     0.9391    0.9296    0.9343       199
     country     0.7664    0.8241    0.7942       199
       disco     0.8250    0.8250    0.8250       200
      hiphop     0.9259    0.8750    0.8997       200
        jazz     0.8333    0.8750    0.8537       200
       metal     0.9343    0.9250    0.9296       200
         pop     0.9050    0.8100    0.8549       200
      reggae     0.8230    0.8600    0.8411       200
        rock     0.7673    0.7750    0.7711       200

    accuracy                         0.8544      1998
   macro avg     0.8564    0.8544    0.8549      1998
weighted avg     0.8564    0.8544    0.8549      1998

```
