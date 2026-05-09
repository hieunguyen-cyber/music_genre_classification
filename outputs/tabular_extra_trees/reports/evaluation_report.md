# Evaluation Report

- Run: `tabular_extra_trees`
- Timestamp: 2026-05-09T22:39:24
- Device: `cuda`

## Summary

- Accuracy: **0.8904**
- Macro F1: **0.8900**
- Weighted F1: **0.8900**
- ROC-AUC (OvR, macro): **0.9902**

## Classification Report

```
              precision    recall  f1-score   support

       blues     0.8683    0.8900    0.8790       200
   classical     0.9198    0.9799    0.9489       199
     country     0.8333    0.8291    0.8312       199
       disco     0.8647    0.8950    0.8796       200
      hiphop     0.9368    0.8900    0.9128       200
        jazz     0.8873    0.9050    0.8960       200
       metal     0.8977    0.9650    0.9301       200
         pop     0.9448    0.8550    0.8976       200
      reggae     0.8295    0.9000    0.8633       200
        rock     0.9408    0.7950    0.8618       200

    accuracy                         0.8904      1998
   macro avg     0.8923    0.8904    0.8900      1998
weighted avg     0.8923    0.8904    0.8900      1998

```
