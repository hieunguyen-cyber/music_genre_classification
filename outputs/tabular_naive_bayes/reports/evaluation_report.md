# Evaluation Report

- Run: `tabular_naive_bayes`
- Timestamp: 2026-06-20T23:51:26
- Device: `mps`

## Summary

- Accuracy: **0.5065**
- Macro F1: **0.4846**
- Weighted F1: **0.4846**
- ROC-AUC (OvR, macro): **0.8694**

## Classification Report

```
              precision    recall  f1-score   support

       blues     0.4766    0.2550    0.3322       200
   classical     0.7231    0.9400    0.8174       200
     country     0.4218    0.5800    0.4884       200
       disco     0.3981    0.2050    0.2706       200
      hiphop     0.6496    0.3800    0.4795       200
        jazz     0.5899    0.4100    0.4838       200
       metal     0.4396    0.8000    0.5674       200
         pop     0.5763    0.7550    0.6537       200
      reggae     0.4623    0.4900    0.4757       200
        rock     0.3106    0.2500    0.2770       200

    accuracy                         0.5065      2000
   macro avg     0.5048    0.5065    0.4846      2000
weighted avg     0.5048    0.5065    0.4846      2000

```
