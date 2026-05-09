# Evaluation Report

- Run: `tabular_naive_bayes`
- Timestamp: 2026-05-09T22:38:13
- Device: `cuda`

## Summary

- Accuracy: **0.4990**
- Macro F1: **0.4795**
- Weighted F1: **0.4794**
- ROC-AUC (OvR, macro): **0.8807**

## Classification Report

```
              precision    recall  f1-score   support

       blues     0.4112    0.2200    0.2866       200
   classical     0.7236    0.8945    0.8000       199
     country     0.3918    0.6281    0.4826       199
       disco     0.4178    0.3050    0.3526       200
      hiphop     0.7500    0.3300    0.4583       200
        jazz     0.5929    0.3350    0.4281       200
       metal     0.4349    0.8850    0.5832       200
         pop     0.6168    0.6600    0.6377       200
      reggae     0.4798    0.4750    0.4774       200
        rock     0.3250    0.2600    0.2889       200

    accuracy                         0.4990      1998
   macro avg     0.5144    0.4993    0.4795      1998
weighted avg     0.5143    0.4990    0.4794      1998

```
