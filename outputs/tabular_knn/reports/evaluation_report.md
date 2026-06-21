# Evaluation Report

- Run: `tabular_knn`
- Timestamp: 2026-06-20T23:50:39
- Device: `mps`

## Summary

- Accuracy: **0.6435**
- Macro F1: **0.6434**
- Weighted F1: **0.6434**
- ROC-AUC (OvR, macro): **0.8681**

## Classification Report

```
              precision    recall  f1-score   support

       blues     0.6384    0.7150    0.6745       200
   classical     0.7597    0.8850    0.8176       200
     country     0.5198    0.5900    0.5527       200
       disco     0.4894    0.5750    0.5287       200
      hiphop     0.6532    0.5650    0.6059       200
        jazz     0.7596    0.6950    0.7258       200
       metal     0.7841    0.6900    0.7340       200
         pop     0.7104    0.6500    0.6789       200
      reggae     0.6062    0.5850    0.5954       200
        rock     0.5607    0.4850    0.5201       200

    accuracy                         0.6435      2000
   macro avg     0.6481    0.6435    0.6434      2000
weighted avg     0.6481    0.6435    0.6434      2000

```
