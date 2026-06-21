# Evaluation Report

- Run: `tabular_random_forest`
- Timestamp: 2026-06-20T23:51:54
- Device: `mps`

## Summary

- Accuracy: **0.7155**
- Macro F1: **0.7117**
- Weighted F1: **0.7117**
- ROC-AUC (OvR, macro): **0.9562**

## Classification Report

```
              precision    recall  f1-score   support

       blues     0.7186    0.7150    0.7168       200
   classical     0.9052    0.9550    0.9294       200
     country     0.6791    0.7300    0.7036       200
       disco     0.7039    0.5350    0.6080       200
      hiphop     0.6359    0.6900    0.6619       200
        jazz     0.7179    0.8400    0.7742       200
       metal     0.8109    0.8150    0.8130       200
         pop     0.6934    0.7350    0.7136       200
      reggae     0.6293    0.6450    0.6370       200
        rock     0.6429    0.4950    0.5593       200

    accuracy                         0.7155      2000
   macro avg     0.7137    0.7155    0.7117      2000
weighted avg     0.7137    0.7155    0.7117      2000

```
