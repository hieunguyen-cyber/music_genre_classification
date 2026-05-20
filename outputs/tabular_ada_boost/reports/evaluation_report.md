# Evaluation Report

- Run: `tabular_ada_boost`
- Timestamp: 2026-05-19T22:53:50
- Device: `mps`

## Summary

- Accuracy: **0.4665**
- Macro F1: **0.4655**
- Weighted F1: **0.4655**
- ROC-AUC (OvR, macro): **0.8249**

## Classification Report

```
              precision    recall  f1-score   support

       blues     0.4367    0.5000    0.4662       200
   classical     0.9529    0.4050    0.5684       200
     country     0.3464    0.3100    0.3272       200
       disco     0.3345    0.4950    0.3992       200
      hiphop     0.5556    0.2750    0.3679       200
        jazz     0.3959    0.6750    0.4991       200
       metal     0.7226    0.5600    0.6310       200
         pop     0.5273    0.7250    0.6105       200
      reggae     0.5886    0.4650    0.5196       200
        rock     0.2787    0.2550    0.2663       200

    accuracy                         0.4665      2000
   macro avg     0.5139    0.4665    0.4655      2000
weighted avg     0.5139    0.4665    0.4655      2000

```
