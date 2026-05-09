# Evaluation Report

- Run: `tabular_xgboost`
- Timestamp: 2026-05-09T22:54:29
- Device: `cuda`

## Summary

- Accuracy: **0.9044**
- Macro F1: **0.9042**
- Weighted F1: **0.9042**
- ROC-AUC (OvR, macro): **0.9946**

## Classification Report

```
              precision    recall  f1-score   support

       blues     0.9021    0.8750    0.8883       200
   classical     0.9286    0.9799    0.9535       199
     country     0.8663    0.8794    0.8728       199
       disco     0.8673    0.9150    0.8905       200
      hiphop     0.9133    0.8950    0.9040       200
        jazz     0.8971    0.9150    0.9059       200
       metal     0.9552    0.9600    0.9576       200
         pop     0.9389    0.8450    0.8895       200
      reggae     0.8835    0.9100    0.8966       200
        rock     0.8969    0.8700    0.8832       200

    accuracy                         0.9044      1998
   macro avg     0.9049    0.9044    0.9042      1998
weighted avg     0.9049    0.9044    0.9042      1998

```
