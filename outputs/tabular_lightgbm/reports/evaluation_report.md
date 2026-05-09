# Evaluation Report

- Run: `tabular_lightgbm`
- Timestamp: 2026-05-09T22:54:59
- Device: `cuda`

## Summary

- Accuracy: **0.9129**
- Macro F1: **0.9127**
- Weighted F1: **0.9127**
- ROC-AUC (OvR, macro): **0.9955**

## Classification Report

```
              precision    recall  f1-score   support

       blues     0.8981    0.9250    0.9113       200
   classical     0.9466    0.9799    0.9630       199
     country     0.8700    0.8744    0.8722       199
       disco     0.8932    0.9200    0.9064       200
      hiphop     0.9275    0.8950    0.9109       200
        jazz     0.9064    0.9200    0.9132       200
       metal     0.9554    0.9650    0.9602       200
         pop     0.9399    0.8600    0.8982       200
      reggae     0.8932    0.9200    0.9064       200
        rock     0.9016    0.8700    0.8855       200

    accuracy                         0.9129      1998
   macro avg     0.9132    0.9129    0.9127      1998
weighted avg     0.9132    0.9129    0.9127      1998

```
