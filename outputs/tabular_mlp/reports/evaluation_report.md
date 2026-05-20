# Evaluation Report

- Run: `tabular_mlp`
- Timestamp: 2026-05-19T23:35:54
- Device: `mps`

## Summary

- Accuracy: **0.6785**
- Macro F1: **0.6719**
- Weighted F1: **0.6719**
- ROC-AUC (OvR, macro): **0.9368**

## Classification Report

```
              precision    recall  f1-score   support

       blues     0.6195    0.6350    0.6272       200
   classical     0.8268    0.9550    0.8863       200
     country     0.6761    0.5950    0.6330       200
       disco     0.6067    0.4550    0.5200       200
      hiphop     0.6749    0.6850    0.6799       200
        jazz     0.7330    0.8100    0.7696       200
       metal     0.8376    0.8250    0.8312       200
         pop     0.6190    0.7800    0.6903       200
      reggae     0.5753    0.6300    0.6014       200
        rock     0.5685    0.4150    0.4798       200

    accuracy                         0.6785      2000
   macro avg     0.6738    0.6785    0.6719      2000
weighted avg     0.6738    0.6785    0.6719      2000

```
