# Evaluation Report

- Run: `tabular_gbdt`
- Timestamp: 2026-05-19T23:00:55
- Device: `mps`

## Summary

- Accuracy: **0.7440**
- Macro F1: **0.7428**
- Weighted F1: **0.7428**
- ROC-AUC (OvR, macro): **0.9589**

## Classification Report

```
              precision    recall  f1-score   support

       blues     0.7634    0.7100    0.7358       200
   classical     0.9183    0.9550    0.9363       200
     country     0.6567    0.7650    0.7067       200
       disco     0.7030    0.5800    0.6356       200
      hiphop     0.6786    0.7600    0.7170       200
        jazz     0.7767    0.8350    0.8048       200
       metal     0.8750    0.8050    0.8385       200
         pop     0.7402    0.7550    0.7475       200
      reggae     0.6882    0.6400    0.6632       200
        rock     0.6513    0.6350    0.6430       200

    accuracy                         0.7440      2000
   macro avg     0.7451    0.7440    0.7428      2000
weighted avg     0.7451    0.7440    0.7428      2000

```
