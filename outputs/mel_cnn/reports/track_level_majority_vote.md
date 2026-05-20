# Track-Level Evaluation Report

**Aggregation mode:** `majority_vote`  
**Total tracks:** 200

## Summary

- Track-level Accuracy: **0.6900**
- ROC-AUC (OvR, macro): **0.9448**

## Classification Report

```
              precision    recall  f1-score   support

       blues     0.7500    0.7500    0.7500        20
   classical     0.6667    1.0000    0.8000        20
     country     0.5625    0.4500    0.5000        20
       disco     0.5769    0.7500    0.6522        20
      hiphop     0.6786    0.9500    0.7917        20
        jazz     1.0000    0.6000    0.7500        20
       metal     0.7600    0.9500    0.8444        20
         pop     1.0000    0.3500    0.5185        20
      reggae     0.8000    0.6000    0.6857        20
        rock     0.4762    0.5000    0.4878        20

    accuracy                         0.6900       200
   macro avg     0.7271    0.6900    0.6780       200
weighted avg     0.7271    0.6900    0.6780       200

```

## Confusion Matrix

Rows = true, Cols = predicted (classes: blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock)

```
    15    0    2    2    0    0    0    0    1    0
     0   20    0    0    0    0    0    0    0    0
     2    3    9    1    0    0    0    0    0    5
     0    1    0   15    3    0    1    0    0    0
     0    0    0    1   19    0    0    0    0    0
     1    5    0    0    0   12    0    0    0    2
     0    0    0    0    0    0   19    0    0    1
     0    0    1    4    4    0    2    7    1    1
     1    0    1    2    2    0    0    0   12    2
     1    1    3    1    0    0    3    0    1   10
```
