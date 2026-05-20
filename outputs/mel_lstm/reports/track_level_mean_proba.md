# Track-Level Evaluation Report

**Aggregation mode:** `mean_proba`  
**Total tracks:** 200

## Summary

- Track-level Accuracy: **0.3200**
- ROC-AUC (OvR, macro): **0.8373**

## Classification Report

```
              precision    recall  f1-score   support

       blues     0.2500    0.4500    0.3214        20
   classical     0.5000    0.9500    0.6552        20
     country     0.4286    0.3000    0.3529        20
       disco     0.1667    0.1000    0.1250        20
      hiphop     0.0000    0.0000    0.0000        20
        jazz     0.2745    0.7000    0.3944        20
       metal     1.0000    0.3000    0.4615        20
         pop     0.2609    0.3000    0.2791        20
      reggae     0.0000    0.0000    0.0000        20
        rock     0.1053    0.1000    0.1026        20

    accuracy                         0.3200       200
   macro avg     0.2986    0.3200    0.2692       200
weighted avg     0.2986    0.3200    0.2692       200

```

## Confusion Matrix

Rows = true, Cols = predicted (classes: blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock)

```
     9    2    1    0    0    6    0    1    0    1
     0   19    0    0    0    1    0    0    0    0
     2    3    6    0    0    7    0    0    0    2
     6    1    0    2    0    5    0    6    0    0
     5    1    0    3    0    3    0    6    1    1
     1    5    0    0    0   14    0    0    0    0
     3    0    1    1    0    1    6    0    0    8
     5    3    1    3    0    0    0    6    0    2
     5    0    0    3    0    5    0    4    0    3
     0    4    5    0    0    9    0    0    0    2
```
