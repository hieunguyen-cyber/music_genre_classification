# Track-Level Evaluation Report

**Aggregation mode:** `mean_proba`  
**Total tracks:** 200

## Summary

- Track-level Accuracy: **0.7300**
- ROC-AUC (OvR, macro): **0.9627**

## Classification Report

```
              precision    recall  f1-score   support

       blues     0.7059    0.6000    0.6486        20
   classical     0.8696    1.0000    0.9302        20
     country     0.6250    0.7500    0.6818        20
       disco     0.6429    0.4500    0.5294        20
      hiphop     0.8125    0.6500    0.7222        20
        jazz     0.7826    0.9000    0.8372        20
       metal     0.9000    0.9000    0.9000        20
         pop     0.6000    0.9000    0.7200        20
      reggae     1.0000    0.6000    0.7500        20
        rock     0.5238    0.5500    0.5366        20

    accuracy                         0.7300       200
   macro avg     0.7462    0.7300    0.7256       200
weighted avg     0.7462    0.7300    0.7256       200

```

## Confusion Matrix

Rows = true, Cols = predicted (classes: blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock)

```
    12    0    3    0    0    2    0    0    0    3
     0   20    0    0    0    0    0    0    0    0
     3    0   15    0    0    0    0    0    0    2
     0    1    0    9    2    0    0    6    0    2
     1    0    0    1   13    1    0    4    0    0
     0    1    1    0    0   18    0    0    0    0
     0    0    0    0    0    0   18    0    0    2
     0    0    1    0    0    0    0   18    0    1
     1    0    1    3    1    1    1    0   12    0
     0    1    3    1    0    1    1    2    0   11
```
