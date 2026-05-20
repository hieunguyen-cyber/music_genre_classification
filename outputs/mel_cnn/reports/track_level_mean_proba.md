# Track-Level Evaluation Report

**Aggregation mode:** `mean_proba`  
**Total tracks:** 200

## Summary

- Track-level Accuracy: **0.7150**
- ROC-AUC (OvR, macro): **0.9448**

## Classification Report

```
              precision    recall  f1-score   support

       blues     0.7619    0.8000    0.7805        20
   classical     0.6667    1.0000    0.8000        20
     country     0.5714    0.4000    0.4706        20
       disco     0.6522    0.7500    0.6977        20
      hiphop     0.7143    1.0000    0.8333        20
        jazz     1.0000    0.6500    0.7879        20
       metal     0.8261    0.9500    0.8837        20
         pop     0.9091    0.5000    0.6452        20
      reggae     0.8000    0.6000    0.6857        20
        rock     0.4545    0.5000    0.4762        20

    accuracy                         0.7150       200
   macro avg     0.7356    0.7150    0.7061       200
weighted avg     0.7356    0.7150    0.7061       200

```

## Confusion Matrix

Rows = true, Cols = predicted (classes: blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock)

```
    16    0    1    2    0    0    0    0    1    0
     0   20    0    0    0    0    0    0    0    0
     2    4    8    1    0    0    0    0    0    5
     0    1    0   15    3    0    1    0    0    0
     0    0    0    0   20    0    0    0    0    0
     1    4    0    0    0   13    0    0    0    2
     0    0    0    0    0    0   19    0    0    1
     0    0    1    2    3    0    1   10    1    2
     1    0    1    2    2    0    0    0   12    2
     1    1    3    1    0    0    2    1    1   10
```
