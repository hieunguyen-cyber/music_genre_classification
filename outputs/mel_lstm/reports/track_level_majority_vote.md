# Track-Level Evaluation Report

**Aggregation mode:** `majority_vote`  
**Total tracks:** 200

## Summary

- Track-level Accuracy: **0.3100**
- ROC-AUC (OvR, macro): **0.8373**

## Classification Report

```
              precision    recall  f1-score   support

       blues     0.2045    0.4500    0.2812        20
   classical     0.5263    1.0000    0.6897        20
     country     0.2727    0.1500    0.1935        20
       disco     0.2000    0.1000    0.1333        20
      hiphop     0.0000    0.0000    0.0000        20
        jazz     0.3191    0.7500    0.4478        20
       metal     0.8333    0.2500    0.3846        20
         pop     0.2308    0.3000    0.2609        20
      reggae     0.0000    0.0000    0.0000        20
        rock     0.1111    0.1000    0.1053        20

    accuracy                         0.3100       200
   macro avg     0.2698    0.3100    0.2496       200
weighted avg     0.2698    0.3100    0.2496       200

```

## Confusion Matrix

Rows = true, Cols = predicted (classes: blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock)

```
     9    2    2    0    0    6    0    0    0    1
     0   20    0    0    0    0    0    0    0    0
     2    4    3    0    0    7    0    0    0    4
     6    1    0    2    0    4    0    7    0    0
     7    1    0    2    0    2    1    7    0    0
     1    4    0    0    0   15    0    0    0    0
     4    0    1    1    0    1    5    0    0    8
     6    2    1    3    0    0    0    6    0    2
     6    1    0    2    0    4    0    6    0    1
     3    3    4    0    0    8    0    0    0    2
```
