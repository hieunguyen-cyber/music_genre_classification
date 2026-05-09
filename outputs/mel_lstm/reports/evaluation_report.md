# Evaluation Report

- Run: `mel_lstm`
- Timestamp: 2026-05-09T23:44:35
- Device: `cuda`

## Summary

- Accuracy: **0.5195**
- Macro F1: **0.5168**
- Weighted F1: **0.5168**
- ROC-AUC (OvR, macro): **0.8782**

## Classification Report

```
              precision    recall  f1-score   support

       blues     0.5123    0.4150    0.4586       200
   classical     0.8280    0.7700    0.7979       200
     country     0.4550    0.4550    0.4550       200
       disco     0.3556    0.2400    0.2866       200
      hiphop     0.4748    0.6600    0.5523       200
        jazz     0.6782    0.5900    0.6310       200
       metal     0.7254    0.7000    0.7125       200
         pop     0.5061    0.6250    0.5593       200
      reggae     0.3901    0.4350    0.4113       200
        rock     0.3020    0.3050    0.3035       200

    accuracy                         0.5195      2000
   macro avg     0.5227    0.5195    0.5168      2000
weighted avg     0.5227    0.5195    0.5168      2000

```
