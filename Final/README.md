# Final Project

This final project referenced google-research slot-attention.
Slot-Attention: (https://github.com/google-research/google-research/tree/master/slot_attention)

## Slot-Attention

To train the set prediction model, navigate to the parent directory 
run:

```
python -m slot_attention.set_prediction.train
```

## VQA Training

```
python -m Final.vqa
```

## VQA Test

```
python -m Final.vqa_eval
```

## Pre-trained model checkpoints

We provide checkpoints of pre-trained models on the CLEVR dataset.

Slot-Attention Set prediction model dir: "/tmp/set_prediction/".

VQA model dir: "/tmp/vqa/".

The checkpoints are available on Google Cloud Storage:
* Slot-Attention Set prediction: [gs://FinalProject/set_prediction](https://drive.google.com/drive/folders/1xyrDDiiNmBDjOhgbvYFyVvRewzBbcRCK?usp=sharing)
* VQA model: [gs://FinalProject/VQA](https://drive.google.com/drive/folders/1lHClh1SEhzrCCorEDvvnzXKUuLKhdEdw?usp=sharing)
