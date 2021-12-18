# Final Project

This final project referenced google-research slot-attention.
Slot-Attention: (https://github.com/google-research/google-research/tree/master/slot_attention)

## Slot-Attention

To train the object discovery model, navigate to the parent directory
(`google-research`) and run:

```
python -m slot_attention.object_discovery.train
```

## VQA Training

```
python -m slot_attention.vqa
```

## VQA Test

```
python -m slot_attention.vqa_eval
```

## Pre-trained model checkpoints

We provide checkpoints of pre-trained models on the CLEVR dataset. 
The checkpoints are available on Google Cloud Storage:
* Set prediction: [gs://gresearch/slot-attention/set-prediction](https://console.cloud.google.com/storage/browser/gresearch/slot-attention/set-prediction) (~5MB)
* VQA model: 
