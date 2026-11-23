---
license: cc-by-sa-4.0
dataset_info:
  features:
  - name: id
    dtype: string
  - name: question
    dtype: string
  - name: choices
    sequence:
    - name: text
      dtype: string
    - name: label
      dtype: string
  - name: answerKey
    dtype: string
  splits:
  - name: train
    num_bytes: 1041020
    num_examples: 2251
  - name: test
    num_bytes: 1106644
    num_examples: 2376
  - name: validation
    num_bytes: 264848
    num_examples: 570
  download_size: 1094042
  dataset_size: 2412512
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
  - split: test
    path: data/test-*
  - split: validation
    path: data/validation-*
---
