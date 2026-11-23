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
    num_bytes: 593735
    num_examples: 1119
  - name: test
    num_bytes: 639866
    num_examples: 1172
  - name: validation
    num_bytes: 166067
    num_examples: 299
  download_size: 647884
  dataset_size: 1399668
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
