---
license: mit
dataset_info:
  features:
  - name: question
    dtype: string
  - name: subject
    dtype: string
  - name: choices
    sequence: string
  - name: answer
    dtype:
      class_label:
        names:
          '0': A
          '1': B
          '2': C
          '3': D
  splits:
  - name: test
    num_bytes: 12194284
    num_examples: 14042
  - name: validation
    num_bytes: 1336478
    num_examples: 1531
  - name: dev
    num_bytes: 218505
    num_examples: 285
  download_size: 5845296
  dataset_size: 13749267
configs:
- config_name: default
  data_files:
  - split: test
    path: data/test-*
  - split: validation
    path: data/validation-*
  - split: dev
    path: data/dev-*
---
Example record in the dataset:

```
{
  'question': 'Кое от следните най-добре описва баланса, който Върховният съд е постигнал между клаузата за установяване и клаузата за свободно упражняване?',
  'subject': 'high_school_government_and_politics',
  'choices': [
     'Свободата на словото е защитена, освен в определени ситуации, като крещенето „пожар“ в претъпкан театър.',
     'След като една църква бъде призната от федералното правителство, нейният статут на освободена от данъци никога не може да бъде отменен.',
     'След като Конгресът създаде административна агенция, тази агенция може да бъде разпусната само чрез конституционна поправка.',
     'Спонсорираната от държавата молитва по време на учебните часове е забранена, но доброволната молитва от ученически групи преди училище е разрешена.'
  ],
  'answer': 3
}
```