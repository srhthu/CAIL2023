## Data Format
For stage 1:
- `train/train_query.json`: query cases. Each line is a json string:
    - `id` (*int*): identifier
    - `query` (*str*): the first paragraph and the `fact` content
    - `fact` (*str*): the accucation of procuratorate


- `train/train_label.trec`: similar case labels. Each line is a tuple saperated by `\t`
  - `<qid>`: query id that match the `id` in `train_query.json`
  - `0`. disgard
  - `<pid>`: candidate id 
  - `<label>`: take the value of 0,1,2,3  
  19169 lines.

The candidates are from LeCaRDv2.

Note: for some samples (78 of 640) in `train_query.json`, the `fact` cannot match part of the `query`. The possible reason is that fact is extracted in early days.