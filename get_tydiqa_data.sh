mkdir data
wget https://storage.googleapis.com/tydiqa/v1.0/tydiqa-v1.0-train.jsonl.gz -P data
wget https://storage.googleapis.com/tydiqa/v1.0/tydiqa-v1.0-dev.jsonl.gz -P data
cd data
gunzip tydiqa-v1.0-train.jsonl.gz
gunzip tydiqa-v1.0-dev.jsonl.gz