# English
python3 make_templates.py -l en -d data/tydiqa-v1.0-train.jsonl -ft tydiqa -idf templates/en/idf_en.csv -cf

# Finnish
python3 make_templates.py -l fi -d data/tydiqa-v1.0-train.jsonl -ft tydiqa -idf templates/fi/idf_fi.csv -cf

# Russian
python3 make_templates.py -l ru -d data/tydiqa-v1.0-train.jsonl -ft tydiqa -idf templates/ru/idf_ru.csv -cf -rd

# Indonesian
python3 make_templates.py -l id -d data/tydiqa-v1.0-train.jsonl -ft tydiqa -idf templates/id/idf_id.csv -cf

# Japanese
python3 make_templates.py -l ja -d data/tydiqa-v1.0-train.jsonl -ft tydiqa -idf templates/ja/idf_ja.csv -rp -rtl

# Telugu
python3 make_templates.py -l te -d data/tydiqa-v1.0-train.jsonl -ft tydiqa -idf templates/te/idf_te.csv -rp -rtl

# Arabic
python3 make_templates.py -l ar -d data/tydiqa-v1.0-train.jsonl -ft tydiqa -idf templates/ar/idf_ar.csv -rp

# Korean
python3 make_templates.py -l ko -d data/tydiqa-v1.0-train.jsonl -ft tydiqa -idf templates/ko/idf_ko.csv -rtl