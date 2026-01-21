Мберт + осетинские эмбеддинги:
```
pip install -r requirements.txt
python ossetic_mlm.py \
--output_dir <ваша директория> \
--hf_token <ваш токен> \
--model_name ania3000/untrained-ossbert-e \
--tokenizer_name ania3000/ossbert-e-tokenizer \
--dataset_name ossetic-encoders/onc-unlab \
--hub_model_id ossetic-encoders/ossbert-e \
--num_train_epochs <число эпох> или --max_steps <число шагов>
```
Мберт + мультиязычные эмбеддинги:
```
pip install -r requirements.txt
python ossetic_mlm.py \
--output_dir <ваша директория> \
--hf_token <ваш токен> \
--model_name google-bert/bert-base-multilingual-cased \
--tokenizer_name ania3000/ossbert-tokenizer \
--dataset_name ossetic-encoders/onc-unlab \
--hub_model_id ossetic-encoders/ossbert \
--num_train_epochs <число эпох> или --max_steps <число шагов>
```
Установлены по умолчанию на значения, с которыми я учила, но можно менять:
```
--learning_rate (по дефолту 5e-5)
--eval_steps (по дефолту 200)
--save_steps (по дефолту 600)
--per_device_train_batch_size (по дефолту 8)
--per_device_eval_batch_size (по дефолту 8)
```
Чтобы взять только N первых предложений из выборки (по умолчанию не используется):
```
--sub_train <N>
--sub_test <N>
```
