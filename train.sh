python prepare_train_data.py

mkdir models &>/dev/null

for model in tiny mini small medium
do
  python create_model.py --base-model $model  models/bert-ner-$model || exit
  tensorflowjs_converter --input_format=tf_saved_model --output_format=tfjs_graph_model models/bert-ner-$model models/bert-ner-$model-js
  tensorflowjs_converter --quantize_uint16 --input_format=tf_saved_model --output_format=tfjs_graph_model models/bert-ner-$model models/bert-ner-$model-js16
  tensorflowjs_converter --quantize_uint8 --input_format=tf_saved_model --output_format=tfjs_graph_model models/bert-ner-$model models/bert-ner-$model-js8
  python save_model_metadata.py models/bert-ner-$model-js
  python save_model_metadata.py models/bert-ner-$model-js16
  python save_model_metadata.py models/bert-ner-$model-js8
done



