import os

import streamlit as st
import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow import keras

import bert
from bert import BertModelLayer
from bert.loader import StockBertConfig, map_stock_config_to_params, load_stock_weights
from bert.tokenization.bert_tokenization import FullTokenizer

import SessionState

max_seq_len = 512
model_name = "multi_cased_L-12_H-768_A-12"

@st.cache(allow_output_mutation=True)
def get_model_dir(model_name):
    return bert.fetch_google_bert_model(model_name, ".models")


model_dir = get_model_dir(model_name)
model_ckpt = os.path.join(model_dir, "bert_model.ckpt")

@st.cache(allow_output_mutation=True)
def get_tokenizer(model_name, model_ckpt):
    do_lower_case = not (model_name.find("cased") == 0 or model_name.find("multi_cased") == 0)
    bert.bert_tokenization.validate_case_matches_checkpoint(do_lower_case, model_ckpt)
    vocab_file = os.path.join(model_dir, "vocab.txt")
    print(f'Do lower case: {do_lower_case}')
    return bert.bert_tokenization.FullTokenizer(vocab_file, do_lower_case)


tokenizer = get_tokenizer(model_name, model_ckpt)

classes = ['Трудовое право',
           'Интеллектуальная собственность, ИТ, цифровые права',
           'Коммерческие операции и поставки',
           'Закупки (юридические вопросы)',
           'Перевозки и хранение',
           'Международные санкции',
           'Экологическое право',
           'Налоговое право',
           'Антимонопольное, тарифное регулирование',
           'Строительство, недвижимость и промышленная безопасность',
           'Таможенное, валютное регулирование',
           'Недропользование (поиск, оценка месторождений УВС, разведка и добыча)']

# This should not be hashed by Streamlit when using st.cache.
# TF_HASH_FUNCS = {
#     tf.Session: id
# }

import params_flow as pf

def create_model(max_seq_len, model_dir, model_ckpt, freeze=True, adapter_size=4):
    bert_params = bert.params_from_pretrained_ckpt(model_dir)
    print(f'bert params: {bert_params}')
    bert_params.adapter_size = adapter_size
    bert_params.adapter_init_scale = 1e-5
    l_bert = bert.BertModelLayer.from_params(bert_params, name="bert")

    input_ids = keras.layers.Input(shape=(max_seq_len,), dtype='int32', name="input_ids")
    bert_output = l_bert(input_ids)

    print("bert shape", bert_output.shape)

    cls_out = keras.layers.Lambda(lambda seq: seq[:, 0, :], name='lambda')(bert_output)
    cls_out = keras.layers.Dropout(0.5)(cls_out)
    logits = keras.layers.Dense(name='dense_sin', units=768, activation=tf.math.sin)(cls_out)
    # logits = keras.layers.Dense(name='dense_tanh', units=768, activation="tanh")(cls_out)
    # logits = keras.layers.Dense(name='dense_relu', units=256, activation="relu")(cls_out)
    logits = keras.layers.Dropout(0.5)(logits)
    logits = keras.layers.Dense(name='initial_predictions', units=len(classes), activation="softmax")(logits)

    model = keras.Model(inputs=input_ids, outputs=logits)
    model.build(input_shape=(None, max_seq_len))

    model.summary()
    if freeze:
        l_bert.apply_adapter_freeze()
        l_bert.embeddings_layer.trainable = False
        model.summary()

    # Дополнительная инфа https://arxiv.org/abs/1902.00751
    # apply global regularization on all trainable dense layers
    pf.utils.add_dense_layer_loss(model,
                                  kernel_regularizer=keras.regularizers.l2(0.01),
                                  bias_regularizer=keras.regularizers.l2(0.01))

    model.compile(optimizer=pf.optimizers.RAdam(),
                  # loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")])

    bert.load_stock_weights(l_bert, model_ckpt)
    # bert.load_bert_weights(l_bert, model_ckpt)

    return model


def get_token_ids_faster(tokenizer, sentences, max_seq_len=512):
  pred_token_ids = []
  for sent in sentences:
    tokens = tokenizer.tokenize(sent)
    tokens = ["[CLS]"] + tokens[:min(len(tokens), max_seq_len-2)] + ["[SEP]"]
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    #pad
    token_ids = token_ids +[0]*(max_seq_len-len(token_ids))
    pred_token_ids.append(token_ids)
  return pred_token_ids


def get_token_ids(tokenizer, sentences):
    return get_token_ids_faster(tokenizer, sentences, max_seq_len=max_seq_len)

import tensorflow.keras.backend as K

# @st.cache(allow_output_mutation=True, hash_funcs=TF_HASH_FUNCS)
@st.cache(allow_output_mutation=True)
def load_model():
    # config = tf.ConfigProto(allow_soft_placement=True)
    # session = tf.Session(config=config)
    #
    #
    # with session.as_default():
    session = None

    model = create_model(max_seq_len, model_dir, model_ckpt, adapter_size=6)
    model.load_weights('.models/final.h5')
    # model._make_predict_function()
    print('load main model')

    layer_name = 'lambda'
    embedding_model = keras.Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    # embedding_model._make_predict_function()
    print('load embeddings model')

    outlier_model = create_outlier_model(input_dim=768, encoding_dim=768, hidden_dim=256)
    outlier_model.load_weights('.models/outlier_best_model_2020-08-05-659-0.019.h5')
    # outlier_model._make_predict_function()
    print('load outlier model')

    # session = K.get_session()
    return session, model, embedding_model, outlier_model


def create_outlier_model(input_dim, encoding_dim=256, hidden_dim=128):
    input_layer = keras.layers.Input(shape=(input_dim,), name='input_embeddings')

    encoder = keras.layers.Dense(input_dim, activation=tf.math.sin, activity_regularizer=keras.regularizers.l1(10e-5))(
        input_layer)
    encoder = keras.layers.BatchNormalization()(encoder)
    # encoder = keras.layers.Dropout(0.5)(encoder)
    encoder = keras.layers.Dense(hidden_dim, activation=tf.math.sin)(encoder)
    # encoder = keras.layers.Dropout(0.5)(encoder)
    encoder = keras.layers.BatchNormalization()(encoder)

    decoder = keras.layers.Dense(hidden_dim, activation=tf.math.sin)(encoder)
    # decoder = keras.layers.BatchNormalization()(decoder)

    decoder = keras.layers.Dense(encoding_dim, activation=tf.math.sin)(decoder)
    # decoder = keras.layers.Dropout(0.5)(decoder)
    # decoder = keras.layers.BatchNormalization()(decoder)
    decoder = keras.layers.Dense(input_dim, activation=tf.math.sin)(decoder)
    autoencoder = keras.Model(inputs=input_layer, outputs=decoder)
    autoencoder.build(input_shape=(None, input_dim))

    autoencoder.summary()

    mae = keras.losses.MeanAbsoluteError()
    adam = keras.optimizers.Adam(learning_rate=0.0021)
    autoencoder.compile(
        # optimizer=pf.optimizers.RAdam(),
        optimizer=adam,
        loss='mse',
        # loss = mae,
        metrics=['accuracy'])

    return autoencoder


def get_embeddings(embedding_model, tokenizer, vals):
  token_ids = get_token_ids(tokenizer, vals)
  return embedding_model.predict(token_ids)


def calculte_reconstruction_error(outlier_model, embeddings):
  embeddings_pred = outlier_model.predict(embeddings)

  mse = np.mean(np.power(embeddings - embeddings_pred, 2), axis=1)
  mae = np.mean(np.abs(embeddings_pred - embeddings), axis=1)
  return (mse, mae)


session, model, embedding_model, outlier_model = load_model()
mse_threshold = 0.05


def predict(text):
    outlier_embeddings = get_embeddings(embedding_model, tokenizer, [text])
    mse, mae = calculte_reconstruction_error(outlier_model, outlier_embeddings)
    print("MSE: " + str(mse) + " MAE: " + str(mae))
    result = []
    if mse[0] > mse_threshold:
        result.append((np.float32(1.-abs(mse[0]-mse_threshold)), f"Прочее (MSE: {mse})"))

    pred_token_ids = get_token_ids(tokenizer, [text])
    probabilities = model.predict(pred_token_ids)[0]
    predictions = probabilities.argmax(axis=-1)
    print(predictions)
    print(probabilities)
    print(f"text: {text}\ncategory: {classes[predictions]} prob: {probabilities[predictions]}")
    result.extend(sorted(zip(probabilities, classes), reverse=True))
    print(f'sorted: {result}')
    return result


# predict("Требуется определить порядок возмещения вреда почве в случае выявления разливов нефти при првоедении очередной плановой проверки РПН, Акт составлен, предписание выдано. Не обжаловано")
# predict("На основании ст.193 ТК РФ дисциплинарное взыскание применяется не позднее одного месяца со дня обнаружения проступка. С персональными нарушениями (проступками), когда вина конкретных людей очевидна, все понятно. Точкой отсчета будет считаться документ, фиксирующий событие (Акт нарушения, служебная записка). А что делать с Происшествиями?  Если на объекте случился пожар 01.06.  В течение 15 р.д. ведет работу комиссия по расследованию происшествий (Стандарт компании). В ходе работы комиссии производится разбирательство.  16.06. - результатом работы такой комиссии станет Акт, подписанный всеми членами. Только в этот момент становится понятно, кто персонально виновен.    Какая дата станет днем обнаружения проступка? В принципе, именно в этот момент хотелось бы начинать процесс применения дисциплинарного взыскания. Насколько это законно?   Или 15 р.д. работы комиссии, должны войти в озвученный ранее период в 1 месяц?")


import seaborn as sns
import matplotlib.pyplot as plt

top = 3
st.title("Система Юридических Консультаций")
state = SessionState.get(key=0)
if st.button('Очистить'):
    state.key += 1

ta_placeholder = st.empty()

text = ta_placeholder.text_area("Текст обращения", height=300, value='', key=state.key)

if st.button("Определить тему"):
    with st.spinner("Определение темы"):
        result = predict(text)

    st.header(f"Результат: {top} наиболее вероятные темы")
    x = []
    y = []
    for pred in result[:3]:
        # st.write(pred[1] + " " + str(pred[0].item()))
        x.append(pred[1])
        y.append(pred[0])

    data = pd.DataFrame({
        'Тема': x,
        'Вероятность': y,
    })
    st.table(data)

    plt.figure(figsize=(8, 3))
    bp = sns.barplot(x=y, y=x)
    plt.title(f'Вероятные темы:', fontsize=14)
    plt.ylabel('Тема', fontsize=12)
    plt.xlabel('Вероятность', fontsize=12)
    plt.yticks(range(len(x)), x)

    # bp.set_xticklabels(bp.get_xticklabels(), rotation=45)
    # plt.show()
    st.pyplot()

