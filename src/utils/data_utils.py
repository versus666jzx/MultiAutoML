from typing import List, Tuple, Union

import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

GLOBAL_RS = 25
Uploaded_file_types = Union[st.runtime.uploaded_file_manager.UploadedFile, List[st.runtime.uploaded_file_manager.UploadedFile]]
Processed_file_types = Tuple[pd.DataFrame, pd.DataFrame]


def preprocess_data(data: Uploaded_file_types, test_size: float = .3) -> Processed_file_types:
	train = test = None
	# если пользователь загрузил трейн и тест раздельно
	if type(data) is list:
		for data_file in data:
			# если имя файла train.csv, то загружаем в обучающую выборку
			if 'train' in data_file.name.lower():
				try:
					train = pd.read_csv(data_file)
				except Exception as err:
					st.error(f'Не удалось прочитать данные из файла {data_file.name} т.к.: {err}')
			# если имя файла test.csv, то загружаем в тестовую выборку
			elif 'test'in data_file.name.lower():
				try:
					test = pd.read_csv(data_file)
				except Exception as err:
					st.error(f'Не удалось прочитать данные из файла {data_file.name} т.к.: {err}')
		return train, test

	else:  # если пользователь загрузил данные одним файлом
		try:
			pd_data = pd.read_csv(data)
		except Exception as err:
			st.error(f'Не удалось прочитать данные из файла {data.name} т.к.: {err}')
		train, test = train_test_split(pd_data, test_size=test_size, random_state=GLOBAL_RS)
		return train, test


def check_correct_uploaded_data(data: Uploaded_file_types, data_split_type: str) -> None:
	# if data_split_type == 'Одним файлом' and data is None:
	# 	st.warning('Для начала загрузите данные.')
	# 	st.stop()
	if data_split_type == 'Train и Test раздельно':
		# if len(data) == 0:
		# 	st.warning('Для начала загрузите данные.')
		# 	st.stop()
		if len(data) != 2:
			st.error('Выбрано неверное количество файлов. Выберите только train.csv и test.csv.')
			# st.stop()
		if len(data) == 2:
			names = [data_file.name.lower() for data_file in data]
			if 'train.csv' not in names or 'test.csv' not in names:
				st.error('Неверные имена загружаемых файлов. Имена файлов должны быть train.csv и test.csv.')
				# st.stop()


def validate_continue(train, test):
	if train is None or test is None:
		st.info('Загрузите данные.')
		st.stop()


def task_type_mapper(task_type: str) -> str:
	maper = {
		'Бинарная классификация': 'binary',
		'Мультиклассовая классификация': 'multiclass',
		'Регрессия': 'reg'
	}
	return maper[task_type]


def metric_mapper(metricks: List[str]):
	metrics_mapper = {
		'F1': f1_score,
		'Accuracy': accuracy_score,
		'ROC-AUC': roc_auc_score
	}
	return [metrics_mapper[metric] for metric in metricks]
