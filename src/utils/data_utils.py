from typing import List, Tuple, Union

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
	# классификация
	accuracy_score,
	f1_score,
	roc_auc_score,
	precision_score,
	recall_score,
	average_precision_score,
	classification_report,
	confusion_matrix,
	precision_recall_curve,
	# регрессия
	mean_absolute_error,
	mean_squared_error,
	r2_score,
	mean_absolute_percentage_error,
	# AUC
	roc_curve
)

GLOBAL_RS = 25
Uploaded_file_types = Union[st.runtime.uploaded_file_manager.UploadedFile, List[st.runtime.uploaded_file_manager.UploadedFile]]
Processed_file_types = Tuple[pd.DataFrame, pd.DataFrame]


def preprocess_data(data: Uploaded_file_types, test_size: float = .3) -> Processed_file_types:
	"""
	Читает загруженные пользователем в streamlit данные и разделяет их на train и test
	если данные загружены одним файлом.

	"""
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
			elif 'test' in data_file.name.lower():
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
	"""
	Проверка корректности загруженных данных.
	Выдаст пользователю ошибку, если:
	 - загружено больше 2-х файлов
	 - файлы называются не train.csv и test.csv

	"""

	if data_split_type == 'Train и Test раздельно':
		if len(data) != 2:
			st.error('Выбрано неверное количество файлов. Выберите только train.csv и test.csv.')
		if len(data) == 2:
			names = [data_file.name.lower() for data_file in data]
			for name in names:
				if 'train' not in name:
					if 'test' not in name:
						st.error('Неверные имена загружаемых файлов. Имена файлов должны содержать слова train и test.')


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


def smape(y_true, y_pred):
	y_true, y_pred = np.array(y_true), np.array(y_pred)
	return 100/len(y_true) * np.sum(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))


def metric_mapper(metricks: List[str]) -> List:
	metrics_mapper = {
		# классификация
		'F1': f1_score,
		'Accuracy': accuracy_score,
		'ROC-AUC': roc_auc_score,
		'Precision': precision_score,
		'Recall': recall_score,
		'PR-AUC': average_precision_score,
		# регрессия
		'MAE': mean_absolute_error,
		'MSE': mean_squared_error,
		'R2': r2_score,
		'MAPE': mean_absolute_percentage_error,
		'SMAPE': smape

	}
	return [metrics_mapper[metric] for metric in metricks]


def get_metricks(task_type) -> List[str]:
	if task_type == 'Бинарная классификация':
		return ['F1', 'Accuracy', 'ROC-AUC', 'Precision', 'Recall', 'PR-AUC']
	elif task_type == 'Мультиклассовая классификация':
		return ['F1', 'Accuracy', 'Precision', 'Recall']
	elif task_type == 'Регрессия':
		return ['MAE', 'MSE', 'R2', 'MAPE', 'SMAPE']


def evals_to_clear_table(evals: dict) -> pd.DataFrame:
	clear_table = pd.DataFrame(
		columns=dict(list(evals.values())[0]).keys(),
		index=evals.keys()
		)

	for idx, key in enumerate(evals.keys()):
		metrick_values = []
		for value in evals[key].values():
			metrick_values.append(value)
		clear_table.iloc[idx, :] = metrick_values

	return clear_table


def get_classification_reports(y_true, y_pred):
	reports = []
	for model in y_pred.keys():
		report = classification_report(y_true, y_pred[model])
		st.write(report)
		reports.append(evals_to_clear_table(report))


def get_conflusion_matrix(y_true, y_pred):
	for model in y_pred.keys():
		fig = px.imshow(
			confusion_matrix(y_true, y_pred[model]),
			title=model,
			text_auto=True,
			color_continuous_scale=px.colors.sequential.Darkmint_r,

		)

		st.plotly_chart(fig)


def plot_roc_auc_curve(target_valid: pd.DataFrame, probabilities_one_valid: pd.DataFrame) -> None:
	"""
	Plotting roc_auc curve function

	:param target_valid:
	:param probabilities_one_valid:
	:return:
	"""

	for model in probabilities_one_valid.keys():

		fpr, tpr, thresholds = roc_curve(target_valid, probabilities_one_valid[model])

		# линия предсказания модели
		fig = px.area(
			x=fpr, y=tpr,
			title=f'ROC Curve для модели {model}',
			labels=dict(x='False Positive Rate', y='True Positive Rate'),
			width=700, height=500
		)
		# линия предсказания случайной модели
		fig.add_shape(
			type='line', line=dict(dash='dash'),
			x0=0, x1=1, y0=0, y1=1
		)

		st.plotly_chart(fig)


def plot_pr_curve(target_valid: pd.DataFrame, probabilities_one_valid: pd.DataFrame) -> None:
	"""
	Plotting roc_auc curve function

	:param target_valid:
	:param probabilities_one_valid:
	:return:
	"""

	for model in probabilities_one_valid.keys():

		fpr, tpr, thresholds = precision_recall_curve(target_valid, probabilities_one_valid[model])

		# линия предсказания модели
		fig = px.area(
			x=fpr, y=tpr,
			title=f'Precision Recall Curve для модели {model}',
			labels=dict(x='False Positive Rate', y='True Positive Rate'),
			width=700, height=500
		)
		# линия предсказания случайной модели
		fig.add_shape(
			type='line', line=dict(dash='dash'),
			x0=0, x1=1, y0=0, y1=1
		)

		st.plotly_chart(fig)
