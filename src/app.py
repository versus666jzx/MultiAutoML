import numpy as np
import pandas as pd
import streamlit as st

from utils import data_utils, multi_light_auto_ml


with st.sidebar:

	train = test = None

	task_type = st.selectbox(
		label='Выберите тип задачи',
		options=[
			'Бинарная классификация',
			'Мультиклассовая классификация',
			'Регрессия'
		]
	)

	st.markdown('---')

	data_split_type = st.radio(
		label='Представление данных',
		options=[
			'Одним файлом',
			'Train и Test раздельно'
		]
	)

	data = st.file_uploader(
		label='Загрузить данные',
		accept_multiple_files=True if data_split_type == 'Train и Test раздельно' else False,
		type=['csv'],
		help='Файлы должны называться train.csv и test.csv'
	)

	if data_split_type == 'Одним файлом':
		test_size = st.number_input(
			label='Выберите размер теста (%)',
			min_value=5,
			max_value=40,
			step=5
		) / 100  # переводим в % для train_test_split

	if data is not None or data_split_type == 'Train и Test раздельно' and len(data) > 0:
		data_utils.check_correct_uploaded_data(data=data, data_split_type=data_split_type)
		if data_split_type == 'Одним файлом':
			train, test = data_utils.preprocess_data(data=data, test_size=test_size)
		else:
			train, test = data_utils.preprocess_data(data=data)

		if train is not None:
			if task_type == 'Бинарная классификация' or task_type == 'Регрессия':

				target_column_name = st.selectbox(
					label='Выберите колонку с таргетом',
					options=train.columns
				)

			elif task_type == 'Мультиклассовая классификация':
				target_column_name = st.multiselect(
					label='Выберите колонки с таргетом',
					options=train.columns
				)

			columns_to_drop = st.multiselect(
				label='Выберите какую колонку исключить',
				options=train.columns
			)

			metrics = st.multiselect(
				label='Выберите метрику оценки',
				options=[
					'F1',
					'Accuracy',
					'ROC-AUC'
				],
				default='F1'
			)

			timeout_learn = st.number_input(
				label='Таймаут обучения (сек)',
				min_value=10,
				max_value=7200,
				step=10,
				value=3600
			)

if task_type == 'Бинарная классификация':
	data_utils.validate_continue(train, test)
	start_automl = st.button('Запустить AutoML')
	if start_automl:
		automl = multi_light_auto_ml.MultiAutoML(task_type=data_utils.task_type_mapper(task_type), timeout=timeout_learn)
		train_pred = automl.fit_predict(train, target_column=target_column_name, drop_columns=columns_to_drop)
		test_pred = automl.predict(test)
		try:
			evals = automl.evaluate(test[target_column_name], test_pred, data_utils.metric_mapper(metrics))
		except KeyError as err:
			evals = None
			st.error(f'В тестовой выборке отсутствует target стоблец {target_column_name}. {err}')
		st.write(evals)