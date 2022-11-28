import streamlit as st

from utils import data_utils, multi_light_auto_ml

if __name__ == '__main__':
	with st.sidebar:
		# объявляем переменные для train и test
		train = test = None
		# выбор типа решаемой задачи
		task_type = st.selectbox(
			label='Выберите тип задачи',
			options=[
				'Бинарная классификация',
				'Мультиклассовая классификация',
				'Регрессия'
			]
		)

		st.markdown('---')
		# выбор представления данных, которые пользователь будет загружать
		data_split_type = st.radio(
			label='Представление данных',
			options=[
				'Одним файлом',
				'Train и Test раздельно'
			]
		)
		# загрузчик данных
		data = st.file_uploader(
			label='Загрузить данные',
			accept_multiple_files=True if data_split_type == 'Train и Test раздельно' else False,
			type=['csv'],
			help='Файлы должны называться train.csv и test.csv'
		)
		# если данные представлены одним файлом, надо выборать размер тестовой выборки в %
		if data_split_type == 'Одним файлом':
			test_size = st.number_input(
				label='Выберите размер теста (%)',
				min_value=5,
				max_value=40,
				step=5
			) / 100  # переводим в % для train_test_split
		# если данные загружены
		if data is not None or data_split_type == 'Train и Test раздельно' and len(data) > 0:
			# проверяем корректность загруженных данных
			data_utils.check_correct_uploaded_data(data=data, data_split_type=data_split_type)
			# если данные загружены одним файлом
			if data_split_type == 'Одним файлом':
				# разделяем на трейн и тест с учетом введенного пользователем размера тестовой выборки
				train, test = data_utils.preprocess_data(data=data, test_size=test_size)
			else:  # иначе просто считываем файлы обучающей и тестовой выборки в Dataframe
				train, test = data_utils.preprocess_data(data=data)

			if train is not None:
				if task_type == 'Бинарная классификация' or task_type == 'Регрессия':
					# пользователь должен выбрать таргет
					target_column_name = st.selectbox(
						label='Выберите колонку с таргетом',
						options=train.columns
					)

				elif task_type == 'Мультиклассовая классификация':
					# пользователь должен выбрать несколько таргетов
					target_column_name = st.multiselect(
						label='Выберите колонки с таргетом',
						options=train.columns
					)
				# при необходимости можно дропнуть колонки из обучающей и тестовой выборки
				columns_to_drop = st.multiselect(
					label='Выберите какую колонку исключить',
					options=train.columns
				)
				# проверим что не выбран таргет для дропа
				if target_column_name in columns_to_drop:
					st.error('Нельзя удалять target')

				# выбор метрик
				metrics = st.multiselect(
					label='Выберите метрику оценки',
					options=data_utils.get_metricks(task_type)
				)
				# ограничивает таймаут обучения
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
			# создаем инстансы базового класса модели
			# для примера созданы два одинаковых инстанса LightAutoML
			light_automl_base_model = multi_light_auto_ml.create_base_lightautoml_model(
				model_name='LightAutoML',
				task_type=data_utils.task_type_mapper(task_type),
				timeout=timeout_learn,
				columns_to_drop=columns_to_drop
			)
			light_automl_base_model_2 = multi_light_auto_ml.create_base_lightautoml_model(
				model_name='LightAutoML_2',
				task_type=data_utils.task_type_mapper(task_type),
				timeout=timeout_learn,
				columns_to_drop=columns_to_drop
			)
			# создаем MultiLightAutoML класс из списка базовых классов модели
			multiautoml = multi_light_auto_ml.MultiAutoML(
				[light_automl_base_model, light_automl_base_model_2]
			)
			# обучаем все базовые модели
			multiautoml.fit(train, target_column_name)
			# получаем предикты всех базовых моделей
			test_pred = multiautoml.predict(test)
			try:
				# получаем оценки для всех базовых моделей
				evals = multiautoml.evaluate(test, test[target_column_name], data_utils.metric_mapper(metrics))
			except KeyError as err:
				evals = None
				st.error(f'В тестовой выборке отсутствует target стоблец {target_column_name}. {err}')
			st.write(evals)

	if task_type == 'Регрессия':
		data_utils.validate_continue(train, test)
		start_automl = st.button('Запустить AutoML')
		if start_automl:
			# создаем инстансы базового класса модели
			# для примера созданы два одинаковых инстанса LightAutoML
			light_automl_base_model = multi_light_auto_ml.create_base_lightautoml_model(
				model_name='LightAutoML',
				task_type=data_utils.task_type_mapper(task_type),
				timeout=timeout_learn,
				columns_to_drop=columns_to_drop
			)
			light_automl_base_model_2 = multi_light_auto_ml.create_base_lightautoml_model(
				model_name='LightAutoML_2',
				task_type=data_utils.task_type_mapper(task_type),
				timeout=timeout_learn,
				columns_to_drop=columns_to_drop
			)
			# создаем MultiLightAutoML класс из списка базовых классов модели
			multiautoml = multi_light_auto_ml.MultiAutoML(
				[light_automl_base_model, light_automl_base_model_2]
			)
			# обучаем все базовые модели
			multiautoml.fit(train, target_column_name)
			# получаем предикты всех базовых моделей
			test_pred = multiautoml.predict(test)
			try:
				# получаем оценки для всех базовых моделей
				evals = multiautoml.evaluate(test, test[target_column_name], data_utils.metric_mapper(metrics))
			except KeyError as err:
				evals = None
				st.error(f'В тестовой выборке отсутствует target стоблец {target_column_name}. {err}')
			st.write(evals)