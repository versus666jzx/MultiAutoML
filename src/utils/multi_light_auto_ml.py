from typing import Any, List, Dict, Optional

import numpy as np
from lightautoml.automl.presets.tabular_presets import TabularAutoML
from lightautoml.tasks import Task
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils import data_utils


class MultiAutoML:
	def __init__(self, models):
		self.models: List = models
		self.fitted_models: List = []
		self._all_preds: Dict = {}
		self._all_eval_metrics: Dict = {}

	@property
	def get_all_preds(self) -> Dict:
		return self._all_preds

	@property
	def get_eval_all_metrics(self) -> Dict:
		return self._all_eval_metrics

	# Подставляем y_column_name а не y, т.к. LightAutoML нужно имя колонки с таргетом,
	# а не таргет
	def fit(self, X, y_column_name: str):
		# создаем плейсхолдеры для отображения хода обуччения
		message = st.empty()
		progress = st.empty()
		# создаем progress bar
		progress_bar = progress.progress(0)
		progress_step = (100 / len(self.models)) / 100
		for model in self.models:
			message.write(f'Обучается {model.model_name} ...')
			model.fit(X, y_column_name)
			self.fitted_models.append(model)
			# обновляем шаг прогресс бара
			progress_bar.progress(progress_step)
			progress_step += progress_step
		# обнуляем плейсхолдеры
		message.empty()
		progress.empty()

	# Подставляем y_column_name а не y, т.к. LightAutoML нужно имя колонки с таргетом,
	# а не таргет
	def fit_predict(self, X, y_column_name: str) -> Dict:
		for model in self.models:
			res = {model.model_name: model.fit_predict(X, y_column_name)}
			# если модель не добавлена в список обученных моделей
			if model not in self.fitted_models:
				# добавляем
				self.fitted_models.append(model)
			# если предикт для данной модели еще не был сохранен
			if model.model_name not in self._all_preds:
				# добавляем
				self._all_preds.update(res)

		return self.get_all_preds

	def predict(self, X) -> Dict:
		res = {}
		for model in self.models:
			res[model.model_name] = np.round(model.predict(X).data)
		return res

	def evaluate(self, X, y, metrics: List) -> Optional[Dict[Any, Dict[Any, Any]]]:
		if len(self.fitted_models) == 0:
			print('No one fitted model detected. Fit AutoML before evaluating.')
			return None
		res = {}
		for model in self.models:
			res[model.model_name] = model.evaluate(X, y, metrics)
		#
		self._all_eval_metrics = res
		return res


class BaseModel:
	"""
	Базовый класс для AutoML модели, реализующий стандартные методы:
		- fit
		- predict
		- fit_predict
		- evaluate
	"""
	def __init__(
		self,
		model,
		model_name: str,
		columns_to_drop: List[str] = None


	):
		self.model = model
		self.fitted_models = []
		self.fitted_model_names = []
		self.model_name = model_name
		self.columns_to_drop = columns_to_drop
		self._is_fitted = False

	@property
	def is_fitted(self) -> bool:
		return self._is_fitted

	def fit(self, X, y_column_name):
		# в LightAutoML нет метода fit, поэтому приходится вызывать fit_predict
		self.fit_predict(X, y_column_name)

	def fit_predict(self, X, y_column_name) -> dict:
		model = self.model
		if self.columns_to_drop is None:
			self.columns_to_drop = []
		preds = model.fit_predict(
			train_data=X,
			roles={'target': y_column_name, 'drop': self.columns_to_drop}
		)
		self.model = model
		self._is_fitted = True
		return preds

	def predict(self, X) -> dict:
		if not self._is_fitted:
			print(f'Fit {self.model_name} before predict.')
		# округляем, т.к. LightAutoML возвращает вероятности класса
		return np.round(self.model.predict(X).data)

	def evaluate(self, X, y, metrics: list) -> Optional[Dict[Any, Any]]:
		if not self._is_fitted:
			print(f'Fit {self.model_name} before evaluate.')
			return None
		preds = np.round(self.model.predict(X).data)
		res = {}
		for metric in metrics:
			res[metric.__name__] = metric(y, preds)
		return res


def create_base_lightautoml_model(
	model_name: str,
	task_type: str,
	timeout: int,
	columns_to_drop: List[str] = None,
	cpu_limit: int = -1
):
	base_model = BaseModel(
		model=TabularAutoML(
			task=Task(name=task_type),
			cpu_limit=cpu_limit,
			timeout=timeout
		),
		model_name=model_name,
		columns_to_drop=columns_to_drop
	)
	return base_model
