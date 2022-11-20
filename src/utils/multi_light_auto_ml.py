import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from lightautoml.automl.presets.tabular_presets import TabularAutoML
from lightautoml.tasks import Task
from typing import List


class MultiAutoML:
	def __init__(
		self,
		task_type: str,
		cpu_limit: int = -1,
		timeout: int = 3600
	):
		self.LAMA_Task = Task(name=task_type)
		self.models = [TabularAutoML(task=self.LAMA_Task, cpu_limit=cpu_limit, timeout=timeout)]
		self.fitted_models = []
		self.fitted_model_names = []

	def fit_predict(self, X, target_column: str, drop_columns: List[str] = None) -> dict:
		if drop_columns is None:
			drop_columns = []
		res = {}
		for model in self.models:
			try:
				model_name = model.__name__
			except AttributeError:
				model_name = str(model.__class__)
			res[model_name] = model.fit_predict(
				train_data=X,
				roles={'target': target_column, 'drop': drop_columns}
			)
			self.fitted_models.append(model)
			self.fitted_model_names.append(model_name)
		return res

	def predict(self, X) -> dict:
		res = {}
		for model, model_name in zip(self.fitted_models, self.fitted_model_names):
			res[model_name] = np.round(model.predict(X).data)
		return res

	def evaluate(self, y_true, y_test, metricks: list) -> dict:
		if len(self.fitted_models) == 0:
			print('No one fitted model detected. Fit AutoML before evaluating.')
		res = {}
		for model, model_name in zip(self.fitted_models, self.fitted_model_names):
			res[model_name] = {}
			for metrick in metricks:
				res[model_name][metrick.__name__] = metrick(y_true, y_test[model_name])
		return res

