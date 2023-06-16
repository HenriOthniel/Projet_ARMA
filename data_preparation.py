import pandas
import matplotlib.pyplot as plt
import seaborn
import numpy

class DataPreparation:
	"""Cette classe me permet de gérer le jeu de données"""
	def __init__(self, csv_path, ratio):
		self.number_of_travelers_df = pandas.read_csv(csv_path, sep=",")
		self.dataset_length = len(self.number_of_travelers_df)
		self.ratio = ratio
		self.index_split_1 = int(self.dataset_length * self.ratio[0])
		self.index_split_2 = int(self.dataset_length * (self.ratio[0] + self.ratio[1]))
		

	def prepare_data_for_arma_model(self, periodicity):
		coefficient_normalisation = self.number_of_travelers_df["passengers"].values.max()
		self.number_of_travelers_df["passengers"] /= coefficient_normalisation

		self.number_of_travelers_df["index_time"] = numpy.array([index_date for index_date in range(0, self.dataset_length)])

		train_dataset_df = self.number_of_travelers_df.loc[:self.index_split_1 - 1]
		validation_dataset_df = self.number_of_travelers_df.loc[self.index_split_1:self.index_split_2 - 1]
		test_dataset_df = self.number_of_travelers_df.loc[self.index_split_2:]
		
		x_train = train_dataset_df["index_time"].values[periodicity:]
		x_validation = validation_dataset_df["index_time"].values[periodicity:]
		x_test = test_dataset_df["index_time"].values[periodicity:]

		x_train_arma = []  # liste de feature (feature est une liste qui contient les p précédentes valeurs)
		for index_feature in range(0, len(train_dataset_df) - periodicity):
			x_train_arma.append(train_dataset_df["passengers"].values[index_feature: index_feature + periodicity])

		x_validation_arma = []  # liste de feature (feature est une liste qui contient les p précédentes valeurs)
		for index_feature in range(0, len(validation_dataset_df) - periodicity):
			x_validation_arma.append(validation_dataset_df["passengers"].values[index_feature: index_feature + periodicity])

		x_test_arma = []  # liste de feature (feature est une liste qui contient les p précédentes valeurs)
		for index_feature in range(0, len(test_dataset_df) - periodicity):
			x_test_arma.append(test_dataset_df["passengers"].values[index_feature: index_feature + periodicity])


		y_train_arma = train_dataset_df["passengers"].values[periodicity:]
		y_validation_arma = validation_dataset_df["passengers"].values[periodicity:]
		y_test_arma = test_dataset_df["passengers"].values[periodicity:]

		return coefficient_normalisation, x_train, x_validation, x_test, x_train_arma, x_validation_arma, x_test_arma, y_train_arma, y_validation_arma, y_test_arma

	def show_dataset(self):
		plt.figure(figsize=(15, 6))
		seaborn.scatterplot(x="index_time", y="passengers", data=self.number_of_travelers_df)
		plt.show()