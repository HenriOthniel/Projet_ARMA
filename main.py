import data_preparation, arma_model

data_preparation_object = data_preparation.DataPreparation(csv_path="number of travelers.csv", ratio=(0.7, 0.15))
# data_preparation_object.show_dataset()


coefficient_normalisation, x_train, x_validation, x_test, x_train_arma, x_validation_arma, x_test_arma, y_train_arma, y_validation_arma, y_test_arma = data_preparation_object.prepare_data_for_arma_model(periodicity=12)
arma_model_object = arma_model.ARMAModel(coefficient_normalisation, x_train, x_validation, x_test, x_train_arma, x_validation_arma, x_test_arma, y_train_arma, y_validation_arma, y_test_arma)
arma_model_object.training_gradient_descent(number_epochs=100000, learning_rate=0.00002, batch_size=2, p=12, q=1)
arma_model_object.show_forecast_of_arma_model(p=12, q=1)