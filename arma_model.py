import matplotlib.pyplot as plt
from more_itertools import chunked
import numpy

class ARMAModel:
    def __init__(self, coefficient_normalisation, x_train, x_validation, x_test, x_train_arma, x_validation_arma, x_test_arma, y_train_arma, y_validation_arma, y_test_arma):
        self.coefficient_normalisation = coefficient_normalisation
        self.x_train = x_train
        self.x_validation = x_validation
        self.x_test = x_test
        self.x_train_arma = x_train_arma
        self.x_validation_arma = x_validation_arma
        self.x_test_arma = x_test_arma
        self.y_train_arma = y_train_arma
        self.y_validation_arma = y_validation_arma
        self.y_test_arma = y_test_arma

    def training_gradient_descent(self, number_epochs, learning_rate, batch_size, p, q):
        train_length = len(self.x_train_arma)
        validation_length = len(self.x_validation_arma)

        self.parameters = numpy.zeros(p + q + 1)  # p : les coefficients AR, q : les coefficients MA, le dernier terme est le biais.
        self.residuals = numpy.zeros(max(p, q))
        
        def arma_model_forecasting(feature):
            ar_term = numpy.dot(self.parameters[:p], feature)
            ma_term = numpy.dot(self.parameters[p:p+q], self.residuals[-q:])
            return ar_term + ma_term + self.parameters[-1]

        loss_function_mse_train = lambda predictions: sum([(predictions[index_feature] - self.y_train_arma[index_feature]) ** 2 for index_feature in range(train_length)]) / (2 * train_length)
        loss_function_mse_validation = lambda predictions: sum([(predictions[index_feature] - self.y_validation_arma[index_feature]) ** 2 for index_feature in range(validation_length)]) / (2 * validation_length)

        list_epochs = []
        list_mse_loss_values_for_train_per_epoch = []
        list_mse_loss_values_for_validation_per_epoch = []

        x_train_arma_batches = numpy.array(list(chunked(self.x_train_arma, batch_size)))
        y_train_arma_batches = numpy.array(list(chunked(self.y_train_arma, batch_size)))

        number_of_batches = len(x_train_arma_batches)

        for epoch in range(number_epochs):

            for index_batch in range(number_of_batches):
                number_of_elements_current_batch = len(x_train_arma_batches[index_batch])

                current_batch_train_predictions = list(map(arma_model_forecasting, x_train_arma_batches[index_batch]))

                for index_periodicity in range(p):
                    dloss_dak = (1 / number_of_elements_current_batch) * sum([x_train_arma_batches[index_batch][index_feature][index_periodicity] * (current_batch_train_predictions[index_feature] - y_train_arma_batches[index_batch][index_feature])\
                                                                                                                                    for index_feature in range(number_of_elements_current_batch)])
                    self.parameters[index_periodicity] -= learning_rate * dloss_dak

                for index_periodicity in range(q):
                    dloss_dbk = (1 / number_of_elements_current_batch) * sum(self.residuals[-q+index_periodicity] * ((current_batch_train_predictions) - y_train_arma_batches[index_batch]))
                    self.parameters[p + index_periodicity] -= learning_rate * dloss_dbk

                dloss_db = (1 / number_of_elements_current_batch) * sum([(current_batch_train_predictions[index_feature] - y_train_arma_batches[index_batch][index_feature]) for index_feature in range(number_of_elements_current_batch)])
                self.parameters[-1] -= learning_rate * dloss_db

            train_predictions_current_epoch = list(map(arma_model_forecasting, self.x_train_arma))
            validation_predictions_current_epoch = list(map(arma_model_forecasting, self.x_validation_arma))
            
            self.residuals = (self.y_train_arma) - (train_predictions_current_epoch)

            loss_train_current_epoch = loss_function_mse_train(train_predictions_current_epoch)
            loss_validation_current_epoch = loss_function_mse_validation(validation_predictions_current_epoch)

            list_epochs.append(epoch)
            list_mse_loss_values_for_train_per_epoch.append(loss_train_current_epoch)
            list_mse_loss_values_for_validation_per_epoch.append(loss_validation_current_epoch)

            if epoch % 1000 == 0 and epoch != 0:
                print(f"epoch : {epoch}")
                print(loss_train_current_epoch, loss_validation_current_epoch)
                print(self.parameters)
                print("#" * 20)

        plt.figure(figsize=(15, 6))
        plt.plot(list_epochs[100:], list_mse_loss_values_for_train_per_epoch[100:], "b")
        plt.plot(list_epochs[100:], list_mse_loss_values_for_validation_per_epoch[100:], "r")
        plt.show()

    def show_forecast_of_arma_model(self, p, q):
        if not hasattr(self, "parameters"):
            print("Error: You must first train your model!")
        else:
            arma_model_forecast = lambda feature: numpy.dot(self.parameters[:p], feature) + numpy.dot(self.parameters[p:p+q], self.residuals[-q:]) + self.parameters[-1]

            train_predictions = list(map(arma_model_forecast, self.x_train_arma))
            validation_predictions = list(map(arma_model_forecast, self.x_validation_arma))
            test_predictions = list(map(arma_model_forecast, self.x_test_arma))

            plt.figure(figsize=(15, 6))
            plt.plot(self.x_train, self.y_train_arma, "o", color="cyan", linestyle='dashed')
            plt.plot(self.x_train, train_predictions, "o", color="blue", linestyle='dashed')

            plt.plot(self.x_validation, self.y_validation_arma, "o", color="cyan", linestyle='dashed')
            plt.plot(self.x_validation, validation_predictions, "o", color="blue", linestyle='dashed')

            plt.plot(self.x_test, self.y_test_arma, "o", color="orange", linestyle='dashed')
            plt.plot(self.x_test, test_predictions, "o", color="red", linestyle='dashed')

            plt.show()