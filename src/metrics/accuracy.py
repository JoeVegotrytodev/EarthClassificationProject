import torch

from src.PictToClass import pic_to_class
def accuracy_calc(test_prediction, answer_test, batch_size):
    true_answers_counter = 0
    position = 0

    model_pred_tensor = torch.argmax(test_prediction, dim=-1)
    # print("trch_argmax :")
    # print(model_pred_tensor, "\n")

    classes_by_pict = pic_to_class(answer_test).transpose(0, 1)
    # classes_by_pict = classes_by_pict.transpose(0, 1)
    # print("pic_to_class : \n", classes_by_pict, "\n")
    answers_vect = classesToVector(classes_by_pict)
    # print("answers_vect :", answers_vect)

    for index_of_row, pred in enumerate(model_pred_tensor):
        value = pred.item()
        if value == answers_vect[position]:
            # print(value, "=", answers[position])
            true_answers_counter = true_answers_counter + 1
        position = position + 1

    print("accuracy  =", true_answers_counter / batch_size)
    return true_answers_counter / batch_size

def classesToVector(tensor_of_classes):
    vector_of_classes = [0] * 10
    # print("tensor of classes \n", tensor_of_classes)
    # print("tensor of classes shape =", tensor_of_classes.shape)
    for idx_of_row, row_tensor in enumerate(tensor_of_classes):
        # print("rows =", row_tensor)
        # print("idx_of_row = ", idx_of_row)
        # создать счетчик по которому сверять
        vector_position = 0
        for elem in row_tensor:
            value = elem.item()
            if value == 1.0 and idx_of_row == 0:
                vector_of_classes.insert(vector_position, 0)
                vector_of_classes.pop(vector_position + 1)
                # print("vector_position =", vector_position)
                vector_position = vector_position + 1
                continue
            elif value == 1.0 and idx_of_row == 1:
                vector_of_classes.insert(vector_position, 1)
                vector_of_classes.pop(vector_position + 1)
                # print("adding_position =", vector_position)
                vector_position = vector_position + 1
                continue
            elif value == 1.0 and idx_of_row == 2:
                vector_of_classes.insert(vector_position, 2)
                vector_of_classes.pop(vector_position + 1)
                # print("adding_position =", vector_position)
                vector_position = vector_position + 1
                continue
            vector_position = vector_position + 1

    return vector_of_classes