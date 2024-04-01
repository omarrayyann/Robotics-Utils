import numpy as np
from utils import Utils
import inspect


# Manipulator Class
class Manipulator:
    def __init__(self, transformation_functions=None, transformations_inputs=None):
        self.transformation_functions = []
        self.transformations_inputs = []
        self.number_of_transformations = 0
        if transformation_functions is not None:
            for index, function in enumerate(transformation_functions):
                function_input = None
                if transformations_inputs is not None and index < len(transformations_inputs):
                    function_input = transformations_inputs[index]
                self.add_transformation(function, initial_function_input=function_input)
                self.number_of_transformations += 1

    def add_transformation(self, function, initial_function_input=None):
        sig = inspect.signature(function)
        num_params = len(sig.parameters)
        if initial_function_input is None:
            initial_function_input = np.zeros(num_params).tolist()
        if num_params != len(initial_function_input):
            raise ValueError(f"Number of values in input ({len(initial_function_input)}) not equal to number of arguments in function ({num_params}).")
        self.transformation_functions.append(function)
        self.transformations_inputs.append(initial_function_input)

    def set_transformation_inputs(self, index, transformation_inputs):
        sig = inspect.signature(self.transformation_functions[index])
        num_params = len(sig.parameters)
        if num_params != len(transformation_inputs):
            raise ValueError(f"Number of values in input ({len(transformation_inputs)}) not equal to number of arguments in function ({num_params}).")
        self.transformations_inputs[index] = transformation_inputs

    def set_transformations_inputs(self, transformations_inputs):
       for index, transformation_inputs in enumerate(transformations_inputs):
            sig = inspect.signature(self.transformation_functions[index])
            num_params = len(sig.parameters)
            if num_params != len(transformation_inputs):
                raise ValueError(f"Number of values in input ({len(transformation_inputs)}) not equal to number of arguments in function ({num_params}).")
            self.transformations_inputs[index] = transformation_inputs

    # Computes the forward kinematics up to a given transformation
    def forward_kinematics(self, transformation_inputs=None, transformation_index=None):
        if transformation_index is None:
            transformation_index = self.number_of_transformations - 1
        if transformation_inputs is not None:
            self.set_transformations_inputs(transformation_inputs)
        transformation_matrix = np.eye(4)
        for i in range(transformation_index+1):
            transformation_matrix = transformation_matrix @ self.transformation_functions[i](*self.transformations_inputs[i])
        return transformation_matrix









