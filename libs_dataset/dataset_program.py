import numpy
import torch
import string


class ProgramGenerator:

    def __init__(self, variables_count, num_range):
        self.variables_count    = variables_count
        self.num_range          = num_range


    def generate_new(self):
        self.program_str = self._generate_program()


    def eval(self, variables = None, include_vars = False):

        if variables is None:
            variables = numpy.random.randint(0, self.num_range, self.variables_count)

           
        loc = {}
        for i in range(self.variables_count):
            loc[chr(i + ord('a'))] = variables[i]

        exec(self.program_str, globals(), loc)
        program_result = loc["y"]

        program_str = ""
        if include_vars:
            for i in range(self.variables_count):
                program_str+= str(variables[i]) + " "

        program_str = program_str + self.program_str

        return variables, program_str, program_result
    
    def _generate_program(self):

        s = ""

        s_left  = ""
        s_right = ""

        for i in range(self.variables_count):
            op_id = numpy.random.randint(3)

            if op_id == 0:
                op = "+"
            elif op_id == 1:
                op = "-"
            elif op_id == 2:
                op = "*"
           
            if numpy.random.rand() < 0.5:
                s_left+= chr(i+ord("a"))  + op
            else:
                s_right+= chr(i+ord("a")) + op

        val_left    = numpy.random.randint(self.num_range)
        val_right   = numpy.random.randint(self.num_range)

        s_left+=    str(val_left)
        s_right+=   str(val_right)

        rnd = numpy.random.randint(6)

        if rnd == 0:
            operation = "=="
        elif rnd == 1:
            operation = ">"
        elif rnd == 2:
            operation = "<"
        elif rnd == 3:
            operation = ">="
        elif rnd == 4:
            operation = "<="
        elif rnd == 5:
            operation = "!="
        
        s+= "y=" + s_left + operation + s_right

        return s



class DatasetProgram:

    def __init__(self):

        self.training_count = 10000
        self.testing_count  = 1000

        variables_count     = 8
        num_range           = 1000

        self.seq_length     = 64
        self.channels       = len(string.printable)

        self.classes_count  = 2

        self.input_shape   = (self.seq_length, self.channels)
        self.output_shape  = (self.classes_count, )



        generator = ProgramGenerator(variables_count, num_range)


        self.training_x = numpy.zeros((self.training_count, ) + self.input_shape, dtype=numpy.float32)
        self.training_y = numpy.zeros((self.training_count, ) + self.output_shape, dtype=numpy.float32)
        self.testing_x  = numpy.zeros((self.testing_count, ) + self.input_shape, dtype=numpy.float32)
        self.testing_y  = numpy.zeros((self.testing_count, ) + self.output_shape, dtype=numpy.float32)


        for i in range(self.training_count):
            generator.generate_new()
            _, program_str, program_result = generator.eval(None, True)

            if i%1000 == 0:
                print(program_str, program_result)

            program_one_hot = self._string_vectorizer(program_str)
            program_one_hot = numpy.array(program_one_hot)

            length = program_one_hot.shape[0]
            
            self.training_x[i][0:length,:] = program_one_hot.copy()

            if program_result == True:
                self.training_y[i][1] = 1.0
            else:
                self.training_y[i][0] = 1.0
 
        
        for i in range(self.testing_count):
            generator.generate_new()
            _, program_str, program_result = generator.eval(None, True)

            program_one_hot = self._string_vectorizer(program_str)
            program_one_hot = numpy.array(program_one_hot)

            length = program_one_hot.shape[0]
            
            self.testing_x[i][0:length,:] = program_one_hot.copy()

            if program_result == True:
                self.testing_y[i][1] = 1.0
            else:
                self.testing_y[i][0] = 1.0
   
        

        print("\n\n\n\n")
        print("dataset summary : \n")
        print("training_count = ", self.get_training_count())
        print("testing_count  = ", self.get_testing_count())
        print("seq_length       = ", self.seq_length)
        print("channels         = ", self.channels)

        print("classes_count =  ", self.classes_count)
        print("training_x shape ", self.training_x.shape)
        print("training_y shape ", self.training_y.shape)
        print("testing_x shape  ", self.testing_x.shape) 
        print("testing_y shape  ", self.testing_y.shape)
        print("\n")


    
    def get_training_count(self):
        return len(self.training_x)

    def get_testing_count(self):
        return len(self.testing_x)

    def get_training_batch(self, batch_size = 32):
        return self._get_batch(self.training_x, self.training_y, batch_size)

    def get_testing_batch(self, batch_size = 32):
        return self._get_batch(self.testing_x, self.testing_y, batch_size)

    def _get_batch(self, x, y, batch_size = 32):
        result_x = torch.zeros((batch_size, ) + self.input_shape)
        result_y = torch.zeros((batch_size, self.classes_count))

        for i in range(batch_size):  
            idx = numpy.random.randint(len(x))

            result_x[i]  = torch.from_numpy(x[idx]).float()
            result_y[i]  = torch.from_numpy(y[idx]).float()

        return result_x, result_y

 
    def _string_vectorizer(self, str, alphabet=string.printable):
        vector = [[0 if char != letter else 1 for char in alphabet] 
                  for letter in str]
        return vector



if __name__ == "__main__":
    dataset = DatasetProgram()

    '''
    results = []
    lengths = []

    program_generator = ProgramGenerator(8, 10)
    for i in range(5000):

        program_generator.generate_new()

        variables, program_str, program_result = program_generator.eval(None, True)

        results.append(program_result)
        lengths.append(len(program_str))

    print("\n\n")
    print(program_str)
    print("\n\n")

    print(numpy.mean(results))
    print(numpy.mean(lengths), numpy.max(lengths))

    '''


