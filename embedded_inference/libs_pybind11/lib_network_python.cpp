#include <pybind11/pybind11.h>
#include <iostream>
#include <vector>

class ModelInterfacePython
{
    public:
        ModelInterfacePython()
        {
            std::cout << "ModelInterfacePython init done\n";
        }

        std::vector<float>& forward(std::vector<float> &x)
        {
            result.resize(x.size());

            for (unsigned int i = 0; i < x.size(); i++)
                result[i] = x[i] + 1.0;

            return result;
        }

    private:
        std::vector<float> result;
};

namespace py = pybind11;

PYBIND11_MODULE(lib_network_python, m) 
{
    py::class_<ModelInterfacePython>(m, "ModelInterfacePython")
        .def(py::init<>())
        .def("forward", &ModelInterfacePython::forward);
}