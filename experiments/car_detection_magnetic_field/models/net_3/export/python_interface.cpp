#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iostream>
#include <vector>

#include <MyModel.h>

namespace py = pybind11;

class ModelInterfacePython
{
    public:
        ModelInterfacePython()
        {
            std::cout << "ModelInterfacePython init done\n";
        }

        std::vector<float> forward(std::vector<float> &x)
        { 
            auto in_count   = model.input_channels*model.input_height*model.input_width;
            auto out_count  = model.output_channels*model.output_height*model.output_width;

            
            for (unsigned int i = 0; i < in_count; i++)
                model.input_buffer()[i] = x[i]; 
            
            /*
            for (unsigned int i = 0; i < in_count; i++)
            {
                auto v = x[i]*4;

                if (v > 127) 
                    v = 127; 

                if (v < -127)
                    v = -127;
                    
                model.input_buffer()[i] = v; 
            }
            */


            model.forward(); 

            std::vector<float> result(out_count); 

            for (unsigned int i = 0; i < out_count; i++)
                result[i] = model.output_buffer()[i];
            
            return result;
        }

    private:
        MyModel model;
        //MagnetometetNetworkInt8 model;
};




PYBIND11_PLUGIN(LibEmbeddedNetwork)
{
    py::module mod("LibEmbeddedNetwork");

    py::class_<ModelInterfacePython>(mod, "ModelInterfacePython")
        .def(py::init<>())
        .def("forward", &ModelInterfacePython::forward);

    return mod.ptr();
}
