#include <torch/extension.h>
cpu_adam_impl.cpp
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
m.def("ds_adam_step", torch::wrap_pybind_function(ds_adam_step), "ds_adam_step");
m.def("ds_adam_step_plus_copy", torch::wrap_pybind_function(ds_adam_step_plus_copy), "ds_adam_step_plus_copy");
m.def("create_adam_optimizer", torch::wrap_pybind_function(create_adam_optimizer), "create_adam_optimizer");
m.def("destroy_adam_optimizer", torch::wrap_pybind_function(destroy_adam_optimizer), "destroy_adam_optimizer");
}