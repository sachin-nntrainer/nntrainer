#include <iostream>
#include <vector>
#include <memory>
#include <layer.h>
#include <model.h>
#include <neuralnet.h>
#include <nntrainer-api-common.h>
#include <optimizer.h>
#include <tensor.h>
#include <util_func.h>
#include <dataset.h>

using LayerHandle = std::shared_ptr<ml::train::Layer>;
using ModelHandle = std::unique_ptr<ml::train::Model>;

std::vector<LayerHandle> createSimpleGraph() {
  using ml::train::createLayer;

  std::vector<LayerHandle> layers;

  // Input layer
  layers.push_back(
    createLayer("input", {nntrainer::withKey("name", "input0"),
                          nntrainer::withKey("input_shape", "1:1:768")}));

  // Hidden layer
  layers.push_back(
    createLayer("fully_connected",
                {nntrainer::withKey("unit", 256),
                 nntrainer::withKey("activation", "relu")}));

  // Output layer
  layers.push_back(
    createLayer("fully_connected",
                {nntrainer::withKey("unit", 128),
                 nntrainer::withKey("activation", "relu")}));
   
   layers.push_back(
    createLayer("fully_connected",
                {nntrainer::withKey("unit", 10)}));              

  // Loss layer
  layers.push_back(createLayer("cross_softmax"));

  return layers;
}

int main(){

    std::cout << "Demonstrating trainMeZO with a simple neural network" << std::endl;

    // Create model
    auto model = ml::train::createModel();

    // Add layers
    auto layers = createSimpleGraph();
    for (auto &layer : layers) {
        model->addLayer(layer);
    }

    try {
        // Compile the model
        model->compile();
        std::cout << "Model compiled successfully." << std::endl;
    } catch (const std::exception &e) {
        std::cerr << "Error compiling model: " << e.what() << std::endl;
        return 1;
    }

    try {
        // Initialize the model
        model->initialize();
        std::cout << "Model initialized successfully." << std::endl;
    } catch (const std::exception &e) {
        std::cerr << "Error initializing model: " << e.what() << std::endl;
        return 1;
    }

    model->summarize(std::cout, ML_TRAIN_SUMMARY_MODEL);

    auto nn = static_cast<nntrainer::NeuralNetwork*>(model.get());

    float *input = new float[768];

    for(int i=0;i<768;i++)
     input[i]=i;

    float *output=model->inference(1, {input})[0];

    for(int i=0;i<10;i++)
      std::cout<<output[i]<<" ";
    
    std::cout<<std::endl;  

    // Get parameter pointers before training
    std::vector<nntrainer::Tensor*> params = nn->getParameterPointers();
    std::cout << "Number of weight tensors before training: " << params.size() << std::endl;

    for(auto *t:params){
      std::cout<<t->getName()<<" "<<" "<<std::endl;
      int size = t->getDim().getDataLen();
      float* val = (float*)(t->getData());

      std::cout<<val[0]<<" ";
      val[0]=5;
      std::cout<<val[0]<<" ";
      float* val1 = (float*)(t->getData());
      std::cout<<val1[0]<<" ";

      std::cout<<std::endl;
    }

    output=model->inference(1, {input})[0];

    return 0;
}

// using LayerHandle = std::shared_ptr<ml::train::Layer>;
// using ModelHandle = std::unique_ptr<ml::train::Model>;
// using DatasetHandle = std::unique_ptr<ml::train::Dataset>;

// /**
//  * @brief Create a simple neural network for demonstration
//  */


// /**
//  * @brief Simple data generator for training
//  */
// int getSample_train(float **outVec, float **outLabel, bool *last, void *user_data) {
//   static int sample_count = 0;
//   static const int num_samples = 3;

//   // Simple XOR-like data
//   static const std::vector<std::vector<float>> inputs = {
//     {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}, // 0
//     {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f}, // 1
//     {0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f}, // 0.5
//   };

//   static const std::vector<std::vector<float>> labels = {
//     {0.0f}, // label for 0
//     {1.0f}, // label for 1
//     {0.5f}, // label for 0.5
//   };

//   if (sample_count >= num_samples) {
//     *last = true;
//     sample_count = 0;
//     return 0;
//   }

//   // Copy input data
//   for (size_t i = 0; i < inputs[sample_count].size(); ++i) {
//     (*outVec)[i] = inputs[sample_count][i];
//   }

//   // Copy label data
//   for (size_t i = 0; i < labels[sample_count].size(); ++i) {
//     (*outLabel)[i] = labels[sample_count][i];
//   }

//   sample_count++;
//   *last = false;

//   return 0;
// }

// /**
//  * @brief Create a simple dataset for training
//  */
// DatasetHandle createSimpleDataset() {
//   auto dataset = ml::train::createDataset(ml::train::DatasetType::GENERATOR, getSample_train);

//   // Set dataset properties
//   dataset->setProperty({
//     "buffer_size=3",
//     "input_shape=1:1:10",
//     "label_shape=1:1:1"
//   });

//   return dataset;
// }

// int main() {
//     std::cout << "Demonstrating trainMeZO with a simple neural network" << std::endl;

//     // Create model
//     auto model = ml::train::createModel();

//     // Add layers
//     auto layers = createSimpleGraph();
//     for (auto &layer : layers) {
//         model->addLayer(layer);
//     }

//     // Set optimizer to MeZO (though trainMeZO doesn't use it directly)
//     auto optimizer = ml::train::createOptimizer("mezo");
//     model->setOptimizer(optimizer);

//     // Add dataset
//     auto dataset = createSimpleDataset();
//     model->setDataset(ml::train::DatasetModeType::MODE_TRAIN, dataset);

//     try {
//         // Compile the model
//         model->compile();
//         std::cout << "Model compiled successfully." << std::endl;
//     } catch (const std::exception &e) {
//         std::cerr << "Error compiling model: " << e.what() << std::endl;
//         return 1;
//     }

//     try {
//         // Initialize the model
//         model->initialize();
//         std::cout << "Model initialized successfully." << std::endl;
//     } catch (const std::exception &e) {
//         std::cerr << "Error initializing model: " << e.what() << std::endl;
//         return 1;
//     }

//     // Cast to NeuralNetwork to access trainMeZO
//     auto nn = static_cast<nntrainer::NeuralNetwork*>(model.get());

//     // Get parameter pointers before training
//     std::vector<nntrainer::Tensor*> params = nn->getParameterPointers();
//     std::cout << "Number of weight tensors before training: " << params.size() << std::endl;

//     // Print initial weights
//     if (!params.empty()) {
//         const auto& first_tensor = *params[0];
//         if (first_tensor.getDataType() == nntrainer::Tdatatype::FP32) {
//             float* data = reinterpret_cast<float*>(first_tensor.getData());
//             if (first_tensor.size() > 0) {
//                 std::cout << "Initial first weight value: " << data[0] << std::endl;
//             }
//         }
//     }

//     // Train with MeZO
//     try {
//         std::cout << "Starting MeZO training..." << std::endl;
//         auto stats = nn->trainMeZO({
//             "epochs=5",
//             "batch_size=1",
//             "learning_rate=0.1",
//             "mezo_epsilon=0.01"
//         });
//         std::cout << "MeZO training completed successfully!" << std::endl;
//         std::cout << "Training stats: epochs=" << stats.max_epoch << std::endl;
//     } catch (const std::exception &e) {
//         std::cerr << "Error during MeZO training: " << e.what() << std::endl;
//         return 1;
//     }

//     // Get parameter pointers after training
//     params = nn->getParameterPointers();
//     std::cout << "Number of weight tensors after training: " << params.size() << std::endl;

//     // Print weights after training
//     if (!params.empty()) {
//         const auto& first_tensor = *params[0];
//         if (first_tensor.getDataType() == nntrainer::Tdatatype::FP32) {
//             float* data = reinterpret_cast<float*>(first_tensor.getData());
//             if (first_tensor.size() > 0) {
//                 std::cout << "Final first weight value: " << data[0] << std::endl;
//             }
//         }
//     }

//     std::cout << "trainMeZO demonstration completed!" << std::endl;

//     return 0;
// }
// #include <iostream>
// #include <vector>
// #include <memory>
// #include <layer.h>
// #include <model.h>
// #include <neuralnet.h>
// #include <nntrainer-api-common.h>
// #include <optimizer.h>
// #include <tensor.h>
// #include <util_func.h>

// /**
//  * Function to print tensor information
//  */
// void printTensorInfo(const nntrainer::Tensor& tensor, size_t index) {
//     std::cout << "Tensor " << index << ":" << std::endl;
//     std::cout << "  Shape: [";
//     auto dim = tensor.getDim();
//     for (unsigned int i = 0; i < 4; ++i) {
//         std::cout << dim[i];
//         if (i < 3) std::cout << ", ";
//     }
//     std::cout << "]" << std::endl;
    
//     std::cout << "  Data type: " << static_cast<int>(tensor.getDataType()) << std::endl;
//     std::cout << "  Number of elements: " << tensor.size() << std::endl;
//     std::cout << "  Memory size (bytes): " << tensor.getDim().getDataLen() * sizeof(float) << std::endl;
//     std::cout << "  Batch size: " << dim.batch() << std::endl;
//     std::cout << "  Channel size: " << dim.channel() << std::endl;
//     std::cout << "  Height size: " << dim.height() << std::endl;
//     std::cout << "  Width size: " << dim.width() << std::endl;
// }

// /**
//  * Function to demonstrate how to determine the size of tensor data
//  */
// void demonstrateTensorDataSize(const nntrainer::Tensor& tensor, size_t index) {
//     std::cout << "\n=== Tensor " << index << " Data Size Analysis ===" << std::endl;
    
//     // Method 1: Using tensor.size() - number of elements
//     size_t num_elements = tensor.size();
//     std::cout << "Number of elements: " << num_elements << std::endl;
    
//     // Method 2: Using tensor dimensions
//     auto dim = tensor.getDim();
//     size_t calculated_elements = dim.batch() * dim.channel() * dim.height() * dim.width();
//     std::cout << "Calculated elements: " << calculated_elements << std::endl;
    
//     // Method 3: Memory size calculation
//     size_t element_size = sizeof(float);  // Assuming FP32
//     size_t total_memory_bytes = num_elements * element_size;
//     std::cout << "Element size: " << element_size << " bytes" << std::endl;
//     std::cout << "Total memory: " << total_memory_bytes << " bytes" << std::endl;
    
//     // Method 4: Pointer size vs data size
//     void* data_ptr = tensor.getData();
//     std::cout << "Pointer size: " << sizeof(data_ptr) << " bytes (this is just the pointer!)" << std::endl;
//     std::cout << "Actual data size: " << total_memory_bytes << " bytes" << std::endl;
// }

// /**
//  * Function to print first few values of a tensor
//  */
// void printTensorValues(const nntrainer::Tensor& tensor, size_t index, size_t max_values = 10) {
//     std::cout << "\n=== Tensor " << index << " Values (first " << max_values << ") ===" << std::endl;
    
//     if (tensor.getDataType() == nntrainer::TensorDim::DataType::FP32) {
//         const float* data = tensor.getData<float>();
//         size_t num_elements = std::min(tensor.size(), max_values);
        
//         std::cout << "Values: ";
//         for (size_t i = 0; i < num_elements; ++i) {
//             std::cout << data[i] << " ";
//         }
//         std::cout << std::endl;
//     } else {
//         std::cout << "Unsupported data type for printing values" << std::endl;
//     }
// }

// /**
//  * Function to create input tensors for inference
//  */
// std::vector<nntrainer::Tensor> createInputTensors(ml::train::Model* model) {
//     std::cout << "\n=== Creating Input Tensors ===" << std::endl;
    
//     // Get model input dimensions
//     auto input_dims = model->getInputDimension();
//     std::vector<nntrainer::Tensor> inputs;
    
//     std::cout << "Number of input tensors required: " << input_dims.size() << std::endl;
    
//     for (size_t i = 0; i < input_dims.size(); ++i) {
//         std::cout << "Input " << i << " dimensions: [";
//         for (unsigned int j = 0; j < input_dims[i].getDim(); ++j) {
//             std::cout << input_dims[i][j];
//             if (j < input_dims[i].getDim() - 1) std::cout << ", ";
//         }
//         std::cout << "]" << std::endl;
        
//         // Create tensor with the required dimensions
//         nntrainer::Tensor input_tensor(input_dims[i]);
        
//         // Initialize with random values or zeros
//         // For demonstration, we'll initialize with some pattern
//         if (input_tensor.getDataType() == nntrainer::TensorDim::DataType::FP32) {
//             float* data = input_tensor.getData<float>();
//             size_t num_elements = input_tensor.size();
            
//             for (size_t j = 0; j < num_elements; ++j) {
//                 data[j] = static_cast<float>(j % 10) / 10.0f;  // Values between 0.0 and 0.9
//             }
//         }
        
//         inputs.push_back(input_tensor);
//         std::cout << "Created input tensor " << i << " with " << input_tensor.size() << " elements" << std::endl;
//     }
    
//     return inputs;
// }

// /**
//  * Function to run model inference
//  */
// std::vector<nntrainer::Tensor> runInference(ml::train::Model* model, 
//                                           const std::vector<nntrainer::Tensor>& inputs) {
//     std::cout << "\n=== Running Model Inference ===" << std::endl;
    
//     try {
//         // Run inference
//         auto outputs = model->inference(inputs);
        
//         std::cout << "Inference completed successfully!" << std::endl;
//         std::cout << "Number of output tensors: " << outputs.size() << std::endl;
        
//         // Print output information
//         for (size_t i = 0; i < outputs.size(); ++i) {
//             std::cout << "Output " << i << " shape: [";
//             auto dim = outputs[i]->getDim();
//             for (unsigned int j = 0; j < dim.getDim(); ++j) {
//                 std::cout << dim[j];
//                 if (j < dim.getDim() - 1) std::cout << ", ";
//             }
//             std::cout << "]" << std::endl;
//             std::cout << "  Number of elements: " << outputs[i]->size() << std::endl;
//         }
        
//         return outputs;
//     } catch (const std::exception& e) {
//         std::cerr << "Error during inference: " << e.what() << std::endl;
//         throw;
//     }
// }

// int main() {
//     // Hardcoded ONNX model path
//     std::string onnx_path = "/sachin/personalization/nntrainer/Applications/ONNX/jni/simple_model.onnx";  // Replace with actual path
//     std::string weight_path =
//     "/sachin/personalization/nntrainer/Applications/ONNX/python/qwen3/multi-token/bins/";

//     // Create model
//     auto model = ml::train::createModel();

//     try {
//         // Load ONNX model
//         model->load(onnx_path, ml::train::ModelFormat::MODEL_FORMAT_ONNX);
//         std::cout << "ONNX model loaded successfully from: " << onnx_path << std::endl;
//     } catch (const std::exception &e) {
//         std::cerr << "Error loading ONNX model: " << e.what() << std::endl;
//         return 1;
//     }

//     try {
//         // Compile the model
//         model->compile();
//         std::cout << "Model compiled successfully." << std::endl;
//     } catch (const std::exception &e) {
//         std::cerr << "Error compiling model: " << e.what() << std::endl;
//         return 1;
//     }

//     try {
//         // Initialize the model
//         model->initialize();
//         std::cout << "Model initialized successfully." << std::endl;
//     } catch (const std::exception &e) {
//         std::cerr << "Error initializing model: " << e.what() << std::endl;
//         return 1;
//     }

//     model->summarize(std::cout, ML_TRAIN_SUMMARY_MODEL);

//     std::cout << "--------------------------------------Summarize Model "
//                "Done--------------------------------------"
//             << std::endl;
            

//     // Cast to NeuralNetwork to access getParameterPointers
//     auto nn = static_cast<nntrainer::NeuralNetwork*>(model.get());

//     // Get parameter pointers
//     std::vector<nntrainer::Tensor*> params = nn->getParameterPointers();

//     std::cout << "\nNumber of weight tensors: " << params.size() << std::endl;

//     // Analyze each tensor
//     for (size_t i = 0; i < params.size(); ++i) {
//         const auto& tensor = *params[i];
        
//         // Print tensor information
//         printTensorInfo(tensor, i);
        
//         // Demonstrate data size determination
//         demonstrateTensorDataSize(tensor, i);
        
//         // Print some values
//         printTensorValues(tensor, i, 5);
        
//         std::cout << std::endl;
//     }

//     // Demonstrate tensor data manipulation
//     std::cout << "\n=== Tensor Data Manipulation ===" << std::endl;
//     for (size_t i = 0; i < std::min(params.size(), size_t(2)); ++i) {
//         auto& tensor = *params[i];
        
//         if (tensor.getDataType() == nntrainer::TensorDim::DataType::FP32 && tensor.size() > 0) {
//             std::cout << "\nManipulating Tensor " << i << ":" << std::endl;
            
//             float* data = tensor.getData<float>();
//             size_t num_elements = tensor.size();
            
//             std::cout << "Original first value: " << data[0] << std::endl;
//             std::cout << "Original last value: " << data[num_elements - 1] << std::endl;
            
//             // Modify some values (be careful with trained models!)
//             // data[0] += 0.1f;  // Modify first element
//             // data[num_elements - 1] -= 0.1f;  // Modify last element
            
//             std::cout << "Modified first value: " << data[0] << std::endl;
//             std::cout << "Modified last value: " << data[num_elements - 1] << std::endl;
//         }
//     }

//     // Demonstrate key concepts about pointer vs data size
//     std::cout << "\n=== Key Concepts: Pointer vs Data Size ===" << std::endl;
//     if (!params.empty()) {
//         auto& first_tensor = *params[0];
//         void* data_ptr = first_tensor.getData();
        
//         std::cout << "Pointer to tensor data: " << data_ptr << std::endl;
//         std::cout << "Size of pointer itself: " << sizeof(data_ptr) << " bytes" << std::endl;
//         std::cout << "Size of actual tensor data: " << first_tensor.size() * sizeof(float) << " bytes" << std::endl;
//         std::cout << "Number of elements in tensor: " << first_tensor.size() << std::endl;
//         std::cout << "Size of each element: " << sizeof(float) << " bytes" << std::endl;
//     }

//     std::cout << "\nDone playing with tensors!" << std::endl;

//     // Run model inference demonstration
//     std::cout << "\n=== Model Inference Demonstration ===" << std::endl;
    
//     try {
//         // Create input tensors for inference
//         auto input_tensors = createInputTensors(model.get());
        
//         // Run inference
//         auto output_tensors = runInference(model.get(), input_tensors);
        
//         // Print some output values
//         std::cout << "\n=== Inference Results ===" << std::endl;
//         for (size_t i = 0; i < output_tensors.size(); ++i) {
//             std::cout << "Output tensor " << i << " first 5 values: ";
//             if (output_tensors[i]->getDataType() == nntrainer::TensorDim::DataType::FP32) {
//                 const float* data = output_tensors[i]->getData<float>();
//                 size_t num_elements = std::min(output_tensors[i]->size(), size_t(5));
//                 for (size_t j = 0; j < num_elements; ++j) {
//                     std::cout << data[j] << " ";
//                 }
//                 std::cout << std::endl;
//             }
//         }
        
//     } catch (const std::exception& e) {
//         std::cerr << "Error during inference demonstration: " << e.what() << std::endl;
//     }

//     std::cout << "\n=== Complete Demonstration Finished ===" << std::endl;
//     return 0;
// }
