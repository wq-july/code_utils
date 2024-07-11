#include "tensorRT/generic.h"

namespace TensorRT {

static auto cuda_stream = TensorRT::MakeCudaStream();

GenericInference::GenericInference(const TensorRTConfig::Config& config) : config_(config) {}

bool GenericInference::LoadTRTEngine(const std::string& engin_path) {
  LOG(INFO) << "Loading TensorRT engine...";
  std::ifstream file(engin_path.c_str(), std::ios::binary);
  if (file.is_open()) {
    file.seekg(0, std::ifstream::end);
    size_t size = file.tellg();
    file.seekg(0, std::ifstream::beg);
    char* model_stream = new char[size];
    file.read(model_stream, size);
    file.close();
    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(gLogger);
    if (runtime == nullptr) {
      LOG(ERROR) << "Runtime creat failed!";
      delete[] model_stream;
      return false;
    }
    engine_ =
        std::shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(model_stream, size));
    if (engine_ == nullptr) {
      LOG(ERROR) << "TensorRT engine creat failed!";
      delete[] model_stream;
      return false;
    }
    delete[] model_stream;
    return true;
  }
  LOG(WARNING) << "TensorRT engine load failed at: " << engin_path;
  return false;
}

bool GenericInference::LoadTRTONNXEngine(const std::string& onnx_path) {
  LOG(INFO) << "Loading TensorRT ONNX...";
  // ========== 1. 创建builder：创建优化的执行引擎（ICudaEngine）的关键工具 ==========
  /*
  创建 IBuilder 对象，并使用 std::unique_ptr 管理其内存
  IBuilder 是 TensorRT 中的一个接口，负责创建一个优化的 ICudaEngine。
  ICudaEngine 是一个优化后的可执行网络，可以用于实际执行推理任务。
  几乎在所有使用 TensorRT
  的场合都会使用到IBuilder，因为它是创建优化执行引擎（ICudaEngine）的关键工具。
  不论你的模型是什么，只要你想用 TensorRT 来进行优化和部署，都需要创建和使用 IBuilder。
  因此，创建 IBuilder 是必要的步骤，因为它是创建 ICudaEngine 的关键工具。
  在创建时需要提供一个日志对象以接收错误、警告和其他信息。
  这个步骤是整个模型创建流程的起点，没有 IBuilder，我们就无法创建网络（模型）。
  */
  auto builder =
      TensorRTUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(gLogger.getTRTLogger()));
  CHECK(builder != nullptr) << "Create infer builder failed !";
  // ========== 2. 创建network：builder--->network ==========
  // 设置batch, 数据输入的批次量大小,显性设置batch
  /*
    创建 network 是因为我们需要一个 INetworkDefinition 对象来描述我们的模型。我们使用 builder 的
      createNetworkV2
     方法来创建一个网络，并为其提供一组标志来指定网络的行为。在这个例子中，我们设置了
    kEXPLICIT_BATCH标志，以启用显式批处理。Batch通常在你需要进行大量数据的推理时会用到。在训练和推理阶段，我们通常不会一次只处理一个样本，
    而是会一次处理一批样本，这样可以更有效地利用计算资源，特别是在GPU上进行计算时。同时，选择合适的Batch大小也是一个需要平衡的问题，
    一方面，较大的Batch大小可以更好地利用硬件并行性，提高计算效率，另一方面，较大的Batch大小会消耗更多的内存资源，所以需要根据具体情况进行选择。
  */
  const auto explicit_batch =
      1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
  auto network =
      TensorRTUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicit_batch));
  CHECK(network != nullptr) << "Create network failed !";

  // ========== 3. 创建config配置：builder--->config ==========
  // 配置解析器
  // 在这个步骤中，我们创建一个 IBuilderConfig
  // 对象，用于保存模型优化过程中的各种设置。我们向其添加我们之前创建的优化配置文件，并设置一些标志，如
  // kFP16，来启用对 FP16 精度的支持。我们还设置了最大批处理大小，这是优化过程的一部分。
  // 在 TensorRT 中，配置文件（IBuilderConfig对象）用于保存模型优化过程中的各种设置。这些设置包括：
  // 优化精度模式：你可以选择 FP32、FP16 或 INT8 精度模式。使用更低的精度（如 FP16 或
  // INT8）可以减少计算资源的使用，从而提高推理速度，但可能会牺牲一些推理精度。
  // 最大批处理大小：这是模型优化过程中可以处理的最大输入批处理数量。
  // 工作空间大小：这是 TensorRT 在优化和执行模型时可以使用的最大内存数量。
  // 优化配置文件：优化配置文件描述了模型输入尺寸的可能范围。根据这些信息，TensorRT
  // 可以创建一个针对各种输入尺寸优化的模型。 配置文件还可以包含其他的设置，例如 GPU
  // 设备选择、层策略选择等等。这些设置都会影响 TensorRT 如何优化模型，以及优化后的模型的性能。
  auto config = TensorRTUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
  CHECK(config != nullptr) << "Create builder config failed !";

  auto parser = TensorRTUniquePtr<nvonnxparser::IParser>(
      nvonnxparser::createParser(*network, gLogger.getTRTLogger()));
  CHECK(parser != nullptr) << "Create parser failed !";

  auto profile = builder->createOptimizationProfile();
  CHECK(profile != nullptr) << "Create optimization profile failed !";
  SetIBuilderConfigProfile(config.get(), profile);

  auto constructed = ConstructNetwork(builder, network, config, parser);
  CHECK(constructed) << "Construct network failed !";

  // 创建 CUDA 流并设置 config 的 profile stream
  // 这里我们创建一个 CUDA 流，用于异步执行 GPU 操作。然后我们把这个 CUDA 流设置为配置对象的profile
  // stream。Profile stream 用于确定何时可以开始收集内核时间，以及何时可以读取收集到的时间。
  // 这里的 profileStream 是 CUDA 流（CUDA Stream），而不是前面的优化配置文件（optimization
  // profile）。CUDA 流是一个有序的操作队列，它们在 GPU
  // 上异步执行。流可以用于组织和控制执行的并发性和依赖关系。
  // 在你的代码中，profileStream 是通过 samplesCommon::makeCudaStream()
  // 创建的。这个流会被传递给配置对象 config，然后 TensorRT
  // 会在这个流中执行和优化配置文件（profile）相关的操作。这允许这些操作并发执行，从而提高了执行效率。
  // 需要注意的是，这里的 CUDA
  // 流和前面的优化配置文件是两个完全不同的概念。优化配置文件包含了模型输入尺寸的信息，
  // 用于指导模型的优化过程；而CUDA 流则是用于管理 GPU
  // 操作的执行顺序，以提高执行效率。两者虽然名字类似，但实际上在功能和用途上是完全不同的。

  // auto cuda_stream = TensorRT::MakeCudaStream();
  // CHECK(cuda_stream != nullptr) << "Cuda stream nullptr !";

  config->setProfileStream(*cuda_stream);
  CHECK(config->getProfileStream()) << "Cuda stream set failed !";

  // ========== 5. 序列化保存engine ==========
  // 使用之前创建并配置的 builder、network 和 config 对象来构建并序列化一个优化过的模型。
  TensorRTUniquePtr<nvinfer1::IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};
  CHECK(plan != nullptr) << "Create IHostMemory plan failed !";

  TensorRTUniquePtr<nvinfer1::IRuntime> runtime{
      nvinfer1::createInferRuntime(gLogger.getTRTLogger())};
  CHECK(runtime != nullptr) << "Create IRuntime failed !";
  engine_ = std::shared_ptr<nvinfer1::ICudaEngine>(
      runtime->deserializeCudaEngine(plan->data(), plan->size()));
  CHECK(engine_ != nullptr) << "Create engine failed !";
  CHECK(SaveEngine()) << "Save engine failed !";
  CHECK(VerifyEngine(network.get())) << "VerifyEngine failed !";
  LOG(INFO) << "Build Engine Success !";
  return true;
}

void GenericInference::SetIBuilderConfigProfile(nvinfer1::IBuilderConfig* const config,
                                                nvinfer1::IOptimizationProfile* const profile) {
  CHECK(config != nullptr) << "IBuilderConfig is nullptr!";
  CHECK(profile != nullptr) << "IOptimizationProfile is nullptr!";
  // 对于SuperPoint和SuperGlue会进行不同的设置， 虚函数，手动去修改, 维度不同
  // 配置网络参数
  // 我们需要告诉tensorrt我们最终运行时，输入图像的范围，batch
  // size的范围。这样tensorrt才能对应为我们进行模型构建与优化。
  // nvinfer1::ITensor* input = network->getInput(0);  // 获取了网络的第一个输入节点。
  // 网络的输入节点就是模型的输入层，它接收模型的输入数据。
  // 在 TensorRT 中，优化配置文件（Optimization Profile）用于描述模型的输入尺寸和动态尺寸范围。
  // 通过优化配置文件，可以告诉 TensorRT
  // 输入数据的可能尺寸范围，使其可以创建一个适应各种输入尺寸的优化后的模型。
  // 设置最小尺寸
  // profile->setDimensions(
  //     input->getName(), nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4(1, 3, 640, 640));
  // 设置最优尺寸
  // profile->setDimensions(
  //     input->getName(), nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4(1, 3, 640, 640));
  // 设置最大尺寸
  // profile->setDimensions(
  //     input->getName(), nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4(1, 3, 640, 640));
  //  example, SuperPoint的配置
  LOG(WARNING) << "Loading default profile !";

  profile->setDimensions(config_.input_tensor_names()[0].c_str(),
                         nvinfer1::OptProfileSelector::kMIN,
                         nvinfer1::Dims4(1, 1, 100, 100));
  profile->setDimensions(config_.input_tensor_names()[0].c_str(),
                         nvinfer1::OptProfileSelector::kOPT,
                         nvinfer1::Dims4(1, 1, 500, 500));
  profile->setDimensions(config_.input_tensor_names()[0].c_str(),
                         nvinfer1::OptProfileSelector::kMAX,
                         nvinfer1::Dims4(1, 1, 1500, 1500));
  // config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1 << 30);
  // config->setAvgTimingIterations(1);
  config->addOptimizationProfile(profile);
}

bool GenericInference::SaveEngine() {
  if (config_.engine_file().empty()) {
    LOG(ERROR) << "Empty engine file path!";
    return false;
  }
  if (engine_ != nullptr) {
    std::unique_ptr<nvinfer1::IHostMemory> serialized_engine{engine_->serialize()};
    if (!serialized_engine) {
      LOG(FATAL) << "Serializ engine failed!";
      return false;
    }
    std::ofstream engine_file(config_.engine_file().c_str(), std::ios::binary);
    if (!engine_file) {
      LOG(ERROR) << "Cannot open engine file: " << config_.engine_file();
      return false;
    }
    engine_file.write(static_cast<char*>(serialized_engine->data()), serialized_engine->size());
    return !engine_file.fail();
  }
  return false;
}

bool GenericInference::VerifyEngine(const nvinfer1::INetworkDefinition* network) {
  CHECK(engine_->getNbIOTensors() ==
        config_.input_tensor_names().size() + config_.output_tensor_names().size())
      << "Error tensors nums !";
  return true;
}

bool GenericInference::Infer() {
  LOG(INFO) << "Starting Infer ! ";
  // 在TensorRT中，ICudaEngine对象代表了优化后的网络，而IExecutionContext则封装了执行推理所需要的上下文信息，
  // 比如输入/输出的内存、CUDA流等。每个IExecutionContext都与一个特定的ICudaEngine相关联，
  // 并且在执行推理时会使用ICudaEngine中的模型。
  // 创建IExecutionContext的过程是通过调用ICudaEngine的createExecutionContext()方法完成的。
  // 在 TensorRT 中，推理的执行是通过执行上下文 (ExecutionContext)
  // 来进行的.ExecutionContext封装了推理运行时所需要的所有信息，包括存储设备上的缓冲区地址、内核参数以及其他设备
  // 信息。因此，当你想在设备上执行模型推理时必须要有一个ExecutionContext。
  // 每一个ExecutionContext是和一个特定的ICudaEngine(优化后的网络)
  // 相关联的，它包含了网络推理的所有运行时信息。因此，没有执行上下文，就无法进行模型推理。
  // 此外，每个ExecutionContext也和一个特定的CUDA流关联，它允许在同一个流中并行地执行多个模型推理，
  // 这使得能够在多个设备或多个线程之间高效地切换。
  // 深化理解ICudaEngine
  // ICudaEngine 是 NVIDIA TensorRT 库中的一个关键接口，它提供了在 GPU
  // 上执行推断的所有必要的方法。每个 ICudaEngine 对象都表示了一个已经优化过的神经网络模型。
  // 以下是一些你可以使用 ICudaEngine 完成的操作：
  // 1. 创建执行上下文：使用 ICudaEngine 对象，你可以创建一个或多个执行上下文
  // (IExecutionContext)，每个上下文在给定的引擎中定义了网络的一次执行。
  // 这对于并发执行或使用不同的批量大小进行推断很有用。
  // 2. 获取网络层信息：ICudaEngine
  // 提供了方法来查询网络的输入和输出张量以及网络层的相关信息，例如张量的尺寸、数据类型等。
  // 3. 序列化和反序列化：你可以将 ICudaEngine
  // 对象序列化为一个字符串，然后将这个字符串保存到磁盘，以便以后使用。这对于保存优化后的模型并在以后的会话中
  // 重新加载它们很有用。相对应地，你也可以使用deserializeCudaEngine() 方法从字符串或磁盘文件中恢复
  // ICudaEngine 对象。
  // 通常，ICudaEngine用于以下目的：
  // 执行推断：最主要的应用就是使用 ICudaEngine
  // 执行推断。你可以创建一个执行上下文，并使用它将输入数据提供给模型，然后从模型中获取输出结果。
  // 加速模型加载：通过序列化 ICudaEngine
  // 对象并将它们保存到磁盘，你可以在以后的会话中快速加载优化后的模型，无需重新进行优化。
  // 管理资源：在并发执行或使用不同的批量大小进行推断时，你可以创建多个执行上下文以管理 GPU 资源。
  // 总的来说，ICudaEngine 是 TensorRT 中最重要的接口之一，它提供了一种高效、灵活的方式来在 GPU
  // 上执行推断。
  auto context = TensorRTUniquePtr<nvinfer1::IExecutionContext>(engine_->createExecutionContext());
  CHECK(context != nullptr) << "Creat context failed!";

  SetContext(context.get());

  // 创建输入输出缓冲区：输入输出缓冲区在深度学习推理中起着桥梁的作用，它们连接着原始输入数据和模型处理结果，
  // 使得我们可以有效地执行模型推理，并获取推理结果。
  // 输入输出缓冲区在深度学习推理中发挥着至关重要的作用，它们充当了处理过程中数据的"桥梁"，
  // 使得我们能够将输入数据顺利地送入模型，并从模型中获取处理结果。这大大简化了深度学习推理的过程，
  // 使得我们可以更专注于实现模型的逻辑，而不需要过多地关心数据的传输和存储问题。
  // TensorRT 提供的 BufferManager
  // 类用于简化这个过程，可以自动创建并管理这些缓冲区，使得在进行推理时不需要手动创建和管理这些缓冲区。
  // TensorRT中的BufferManager类是一个辅助类，它的主要目的是简化输入/输出缓冲区的创建和管理。在进行模型推理时，
  // 不需要手动管理这些缓冲区，只需要将输入数据放入BufferManager管理的输入缓冲区中，然后调用推理函数。
  // 待推理完成后，可以从BufferManager管理的输出缓冲区中获取模型的推理结果。
  // --------------------------------------------------------------------
  /*
  输入缓冲区：
  输入缓冲区主要用于存储需要进行处理的数据。在深度学习推理中，输入缓冲区通常用于存储模型的输入数据。
  比如，如果你的模型是一个图像识别模型，那么输入缓冲区可能会存储待识别的图像数据。当执行模型推理时，模型会从输入缓冲区中读取数据进行处理。

  输出缓冲区：
  输出缓冲区主要用于存储处理过的数据，即处理结果。在深度学习推理中，输出缓冲区通常用于存储模型的输出结果。继续上述图像识别模型的例子，
  一旦模型完成了图像的识别处理，识别结果（例如，图像中物体的类别）就会存储在输出缓冲区中。我们可以从输出缓冲区中获取这些结果，进行进一步的处理或分析。

  总的来说，输入输出缓冲区在深度学习推理中起着桥梁的作用，它们连接着原始输入数据和模型处理结果，使得我们可以有效地执行模型推理，并获取推理结果。
  */
  // --------------------------------------------------------------------
  TensorRT::BufferManager buffers(engine_, 0, context.get());
  // TensorRT::BufferManager buffers(engine_);

  // ? 是否要进行这一步的设置呢？
  // for (int32_t i = 0, e = engine_->getNbIOTensors(); i < e; i++) {
  //   auto const name = engine_->getIOTensorName(i);
  //   context->setTensorAddress(name, buffers.GetDeviceBuffer(name));
  // }

  // Read the input data into the managed buffers
  CHECK(ProcessInput(buffers)) << "Process input failed !";
  // Memcpy from host input buffers to device input buffers
  buffers.CopyInputToDevice();
  CHECK(context->executeV2(buffers.GetDeviceBindings().data())) << "Execute buffers failed !";
  // Memcpy from device output buffers to host output buffers
  buffers.CopyOutputToHost();
  // Verify results
  CHECK(ProcessOutput(buffers)) << "Process output failed !";
  LOG(INFO) << "Infer Done !";
  return true;
}

bool GenericInference::Build() {
  if (!LoadTRTEngine(config_.engine_file())) {
    LOG(WARNING) << "Load local engine failed, starting creating engine...";
  } else {
    LOG(INFO) << "Load local engine success !";
    return true;
  }

  if (!LoadTRTONNXEngine(config_.onnx_file())) {
    LOG(ERROR) << "Creating engine failed !";
    return false;
  }
  return true;
}

bool GenericInference::ConstructNetwork(TensorRTUniquePtr<nvinfer1::IBuilder>& builder,
                                        TensorRTUniquePtr<nvinfer1::INetworkDefinition>& network,
                                        TensorRTUniquePtr<nvinfer1::IBuilderConfig>& config,
                                        TensorRTUniquePtr<nvonnxparser::IParser>& parser) const {
  auto parsed = parser->parseFromFile(config_.onnx_file().c_str(),
                                      static_cast<int>(gLogger.getReportableSeverity()));
  CHECK(parsed) << "Parse onnx file failed!";
  if (config_.fp16()) {
    LOG(INFO) << "Set kFP16";
    config->setFlag(nvinfer1::BuilderFlag::kFP16);
  }
  if (config_.bf16()) {
    LOG(INFO) << "Set kBF16";
    config->setFlag(nvinfer1::BuilderFlag::kBF16);
  }
  if (config_.int8()) {
    LOG(INFO) << "Set kINT8";
    config->setFlag(nvinfer1::BuilderFlag::kINT8);
    TensorRT::SetAllDynamicRanges(network.get(), 127.0F, 127.0F);
  }
  TensorRT::EnableDLA(builder.get(), config.get(), config_.dla_core());
  return true;
}

bool GenericInference::ProcessInput(const TensorRT::BufferManager& buffers) {
  LOG(WARNING) << "需要重载这个函数，来处理你的输入！";
  return true;
}

bool GenericInference::ProcessOutput(const TensorRT::BufferManager& buffers) {
  LOG(WARNING) << "需要重载这个函数，来处理你的输出结果！";
  return true;
}

void GenericInference::SetContext(nvinfer1::IExecutionContext* const context) {
  LOG(WARNING) << "需要重载这个函数，来处理你的context！";
}

}  // namespace TensorRT