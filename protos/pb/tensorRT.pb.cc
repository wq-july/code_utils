// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: tensorRT.proto

#include "tensorRT.pb.h"

#include <algorithm>

#include <google/protobuf/stubs/common.h>
#include <google/protobuf/stubs/port.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/wire_format_lite_inl.h>
#include <google/protobuf/descriptor.h>
#include <google/protobuf/generated_message_reflection.h>
#include <google/protobuf/reflection_ops.h>
#include <google/protobuf/wire_format.h>
// This is a temporary google only hack
#ifdef GOOGLE_PROTOBUF_ENFORCE_UNIQUENESS
#include "third_party/protobuf/version.h"
#endif
// @@protoc_insertion_point(includes)

namespace TensorRTConfig {
class ConfigDefaultTypeInternal {
 public:
  ::google::protobuf::internal::ExplicitlyConstructed<Config>
      _instance;
} _Config_default_instance_;
}  // namespace TensorRTConfig
namespace protobuf_tensorRT_2eproto {
static void InitDefaultsConfig() {
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  {
    void* ptr = &::TensorRTConfig::_Config_default_instance_;
    new (ptr) ::TensorRTConfig::Config();
    ::google::protobuf::internal::OnShutdownDestroyMessage(ptr);
  }
  ::TensorRTConfig::Config::InitAsDefaultInstance();
}

::google::protobuf::internal::SCCInfo<0> scc_info_Config =
    {{ATOMIC_VAR_INIT(::google::protobuf::internal::SCCInfoBase::kUninitialized), 0, InitDefaultsConfig}, {}};

void InitDefaults() {
  ::google::protobuf::internal::InitSCC(&scc_info_Config.base);
}

::google::protobuf::Metadata file_level_metadata[1];

const ::google::protobuf::uint32 TableStruct::offsets[] GOOGLE_PROTOBUF_ATTRIBUTE_SECTION_VARIABLE(protodesc_cold) = {
  ~0u,  // no _has_bits_
  GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(::TensorRTConfig::Config, _internal_metadata_),
  ~0u,  // no _extensions_
  ~0u,  // no _oneof_case_
  ~0u,  // no _weak_field_map_
  GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(::TensorRTConfig::Config, input_tensor_names_),
  GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(::TensorRTConfig::Config, output_tensor_names_),
  GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(::TensorRTConfig::Config, onnx_file_),
  GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(::TensorRTConfig::Config, engine_file_),
  GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(::TensorRTConfig::Config, dla_core_),
  GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(::TensorRTConfig::Config, batch_size_),
  GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(::TensorRTConfig::Config, int8_),
  GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(::TensorRTConfig::Config, fp16_),
  GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(::TensorRTConfig::Config, bf16_),
};
static const ::google::protobuf::internal::MigrationSchema schemas[] GOOGLE_PROTOBUF_ATTRIBUTE_SECTION_VARIABLE(protodesc_cold) = {
  { 0, -1, sizeof(::TensorRTConfig::Config)},
};

static ::google::protobuf::Message const * const file_default_instances[] = {
  reinterpret_cast<const ::google::protobuf::Message*>(&::TensorRTConfig::_Config_default_instance_),
};

void protobuf_AssignDescriptors() {
  AddDescriptors();
  AssignDescriptors(
      "tensorRT.proto", schemas, file_default_instances, TableStruct::offsets,
      file_level_metadata, NULL, NULL);
}

void protobuf_AssignDescriptorsOnce() {
  static ::google::protobuf::internal::once_flag once;
  ::google::protobuf::internal::call_once(once, protobuf_AssignDescriptors);
}

void protobuf_RegisterTypes(const ::std::string&) GOOGLE_PROTOBUF_ATTRIBUTE_COLD;
void protobuf_RegisterTypes(const ::std::string&) {
  protobuf_AssignDescriptorsOnce();
  ::google::protobuf::internal::RegisterAllTypes(file_level_metadata, 1);
}

void AddDescriptorsImpl() {
  InitDefaults();
  static const char descriptor[] GOOGLE_PROTOBUF_ATTRIBUTE_SECTION_VARIABLE(protodesc_cold) = {
      "\n\016tensorRT.proto\022\016TensorRTConfig\"\271\001\n\006Con"
      "fig\022\032\n\022input_tensor_names\030\001 \003(\t\022\033\n\023outpu"
      "t_tensor_names\030\002 \003(\t\022\021\n\tonnx_file\030\003 \001(\t\022"
      "\023\n\013engine_file\030\004 \001(\t\022\020\n\010dla_core\030\005 \001(\005\022\022"
      "\n\nbatch_size\030\006 \001(\005\022\014\n\004int8\030\007 \001(\010\022\014\n\004fp16"
      "\030\010 \001(\010\022\014\n\004bf16\030\t \001(\010b\006proto3"
  };
  ::google::protobuf::DescriptorPool::InternalAddGeneratedFile(
      descriptor, 228);
  ::google::protobuf::MessageFactory::InternalRegisterGeneratedFile(
    "tensorRT.proto", &protobuf_RegisterTypes);
}

void AddDescriptors() {
  static ::google::protobuf::internal::once_flag once;
  ::google::protobuf::internal::call_once(once, AddDescriptorsImpl);
}
// Force AddDescriptors() to be called at dynamic initialization time.
struct StaticDescriptorInitializer {
  StaticDescriptorInitializer() {
    AddDescriptors();
  }
} static_descriptor_initializer;
}  // namespace protobuf_tensorRT_2eproto
namespace TensorRTConfig {

// ===================================================================

void Config::InitAsDefaultInstance() {
}
#if !defined(_MSC_VER) || _MSC_VER >= 1900
const int Config::kInputTensorNamesFieldNumber;
const int Config::kOutputTensorNamesFieldNumber;
const int Config::kOnnxFileFieldNumber;
const int Config::kEngineFileFieldNumber;
const int Config::kDlaCoreFieldNumber;
const int Config::kBatchSizeFieldNumber;
const int Config::kInt8FieldNumber;
const int Config::kFp16FieldNumber;
const int Config::kBf16FieldNumber;
#endif  // !defined(_MSC_VER) || _MSC_VER >= 1900

Config::Config()
  : ::google::protobuf::Message(), _internal_metadata_(NULL) {
  ::google::protobuf::internal::InitSCC(
      &protobuf_tensorRT_2eproto::scc_info_Config.base);
  SharedCtor();
  // @@protoc_insertion_point(constructor:TensorRTConfig.Config)
}
Config::Config(const Config& from)
  : ::google::protobuf::Message(),
      _internal_metadata_(NULL),
      input_tensor_names_(from.input_tensor_names_),
      output_tensor_names_(from.output_tensor_names_) {
  _internal_metadata_.MergeFrom(from._internal_metadata_);
  onnx_file_.UnsafeSetDefault(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
  if (from.onnx_file().size() > 0) {
    onnx_file_.AssignWithDefault(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), from.onnx_file_);
  }
  engine_file_.UnsafeSetDefault(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
  if (from.engine_file().size() > 0) {
    engine_file_.AssignWithDefault(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), from.engine_file_);
  }
  ::memcpy(&dla_core_, &from.dla_core_,
    static_cast<size_t>(reinterpret_cast<char*>(&bf16_) -
    reinterpret_cast<char*>(&dla_core_)) + sizeof(bf16_));
  // @@protoc_insertion_point(copy_constructor:TensorRTConfig.Config)
}

void Config::SharedCtor() {
  onnx_file_.UnsafeSetDefault(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
  engine_file_.UnsafeSetDefault(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
  ::memset(&dla_core_, 0, static_cast<size_t>(
      reinterpret_cast<char*>(&bf16_) -
      reinterpret_cast<char*>(&dla_core_)) + sizeof(bf16_));
}

Config::~Config() {
  // @@protoc_insertion_point(destructor:TensorRTConfig.Config)
  SharedDtor();
}

void Config::SharedDtor() {
  onnx_file_.DestroyNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
  engine_file_.DestroyNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
}

void Config::SetCachedSize(int size) const {
  _cached_size_.Set(size);
}
const ::google::protobuf::Descriptor* Config::descriptor() {
  ::protobuf_tensorRT_2eproto::protobuf_AssignDescriptorsOnce();
  return ::protobuf_tensorRT_2eproto::file_level_metadata[kIndexInFileMessages].descriptor;
}

const Config& Config::default_instance() {
  ::google::protobuf::internal::InitSCC(&protobuf_tensorRT_2eproto::scc_info_Config.base);
  return *internal_default_instance();
}


void Config::Clear() {
// @@protoc_insertion_point(message_clear_start:TensorRTConfig.Config)
  ::google::protobuf::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  input_tensor_names_.Clear();
  output_tensor_names_.Clear();
  onnx_file_.ClearToEmptyNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
  engine_file_.ClearToEmptyNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
  ::memset(&dla_core_, 0, static_cast<size_t>(
      reinterpret_cast<char*>(&bf16_) -
      reinterpret_cast<char*>(&dla_core_)) + sizeof(bf16_));
  _internal_metadata_.Clear();
}

bool Config::MergePartialFromCodedStream(
    ::google::protobuf::io::CodedInputStream* input) {
#define DO_(EXPRESSION) if (!GOOGLE_PREDICT_TRUE(EXPRESSION)) goto failure
  ::google::protobuf::uint32 tag;
  // @@protoc_insertion_point(parse_start:TensorRTConfig.Config)
  for (;;) {
    ::std::pair<::google::protobuf::uint32, bool> p = input->ReadTagWithCutoffNoLastTag(127u);
    tag = p.first;
    if (!p.second) goto handle_unusual;
    switch (::google::protobuf::internal::WireFormatLite::GetTagFieldNumber(tag)) {
      // repeated string input_tensor_names = 1;
      case 1: {
        if (static_cast< ::google::protobuf::uint8>(tag) ==
            static_cast< ::google::protobuf::uint8>(10u /* 10 & 0xFF */)) {
          DO_(::google::protobuf::internal::WireFormatLite::ReadString(
                input, this->add_input_tensor_names()));
          DO_(::google::protobuf::internal::WireFormatLite::VerifyUtf8String(
            this->input_tensor_names(this->input_tensor_names_size() - 1).data(),
            static_cast<int>(this->input_tensor_names(this->input_tensor_names_size() - 1).length()),
            ::google::protobuf::internal::WireFormatLite::PARSE,
            "TensorRTConfig.Config.input_tensor_names"));
        } else {
          goto handle_unusual;
        }
        break;
      }

      // repeated string output_tensor_names = 2;
      case 2: {
        if (static_cast< ::google::protobuf::uint8>(tag) ==
            static_cast< ::google::protobuf::uint8>(18u /* 18 & 0xFF */)) {
          DO_(::google::protobuf::internal::WireFormatLite::ReadString(
                input, this->add_output_tensor_names()));
          DO_(::google::protobuf::internal::WireFormatLite::VerifyUtf8String(
            this->output_tensor_names(this->output_tensor_names_size() - 1).data(),
            static_cast<int>(this->output_tensor_names(this->output_tensor_names_size() - 1).length()),
            ::google::protobuf::internal::WireFormatLite::PARSE,
            "TensorRTConfig.Config.output_tensor_names"));
        } else {
          goto handle_unusual;
        }
        break;
      }

      // string onnx_file = 3;
      case 3: {
        if (static_cast< ::google::protobuf::uint8>(tag) ==
            static_cast< ::google::protobuf::uint8>(26u /* 26 & 0xFF */)) {
          DO_(::google::protobuf::internal::WireFormatLite::ReadString(
                input, this->mutable_onnx_file()));
          DO_(::google::protobuf::internal::WireFormatLite::VerifyUtf8String(
            this->onnx_file().data(), static_cast<int>(this->onnx_file().length()),
            ::google::protobuf::internal::WireFormatLite::PARSE,
            "TensorRTConfig.Config.onnx_file"));
        } else {
          goto handle_unusual;
        }
        break;
      }

      // string engine_file = 4;
      case 4: {
        if (static_cast< ::google::protobuf::uint8>(tag) ==
            static_cast< ::google::protobuf::uint8>(34u /* 34 & 0xFF */)) {
          DO_(::google::protobuf::internal::WireFormatLite::ReadString(
                input, this->mutable_engine_file()));
          DO_(::google::protobuf::internal::WireFormatLite::VerifyUtf8String(
            this->engine_file().data(), static_cast<int>(this->engine_file().length()),
            ::google::protobuf::internal::WireFormatLite::PARSE,
            "TensorRTConfig.Config.engine_file"));
        } else {
          goto handle_unusual;
        }
        break;
      }

      // int32 dla_core = 5;
      case 5: {
        if (static_cast< ::google::protobuf::uint8>(tag) ==
            static_cast< ::google::protobuf::uint8>(40u /* 40 & 0xFF */)) {

          DO_((::google::protobuf::internal::WireFormatLite::ReadPrimitive<
                   ::google::protobuf::int32, ::google::protobuf::internal::WireFormatLite::TYPE_INT32>(
                 input, &dla_core_)));
        } else {
          goto handle_unusual;
        }
        break;
      }

      // int32 batch_size = 6;
      case 6: {
        if (static_cast< ::google::protobuf::uint8>(tag) ==
            static_cast< ::google::protobuf::uint8>(48u /* 48 & 0xFF */)) {

          DO_((::google::protobuf::internal::WireFormatLite::ReadPrimitive<
                   ::google::protobuf::int32, ::google::protobuf::internal::WireFormatLite::TYPE_INT32>(
                 input, &batch_size_)));
        } else {
          goto handle_unusual;
        }
        break;
      }

      // bool int8 = 7;
      case 7: {
        if (static_cast< ::google::protobuf::uint8>(tag) ==
            static_cast< ::google::protobuf::uint8>(56u /* 56 & 0xFF */)) {

          DO_((::google::protobuf::internal::WireFormatLite::ReadPrimitive<
                   bool, ::google::protobuf::internal::WireFormatLite::TYPE_BOOL>(
                 input, &int8_)));
        } else {
          goto handle_unusual;
        }
        break;
      }

      // bool fp16 = 8;
      case 8: {
        if (static_cast< ::google::protobuf::uint8>(tag) ==
            static_cast< ::google::protobuf::uint8>(64u /* 64 & 0xFF */)) {

          DO_((::google::protobuf::internal::WireFormatLite::ReadPrimitive<
                   bool, ::google::protobuf::internal::WireFormatLite::TYPE_BOOL>(
                 input, &fp16_)));
        } else {
          goto handle_unusual;
        }
        break;
      }

      // bool bf16 = 9;
      case 9: {
        if (static_cast< ::google::protobuf::uint8>(tag) ==
            static_cast< ::google::protobuf::uint8>(72u /* 72 & 0xFF */)) {

          DO_((::google::protobuf::internal::WireFormatLite::ReadPrimitive<
                   bool, ::google::protobuf::internal::WireFormatLite::TYPE_BOOL>(
                 input, &bf16_)));
        } else {
          goto handle_unusual;
        }
        break;
      }

      default: {
      handle_unusual:
        if (tag == 0) {
          goto success;
        }
        DO_(::google::protobuf::internal::WireFormat::SkipField(
              input, tag, _internal_metadata_.mutable_unknown_fields()));
        break;
      }
    }
  }
success:
  // @@protoc_insertion_point(parse_success:TensorRTConfig.Config)
  return true;
failure:
  // @@protoc_insertion_point(parse_failure:TensorRTConfig.Config)
  return false;
#undef DO_
}

void Config::SerializeWithCachedSizes(
    ::google::protobuf::io::CodedOutputStream* output) const {
  // @@protoc_insertion_point(serialize_start:TensorRTConfig.Config)
  ::google::protobuf::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  // repeated string input_tensor_names = 1;
  for (int i = 0, n = this->input_tensor_names_size(); i < n; i++) {
    ::google::protobuf::internal::WireFormatLite::VerifyUtf8String(
      this->input_tensor_names(i).data(), static_cast<int>(this->input_tensor_names(i).length()),
      ::google::protobuf::internal::WireFormatLite::SERIALIZE,
      "TensorRTConfig.Config.input_tensor_names");
    ::google::protobuf::internal::WireFormatLite::WriteString(
      1, this->input_tensor_names(i), output);
  }

  // repeated string output_tensor_names = 2;
  for (int i = 0, n = this->output_tensor_names_size(); i < n; i++) {
    ::google::protobuf::internal::WireFormatLite::VerifyUtf8String(
      this->output_tensor_names(i).data(), static_cast<int>(this->output_tensor_names(i).length()),
      ::google::protobuf::internal::WireFormatLite::SERIALIZE,
      "TensorRTConfig.Config.output_tensor_names");
    ::google::protobuf::internal::WireFormatLite::WriteString(
      2, this->output_tensor_names(i), output);
  }

  // string onnx_file = 3;
  if (this->onnx_file().size() > 0) {
    ::google::protobuf::internal::WireFormatLite::VerifyUtf8String(
      this->onnx_file().data(), static_cast<int>(this->onnx_file().length()),
      ::google::protobuf::internal::WireFormatLite::SERIALIZE,
      "TensorRTConfig.Config.onnx_file");
    ::google::protobuf::internal::WireFormatLite::WriteStringMaybeAliased(
      3, this->onnx_file(), output);
  }

  // string engine_file = 4;
  if (this->engine_file().size() > 0) {
    ::google::protobuf::internal::WireFormatLite::VerifyUtf8String(
      this->engine_file().data(), static_cast<int>(this->engine_file().length()),
      ::google::protobuf::internal::WireFormatLite::SERIALIZE,
      "TensorRTConfig.Config.engine_file");
    ::google::protobuf::internal::WireFormatLite::WriteStringMaybeAliased(
      4, this->engine_file(), output);
  }

  // int32 dla_core = 5;
  if (this->dla_core() != 0) {
    ::google::protobuf::internal::WireFormatLite::WriteInt32(5, this->dla_core(), output);
  }

  // int32 batch_size = 6;
  if (this->batch_size() != 0) {
    ::google::protobuf::internal::WireFormatLite::WriteInt32(6, this->batch_size(), output);
  }

  // bool int8 = 7;
  if (this->int8() != 0) {
    ::google::protobuf::internal::WireFormatLite::WriteBool(7, this->int8(), output);
  }

  // bool fp16 = 8;
  if (this->fp16() != 0) {
    ::google::protobuf::internal::WireFormatLite::WriteBool(8, this->fp16(), output);
  }

  // bool bf16 = 9;
  if (this->bf16() != 0) {
    ::google::protobuf::internal::WireFormatLite::WriteBool(9, this->bf16(), output);
  }

  if ((_internal_metadata_.have_unknown_fields() &&  ::google::protobuf::internal::GetProto3PreserveUnknownsDefault())) {
    ::google::protobuf::internal::WireFormat::SerializeUnknownFields(
        (::google::protobuf::internal::GetProto3PreserveUnknownsDefault()   ? _internal_metadata_.unknown_fields()   : _internal_metadata_.default_instance()), output);
  }
  // @@protoc_insertion_point(serialize_end:TensorRTConfig.Config)
}

::google::protobuf::uint8* Config::InternalSerializeWithCachedSizesToArray(
    bool deterministic, ::google::protobuf::uint8* target) const {
  (void)deterministic; // Unused
  // @@protoc_insertion_point(serialize_to_array_start:TensorRTConfig.Config)
  ::google::protobuf::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  // repeated string input_tensor_names = 1;
  for (int i = 0, n = this->input_tensor_names_size(); i < n; i++) {
    ::google::protobuf::internal::WireFormatLite::VerifyUtf8String(
      this->input_tensor_names(i).data(), static_cast<int>(this->input_tensor_names(i).length()),
      ::google::protobuf::internal::WireFormatLite::SERIALIZE,
      "TensorRTConfig.Config.input_tensor_names");
    target = ::google::protobuf::internal::WireFormatLite::
      WriteStringToArray(1, this->input_tensor_names(i), target);
  }

  // repeated string output_tensor_names = 2;
  for (int i = 0, n = this->output_tensor_names_size(); i < n; i++) {
    ::google::protobuf::internal::WireFormatLite::VerifyUtf8String(
      this->output_tensor_names(i).data(), static_cast<int>(this->output_tensor_names(i).length()),
      ::google::protobuf::internal::WireFormatLite::SERIALIZE,
      "TensorRTConfig.Config.output_tensor_names");
    target = ::google::protobuf::internal::WireFormatLite::
      WriteStringToArray(2, this->output_tensor_names(i), target);
  }

  // string onnx_file = 3;
  if (this->onnx_file().size() > 0) {
    ::google::protobuf::internal::WireFormatLite::VerifyUtf8String(
      this->onnx_file().data(), static_cast<int>(this->onnx_file().length()),
      ::google::protobuf::internal::WireFormatLite::SERIALIZE,
      "TensorRTConfig.Config.onnx_file");
    target =
      ::google::protobuf::internal::WireFormatLite::WriteStringToArray(
        3, this->onnx_file(), target);
  }

  // string engine_file = 4;
  if (this->engine_file().size() > 0) {
    ::google::protobuf::internal::WireFormatLite::VerifyUtf8String(
      this->engine_file().data(), static_cast<int>(this->engine_file().length()),
      ::google::protobuf::internal::WireFormatLite::SERIALIZE,
      "TensorRTConfig.Config.engine_file");
    target =
      ::google::protobuf::internal::WireFormatLite::WriteStringToArray(
        4, this->engine_file(), target);
  }

  // int32 dla_core = 5;
  if (this->dla_core() != 0) {
    target = ::google::protobuf::internal::WireFormatLite::WriteInt32ToArray(5, this->dla_core(), target);
  }

  // int32 batch_size = 6;
  if (this->batch_size() != 0) {
    target = ::google::protobuf::internal::WireFormatLite::WriteInt32ToArray(6, this->batch_size(), target);
  }

  // bool int8 = 7;
  if (this->int8() != 0) {
    target = ::google::protobuf::internal::WireFormatLite::WriteBoolToArray(7, this->int8(), target);
  }

  // bool fp16 = 8;
  if (this->fp16() != 0) {
    target = ::google::protobuf::internal::WireFormatLite::WriteBoolToArray(8, this->fp16(), target);
  }

  // bool bf16 = 9;
  if (this->bf16() != 0) {
    target = ::google::protobuf::internal::WireFormatLite::WriteBoolToArray(9, this->bf16(), target);
  }

  if ((_internal_metadata_.have_unknown_fields() &&  ::google::protobuf::internal::GetProto3PreserveUnknownsDefault())) {
    target = ::google::protobuf::internal::WireFormat::SerializeUnknownFieldsToArray(
        (::google::protobuf::internal::GetProto3PreserveUnknownsDefault()   ? _internal_metadata_.unknown_fields()   : _internal_metadata_.default_instance()), target);
  }
  // @@protoc_insertion_point(serialize_to_array_end:TensorRTConfig.Config)
  return target;
}

size_t Config::ByteSizeLong() const {
// @@protoc_insertion_point(message_byte_size_start:TensorRTConfig.Config)
  size_t total_size = 0;

  if ((_internal_metadata_.have_unknown_fields() &&  ::google::protobuf::internal::GetProto3PreserveUnknownsDefault())) {
    total_size +=
      ::google::protobuf::internal::WireFormat::ComputeUnknownFieldsSize(
        (::google::protobuf::internal::GetProto3PreserveUnknownsDefault()   ? _internal_metadata_.unknown_fields()   : _internal_metadata_.default_instance()));
  }
  // repeated string input_tensor_names = 1;
  total_size += 1 *
      ::google::protobuf::internal::FromIntSize(this->input_tensor_names_size());
  for (int i = 0, n = this->input_tensor_names_size(); i < n; i++) {
    total_size += ::google::protobuf::internal::WireFormatLite::StringSize(
      this->input_tensor_names(i));
  }

  // repeated string output_tensor_names = 2;
  total_size += 1 *
      ::google::protobuf::internal::FromIntSize(this->output_tensor_names_size());
  for (int i = 0, n = this->output_tensor_names_size(); i < n; i++) {
    total_size += ::google::protobuf::internal::WireFormatLite::StringSize(
      this->output_tensor_names(i));
  }

  // string onnx_file = 3;
  if (this->onnx_file().size() > 0) {
    total_size += 1 +
      ::google::protobuf::internal::WireFormatLite::StringSize(
        this->onnx_file());
  }

  // string engine_file = 4;
  if (this->engine_file().size() > 0) {
    total_size += 1 +
      ::google::protobuf::internal::WireFormatLite::StringSize(
        this->engine_file());
  }

  // int32 dla_core = 5;
  if (this->dla_core() != 0) {
    total_size += 1 +
      ::google::protobuf::internal::WireFormatLite::Int32Size(
        this->dla_core());
  }

  // int32 batch_size = 6;
  if (this->batch_size() != 0) {
    total_size += 1 +
      ::google::protobuf::internal::WireFormatLite::Int32Size(
        this->batch_size());
  }

  // bool int8 = 7;
  if (this->int8() != 0) {
    total_size += 1 + 1;
  }

  // bool fp16 = 8;
  if (this->fp16() != 0) {
    total_size += 1 + 1;
  }

  // bool bf16 = 9;
  if (this->bf16() != 0) {
    total_size += 1 + 1;
  }

  int cached_size = ::google::protobuf::internal::ToCachedSize(total_size);
  SetCachedSize(cached_size);
  return total_size;
}

void Config::MergeFrom(const ::google::protobuf::Message& from) {
// @@protoc_insertion_point(generalized_merge_from_start:TensorRTConfig.Config)
  GOOGLE_DCHECK_NE(&from, this);
  const Config* source =
      ::google::protobuf::internal::DynamicCastToGenerated<const Config>(
          &from);
  if (source == NULL) {
  // @@protoc_insertion_point(generalized_merge_from_cast_fail:TensorRTConfig.Config)
    ::google::protobuf::internal::ReflectionOps::Merge(from, this);
  } else {
  // @@protoc_insertion_point(generalized_merge_from_cast_success:TensorRTConfig.Config)
    MergeFrom(*source);
  }
}

void Config::MergeFrom(const Config& from) {
// @@protoc_insertion_point(class_specific_merge_from_start:TensorRTConfig.Config)
  GOOGLE_DCHECK_NE(&from, this);
  _internal_metadata_.MergeFrom(from._internal_metadata_);
  ::google::protobuf::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  input_tensor_names_.MergeFrom(from.input_tensor_names_);
  output_tensor_names_.MergeFrom(from.output_tensor_names_);
  if (from.onnx_file().size() > 0) {

    onnx_file_.AssignWithDefault(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), from.onnx_file_);
  }
  if (from.engine_file().size() > 0) {

    engine_file_.AssignWithDefault(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), from.engine_file_);
  }
  if (from.dla_core() != 0) {
    set_dla_core(from.dla_core());
  }
  if (from.batch_size() != 0) {
    set_batch_size(from.batch_size());
  }
  if (from.int8() != 0) {
    set_int8(from.int8());
  }
  if (from.fp16() != 0) {
    set_fp16(from.fp16());
  }
  if (from.bf16() != 0) {
    set_bf16(from.bf16());
  }
}

void Config::CopyFrom(const ::google::protobuf::Message& from) {
// @@protoc_insertion_point(generalized_copy_from_start:TensorRTConfig.Config)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

void Config::CopyFrom(const Config& from) {
// @@protoc_insertion_point(class_specific_copy_from_start:TensorRTConfig.Config)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool Config::IsInitialized() const {
  return true;
}

void Config::Swap(Config* other) {
  if (other == this) return;
  InternalSwap(other);
}
void Config::InternalSwap(Config* other) {
  using std::swap;
  input_tensor_names_.InternalSwap(CastToBase(&other->input_tensor_names_));
  output_tensor_names_.InternalSwap(CastToBase(&other->output_tensor_names_));
  onnx_file_.Swap(&other->onnx_file_, &::google::protobuf::internal::GetEmptyStringAlreadyInited(),
    GetArenaNoVirtual());
  engine_file_.Swap(&other->engine_file_, &::google::protobuf::internal::GetEmptyStringAlreadyInited(),
    GetArenaNoVirtual());
  swap(dla_core_, other->dla_core_);
  swap(batch_size_, other->batch_size_);
  swap(int8_, other->int8_);
  swap(fp16_, other->fp16_);
  swap(bf16_, other->bf16_);
  _internal_metadata_.Swap(&other->_internal_metadata_);
}

::google::protobuf::Metadata Config::GetMetadata() const {
  protobuf_tensorRT_2eproto::protobuf_AssignDescriptorsOnce();
  return ::protobuf_tensorRT_2eproto::file_level_metadata[kIndexInFileMessages];
}


// @@protoc_insertion_point(namespace_scope)
}  // namespace TensorRTConfig
namespace google {
namespace protobuf {
template<> GOOGLE_PROTOBUF_ATTRIBUTE_NOINLINE ::TensorRTConfig::Config* Arena::CreateMaybeMessage< ::TensorRTConfig::Config >(Arena* arena) {
  return Arena::CreateInternal< ::TensorRTConfig::Config >(arena);
}
}  // namespace protobuf
}  // namespace google

// @@protoc_insertion_point(global_scope)
