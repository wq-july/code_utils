// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: all_config.proto

#include "all_config.pb.h"

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

namespace protobuf_camera_2eproto {
extern PROTOBUF_INTERNAL_EXPORT_protobuf_camera_2eproto ::google::protobuf::internal::SCCInfo<2> scc_info_Config;
}  // namespace protobuf_camera_2eproto
namespace protobuf_imu_2eproto {
extern PROTOBUF_INTERNAL_EXPORT_protobuf_imu_2eproto ::google::protobuf::internal::SCCInfo<3> scc_info_Config;
}  // namespace protobuf_imu_2eproto
namespace AllConfigs {
class ConfigDefaultTypeInternal {
 public:
  ::google::protobuf::internal::ExplicitlyConstructed<Config>
      _instance;
} _Config_default_instance_;
}  // namespace AllConfigs
namespace protobuf_all_5fconfig_2eproto {
static void InitDefaultsConfig() {
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  {
    void* ptr = &::AllConfigs::_Config_default_instance_;
    new (ptr) ::AllConfigs::Config();
    ::google::protobuf::internal::OnShutdownDestroyMessage(ptr);
  }
  ::AllConfigs::Config::InitAsDefaultInstance();
}

::google::protobuf::internal::SCCInfo<2> scc_info_Config =
    {{ATOMIC_VAR_INIT(::google::protobuf::internal::SCCInfoBase::kUninitialized), 2, InitDefaultsConfig}, {
      &protobuf_imu_2eproto::scc_info_Config.base,
      &protobuf_camera_2eproto::scc_info_Config.base,}};

void InitDefaults() {
  ::google::protobuf::internal::InitSCC(&scc_info_Config.base);
}

::google::protobuf::Metadata file_level_metadata[1];

const ::google::protobuf::uint32 TableStruct::offsets[] GOOGLE_PROTOBUF_ATTRIBUTE_SECTION_VARIABLE(protodesc_cold) = {
  ~0u,  // no _has_bits_
  GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(::AllConfigs::Config, _internal_metadata_),
  ~0u,  // no _extensions_
  ~0u,  // no _oneof_case_
  ~0u,  // no _weak_field_map_
  GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(::AllConfigs::Config, imu_config_),
  GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(::AllConfigs::Config, camera_config_),
};
static const ::google::protobuf::internal::MigrationSchema schemas[] GOOGLE_PROTOBUF_ATTRIBUTE_SECTION_VARIABLE(protodesc_cold) = {
  { 0, -1, sizeof(::AllConfigs::Config)},
};

static ::google::protobuf::Message const * const file_default_instances[] = {
  reinterpret_cast<const ::google::protobuf::Message*>(&::AllConfigs::_Config_default_instance_),
};

void protobuf_AssignDescriptors() {
  AddDescriptors();
  AssignDescriptors(
      "all_config.proto", schemas, file_default_instances, TableStruct::offsets,
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
      "\n\020all_config.proto\022\nAllConfigs\032\013utils.pr"
      "oto\032\timu.proto\032\014camera.proto\"\\\n\006Config\022%"
      "\n\nimu_config\030\001 \001(\0132\021.IMUConfig.Config\022+\n"
      "\rcamera_config\030\002 \001(\0132\024.CameraConfig.Conf"
      "igb\006proto3"
  };
  ::google::protobuf::DescriptorPool::InternalAddGeneratedFile(
      descriptor, 170);
  ::google::protobuf::MessageFactory::InternalRegisterGeneratedFile(
    "all_config.proto", &protobuf_RegisterTypes);
  ::protobuf_utils_2eproto::AddDescriptors();
  ::protobuf_imu_2eproto::AddDescriptors();
  ::protobuf_camera_2eproto::AddDescriptors();
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
}  // namespace protobuf_all_5fconfig_2eproto
namespace AllConfigs {

// ===================================================================

void Config::InitAsDefaultInstance() {
  ::AllConfigs::_Config_default_instance_._instance.get_mutable()->imu_config_ = const_cast< ::IMUConfig::Config*>(
      ::IMUConfig::Config::internal_default_instance());
  ::AllConfigs::_Config_default_instance_._instance.get_mutable()->camera_config_ = const_cast< ::CameraConfig::Config*>(
      ::CameraConfig::Config::internal_default_instance());
}
void Config::clear_imu_config() {
  if (GetArenaNoVirtual() == NULL && imu_config_ != NULL) {
    delete imu_config_;
  }
  imu_config_ = NULL;
}
void Config::clear_camera_config() {
  if (GetArenaNoVirtual() == NULL && camera_config_ != NULL) {
    delete camera_config_;
  }
  camera_config_ = NULL;
}
#if !defined(_MSC_VER) || _MSC_VER >= 1900
const int Config::kImuConfigFieldNumber;
const int Config::kCameraConfigFieldNumber;
#endif  // !defined(_MSC_VER) || _MSC_VER >= 1900

Config::Config()
  : ::google::protobuf::Message(), _internal_metadata_(NULL) {
  ::google::protobuf::internal::InitSCC(
      &protobuf_all_5fconfig_2eproto::scc_info_Config.base);
  SharedCtor();
  // @@protoc_insertion_point(constructor:AllConfigs.Config)
}
Config::Config(const Config& from)
  : ::google::protobuf::Message(),
      _internal_metadata_(NULL) {
  _internal_metadata_.MergeFrom(from._internal_metadata_);
  if (from.has_imu_config()) {
    imu_config_ = new ::IMUConfig::Config(*from.imu_config_);
  } else {
    imu_config_ = NULL;
  }
  if (from.has_camera_config()) {
    camera_config_ = new ::CameraConfig::Config(*from.camera_config_);
  } else {
    camera_config_ = NULL;
  }
  // @@protoc_insertion_point(copy_constructor:AllConfigs.Config)
}

void Config::SharedCtor() {
  ::memset(&imu_config_, 0, static_cast<size_t>(
      reinterpret_cast<char*>(&camera_config_) -
      reinterpret_cast<char*>(&imu_config_)) + sizeof(camera_config_));
}

Config::~Config() {
  // @@protoc_insertion_point(destructor:AllConfigs.Config)
  SharedDtor();
}

void Config::SharedDtor() {
  if (this != internal_default_instance()) delete imu_config_;
  if (this != internal_default_instance()) delete camera_config_;
}

void Config::SetCachedSize(int size) const {
  _cached_size_.Set(size);
}
const ::google::protobuf::Descriptor* Config::descriptor() {
  ::protobuf_all_5fconfig_2eproto::protobuf_AssignDescriptorsOnce();
  return ::protobuf_all_5fconfig_2eproto::file_level_metadata[kIndexInFileMessages].descriptor;
}

const Config& Config::default_instance() {
  ::google::protobuf::internal::InitSCC(&protobuf_all_5fconfig_2eproto::scc_info_Config.base);
  return *internal_default_instance();
}


void Config::Clear() {
// @@protoc_insertion_point(message_clear_start:AllConfigs.Config)
  ::google::protobuf::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  if (GetArenaNoVirtual() == NULL && imu_config_ != NULL) {
    delete imu_config_;
  }
  imu_config_ = NULL;
  if (GetArenaNoVirtual() == NULL && camera_config_ != NULL) {
    delete camera_config_;
  }
  camera_config_ = NULL;
  _internal_metadata_.Clear();
}

bool Config::MergePartialFromCodedStream(
    ::google::protobuf::io::CodedInputStream* input) {
#define DO_(EXPRESSION) if (!GOOGLE_PREDICT_TRUE(EXPRESSION)) goto failure
  ::google::protobuf::uint32 tag;
  // @@protoc_insertion_point(parse_start:AllConfigs.Config)
  for (;;) {
    ::std::pair<::google::protobuf::uint32, bool> p = input->ReadTagWithCutoffNoLastTag(127u);
    tag = p.first;
    if (!p.second) goto handle_unusual;
    switch (::google::protobuf::internal::WireFormatLite::GetTagFieldNumber(tag)) {
      // .IMUConfig.Config imu_config = 1;
      case 1: {
        if (static_cast< ::google::protobuf::uint8>(tag) ==
            static_cast< ::google::protobuf::uint8>(10u /* 10 & 0xFF */)) {
          DO_(::google::protobuf::internal::WireFormatLite::ReadMessage(
               input, mutable_imu_config()));
        } else {
          goto handle_unusual;
        }
        break;
      }

      // .CameraConfig.Config camera_config = 2;
      case 2: {
        if (static_cast< ::google::protobuf::uint8>(tag) ==
            static_cast< ::google::protobuf::uint8>(18u /* 18 & 0xFF */)) {
          DO_(::google::protobuf::internal::WireFormatLite::ReadMessage(
               input, mutable_camera_config()));
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
  // @@protoc_insertion_point(parse_success:AllConfigs.Config)
  return true;
failure:
  // @@protoc_insertion_point(parse_failure:AllConfigs.Config)
  return false;
#undef DO_
}

void Config::SerializeWithCachedSizes(
    ::google::protobuf::io::CodedOutputStream* output) const {
  // @@protoc_insertion_point(serialize_start:AllConfigs.Config)
  ::google::protobuf::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  // .IMUConfig.Config imu_config = 1;
  if (this->has_imu_config()) {
    ::google::protobuf::internal::WireFormatLite::WriteMessageMaybeToArray(
      1, this->_internal_imu_config(), output);
  }

  // .CameraConfig.Config camera_config = 2;
  if (this->has_camera_config()) {
    ::google::protobuf::internal::WireFormatLite::WriteMessageMaybeToArray(
      2, this->_internal_camera_config(), output);
  }

  if ((_internal_metadata_.have_unknown_fields() &&  ::google::protobuf::internal::GetProto3PreserveUnknownsDefault())) {
    ::google::protobuf::internal::WireFormat::SerializeUnknownFields(
        (::google::protobuf::internal::GetProto3PreserveUnknownsDefault()   ? _internal_metadata_.unknown_fields()   : _internal_metadata_.default_instance()), output);
  }
  // @@protoc_insertion_point(serialize_end:AllConfigs.Config)
}

::google::protobuf::uint8* Config::InternalSerializeWithCachedSizesToArray(
    bool deterministic, ::google::protobuf::uint8* target) const {
  (void)deterministic; // Unused
  // @@protoc_insertion_point(serialize_to_array_start:AllConfigs.Config)
  ::google::protobuf::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  // .IMUConfig.Config imu_config = 1;
  if (this->has_imu_config()) {
    target = ::google::protobuf::internal::WireFormatLite::
      InternalWriteMessageToArray(
        1, this->_internal_imu_config(), deterministic, target);
  }

  // .CameraConfig.Config camera_config = 2;
  if (this->has_camera_config()) {
    target = ::google::protobuf::internal::WireFormatLite::
      InternalWriteMessageToArray(
        2, this->_internal_camera_config(), deterministic, target);
  }

  if ((_internal_metadata_.have_unknown_fields() &&  ::google::protobuf::internal::GetProto3PreserveUnknownsDefault())) {
    target = ::google::protobuf::internal::WireFormat::SerializeUnknownFieldsToArray(
        (::google::protobuf::internal::GetProto3PreserveUnknownsDefault()   ? _internal_metadata_.unknown_fields()   : _internal_metadata_.default_instance()), target);
  }
  // @@protoc_insertion_point(serialize_to_array_end:AllConfigs.Config)
  return target;
}

size_t Config::ByteSizeLong() const {
// @@protoc_insertion_point(message_byte_size_start:AllConfigs.Config)
  size_t total_size = 0;

  if ((_internal_metadata_.have_unknown_fields() &&  ::google::protobuf::internal::GetProto3PreserveUnknownsDefault())) {
    total_size +=
      ::google::protobuf::internal::WireFormat::ComputeUnknownFieldsSize(
        (::google::protobuf::internal::GetProto3PreserveUnknownsDefault()   ? _internal_metadata_.unknown_fields()   : _internal_metadata_.default_instance()));
  }
  // .IMUConfig.Config imu_config = 1;
  if (this->has_imu_config()) {
    total_size += 1 +
      ::google::protobuf::internal::WireFormatLite::MessageSize(
        *imu_config_);
  }

  // .CameraConfig.Config camera_config = 2;
  if (this->has_camera_config()) {
    total_size += 1 +
      ::google::protobuf::internal::WireFormatLite::MessageSize(
        *camera_config_);
  }

  int cached_size = ::google::protobuf::internal::ToCachedSize(total_size);
  SetCachedSize(cached_size);
  return total_size;
}

void Config::MergeFrom(const ::google::protobuf::Message& from) {
// @@protoc_insertion_point(generalized_merge_from_start:AllConfigs.Config)
  GOOGLE_DCHECK_NE(&from, this);
  const Config* source =
      ::google::protobuf::internal::DynamicCastToGenerated<const Config>(
          &from);
  if (source == NULL) {
  // @@protoc_insertion_point(generalized_merge_from_cast_fail:AllConfigs.Config)
    ::google::protobuf::internal::ReflectionOps::Merge(from, this);
  } else {
  // @@protoc_insertion_point(generalized_merge_from_cast_success:AllConfigs.Config)
    MergeFrom(*source);
  }
}

void Config::MergeFrom(const Config& from) {
// @@protoc_insertion_point(class_specific_merge_from_start:AllConfigs.Config)
  GOOGLE_DCHECK_NE(&from, this);
  _internal_metadata_.MergeFrom(from._internal_metadata_);
  ::google::protobuf::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  if (from.has_imu_config()) {
    mutable_imu_config()->::IMUConfig::Config::MergeFrom(from.imu_config());
  }
  if (from.has_camera_config()) {
    mutable_camera_config()->::CameraConfig::Config::MergeFrom(from.camera_config());
  }
}

void Config::CopyFrom(const ::google::protobuf::Message& from) {
// @@protoc_insertion_point(generalized_copy_from_start:AllConfigs.Config)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

void Config::CopyFrom(const Config& from) {
// @@protoc_insertion_point(class_specific_copy_from_start:AllConfigs.Config)
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
  swap(imu_config_, other->imu_config_);
  swap(camera_config_, other->camera_config_);
  _internal_metadata_.Swap(&other->_internal_metadata_);
}

::google::protobuf::Metadata Config::GetMetadata() const {
  protobuf_all_5fconfig_2eproto::protobuf_AssignDescriptorsOnce();
  return ::protobuf_all_5fconfig_2eproto::file_level_metadata[kIndexInFileMessages];
}


// @@protoc_insertion_point(namespace_scope)
}  // namespace AllConfigs
namespace google {
namespace protobuf {
template<> GOOGLE_PROTOBUF_ATTRIBUTE_NOINLINE ::AllConfigs::Config* Arena::CreateMaybeMessage< ::AllConfigs::Config >(Arena* arena) {
  return Arena::CreateInternal< ::AllConfigs::Config >(arena);
}
}  // namespace protobuf
}  // namespace google

// @@protoc_insertion_point(global_scope)