// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: all_config.proto

#ifndef PROTOBUF_INCLUDED_all_5fconfig_2eproto
#define PROTOBUF_INCLUDED_all_5fconfig_2eproto

#include <string>

#include <google/protobuf/stubs/common.h>

#if GOOGLE_PROTOBUF_VERSION < 3006001
#error This file was generated by a newer version of protoc which is
#error incompatible with your Protocol Buffer headers.  Please update
#error your headers.
#endif
#if 3006001 < GOOGLE_PROTOBUF_MIN_PROTOC_VERSION
#error This file was generated by an older version of protoc which is
#error incompatible with your Protocol Buffer headers.  Please
#error regenerate this file with a newer version of protoc.
#endif

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/arena.h>
#include <google/protobuf/arenastring.h>
#include <google/protobuf/generated_message_table_driven.h>
#include <google/protobuf/generated_message_util.h>
#include <google/protobuf/inlined_string_field.h>
#include <google/protobuf/metadata.h>
#include <google/protobuf/message.h>
#include <google/protobuf/repeated_field.h>  // IWYU pragma: export
#include <google/protobuf/extension_set.h>  // IWYU pragma: export
#include <google/protobuf/unknown_field_set.h>
#include "utils.pb.h"
#include "imu.pb.h"
#include "camera.pb.h"
// @@protoc_insertion_point(includes)
#define PROTOBUF_INTERNAL_EXPORT_protobuf_all_5fconfig_2eproto 

namespace protobuf_all_5fconfig_2eproto {
// Internal implementation detail -- do not use these members.
struct TableStruct {
  static const ::google::protobuf::internal::ParseTableField entries[];
  static const ::google::protobuf::internal::AuxillaryParseTableField aux[];
  static const ::google::protobuf::internal::ParseTable schema[1];
  static const ::google::protobuf::internal::FieldMetadata field_metadata[];
  static const ::google::protobuf::internal::SerializationTable serialization_table[];
  static const ::google::protobuf::uint32 offsets[];
};
void AddDescriptors();
}  // namespace protobuf_all_5fconfig_2eproto
namespace AllConfigs {
class Config;
class ConfigDefaultTypeInternal;
extern ConfigDefaultTypeInternal _Config_default_instance_;
}  // namespace AllConfigs
namespace google {
namespace protobuf {
template<> ::AllConfigs::Config* Arena::CreateMaybeMessage<::AllConfigs::Config>(Arena*);
}  // namespace protobuf
}  // namespace google
namespace AllConfigs {

// ===================================================================

class Config : public ::google::protobuf::Message /* @@protoc_insertion_point(class_definition:AllConfigs.Config) */ {
 public:
  Config();
  virtual ~Config();

  Config(const Config& from);

  inline Config& operator=(const Config& from) {
    CopyFrom(from);
    return *this;
  }
  #if LANG_CXX11
  Config(Config&& from) noexcept
    : Config() {
    *this = ::std::move(from);
  }

  inline Config& operator=(Config&& from) noexcept {
    if (GetArenaNoVirtual() == from.GetArenaNoVirtual()) {
      if (this != &from) InternalSwap(&from);
    } else {
      CopyFrom(from);
    }
    return *this;
  }
  #endif
  static const ::google::protobuf::Descriptor* descriptor();
  static const Config& default_instance();

  static void InitAsDefaultInstance();  // FOR INTERNAL USE ONLY
  static inline const Config* internal_default_instance() {
    return reinterpret_cast<const Config*>(
               &_Config_default_instance_);
  }
  static constexpr int kIndexInFileMessages =
    0;

  void Swap(Config* other);
  friend void swap(Config& a, Config& b) {
    a.Swap(&b);
  }

  // implements Message ----------------------------------------------

  inline Config* New() const final {
    return CreateMaybeMessage<Config>(NULL);
  }

  Config* New(::google::protobuf::Arena* arena) const final {
    return CreateMaybeMessage<Config>(arena);
  }
  void CopyFrom(const ::google::protobuf::Message& from) final;
  void MergeFrom(const ::google::protobuf::Message& from) final;
  void CopyFrom(const Config& from);
  void MergeFrom(const Config& from);
  void Clear() final;
  bool IsInitialized() const final;

  size_t ByteSizeLong() const final;
  bool MergePartialFromCodedStream(
      ::google::protobuf::io::CodedInputStream* input) final;
  void SerializeWithCachedSizes(
      ::google::protobuf::io::CodedOutputStream* output) const final;
  ::google::protobuf::uint8* InternalSerializeWithCachedSizesToArray(
      bool deterministic, ::google::protobuf::uint8* target) const final;
  int GetCachedSize() const final { return _cached_size_.Get(); }

  private:
  void SharedCtor();
  void SharedDtor();
  void SetCachedSize(int size) const final;
  void InternalSwap(Config* other);
  private:
  inline ::google::protobuf::Arena* GetArenaNoVirtual() const {
    return NULL;
  }
  inline void* MaybeArenaPtr() const {
    return NULL;
  }
  public:

  ::google::protobuf::Metadata GetMetadata() const final;

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  // .IMUConfig.Config imu_config = 1;
  bool has_imu_config() const;
  void clear_imu_config();
  static const int kImuConfigFieldNumber = 1;
  private:
  const ::IMUConfig::Config& _internal_imu_config() const;
  public:
  const ::IMUConfig::Config& imu_config() const;
  ::IMUConfig::Config* release_imu_config();
  ::IMUConfig::Config* mutable_imu_config();
  void set_allocated_imu_config(::IMUConfig::Config* imu_config);

  // .CameraConfig.Config camera_config = 2;
  bool has_camera_config() const;
  void clear_camera_config();
  static const int kCameraConfigFieldNumber = 2;
  private:
  const ::CameraConfig::Config& _internal_camera_config() const;
  public:
  const ::CameraConfig::Config& camera_config() const;
  ::CameraConfig::Config* release_camera_config();
  ::CameraConfig::Config* mutable_camera_config();
  void set_allocated_camera_config(::CameraConfig::Config* camera_config);

  // @@protoc_insertion_point(class_scope:AllConfigs.Config)
 private:

  ::google::protobuf::internal::InternalMetadataWithArena _internal_metadata_;
  ::IMUConfig::Config* imu_config_;
  ::CameraConfig::Config* camera_config_;
  mutable ::google::protobuf::internal::CachedSize _cached_size_;
  friend struct ::protobuf_all_5fconfig_2eproto::TableStruct;
};
// ===================================================================


// ===================================================================

#ifdef __GNUC__
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif  // __GNUC__
// Config

// .IMUConfig.Config imu_config = 1;
inline bool Config::has_imu_config() const {
  return this != internal_default_instance() && imu_config_ != NULL;
}
inline const ::IMUConfig::Config& Config::_internal_imu_config() const {
  return *imu_config_;
}
inline const ::IMUConfig::Config& Config::imu_config() const {
  const ::IMUConfig::Config* p = imu_config_;
  // @@protoc_insertion_point(field_get:AllConfigs.Config.imu_config)
  return p != NULL ? *p : *reinterpret_cast<const ::IMUConfig::Config*>(
      &::IMUConfig::_Config_default_instance_);
}
inline ::IMUConfig::Config* Config::release_imu_config() {
  // @@protoc_insertion_point(field_release:AllConfigs.Config.imu_config)
  
  ::IMUConfig::Config* temp = imu_config_;
  imu_config_ = NULL;
  return temp;
}
inline ::IMUConfig::Config* Config::mutable_imu_config() {
  
  if (imu_config_ == NULL) {
    auto* p = CreateMaybeMessage<::IMUConfig::Config>(GetArenaNoVirtual());
    imu_config_ = p;
  }
  // @@protoc_insertion_point(field_mutable:AllConfigs.Config.imu_config)
  return imu_config_;
}
inline void Config::set_allocated_imu_config(::IMUConfig::Config* imu_config) {
  ::google::protobuf::Arena* message_arena = GetArenaNoVirtual();
  if (message_arena == NULL) {
    delete reinterpret_cast< ::google::protobuf::MessageLite*>(imu_config_);
  }
  if (imu_config) {
    ::google::protobuf::Arena* submessage_arena = NULL;
    if (message_arena != submessage_arena) {
      imu_config = ::google::protobuf::internal::GetOwnedMessage(
          message_arena, imu_config, submessage_arena);
    }
    
  } else {
    
  }
  imu_config_ = imu_config;
  // @@protoc_insertion_point(field_set_allocated:AllConfigs.Config.imu_config)
}

// .CameraConfig.Config camera_config = 2;
inline bool Config::has_camera_config() const {
  return this != internal_default_instance() && camera_config_ != NULL;
}
inline const ::CameraConfig::Config& Config::_internal_camera_config() const {
  return *camera_config_;
}
inline const ::CameraConfig::Config& Config::camera_config() const {
  const ::CameraConfig::Config* p = camera_config_;
  // @@protoc_insertion_point(field_get:AllConfigs.Config.camera_config)
  return p != NULL ? *p : *reinterpret_cast<const ::CameraConfig::Config*>(
      &::CameraConfig::_Config_default_instance_);
}
inline ::CameraConfig::Config* Config::release_camera_config() {
  // @@protoc_insertion_point(field_release:AllConfigs.Config.camera_config)
  
  ::CameraConfig::Config* temp = camera_config_;
  camera_config_ = NULL;
  return temp;
}
inline ::CameraConfig::Config* Config::mutable_camera_config() {
  
  if (camera_config_ == NULL) {
    auto* p = CreateMaybeMessage<::CameraConfig::Config>(GetArenaNoVirtual());
    camera_config_ = p;
  }
  // @@protoc_insertion_point(field_mutable:AllConfigs.Config.camera_config)
  return camera_config_;
}
inline void Config::set_allocated_camera_config(::CameraConfig::Config* camera_config) {
  ::google::protobuf::Arena* message_arena = GetArenaNoVirtual();
  if (message_arena == NULL) {
    delete reinterpret_cast< ::google::protobuf::MessageLite*>(camera_config_);
  }
  if (camera_config) {
    ::google::protobuf::Arena* submessage_arena = NULL;
    if (message_arena != submessage_arena) {
      camera_config = ::google::protobuf::internal::GetOwnedMessage(
          message_arena, camera_config, submessage_arena);
    }
    
  } else {
    
  }
  camera_config_ = camera_config;
  // @@protoc_insertion_point(field_set_allocated:AllConfigs.Config.camera_config)
}

#ifdef __GNUC__
  #pragma GCC diagnostic pop
#endif  // __GNUC__

// @@protoc_insertion_point(namespace_scope)

}  // namespace AllConfigs

// @@protoc_insertion_point(global_scope)

#endif  // PROTOBUF_INCLUDED_all_5fconfig_2eproto
