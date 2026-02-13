// cnpy - C++ library for loading and saving NumPy npy and npz files.
// This is a trimmed-down subset of the upstream project
//   https://github.com/rogersce/cnpy
// that is sufficient for MLC-LLM's LoRA loader.  Only the pieces required
// for reading .npz archives (zip of .npy files) are kept.  The implementation
// is header-only for ease of integration on all platforms.
//
// License: MIT
#pragma once

#include <cstdint>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// We depend on <zlib>.  It is available on Linux and macOS by default; on
// Windows we rely on the system's zlib development package (or vcpkg).
#include <zlib.h>

namespace cnpy {

struct NpyArray {
  std::vector<size_t> shape;
  bool fortran_order{false};
  size_t word_size{0};                             // bytes per element
  std::shared_ptr<std::vector<char>> data_holder;  // shared so copies are cheap

  template <typename T>
  T* data() {
    return reinterpret_cast<T*>(data_holder->data());
  }
  template <typename T>
  const T* data() const {
    return reinterpret_cast<const T*>(data_holder->data());
  }
};

namespace detail {

// Read little-endian 4-byte unsigned int.
inline uint32_t read_le_uint32(std::istream& is) {
  uint32_t val;
  is.read(reinterpret_cast<char*>(&val), sizeof(val));
  return val;
}

// Validate magic string (\x93NUMPY) and version 1.0/2.0.
inline void parse_npy_header(std::istream& is, NpyArray& arr, std::string& descr_dtype) {
  char magic[6];
  is.read(magic, 6);
  if (std::memcmp(magic, "\x93NUMPY", 6) != 0) {
    throw std::runtime_error("Invalid .npy file – bad magic");
  }
  uint8_t major, minor;
  is.read(reinterpret_cast<char*>(&major), 1);
  is.read(reinterpret_cast<char*>(&minor), 1);
  uint16_t header_len16;
  if (major == 1) {
    header_len16 = static_cast<uint16_t>(read_le_uint32(is));
  } else if (major == 2) {
    header_len16 = static_cast<uint16_t>(read_le_uint32(is));
  } else {
    throw std::runtime_error("Unsupported .npy version");
  }
  std::string header(header_len16, '\0');
  is.read(header.data(), header_len16);

  // Parse header dictionary – extremely small, so simple string parsing is ok.
  auto loc_descr = header.find("'descr':");
  auto loc_shape = header.find("'shape':");
  auto loc_fortran = header.find("'fortran_order':");
  if (loc_descr == std::string::npos || loc_shape == std::string::npos) {
    throw std::runtime_error("Malformed .npy header");
  }
  // dtype string is delimited by quotes.
  auto start = header.find("'", loc_descr + 7) + 1;
  auto end = header.find("'", start);
  descr_dtype = header.substr(start, end - start);

  // Parse shape tuple, e.g. (3, 4, 5)
  start = header.find("(", loc_shape);
  end = header.find(")", start);
  std::string shape_str = header.substr(start + 1, end - start - 1);
  size_t pos = 0;
  while (true) {
    size_t comma = shape_str.find(',', pos);
    std::string dim = shape_str.substr(pos, comma - pos);
    if (!dim.empty()) {
      arr.shape.push_back(static_cast<size_t>(std::stoul(dim)));
    }
    if (comma == std::string::npos) break;
    pos = comma + 1;
  }

  // fortran_order
  if (loc_fortran != std::string::npos) {
    size_t loc_true = header.find("True", loc_fortran);
    arr.fortran_order = (loc_true != std::string::npos && loc_true < header.find(',', loc_fortran));
  }
}

inline size_t dtype_to_word_size(const std::string& descr) {
  if (descr == "<f4" || descr == "|f4") return 4;
  if (descr == "<f2" || descr == "|f2") return 2;
  if (descr == "<f8" || descr == "|f8") return 8;
  throw std::runtime_error("Unsupported dtype in .npy: " + descr);
}

}  // namespace detail

// Load a single .npy from an std::istream positioned at the array.
inline NpyArray load_npy_stream(std::istream& is) {
  NpyArray arr;
  std::string dtype;
  detail::parse_npy_header(is, arr, dtype);
  arr.word_size = detail::dtype_to_word_size(dtype);
  size_t num_elems = 1;
  for (size_t d : arr.shape) num_elems *= d;
  size_t bytes = num_elems * arr.word_size;
  arr.data_holder = std::make_shared<std::vector<char>>(bytes);
  is.read(arr.data_holder->data(), bytes);
  return arr;
}

// Load *all* arrays from an .npz archive.  This minimal implementation works
// because our LoRA adapters store tens of small arrays at most.
inline std::map<std::string, NpyArray> npz_load(const std::string& fname) {
  std::map<std::string, NpyArray> arrays;
  // Open zip file via zlib's unz API (minizip).  For portability we use the
  // simpler gz* interface + .tar hack: not ideal but avoids adding minizip.
  // Instead, we fall back to famous observation that .npz is a normal zip:
  // Here we only support *stored* (compression method 0) entries which is the
  // default for numpy (since 2023).  If the file uses DEFLATE we error out.

  // To keep integration simple and header-only, we restrict to uncompressed
  // archives: each member is concatenated so we can parse manually.
  std::ifstream fs(fname, std::ios::binary);
  if (!fs) throw std::runtime_error("Cannot open npz file: " + fname);

  // Very small, naive ZIP reader.  We scan for "PK\x03\x04" local headers and
  // read the contained .npy blobs.  Enough for CI/sanity tests.
  const uint32_t kSig = 0x04034b50;  // little-endian PK\x03\x04
  while (true) {
    uint32_t sig;
    fs.read(reinterpret_cast<char*>(&sig), 4);
    if (!fs) break;  // EOF
    if (sig != kSig) {
      throw std::runtime_error("Unsupported compression in npz (need stored) or bad signature");
    }
    uint16_t version, flags, method;
    uint16_t modtime, moddate;
    uint32_t crc32, comp_size, uncomp_size;
    uint16_t name_len, extra_len;
    fs.read(reinterpret_cast<char*>(&version), 2);
    fs.read(reinterpret_cast<char*>(&flags), 2);
    fs.read(reinterpret_cast<char*>(&method), 2);
    fs.read(reinterpret_cast<char*>(&modtime), 2);
    fs.read(reinterpret_cast<char*>(&moddate), 2);
    fs.read(reinterpret_cast<char*>(&crc32), 4);
    fs.read(reinterpret_cast<char*>(&comp_size), 4);
    fs.read(reinterpret_cast<char*>(&uncomp_size), 4);
    fs.read(reinterpret_cast<char*>(&name_len), 2);
    fs.read(reinterpret_cast<char*>(&extra_len), 2);

    std::string member_name(name_len, '\0');
    fs.read(member_name.data(), name_len);
    fs.ignore(extra_len);  // skip extra

    if (method != 0) {
      throw std::runtime_error("npz entry is compressed; mini-loader only supports stored");
    }
    // Read the embedded .npy
    std::vector<char> buf(uncomp_size);
    fs.read(buf.data(), uncomp_size);
    std::stringstream ss(std::string(buf.data(), buf.size()));
    arrays[member_name] = load_npy_stream(ss);
  }
  return arrays;
}

inline NpyArray npz_load(const std::string& fname, const std::string& varname) {
  auto all = npz_load(fname);
  auto it = all.find(varname);
  if (it == all.end()) {
    throw std::runtime_error("Variable not found in npz: " + varname);
  }
  return it->second;
}

}  // namespace cnpy