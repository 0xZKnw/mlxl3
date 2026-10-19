#pragma once

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <dispatch/dispatch.h>
#include <exception>
#include <limits>
#include <stdexcept>
#include <string>
#include <sys/uio.h>
#include <unistd.h>
#include <utility>
#include <vector>

namespace mlxl3 {
struct DiskRead {
  int fd;
  uint64_t offset;
  std::vector<iovec> destinations;
  uint64_t bytes_read = 0;
};

inline void copy_disk_record(const uint8_t* source, size_t size, const std::vector<iovec>& destinations) {
  if (!source || !size || destinations.empty() || destinations.size() > 64)
    throw std::invalid_argument("invalid staged record");
  const auto source_start = reinterpret_cast<uintptr_t>(source);
  if (size > std::numeric_limits<uintptr_t>::max() - source_start)
    throw std::invalid_argument("invalid staged source interval");
  std::vector<std::pair<uintptr_t, uintptr_t>> ranges;
  size_t total = 0;
  for (const auto& v : destinations) {
    auto start = reinterpret_cast<uintptr_t>(v.iov_base);
    if (!start || !v.iov_len || v.iov_len > std::numeric_limits<uintptr_t>::max() - start ||
        v.iov_len > size - total ||
        (start < source_start + size && source_start < start + v.iov_len))
      throw std::invalid_argument("invalid staged destination interval");
    total += v.iov_len;
    ranges.emplace_back(start, start + v.iov_len);
  }
  if (total != size) throw std::invalid_argument("incomplete staged record");
  std::sort(ranges.begin(), ranges.end());
  for (size_t i = 1; i < ranges.size(); ++i)
    if (ranges[i].first < ranges[i - 1].second)
      throw std::invalid_argument("overlapping staged destinations");
  size_t offset = 0;
  for (const auto& v : destinations) {
    std::memcpy(v.iov_base, source + offset, v.iov_len);
    offset += v.iov_len;
  }
}

inline void read_disk_record(DiskRead& read) {
  size_t index = 0, done = 0;
  read.bytes_read = 0;
  while (index < read.destinations.size()) {
    auto* vectors = read.destinations.data();
    auto n = ::preadv(read.fd, vectors + index,
                     int(read.destinations.size() - index), off_t(read.offset + done));
    if (n < 0 && errno == EINTR) continue;
    if (n < 0) {
      const auto code = errno;
      char message[256];
      const auto detail = ::strerror_r(code, message, sizeof(message)) == 0
          ? std::string(message) : std::to_string(code);
      throw std::runtime_error("preadv failed: " + detail);
    }
    if (n == 0) throw std::runtime_error("checkpoint truncated during disk record read");
    done += size_t(n);
    read.bytes_read += uint64_t(n);
    auto remaining = size_t(n);
    while (remaining && index < read.destinations.size()) {
      auto consumed = std::min(remaining, vectors[index].iov_len);
      vectors[index].iov_base = static_cast<uint8_t*>(vectors[index].iov_base) + consumed;
      vectors[index].iov_len -= consumed;
      remaining -= consumed;
      if (!vectors[index].iov_len) ++index;
    }
  }
}

inline void validate_disk_reads(const std::vector<DiskRead>& reads) {
  if (reads.empty() || reads.size() > 512)
    throw std::invalid_argument("invalid disk reader batch");
  std::vector<std::pair<uintptr_t, uintptr_t>> ranges;
  for (const auto& read : reads) {
    if (read.destinations.empty() || read.destinations.size() > 64)
      throw std::invalid_argument("invalid disk record destinations");
    uint64_t total = 0;
    for (const auto& v : read.destinations) {
      auto start = reinterpret_cast<uintptr_t>(v.iov_base);
      if (!start || !v.iov_len || v.iov_len > std::numeric_limits<uintptr_t>::max() - start ||
          v.iov_len > uint64_t(std::numeric_limits<off_t>::max()) - total)
        throw std::invalid_argument("invalid disk destination interval");
      ranges.emplace_back(start, start + v.iov_len);
      total += v.iov_len;
    }
    if (read.offset > uint64_t(std::numeric_limits<off_t>::max()) - total)
      throw std::overflow_error("disk record offset overflow");
  }
  std::sort(ranges.begin(), ranges.end());
  for (size_t i = 1; i < ranges.size(); ++i)
    if (ranges[i].first < ranges[i - 1].second)
      throw std::invalid_argument("overlapping disk destinations");
}

// Destinations must remain owned until this synchronous call returns, including on error.
inline void read_disk_records(std::vector<DiskRead>& reads, size_t workers) {
  if (!workers || workers > 4) throw std::invalid_argument("invalid disk worker count");
  validate_disk_reads(reads);
  workers = std::min(workers, reads.size());
  std::vector<std::exception_ptr> errors(reads.size());
  struct Work {
    std::vector<DiskRead>& reads;
    std::vector<std::exception_ptr>& errors;
    size_t workers;
  } context{reads, errors, workers};
  auto apply = [](void* opaque, size_t worker) noexcept {
    auto& work = *static_cast<Work*>(opaque);
    for (size_t i = worker; i < work.reads.size(); i += work.workers) {
      try { read_disk_record(work.reads[i]); }
      catch (...) { work.errors[i] = std::current_exception(); }
    }
  };
  if (workers == 1) apply(&context, 0);
  else dispatch_apply_f(workers, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0),
                        &context, apply);
  for (const auto& error : errors)
    if (error) std::rethrow_exception(error);
}
}  // namespace mlxl3
