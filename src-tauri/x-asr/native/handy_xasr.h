#pragma once
#include <stddef.h>
#include <stdint.h>

#ifdef _WIN32
#define HANDY_XASR_API __declspec(dllexport)
#else
#define HANDY_XASR_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
#define HANDY_XASR_NOEXCEPT noexcept
extern "C" {
#else
#define HANDY_XASR_NOEXCEPT
#endif

typedef struct HandyXAsrModel HandyXAsrModel;
typedef struct HandyXAsrStream HandyXAsrStream;

// Every entry point returns zero on success. No C++ exception crosses this ABI.
// Text is UTF-8, borrowed until the next operation on its owning model/stream.
HANDY_XASR_API int handy_xasr_ort_api(uint32_t version, const void **api,
    char *error, size_t capacity) HANDY_XASR_NOEXCEPT;
HANDY_XASR_API int handy_xasr_load(const char *directory, int offline,
    HandyXAsrModel **model, char *error, size_t capacity) HANDY_XASR_NOEXCEPT;
HANDY_XASR_API int handy_xasr_destroy(HandyXAsrModel *model,
    char *error, size_t capacity) HANDY_XASR_NOEXCEPT;
HANDY_XASR_API int handy_xasr_offline(HandyXAsrModel *model,
    const float *samples, int32_t count, const char **text, size_t *length,
    char *error, size_t capacity) HANDY_XASR_NOEXCEPT;
HANDY_XASR_API int handy_xasr_start(HandyXAsrModel *model,
    HandyXAsrStream **stream, char *error, size_t capacity) HANDY_XASR_NOEXCEPT;
HANDY_XASR_API int handy_xasr_feed(HandyXAsrStream *stream,
    const float *samples, int32_t count, const char **text, size_t *length,
    char *error, size_t capacity) HANDY_XASR_NOEXCEPT;
HANDY_XASR_API int handy_xasr_finish(HandyXAsrStream *stream,
    const char **text, size_t *length, char *error, size_t capacity) HANDY_XASR_NOEXCEPT;
HANDY_XASR_API int handy_xasr_cancel(HandyXAsrStream *stream,
    char *error, size_t capacity) HANDY_XASR_NOEXCEPT;

#ifdef __cplusplus
}
#endif
