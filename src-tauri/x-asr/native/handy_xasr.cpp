#include "handy_xasr.h"

#include <array>
#include <cstdio>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include "onnxruntime_c_api.h"
#include "sherpa-onnx/csrc/offline-recognizer.h"
#include "sherpa-onnx/csrc/online-recognizer.h"

struct HandyXAsrModel {
  std::unique_ptr<sherpa_onnx::OnlineRecognizer> online;
  std::unique_ptr<sherpa_onnx::OfflineRecognizer> offline;
  std::string text;
};

struct HandyXAsrStream {
  HandyXAsrModel *model;
  std::unique_ptr<sherpa_onnx::OnlineStream> stream;
  std::string text;
  bool has_samples = false;
  bool finished = false;
};

namespace {
template <class F>
int guarded(char *error, size_t capacity, F &&operation) noexcept {
  if (error && capacity) error[0] = '\0';
  try {
    operation();
    return 0;
  } catch (const std::exception &e) {
    if (error && capacity) std::snprintf(error, capacity, "%s", e.what());
  } catch (...) {
    if (error && capacity)
      std::snprintf(error, capacity, "unknown X-ASR native exception");
  }
  return 1;
}

void result(const std::string &value, const char **text, size_t *length) {
  if (!text || !length) throw std::invalid_argument("missing result output");
  *text = value.data();
  *length = value.size();
}

bool drain(HandyXAsrStream &s) {
  bool decoded = false;
  while (s.model->online->IsReady(s.stream.get())) {
    s.model->online->DecodeStream(s.stream.get());
    decoded = true;
  }
  if (!decoded) return false;
  auto text = s.model->online->GetResult(s.stream.get()).text;
  if (text == s.text) return false;
  s.text = std::move(text);
  return true;
}
}  // namespace

int handy_xasr_ort_api(uint32_t version, const void **api, char *error,
                       size_t capacity) noexcept {
  return guarded(error, capacity, [&] {
    if (!api) throw std::invalid_argument("missing ORT API output");
    const auto *base = OrtGetApiBase();
    if (!base) throw std::runtime_error("linked ONNX Runtime has no API base");
    *api = base->GetApi(version);
    if (!*api)
      throw std::runtime_error(std::string("ONNX Runtime ") + base->GetVersionString() +
                               " does not support required API " + std::to_string(version));
  });
}

int handy_xasr_load(const char *directory, int offline, HandyXAsrModel **model,
                    char *error, size_t capacity) noexcept {
  return guarded(error, capacity, [&] {
    if (!directory || !model) throw std::invalid_argument("missing model directory/output");
    *model = nullptr;
    const std::string root = std::string(directory) + "/";
    auto owner = std::make_unique<HandyXAsrModel>();
    if (offline) {
      sherpa_onnx::OfflineRecognizerConfig config;
      config.feat_config.sampling_rate = 16000;
      config.feat_config.feature_dim = 80;
      config.model_config.transducer.encoder_filename = root + "encoder-epoch-99-avg-1.int8.onnx";
      config.model_config.transducer.decoder_filename = root + "decoder-epoch-99-avg-1.onnx";
      config.model_config.transducer.joiner_filename = root + "joiner-epoch-99-avg-1.int8.onnx";
      config.model_config.tokens = root + "tokens.txt";
      // The pinned export is an icefall transducer. Avoid loading the encoder
      // once for architecture detection and again for actual inference.
      config.model_config.model_type = "transducer";
      config.model_config.num_threads = 1;
      config.model_config.provider = "cpu";
      config.decoding_method = "greedy_search";
      if (!config.Validate()) throw std::runtime_error("invalid X-ASR offline model configuration");
      owner->offline = std::make_unique<sherpa_onnx::OfflineRecognizer>(config);
    } else {
      sherpa_onnx::OnlineRecognizerConfig config;
      config.feat_config.sampling_rate = 16000;
      config.feat_config.feature_dim = 80;
      config.model_config.transducer.encoder = root + "encoder-480ms.onnx";
      config.model_config.transducer.decoder = root + "decoder-480ms.onnx";
      config.model_config.transducer.joiner = root + "joiner-480ms.onnx";
      config.model_config.tokens = root + "tokens.txt";
      config.model_config.model_type = "zipformer2";
      config.model_config.num_threads = 1;
      config.model_config.provider_config.provider = "cpu";
      config.decoding_method = "greedy_search";
      config.enable_endpoint = false;
      if (!config.Validate()) throw std::runtime_error("invalid X-ASR streaming model configuration");
      owner->online = std::make_unique<sherpa_onnx::OnlineRecognizer>(config);
    }
    *model = owner.release();
  });
}

int handy_xasr_destroy(HandyXAsrModel *model, char *error, size_t capacity) noexcept {
  return guarded(error, capacity, [&] { delete model; });
}

int handy_xasr_offline(HandyXAsrModel *model, const float *samples, int32_t count,
                       const char **text, size_t *length, char *error, size_t capacity) noexcept {
  return guarded(error, capacity, [&] {
    if (!model || !model->offline) throw std::invalid_argument("not an offline X-ASR model");
    // The Rust segmenter pads nonempty short inputs and preserves all samples.
    if (!samples || count < 1600 || count > 480000)
      throw std::invalid_argument("offline X-ASR segment must contain 100ms to 30s of 16kHz audio");
    auto stream = model->offline->CreateStream();
    if (!stream) throw std::runtime_error("could not create offline X-ASR stream");
    stream->AcceptWaveform(16000, samples, count);
    model->offline->DecodeStream(stream.get());
    model->text = stream->GetResult().text;
    result(model->text, text, length);
  });
}

int handy_xasr_start(HandyXAsrModel *model, HandyXAsrStream **stream,
                     char *error, size_t capacity) noexcept {
  return guarded(error, capacity, [&] {
    if (!model || !model->online || !stream)
      throw std::invalid_argument("not a streaming X-ASR model or missing stream output");
    *stream = nullptr;
    auto owner = std::make_unique<HandyXAsrStream>();
    owner->model = model;
    owner->stream = model->online->CreateStream();
    if (!owner->stream) throw std::runtime_error("could not create online X-ASR stream");
    *stream = owner.release();
  });
}

int handy_xasr_feed(HandyXAsrStream *stream, const float *samples, int32_t count,
                    const char **text, size_t *length, char *error, size_t capacity) noexcept {
  return guarded(error, capacity, [&] {
    if (!stream || stream->finished || count < 0 || (count && !samples) || !text || !length)
      throw std::invalid_argument("invalid or finished online X-ASR stream");
    *text = nullptr;
    *length = 0;
    if (count) {
      stream->has_samples = true;
      stream->stream->AcceptWaveform(16000, samples, count);
      if (drain(*stream)) result(stream->text, text, length);
    }
  });
}

int handy_xasr_finish(HandyXAsrStream *stream, const char **text, size_t *length,
                      char *error, size_t capacity) noexcept {
  return guarded(error, capacity, [&] {
    if (!stream || stream->finished) throw std::invalid_argument("invalid or finished X-ASR stream");
    stream->finished = true;
    if (stream->has_samples) {
      // input_finished alone loses X-ASR sentence tails. Drain 1.5s of silence
      // through the same stream/cache before marking the input complete.
      static constexpr std::array<float, 24000> silence{};
      stream->stream->AcceptWaveform(16000, silence.data(), silence.size());
      drain(*stream);
      stream->stream->InputFinished();
      drain(*stream);
    }
    result(stream->text, text, length);
  });
}

int handy_xasr_cancel(HandyXAsrStream *stream, char *error, size_t capacity) noexcept {
  return guarded(error, capacity, [&] { delete stream; });
}
