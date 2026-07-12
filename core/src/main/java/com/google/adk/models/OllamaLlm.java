package com.google.adk.models;

import com.google.common.collect.ImmutableList;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import com.google.genai.types.Content;
import com.google.genai.types.GenerateContentConfig;
import com.google.genai.types.Part;
import io.github.amithkoujalgi.ollama4j.core.OllamaAPI;
import io.github.amithkoujalgi.ollama4j.core.models.chat.OllamaChatMessage;
import io.github.amithkoujalgi.ollama4j.core.models.chat.OllamaChatMessageRole;
import io.github.amithkoujalgi.ollama4j.core.models.chat.OllamaChatResult;
import io.reactivex.rxjava3.core.Flowable;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.stream.Collectors;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Represents an Ollama-backed LLM.
 *
 * <p>This implementation uses ollama4j's chat API. Model names registered through the ADK registry
 * use the {@code ollama/<model>} prefix, for example {@code ollama/llama3.1}. The prefix is removed
 * before calling the Ollama server.
 */
public class OllamaLlm extends BaseLlm {

  private static final Logger logger = LoggerFactory.getLogger(OllamaLlm.class);
  private static final String OLLAMA_MODEL_PREFIX = "ollama/";
  private static final String DEFAULT_HOST = "http://localhost:11434";
  private static final long DEFAULT_TIMEOUT_SECONDS = 120;

  private final OllamaAPI client;
  private final String ollamaModelName;

  public OllamaLlm(String model) {
    this(model, defaultClient());
  }

  public OllamaLlm(String model, OllamaAPI client) {
    super(model);
    this.client = Objects.requireNonNull(client, "client must be set.");
    this.ollamaModelName = stripOllamaPrefix(model);
  }

  @Override
  public Flowable<LlmResponse> generateContent(LlmRequest llmRequest, boolean stream) {
    if (hasTools(llmRequest)) {
      throw new UnsupportedOperationException(
          "OllamaLlm does not support ADK tool/function calling through ollama4j yet.");
    }

    List<OllamaChatMessage> messages = new ArrayList<>();
    extractSystemText(llmRequest)
        .ifPresent(text -> messages.add(new OllamaChatMessage(OllamaChatMessageRole.SYSTEM, text)));

    for (Content content : llmRequest.contents()) {
      contentToOllamaMessage(content).ifPresent(messages::add);
    }

    String requestModel =
        llmRequest.model().map(OllamaLlm::stripOllamaPrefix).orElse(ollamaModelName);
    try {
      OllamaChatResult result = client.chat(requestModel, messages);
      logger.debug("Ollama response: {}", result);
      return Flowable.just(toLlmResponse(result.getResponse()));
    } catch (Exception e) {
      return Flowable.error(
          new RuntimeException("Failed to generate content with Ollama model " + requestModel, e));
    }
  }

  private static Optional<String> extractSystemText(LlmRequest llmRequest) {
    Optional<GenerateContentConfig> configOpt = llmRequest.config();
    if (configOpt.isEmpty() || configOpt.get().systemInstruction().isEmpty()) {
      return Optional.empty();
    }
    String systemText =
        configOpt.get().systemInstruction().get().parts().orElse(ImmutableList.of()).stream()
            .filter(part -> part.text().isPresent())
            .map(part -> part.text().get())
            .collect(Collectors.joining("\n"));
    return systemText.isBlank() ? Optional.empty() : Optional.of(systemText);
  }

  private static Optional<OllamaChatMessage> contentToOllamaMessage(Content content) {
    List<Part> parts = content.parts().orElse(ImmutableList.of());
    String textContent =
        parts.stream()
            .filter(part -> part.text().isPresent())
            .map(part -> part.text().get())
            .collect(Collectors.joining("\n"));
    if (textContent.isBlank()) {
      return Optional.empty();
    }
    return Optional.of(new OllamaChatMessage(toOllamaRole(content.role().orElse("")), textContent));
  }

  private static OllamaChatMessageRole toOllamaRole(String role) {
    if ("model".equals(role) || "assistant".equals(role)) {
      return OllamaChatMessageRole.ASSISTANT;
    }
    if ("system".equals(role)) {
      return OllamaChatMessageRole.SYSTEM;
    }
    return OllamaChatMessageRole.USER;
  }

  private static boolean hasTools(LlmRequest llmRequest) {
    return llmRequest.config().isPresent()
        && llmRequest.config().get().tools().isPresent()
        && !llmRequest.config().get().tools().get().isEmpty();
  }

  private static LlmResponse toLlmResponse(String text) {
    return LlmResponse.builder()
        .content(
            Content.builder()
                .role("model")
                .parts(ImmutableList.of(Part.builder().text(text == null ? "" : text).build()))
                .build())
        .build();
  }

  private static String stripOllamaPrefix(String modelName) {
    if (modelName != null && modelName.startsWith(OLLAMA_MODEL_PREFIX)) {
      return modelName.substring(OLLAMA_MODEL_PREFIX.length());
    }
    return modelName;
  }

  private static OllamaAPI defaultClient() {
    String host = envOrDefault("OLLAMA_HOST", DEFAULT_HOST);
    OllamaAPI api = new OllamaAPI(host);
    api.setVerbose(false);
    api.setRequestTimeoutSeconds(readTimeoutSeconds());
    String username = System.getenv("OLLAMA_BASIC_AUTH_USERNAME");
    String password = System.getenv("OLLAMA_BASIC_AUTH_PASSWORD");
    if (username != null && !username.isBlank() && password != null && !password.isBlank()) {
      api.setBasicAuth(username, password);
    }
    return api;
  }

  private static long readTimeoutSeconds() {
    String value = System.getenv("OLLAMA_REQUEST_TIMEOUT_SECONDS");
    if (value == null || value.isBlank()) {
      return DEFAULT_TIMEOUT_SECONDS;
    }
    try {
      return Long.parseLong(value);
    } catch (NumberFormatException e) {
      logger.warn(
          "Invalid OLLAMA_REQUEST_TIMEOUT_SECONDS '{}', using default {}",
          value,
          DEFAULT_TIMEOUT_SECONDS);
      return DEFAULT_TIMEOUT_SECONDS;
    }
  }

  private static String envOrDefault(String name, String defaultValue) {
    String value = System.getenv(name);
    return value == null || value.isBlank() ? defaultValue : value;
  }

  @Override
  public BaseLlmConnection connect(LlmRequest llmRequest) {
    throw new UnsupportedOperationException("Live connection is not supported for Ollama models.");
  }

  public static Builder builder() {
    return new Builder();
  }

  public static class Builder {
    private String modelName;
    private OllamaAPI apiClient;

    private Builder() {}

    @CanIgnoreReturnValue
    public Builder modelName(String modelName) {
      this.modelName = modelName;
      return this;
    }

    @CanIgnoreReturnValue
    public Builder apiClient(OllamaAPI apiClient) {
      this.apiClient = apiClient;
      return this;
    }

    public OllamaLlm build() {
      Objects.requireNonNull(modelName, "modelName must be set.");
      return apiClient == null ? new OllamaLlm(modelName) : new OllamaLlm(modelName, apiClient);
    }
  }
}
