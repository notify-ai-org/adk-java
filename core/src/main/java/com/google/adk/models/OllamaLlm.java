package com.google.adk.models;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.google.adk.JsonBaseModel;
import com.google.common.collect.ImmutableList;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import com.google.genai.types.Content;
import com.google.genai.types.FunctionCall;
import com.google.genai.types.FunctionDeclaration;
import com.google.genai.types.FunctionResponse;
import com.google.genai.types.GenerateContentConfig;
import com.google.genai.types.Part;
import com.google.genai.types.Schema;
import io.github.ollama4j.Ollama;
import io.github.ollama4j.models.chat.OllamaChatMessage;
import io.github.ollama4j.models.chat.OllamaChatMessageRole;
import io.github.ollama4j.models.chat.OllamaChatRequest;
import io.github.ollama4j.models.chat.OllamaChatResult;
import io.github.ollama4j.models.chat.OllamaChatToolCalls;
import io.github.ollama4j.tools.OllamaToolCallsFunction;
import io.github.ollama4j.tools.Tools;
import io.reactivex.rxjava3.core.Flowable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
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
  private static final long DEFAULT_TIMEOUT_SECONDS = 900;
  private static final String TIMEOUT_PROPERTY = "ollama.request-timeout-seconds";
  private static final String TIMEOUT_ENV = "OLLAMA_REQUEST_TIMEOUT_SECONDS";

  private final Ollama client;
  private final String ollamaModelName;

  public OllamaLlm(String model) {
    this(model, defaultClient());
  }

  public OllamaLlm(String model, Ollama client) {
    super(model);
    this.client = Objects.requireNonNull(client, "client must be set.");
    this.client.setMaxChatToolCallRetries(0);
    this.ollamaModelName = stripOllamaPrefix(model);
  }

  @Override
  public Flowable<LlmResponse> generateContent(LlmRequest llmRequest, boolean stream) {

    List<OllamaChatMessage> messages = new ArrayList<>();
    extractSystemText(llmRequest)
        .ifPresent(text -> messages.add(new OllamaChatMessage(OllamaChatMessageRole.SYSTEM, text)));

    for (Content content : llmRequest.contents()) {
      contentToOllamaMessage(content).ifPresent(messages::add);
    }

    String requestModel =
        llmRequest.model().map(OllamaLlm::stripOllamaPrefix).orElse(ollamaModelName);
    try {
      List<Tools.Tool> tools = extractTools(llmRequest);
      OllamaChatRequest request =
          OllamaChatRequest.builder()
              .withModel(requestModel)
              .withMessages(messages)
              .withUseTools(false);
      if (!tools.isEmpty()) {
        request.withTools(tools);
      }
      request.build();
      OllamaChatResult result = client.chat(request, null);
      logger.debug("Ollama response: {}", result);
      return Flowable.just(toLlmResponse(result));
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

    List<OllamaChatToolCalls> toolCalls =
        parts.stream()
            .filter(part -> part.functionCall().isPresent())
            .map(part -> toOllamaToolCall(part.functionCall().get()))
            .collect(Collectors.toList());

    List<FunctionResponse> functionResponses =
        parts.stream()
            .filter(part -> part.functionResponse().isPresent())
            .map(part -> part.functionResponse().get())
            .collect(Collectors.toList());
    if (!functionResponses.isEmpty()) {
      return Optional.of(
          new OllamaChatMessage(
              OllamaChatMessageRole.TOOL,
              functionResponses.stream()
                  .map(OllamaLlm::functionResponseToContent)
                  .collect(Collectors.joining("\n"))));
    }

    if (textContent.isBlank() && toolCalls.isEmpty()) {
      return Optional.empty();
    }
    OllamaChatMessage message =
        new OllamaChatMessage(toOllamaRole(content.role().orElse("")), textContent);
    if (!toolCalls.isEmpty()) {
      message.setToolCalls(toolCalls);
    }
    return Optional.of(message);
  }

  private static OllamaChatToolCalls toOllamaToolCall(FunctionCall functionCall) {
    return new OllamaChatToolCalls(
        functionCall.id().orElse(null),
        new OllamaToolCallsFunction(
            functionCall.name().orElse(""),
            functionCall.args().map(HashMap::new).orElseGet(HashMap::new)));
  }

  private static String functionResponseToContent(FunctionResponse functionResponse) {
    try {
      return JsonBaseModel.getMapper()
          .writeValueAsString(functionResponse.response().orElse(Map.of()));
    } catch (JsonProcessingException e) {
      throw new IllegalStateException("Failed to serialize function response.", e);
    }
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

  private static LlmResponse toLlmResponse(OllamaChatResult result) {
    OllamaChatMessage message = extractResponseMessage(result);
    List<Part> parts = new ArrayList<>();
    if (message != null && message.getResponse() != null && !message.getResponse().isBlank()) {
      parts.add(Part.builder().text(message.getResponse()).build());
    }
    if (message != null && message.getToolCalls() != null) {
      message.getToolCalls().stream()
          .map(OllamaLlm::toFunctionCallPart)
          .flatMap(Optional::stream)
          .forEach(parts::add);
    }

    return LlmResponse.builder()
        .content(
            Content.builder()
                .role("model")
                .parts(
                    parts.isEmpty()
                        ? ImmutableList.of(Part.builder().text("").build())
                        : ImmutableList.copyOf(parts))
                .build())
        .build();
  }

  private static OllamaChatMessage extractResponseMessage(OllamaChatResult result) {
    if (result == null || result.getResponseModel() == null) {
      return null;
    }
    return result.getResponseModel().getMessage();
  }

  private static Optional<Part> toFunctionCallPart(OllamaChatToolCalls toolCall) {
    if (toolCall == null || toolCall.getFunction() == null) {
      return Optional.empty();
    }
    OllamaToolCallsFunction function = toolCall.getFunction();
    return Optional.of(
        Part.builder()
            .functionCall(
                FunctionCall.builder()
                    .id(toolCall.getId())
                    .name(function.getName())
                    .args(function.getArguments() == null ? Map.of() : function.getArguments())
                    .build())
            .build());
  }

  private static List<Tools.Tool> extractTools(LlmRequest llmRequest) {
    return llmRequest.config().flatMap(GenerateContentConfig::tools).stream()
        .flatMap(List::stream)
        .flatMap(tool -> tool.functionDeclarations().stream())
        .flatMap(List::stream)
        .map(OllamaLlm::toOllamaTool)
        .collect(Collectors.toList());
  }

  private static Tools.Tool toOllamaTool(FunctionDeclaration declaration) {
    return Tools.Tool.builder()
        .toolSpec(
            Tools.ToolSpec.builder()
                .name(declaration.name().orElse(""))
                .description(declaration.description().orElse(""))
                .parameters(toOllamaParameters(declaration.parameters()))
                .build())
        .build();
  }

  private static Tools.Parameters toOllamaParameters(Optional<Schema> parameters) {
    if (parameters.isEmpty() || parameters.get().properties().isEmpty()) {
      return Tools.Parameters.of(Map.of());
    }
    Schema schema = parameters.get();
    List<String> required = schema.required().orElse(List.of());
    Map<String, Tools.Property> properties = new HashMap<>();
    schema
        .properties()
        .get()
        .forEach(
            (name, propertySchema) -> {
              Tools.Property property =
                  Tools.Property.builder()
                      .type(toOllamaType(propertySchema))
                      .description(propertySchema.description().orElse(""))
                      .enumValues(propertySchema.enum_().orElse(null))
                      .required(required.contains(name))
                      .build();
              properties.put(name, property);
            });
    Tools.Parameters ollamaParameters = Tools.Parameters.of(properties);
    ollamaParameters.setRequired(new ArrayList<>(required));
    return ollamaParameters;
  }

  private static String toOllamaType(Schema schema) {
    return schema
        .type()
        .map(
            type ->
                type.knownEnum() == com.google.genai.types.Type.Known.TYPE_UNSPECIFIED
                    ? type.toString().toLowerCase()
                    : type.knownEnum().name().toLowerCase())
        .orElse("string");
  }

  private static String stripOllamaPrefix(String modelName) {
    if (modelName != null && modelName.startsWith(OLLAMA_MODEL_PREFIX)) {
      return modelName.substring(OLLAMA_MODEL_PREFIX.length());
    }
    return modelName;
  }

  private static Ollama defaultClient() {
    String host = envOrDefault("OLLAMA_HOST", DEFAULT_HOST);
    Ollama api = new Ollama(host);
    api.setRequestTimeoutSeconds(readTimeoutSeconds());
    String username = System.getenv("OLLAMA_BASIC_AUTH_USERNAME");
    String password = System.getenv("OLLAMA_BASIC_AUTH_PASSWORD");
    if (username != null && !username.isBlank() && password != null && !password.isBlank()) {
      api.setBasicAuth(username, password);
    }
    return api;
  }

  private static long readTimeoutSeconds() {
    String value = System.getProperty(TIMEOUT_PROPERTY);
    if (value == null || value.isBlank()) {
      value = System.getenv(TIMEOUT_ENV);
    }
    if (value == null || value.isBlank()) {
      return DEFAULT_TIMEOUT_SECONDS;
    }
    try {
      return Long.parseLong(value);
    } catch (NumberFormatException e) {
      logger.warn(
          "Invalid Ollama request timeout '{}', using default {} seconds",
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
    private Ollama apiClient;

    private Builder() {}

    @CanIgnoreReturnValue
    public Builder modelName(String modelName) {
      this.modelName = modelName;
      return this;
    }

    @CanIgnoreReturnValue
    public Builder apiClient(Ollama apiClient) {
      this.apiClient = apiClient;
      return this;
    }

    public OllamaLlm build() {
      Objects.requireNonNull(modelName, "modelName must be set.");
      return apiClient == null ? new OllamaLlm(modelName) : new OllamaLlm(modelName, apiClient);
    }
  }
}
